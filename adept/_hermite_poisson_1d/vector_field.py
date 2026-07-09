"""
Core vector field for the 1D Hermite-Poisson solver.

State: Ck_electrons(Nn_e, Nx), Ck_ions(Nn_i, Nx) [complex, stored as float64 views],
       a(Nx+2), prev_a(Nx+2) [real, vector potential with boundary cells],
       e(Nx) [real, electrostatic field for diagnostics].

Integration: Lawson-RK4 for Ck (free-streaming exact, E-field/ponderomotive explicit)
             + explicit leapfrog WaveSolver for a.
             Uses the Stepper (discrete-map) convention from adept._base_.
"""

import jax
import jax.numpy as jnp
from jax import Array

from adept._base_ import get_envelope
from adept._vlasov1d.solvers.pushers.field import WaveSolver


class TransverseWaveDriver:
    """Wave-equation source term, CALIBRATED so the radiated wave has amplitude a0.

    Two source modes per pulse (pulse["source"], default "extended"):

    "point" (recommended; identical to vlasov1d's TransverseCurrentSourceDriver):
        S = (F0/dx) · time_envelope(t) · δ_{i0} · sin(w·t),  F0 = 2·w·c·a0.
        The 1D Green's function for S = F0·δ(x−x0)·sin(wt) gives outgoing waves
        of amplitude F0/(2·k·c²); with vacuum k = w/c that is exactly a0.
        Radiates both directions — place near an absorbing boundary.

    "extended": S = −w²·a0·(2c/(w·G))·envelope(x,t)·sin(kx − wt), where
        G = ∫env_x dx. A phase-matched traveling source of integrated strength
        S0·G radiates amplitude S0·G/(2·w·c), so the 2c/(w·G) factor calibrates
        the outgoing amplitude to a0 (accurate for antenna width ≳ wavelength).

    REGRESSION NOTE: the extended mode originally had NO calibration — the
    radiated amplitude was a0·(w·G/2c), a factor ~9 (81× intensity) for the
    production SRS geometry (1 µm antenna, 351 nm pump at n_c normalization).
    Every HP campaign before 2026-07-04 was driven at ~81× nominal intensity;
    measured pump/nominal = 9.00 at all four campaign intensities, matching
    (w/2c)·G = 8.95 analytically. The vlasov1d reference campaigns used its
    calibrated point source and were unaffected.

    Reads the raw normalized driver_config dict. Output shape: (Nx+2,),
    matching WaveSolver's djy_array expectation.
    """

    def __init__(self, x_a: Array, ey_driver_cfg: dict, c: float = 1.0):
        """
        Args:
            x_a: Extended x grid with ghost cells, shape (Nx+2,).
            ey_driver_cfg: dict mapping pulse_name → pulse_dict from cfg["drivers"]["ey"].
            c: normalized light speed (1.0 in skin-depth units).
        """
        self.x_a = x_a
        dx = float(x_a[1] - x_a[0])
        # Pre-parse all scalars at init so __call__ contains no float() on JAX arrays.
        x_a_last = float(x_a[-1])
        parsed = []
        for _, pulse in ey_driver_cfg.items() if isinstance(ey_driver_cfg, dict) else []:
            if not isinstance(pulse, dict):
                continue
            w_total = float(pulse["w0"]) + float(pulse.get("dw0", 0.0))
            t_center = float(pulse.get("t_center", 0.0))
            t_half = 0.5 * float(pulse.get("t_width", 1e10))
            t_rise = float(pulse.get("t_rise", 0.0))
            x_center = float(pulse.get("x_center", 0.5 * x_a_last))
            x_half = 0.5 * float(pulse.get("x_width", 1e10))
            x_rise = float(pulse.get("x_rise", 0.0))
            source = str(pulse.get("source", "extended"))
            if source not in ("extended", "point"):
                raise ValueError(f"drivers.ey pulse source must be 'extended' or 'point', got {source!r}")

            if source == "point":
                i0 = int(jnp.argmin(jnp.abs(x_a - x_center)))
                mask = jnp.zeros_like(x_a).at[i0].set(1.0)
                # F0 = 2*w*c*a0 (vacuum-dispersion Green's factor), applied as F0/dx
                scale = 2.0 * w_total * float(c) * float(pulse["a0"]) / dx
                parsed.append(("point", scale, w_total, t_center, t_half, t_rise, mask))
            else:
                env_x = get_envelope(x_rise, x_rise, x_center - x_half, x_center + x_half, x_a)
                G = float(jnp.trapezoid(env_x, x_a))
                if G <= 0.0:
                    raise ValueError("drivers.ey: extended-source spatial envelope integrates to zero")
                # Radiated amplitude of the uncalibrated antenna is a0*(w*G/2c);
                # divide it out so the launched wave has amplitude a0.
                cal = 2.0 * float(c) / (w_total * G)
                amp = -(w_total**2) * float(pulse["a0"]) * cal
                parsed.append(("extended", amp, w_total, t_center, t_half, t_rise, (float(pulse["k0"]), env_x)))
        self.parsed_pulses = parsed

    def __call__(self, t: float, args) -> Array:
        total = jnp.zeros_like(self.x_a)
        for source, amp, w_total, t_center, t_half, t_rise, extra in self.parsed_pulses:
            env_t = get_envelope(t_rise, t_rise, t_center - t_half, t_center + t_half, t)
            if source == "point":
                total = total + amp * env_t * extra * jnp.sin(w_total * t)
            else:
                k0, env_x = extra
                total = total + env_t * env_x * amp * jnp.sin(k0 * self.x_a - w_total * t)
        return total


class LongitudinalElectricFieldDriver:
    """External longitudinal field E_drive(x, t) = Σ_pulses env(x,t) (w0+dw0) a0 sin(k0 x − (w0+dw0) t).

    The Hermite-Poisson counterpart of adept._vlasov1d...field.LongitudinalElectricFieldDriver:
    a prescribed Ex that is added to the self-consistent Poisson field inside the
    velocity-space force term (so it drives the plasma directly, e.g. a resonant EPW
    kick), without ever entering the Poisson or wave-equation solves. Reads the same
    flat normalized driver dict as TransverseWaveDriver (cfg["drivers"]["ex"]).

    Amplitude convention matches vlasov1d: the prefactor is the total angular frequency
    w0+dw0 (not |k0|, which is the spectrax1d Ex convention). Output shape (Nx,), matching
    the interior force-coupling grid (no ghost cells).
    """

    def __init__(self, x: Array, ex_driver_cfg: dict):
        """
        Args:
            x: Interior x grid (no ghost cells), shape (Nx,).
            ex_driver_cfg: dict mapping pulse_name → pulse_dict from cfg["drivers"]["ex"].
        """
        self.x = x
        # Pre-parse all scalars at init so __call__ contains no float() on JAX arrays.
        x_last = float(x[-1])
        parsed = []
        for _, pulse in ex_driver_cfg.items() if isinstance(ex_driver_cfg, dict) else []:
            if not isinstance(pulse, dict):
                continue
            w_total = float(pulse["w0"]) + float(pulse.get("dw0", 0.0))
            t_center = float(pulse.get("t_center", 0.0))
            t_half = 0.5 * float(pulse.get("t_width", 1e10))
            t_rise = float(pulse.get("t_rise", 0.0))
            x_center = float(pulse.get("x_center", 0.5 * x_last))
            x_half = 0.5 * float(pulse.get("x_width", 1e10))
            x_rise = float(pulse.get("x_rise", 0.0))
            parsed.append(
                (
                    float(pulse["k0"]),
                    w_total,
                    float(pulse["a0"]),
                    t_center,
                    t_half,
                    t_rise,
                    x_center,
                    x_half,
                    x_rise,
                )
            )
        self.parsed_pulses = parsed

    def __call__(self, t: float, args) -> Array:
        total = jnp.zeros_like(self.x)
        for k0, w_total, a0, t_center, t_half, t_rise, x_center, x_half, x_rise in self.parsed_pulses:
            env_t = get_envelope(t_rise, t_rise, t_center - t_half, t_center + t_half, t)
            env_x = get_envelope(x_rise, x_rise, x_center - x_half, x_center + x_half, self.x)
            total = total + env_t * env_x * w_total * a0 * jnp.sin(k0 * self.x - w_total * t)
        return total


# ---------------------------------------------------------------------------
# Linear exponential operators for (Nn, Nx) arrays
# ---------------------------------------------------------------------------


class FreeStreamingExp1D:
    """Exact exponential for free-streaming in x (acts on the n-axis, varies with kx).

    Prediagonalizes the symmetric tridiagonal streaming matrix T where
        T[n, n+1] = T[n+1, n] = sqrt((n+1)/2)   (+ diagonal drift/alpha term)
    and stores V (eigenvectors) and eigenvalues so that
        exp(-i*kx*alpha * T * s) = V * diag(exp(prefactor * kx * eigenvalues * s)) * V^T.

    kx_1d must already be in physical units (2*pi*fftfreq(Nx)*Nx/Lx) — the same
    array modules.py builds and also feeds to the Poisson solver and the
    hyper-diffusion term. (An earlier version divided by Lx again here, making
    free-streaming Lx-times too slow; caught by
    tests/test_hermite_poisson_1d/test_landau_damping.py.)
    """

    def __init__(self, Nn: int, alpha: float, u: float, Lx: float, kx_1d: Array):
        """
        Args:
            Nn: Number of Hermite modes.
            alpha: Thermal velocity (vth/c in skin-depth units).
            u: Drift velocity (usually 0).
            Lx: Domain length in normalized units (unused; kept for API stability).
            kx_1d: 1D wavenumber array, shape (Nx,), physical units
                   (2*pi*fftfreq(Nx)*Nx/Lx).
        """
        self.prefactor = -1j * float(alpha)
        self.kx_1d = kx_1d

        if Nn == 1:
            self.V = jnp.ones((1, 1))
            self.eigenvalues = jnp.array([float(u / alpha)])
        else:
            n = jnp.arange(Nn, dtype=jnp.float64)
            off_diag = jnp.sqrt((n + 1.0) / 2.0)[:-1]  # sqrt((n+1)/2) for n=0..Nn-2
            T = jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
            T = T + (float(u) / float(alpha)) * jnp.eye(Nn)
            eigenvalues, V = jnp.linalg.eigh(T)
            self.V = V
            self.eigenvalues = eigenvalues

    def apply(self, Ck: Array, s: float) -> Array:
        """Apply exp(L_stream * s) to Ck, shape (Nn, Nx)."""
        # Project: C_eig[i, kx] = V^T[i, n] * Ck[n, kx]
        C_eig = self.V.T @ Ck  # (Nn, Nx)
        # Scale: exp(prefactor * kx * eigenvalue * s)
        exp_fac = jnp.exp(self.prefactor * s * self.eigenvalues[:, None] * self.kx_1d[None, :])
        # Project back
        return self.V @ (C_eig * exp_fac)  # (Nn, Nx)


class DiagonalExp1D:
    """Exact exponential for hypercollision + hyper-diffusion + Hou-Li filter (diagonal in mode/k space)."""

    def __init__(
        self,
        nu: float,
        col_1d: Array,
        D: float,
        kx_sq_1d: Array,
        hou_li_strength: float = 0.0,
        hou_li_col_1d: Array | None = None,
    ):
        """
        Args:
            nu: Collision frequency.
            col_1d: Hypercollisional damping coefficients, shape (Nn,).
            D: Hyper-diffusion coefficient.
            kx_sq_1d: kx^2 array, shape (Nx,).
            hou_li_strength: Hou-Li filter strength (0 to disable).
            hou_li_col_1d: Hou-Li profile (n/Nn)^order, shape (Nn,). Required if strength > 0.
        """
        self.nu = nu
        self.col_1d = col_1d
        self.D = D
        self.kx_sq_1d = kx_sq_1d
        self.hou_li_strength = hou_li_strength
        self.hou_li_col_1d = hou_li_col_1d

    def apply(self, Ck: Array, s: float) -> Array:
        """Apply exp((-nu*col - D*kx^2 - hou_li_strength*hou_li_col) * s) to Ck, shape (Nn, Nx)."""
        col_fac = -self.nu * self.col_1d[:, None] * s  # (Nn, Nx)
        diff_fac = -self.D * self.kx_sq_1d[None, :] * s  # (1, Nx)
        if self.hou_li_strength > 0.0 and self.hou_li_col_1d is not None:
            hl_fac = -self.hou_li_strength * self.hou_li_col_1d[:, None] * s
        else:
            hl_fac = 0.0
        return Ck * jnp.exp(col_fac + diff_fac + hl_fac)


class LinearExp1D:
    """Combined free-streaming + diagonal exponential for one species."""

    def __init__(self, free_streaming: FreeStreamingExp1D, diagonal: DiagonalExp1D):
        self.free_streaming = free_streaming
        self.diagonal = diagonal

    def apply(self, Ck: Array, s: float) -> Array:
        Ck = self.free_streaming.apply(Ck, s)
        Ck = self.diagonal.apply(Ck, s)
        return Ck


class CombinedLinearExp1D:
    """Applies exp(L*s) to the full state dict (electrons + ions).

    Passes Ck_electrons and Ck_ions through their respective LinearExp1D operators.
    Real-valued fields (a, prev_a, e, da) are passed through unchanged.
    """

    def __init__(
        self,
        linear_e: LinearExp1D,
        linear_i: LinearExp1D,
        static_ions: bool = False,
    ):
        self.linear_e = linear_e
        self.linear_i = linear_i
        self.static_ions = static_ions

    def apply(self, state: dict, s: float) -> dict:
        """Apply exp(L*s) to the Ck components; pass real arrays unchanged."""
        Ck_e = state["Ck_electrons"].view(jnp.complex128)
        Ck_e_new = self.linear_e.apply(Ck_e, s)

        if self.static_ions:
            Ck_i_new_view = state["Ck_ions"]
        else:
            Ck_i = state["Ck_ions"].view(jnp.complex128)
            Ck_i_new_view = self.linear_i.apply(Ck_i, s).view(jnp.float64)

        out = dict(state)
        out["Ck_electrons"] = Ck_e_new.view(jnp.float64)
        out["Ck_ions"] = Ck_i_new_view
        return out


# ---------------------------------------------------------------------------
# Poisson solver (electrostatic field from density)
# ---------------------------------------------------------------------------


class PoissonSolver1D:
    """Spectral Poisson solver: Ex = -i/kx * FFT(n_i - n_e).

    n_e(x) = alpha_e^3 * IFFT(Ck_e[0, :]).real  (3D normalization from spectrax)
    n_i(x) = alpha_i^3 * IFFT(Ck_i[0, :]).real  (or static background)
    """

    def __init__(
        self,
        one_over_kx: Array,
        alpha_e: float,
        alpha_i: float,
        static_ion_density: Array | None = None,
    ):
        """
        Args:
            one_over_kx: 1/kx array (0 at k=0), shape (Nx,).
            alpha_e, alpha_i: Thermal velocities.
            static_ion_density: Fixed ion density profile shape (Nx,) for static_ions=True.
                                If None, ion density is computed from Ck_ions each step.
        """
        self.one_over_kx = one_over_kx
        self.alpha_e = float(alpha_e)
        self.alpha_i = float(alpha_i)
        self.static_ion_density = static_ion_density

    def electron_density(self, Ck_e: Array) -> Array:
        """n_e(x) from Hermite coefficient C_0 in k-space."""
        C0_e = jnp.fft.ifft(Ck_e[0, :], norm="forward").real
        return (self.alpha_e**3) * C0_e

    def ion_density(self, Ck_i: Array) -> Array:
        """n_i(x) from Hermite coefficient C_0 in k-space (or static background)."""
        if self.static_ion_density is not None:
            return self.static_ion_density
        C0_i = jnp.fft.ifft(Ck_i[0, :], norm="forward").real
        return (self.alpha_i**3) * C0_i

    def __call__(self, Ck_e: Array, Ck_i: Array) -> Array:
        """Return Ex(x), shape (Nx,)."""
        n_e = self.electron_density(Ck_e)
        n_i = self.ion_density(Ck_i)
        rho = n_i - n_e  # charge density (positive where ions > electrons)
        rho_k = jnp.fft.fft(rho, norm="forward")
        # Poisson: -kx^2 phi_k = -rho_k → phi_k = rho_k/kx^2
        # Ex = -d phi/dx → Ex_k = -i*kx*phi_k = -i/kx * rho_k
        Ex_k = -1j * self.one_over_kx * rho_k
        return jnp.fft.ifft(Ex_k, norm="forward").real


# ---------------------------------------------------------------------------
# Nonlinear RHS: E-field + ponderomotive Hermite coupling
# ---------------------------------------------------------------------------


def _hermite_e_coupling(
    Ck: Array,
    F_total: Array,
    sqrt_n_minus: Array,
    q_over_m: float,
    alpha: float,
    Omega_ce_tau: float,
    mask23: Array | None = None,
) -> Array:
    """Compute E-field (+ ponderomotive) coupling term for one species.

    dCk_n/dt|E = (q/m) * Omega_ce_tau * FFT[sqrt(2n)/alpha * F(x) * C[n-1](x)]

    where C[n-1] means the real-space coefficient at mode n-1 (zero row at n=0).
    This is the AW-Hermite force term of Parker & Dellar (2015) eq. 3.11:
    differentiation raises the Hermite-function index, so E·∂_v f projects onto
    mode n from mode n-1. Matches the validated spectrax1d term
    (sqrt_n_minus · √2/α · F · shift_multi(C, dn=-1), where dn=-1 yields C[n-1]).
    A uniform E applied to a Maxwellian (only C_0 ≠ 0) must drive current:
    dC_1/dt = (q/m)·√2/α·E·C_0.

    Args:
        Ck: (Nn, Nx) complex, in k-space.
        F_total: (Nx,) real, force field. CONTRACT: q_over_m * F_total must be
            the species ACCELERATION. E-field terms may be passed bare with
            q_over_m=q/m, but the ponderomotive term is ∝ q²/m² (charge-sign
            independent) and must be pre-composed by the caller — pass
            accel = (q/m)·E + (q/m)²·pond with q_over_m=1 (see _nonlinear_rhs).
        sqrt_n_minus: (Nn,) = sqrt([0,1,...,Nn-1]).
        q_over_m: Charge-to-mass ratio applied as an overall factor.
        alpha: Thermal velocity.
        Omega_ce_tau: Normalization constant (1.0 in current configs).
        mask23: Optional (Nx,) bool 2/3-rule dealiasing mask in FFT ordering.
            The F*C product is quadratic: without truncating both factors to
            |k| <= Nx//3 and masking the result, beyond-Nyquist products alias
            back into the resolved band (the spectrax1d base applies the same
            mask23 to every Lorentz product; the vlasov1d reference runs a
            Hou-Li grid-scale filter in x for the same reason).
    """
    Nn, Nx = Ck.shape
    if mask23 is not None:
        Ck = Ck * mask23[None, :]
        F_total = jnp.fft.ifft(jnp.fft.fft(F_total, norm="forward") * mask23, norm="forward").real

    # IFFT to real space: C(n, x)
    C = jnp.fft.ifft(Ck, axis=-1, norm="forward")  # (Nn, Nx) complex → real part matters

    # Source at n-1: result[n] = C[n-1], zero row at n=0
    C_up = jnp.concatenate([jnp.zeros((1, Nx), dtype=C.dtype), C[:-1, :]], axis=0)

    # Coupling: sqrt(2n)/alpha * F(x) * C[n-1](x)
    integrand = sqrt_n_minus[:, None] * jnp.sqrt(2.0) / alpha * F_total[None, :] * C_up  # (Nn, Nx)

    # FFT back and apply q/m * Omega_ce_tau
    out = q_over_m * Omega_ce_tau * jnp.fft.fft(integrand, axis=-1, norm="forward")
    if mask23 is not None:
        out = out * mask23[None, :]
    return out


def _exp_force_substep(
    Ck: Array,
    accel: Array,
    sqrt_n_minus: Array,
    alpha: float,
    Omega_ce_tau: float,
    dt: float,
    mask23: Array | None = None,
    n_terms: int = 24,
) -> Array:
    """Exact exponential step for the force coupling dC_n/dt = kappa_n·accel(x)·C_{n-1}.

    kappa_n = Omega_ce_tau·√(2n)/alpha. `accel` is the species ACCELERATION —
    (q/m)·E-terms + (q/m)²·ponderomotive, composed by the caller (same contract
    as _hermite_e_coupling called with q_over_m=1).

    With accel frozen over the substep, the force flow is exactly a velocity
    shift f(v) → f(v − accel·dt) — the same sub-flow the Fourier-in-v vlasov1d
    solver applies as a unitary phase, which is why that solver has NO E-field
    stability bound. In the AW-Hermite basis the generator A is strictly lower
    bidiagonal (nilpotent), so exp(dt·A) is the FINITE series Σ (dt·A)^k/k!,
    applied here as n_terms cheap bidiagonal products (O(n_terms·Nn·Nx), far
    below the streaming matmul). Truncating at n_terms is accurate when
    x_max = dt·|accel|·√(2·Nn)·√2/α satisfies x_max^n_terms/n_terms! ≪ 1 —
    n_terms=24 covers x_max ≲ 6, i.e. fields ~2× past the explicit-RK4
    divergence bound (~2.8) that killed the Lawson path in production.

    NOT Crank-Nicolson: the trapezoidal rational approximation is spectrally
    stable but its non-normal transient response down the nilpotent ladder
    behaves like (2x)^k where the true flow has x^k/k! — it detonates in
    exactly the strong-force regime this substep exists to survive.
    """
    Nn, Nx = Ck.shape
    if mask23 is not None:
        Ck = Ck * mask23[None, :]
        accel = jnp.fft.ifft(jnp.fft.fft(accel, norm="forward") * mask23, norm="forward").real

    C = jnp.fft.ifft(Ck, axis=-1, norm="forward")  # (Nn, Nx) complex
    kappa = Omega_ce_tau * jnp.sqrt(2.0) / alpha * sqrt_n_minus  # (Nn,)
    kdt = (kappa * dt)[:, None] * accel[None, :]  # (Nn, Nx) real

    zero_row = jnp.zeros((1, Nx), dtype=C.dtype)

    def apply_A_dt(V: Array) -> Array:
        # (dt·A·V)_n = dt·kappa_n·accel·V_{n-1}; row 0 = 0
        return kdt * jnp.concatenate([zero_row, V[:-1, :]], axis=0)

    out_C = C
    term = C
    k_max = min(int(n_terms), Nn - 1) if Nn > 1 else 0
    for k in range(1, k_max + 1):
        term = apply_A_dt(term) / k
        out_C = out_C + term

    out = jnp.fft.fft(out_C, axis=-1, norm="forward")
    if mask23 is not None:
        out = out * mask23[None, :]
    return out


# ---------------------------------------------------------------------------
# Top-level vector field (Stepper/discrete-map convention)
# ---------------------------------------------------------------------------


def _tree_add(a: dict, b: dict) -> dict:
    import jax

    return jax.tree.map(lambda x, y: x + y, a, b)


def _tree_scale(a: dict, c: float) -> dict:
    import jax

    return jax.tree.map(lambda x: c * x, a)


class HermitePoisson1DVectorField:
    """Advances the 1D Hermite-Poisson state by one timestep dt.

    Two integrators (select via `integrator`):

    "strang-exp" (default): Strang split exp(L·dt/2) → exact force-exponential
      substep (finite nilpotent series; the frozen-force flow is exactly a
      velocity shift, so this is the Hermite analog of vlasov1d's unitary
      e^{i·k_v·E·dt} push and carries no explicit-RK force stability bound)
      → exp(L·dt/2), with every wave↔plasma coupling time-centered: the force
      substep sees Poisson-Ex from the mid-step C_0, the vector potential
      linearly extrapolated to t+dt/2, and the Ex driver at t+dt/2; the
      leapfrog wave update sees n_e and the source at t_n (its own centering
      point). Second order, ~3× cheaper per step than Lawson-RK4.

    "lawson-rk4" (legacy): Lawson-RK4 for Ck with the force treated explicitly
      (stability bound |F|·√(2Nn)·√2/α·dt ≲ 2.8) and two half-step-off-center
      couplings kept bug-compatible for A/B: the ponderomotive uses a frozen at
      step START (half-step early) and the wave equation uses ½(n_eⁿ+n_eⁿ⁺¹)
      and the source at t+dt/2 (half-step late). These opposite first-order
      offsets in the two legs of the parametric loop are the suspected source
      of the anomalous γ ≈ a0·ωp0 pump-proportional gain observed at
      quarter-critical in SRS runs (4–5× the physical γ0), and the explicit
      force bound is the observed super-exponential runaway once fields reach
      E_crit ≈ 0.034 (Nn=1024, dt=0.1).

    Called by adept._base_.Stepper which treats the return value as the new state
    (not dy/dt). dt is stored internally from grid.dt at construction time.
    """

    def __init__(
        self,
        combined_exp: CombinedLinearExp1D,
        poisson: PoissonSolver1D,
        wave_solver: WaveSolver,
        ey_driver: TransverseWaveDriver,
        ex_driver: LongitudinalElectricFieldDriver,
        sqrt_n_minus_e: Array,
        sqrt_n_minus_i: Array,
        alpha_e: float,
        alpha_i: float,
        q_e: float,
        q_i: float,
        mi_me: float,
        Omega_ce_tau: float,
        dx: float,
        dt: float,
        static_ions: bool = False,
        sponge_plasma: Array | None = None,
        sponge_fields: Array | None = None,
        mask23: Array | None = None,
        noise_amplitude: float = 0.0,
        noise_seed: int = 0,
        noise_spatial_profile: Array | None = None,
        integrator: str = "strang-exp",
        force_exp_terms: int = 24,
        force_cap: float | None = None,
        wave_density_max: float | None = None,
    ):
        if integrator not in ("strang-exp", "lawson-rk4"):
            raise ValueError(f"Unknown integrator {integrator!r}; use 'strang-exp' or 'lawson-rk4'")
        self.integrator = integrator
        self.force_exp_terms = int(force_exp_terms)
        # LPSE-style stabilization clamps (None = off). See _strang_exp_step.
        # force_cap: smooth tanh saturation of the species acceleration —
        # accel -> cap*tanh(accel/cap). Transparent while |accel| << cap
        # (relative distortion (accel/cap)^2/3), artificially saturates the
        # parametric loop gain above it, and bounds the hierarchy-forcing
        # parameter x = dt*|accel|*sqrt(4*Nn)/alpha below the truncated-AW
        # detonation threshold. Above-threshold runs become stable but NOT
        # quantitatively accurate at saturation amplitude (by design).
        # wave_density_max: clip on the electron density the wave equation
        # sees (safety net against runaway omega_pe^2*dt^2 at huge delta-n).
        self.force_cap = None if force_cap is None else float(force_cap)
        self.wave_density_max = None if wave_density_max is None else float(wave_density_max)
        self.combined_exp = combined_exp
        self.poisson = poisson
        self.wave_solver = wave_solver
        self.ey_driver = ey_driver
        self.ex_driver = ex_driver

        self.sqrt_n_minus_e = sqrt_n_minus_e
        self.sqrt_n_minus_i = sqrt_n_minus_i
        self.alpha_e = float(alpha_e)
        self.alpha_i = float(alpha_i)
        # q/m in normalized units: electrons q=-1/m=1, ions q=+1/m=mi_me
        self.qm_e = float(q_e) / 1.0
        self.qm_i = float(q_i) / float(mi_me)
        self.Omega_ce_tau = float(Omega_ce_tau)
        self.dx = float(dx)
        self.dt = float(dt)
        self.static_ions = static_ions
        self.sponge_plasma = sponge_plasma
        self.sponge_fields = sponge_fields
        self.mask23 = mask23
        # Per-step stochastic Ex force noise (vlasov1d stochastic_noise parity):
        # white in time, Gaussian in space, density-weighted; drawn ONCE per step
        # (frozen across the Lawson-RK4 stages, like a_frozen) and added to the
        # v-advection force only — never to Poisson or the wave solver.
        self.noise_amplitude = float(noise_amplitude)
        self.noise_base_key = jax.random.PRNGKey(int(noise_seed))
        self.noise_spatial_profile = noise_spatial_profile

    # ------------------------------------------------------------------
    # Nonlinear RHS (E-field + ponderomotive coupling only)
    # ------------------------------------------------------------------

    def _draw_noise(self, t: float) -> Array:
        """Deterministic per-step noise: key = fold_in(seed, round(t/dt)).
        Same t -> same realization, so RK4 stages within a step see frozen noise
        when the draw happens once at the step head."""
        step = jnp.uint32(jnp.round(t / self.dt))
        key = jax.random.fold_in(self.noise_base_key, step)
        white = jax.random.normal(key, self.noise_spatial_profile.shape)
        return self.noise_amplitude * white * self.noise_spatial_profile

    def _nonlinear_rhs(
        self, t: float, state: dict, a_frozen: Array, args: dict, noise_frozen: Array | float = 0.0
    ) -> dict:
        """Compute the nonlinear part of dCk/dt for both species.

        a_frozen: interior vector potential (Nx,) frozen from start of step.
        Returns a dict with same keys as state but only Ck entries nonzero.
        """
        Ck_e = state["Ck_electrons"].view(jnp.complex128)
        Ck_i = state["Ck_ions"].view(jnp.complex128)

        # Electrostatic field from Poisson
        Ex = self.poisson(Ck_e, Ck_i)  # (Nx,)

        # External longitudinal E-field driver, evaluated at this substep time.
        # Lawson-RK4 calls _nonlinear_rhs at t, t+dt/2, t+dt, so this captures the
        # driver's time variation within the step (same as vlasov1d's dex_array).
        Edrive = self.ex_driver(t, args)  # (Nx,)

        # Ponderomotive force per unit charge²/mass: -0.5 * d(a^2)/dx.
        # This term is ∝ q²/m² — charge-sign INDEPENDENT. It must NOT be folded
        # under the q/m factor that multiplies the E-field terms: doing so flips
        # its sign for electrons, which flips the sign of one leg of the SRS
        # three-wave coupling (γ² ∝ product of legs) — Stokes backscatter becomes
        # structurally stable (no SRS, ever) while the normally-stable anti-Stokes
        # pair goes unstable and detonates. Reference: vlasov1d does this
        # correctly (solvers/pushers/vlasov.py: force = q*e + (q²/m)*pond).
        a_sq = a_frozen**2
        Fp = -0.5 * jnp.gradient(a_sq, self.dx)  # (Nx,)

        # Per-species acceleration: (q/m)·(Ex + Edrive + noise) + (q/m)²·Fp,
        # passed with q_over_m=1 so the coupling applies no further charge factor.
        accel_e = self.qm_e * (Ex + Edrive + noise_frozen) + self.qm_e**2 * Fp
        dCk_e = _hermite_e_coupling(
            Ck_e,
            accel_e,
            self.sqrt_n_minus_e,
            1.0,
            self.alpha_e,
            self.Omega_ce_tau,
            mask23=self.mask23,
        )

        if self.static_ions:
            dCk_i = jnp.zeros_like(state["Ck_ions"])
        else:
            # Ion ponderomotive is (q/m)² = 1/mi_me² — kept for correctness, negligible.
            accel_i = self.qm_i * (Ex + Edrive + noise_frozen) + self.qm_i**2 * Fp
            dCk_i = _hermite_e_coupling(
                Ck_i,
                accel_i,
                self.sqrt_n_minus_i,
                1.0,
                self.alpha_i,
                self.Omega_ce_tau,
                mask23=self.mask23,
            ).view(jnp.float64)

        # Real-valued fields have zero nonlinear derivative in the Lawson frame
        # (wave solver is handled outside the Lawson loop)
        out = dict(state)
        out["Ck_electrons"] = dCk_e.view(jnp.float64)
        out["Ck_ions"] = dCk_i
        # Zero out non-Ck entries so tree arithmetic works cleanly
        for k in ["a", "prev_a", "e", "da", "de"]:
            if k in out:
                out[k] = jnp.zeros_like(state[k])
        return out

    # ------------------------------------------------------------------
    # Lawson-RK4 for Ck
    # ------------------------------------------------------------------

    def _lawson_rk4(self, t: float, state: dict, a_frozen: Array, args: dict) -> dict:
        """One Lawson-RK4 step for Ck (wave-eq fields not touched)."""
        dt = self.dt
        exp_L = self.combined_exp.apply

        if self.noise_amplitude > 0.0 and self.noise_spatial_profile is not None:
            noise_frozen = self._draw_noise(t)
        else:
            noise_frozen = 0.0

        Eh_y = exp_L(state, dt / 2)
        Ef_y = exp_L(state, dt)

        N1 = self._nonlinear_rhs(t, state, a_frozen, args, noise_frozen)

        Eh_N1 = exp_L(N1, dt / 2)
        y_star = _tree_add(Eh_y, _tree_scale(Eh_N1, dt / 2))
        N2 = self._nonlinear_rhs(t + dt / 2, y_star, a_frozen, args, noise_frozen)

        y_dstar = _tree_add(Eh_y, _tree_scale(N2, dt / 2))
        N3 = self._nonlinear_rhs(t + dt / 2, y_dstar, a_frozen, args, noise_frozen)

        Eh_N3 = exp_L(N3, dt / 2)
        y_tstar = _tree_add(Ef_y, _tree_scale(Eh_N3, dt))
        N4 = self._nonlinear_rhs(t + dt, y_tstar, a_frozen, args, noise_frozen)

        Ef_N1 = exp_L(N1, dt)
        Eh_N2 = exp_L(N2, dt / 2)

        weighted = _tree_scale(
            _tree_add(
                _tree_add(Ef_N1, _tree_scale(Eh_N2, 2.0)),
                _tree_add(_tree_scale(Eh_N3, 2.0), N4),
            ),
            dt / 6.0,
        )
        return _tree_add(Ef_y, weighted)

    # ------------------------------------------------------------------
    # Sponge damping
    # ------------------------------------------------------------------

    def _apply_sponge(self, state: dict) -> dict:
        """Apply exponential sponge damping post-step."""
        dt = self.dt
        out = dict(state)

        if self.sponge_plasma is not None:
            # Damp all non-density Hermite modes in real space
            for key, alpha in [("Ck_electrons", self.alpha_e), ("Ck_ions", self.alpha_i)]:
                Ck = out[key].view(jnp.complex128)  # (Nn, Nx)
                Nn = Ck.shape[0]
                C = jnp.fft.ifft(Ck, axis=-1, norm="forward")
                # Zero mode (density) undamped; all others damped
                n_idx = jnp.arange(Nn)[:, None]  # (Nn, 1)
                damping = jnp.where(n_idx == 0, 1.0, jnp.exp(-self.sponge_plasma[None, :] * dt))
                C_damped = C * damping
                out[key] = jnp.fft.fft(C_damped, axis=-1, norm="forward").view(jnp.float64)

        if self.sponge_fields is not None:
            # Damp vector potential interior (not boundary cells)
            a = out["a"]
            damping = jnp.exp(-self.sponge_fields * dt)
            a_interior = a[1:-1] * damping
            out["a"] = a.at[1:-1].set(a_interior)
            prev_a = out["prev_a"]
            prev_a_interior = prev_a[1:-1] * damping
            out["prev_a"] = prev_a.at[1:-1].set(prev_a_interior)

        return out

    # ------------------------------------------------------------------
    # Top-level __call__ (Stepper convention: returns new state)
    # ------------------------------------------------------------------

    def __call__(self, t: float, y: dict, args: dict) -> dict:
        """Advance state by one timestep dt. Returns new state dict."""
        if self.integrator == "strang-exp":
            return self._strang_exp_step(t, y, args)
        return self._lawson_step(t, y, args)

    def _strang_exp_step(self, t: float, y: dict, args: dict) -> dict:
        """Strang split: exp(L·dt/2) → exact force exponential → exp(L·dt/2) + leapfrog wave.

        All couplings time-centered; see class docstring.
        """
        dt = self.dt
        exp_L = self.combined_exp.apply

        if self.noise_amplitude > 0.0 and self.noise_spatial_profile is not None:
            noise_frozen = self._draw_noise(t)
        else:
            noise_frozen = 0.0

        # Wave-equation inputs at t_n — the leapfrog update
        # a_{n+1} = 2a_n − a_{n−1} + dt²(∂xx a − n_e·a + S) is centered at t_n,
        # so n_e and S belong at t_n (pre-step C_0), not averaged/half-shifted.
        Ck_e_n = y["Ck_electrons"].view(jnp.complex128)
        n_e_n = self.poisson.electron_density(Ck_e_n)
        if self.wave_density_max is not None:
            n_e_n = jnp.clip(n_e_n, 0.0, self.wave_density_max)
        djy = self.ey_driver(t, args)

        # 1. First linear half-step (streaming + collisions + filter, exact)
        y_h = exp_L(y, dt / 2)
        Ck_e_h = y_h["Ck_electrons"].view(jnp.complex128)
        Ck_i_h = y_h["Ck_ions"].view(jnp.complex128)

        # 2. Time-centered force substep over the full dt.
        # Ex from the mid-step C_0 (streaming has advanced it to t+dt/2; the
        # force operator itself never changes C_0, so Ex is constant across
        # the substep and the substep is linear with frozen coefficients).
        Ex = self.poisson(Ck_e_h, Ck_i_h)
        Edrive = self.ex_driver(t + 0.5 * dt, args)
        # Vector potential linearly extrapolated to t+dt/2 from (a_n, a_{n-1});
        # error O((ω0·dt)²) vs the half-step-early freeze's O(ω0·dt).
        a_mid = 1.5 * y["a"][1:-1] - 0.5 * y["prev_a"][1:-1]
        Fp = -0.5 * jnp.gradient(a_mid**2, self.dx)

        accel_e = self.qm_e * (Ex + Edrive + noise_frozen) + self.qm_e**2 * Fp
        if self.force_cap is not None:
            accel_e = self.force_cap * jnp.tanh(accel_e / self.force_cap)
        Ck_e_f = _exp_force_substep(
            Ck_e_h,
            accel_e,
            self.sqrt_n_minus_e,
            self.alpha_e,
            self.Omega_ce_tau,
            dt,
            mask23=self.mask23,
            n_terms=self.force_exp_terms,
        )
        if self.static_ions:
            Ck_i_f_view = y_h["Ck_ions"]
        else:
            accel_i = self.qm_i * (Ex + Edrive + noise_frozen) + self.qm_i**2 * Fp
            if self.force_cap is not None:
                accel_i = self.force_cap * jnp.tanh(accel_i / self.force_cap)
            Ck_i_f_view = _exp_force_substep(
                Ck_i_h,
                accel_i,
                self.sqrt_n_minus_i,
                self.alpha_i,
                self.Omega_ce_tau,
                dt,
                mask23=self.mask23,
                n_terms=self.force_exp_terms,
            ).view(jnp.float64)

        y_f = dict(y_h)
        y_f["Ck_electrons"] = Ck_e_f.view(jnp.float64)
        y_f["Ck_ions"] = Ck_i_f_view

        # 3. Second linear half-step
        y1 = exp_L(y_f, dt / 2)

        # 4. Wave solver (inputs at t_n, computed above)
        a_result = self.wave_solver(
            a=y["a"],
            aold=y["prev_a"],
            djy_array=djy,
            electron_density=n_e_n,
        )

        # 5. Diagnostics + assemble
        Ck_e_np1 = y1["Ck_electrons"].view(jnp.complex128)
        Ck_i_np1 = y1["Ck_ions"].view(jnp.complex128)
        Ex_out = self.poisson(Ck_e_np1, Ck_i_np1)
        de = self.ex_driver(t, args)

        new_state = {
            "Ck_electrons": y1["Ck_electrons"],
            "Ck_ions": y1["Ck_ions"],
            "a": a_result["a"],
            "prev_a": a_result["prev_a"],
            "e": Ex_out,
            "da": djy,
            "de": de,
        }
        return self._apply_sponge(new_state)

    def _lawson_step(self, t: float, y: dict, args: dict) -> dict:
        """Legacy Lawson-RK4 step (explicit force; see class docstring for caveats)."""

        # Frozen interior vector potential for the whole Lawson step
        a_frozen = y["a"][1:-1]  # (Nx,)

        # 1. Lawson-RK4 for Ck
        y_after_lawson = self._lawson_rk4(t, y, a_frozen, args)

        # 2. Wave solver for a
        # Electron density at start and end of Ck step (for wave eq. plasma term)
        Ck_e_n = y["Ck_electrons"].view(jnp.complex128)
        Ck_e_np1 = y_after_lawson["Ck_electrons"].view(jnp.complex128)
        Ck_i_n = y["Ck_ions"].view(jnp.complex128)
        Ck_i_np1 = y_after_lawson["Ck_ions"].view(jnp.complex128)

        n_e_n = self.poisson.electron_density(Ck_e_n)
        n_e_np1 = self.poisson.electron_density(Ck_e_np1)
        # Wave eq. plasma term: averaged electron density (ωpe² * a)
        electron_density_avg = 0.5 * (n_e_n + n_e_np1)

        # Transverse current driver (evaluated at half-step for centered scheme)
        djy = self.ey_driver(t + 0.5 * self.dt, args)

        a_result = self.wave_solver(
            a=y["a"],
            aold=y["prev_a"],
            djy_array=djy,
            electron_density=electron_density_avg,
        )

        # 3. Electrostatic field for diagnostics/output
        Ex = self.poisson(Ck_e_np1, Ck_i_np1)

        # External longitudinal driver field at the start of the step (diagnostic only;
        # it is applied to the force inside the Lawson loop, not here).
        de = self.ex_driver(t, args)

        # 4. Assemble new state
        new_state = {
            "Ck_electrons": y_after_lawson["Ck_electrons"],
            "Ck_ions": y_after_lawson["Ck_ions"],
            "a": a_result["a"],
            "prev_a": a_result["prev_a"],
            "e": Ex,
            "da": djy,
            "de": de,
        }

        # 5. Apply sponge damping
        new_state = self._apply_sponge(new_state)

        return new_state
