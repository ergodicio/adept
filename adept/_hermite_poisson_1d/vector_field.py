"""
Core vector field for the 1D Hermite-Poisson solver.

State: Ck_electrons(Nn_e, Nx), Ck_ions(Nn_i, Nx) [complex, stored as float64 views],
       a(Nx+2), prev_a(Nx+2) [real, vector potential with boundary cells],
       e(Nx) [real, electrostatic field for diagnostics].

Integration: Lawson-RK4 for Ck (free-streaming exact, E-field/ponderomotive explicit)
             + explicit leapfrog WaveSolver for a.
             Uses the Stepper (discrete-map) convention from adept._base_.
"""

import jax.numpy as jnp
from jax import Array

from adept._base_ import get_envelope
from adept._vlasov1d.solvers.pushers.field import WaveSolver


class TransverseWaveDriver:
    """Wave-equation source term S(x, t) = Σ_pulses -w² a0 envelope(x,t) sin(kx − wt).

    Reads directly from the raw normalized driver_config dict (same format as
    adept._spectrax1d.driver.Driver) so no Pydantic/EMDriver chain is needed.
    Output shape: (Nx+2,), matching WaveSolver's djy_array expectation.
    """

    def __init__(self, x_a: Array, ey_driver_cfg: dict):
        """
        Args:
            x_a: Extended x grid with ghost cells, shape (Nx+2,).
            ey_driver_cfg: dict mapping pulse_name → pulse_dict from cfg["drivers"]["ey"].
        """
        self.x_a = x_a
        # Pre-parse all scalars at init so __call__ contains no float() on JAX arrays.
        x_a_last = float(x_a[-1])
        parsed = []
        for _, pulse in (ey_driver_cfg.items() if isinstance(ey_driver_cfg, dict) else []):
            if not isinstance(pulse, dict):
                continue
            w_total = float(pulse["w0"]) + float(pulse.get("dw0", 0.0))
            t_center = float(pulse.get("t_center", 0.0))
            t_half = 0.5 * float(pulse.get("t_width", 1e10))
            t_rise = float(pulse.get("t_rise", 0.0))
            x_center = float(pulse.get("x_center", 0.5 * x_a_last))
            x_half = 0.5 * float(pulse.get("x_width", 1e10))
            x_rise = float(pulse.get("x_rise", 0.0))
            parsed.append((
                float(pulse["k0"]), w_total, float(pulse["a0"]),
                t_center, t_half, t_rise,
                x_center, x_half, x_rise,
            ))
        self.parsed_pulses = parsed

    def __call__(self, t: float, args) -> Array:
        total = jnp.zeros_like(self.x_a)
        for k0, w_total, a0, t_center, t_half, t_rise, x_center, x_half, x_rise in self.parsed_pulses:
            env_t = get_envelope(t_rise, t_rise, t_center - t_half, t_center + t_half, t)
            env_x = get_envelope(x_rise, x_rise, x_center - x_half, x_center + x_half, self.x_a)
            total = total + env_t * env_x * (-(w_total ** 2)) * a0 * jnp.sin(k0 * self.x_a - w_total * t)
        return total


# ---------------------------------------------------------------------------
# Linear exponential operators for (Nn, Nx) arrays
# ---------------------------------------------------------------------------


class FreeStreamingExp1D:
    """Exact exponential for free-streaming in x (acts on the n-axis, varies with kx).

    Prediagonalizes the symmetric tridiagonal streaming matrix T where
        T[n, n+1] = T[n+1, n] = sqrt((n+1)/2)   (+ diagonal drift/alpha term)
    and stores V (eigenvectors) and eigenvalues so that
        exp(-i*kx*alpha/Lx * T * s) = V * diag(exp(prefactor * kx * eigenvalues * s)) * V^T.
    """

    def __init__(self, Nn: int, alpha: float, u: float, Lx: float, kx_1d: Array):
        """
        Args:
            Nn: Number of Hermite modes.
            alpha: Thermal velocity (vth/c in skin-depth units).
            u: Drift velocity (usually 0).
            Lx: Domain length in normalized units.
            kx_1d: 1D wavenumber array, shape (Nx,), values 2*pi*fftfreq(Nx)*Nx.
        """
        self.prefactor = -1j * float(alpha) / float(Lx)
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
        col_fac = -self.nu * self.col_1d[:, None] * s       # (Nn, Nx)
        diff_fac = -self.D * self.kx_sq_1d[None, :] * s    # (1, Nx)
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
        return (self.alpha_e ** 3) * C0_e

    def ion_density(self, Ck_i: Array) -> Array:
        """n_i(x) from Hermite coefficient C_0 in k-space (or static background)."""
        if self.static_ion_density is not None:
            return self.static_ion_density
        C0_i = jnp.fft.ifft(Ck_i[0, :], norm="forward").real
        return (self.alpha_i ** 3) * C0_i

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
) -> Array:
    """Compute E-field (+ ponderomotive) coupling term for one species.

    dCk_n/dt|E = (q/m) * Omega_ce_tau * FFT[sqrt(n)*sqrt(2)/alpha * F(x) * C[n+1](x)]

    where C[n+1] means the real-space coefficient at mode n+1 (zero-padded at Nn).

    Args:
        Ck: (Nn, Nx) complex, in k-space.
        F_total: (Nx,) real, total force field (Ex + ponderomotive).
        sqrt_n_minus: (Nn,) = sqrt([0,1,...,Nn-1]).
        q_over_m: Charge-to-mass ratio (e.g. -1 for electrons).
        alpha: Thermal velocity.
        Omega_ce_tau: Normalization constant (1.0 in current configs).
    """
    Nn, Nx = Ck.shape
    # IFFT to real space: C(n, x)
    C = jnp.fft.ifft(Ck, axis=-1, norm="forward")  # (Nn, Nx) complex → real part matters

    # Shift down by 1: result[n] = C[n+1], zero at n=Nn-1
    C_down = jnp.concatenate([C[1:, :], jnp.zeros((1, Nx), dtype=C.dtype)], axis=0)

    # Coupling: sqrt(n)*sqrt(2)/alpha * F(x) * C[n+1](x)
    integrand = (
        sqrt_n_minus[:, None] * jnp.sqrt(2.0) / alpha
        * F_total[None, :]
        * C_down
    )  # (Nn, Nx)

    # FFT back and apply q/m * Omega_ce_tau
    return q_over_m * Omega_ce_tau * jnp.fft.fft(integrand, axis=-1, norm="forward")


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

    Uses Lawson-RK4 for Ck (free-streaming exact, E-field/ponderomotive explicit)
    and explicit leapfrog WaveSolver for the transverse vector potential a.

    Called by adept._base_.Stepper which treats the return value as the new state
    (not dy/dt). dt is stored internally from grid.dt at construction time.
    """

    def __init__(
        self,
        combined_exp: CombinedLinearExp1D,
        poisson: PoissonSolver1D,
        wave_solver: WaveSolver,
        ey_driver: TransverseWaveDriver,
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
    ):
        self.combined_exp = combined_exp
        self.poisson = poisson
        self.wave_solver = wave_solver
        self.ey_driver = ey_driver

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

    # ------------------------------------------------------------------
    # Nonlinear RHS (E-field + ponderomotive coupling only)
    # ------------------------------------------------------------------

    def _nonlinear_rhs(self, t: float, state: dict, a_frozen: Array, args: dict) -> dict:
        """Compute the nonlinear part of dCk/dt for both species.

        a_frozen: interior vector potential (Nx,) frozen from start of step.
        Returns a dict with same keys as state but only Ck entries nonzero.
        """
        Ck_e = state["Ck_electrons"].view(jnp.complex128)
        Ck_i = state["Ck_ions"].view(jnp.complex128)

        # Electrostatic field from Poisson
        Ex = self.poisson(Ck_e, Ck_i)  # (Nx,)

        # Ponderomotive force: -0.5 * d(a^2)/dx  (on electrons)
        a_sq = a_frozen ** 2
        Fp = -0.5 * jnp.gradient(a_sq, self.dx)  # (Nx,)

        # Electron E-field + ponderomotive coupling
        dCk_e = _hermite_e_coupling(
            Ck_e, Ex + Fp,
            self.sqrt_n_minus_e, self.qm_e, self.alpha_e, self.Omega_ce_tau,
        )

        if self.static_ions:
            dCk_i = jnp.zeros_like(state["Ck_ions"])
        else:
            dCk_i = _hermite_e_coupling(
                Ck_i, Ex,  # ponderomotive negligible for ions
                self.sqrt_n_minus_i, self.qm_i, self.alpha_i, self.Omega_ce_tau,
            ).view(jnp.float64)

        # Real-valued fields have zero nonlinear derivative in the Lawson frame
        # (wave solver is handled outside the Lawson loop)
        out = dict(state)
        out["Ck_electrons"] = dCk_e.view(jnp.float64)
        out["Ck_ions"] = dCk_i if self.static_ions else dCk_i
        # Zero out non-Ck entries so tree arithmetic works cleanly
        for k in ["a", "prev_a", "e", "da"]:
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

        Eh_y = exp_L(state, dt / 2)
        Ef_y = exp_L(state, dt)

        N1 = self._nonlinear_rhs(t, state, a_frozen, args)

        Eh_N1 = exp_L(N1, dt / 2)
        y_star = _tree_add(Eh_y, _tree_scale(Eh_N1, dt / 2))
        N2 = self._nonlinear_rhs(t + dt / 2, y_star, a_frozen, args)

        y_dstar = _tree_add(Eh_y, _tree_scale(N2, dt / 2))
        N3 = self._nonlinear_rhs(t + dt / 2, y_dstar, a_frozen, args)

        Eh_N3 = exp_L(N3, dt / 2)
        y_tstar = _tree_add(Ef_y, _tree_scale(Eh_N3, dt))
        N4 = self._nonlinear_rhs(t + dt, y_tstar, a_frozen, args)

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

        # 4. Assemble new state
        new_state = {
            "Ck_electrons": y_after_lawson["Ck_electrons"],
            "Ck_ions": y_after_lawson["Ck_ions"],
            "a": a_result["a"],
            "prev_a": a_result["prev_a"],
            "e": Ex,
            "da": djy,
        }

        # 5. Apply sponge damping
        new_state = self._apply_sponge(new_state)

        return new_state
