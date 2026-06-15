"""
Core vector field for the 1D mixed Hermite-Legendre Vlasov-Poisson solver.

Implements the mixed method of Issan, Delzanno & Roytershteyn (arXiv:2606.12322,
"Mixed Hermite-Legendre spectral method for kinetic plasma simulations").

The electron distribution is split f = f0 + df, with
  f0(x, v, t) = sum_{n=0}^{Nh-1} C_n(x, t) psi_n(v; alpha, u)      [AW Hermite]
  df(x, v, t) = sum_{m=0}^{Nl-1} B_m(x, t) xi_m(v; v_a, v_b)        [Legendre]

evolved by the coupled system (paper eqns 23-25) on a periodic spatial domain.
Spatial dependence is carried in Fourier space (d_x -> i*kx); velocity is spectral.

Normalization (paper sec 2.1): t*wpe, x/lambda_D, v/vthe, phi*e*lambda_D/Te, so
the electric field is E = -d_x phi and the ion background density is 1.

Integration: Lawson-RK4 (free-streaming exact for both bases via prediagonalized
symmetric-tridiagonal streaming matrices; E-field force, Dirichlet penalty, and the
Hermite->Legendre coupling are explicit). Uses the Stepper (discrete-map) convention
from adept._base_, mirroring adept._hermite_poisson_1d.vector_field.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from adept._base_ import get_envelope

# ---------------------------------------------------------------------------
# External longitudinal field driver
# ---------------------------------------------------------------------------


class ExternalExDriver:
    """Prescribed longitudinal field E_drive(x, t) added to the velocity-space force.

    E_drive(x, t) = sum_pulses env(x, t) (w0+dw0) a0 sin(k0 x - (w0+dw0) t)

    The driver enters only the E.d_v f force term (it drives EPWs directly, e.g. a
    resonant kick for a Landau-damping measurement) and never the Poisson solve, so
    the self-consistent field energy diagnostic excludes it. Mirrors the convention
    of adept._hermite_poisson_1d.vector_field.LongitudinalElectricFieldDriver and of
    the vlasov1d ``ex`` driver. Reads cfg["drivers"]["ex"] (pulse_name -> pulse_dict).
    Output shape (Nx,), on the interior x grid.
    """

    def __init__(self, x: Array, ex_driver_cfg: dict):
        self.x = x
        x_last = float(x[-1])
        parsed = []
        for _, pulse in ex_driver_cfg.items() if isinstance(ex_driver_cfg, dict) else []:
            if not isinstance(pulse, dict):
                continue
            parsed.append(
                (
                    float(pulse["k0"]),
                    float(pulse["w0"]) + float(pulse.get("dw0", 0.0)),
                    float(pulse["a0"]),
                    float(pulse.get("t_center", 0.0)),
                    0.5 * float(pulse.get("t_width", 1e10)),
                    float(pulse.get("t_rise", 0.0)),
                    float(pulse.get("x_center", 0.5 * x_last)),
                    0.5 * float(pulse.get("x_width", 1e10)),
                    float(pulse.get("x_rise", 0.0)),
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
# Basis constants
# ---------------------------------------------------------------------------


def safe_col(N: int) -> Array:
    """Artificial-collision profile col[n] = n(n-1)(n-2) / ((N-1)(N-2)(N-3)).

    This is the Lenard-Bernstein-based hyper-collision spectrum of paper sec 2.5:
    cubic in the mode index, normalized to 1 at n = N-1, and identically zero for
    n = 0, 1, 2 so the operator conserves mass, momentum, and energy exactly. The
    damping is applied per Lawson substep as exp(-nu * col[n] * s).
    """
    n = jnp.arange(N, dtype=jnp.float64)
    term = n * (n - 1) * (n - 2)
    denom = (N - 1) * (N - 2) * (N - 3) if N > 3 else 1.0
    return jnp.where(N > 3, term / denom, jnp.zeros(N, dtype=jnp.float64))


def hermite_streaming_matrix(Nh: int, u: float, alpha: float) -> np.ndarray:
    """Symmetric tridiagonal T_H for AW-Hermite free streaming (paper eqn 7).

    d_t C_n + alpha*sqrt((n+1)/2) d_x C_{n+1} + alpha*sqrt(n/2) d_x C_{n-1}
            + u d_x C_n = 0   ==>   d_t C = -alpha * T_H * (d_x C),
    with off-diagonal sqrt((n+1)/2) and diagonal u/alpha. The scalar prefactor
    -i*alpha*kx is applied by StreamingExp1D.
    """
    n = np.arange(Nh, dtype=np.float64)
    T = np.zeros((Nh, Nh), dtype=np.float64)
    if Nh > 1:
        off = np.sqrt((n[:-1] + 1.0) / 2.0)  # sqrt((n+1)/2), n = 0..Nh-2
        T += np.diag(off, 1) + np.diag(off, -1)
    T += np.diag(np.full(Nh, u / alpha))
    return T


def legendre_constants(Nl: int, v_a: float, v_b: float) -> dict:
    """Constants for the shifted/scaled Legendre basis xi_m on [v_a, v_b].

    Returns:
      T_L:    symmetric tridiagonal velocity matrix (paper eqn 11),
              off-diagonal sigma_{n+1}, diagonal sigma_bar = (v_a+v_b)/2.
      sigma:  sub/super-diagonal coefficients sigma_n (sigma_0 = 0).
      sigma_bar: scalar (v_a+v_b)/2.
      deriv:  strictly-lower-triangular derivative matrix sigma_{m,i} (paper eqn 10),
              d xi_m / dv = sum_{i<m} sigma_{m,i} xi_i.
      xi_b:   boundary values xi_m(v_b) = sqrt(2m+1).
      xi_a:   boundary values xi_m(v_a) = sqrt(2m+1) (-1)^m.
    """
    width = float(v_b - v_a)
    sigma_bar = 0.5 * (v_a + v_b)

    n = np.arange(Nl, dtype=np.float64)
    sigma = np.zeros(Nl, dtype=np.float64)
    if Nl > 1:
        sigma[1:] = (width / 2.0) * n[1:] / np.sqrt((2.0 * n[1:] + 1.0) * (2.0 * n[1:] - 1.0))

    T_L = np.diag(np.full(Nl, sigma_bar))
    if Nl > 1:
        off = sigma[1:]  # T_L[m, m+1] = T_L[m+1, m] = sigma_{m+1}
        T_L += np.diag(off, 1) + np.diag(off, -1)

    # Derivative matrix sigma_{m,i}: nonzero only when (m - i) is odd, i < m.
    deriv = np.zeros((Nl, Nl), dtype=np.float64)
    for m in range(Nl):
        for i in range(m):
            if (m - i) % 2 == 1:
                deriv[m, i] = 2.0 * np.sqrt((2.0 * m + 1.0) * (2.0 * i + 1.0)) / width

    sqrt_2m1 = np.sqrt(2.0 * n + 1.0)
    xi_b = sqrt_2m1
    xi_a = sqrt_2m1 * ((-1.0) ** n)

    return {
        "T_L": T_L,
        "sigma": jnp.asarray(sigma),
        "sigma_bar": float(sigma_bar),
        "deriv": jnp.asarray(deriv),
        "xi_b": jnp.asarray(xi_b),
        "xi_a": jnp.asarray(xi_a),
    }


def _hermite_function_values(Nh_plus_1: int, v: np.ndarray, u: float, alpha: float) -> np.ndarray:
    """psi_n(v; u, alpha) for n = 0..Nh, shape (Nh+1, len(v)). Used for J integrals.

    psi_n = (pi 2^n n!)^{-1/2} H_n(z) exp(-z^2),  z = (v - u)/alpha. Built from the
    *normalized* three-term recurrence

        psi_0 = pi^{-1/2} exp(-z^2),   psi_1 = sqrt(2) z psi_0,
        psi_{n+1} = z sqrt(2/(n+1)) psi_n - sqrt(n/(n+1)) psi_{n-1},

    which keeps every psi_n O(1). Forming H_n and 2^n n! separately (as a naive
    implementation would) overflows float64 for n >= 171, silently zeroing the
    high-order coupling integrals.
    """
    z = (v - u) / alpha
    psi = np.zeros((Nh_plus_1, v.size), dtype=np.float64)
    psi[0] = np.pi**-0.5 * np.exp(-(z**2))
    if Nh_plus_1 > 1:
        psi[1] = np.sqrt(2.0) * z * psi[0]
    for n in range(1, Nh_plus_1 - 1):
        psi[n + 1] = z * np.sqrt(2.0 / (n + 1)) * psi[n] - np.sqrt(n / (n + 1)) * psi[n - 1]
    return psi


def _legendre_basis_values(Nl: int, v: np.ndarray, v_a: float, v_b: float) -> np.ndarray:
    """xi_m(v; v_a, v_b) = sqrt(2m+1) L_m(s), s = (2v-(v_a+v_b))/(v_b-v_a),
    for m = 0..Nl-1, shape (Nl, len(v)). Built from the Legendre recurrence."""
    s = (2.0 * v - (v_a + v_b)) / (v_b - v_a)
    L = np.zeros((Nl, v.size), dtype=np.float64)
    L[0] = 1.0
    if Nl > 1:
        L[1] = s
    for k in range(1, Nl - 1):
        L[k + 1] = ((2.0 * k + 1.0) * s * L[k] - k * L[k - 1]) / (k + 1.0)
    scale = np.sqrt(2.0 * np.arange(Nl) + 1.0)
    return scale[:, None] * L


def hermite_legendre_coupling_vector(
    Nh: int, Nl: int, alpha: float, u: float, v_a: float, v_b: float, enforce_conservation: bool = True
) -> Array:
    """J_{Nh, m} = integral_{v_a}^{v_b} psi_{Nh}(v; alpha, u) xi_m(v) dv, m = 0..Nl-1.

    This is the one-way coupling from the (closed-off) highest Hermite coefficient
    C_{Nh-1} into the Legendre modes (paper eqn 24). Evaluated by Gauss-Legendre
    quadrature on [v_a, v_b]. When ``enforce_conservation`` is set, J_{Nh,0..2} are
    zeroed (paper sec 3.4/4): removing the coupling to the first three Legendre
    coefficients makes the discrete method conserve mass, momentum, and energy
    independent of Nh parity and the spectral shift/scale parameters.
    """
    deg = max(4 * (Nh + Nl), 200)
    nodes, weights = np.polynomial.legendre.leggauss(deg)
    v = 0.5 * (v_b - v_a) * nodes + 0.5 * (v_b + v_a)
    w = 0.5 * (v_b - v_a) * weights

    psi_Nh = _hermite_function_values(Nh + 1, v, u, alpha)[Nh]  # (len(v),)
    xi = _legendre_basis_values(Nl, v, v_a, v_b)  # (Nl, len(v))
    J = xi @ (psi_Nh * w)  # (Nl,)

    if enforce_conservation:
        J[: min(3, Nl)] = 0.0
    return jnp.asarray(J)


# ---------------------------------------------------------------------------
# Exact exponential operators
# ---------------------------------------------------------------------------


class StreamingExp1D:
    """Exact free-streaming exponential exp(prefactor * kx * T * s) for one basis.

    Prediagonalizes the symmetric tridiagonal streaming matrix T (Hermite or
    Legendre) once, then applies exp(L*s) to a coefficient array (Nmodes, Nx) by
    rotating into the eigenbasis, scaling by exp(prefactor * kx * eigval * s), and
    rotating back. Mirrors adept._hermite_poisson_1d.vector_field.FreeStreamingExp1D
    (Hermite: prefactor = -i*alpha; Legendre: prefactor = -i).
    """

    def __init__(self, T: np.ndarray, prefactor: complex, kx_1d: Array):
        self.prefactor = prefactor
        self.kx_1d = kx_1d
        eigvals, V = np.linalg.eigh(T)
        self.V = jnp.asarray(V)
        self.eigenvalues = jnp.asarray(eigvals)

    def apply(self, Ck: Array, s: float) -> Array:
        C_eig = self.V.T @ Ck  # (Nmodes, Nx)
        exp_fac = jnp.exp(self.prefactor * s * self.eigenvalues[:, None] * self.kx_1d[None, :])
        return self.V @ (C_eig * exp_fac)


class DiagonalCollisionExp1D:
    """Exact exponential exp(-nu * col[n] * s), diagonal in the mode index."""

    def __init__(self, nu: float, col: Array):
        self.nu = float(nu)
        self.col = col

    def apply(self, Ck: Array, s: float) -> Array:
        if self.nu == 0.0:
            return Ck
        return Ck * jnp.exp(-self.nu * self.col[:, None] * s)


class CombinedLinearExp1D:
    """Applies exp(L*s) to the full {Ck, Bk} state; real diagnostics pass through."""

    def __init__(
        self,
        hermite_stream: StreamingExp1D,
        legendre_stream: StreamingExp1D,
        hermite_coll: DiagonalCollisionExp1D,
        legendre_coll: DiagonalCollisionExp1D,
    ):
        self.hermite_stream = hermite_stream
        self.legendre_stream = legendre_stream
        self.hermite_coll = hermite_coll
        self.legendre_coll = legendre_coll

    def apply(self, state: dict, s: float) -> dict:
        Ck = state["Ck"].view(jnp.complex128)
        Bk = state["Bk"].view(jnp.complex128)
        Ck = self.hermite_coll.apply(self.hermite_stream.apply(Ck, s), s)
        Bk = self.legendre_coll.apply(self.legendre_stream.apply(Bk, s), s)
        out = dict(state)
        out["Ck"] = Ck.view(jnp.float64)
        out["Bk"] = Bk.view(jnp.float64)
        return out


# ---------------------------------------------------------------------------
# Poisson solver
# ---------------------------------------------------------------------------


class PoissonSolver1D:
    """Spectral Poisson solve for the mixed-method charge density (paper eqn 25).

    rho(x) = 1 - alpha*C_0(x) - (v_b - v_a)*B_0(x)   (immobile ion background = 1)
    -d_x^2 phi = rho  ==>  phi_k = rho_k / kx^2,  E = -d_x phi  ==>  E_k = -i rho_k / kx.
    The k=0 component is set to zero (quasineutral domain).
    """

    def __init__(self, one_over_kx: Array, kx_sq: Array, alpha: float, width: float):
        self.one_over_kx = one_over_kx
        self.kx_sq = kx_sq
        self.alpha = float(alpha)
        self.width = float(width)

    def _rho_k(self, Ck: Array, Bk: Array) -> Array:
        # C_0(x), B_0(x) from their Fourier representations (norm="forward" -> [0] is mean)
        n_f0 = self.alpha * jnp.fft.ifft(Ck[0], norm="forward").real
        n_df = self.width * jnp.fft.ifft(Bk[0], norm="forward").real
        rho = 1.0 - n_f0 - n_df
        return jnp.fft.fft(rho, norm="forward")

    def electric_field(self, Ck: Array, Bk: Array) -> Array:
        E_k = -1j * self.one_over_kx * self._rho_k(Ck, Bk)
        return jnp.fft.ifft(E_k, norm="forward").real

    def potential(self, Ck: Array, Bk: Array) -> Array:
        rho_k = self._rho_k(Ck, Bk)
        phi_k = jnp.where(self.kx_sq > 0, rho_k / jnp.where(self.kx_sq > 0, self.kx_sq, 1.0), 0.0)
        return jnp.fft.ifft(phi_k, norm="forward").real


# ---------------------------------------------------------------------------
# Explicit nonlinear terms (E-field force, penalty, Hermite->Legendre coupling)
# ---------------------------------------------------------------------------


def _to_real(Ck: Array, mask23: Array | None) -> Array:
    """IFFT (Nmodes, Nx) k-space coeffs to real space, optionally 2/3-dealiased."""
    if mask23 is not None:
        Ck = Ck * mask23[None, :]
    return jnp.fft.ifft(Ck, axis=-1, norm="forward")


def _hermite_force(Ck: Array, E: Array, sqrt_2n_over_alpha: Array, mask23: Array | None) -> Array:
    """d_t C_n |force = -(sqrt(2n)/alpha) E(x) C_{n-1}(x)  (electron, paper eqn 19/23).

    E = -d_x phi is the self-consistent field; the n-th coefficient is forced by the
    (n-1)-th (Hermite differentiation raises the index). Returns k-space (Nh, Nx).
    """
    Nh, Nx = Ck.shape
    if mask23 is not None:
        E = jnp.fft.ifft(jnp.fft.fft(E, norm="forward") * mask23, norm="forward").real
    C = _to_real(Ck, mask23)
    C_up = jnp.concatenate([jnp.zeros((1, Nx), dtype=C.dtype), C[:-1, :]], axis=0)  # C_{n-1}
    integrand = -sqrt_2n_over_alpha[:, None] * E[None, :] * C_up
    out = jnp.fft.fft(integrand, axis=-1, norm="forward")
    if mask23 is not None:
        out = out * mask23[None, :]
    return out


def _legendre_force(
    Bk: Array,
    E: Array,
    deriv: Array,
    gamma_vec: Array,
    xi_a: Array,
    xi_b: Array,
    width: float,
    mask23: Array | None,
) -> Array:
    """d_t B_m |force = -E(x) [ sum_{i<m} sigma_{m,i} B_i - gamma_m delta_v[df xi_m] ]
    (paper eqn 24). The penalty enforces the weak Dirichlet BC df(v_a)=df(v_b)=0 with

      delta_v[df xi_m]_{v_a}^{v_b} = [ df(x,v_b) xi_m(v_b) - df(x,v_a) xi_m(v_a) ] / (v_b - v_a)

    where df(x,v_b) = sum_m B_m xi_m(v_b) and df(x,v_a) = sum_m B_m xi_m(v_a).
    Returns k-space (Nl, Nx).
    """
    Nl, Nx = Bk.shape
    if mask23 is not None:
        E = jnp.fft.ifft(jnp.fft.fft(E, norm="forward") * mask23, norm="forward").real
    B = _to_real(Bk, mask23)  # (Nl, Nx)

    deriv_term = deriv @ B  # sum_{i<m} sigma_{m,i} B_i, shape (Nl, Nx)

    df_b = jnp.tensordot(xi_b, B, axes=([0], [0]))  # df(x, v_b), shape (Nx,)
    df_a = jnp.tensordot(xi_a, B, axes=([0], [0]))  # df(x, v_a), shape (Nx,)
    penalty = (gamma_vec[:, None] / width) * (xi_b[:, None] * df_b[None, :] - xi_a[:, None] * df_a[None, :])

    integrand = -E[None, :] * (deriv_term - penalty)
    out = jnp.fft.fft(integrand, axis=-1, norm="forward")
    if mask23 is not None:
        out = out * mask23[None, :]
    return out


def _cross_coupling(
    Ck: Array, E: Array, kx_1d: Array, coupling_vec: Array, alpha: float, width: float, mask23: Array | None
) -> Array:
    """Hermite -> Legendre coupling (paper eqn 24, RHS):

      d_t B_m |coupling = -(alpha/width) J_{Nh,m} sqrt(Nh/2) [ d_x C_{Nh-1} + (2/alpha^2) E C_{Nh-1} ].

    The d_x C_{Nh-1} part is linear (i*kx in Fourier); the E*C_{Nh-1} part is the
    nonlinear product. Returns k-space (Nl, Nx). coupling_vec already folds in
    -(alpha/width) sqrt(Nh/2) J_{Nh,m} (and the J_{Nh,0..2}=0 conservation gate).
    """
    Nh = Ck.shape[0]
    C_last_k = Ck[Nh - 1]  # (Nx,) k-space
    dx_C = 1j * kx_1d * C_last_k  # d_x C_{Nh-1} in k-space

    C_last_real = _to_real(Ck[Nh - 1 : Nh], mask23)[0]  # (Nx,)
    if mask23 is not None:
        E = jnp.fft.ifft(jnp.fft.fft(E, norm="forward") * mask23, norm="forward").real
    nl_real = (2.0 / alpha**2) * E * C_last_real
    nl_k = jnp.fft.fft(nl_real, norm="forward")
    if mask23 is not None:
        nl_k = nl_k * mask23

    bracket = dx_C + nl_k  # (Nx,) k-space
    return coupling_vec[:, None] * bracket[None, :]  # (Nl, Nx)


# ---------------------------------------------------------------------------
# Implicit (IMEX) force operators
# ---------------------------------------------------------------------------


def hermite_force_operator(Nh: int, alpha: float) -> np.ndarray:
    """G_C with (G_C C)_n = -sqrt(2n)/alpha C_{n-1} (strictly lower-bidiagonal).

    The Hermite Lorentz force is dC/dt|force = E(x) G_C C. G_C is nilpotent, with
    operator norm ~ sqrt(2 Nh)/alpha * |E|, so explicit RK4 has a CFL-like limit
    that tightens with Nh. Backward Euler on it is unconditionally stable.
    """
    G = np.zeros((Nh, Nh), dtype=np.float64)
    n = np.arange(1, Nh)
    G[n, n - 1] = -np.sqrt(2.0 * n) / alpha
    return G


def legendre_force_operator(deriv: Array, gamma_vec: Array, xi_a: Array, xi_b: Array, width: float) -> np.ndarray:
    """G_B = P - D for the Legendre Lorentz force dB/dt|force = E(x) G_B B.

    D = deriv (strictly lower-triangular d_v matrix, paper eqn 10) and P is the
    rank-2 Dirichlet penalty P[m, j] = (gamma_m/width)(xi_b[m] xi_b[j] - xi_a[m] xi_a[j]).
    The d_v operator norm scales as ~ Nl^2/width, making this the *dominant*
    stiffness in the mixed method -- hence it is treated implicitly in IMEX.
    """
    D = np.asarray(deriv)
    g = np.asarray(gamma_vec) / width
    xa, xb = np.asarray(xi_a), np.asarray(xi_b)
    P = g[:, None] * (np.outer(xb, xb) - np.outer(xa, xa))
    return P - D


# ---------------------------------------------------------------------------
# Lawson-RK4 vector field (Stepper / discrete-map convention)
# ---------------------------------------------------------------------------


def _tree_add(a: dict, b: dict) -> dict:
    return jax.tree.map(lambda x, y: x + y, a, b)


def _tree_scale(a: dict, c: float) -> dict:
    return jax.tree.map(lambda x: c * x, a)


class HermiteLegendre1DVectorField:
    """Advances the mixed Hermite-Legendre state by one timestep dt.

    State dict (complex arrays stored as float64 views, like _hermite_poisson_1d):
      Ck: (Nh, Nx) Hermite-Fourier coefficients of f0
      Bk: (Nl, Nx) Legendre-Fourier coefficients of df
      e:   (Nx,) electric field   (diagnostic)
      phi: (Nx,) potential        (diagnostic)

    Called by adept._base_.Stepper, which treats the return value as the new state.
    """

    def __init__(
        self,
        combined_exp: CombinedLinearExp1D,
        poisson: PoissonSolver1D,
        kx_1d: Array,
        sqrt_2n_over_alpha: Array,
        deriv: Array,
        gamma_vec: Array,
        xi_a: Array,
        xi_b: Array,
        coupling_vec: Array,
        alpha: float,
        width: float,
        dt: float,
        mask23: Array | None = None,
        field_on: bool = True,
        ex_driver: "ExternalExDriver | None" = None,
        imex: bool = False,
        G_C: Array | None = None,
        G_B: Array | None = None,
        implicit: bool = False,
        T_H: Array | None = None,
        T_L: Array | None = None,
        col_e: Array | None = None,
        col_l: Array | None = None,
        nu_H: float = 0.0,
        nu_L: float = 0.0,
        newton_iters: int = 3,
        gmres_restart: int = 20,
        gmres_maxiter: int = 4,
        gmres_tol: float = 1e-8,
        precondition: bool = True,
    ):
        self.combined_exp = combined_exp
        self.poisson = poisson
        self.kx_1d = kx_1d
        self.sqrt_2n_over_alpha = sqrt_2n_over_alpha
        self.deriv = deriv
        self.gamma_vec = gamma_vec
        self.xi_a = xi_a
        self.xi_b = xi_b
        self.coupling_vec = coupling_vec
        self.alpha = float(alpha)
        self.width = float(width)
        self.dt = float(dt)
        self.mask23 = mask23
        self.field_on = bool(field_on)
        self.ex_driver = ex_driver
        # IMEX: treat the (stiff) E.d_v f Lorentz force implicitly via a frozen-E
        # backward-Euler substep, keeping the rest of the RHS in the explicit Lawson step.
        self.imex = bool(imex)
        self.G_C = None if G_C is None else jnp.asarray(G_C, dtype=jnp.complex128)
        self.G_B = None if G_B is None else jnp.asarray(G_B, dtype=jnp.complex128)
        # Implicit midpoint (AD-JFNK): needs the raw streaming/collision RHS operators.
        self.implicit = bool(implicit)
        self.T_H = None if T_H is None else jnp.asarray(T_H)
        self.T_L = None if T_L is None else jnp.asarray(T_L)
        self.col_e = None if col_e is None else jnp.asarray(col_e)
        self.col_l = None if col_l is None else jnp.asarray(col_l)
        self.nu_H = float(nu_H)
        self.nu_L = float(nu_L)
        self.newton_iters = int(newton_iters)
        self.gmres_restart = int(gmres_restart)
        self.gmres_maxiter = int(gmres_maxiter)
        self.gmres_tol = float(gmres_tol)
        self.precondition = bool(precondition)
        if self.implicit and self.precondition:
            self._setup_stream_preconditioner()

    def _setup_stream_preconditioner(self) -> None:
        """Precompute the per-k tridiagonal bands of M = I - (dt/2)(L_stream + L_coll).

        L_stream is block-diagonal in k and tridiagonal in mode (the streaming matrices
        T_H, T_L are tridiagonal), so M^{-1} is a batched tridiagonal solve. Including the
        diagonal collision term improves conditioning for collisional runs. This M
        captures the stiff (imaginary) streaming spectrum that otherwise cripples GMRES.
        """
        half = 0.5 * self.dt

        def bands(T, prefac, nu, col):
            T = jnp.asarray(T)
            diagT = jnp.diagonal(T)  # (N,)
            off = jnp.diagonal(T, offset=1)  # (N-1,)
            N = diagT.shape[0]
            c = prefac * half * self.kx_1d  # (Nx,) ; prefac = -1 * (-i alpha) etc. -> see below
            # M = I + (dt/2)*(i*coef*kx*T) + (dt/2)*nu*col   (coef = alpha for H, 1 for L)
            d = 1.0 + c[:, None] * diagT[None, :] + half * nu * col[None, :]
            sub = jnp.concatenate([jnp.zeros(1, dtype=off.dtype), off])  # sub[n]=T[n,n-1]
            sup = jnp.concatenate([off, jnp.zeros(1, dtype=off.dtype)])  # sup[n]=T[n,n+1]
            dl = c[:, None] * sub[None, :]
            du = c[:, None] * sup[None, :]
            return dl.astype(jnp.complex128), d.astype(jnp.complex128), du.astype(jnp.complex128)

        # L_stream C = -i*alpha*kx*(T_H C)  ->  M_H = I + i*(dt/2)*alpha*kx*T_H
        self._pc_h = bands(self.T_H, 1j * self.alpha, self.nu_H, self.col_e)
        # L_stream B = -i*kx*(T_L B)        ->  M_L = I + i*(dt/2)*kx*T_L
        self._pc_l = bands(self.T_L, 1j, self.nu_L, self.col_l)

    def _stream_precond_apply(self, v: dict) -> dict:
        """Apply M^{-1} (streaming+collision preconditioner) to a real (re,im) pytree."""
        Ck = v["Cr"] + 1j * v["Ci"]  # (Nh, Nx)
        Bk = v["Br"] + 1j * v["Bi"]  # (Nl, Nx)
        dl_h, d_h, du_h = self._pc_h
        dl_l, d_l, du_l = self._pc_l
        x = jax.lax.linalg.tridiagonal_solve(dl_h, d_h, du_h, Ck.T[..., None])[..., 0].T
        y = jax.lax.linalg.tridiagonal_solve(dl_l, d_l, du_l, Bk.T[..., None])[..., 0].T
        return {"Cr": x.real, "Ci": x.imag, "Br": y.real, "Bi": y.imag}

    def _force_precond_apply(self, v: dict, sC: Array) -> dict:
        """Apply (I - dt/2 E0 G_force)^{-1} (the Lorentz-force preconditioner), per x.

        sC = (dt/2) E0(x) is the frozen-field scale. The Hermite force operator is
        lower-bidiagonal -> a per-x tridiagonal (forward-substitution) solve; the
        Legendre force is dominated by the derivative matrix D (norm ~Nl^2/width), a
        per-x lower-triangular solve. The rank-2 Dirichlet penalty is left to GMRES (a
        preconditioner need not be exact). This is the piece that becomes stiff at
        saturation, where streaming-only preconditioning stalls.
        """
        Ck = v["Cr"] + 1j * v["Ci"]  # (Nh, Nx)
        Bk = v["Br"] + 1j * v["Bi"]  # (Nl, Nx)
        Nh, Nx = Ck.shape
        Nl = Bk.shape[0]
        sC_c = sC.astype(jnp.complex128)

        # Hermite: (I - sC G_C) is lower-bidiagonal; (I-sC G_C)[n,n-1] = sC*sqrt(2n)/alpha
        dl = sC_c[:, None] * self.sqrt_2n_over_alpha[None, :].astype(jnp.complex128)  # (Nx, Nh)
        dl = dl.at[:, 0].set(0.0)
        d = jnp.ones((Nx, Nh), dtype=jnp.complex128)
        du = jnp.zeros((Nx, Nh), dtype=jnp.complex128)
        Cn = jax.lax.linalg.tridiagonal_solve(dl, d, du, Ck.T[..., None])[..., 0].T

        # Legendre: (I + sC D) lower-triangular (unit diagonal); D = deriv matrix
        A0 = jnp.eye(Nl, dtype=jnp.complex128)[None] + sC_c[:, None, None] * self.deriv.astype(jnp.complex128)[None]
        Bn = jax.lax.linalg.triangular_solve(A0, Bk.T[..., None], left_side=True, lower=True)[..., 0].T
        return {"Cr": Cn.real, "Ci": Cn.imag, "Br": Bn.real, "Bi": Bn.imag}

    def _nonlinear_rhs(self, t: float, state: dict, args: dict) -> dict:
        Ck = state["Ck"].view(jnp.complex128)
        Bk = state["Bk"].view(jnp.complex128)

        # field_on=False -> pure advection (phi=0); the linear Hermite->Legendre
        # closure flux (d_x C_{Nh-1}) still acts, only E-dependent terms vanish.
        E = self.poisson.electric_field(Ck, Bk) if self.field_on else jnp.zeros(Ck.shape[1])  # (Nx,)
        # external longitudinal driver (evaluated at the substep time; never enters Poisson)
        if self.ex_driver is not None:
            E = E + self.ex_driver(t, args)

        cross = _cross_coupling(Ck, E, self.kx_1d, self.coupling_vec, self.alpha, self.width, self.mask23)
        if self.imex:
            # E.d_v f (Hermite + Legendre force) is handled by the implicit substep;
            # only the non-stiff Hermite->Legendre closure flux stays explicit.
            dCk = jnp.zeros_like(Ck)
            dBk = cross
        else:
            dCk = _hermite_force(Ck, E, self.sqrt_2n_over_alpha, self.mask23)
            dBk = (
                _legendre_force(Bk, E, self.deriv, self.gamma_vec, self.xi_a, self.xi_b, self.width, self.mask23)
                + cross
            )

        out = dict(state)
        out["Ck"] = dCk.view(jnp.float64)
        out["Bk"] = dBk.view(jnp.float64)
        for k in ("e", "phi", "de"):
            if k in out:
                out[k] = jnp.zeros_like(state[k])
        return out

    def _lawson_rk4(self, t: float, state: dict, args: dict) -> dict:
        dt = self.dt
        exp_L = self.combined_exp.apply

        Eh_y = exp_L(state, dt / 2)
        Ef_y = exp_L(state, dt)

        N1 = self._nonlinear_rhs(t, state, args)
        Eh_N1 = exp_L(N1, dt / 2)
        y_star = _tree_add(Eh_y, _tree_scale(Eh_N1, dt / 2))
        N2 = self._nonlinear_rhs(t + dt / 2, y_star, args)

        y_dstar = _tree_add(Eh_y, _tree_scale(N2, dt / 2))
        N3 = self._nonlinear_rhs(t + dt / 2, y_dstar, args)

        Eh_N3 = exp_L(N3, dt / 2)
        y_tstar = _tree_add(Ef_y, _tree_scale(Eh_N3, dt))
        N4 = self._nonlinear_rhs(t + dt, y_tstar, args)

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

    def _full_rhs_complex(self, t: float, Ck: Array, Bk: Array) -> tuple:
        """Raw RHS dy/dt = streaming + collisions + Lorentz force + closure coupling.

        Unlike the Lawson path (which integrates streaming/collisions exactly via the
        matrix exponential), the implicit-midpoint integrator needs the explicit RHS
        operators: streaming d_t C = -i alpha kx (T_H C), d_t B = -i kx (T_L B), the
        diagonal collision -nu col[n], and the (reused) E.d_v f force + coupling terms.
        """
        E = self.poisson.electric_field(Ck, Bk) if self.field_on else jnp.zeros(Ck.shape[1])
        if self.ex_driver is not None:
            E = E + self.ex_driver(t, None)

        dCk = -1j * self.alpha * self.kx_1d[None, :] * (self.T_H @ Ck) - self.nu_H * self.col_e[:, None] * Ck
        dBk = -1j * self.kx_1d[None, :] * (self.T_L @ Bk) - self.nu_L * self.col_l[:, None] * Bk

        dCk = dCk + _hermite_force(Ck, E, self.sqrt_2n_over_alpha, self.mask23)
        dBk = (
            dBk
            + _legendre_force(Bk, E, self.deriv, self.gamma_vec, self.xi_a, self.xi_b, self.width, self.mask23)
            + _cross_coupling(Ck, E, self.kx_1d, self.coupling_vec, self.alpha, self.width, self.mask23)
        )
        return dCk, dBk

    def _implicit_midpoint_solve(self, t: float, Ck0: Array, Bk0: Array) -> tuple:
        """One implicit-midpoint step via Jacobian-free Newton-Krylov (AD-JFNK).

        Solves y1 = y0 + dt F((y0+y1)/2). Implicit midpoint is A-stable and conserves
        quadratic invariants (energy), so it has no CFL limit and no spurious energy
        growth -- needed for the saturated/long-time regimes where the explicit and
        IMEX paths blow up. The Newton linear solves use a matrix-free GMRES whose
        Jacobian-vector products are EXACT autodiff JVPs (jax.linearize) -- the Jacobian
        is never formed. Complex coefficients are carried as real (re, im) pytree leaves
        so the Krylov inner products are the standard real ones.
        """
        import jax.scipy.sparse.linalg as jsla

        dt = self.dt
        t_mid = t + 0.5 * dt
        y0 = {"Cr": Ck0.real, "Ci": Ck0.imag, "Br": Bk0.real, "Bi": Bk0.imag}

        # Frozen-field scale for the force preconditioner (E at the step-start state).
        E0 = self.poisson.electric_field(Ck0, Bk0) if self.field_on else jnp.zeros(Ck0.shape[1])
        if self.ex_driver is not None:
            E0 = E0 + self.ex_driver(t_mid, None)
        sC = 0.5 * dt * E0

        def rhs(yr):
            Ck = yr["Cr"] + 1j * yr["Ci"]
            Bk = yr["Br"] + 1j * yr["Bi"]
            dCk, dBk = self._full_rhs_complex(t_mid, Ck, Bk)
            return {"Cr": dCk.real, "Ci": dCk.imag, "Br": dBk.real, "Bi": dBk.imag}

        def residual(y1):
            y_mid = jax.tree.map(lambda a, b: 0.5 * (a + b), y0, y1)
            f = rhs(y_mid)
            return jax.tree.map(lambda a, b, ff: a - b - dt * ff, y1, y0, f)

        # Combined preconditioner M^{-1} = M_force^{-1} . M_stream^{-1}: streaming
        # (per-k tridiagonal) handles the linear/growth phase, the force part handles
        # saturation where the Lorentz term dominates the Jacobian.
        if self.precondition:
            def M(vv):
                return self._force_precond_apply(self._stream_precond_apply(vv), sC)
        else:
            M = None
        y1 = y0
        for _ in range(self.newton_iters):
            r, jvp_fn = jax.linearize(residual, y1)  # jvp_fn(v) = J @ v, exact (AD)
            neg_r = jax.tree.map(jnp.negative, r)
            delta, _ = jsla.gmres(
                jvp_fn, neg_r, M=M, tol=self.gmres_tol, atol=0.0, restart=self.gmres_restart, maxiter=self.gmres_maxiter
            )
            y1 = jax.tree.map(jnp.add, y1, delta)

        return y1["Cr"] + 1j * y1["Ci"], y1["Br"] + 1j * y1["Bi"]

    def _implicit_E_substep(self, state: dict, E_real: Array, dt: float) -> dict:
        """Backward-Euler substep for the E.d_v f Lorentz force with frozen E.

        Per spatial point x the force is dC/dt = E(x) G_C C and dB/dt = E(x) G_B B
        (block-diagonal: Hermite force touches only C, Legendre only B). Backward
        Euler gives (I - dt E(x) G) X_new = X, an unconditionally stable per-x linear
        solve (G_C is nilpotent; G_B is lower-triangular + a rank-2 penalty). E(x) is
        diagonal in real space, so we transform k->x, solve per x, and transform back.

        NOTE: this uses a dense per-x solve, O(Nx * N^3). Fine for moderate Nx, but for
        large Nx the structured solve is far cheaper: a bidiagonal forward-substitution
        for the nilpotent Hermite block (cf. _spectrax1d.imex_E) and a Woodbury solve
        (lower-triangular + rank-2) for the Legendre block, both O(Nx * N^2).
        """
        mask = self.mask23
        maskc = mask[None, :] if mask is not None else 1.0
        Ck = state["Ck"].view(jnp.complex128)
        Bk = state["Bk"].view(jnp.complex128)
        Nh, Nl = Ck.shape[0], Bk.shape[0]

        C = jnp.fft.ifft(Ck * maskc, axis=-1, norm="forward")  # (Nh, Nx)
        B = jnp.fft.ifft(Bk * maskc, axis=-1, norm="forward")  # (Nl, Nx)

        scale = (dt * E_real).astype(jnp.complex128)  # (Nx,)
        M_C = jnp.eye(Nh, dtype=jnp.complex128)[None] - scale[:, None, None] * self.G_C[None]  # (Nx, Nh, Nh)
        M_B = jnp.eye(Nl, dtype=jnp.complex128)[None] - scale[:, None, None] * self.G_B[None]  # (Nx, Nl, Nl)

        C_new = jnp.linalg.solve(M_C, C.T[..., None])[..., 0].T  # (Nh, Nx)
        B_new = jnp.linalg.solve(M_B, B.T[..., None])[..., 0].T  # (Nl, Nx)

        Ck_new = jnp.fft.fft(C_new, axis=-1, norm="forward") * maskc
        Bk_new = jnp.fft.fft(B_new, axis=-1, norm="forward") * maskc
        out = dict(state)
        out["Ck"] = Ck_new.view(jnp.float64)
        out["Bk"] = Bk_new.view(jnp.float64)
        return out

    def __call__(self, t: float, y: dict, args: dict) -> dict:
        if self.implicit:
            Ck0 = y["Ck"].view(jnp.complex128)
            Bk0 = y["Bk"].view(jnp.complex128)
            Ck, Bk = self._implicit_midpoint_solve(t, Ck0, Bk0)
            if self.field_on:
                e = self.poisson.electric_field(Ck, Bk)
                phi = self.poisson.potential(Ck, Bk)
            else:
                e = jnp.zeros(Ck.shape[1])
                phi = jnp.zeros(Ck.shape[1])
            de = self.ex_driver(t, args) if self.ex_driver is not None else jnp.zeros(Ck.shape[1])
            return {"Ck": Ck.view(jnp.float64), "Bk": Bk.view(jnp.float64), "e": e, "phi": phi, "de": de}

        y_new = self._lawson_rk4(t, y, args)

        if self.imex:
            # Frozen E from the post-explicit state (self-consistent + external driver
            # at the end of the step), then one implicit Lorentz substep.
            Ckp = y_new["Ck"].view(jnp.complex128)
            Bkp = y_new["Bk"].view(jnp.complex128)
            E_frozen = self.poisson.electric_field(Ckp, Bkp) if self.field_on else jnp.zeros(Ckp.shape[1])
            if self.ex_driver is not None:
                E_frozen = E_frozen + self.ex_driver(t + self.dt, args)
            y_new = self._implicit_E_substep(y_new, E_frozen, self.dt)

        Ck = y_new["Ck"].view(jnp.complex128)
        Bk = y_new["Bk"].view(jnp.complex128)
        if self.field_on:
            e = self.poisson.electric_field(Ck, Bk)
            phi = self.poisson.potential(Ck, Bk)
        else:
            e = jnp.zeros(Ck.shape[1])  # phi = 0 : pure advection
            phi = jnp.zeros(Ck.shape[1])
        de = self.ex_driver(t, args) if self.ex_driver is not None else jnp.zeros(Ck.shape[1])
        return {"Ck": y_new["Ck"], "Bk": y_new["Bk"], "e": e, "phi": phi, "de": de}
