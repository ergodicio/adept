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

    psi_n = (pi 2^n n!)^{-1/2} H_n(z) exp(-z^2),  z = (v - u)/alpha,
    built from the physicists' Hermite recurrence H_{n+1} = 2 z H_n - 2 n H_{n-1}.
    """
    z = (v - u) / alpha
    H = np.zeros((Nh_plus_1, v.size), dtype=np.float64)
    H[0] = 1.0
    if Nh_plus_1 > 1:
        H[1] = 2.0 * z
    for k in range(1, Nh_plus_1 - 1):
        H[k + 1] = 2.0 * z * H[k] - 2.0 * k * H[k - 1]
    norm = 1.0 / np.sqrt(np.pi * (2.0 ** np.arange(Nh_plus_1)) * _factorials(Nh_plus_1))
    return norm[:, None] * H * np.exp(-(z**2))[None, :]


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


def _factorials(n: int) -> np.ndarray:
    out = np.ones(n, dtype=np.float64)
    for k in range(1, n):
        out[k] = out[k - 1] * k
    return out


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

    def _nonlinear_rhs(self, t: float, state: dict, args: dict) -> dict:
        Ck = state["Ck"].view(jnp.complex128)
        Bk = state["Bk"].view(jnp.complex128)

        # field_on=False -> pure advection (phi=0); the linear Hermite->Legendre
        # closure flux (d_x C_{Nh-1}) still acts, only E-dependent terms vanish.
        E = self.poisson.electric_field(Ck, Bk) if self.field_on else jnp.zeros(Ck.shape[1])  # (Nx,)

        dCk = _hermite_force(Ck, E, self.sqrt_2n_over_alpha, self.mask23)
        dBk = _legendre_force(
            Bk, E, self.deriv, self.gamma_vec, self.xi_a, self.xi_b, self.width, self.mask23
        ) + _cross_coupling(Ck, E, self.kx_1d, self.coupling_vec, self.alpha, self.width, self.mask23)

        out = dict(state)
        out["Ck"] = dCk.view(jnp.float64)
        out["Bk"] = dBk.view(jnp.float64)
        for k in ("e", "phi"):
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

    def __call__(self, t: float, y: dict, args: dict) -> dict:
        y_new = self._lawson_rk4(t, y, args)
        Ck = y_new["Ck"].view(jnp.complex128)
        Bk = y_new["Bk"].view(jnp.complex128)
        if self.field_on:
            e = self.poisson.electric_field(Ck, Bk)
            phi = self.poisson.potential(Ck, Bk)
        else:
            e = jnp.zeros(Ck.shape[1])  # phi = 0 : pure advection
            phi = jnp.zeros(Ck.shape[1])
        return {"Ck": y_new["Ck"], "Bk": y_new["Bk"], "e": e, "phi": phi}
