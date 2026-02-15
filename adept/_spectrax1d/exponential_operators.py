"""
Exponential operators for the Lawson-RK4 integrator.

This module provides precomputed and analytically-evaluated exponential operators
for the linear terms in the Vlasov-Maxwell Hermite-Fourier system:
  - Free-streaming: prediagonalized tridiagonal per direction per species
  - Maxwell curls: analytical Rodrigues rotation per k-point
  - Collision + diffusion: diagonal exponentials

These are used by the LawsonRK4Solver to handle linear stiffness exactly,
removing CFL constraints from light waves and free streaming.
"""

import jax
from jax import Array
from jax import numpy as jnp


class FreeStreamingExponential:
    """
    Prediagonalized free-streaming exponential for one species.

    The free-streaming operator for direction d is:
        L_d · Ck = -(i·k_d/L_d)·alpha_d · T_d · Ck
    where T_d is an N_d x N_d real symmetric tridiagonal matrix:
        T_d[n, n+1] = T_d[n+1, n] = sqrt((n+1)/2)
        T_d[n, n]   = u_d / alpha_d

    Prediagonalization: T_d = V_d · Λ_d · V_d^T  (V orthogonal)
    Then: exp(L_d·s) = V_d · diag(exp(prefactor_d · k_d · s · λ_i)) · V_d^T

    Memory: O(N_d^2) per direction instead of O(N_k x N_d^2).
    """

    def __init__(
        self,
        sqrt_n_plus: Array,
        sqrt_n_minus: Array,
        sqrt_m_plus: Array,
        sqrt_m_minus: Array,
        sqrt_p_plus: Array,
        sqrt_p_minus: Array,
        Nn: int,
        Nm: int,
        Np: int,
        alpha: Array,
        u: Array,
        Lx: float,
        Ly: float,
        Lz: float,
        kx_1d: Array,
        ky_1d: Array,
        kz_1d: Array,
    ):
        """
        Args:
            sqrt_n_plus: (Nn,) = sqrt(n+1), ladder operator for n-direction
            sqrt_n_minus: (Nn,) = sqrt(n), ladder operator for n-direction
            sqrt_m_plus: (Nm,) = sqrt(m+1)
            sqrt_m_minus: (Nm,) = sqrt(m)
            sqrt_p_plus: (Np,) = sqrt(p+1)
            sqrt_p_minus: (Np,) = sqrt(p)
            Nn, Nm, Np: Number of Hermite modes per velocity direction
            alpha: (3,) = (alpha_x, alpha_y, alpha_z) thermal velocities
            u: (3,) = (u_x, u_y, u_z) drift velocities
            Lx, Ly, Lz: Domain lengths
            kx_1d, ky_1d, kz_1d: 1D wavenumber arrays (2π·fftfreq·N)
        """
        # Build and diagonalize tridiagonal matrices for each direction
        self.V_x, self.eigenvalues_x = self._build_and_diagonalize(sqrt_n_plus, Nn, float(u[0] / alpha[0]))
        self.V_y, self.eigenvalues_y = self._build_and_diagonalize(sqrt_m_plus, Nm, float(u[1] / alpha[1]))
        self.V_z, self.eigenvalues_z = self._build_and_diagonalize(sqrt_p_plus, Np, float(u[2] / alpha[2]))

        # Prefactors: L_d = prefactor_d * k_d * T_d
        self.prefactor_x = -1j * float(alpha[0]) / Lx
        self.prefactor_y = -1j * float(alpha[1]) / Ly
        self.prefactor_z = -1j * float(alpha[2]) / Lz

        # 1D wavenumber arrays
        self.kx_1d = kx_1d
        self.ky_1d = ky_1d
        self.kz_1d = kz_1d

    @staticmethod
    def _build_and_diagonalize(sqrt_plus: Array, N: int, drift_over_alpha: float):
        """
        Build symmetric tridiagonal matrix and diagonalize.

        T[n, n+1] = T[n+1, n] = sqrt_plus[n] / sqrt(2) = sqrt((n+1)/2)
        T[n, n]   = drift_over_alpha

        Returns:
            V: (N, N) orthogonal eigenvector matrix
            eigenvalues: (N,) eigenvalues
        """
        if N == 1:
            V = jnp.ones((1, 1))
            eigenvalues = jnp.array([drift_over_alpha])
            return V, eigenvalues

        # Off-diagonal: sqrt((n+1)/2) for n = 0, ..., N-2
        off_diag = sqrt_plus[:-1] / jnp.sqrt(2.0)

        # Build symmetric tridiagonal
        T = jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
        T = T + drift_over_alpha * jnp.eye(N)

        # Diagonalize (real symmetric → orthogonal eigenvectors)
        eigenvalues, V = jnp.linalg.eigh(T)
        return V, eigenvalues

    def apply(self, Ck: Array, s: float) -> Array:
        """
        Apply exp(L_stream · s) to Ck.

        Applies x, y, z direction exponentials sequentially (they commute).

        Args:
            Ck: Complex array, shape (Np, Nm, Nn, Ny, Nx, Nz)
            s: Time offset (dt or dt/2)

        Returns:
            exp(L_stream · s) · Ck, same shape
        """
        Ck = self._apply_direction_x(Ck, s)
        Ck = self._apply_direction_y(Ck, s)
        Ck = self._apply_direction_z(Ck, s)
        return Ck

    def _apply_direction_x(self, Ck: Array, s: float) -> Array:
        """Apply x-direction free-streaming exponential (acts on n-axis, varies with kx)."""
        # exp_factors[i, kx_idx] = exp(prefactor_x * kx * s * eigenvalue_i)
        exp_factors = jnp.exp(self.prefactor_x * self.kx_1d[None, :] * s * self.eigenvalues_x[:, None])  # (Nn, Nx)

        # Project into eigenbasis: V^T @ Ck along n-axis (axis 2)
        C_eig = jnp.einsum("in,pmnyxz->pmiyxz", self.V_x.T, Ck)

        # Scale by per-kx exponentials: broadcast (Nn, Nx) over (Np, Nm, _, Ny, _, Nz)
        C_scaled = C_eig * exp_factors[None, None, :, None, :, None]

        # Project back: V @ C_scaled
        return jnp.einsum("ni,pmiyxz->pmnyxz", self.V_x, C_scaled)

    def _apply_direction_y(self, Ck: Array, s: float) -> Array:
        """Apply y-direction free-streaming exponential (acts on m-axis, varies with ky)."""
        exp_factors = jnp.exp(self.prefactor_y * self.ky_1d[None, :] * s * self.eigenvalues_y[:, None])  # (Nm, Ny)

        # Project along m-axis (axis 1)
        C_eig = jnp.einsum("jm,pmnyxz->pjnyxz", self.V_y.T, Ck)

        # Scale: broadcast (Nm, Ny) over (Np, _, Nn, _, Nx, Nz)
        C_scaled = C_eig * exp_factors[None, :, None, :, None, None]

        return jnp.einsum("mj,pjnyxz->pmnyxz", self.V_y, C_scaled)

    def _apply_direction_z(self, Ck: Array, s: float) -> Array:
        """Apply z-direction free-streaming exponential (acts on p-axis, varies with kz)."""
        exp_factors = jnp.exp(self.prefactor_z * self.kz_1d[None, :] * s * self.eigenvalues_z[:, None])  # (Np, Nz)

        # Project along p-axis (axis 0)
        C_eig = jnp.einsum("kp,pmnyxz->kmnyxz", self.V_z.T, Ck)

        # Scale: broadcast (Np, Nz) over (_, Nm, Nn, Ny, Nx, _)
        C_scaled = C_eig * exp_factors[:, None, None, None, None, :]

        return jnp.einsum("pk,kmnyxz->pmnyxz", self.V_z, C_scaled)


class MaxwellExponential:
    """
    Analytical Maxwell curl exponential using Rodrigues-like formula.

    Maxwell curl operator at each k-point with ξ = nabla = (kx/Lx, ky/Ly, kz/Lz):
        dE/dt = i(ξ x B)
        dB/dt = -i(ξ x E)

    This is a 6x6 matrix M. Using ω = |ξ|:
        exp(M·s) = I + (sin(ω·s)/ω)·M + ((1-cos(ω·s))/ω²)·M²

    where M²·[E,B] = [ξx(ξxE), ξx(ξxB)] = [ξ(ξ·E)-ω²E, ξ(ξ·B)-ω²B].
    """

    def __init__(self, nabla: Array):
        """
        Args:
            nabla: (3, Ny, Nx, Nz) = [kx/Lx, ky/Ly, kz/Lz]
        """
        self.nabla = nabla
        self.omega = jnp.sqrt(jnp.sum(nabla**2, axis=0))  # (Ny, Nx, Nz)

    def apply(self, Fk: Array, s: float) -> Array:
        """
        Apply exp(M_maxwell · s) to Fk analytically.

        Args:
            Fk: Complex array, shape (6, Ny, Nx, Nz) = [Ex, Ey, Ez, Bx, By, Bz]
            s: Time offset

        Returns:
            Rotated field array, same shape
        """
        xi = self.nabla  # (3, Ny, Nx, Nz)

        # Use safe_omega to avoid 0/0; at ω=0, M=0 so exp(M·s)=I regardless of coefficients
        safe_omega = jnp.where(self.omega > 0, self.omega, 1.0)
        omega_s = safe_omega * s

        sin_os = jnp.sin(omega_s)
        cos_os = jnp.cos(omega_s)

        c1 = sin_os / safe_omega  # sin(ωs)/ω
        c2 = (1.0 - cos_os) / safe_omega**2  # (1-cos(ωs))/ω²

        E = Fk[:3]  # (3, Ny, Nx, Nz)
        B = Fk[3:]  # (3, Ny, Nx, Nz)

        # M·F: dE = i(ξxB), dB = -i(ξxE)
        M_E = 1j * jnp.cross(xi, B, axis=0)
        M_B = -1j * jnp.cross(xi, E, axis=0)

        # M²·F: ξx(ξxE) = ξ(ξ·E) - ω²E  (BAC-CAB rule)
        xi_dot_E = jnp.sum(xi * E, axis=0, keepdims=True)  # (1, Ny, Nx, Nz)
        xi_dot_B = jnp.sum(xi * B, axis=0, keepdims=True)

        omega_sq = self.omega[None, ...] ** 2  # (1, Ny, Nx, Nz)
        M2_E = xi * xi_dot_E - E * omega_sq
        M2_B = xi * xi_dot_B - B * omega_sq

        # exp(M·s)·F = F + c1·M·F + c2·M²·F
        c1_b = c1[None, ...]  # broadcast to (1, Ny, Nx, Nz)
        c2_b = c2[None, ...]

        E_new = E + c1_b * M_E + c2_b * M2_E
        B_new = B + c1_b * M_B + c2_b * M2_B

        return jnp.concatenate([E_new, B_new], axis=0)


class DiagonalExponential:
    """Diagonal exponential for hypercollision + hyper-diffusion."""

    def __init__(self, nu: float, col: Array, D: float, k2_grid: Array):
        """
        Args:
            nu: Collision frequency
            col: Collision matrix, shape (Np, Nm, Nn)
            D: Hyper-diffusion coefficient
            k2_grid: Squared wavenumber grid, shape (Ny, Nx, Nz)
        """
        self.nu = nu
        self.col = col
        self.D = D
        self.k2_grid = k2_grid

    def apply(self, Ck: Array, s: float) -> Array:
        """
        Apply exp((-nu·col - D·k²)·s) to Ck.

        Args:
            Ck: Complex array, shape (Np, Nm, Nn, Ny, Nx, Nz)
            s: Time offset
        """
        # -nu·col[p,m,n]·s broadcast over spatial dims
        col_factor = -self.nu * self.col[:, :, :, None, None, None] * s
        # -D·k²·s broadcast over Hermite dims
        diff_factor = -self.D * self.k2_grid[None, None, None, :, :, :] * s

        return Ck * jnp.exp(col_factor + diff_factor)


class PlasmaOscillationExponential:
    """
    Exact exponential for the ωpe plasma oscillation coupling C₁ ↔ E.

    The mean-density part of the Vlasov-Maxwell coupling forms a 2×2 linear
    system at each k-point for each velocity direction d:

        dC_{1d}/dt = a_d · E_d(k)
        dE_d/dt    = -b_d · C_{1d}(k)

    where:
        a_d = q_e · Ωc_e · √2/α_d · ⟨C₀₀₀⟩_e    (E → C coupling via Lorentz force)
        b_d = q_e · α₀α₁α₂ · α_d / (√2 · Ωc_e)   (C → E coupling via plasma current)

    Exact solution is a 2×2 rotation at ω = √(a·b):
        C_new = cos(ωs)·C + (a/ω)·sin(ωs)·E
        E_new = cos(ωs)·E - (b/ω)·sin(ωs)·C
    """

    def __init__(
        self,
        q_e: float,
        Omega_c_e: float,
        alpha_e: Array,
        C000_mean_e: float,
        Nn_e: int,
        Nm_e: int,
        Np_e: int,
    ):
        """
        Args:
            q_e: Electron charge (scalar, typically -1)
            Omega_c_e: Electron cyclotron frequency (scalar)
            alpha_e: (3,) electron thermal velocities
            C000_mean_e: Mean electron density in Hermite basis (real scalar = ⟨n_e⟩/α³)
            Nn_e, Nm_e, Np_e: Electron Hermite mode counts
        """
        self.Nn_e = Nn_e
        self.Nm_e = Nm_e
        self.Np_e = Np_e

        alpha_prod = alpha_e[0] * alpha_e[1] * alpha_e[2]

        # Coupling coefficients for each direction d ∈ {x, y, z}
        a = jnp.array([
            q_e * Omega_c_e * jnp.sqrt(2.0) / alpha_e[d] * C000_mean_e
            for d in range(3)
        ])
        b = jnp.array([
            q_e * alpha_prod * alpha_e[d] / (jnp.sqrt(2.0) * Omega_c_e)
            for d in range(3)
        ])

        self.a = a  # (3,) E → C coupling
        self.b = b  # (3,) C → E coupling
        self.omega = jnp.sqrt(jnp.abs(a * b))  # (3,) oscillation frequency per direction

    def _rotate(self, C: Array, E: Array, a: float, b: float, omega: float, s: float):
        """Apply 2×2 rotation for one direction."""
        safe_omega = jnp.where(omega > 0, omega, 1.0)
        cos_os = jnp.cos(safe_omega * s)
        sin_os = jnp.sin(safe_omega * s)
        a_over_w = jnp.where(omega > 0, a / safe_omega, 0.0)
        b_over_w = jnp.where(omega > 0, b / safe_omega, 0.0)
        C_new = cos_os * C + a_over_w * sin_os * E
        E_new = cos_os * E - b_over_w * sin_os * C
        return C_new, E_new

    def apply(self, Ck_e: Array, Fk: Array, s: float):
        """
        Apply the plasma oscillation exponential rotation.

        Args:
            Ck_e: Electron Hermite coefficients, shape (Np, Nm, Nn, Ny, Nx, Nz)
            Fk: EM fields, shape (6, Ny, Nx, Nz)
            s: Time offset

        Returns:
            (Ck_e_new, Fk_new) with rotation applied
        """
        # Direction x: C₁₀₀ [0,0,1] ↔ E_x [0]
        if self.Nn_e > 1:
            C_new, E_new = self._rotate(
                Ck_e[0, 0, 1], Fk[0], self.a[0], self.b[0], self.omega[0], s
            )
            Ck_e = Ck_e.at[0, 0, 1].set(C_new)
            Fk = Fk.at[0].set(E_new)

        # Direction y: C₀₁₀ [0,1,0] ↔ E_y [1]
        if self.Nm_e > 1:
            C_new, E_new = self._rotate(
                Ck_e[0, 1, 0], Fk[1], self.a[1], self.b[1], self.omega[1], s
            )
            Ck_e = Ck_e.at[0, 1, 0].set(C_new)
            Fk = Fk.at[1].set(E_new)

        # Direction z: C₀₀₁ [1,0,0] ↔ E_z [2]
        if self.Np_e > 1:
            C_new, E_new = self._rotate(
                Ck_e[1, 0, 0], Fk[2], self.a[2], self.b[2], self.omega[2], s
            )
            Ck_e = Ck_e.at[1, 0, 0].set(C_new)
            Fk = Fk.at[2].set(E_new)

        return Ck_e, Fk


class CombinedLinearExponential:
    """
    Combines all linear exponentials to apply exp(L·s) to the full state dict.

    Handles the float64-view ↔ complex128 packing required by diffrax.
    """

    def __init__(
        self,
        free_stream_e: FreeStreamingExponential,
        free_stream_i: FreeStreamingExponential,
        maxwell: MaxwellExponential,
        diag_e: DiagonalExponential,
        diag_i: DiagonalExponential,
        plasma_osc: PlasmaOscillationExponential | None = None,
    ):
        self.free_stream_e = free_stream_e
        self.free_stream_i = free_stream_i
        self.maxwell = maxwell
        self.diag_e = diag_e
        self.diag_i = diag_i
        self.plasma_osc = plasma_osc

    def apply(self, y: dict, s: float) -> dict:
        """
        Apply exp(L·s) to full state dictionary.

        Args:
            y: State dict with float64 views: {"Ck_electrons", "Ck_ions", "Fk"}
            s: Time offset (dt or dt/2)

        Returns:
            New state dict with exp(L·s) applied, as float64 views
        """
        # Unpack to complex
        Ck_e = y["Ck_electrons"].view(jnp.complex128)
        Ck_i = y["Ck_ions"].view(jnp.complex128)
        Fk = y["Fk"].view(jnp.complex128)

        # Apply free-streaming + diagonal exponentials to each species
        Ck_e = self.free_stream_e.apply(Ck_e, s)
        Ck_e = self.diag_e.apply(Ck_e, s)

        Ck_i = self.free_stream_i.apply(Ck_i, s)
        Ck_i = self.diag_i.apply(Ck_i, s)

        # Apply Maxwell rotation to fields
        Fk = self.maxwell.apply(Fk, s)

        # Apply plasma oscillation (C₁ ↔ E coupling at ωpe)
        if self.plasma_osc is not None:
            Ck_e, Fk = self.plasma_osc.apply(Ck_e, Fk, s)

        # Pack back to float64 views
        return {
            "Ck_electrons": Ck_e.view(jnp.float64),
            "Ck_ions": Ck_i.view(jnp.float64),
            "Fk": Fk.view(jnp.float64),
        }


def build_combined_exponential(
    grid_quantities_electrons: dict,
    grid_quantities_ions: dict,
    alpha_s: Array,
    u_s: Array,
    nu: float,
    D: float,
    Nn_e: int,
    Nm_e: int,
    Np_e: int,
    Nn_i: int,
    Nm_i: int,
    Np_i: int,
    Nx: int,
    Ny: int,
    Nz: int,
    qs: Array | None = None,
    Omega_cs: Array | None = None,
    C000_mean_e: float | None = None,
) -> CombinedLinearExponential:
    """
    Factory function to build the CombinedLinearExponential from grid quantities.

    Args:
        grid_quantities_electrons: Dict with kx_grid, ky_grid, kz_grid, k2_grid,
            nabla, col, sqrt_n/m/p_plus/minus, Lx, Ly, Lz for electrons
        grid_quantities_ions: Same for ions
        alpha_s: (6,) thermal velocities [alpha_xe, alpha_ye, alpha_ze, alpha_xi, ...]
        u_s: (6,) drift velocities
        nu: Collision frequency
        D: Hyper-diffusion coefficient
        Nn_e, Nm_e, Np_e: Electron Hermite mode counts
        Nn_i, Nm_i, Np_i: Ion Hermite mode counts
        Nx, Ny, Nz: Grid dimensions
        qs: (2,) species charges [q_e, q_i] — needed for plasma oscillation
        Omega_cs: (2,) cyclotron frequencies — needed for plasma oscillation
        C000_mean_e: Mean electron C000 (real scalar) — needed for plasma oscillation

    Returns:
        CombinedLinearExponential instance
    """
    gq_e = grid_quantities_electrons
    gq_i = grid_quantities_ions

    # Extract 1D wavenumber arrays from the 3D meshgrid
    # kx_grid shape (Ny, Nx, Nz), kx varies along axis 1
    kx_1d = jnp.fft.fftfreq(Nx) * Nx * 2 * jnp.pi
    ky_1d = jnp.fft.fftfreq(Ny) * Ny * 2 * jnp.pi
    kz_1d = jnp.fft.fftfreq(Nz) * Nz * 2 * jnp.pi

    # Build free-streaming exponentials
    free_stream_e = FreeStreamingExponential(
        sqrt_n_plus=gq_e["sqrt_n_plus"],
        sqrt_n_minus=gq_e["sqrt_n_minus"],
        sqrt_m_plus=gq_e["sqrt_m_plus"],
        sqrt_m_minus=gq_e["sqrt_m_minus"],
        sqrt_p_plus=gq_e["sqrt_p_plus"],
        sqrt_p_minus=gq_e["sqrt_p_minus"],
        Nn=Nn_e,
        Nm=Nm_e,
        Np=Np_e,
        alpha=alpha_s[:3],
        u=u_s[:3],
        Lx=gq_e["Lx"],
        Ly=gq_e["Ly"],
        Lz=gq_e["Lz"],
        kx_1d=kx_1d,
        ky_1d=ky_1d,
        kz_1d=kz_1d,
    )

    free_stream_i = FreeStreamingExponential(
        sqrt_n_plus=gq_i["sqrt_n_plus"],
        sqrt_n_minus=gq_i["sqrt_n_minus"],
        sqrt_m_plus=gq_i["sqrt_m_plus"],
        sqrt_m_minus=gq_i["sqrt_m_minus"],
        sqrt_p_plus=gq_i["sqrt_p_plus"],
        sqrt_p_minus=gq_i["sqrt_p_minus"],
        Nn=Nn_i,
        Nm=Nm_i,
        Np=Np_i,
        alpha=alpha_s[3:],
        u=u_s[3:],
        Lx=gq_i["Lx"],
        Ly=gq_i["Ly"],
        Lz=gq_i["Lz"],
        kx_1d=kx_1d,
        ky_1d=ky_1d,
        kz_1d=kz_1d,
    )

    # Build Maxwell exponential
    maxwell = MaxwellExponential(nabla=gq_e["nabla"])

    # Build diagonal exponentials (collision + diffusion)
    diag_e = DiagonalExponential(nu=nu, col=gq_e["col"], D=D, k2_grid=gq_e["k2_grid"])
    diag_i = DiagonalExponential(nu=nu, col=gq_i["col"], D=D, k2_grid=gq_i["k2_grid"])

    # Build plasma oscillation exponential if parameters provided
    plasma_osc = None
    if qs is not None and Omega_cs is not None and C000_mean_e is not None:
        plasma_osc = PlasmaOscillationExponential(
            q_e=float(qs[0]),
            Omega_c_e=float(Omega_cs[0]),
            alpha_e=alpha_s[:3],
            C000_mean_e=float(C000_mean_e),
            Nn_e=Nn_e,
            Nm_e=Nm_e,
            Np_e=Np_e,
        )

    return CombinedLinearExponential(
        free_stream_e=free_stream_e,
        free_stream_i=free_stream_i,
        maxwell=maxwell,
        diag_e=diag_e,
        diag_i=diag_i,
        plasma_osc=plasma_osc,
    )


def compute_houli_hermite_filter(
    Nn: int, Nm: int, Np: int, cutoff_fraction: float, strength: float, order: int
) -> Array:
    """Compute Hou-Li exponential filter in Hermite space.

    Returns array of shape (Np, Nm, Nn) with values in (0, 1].
    Modes below cutoff_fraction * h_max are unaffected (value = 1).
    Modes above are damped as exp(-strength * (h/h_cutoff)^order).

    Args:
        Nn, Nm, Np: Number of Hermite modes per velocity direction
        cutoff_fraction: Fraction of max mode number where filtering starts
        strength: Filter strength parameter
        order: Filter order (higher = sharper cutoff)
    """
    n = jnp.arange(Nn)[None, None, :]
    m = jnp.arange(Nm)[None, :, None]
    p = jnp.arange(Np)[:, None, None]

    h_max = jnp.sqrt((Nn - 1) ** 2 + (Nm - 1) ** 2 + (Np - 1) ** 2)
    h_cutoff = cutoff_fraction * h_max
    h = jnp.sqrt(n**2 + m**2 + p**2)

    filter_arg = jnp.where(h > h_cutoff, strength * ((h / h_cutoff) ** order), 0.0)
    return jnp.exp(-filter_arg)
