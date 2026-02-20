"""
Hermite-Fourier ODE system for Vlasov-Maxwell equations.

This module implements the RHS of the Hermite-Fourier moment equations,
computing dCk/dt for the distribution function in spectral space.

This is a refactored class-based version of the external spectrax.Hermite_Fourier_system()
function, with grid quantities stored as class attributes to simplify the call signature.
"""

import equinox as eqx
from jax import Array
from jax import numpy as jnp


class HermiteFourierODE(eqx.Module):
    """
    Compute time derivative of distribution function in Hermite-Fourier basis.

    This class wraps the Hermite-Fourier system ODE, storing grid quantities
    as class attributes to simplify the call signature.

    Physical terms computed:
    1. Spatial advection (streaming with drift velocities)
    2. Lorentz force (E-field and B-field coupling via auxiliary fields)
    3. Collision operator (hypercollisional damping proportional to mode index)
    4. Diffusion operator (high-k artificial viscosity for stabilization)

    Args:
        Nn, Nm, Np: Number of Hermite modes in x, y, z velocity directions
        kx_grid, ky_grid, kz_grid: Wavenumber grids (standard FFT ordering, shape (Ny, Nx, Nz))
        k2_grid: Squared wavenumber magnitude
        Lx, Ly, Lz: Domain lengths
        col: Collision matrix, shape (Np, Nm, Nn)
        sqrt_n_plus, sqrt_n_minus: Hermite ladder operators for n (x-velocity)
        sqrt_m_plus, sqrt_m_minus: Hermite ladder operators for m (y-velocity)
        sqrt_p_plus, sqrt_p_minus: Hermite ladder operators for p (z-velocity)
        mask23: 2/3 dealiasing mask in Fourier space
    """

    # Grid dimensions
    Nn: int
    Nm: int
    Np: int
    Nx: int

    # Domain lengths
    Lx: float
    Ly: float
    Lz: float

    # Collision matrix
    col: Array

    # Hermite ladder operators (1D arrays)
    sqrt_n_plus: Array
    sqrt_n_minus: Array
    sqrt_m_plus: Array
    sqrt_m_minus: Array
    sqrt_p_plus: Array
    sqrt_p_minus: Array

    # k-space grids
    kx_grid: Array
    ky_grid: Array
    kz_grid: Array
    k2_grid: Array
    mask23: Array

    def __init__(
        self,
        Nn: int,
        Nm: int,
        Np: int,
        Nx: int,
        kx_grid: Array,
        ky_grid: Array,
        kz_grid: Array,
        k2_grid: Array,
        Lx: float,
        Ly: float,
        Lz: float,
        col: Array,
        sqrt_n_plus: Array,
        sqrt_n_minus: Array,
        sqrt_m_plus: Array,
        sqrt_m_minus: Array,
        sqrt_p_plus: Array,
        sqrt_p_minus: Array,
        mask23: Array,
    ):
        """Initialize with grid quantities."""
        # Store grid dimensions
        self.Nn = Nn
        self.Nm = Nm
        self.Np = Np
        self.Nx = Nx

        # Store domain lengths
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # Store collision matrix
        self.col = col

        # Store Hermite ladder operators
        # IMPORTANT: Reshape to 1D if spectrax returns broadcasted shapes
        # These should be 1D arrays of length Nn, Nm, Np respectively
        # Use reshape instead of squeeze to ensure we always get 1D arrays
        self.sqrt_n_plus = jnp.reshape(sqrt_n_plus, (Nn,))
        self.sqrt_n_minus = jnp.reshape(sqrt_n_minus, (Nn,))
        self.sqrt_m_plus = jnp.reshape(sqrt_m_plus, (Nm,))
        self.sqrt_m_minus = jnp.reshape(sqrt_m_minus, (Nm,))
        self.sqrt_p_plus = jnp.reshape(sqrt_p_plus, (Np,))
        self.sqrt_p_minus = jnp.reshape(sqrt_p_minus, (Np,))

        # Store k-space grids (no padding - sharding handled at higher level)
        self.kx_grid = kx_grid
        self.ky_grid = ky_grid
        self.kz_grid = kz_grid
        self.k2_grid = k2_grid
        self.mask23 = mask23

    def _pad_hermite_axes(self, Ck: Array) -> Array:
        """
        Zero-pad Hermite axes (p, m, n) by 1 on each side.

        Args:
            Ck: Array with shape (Np, Nm, Nn, Ny, Nx, Nz)

        Returns:
            Padded array with shape (Np+2, Nm+2, Nn+2, Ny, Nx, Nz)
        """
        # Pad +1 on both sides for axes 0,1,2 (p,m,n) only
        return jnp.pad(Ck, ((1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0)))

    def shift_multi(
        self,
        Ck: Array,
        dn: int = 0,
        dm: int = 0,
        dp: int = 0,
        closure_n=None,
        closure_m=None,
        closure_p=None,
    ) -> Array:
        """
        Zero-padded shift along Hermite axes (n, m, p) with optional neural network closure.

        If a closure function is provided for a direction, it will be called to predict
        the value of out-of-range modes instead of using zero padding.

        dn=+1 means 'use source at n-1' (upshift in n)
        dn=-1 means 'use source at n+1' (downshift in n)
        dn=0 means identity (no shift)
        Same for dm, dp.

        Args:
            Ck: Array with shape (Np, Nm, Nn, Ny, Nx, Nz) - Single species
            dn, dm, dp: Shift amounts for n, m, p indices (values in {-1, 0, +1})
            closure_n: Optional closure for n-direction (x-velocity)
            closure_m: Optional closure for m-direction (y-velocity)
            closure_p: Optional closure for p-direction (z-velocity)

        Returns:
            Shifted array with same shape, with closure-predicted or zero-padded boundaries
        """
        Np, Nm, Nn, Ny, Nx, Nz = Ck.shape

        # Start with zero-padding (default behavior)
        P = self._pad_hermite_axes(Ck)

        # Apply neural network closures for out-of-range modes

        # N-direction closure (x-velocity)
        if dn == +1 and closure_n is not None:
            # Accessing n+1 from highest mode (n=Nn-1) -> predict mode Nn
            # closure_n should return shape (Np, Nm, 1, Ny, Nx, Nz)
            C_Nn = closure_n(Ck)
            # Insert predicted mode at the right boundary of padded array
            # P has shape (Np+2, Nm+2, Nn+2, Ny, Nx, Nz)
            # The boundary is at index Nn+1 (0-indexed: 0=pad, 1...Nn=data, Nn+1=pad)
            P = P.at[1:-1, 1:-1, -1, :, :, :].set(C_Nn.squeeze(axis=2))

        # M-direction closure (y-velocity)
        if dm == +1 and closure_m is not None:
            C_Nm = closure_m(Ck)
            P = P.at[1:-1, -1, 1:-1, :, :, :].set(C_Nm.squeeze(axis=1))

        # P-direction closure (z-velocity)
        if dp == +1 and closure_p is not None:
            C_Np = closure_p(Ck)
            P = P.at[-1, 1:-1, 1:-1, :, :, :].set(C_Np.squeeze(axis=0))

        # Extract shifted region (same as before)
        n0 = 1 + dn  # dn=+1 -> 0 ; dn=0 -> 1 ; dn=-1 -> 2
        m0 = 1 + dm
        p0 = 1 + dp

        return P[p0 : p0 + Np, m0 : m0 + Nm, n0 : n0 + Nn, :, :, :]

    def _compute_rhs(
        self,
        Ck: Array,
        C: Array,
        F: Array,
        nu: float,
        D: float,
        alpha: Array,
        u: Array,
        q: float,
        Omega_c: float,
        closure_n=None,
        closure_m=None,
        closure_p=None,
    ) -> Array:
        """
        Compute dCk/dt for the Hermite-Fourier system (core computation).

        This is the inner computation that can be called either directly or via shard_map.

        Args:
            Ck: Spectral Hermite coefficients, shape (Np, Nm, Nn, Ny, Nx, Nz) - Single species
            C: Real-space Hermite coefficients, shape (Np, Nm, Nn, Ny, Nx, Nz) - Single species
            F: Real-space EM fields, shape (6, Ny, Nx, Nz) = (Ex, Ey, Ez, Bx, By, Bz)
            nu: Collision frequency
            D: Hyper-diffusion coefficient
            alpha: Thermal velocities for single species, shape (3,) = (alpha_x, alpha_y, alpha_z)
            u: Drift velocities for single species, shape (3,) = (u_x, u_y, u_z)
            q: Species charge (scalar)
            Omega_c: Cyclotron frequency (scalar)
            closure_n: Optional closure for n-direction (x-velocity)
            closure_m: Optional closure for m-direction (y-velocity)
            closure_p: Optional closure for p-direction (z-velocity)

        Returns:
            dCk/dt with shape (Np, Nm, Nn, Ny, Nx, Nz)
        """
        # Extract shape information (Ck is already in 6D format)
        Np, Nm, Nn, Ny, Nx, Nz = Ck.shape

        # Broadcast fields to match Hermite dimensions
        F = F[:, None, None, None, :, :, :]  # (6, 1, 1, 1, Ny, Nx, Nz)

        # Extract thermal velocity and drift velocity components (scalars broadcast naturally)
        a0, a1, a2 = alpha[0], alpha[1], alpha[2]
        u0, u1, u2 = u[0], u[1], u[2]

        # Reshape Hermite ladder operators for broadcasting with 6D arrays (Np, Nm, Nn, Ny, Nx, Nz)
        # sqrt_n operates on axis 2 (Nn dimension)
        sqrt_n_plus = self.sqrt_n_plus[None, None, :, None, None, None]  # (1, 1, Nn, 1, 1, 1)
        sqrt_n_minus = self.sqrt_n_minus[None, None, :, None, None, None]
        # sqrt_m operates on axis 1 (Nm dimension)
        sqrt_m_plus = self.sqrt_m_plus[None, :, None, None, None, None]  # (1, Nm, 1, 1, 1, 1)
        sqrt_m_minus = self.sqrt_m_minus[None, :, None, None, None, None]
        # sqrt_p operates on axis 0 (Np dimension)
        sqrt_p_plus = self.sqrt_p_plus[:, None, None, None, None, None]  # (Np, 1, 1, 1, 1, 1)
        sqrt_p_minus = self.sqrt_p_minus[:, None, None, None, None, None]

        # Compute auxiliary velocity-space coupling fields for magnetic Lorentz force
        C_aux_x = (
            sqrt_m_minus * sqrt_p_minus * (a2 / a1 - a1 / a2) * self.shift_multi(C, dn=0, dm=-1, dp=-1)
            + sqrt_m_minus * sqrt_p_plus * (a2 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=1)
            - sqrt_m_plus * sqrt_p_minus * (a1 / a2) * self.shift_multi(C, dn=0, dm=1, dp=-1)
            + jnp.sqrt(2) * sqrt_m_minus * (u2 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=0)
            - jnp.sqrt(2) * sqrt_p_minus * (u1 / a2) * self.shift_multi(C, dn=0, dm=0, dp=-1)
        )

        C_aux_y = (
            sqrt_n_minus * sqrt_p_minus * (a0 / a2 - a2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=-1)
            + sqrt_n_plus * sqrt_p_minus * (a0 / a2) * self.shift_multi(C, dn=1, dm=0, dp=-1)
            - sqrt_n_minus * sqrt_p_plus * (a2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=1)
            + jnp.sqrt(2) * sqrt_p_minus * (u0 / a2) * self.shift_multi(C, dn=0, dm=0, dp=-1)
            - jnp.sqrt(2) * sqrt_n_minus * (u2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=0)
        )

        C_aux_z = (
            sqrt_n_minus * sqrt_m_minus * (a1 / a0 - a0 / a1) * self.shift_multi(C, dn=-1, dm=-1, dp=0)
            + sqrt_n_minus * sqrt_m_plus * (a1 / a0) * self.shift_multi(C, dn=-1, dm=1, dp=0)
            - sqrt_n_plus * sqrt_m_minus * (a0 / a1) * self.shift_multi(C, dn=1, dm=-1, dp=0)
            + jnp.sqrt(2) * sqrt_n_minus * (u1 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=0)
            - jnp.sqrt(2) * sqrt_m_minus * (u0 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=0)
        )

        # Collision term (hypercollisional damping)
        Col = -nu * self.col[:, :, :, None, None, None] * Ck

        # Diffusion term (high-k stabilization)
        Diff = -D * self.k2_grid * Ck

        # ODEs for Hermite-Fourier coefficients
        # Closure is achieved by setting to zero coefficients with index out of range (or using NN prediction)
        dCk_s_dt = (
            # Spatial advection in x-direction
            -(self.kx_grid * (1j / self.Lx))
            * a0
            * (
                sqrt_n_plus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=1, dm=0, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + sqrt_n_minus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=-1, dm=0, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + (u0 / a0) * Ck
            )
            # Spatial advection in y-direction
            - (self.ky_grid * (1j / self.Ly))
            * a1
            * (
                sqrt_m_plus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=1, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + sqrt_m_minus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=-1, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + (u1 / a1) * Ck
            )
            # Spatial advection in z-direction
            - (self.kz_grid * 1j / self.Lz)
            * a2
            * (
                sqrt_p_plus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=0, dp=1, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + sqrt_p_minus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=0, dp=-1, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + (u2 / a2) * Ck
            )
            # Lorentz force (E-field + B-field coupling via auxiliary fields)
            + q
            * Omega_c
            * (
                jnp.fft.fftn(
                    (sqrt_n_minus * jnp.sqrt(2) / a0) * F[0] * self.shift_multi(C, dn=-1, dm=0, dp=0)
                    + (sqrt_m_minus * jnp.sqrt(2) / a1) * F[1] * self.shift_multi(C, dn=0, dm=-1, dp=0)
                    + (sqrt_p_minus * jnp.sqrt(2) / a2) * F[2] * self.shift_multi(C, dn=0, dm=0, dp=-1)
                    + F[3] * C_aux_x
                    + F[4] * C_aux_y
                    + F[5] * C_aux_z,
                    axes=(-3, -2, -1),
                    norm="forward",
                )
                * self.mask23
            )
            # Add collision and diffusion terms
            + Col
            + Diff
        )

        return dCk_s_dt

    def _compute_lorentz_rhs(
        self,
        C: Array,
        F: Array,
        alpha: Array,
        u: Array,
        q: float,
        Omega_c: float,
    ) -> Array:
        """
        Compute only the Lorentz force (E-field + B-field) contribution to dCk/dt.

        Used by the Lawson-RK4 exponential integrator where free-streaming, collision,
        and diffusion are handled by exact exponentials.

        Args:
            C: Real-space Hermite coefficients, shape (Np, Nm, Nn, Ny, Nx, Nz) - Single species
            F: Real-space EM fields, shape (6, Ny, Nx, Nz) = (Ex, Ey, Ez, Bx, By, Bz)
            alpha: Thermal velocities, shape (3,) = (alpha_x, alpha_y, alpha_z)
            u: Drift velocities, shape (3,) = (u_x, u_y, u_z)
            q: Species charge (scalar)
            Omega_c: Cyclotron frequency (scalar)

        Returns:
            Lorentz force dCk/dt with shape (Np, Nm, Nn, Ny, Nx, Nz)
        """
        # Extract thermal velocity and drift velocity components
        a0, a1, a2 = alpha[0], alpha[1], alpha[2]
        u0, u1, u2 = u[0], u[1], u[2]

        # Broadcast fields to match Hermite dimensions
        F = F[:, None, None, None, :, :, :]  # (6, 1, 1, 1, Ny, Nx, Nz)

        # Reshape Hermite ladder operators for broadcasting with 6D arrays
        sqrt_n_minus = self.sqrt_n_minus[None, None, :, None, None, None]
        sqrt_m_minus = self.sqrt_m_minus[None, :, None, None, None, None]
        sqrt_p_minus = self.sqrt_p_minus[:, None, None, None, None, None]
        sqrt_n_plus = self.sqrt_n_plus[None, None, :, None, None, None]
        sqrt_m_plus = self.sqrt_m_plus[None, :, None, None, None, None]
        sqrt_p_plus = self.sqrt_p_plus[:, None, None, None, None, None]

        # Compute auxiliary velocity-space coupling fields for magnetic Lorentz force
        C_aux_x = (
            sqrt_m_minus * sqrt_p_minus * (a2 / a1 - a1 / a2) * self.shift_multi(C, dn=0, dm=-1, dp=-1)
            + sqrt_m_minus * sqrt_p_plus * (a2 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=1)
            - sqrt_m_plus * sqrt_p_minus * (a1 / a2) * self.shift_multi(C, dn=0, dm=1, dp=-1)
            + jnp.sqrt(2) * sqrt_m_minus * (u2 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=0)
            - jnp.sqrt(2) * sqrt_p_minus * (u1 / a2) * self.shift_multi(C, dn=0, dm=0, dp=-1)
        )

        C_aux_y = (
            sqrt_n_minus * sqrt_p_minus * (a0 / a2 - a2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=-1)
            + sqrt_n_plus * sqrt_p_minus * (a0 / a2) * self.shift_multi(C, dn=1, dm=0, dp=-1)
            - sqrt_n_minus * sqrt_p_plus * (a2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=1)
            + jnp.sqrt(2) * sqrt_p_minus * (u0 / a2) * self.shift_multi(C, dn=0, dm=0, dp=-1)
            - jnp.sqrt(2) * sqrt_n_minus * (u2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=0)
        )

        C_aux_z = (
            sqrt_n_minus * sqrt_m_minus * (a1 / a0 - a0 / a1) * self.shift_multi(C, dn=-1, dm=-1, dp=0)
            + sqrt_n_minus * sqrt_m_plus * (a1 / a0) * self.shift_multi(C, dn=-1, dm=1, dp=0)
            - sqrt_n_plus * sqrt_m_minus * (a0 / a1) * self.shift_multi(C, dn=1, dm=-1, dp=0)
            + jnp.sqrt(2) * sqrt_n_minus * (u1 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=0)
            - jnp.sqrt(2) * sqrt_m_minus * (u0 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=0)
        )

        # Lorentz force: E-field + B-field coupling via auxiliary fields
        return (
            q
            * Omega_c
            * (
                jnp.fft.fftn(
                    (sqrt_n_minus * jnp.sqrt(2) / a0) * F[0] * self.shift_multi(C, dn=-1, dm=0, dp=0)
                    + (sqrt_m_minus * jnp.sqrt(2) / a1) * F[1] * self.shift_multi(C, dn=0, dm=-1, dp=0)
                    + (sqrt_p_minus * jnp.sqrt(2) / a2) * F[2] * self.shift_multi(C, dn=0, dm=0, dp=-1)
                    + F[3] * C_aux_x
                    + F[4] * C_aux_y
                    + F[5] * C_aux_z,
                    axes=(-3, -2, -1),
                    norm="forward",
                )
                * self.mask23
            )
        )

    def __call__(
        self,
        Ck: Array,
        C: Array,
        F: Array,
        nu: float,
        D: float,
        alpha: Array,
        u: Array,
        q: float,
        Omega_c: float,
        closure_n=None,
        closure_m=None,
        closure_p=None,
    ) -> Array:
        """
        Compute dCk/dt for the Hermite-Fourier system for a single species.

        Args:
            Ck: Spectral Hermite coefficients, shape (Np, Nm, Nn, Ny, Nx, Nz) - Single species
            C: Real-space Hermite coefficients, shape (Np, Nm, Nn, Ny, Nx, Nz) - Single species
            F: Real-space EM fields, shape (6, Ny, Nx, Nz) = (Ex, Ey, Ez, Bx, By, Bz)
            nu: Collision frequency
            D: Hyper-diffusion coefficient
            alpha: Thermal velocities for single species, shape (3,) = (alpha_x, alpha_y, alpha_z)
            u: Drift velocities for single species, shape (3,) = (u_x, u_y, u_z)
            q: Species charge (scalar)
            Omega_c: Cyclotron frequency (scalar)
            closure_n: Optional closure for n-direction (x-velocity)
            closure_m: Optional closure for m-direction (y-velocity)
            closure_p: Optional closure for p-direction (z-velocity)

        Returns:
            dCk/dt with shape (Np, Nm, Nn, Ny, Nx, Nz)
        """
        # Simple wrapper - sharding is now handled at higher level (VectorField)
        return self._compute_rhs(Ck, C, F, nu, D, alpha, u, q, Omega_c, closure_n, closure_m, closure_p)
