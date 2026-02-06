"""
Hermite-Fourier ODE system for Vlasov-Maxwell equations.

This module implements the RHS of the Hermite-Fourier moment equations,
computing dCk/dt for the distribution function in spectral space.

This is a refactored class-based version of the external spectrax.Hermite_Fourier_system()
function, with grid quantities stored as class attributes to simplify the call signature.
"""

from jax import Array, jit
from jax import numpy as jnp


class HermiteFourierODE:
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
        Ns: Number of species
        kx_grid, ky_grid, kz_grid: Wavenumber grids (fftshifted, shape (Ny, Nx, Nz))
        k2_grid: Squared wavenumber magnitude
        Lx, Ly, Lz: Domain lengths
        col: Collision matrix, shape (Np, Nm, Nn)
        sqrt_n_plus, sqrt_n_minus: Hermite ladder operators for n (x-velocity)
        sqrt_m_plus, sqrt_m_minus: Hermite ladder operators for m (y-velocity)
        sqrt_p_plus, sqrt_p_minus: Hermite ladder operators for p (z-velocity)
        mask23: 2/3 dealiasing mask in Fourier space
    """

    def __init__(
        self,
        Nn: int,
        Nm: int,
        Np: int,
        Ns: int,
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
        self.Ns = Ns

        # Store k-space grids
        self.kx_grid = kx_grid
        self.ky_grid = ky_grid
        self.kz_grid = kz_grid
        self.k2_grid = k2_grid

        # Store domain lengths
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # Store collision matrix
        self.col = col

        # Store Hermite ladder operators
        self.sqrt_n_plus = sqrt_n_plus
        self.sqrt_n_minus = sqrt_n_minus
        self.sqrt_m_plus = sqrt_m_plus
        self.sqrt_m_minus = sqrt_m_minus
        self.sqrt_p_plus = sqrt_p_plus
        self.sqrt_p_minus = sqrt_p_minus

        # Store dealiasing mask
        self.mask23 = mask23

    def _pad_hermite_axes(self, Ck: Array) -> Array:
        """
        Zero-pad Hermite axes (p, m, n) by 1 on each side.

        Args:
            Ck: Array with shape (Ns, Np, Nm, Nn, Ny, Nx, Nz)

        Returns:
            Padded array with shape (Ns, Np+2, Nm+2, Nn+2, Ny, Nx, Nz)
        """
        # Pad +1 on both sides for species axis 1,2,3 (p,m,n) only
        return jnp.pad(Ck, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0)))

    def shift_multi(
        self,
        Ck: Array,
        dn: int = 0,
        dm: int = 0,
        dp: int = 0,
        closure_n = None,
        closure_m = None,
        closure_p = None,
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
            Ck: Array with shape (Ns, Np, Nm, Nn, Ny, Nx, Nz)
            dn, dm, dp: Shift amounts for n, m, p indices (values in {-1, 0, +1})
            closure_n: Optional closure for n-direction (x-velocity)
            closure_m: Optional closure for m-direction (y-velocity)
            closure_p: Optional closure for p-direction (z-velocity)

        Returns:
            Shifted array with same shape, with closure-predicted or zero-padded boundaries
        """
        Ns, Np, Nm, Nn, Ny, Nx, Nz = Ck.shape

        # Start with zero-padding (default behavior)
        P = self._pad_hermite_axes(Ck)

        # Apply neural network closures for out-of-range modes

        # N-direction closure (x-velocity)
        if dn == +1 and closure_n is not None:
            # Accessing n+1 from highest mode (n=Nn-1) -> predict mode Nn
            # closure_n should return shape (Ns, Np, Nm, 1, Ny, Nx, Nz)
            C_Nn = closure_n(Ck)
            # Insert predicted mode at the right boundary of padded array
            # P has shape (Ns, Np+2, Nm+2, Nn+2, Ny, Nx, Nz)
            # The boundary is at index Nn+1 (0-indexed: 0=pad, 1...Nn=data, Nn+1=pad)
            P = P.at[:, 1:-1, 1:-1, -1, :, :, :].set(C_Nn.squeeze(axis=3))

        # M-direction closure (y-velocity)
        if dm == +1 and closure_m is not None:
            C_Nm = closure_m(Ck)
            P = P.at[:, 1:-1, -1, 1:-1, :, :, :].set(C_Nm.squeeze(axis=2))

        # P-direction closure (z-velocity)
        if dp == +1 and closure_p is not None:
            C_Np = closure_p(Ck)
            P = P.at[:, -1, 1:-1, 1:-1, :, :, :].set(C_Np.squeeze(axis=1))

        # Extract shifted region (same as before)
        n0 = 1 + dn  # dn=+1 -> 0 ; dn=0 -> 1 ; dn=-1 -> 2
        m0 = 1 + dm
        p0 = 1 + dp

        return P[:, p0 : p0 + Np, m0 : m0 + Nm, n0 : n0 + Nn, :, :, :]

    def __call__(
        self,
        Ck: Array,
        C: Array,
        F: Array,
        nu: float,
        D: float,
        alpha_s: Array,
        u_s: Array,
        qs: Array,
        Omega_cs: Array,
        closure_n=None,
        closure_m=None,
        closure_p=None,
    ) -> Array:
        """
        Compute dCk/dt for the Hermite-Fourier system.

        This is a direct copy of the external spectrax.Hermite_Fourier_system() function,
        refactored to use class attributes for grid quantities.

        Args:
            Ck: Spectral Hermite coefficients, shape (Ns*Np*Nm*Nn, Ny, Nx, Nz)
            C: Real-space Hermite coefficients, shape (Ns*Np*Nm*Nn, Ny, Nx, Nz)
            F: Real-space EM fields, shape (6, Ny, Nx, Nz) = (Ex, Ey, Ez, Bx, By, Bz)
            nu: Collision frequency
            D: Hyper-diffusion coefficient
            alpha_s: Thermal velocities, shape (Ns*3,) or (Ns, 3)
            u_s: Drift velocities, shape (Ns*3,) or (Ns, 3)
            qs: Species charges, shape (Ns,)
            Omega_cs: Cyclotron frequencies, shape (Ns,)
            closure_n: Optional closure for n-direction (x-velocity)
            closure_m: Optional closure for m-direction (y-velocity)
            closure_p: Optional closure for p-direction (z-velocity)

        Returns:
            dCk/dt with shape (Ns, Np, Nm, Nn, Ny, Nx, Nz)
        """
        # Reshape inputs to 7D format (from 4D)
        Ck = Ck.reshape(self.Ns, self.Np, self.Nm, self.Nn, *Ck.shape[-3:])
        C = C.reshape(self.Ns, self.Np, self.Nm, self.Nn, *C.shape[-3:])
        F = F[:, None, None, None, None, :, :, :]  # (6,1,1,1,1,Ny,Nx,Nz) for broadcasting
        Ny, Nx, Nz = Ck.shape[-3], Ck.shape[-2], Ck.shape[-1]

        # Reshape and broadcast species-specific parameters
        alpha = alpha_s.reshape(self.Ns, 3)
        u = u_s.reshape(self.Ns, 3)
        a0 = alpha[:, 0][:, None, None, None, None, None, None]
        a1 = alpha[:, 1][:, None, None, None, None, None, None]
        a2 = alpha[:, 2][:, None, None, None, None, None, None]
        u0 = u[:, 0][:, None, None, None, None, None, None]
        u1 = u[:, 1][:, None, None, None, None, None, None]
        u2 = u[:, 2][:, None, None, None, None, None, None]
        q = qs[:, None, None, None, None, None, None]
        Omega_c = Omega_cs[:, None, None, None, None, None, None]

        # Compute auxiliary velocity-space coupling fields for magnetic Lorentz force
        C_aux_x = (
            self.sqrt_m_minus * self.sqrt_p_minus * (a2 / a1 - a1 / a2) * self.shift_multi(C, dn=0, dm=-1, dp=-1)
            + self.sqrt_m_minus * self.sqrt_p_plus * (a2 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=1)
            - self.sqrt_m_plus * self.sqrt_p_minus * (a1 / a2) * self.shift_multi(C, dn=0, dm=1, dp=-1)
            + jnp.sqrt(2) * self.sqrt_m_minus * (u2 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=0)
            - jnp.sqrt(2) * self.sqrt_p_minus * (u1 / a2) * self.shift_multi(C, dn=0, dm=0, dp=-1)
        )

        C_aux_y = (
            self.sqrt_n_minus * self.sqrt_p_minus * (a0 / a2 - a2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=-1)
            + self.sqrt_n_plus * self.sqrt_p_minus * (a0 / a2) * self.shift_multi(C, dn=1, dm=0, dp=-1)
            - self.sqrt_n_minus * self.sqrt_p_plus * (a2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=1)
            + jnp.sqrt(2) * self.sqrt_p_minus * (u0 / a2) * self.shift_multi(C, dn=0, dm=0, dp=-1)
            - jnp.sqrt(2) * self.sqrt_n_minus * (u2 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=0)
        )

        C_aux_z = (
            self.sqrt_n_minus * self.sqrt_m_minus * (a1 / a0 - a0 / a1) * self.shift_multi(C, dn=-1, dm=-1, dp=0)
            + self.sqrt_n_minus * self.sqrt_m_plus * (a1 / a0) * self.shift_multi(C, dn=-1, dm=1, dp=0)
            - self.sqrt_n_plus * self.sqrt_m_minus * (a0 / a1) * self.shift_multi(C, dn=1, dm=-1, dp=0)
            + jnp.sqrt(2) * self.sqrt_n_minus * (u1 / a0) * self.shift_multi(C, dn=-1, dm=0, dp=0)
            - jnp.sqrt(2) * self.sqrt_m_minus * (u0 / a1) * self.shift_multi(C, dn=0, dm=-1, dp=0)
        )

        # Collision term (hypercollisional damping)
        Col = -nu * self.col[None, :, :, :, None, None, None] * Ck

        # Diffusion term (high-k stabilization)
        Diff = -D * self.k2_grid * Ck

        # ODEs for Hermite-Fourier coefficients
        # Closure is achieved by setting to zero coefficients with index out of range (or using NN prediction)
        dCk_s_dt = (
            # Spatial advection in x-direction
            -(self.kx_grid * (1j / self.Lx))
            * a0
            * (
                self.sqrt_n_plus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=1, dm=0, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + self.sqrt_n_minus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=-1, dm=0, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + (u0 / a0) * Ck
            )
            # Spatial advection in y-direction
            - (self.ky_grid * (1j / self.Ly))
            * a1
            * (
                self.sqrt_m_plus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=1, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + self.sqrt_m_minus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=-1, dp=0, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + (u1 / a1) * Ck
            )
            # Spatial advection in z-direction
            - (self.kz_grid * 1j / self.Lz)
            * a2
            * (
                self.sqrt_p_plus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=0, dp=1, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + self.sqrt_p_minus
                / jnp.sqrt(2)
                * self.shift_multi(Ck, dn=0, dm=0, dp=-1, closure_n=closure_n, closure_m=closure_m, closure_p=closure_p)
                + (u2 / a2) * Ck
            )
            # Lorentz force (E-field + B-field coupling via auxiliary fields)
            + q
            * Omega_c
            * (
                jnp.fft.fftshift(
                    jnp.fft.fftn(
                        (self.sqrt_n_minus * jnp.sqrt(2) / a0) * F[0] * self.shift_multi(C, dn=-1, dm=0, dp=0)
                        + (self.sqrt_m_minus * jnp.sqrt(2) / a1) * F[1] * self.shift_multi(C, dn=0, dm=-1, dp=0)
                        + (self.sqrt_p_minus * jnp.sqrt(2) / a2) * F[2] * self.shift_multi(C, dn=0, dm=0, dp=-1)
                        + F[3] * C_aux_x
                        + F[4] * C_aux_y
                        + F[5] * C_aux_z,
                        axes=(-3, -2, -1),
                        norm="forward",
                    ),
                    axes=(-3, -2, -1),
                )
                * self.mask23
            )
            # Add collision and diffusion terms
            + Col
            + Diff
        )

        return dCk_s_dt
