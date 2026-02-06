"""Vector field class for the Spectrax Vlasov-Maxwell system."""

from jax import Array
from jax import numpy as jnp
from spectrax import cross_product

from adept._spectrax1d.driver import Driver
from adept._spectrax1d.hermite_fourier_ode import HermiteFourierODE


class SpectraxVectorField:
    """
    Vector field class for the Spectrax Vlasov-Maxwell system.

    This class encapsulates the right-hand side of the ODE system for diffrax,
    including external field driving.

    By storing static grid parameters as class attributes, we avoid the need for
    static_argnames in JIT compilation and make the code more readable.

    State and args are now dictionaries for improved clarity:
    - state: {"Ck": Array, "Fk": Array}
    - args: {"qs": Array, "nu": float, "D": float, ...}

    Follows the pattern from adept/_vlasov1d/solvers/vector_field.py

    Complex array handling:
    - Diffrax doesn't fully support complex state, so we use float64 views
    - _unpack_y_() converts float64 views to complex128 at start of __call__
    - _pack_y_() converts back to float64 views at end of __call__
    - This pattern matches LPSE2D implementation

    Args:
        Nx, Ny, Nz: Number of Fourier modes per spatial dimension
        Nn, Nm, Np: Number of Hermite modes per velocity dimension
        Ns: Number of species
        xax: Real-space x grid for driver computation
        driver_config: Driver configuration dict from YAML
        Lx, Ly, Lz: Domain sizes
        kx_grid, ky_grid, kz_grid, k2_grid: Wave vector grids
        nabla: Gradient operator in Fourier space
        col: Collision matrix
        sqrt_n_plus, sqrt_n_minus: Hermite ladder operators for n-direction
        sqrt_m_plus, sqrt_m_minus: Hermite ladder operators for m-direction
        sqrt_p_plus, sqrt_p_minus: Hermite ladder operators for p-direction
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        Nn: int,
        Nm: int,
        Np: int,
        Ns: int,
        xax: Array,
        driver_config: dict,
        Lx: float,
        Ly: float,
        Lz: float,
        kx_grid: Array,
        ky_grid: Array,
        kz_grid: Array,
        k2_grid: Array,
        nabla: Array,
        col: Array,
        sqrt_n_plus: Array,
        sqrt_n_minus: Array,
        sqrt_m_plus: Array,
        sqrt_m_minus: Array,
        sqrt_p_plus: Array,
        sqrt_p_minus: Array,
    ):
        """Initialize with static grid dimensions, grid quantities, and driver configuration."""
        # Store dimensions as attributes
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nn = Nn
        self.Nm = Nm
        self.Np = Np
        self.Ns = Ns

        # Store grid quantities as class attributes (constants)
        self.nabla = nabla

        # Initialize external field driver if configured
        if driver_config.get("ex"):
            self.driver = Driver(xax, Nx, Ny, Nz, driver_key="ex")
            self.driver_config = driver_config
            self.has_driver = True
        else:
            self.driver = None
            self.driver_config = None
            self.has_driver = False

        # Pre-compute the 2/3 dealiasing mask (only depends on grid dimensions)
        # Use fftshifted format to match spectrax convention
        self.mask23 = self._compute_twothirds_mask()

        # Pre-compute Hou-Li filter in Hermite space if enabled
        filter_config = driver_config.get("hermite_filter", {})
        self.use_hermite_filter = filter_config.get("enabled", False)
        if self.use_hermite_filter:
            filter_order = filter_config.get("order", 36)
            filter_strength = filter_config.get("strength", 36.0)
            cutoff_fraction = filter_config.get("cutoff_fraction", 2.0 / 3.0)
            self.hermite_filter = self._compute_houli_hermite_filter(
                Nn, Nm, Np, cutoff_fraction, filter_strength, filter_order
            )
        else:
            self.hermite_filter = None

        # Instantiate class-based Hermite-Fourier ODE system (NEW refactored approach)
        self.hermite_fourier_ode = HermiteFourierODE(
            Nn=Nn,
            Nm=Nm,
            Np=Np,
            Ns=Ns,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            kz_grid=kz_grid,
            k2_grid=k2_grid,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            col=col,
            sqrt_n_plus=sqrt_n_plus,
            sqrt_n_minus=sqrt_n_minus,
            sqrt_m_plus=sqrt_m_plus,
            sqrt_m_minus=sqrt_m_minus,
            sqrt_p_plus=sqrt_p_plus,
            sqrt_p_minus=sqrt_p_minus,
            mask23=self.mask23,
        )

        # List of state variables that are complex (for unpacking/packing)
        self.complex_state_vars = ["Ck", "Fk"]

    def _compute_twothirds_mask(self) -> Array:
        """
        Compute the 2/3 dealiasing mask in fftshifted format (spectrax convention).

        Returns a boolean mask that keeps |k| <= N//3 in each dimension.
        Uses fftshifted ordering (DC at center) to match spectrax library.
        """

        def centered_modes(N):
            # integer mode numbers in fftshifted ordering (DC at center)
            k = jnp.fft.fftfreq(N) * N
            return jnp.fft.fftshift(k)  # Shift to centered ordering

        ky = centered_modes(self.Ny)[:, None, None]
        kx = centered_modes(self.Nx)[None, :, None]
        kz = centered_modes(self.Nz)[None, None, :]

        # cutoffs (keep indices with |k| <= floor(N/3)); if N<3 this naturally keeps only k=0
        cy = self.Ny // 3
        cx = self.Nx // 3
        cz = self.Nz // 3

        return (jnp.abs(ky) <= cy) & (jnp.abs(kx) <= cx) & (jnp.abs(kz) <= cz)

    def _compute_plasma_current(self, Ck: Array, qs: Array, alpha_s: Array, u_s: Array) -> Array:
        """
        Compute the spectral Ampère-Maxwell current from Hermite-Fourier coefficients.

        This replaces the external spectrax.plasma_current() to work directly with 7D Ck
        without requiring a reshape.

        The plasma current is:
            J = sum_s q_s * alpha_s * (drift_velocity_term + thermal_velocity_term)

        Args:
            Ck: Hermite-Fourier coefficients, shape (Ns, Np, Nm, Nn, Ny, Nx, Nz)
            qs: Species charges, shape (Ns,)
            alpha_s: Thermal velocity scaling factors, shape (3*Ns,)
            u_s: Drift velocities, shape (3*Ns,)

        Returns:
            Current density in Fourier space, shape (3, Ny, Nx, Nz) for (Jx, Jy, Jz)
        """
        # Reshape alpha and velocity from (3*Ns,) to (Ns, 3)
        alpha = alpha_s.reshape(self.Ns, 3)
        u = u_s.reshape(self.Ns, 3)

        # Extract specific Hermite modes for current calculation
        # C0 = (p,m,n) = (0,0,0) mode - density
        C0 = Ck[:, 0, 0, 0]  # shape: (Ns, Ny, Nx, Nz)

        # C100 = (0,0,1) mode - x-velocity moment (if Nn > 1, else zero)
        C100 = Ck[:, 0, 0, 1] if self.Nn > 1 else jnp.zeros_like(C0)

        # C010 = (0,1,0) mode - y-velocity moment (if Nm > 1, else zero)
        C010 = Ck[:, 0, 1, 0] if self.Nm > 1 else jnp.zeros_like(C0)

        # C001 = (1,0,0) mode - z-velocity moment (if Np > 1, else zero)
        C001 = Ck[:, 1, 0, 0] if self.Np > 1 else jnp.zeros_like(C0)

        # Extract alpha and u components for each direction
        a0, a1, a2 = alpha[:, 0], alpha[:, 1], alpha[:, 2]  # shape: (Ns,)
        u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]  # shape: (Ns,)

        # Prefactor: q * alpha_x * alpha_y * alpha_z
        pre = qs * a0 * a1 * a2  # shape: (Ns,)

        # Thermal velocity contribution: (1/sqrt(2)) * alpha_i * C_i
        # Shape: (3, Ns, Ny, Nx, Nz)
        term1 = (1.0 / jnp.sqrt(2.0)) * jnp.stack([
            a0[:, None, None, None] * C100,
            a1[:, None, None, None] * C010,
            a2[:, None, None, None] * C001
        ], axis=0)

        # Drift velocity contribution: u_i * C0
        # Shape: (3, Ns, Ny, Nx, Nz)
        term2 = jnp.stack([
            u0[:, None, None, None] * C0,
            u1[:, None, None, None] * C0,
            u2[:, None, None, None] * C0
        ], axis=0)

        # Current per species: (term1 + term2) * prefactor
        # Shape: (3, Ns, Ny, Nx, Nz)
        J_species = (term1 + term2) * pre[None, :, None, None, None]

        # Sum over species to get total current
        # Shape: (3, Ny, Nx, Nz)
        return jnp.sum(J_species, axis=1)

    def _compute_houli_hermite_filter(
        self, Nn: int, Nm: int, Np: int, cutoff_fraction: float, strength: float, order: int
    ) -> Array:
        """
        Compute Hou-Li exponential filter in Hermite space.

        The Hou-Li filter smoothly damps high Hermite modes using an exponential profile:
            H(n, m, p) = exp(-strength * ((h/h_cutoff)^order))
        where h = sqrt(n^2 + m^2 + p^2) is the Hermite mode magnitude.

        Args:
            Nn, Nm, Np: Number of Hermite modes in each direction
            cutoff_fraction: Fraction of max mode where filtering begins (typically 2/3)
            strength: Filter strength parameter (typically 36)
            order: Filter order (typically 36 for sharp cutoff)

        Returns:
            Filter array with shape (Np, Nm, Nn) containing filter coefficients [0, 1]
        """
        # Create Hermite mode index grids
        n = jnp.arange(Nn)[None, None, :]  # Shape (1, 1, Nn)
        m = jnp.arange(Nm)[None, :, None]  # Shape (1, Nm, 1)
        p = jnp.arange(Np)[:, None, None]  # Shape (Np, 1, 1)

        # Compute maximum Hermite mode magnitude
        h_max = jnp.sqrt((Nn - 1) ** 2 + (Nm - 1) ** 2 + (Np - 1) ** 2)

        # Cutoff at specified fraction of max
        h_cutoff = cutoff_fraction * h_max

        # Compute mode magnitude for each (n, m, p)
        h = jnp.sqrt(n**2 + m**2 + p**2)

        # Apply Hou-Li filter: exp(-strength * (h/h_cutoff)^order)
        # Only filter modes above cutoff
        filter_arg = jnp.where(h > h_cutoff, strength * ((h / h_cutoff) ** order), 0.0)
        hermite_filter = jnp.exp(-filter_arg)

        return hermite_filter

    def _unpack_y_(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Unpack state from float64 views to complex128.

        Diffrax doesn't fully support complex state, so arrays are passed as float64 views.
        This method converts them back to complex128 for computation.

        Args:
            y: State dictionary with float64 views

        Returns:
            State dictionary with complex128 arrays
        """
        new_y = {}
        for k in y.keys():
            if k in self.complex_state_vars:
                new_y[k] = y[k].view(jnp.complex128)
            else:
                new_y[k] = y[k].view(jnp.float64)
        return new_y

    def _pack_y_(self, y: dict[str, Array], new_y: dict[str, Array]) -> tuple[dict[str, Array], dict[str, Array]]:
        """
        Pack state and time derivatives back to float64 views.

        Converts complex128 arrays back to float64 views for diffrax.

        Args:
            y: Original state dictionary
            new_y: Time derivatives dictionary

        Returns:
            Tuple of (y, new_y) both with float64 views
        """
        for k in y.keys():
            y[k] = y[k].view(jnp.float64)
            new_y[k] = new_y[k].view(jnp.float64)

        return y, new_y

    def __call__(self, t: float, y: dict, args: dict) -> dict:
        """
        Compute the right-hand side of the Vlasov-Maxwell ODE system.

        This is the interface required by diffrax.ODETerm.

        Args:
            t: Current time
            y: State dictionary with keys "Ck" and "Fk" (float64 views)
            args: Dictionary of physical parameters (qs, nu, D, Omega_cs, etc.)

        Returns:
            Dictionary with time derivatives of Ck and Fk (float64 views)
        """
        # Unpack state from float64 views to complex128
        new_y = self._unpack_y_(y)

        # Unpack state dictionary
        Ck = new_y["Ck"]
        Fk = new_y["Fk"]

        # Unpack physical parameters from args dictionary
        qs = args["qs"]
        nu = args["nu"]
        D = args["D"]
        Omega_cs = args["Omega_cs"]
        alpha_s = args["alpha_s"]
        u_s = args["u_s"]
        # Unpack optional closure functions for neural network Hermite truncation
        closure_n = args.get("closure_n", None)
        closure_m = args.get("closure_m", None)
        closure_p = args.get("closure_p", None)

        # Apply 2/3 dealiasing and inverse FFT to get real-space fields
        # Use fftshifted format (spectrax convention: k=0 at center)
        F = jnp.fft.ifftn(jnp.fft.ifftshift(Fk * self.mask23, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward")
        C = jnp.fft.ifftn(jnp.fft.ifftshift(Ck * self.mask23, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward")

        # Compute time derivative of distribution function
        # Returns 7D array: (Ns, Np, Nm, Nn, Ny, Nx, Nz)

        # NEW: Class-based approach with optional neural network closure
        dCk_s_dt = self.hermite_fourier_ode(
            Ck=Ck,
            C=C,
            F=F,
            nu=nu,
            D=D,
            alpha_s=alpha_s,
            u_s=u_s,
            qs=qs,
            Omega_cs=Omega_cs,
            closure_n=closure_n,
            closure_m=closure_m,
            closure_p=closure_p,
        )

        # Apply Hou-Li filter in Hermite space if enabled
        if self.use_hermite_filter:
            # Apply filter: hermite_filter has shape (Np, Nm, Nn), broadcast over (Ns, Ny, Nx, Nz)
            filter_broadcast = self.hermite_filter[None, :, :, :, None, None, None]  # (1, Np, Nm, Nn, 1, 1, 1)
            dCk_s_dt = dCk_s_dt * filter_broadcast

        # Compute time derivative of magnetic field (Faraday's law)
        dBk_dt = -1j * cross_product(self.nabla, Fk[:3])

        # Get external driver field in Fourier space (if configured)
        # Driver returns complex array of shape (3, Ny, Nx, Nz)
        if self.has_driver:
            driver = self.driver(self.driver_config, t)
        else:
            driver = jnp.zeros_like(Fk[:3])

        # Compute plasma current and time derivative of electric field (Ampere's law)
        # Use our internal 7D-native plasma current calculation
        current = self._compute_plasma_current(Ck, qs, alpha_s, u_s)
        dEk_dt = 1j * cross_product(self.nabla, Fk[3:]) - current / Omega_cs[0] + driver

        # Concatenate field derivatives
        dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)

        # Create time derivatives dictionary
        dy_dt = {"Ck": dCk_s_dt, "Fk": dFk_dt}

        # Pack state and derivatives back to float64 views
        y, dy_dt = self._pack_y_(y, dy_dt)

        # Return dictionary with time derivatives (must match state structure)
        return dy_dt
