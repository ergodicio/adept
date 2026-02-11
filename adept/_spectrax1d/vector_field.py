"""Vector field class for the Spectrax Vlasov-Maxwell system."""

import jax
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
        Nn_electrons, Nm_electrons, Np_electrons: Electron Hermite modes per velocity dimension
        Nn_ions, Nm_ions, Np_ions: Ion Hermite modes per velocity dimension
        Ns: Number of species
        xax: Real-space x grid for driver computation
        driver_config: Driver configuration dict from YAML
        grid_quantities_electrons: Dict with electron-specific ladder operators, collision matrix, k-grids
        grid_quantities_ions: Dict with ion-specific ladder operators, collision matrix, k-grids
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        Nn_electrons: int,
        Nm_electrons: int,
        Np_electrons: int,
        Nn_ions: int,
        Nm_ions: int,
        Np_ions: int,
        Ns: int,
        xax: Array,
        driver_config: dict,
        grid_quantities_electrons: dict,
        grid_quantities_ions: dict,
        dt: float,
        use_shard_map: bool = False,
    ):
        """Initialize with per-species mode counts and grid quantities."""
        # Store dimensions as attributes
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nn_electrons = Nn_electrons
        self.Nm_electrons = Nm_electrons
        self.Np_electrons = Np_electrons
        self.Nn_ions = Nn_ions
        self.Nm_ions = Nm_ions
        self.Np_ions = Np_ions
        self.Ns = Ns

        # Store shared grid quantities (nabla is species-independent)
        self.nabla = grid_quantities_electrons["nabla"]

        # Initialize external field drivers if configured
        self.drivers = {}
        for driver_key in ["ex", "ey", "ez"]:
            if driver_config.get(driver_key):
                self.drivers[driver_key] = Driver(xax, Nx, Ny, Nz, driver_key=driver_key)

        if self.drivers:
            self.driver_config = driver_config
            self.has_driver = True
        else:
            self.driver_config = None
            self.has_driver = False

        # Store timestep for density noise generation
        self.dt = dt

        # Initialize density noise configuration
        noise_config = driver_config.get("density_noise", {})
        self.noise_enabled = noise_config.get("enabled", False)

        if self.noise_enabled:
            import numpy as np

            self.noise_type = noise_config.get("type", "uniform")
            self.noise_amplitude = float(noise_config.get("amplitude", 1.0e-12))
            self.noise_seed = int(noise_config.get("seed", np.random.randint(2**20)))

            # Per-species configuration
            electrons_cfg = noise_config.get("electrons", {})
            ions_cfg = noise_config.get("ions", {})

            self.noise_electrons_enabled = electrons_cfg.get("enabled", True)
            self.noise_ions_enabled = ions_cfg.get("enabled", False)

            self.noise_electrons_amplitude = float(electrons_cfg.get("amplitude", self.noise_amplitude))
            self.noise_ions_amplitude = float(ions_cfg.get("amplitude", self.noise_amplitude))

            # Independent seeds for species (offset by 1000 for independence)
            self.noise_electrons_seed = self.noise_seed
            self.noise_ions_seed = self.noise_seed + 1000

        # Pre-compute the 2/3 dealiasing mask (only depends on grid dimensions)
        # Uses standard FFT ordering
        self.mask23 = self._compute_twothirds_mask()

        # Pre-compute Hou-Li filter in Hermite space if enabled (per-species)
        filter_config = driver_config.get("hermite_filter", {})
        self.use_hermite_filter = filter_config.get("enabled", False)
        if self.use_hermite_filter:
            filter_order = filter_config.get("order", 36)
            filter_strength = filter_config.get("strength", 36.0)
            cutoff_fraction = filter_config.get("cutoff_fraction", 2.0 / 3.0)

            self.hermite_filter_electrons = self._compute_houli_hermite_filter(
                Nn_electrons, Nm_electrons, Np_electrons, cutoff_fraction, filter_strength, filter_order
            )
            self.hermite_filter_ions = self._compute_houli_hermite_filter(
                Nn_ions, Nm_ions, Np_ions, cutoff_fraction, filter_strength, filter_order
            )
        else:
            self.hermite_filter_electrons = None
            self.hermite_filter_ions = None

        # Store use_shard_map flag for VectorField-level sharding
        self.use_shard_map = use_shard_map

        # Create per-species HermiteFourierODE instances with species-specific parameters
        # Note: Sharding is now handled at VectorField level, not in ODE classes
        gq_e = grid_quantities_electrons
        self.hermite_fourier_ode_electrons = HermiteFourierODE(
            Nn=Nn_electrons,
            Nm=Nm_electrons,
            Np=Np_electrons,
            Nx=Nx,
            kx_grid=gq_e["kx_grid"],
            ky_grid=gq_e["ky_grid"],
            kz_grid=gq_e["kz_grid"],
            k2_grid=gq_e["k2_grid"],
            Lx=gq_e["Lx"],
            Ly=gq_e["Ly"],
            Lz=gq_e["Lz"],
            col=gq_e["col"],
            sqrt_n_plus=gq_e["sqrt_n_plus"],
            sqrt_n_minus=gq_e["sqrt_n_minus"],
            sqrt_m_plus=gq_e["sqrt_m_plus"],
            sqrt_m_minus=gq_e["sqrt_m_minus"],
            sqrt_p_plus=gq_e["sqrt_p_plus"],
            sqrt_p_minus=gq_e["sqrt_p_minus"],
            mask23=self.mask23,
        )

        gq_i = grid_quantities_ions
        self.hermite_fourier_ode_ions = HermiteFourierODE(
            Nn=Nn_ions,
            Nm=Nm_ions,
            Np=Np_ions,
            Nx=Nx,
            kx_grid=gq_i["kx_grid"],
            ky_grid=gq_i["ky_grid"],
            kz_grid=gq_i["kz_grid"],
            k2_grid=gq_i["k2_grid"],
            Lx=gq_i["Lx"],
            Ly=gq_i["Ly"],
            Lz=gq_i["Lz"],
            col=gq_i["col"],
            sqrt_n_plus=gq_i["sqrt_n_plus"],
            sqrt_n_minus=gq_i["sqrt_n_minus"],
            sqrt_m_plus=gq_i["sqrt_m_plus"],
            sqrt_m_minus=gq_i["sqrt_m_minus"],
            sqrt_p_plus=gq_i["sqrt_p_plus"],
            sqrt_p_minus=gq_i["sqrt_p_minus"],
            mask23=self.mask23,
        )

        # Setup mesh for automatic sharding if enabled
        if self.use_shard_map:
            from jax.experimental import mesh_utils
            from jax.sharding import Mesh

            devices = mesh_utils.create_device_mesh((len(jax.devices()),))
            self.mesh = Mesh(devices, axis_names=("x",))
            print(f"Automatic sharding: Using {len(jax.devices())} device(s) along Nx dimension")
        else:
            self.mesh = None

        # List of state variables that are complex (for unpacking/packing)
        self.complex_state_vars = ["Ck_electrons", "Ck_ions", "Fk"]

    def _compute_twothirds_mask(self) -> Array:
        """
        Compute the 2/3 dealiasing mask in standard FFT ordering.

        Returns a boolean mask that keeps |k| <= N//3 in each dimension to prevent aliasing.
        """

        def standard_modes(N):
            # Returns integer mode numbers in standard FFT ordering
            return jnp.fft.fftfreq(N) * N

        ky = standard_modes(self.Ny)[:, None, None]
        kx = standard_modes(self.Nx)[None, :, None]
        kz = standard_modes(self.Nz)[None, None, :]

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
        term1 = (1.0 / jnp.sqrt(2.0)) * jnp.stack(
            [a0[:, None, None, None] * C100, a1[:, None, None, None] * C010, a2[:, None, None, None] * C001], axis=0
        )

        # Drift velocity contribution: u_i * C0
        # Shape: (3, Ns, Ny, Nx, Nz)
        term2 = jnp.stack(
            [u0[:, None, None, None] * C0, u1[:, None, None, None] * C0, u2[:, None, None, None] * C0], axis=0
        )

        # Current per species: (term1 + term2) * prefactor
        # Shape: (3, Ns, Ny, Nx, Nz)
        J_species = (term1 + term2) * pre[None, :, None, None, None]

        # Sum over species to get total current
        # Shape: (3, Ny, Nx, Nz)
        return jnp.sum(J_species, axis=1)

    def _compute_plasma_current_single_species(
        self, Ck: Array, q: float, alpha: Array, u: Array, Nn: int, Nm: int, Np: int
    ) -> Array:
        """
        Compute the spectral Ampère-Maxwell current for a single species.

        This version operates on a single species distribution function (6D array)
        instead of the multi-species 7D array.

        Args:
            Ck: Hermite-Fourier coefficients for single species, shape (Np, Nm, Nn, Ny, Nx, Nz)
            q: Species charge (scalar)
            alpha: Thermal velocity scaling factors, shape (3,) = (alpha_x, alpha_y, alpha_z)
            u: Drift velocities, shape (3,) = (u_x, u_y, u_z)
            Nn, Nm, Np: Species-specific mode counts (for bounds checking)

        Returns:
            Current density in Fourier space, shape (3, Ny, Nx, Nz) for (Jx, Jy, Jz)
        """
        # Extract specific Hermite modes for current calculation
        # C0 = (p,m,n) = (0,0,0) mode - density
        C0 = Ck[0, 0, 0]  # shape: (Ny, Nx, Nz)

        # C100 = (0,0,1) mode - x-velocity moment (if Nn > 1, else zero)
        C100 = Ck[0, 0, 1] if Nn > 1 else jnp.zeros_like(C0)

        # C010 = (0,1,0) mode - y-velocity moment (if Nm > 1, else zero)
        C010 = Ck[0, 1, 0] if Nm > 1 else jnp.zeros_like(C0)

        # C001 = (1,0,0) mode - z-velocity moment (if Np > 1, else zero)
        C001 = Ck[1, 0, 0] if Np > 1 else jnp.zeros_like(C0)

        # Extract alpha and u components
        a0, a1, a2 = alpha[0], alpha[1], alpha[2]
        u0, u1, u2 = u[0], u[1], u[2]

        # Prefactor: q * alpha_x * alpha_y * alpha_z
        pre = q * a0 * a1 * a2

        # Thermal velocity contribution: (1/sqrt(2)) * alpha_i * C_i
        # Shape: (3, Ny, Nx, Nz)
        term1 = (1.0 / jnp.sqrt(2.0)) * jnp.stack([a0 * C100, a1 * C010, a2 * C001], axis=0)

        # Drift velocity contribution: u_i * C0
        # Shape: (3, Ny, Nx, Nz)
        term2 = jnp.stack([u0 * C0, u1 * C0, u2 * C0], axis=0)

        # Current for this species: (term1 + term2) * prefactor
        # Shape: (3, Ny, Nx, Nz)
        return (term1 + term2) * pre

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

    def _generate_density_noise(self, t: float, Nx: int, alpha: float, amplitude: float, seed: int) -> Array:
        """Generate density noise in Hermite-Fourier space for (0,0,0) mode.

        The (0,0,0) Hermite coefficient represents the density moment:
            n(x) = Ck[0,0,0,0,kx,0] * alpha^3

        This method generates spatial noise, transforms to k-space via FFT,
        and converts to Hermite coefficient units.

        Args:
            t: Current time
            Nx: Number of spatial grid points in x
            alpha: Thermal velocity parameter (for unit conversion)
            amplitude: Noise amplitude
            seed: Base random seed for this species

        Returns:
            Noise array in Hermite coefficient units, shape (Nx,).
            This should be added to Ck[0,0,0,0,:,0].
        """
        import jax.random

        # Time-dependent seed for reproducibility (following lpse2d pattern)
        time_seed = (t / self.dt).astype(int) + seed
        key = jax.random.PRNGKey(time_seed)

        # Generate random phases in [0, 2π)
        phases = 2.0 * jnp.pi * jax.random.uniform(key, (Nx,))

        # Generate noise based on type
        if self.noise_type == "uniform":
            # Uniform amplitude with random phase
            noise_real = amplitude * jnp.exp(1j * phases)
        elif self.noise_type == "normal":
            # Gaussian amplitude with random phase
            key_amp, key_phase = jax.random.split(key)
            amplitudes = amplitude * jax.random.normal(key_amp, (Nx,))
            phases = 2.0 * jnp.pi * jax.random.uniform(key_phase, (Nx,))
            noise_real = amplitudes * jnp.exp(1j * phases)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        # Transform to k-space
        noise_k = jnp.fft.fft(noise_real, norm="forward")

        # Convert from density units to Hermite coefficient units
        # Density: n = Ck[0,0,0] * alpha^3
        # Therefore: Ck[0,0,0] = n / alpha^3
        noise_hermite = noise_k / (alpha**3)

        return noise_hermite

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

    def _pack_y_(self, new_y: dict[str, Array]) -> dict[str, Array]:
        """
        Pack time derivatives back to float64 views.

        Converts complex128 arrays back to float64 views for diffrax.

        Args:
            new_y: Time derivatives dictionary

        Returns:
            Time derivatives dictionary with float64 views
        """
        packed_dy = {}
        for k in new_y.keys():
            packed_dy[k] = new_y[k].view(jnp.float64)

        return packed_dy

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

        # Unpack state dictionary - per-species distribution functions
        Ck_electrons = new_y["Ck_electrons"]
        Ck_ions = new_y["Ck_ions"]
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
        F = jnp.fft.ifftn(Fk * self.mask23, axes=(-3, -2, -1), norm="forward")

        # Transform each species separately
        C_electrons = jnp.fft.ifftn(Ck_electrons * self.mask23, axes=(-3, -2, -1), norm="forward")
        C_ions = jnp.fft.ifftn(Ck_ions * self.mask23, axes=(-3, -2, -1), norm="forward")

        # Compute time derivative of distribution function for each species
        # Returns 6D array per species: (Np, Nm, Nn, Ny, Nx, Nz)

        # Compute electron derivative using electron-specific ODE
        dCk_electrons_dt = self.hermite_fourier_ode_electrons(
            Ck=Ck_electrons,
            C=C_electrons,
            F=F,
            nu=nu,
            D=D,
            alpha=alpha_s[:3],  # Electron thermal velocities
            u=u_s[:3],  # Electron drift velocities
            q=qs[0],  # Electron charge
            Omega_c=Omega_cs[0],  # Electron cyclotron frequency
            closure_n=closure_n,
            closure_m=closure_m,
            closure_p=closure_p,
        )

        # Compute ion derivative using ion-specific ODE
        dCk_ions_dt = self.hermite_fourier_ode_ions(
            Ck=Ck_ions,
            C=C_ions,
            F=F,
            nu=nu,
            D=D,
            alpha=alpha_s[3:],  # Ion thermal velocities
            u=u_s[3:],  # Ion drift velocities
            q=qs[1],  # Ion charge
            Omega_c=Omega_cs[1],  # Ion cyclotron frequency
            closure_n=closure_n,
            closure_m=closure_m,
            closure_p=closure_p,
        )

        # Apply Hou-Li filter in Hermite space if enabled (per-species)
        if self.use_hermite_filter:
            # Apply per-species filters: each has shape (Np, Nm, Nn), broadcast over spatial dimensions
            filter_broadcast_e = self.hermite_filter_electrons[:, :, :, None, None, None]
            filter_broadcast_i = self.hermite_filter_ions[:, :, :, None, None, None]
            dCk_electrons_dt = dCk_electrons_dt * filter_broadcast_e
            dCk_ions_dt = dCk_ions_dt * filter_broadcast_i

        # Apply density noise if enabled
        if self.noise_enabled:
            if self.noise_electrons_enabled:
                # Electron x-thermal velocity for unit conversion
                alpha_e = alpha_s[0]
                # Generate noise for electron density (0,0,0) mode
                noise_e = self._generate_density_noise(
                    t, self.Nx, alpha_e, self.noise_electrons_amplitude, self.noise_electrons_seed
                )
                # Add to (0,0,0,0,:,0) mode - density moment in k-space
                # Note: noise is already in dCk/dt units, no need to multiply by dt here
                dCk_electrons_dt = dCk_electrons_dt.at[0, 0, 0, 0, :, 0].add(noise_e)

            if self.noise_ions_enabled:
                # Ion x-thermal velocity for unit conversion
                alpha_i = alpha_s[3]
                # Generate noise for ion density (0,0,0) mode
                noise_i = self._generate_density_noise(
                    t, self.Nx, alpha_i, self.noise_ions_amplitude, self.noise_ions_seed
                )
                dCk_ions_dt = dCk_ions_dt.at[0, 0, 0, 0, :, 0].add(noise_i)

        # Compute time derivative of magnetic field (Faraday's law)
        dBk_dt = -1j * cross_product(self.nabla, Fk[:3])

        # Get external driver field in Fourier space (if configured)
        # Driver returns complex array of shape (3, Ny, Nx, Nz)
        if self.has_driver:
            driver = jnp.zeros_like(Fk[:3])
            for drv in self.drivers.values():
                driver = driver + drv(self.driver_config, t)
        else:
            driver = jnp.zeros_like(Fk[:3])

        # Compute plasma current and time derivative of electric field (Ampere's law)
        # Compute current for each species and sum (with species-specific mode counts)
        current_electrons = self._compute_plasma_current_single_species(
            Ck_electrons, qs[0], alpha_s[:3], u_s[:3], self.Nn_electrons, self.Nm_electrons, self.Np_electrons
        )
        current_ions = self._compute_plasma_current_single_species(
            Ck_ions, qs[1], alpha_s[3:], u_s[3:], self.Nn_ions, self.Nm_ions, self.Np_ions
        )
        current = current_electrons + current_ions
        dEk_dt = 1j * cross_product(self.nabla, Fk[3:]) - current / Omega_cs[0] + driver

        # Concatenate field derivatives
        # Note: Sponge damping is now handled by the SplitStepDampingSolver integrator
        dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)

        # Create time derivatives dictionary with keys in same order as input y
        # This ensures pytree structure matches exactly
        dy_dt = {}
        dy_dt["Ck_electrons"] = dCk_electrons_dt.view(jnp.float64)
        dy_dt["Ck_ions"] = dCk_ions_dt.view(jnp.float64)
        dy_dt["Fk"] = dFk_dt.view(jnp.float64)

        # Return dictionary with time derivatives (must match state structure exactly)
        return dy_dt
