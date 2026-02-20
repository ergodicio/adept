"""
Nonlinear vector field for the Lawson-RK4 exponential integrator.

This vector field computes only the nonlinear terms of the Vlasov-Maxwell system:
  - E-field acceleration (convolution: E · ∇_v C)
  - B-field Lorentz force (convolution: (vxB) · ∇_v C)
  - Plasma current (J → dE/dt in Ampère's law)
  - External driver
  - Density noise

Linear terms (free-streaming, Maxwell curls, collision, diffusion) are handled
by the exact exponential operators in the LawsonRK4Solver.
"""

from typing import Optional

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from adept._spectrax1d.driver import Driver
from adept._spectrax1d.hermite_fourier_ode import HermiteFourierODE


class NonlinearVectorField(eqx.Module):
    """
    Nonlinear-only RHS for the Lawson-RK4 exponential integrator.

    Reuses HermiteFourierODE instances for the Lorentz force convolution.
    """

    # Static ions flag
    static_ions: bool

    # Grid dimension (only Nx needed after init for noise generation)
    Nx: int

    # Electron Hermite mode counts
    Nn_electrons: int
    Nm_electrons: int
    Np_electrons: int

    # Ion Hermite mode counts
    Nn_ions: int
    Nm_ions: int
    Np_ions: int

    # Time step
    dt: float

    # Driver configuration
    drivers: dict
    driver_config: dict | None
    has_driver: bool

    # Noise configuration
    noise_enabled: bool
    noise_type: str | None
    noise_electrons_enabled: bool | None
    noise_ions_enabled: bool | None
    noise_electrons_amplitude: float | None
    noise_ions_amplitude: float | None
    noise_electrons_seed: int | None
    noise_ions_seed: int | None

    # 2/3 dealiasing mask
    mask23: Array

    # Hermite filter
    use_hermite_filter: bool
    hermite_filter_electrons: Array | None
    hermite_filter_ions: Array | None

    # Per-species HermiteFourierODE instances
    hermite_fourier_ode_electrons: HermiteFourierODE
    hermite_fourier_ode_ions: HermiteFourierODE

    # Complex state variable names
    complex_state_vars: list

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
        static_ions: bool = False,
    ):
        """Initialize with per-species mode counts and grid quantities."""
        # Note: Ny, Nz, Ns only used locally in __init__, not stored as fields
        self.static_ions = static_ions
        self.Nx = Nx
        self.Nn_electrons = Nn_electrons
        self.Nm_electrons = Nm_electrons
        self.Np_electrons = Np_electrons
        self.Nn_ions = Nn_ions
        self.Nm_ions = Nm_ions
        self.Np_ions = Np_ions
        self.dt = dt

        # Initialize external field drivers
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

        # Density noise configuration
        noise_config = driver_config.get("density_noise", {})
        self.noise_enabled = noise_config.get("enabled", False)

        if self.noise_enabled:
            import numpy as np

            self.noise_type = noise_config.get("type", "uniform")
            # noise_amplitude and noise_seed only used locally to compute per-species values
            noise_amplitude = float(noise_config.get("amplitude", 1.0e-12))
            noise_seed = int(noise_config.get("seed", np.random.randint(2**20)))

            electrons_cfg = noise_config.get("electrons", {})
            ions_cfg = noise_config.get("ions", {})

            self.noise_electrons_enabled = electrons_cfg.get("enabled", True)
            self.noise_ions_enabled = ions_cfg.get("enabled", False)
            self.noise_electrons_amplitude = float(electrons_cfg.get("amplitude", noise_amplitude))
            self.noise_ions_amplitude = float(ions_cfg.get("amplitude", noise_amplitude))
            self.noise_electrons_seed = noise_seed
            self.noise_ions_seed = noise_seed + 1000
        else:
            self.noise_type = None
            self.noise_electrons_enabled = None
            self.noise_ions_enabled = None
            self.noise_electrons_amplitude = None
            self.noise_ions_amplitude = None
            self.noise_electrons_seed = None
            self.noise_ions_seed = None

        # 2/3 dealiasing mask
        self.mask23 = self._compute_twothirds_mask(Nx, Ny, Nz)

        # Hermite filter
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

        # Create per-species HermiteFourierODE instances (reused for Lorentz force)
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

        self.complex_state_vars = ["Ck_electrons", "Ck_ions", "Fk"]

    @staticmethod
    def _compute_twothirds_mask(Nx: int, Ny: int, Nz: int) -> Array:
        """Compute 2/3 dealiasing mask in standard FFT ordering."""

        def standard_modes(N):
            return jnp.fft.fftfreq(N) * N

        ky = standard_modes(Ny)[:, None, None]
        kx = standard_modes(Nx)[None, :, None]
        kz = standard_modes(Nz)[None, None, :]

        cy = Ny // 3
        cx = Nx // 3
        cz = Nz // 3

        return (jnp.abs(ky) <= cy) & (jnp.abs(kx) <= cx) & (jnp.abs(kz) <= cz)

    def _compute_houli_hermite_filter(
        self, Nn: int, Nm: int, Np: int, cutoff_fraction: float, strength: float, order: int
    ) -> Array:
        """Compute Hou-Li exponential filter in Hermite space."""
        n = jnp.arange(Nn)[None, None, :]
        m = jnp.arange(Nm)[None, :, None]
        p = jnp.arange(Np)[:, None, None]

        h_max = jnp.sqrt((Nn - 1) ** 2 + (Nm - 1) ** 2 + (Np - 1) ** 2)
        h_cutoff = cutoff_fraction * h_max
        h = jnp.sqrt(n**2 + m**2 + p**2)

        filter_arg = jnp.where(h > h_cutoff, strength * ((h / h_cutoff) ** order), 0.0)
        return jnp.exp(-filter_arg)

    def _compute_plasma_current_single_species(
        self, Ck: Array, q: float, alpha: Array, u: Array, Nn: int, Nm: int, Np: int
    ) -> Array:
        """Compute spectral Ampère-Maxwell current for a single species."""
        C0 = Ck[0, 0, 0]
        C100 = Ck[0, 0, 1] if Nn > 1 else jnp.zeros_like(C0)
        C010 = Ck[0, 1, 0] if Nm > 1 else jnp.zeros_like(C0)
        C001 = Ck[1, 0, 0] if Np > 1 else jnp.zeros_like(C0)

        a0, a1, a2 = alpha[0], alpha[1], alpha[2]
        u0, u1, u2 = u[0], u[1], u[2]
        pre = q * a0 * a1 * a2

        term1 = (1.0 / jnp.sqrt(2.0)) * jnp.stack([a0 * C100, a1 * C010, a2 * C001], axis=0)
        term2 = jnp.stack([u0 * C0, u1 * C0, u2 * C0], axis=0)

        return (term1 + term2) * pre

    def _generate_density_noise(self, t: float, Nx: int, alpha: float, amplitude: float, seed: int) -> Array:
        """Generate density noise in Hermite-Fourier space for (0,0,0) mode."""
        import jax.random

        time_seed = (t / self.dt).astype(int) + seed
        key = jax.random.PRNGKey(time_seed)
        phases = 2.0 * jnp.pi * jax.random.uniform(key, (Nx,))

        if self.noise_type == "uniform":
            noise_real = amplitude * jnp.exp(1j * phases)
        elif self.noise_type == "normal":
            key_amp, key_phase = jax.random.split(key)
            amplitudes = amplitude * jax.random.normal(key_amp, (Nx,))
            phases = 2.0 * jnp.pi * jax.random.uniform(key_phase, (Nx,))
            noise_real = amplitudes * jnp.exp(1j * phases)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        noise_k = jnp.fft.fft(noise_real, norm="forward")
        return noise_k / (alpha**3)

    def _unpack_y_(self, y: dict[str, Array]) -> dict[str, Array]:
        """Unpack state from float64 views to complex128."""
        new_y = {}
        for k in y.keys():
            if k in self.complex_state_vars:
                new_y[k] = y[k].view(jnp.complex128)
            else:
                new_y[k] = y[k].view(jnp.float64)
        return new_y

    def __call__(self, t: float, y: dict, args: dict) -> dict:
        """
        Compute the nonlinear RHS of the Vlasov-Maxwell system.

        Only computes: Lorentz force, plasma current, driver, noise.
        Does NOT compute: free-streaming, Maxwell curls, collision, diffusion.

        Args:
            t: Current time
            y: State dictionary (float64 views)
            args: Physical parameters

        Returns:
            Dictionary with nonlinear time derivatives (float64 views)
        """
        new_y = self._unpack_y_(y)

        Ck_electrons = new_y["Ck_electrons"]
        Ck_ions = new_y["Ck_ions"]
        Fk = new_y["Fk"]

        # Unpack physical parameters
        qs = args["qs"]
        Omega_cs = args["Omega_cs"]
        alpha_s = args["alpha_s"]
        u_s = args["u_s"]

        # Dealiasing + IFFT to real space for convolution terms
        F = jnp.fft.ifftn(Fk * self.mask23, axes=(-3, -2, -1), norm="forward")
        C_electrons = jnp.fft.ifftn(Ck_electrons * self.mask23, axes=(-3, -2, -1), norm="forward")
        C_ions = jnp.fft.ifftn(Ck_ions * self.mask23, axes=(-3, -2, -1), norm="forward")

        # Compute Lorentz force (E-field + B-field coupling) for each species
        dCk_electrons_dt = self.hermite_fourier_ode_electrons._compute_lorentz_rhs(
            C=C_electrons,
            F=F,
            alpha=alpha_s[:3],
            u=u_s[:3],
            q=qs[0],
            Omega_c=Omega_cs[0],
        )

        if self.static_ions:
            dCk_ions_dt = jnp.zeros_like(Ck_ions)
        else:
            dCk_ions_dt = self.hermite_fourier_ode_ions._compute_lorentz_rhs(
                C=C_ions,
                F=F,
                alpha=alpha_s[3:],
                u=u_s[3:],
                q=qs[1],
                Omega_c=Omega_cs[1],
            )

        # Apply Hermite filter
        if self.use_hermite_filter:
            filter_broadcast_e = self.hermite_filter_electrons[:, :, :, None, None, None]
            dCk_electrons_dt = dCk_electrons_dt * filter_broadcast_e
            if not self.static_ions:
                filter_broadcast_i = self.hermite_filter_ions[:, :, :, None, None, None]
                dCk_ions_dt = dCk_ions_dt * filter_broadcast_i

        # Apply density noise
        if self.noise_enabled:
            if self.noise_electrons_enabled:
                alpha_e = alpha_s[0]
                noise_e = self._generate_density_noise(
                    t, self.Nx, alpha_e, self.noise_electrons_amplitude, self.noise_electrons_seed
                )
                dCk_electrons_dt = dCk_electrons_dt.at[0, 0, 0, 0, :, 0].add(noise_e)

            if self.noise_ions_enabled and not self.static_ions:
                alpha_i = alpha_s[3]
                noise_i = self._generate_density_noise(
                    t, self.Nx, alpha_i, self.noise_ions_amplitude, self.noise_ions_seed
                )
                dCk_ions_dt = dCk_ions_dt.at[0, 0, 0, 0, :, 0].add(noise_i)

        # EM field nonlinear terms: plasma current + driver (no curls — handled by exponential)
        # Get external driver
        if self.has_driver:
            driver = jnp.zeros_like(Fk[:3])
            for drv in self.drivers.values():
                driver = driver + drv(self.driver_config, t)
        else:
            driver = jnp.zeros_like(Fk[:3])

        # Plasma current for Ampère's law
        current_electrons = self._compute_plasma_current_single_species(
            Ck_electrons, qs[0], alpha_s[:3], u_s[:3], self.Nn_electrons, self.Nm_electrons, self.Np_electrons
        )
        if self.static_ions:
            current = current_electrons
        else:
            current_ions = self._compute_plasma_current_single_species(
                Ck_ions, qs[1], alpha_s[3:], u_s[3:], self.Nn_ions, self.Nm_ions, self.Np_ions
            )
            current = current_electrons + current_ions

        # dE/dt contribution: -J/Omega_c + driver (no curl term)
        dEk_dt = -current / Omega_cs[0] + driver

        # dB/dt = 0 from nonlinear terms (Faraday curls are linear, handled by exponential)
        dBk_dt = jnp.zeros_like(Fk[3:])

        dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)

        return {
            "Ck_electrons": dCk_electrons_dt.view(jnp.float64),
            "Ck_ions": dCk_ions_dt.view(jnp.float64),
            "Fk": dFk_dt.view(jnp.float64),
        }
