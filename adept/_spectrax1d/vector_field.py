"""Full vector field for the Spectrax Vlasov-Maxwell system (DoPri8 / explicit integrator path)."""

import jax
import jax.random
from jax import Array
from jax import numpy as jnp

from adept._spectrax1d.driver import Driver
from adept._spectrax1d.hermite_fourier_ode import HermiteFourierODE


class SpectraxVectorField:
    """
    Full RHS for the Hermite-Fourier Vlasov-Maxwell system.

    Computes ALL terms: free-streaming, Maxwell curls, Lorentz force,
    collisions, diffusion, plasma current, and external driver.

    Used with explicit integrators (DoPri8, Tsit5, etc.). The Lawson-RK4
    path uses NonlinearVectorField instead (splits linear/nonlinear).

    State keys: "Ck_electrons", "Ck_ions", "Fk" (float64 views of complex128)
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
        static_ions: bool = False,
    ):
        self.static_ions = static_ions
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
        self.dt = dt

        self.nabla = grid_quantities_electrons["nabla"]

        # External field drivers
        self.drivers = {}
        for driver_key in ["ex", "ey", "ez"]:
            if driver_config.get(driver_key):
                self.drivers[driver_key] = Driver(xax, Nx, Ny, Nz, driver_key=driver_key)
        self.has_driver = bool(self.drivers)
        self.driver_config = driver_config if self.has_driver else None

        # Density noise
        noise_config = driver_config.get("density_noise", {})
        self.noise_enabled = noise_config.get("enabled", False)
        if self.noise_enabled:
            import numpy as np

            self.noise_electrons_enabled = noise_config.get("electrons", {}).get("enabled", True)
            self.noise_ions_enabled = noise_config.get("ions", {}).get("enabled", False)
            self.noise_electrons_amplitude = float(noise_config.get("electrons", {}).get("amplitude", noise_config.get("amplitude", 1e-12)))
            self.noise_ions_amplitude = float(noise_config.get("ions", {}).get("amplitude", noise_config.get("amplitude", 1e-12)))
            seed = int(noise_config.get("seed", np.random.randint(2**20)))
            self.noise_electrons_seed = seed
            self.noise_ions_seed = seed + 1000

        # 2/3 dealiasing mask
        def _modes(N):
            return jnp.fft.fftfreq(N) * N

        ky = _modes(Ny)[:, None, None]
        kx = _modes(Nx)[None, :, None]
        kz = _modes(Nz)[None, None, :]
        self.mask23 = (jnp.abs(ky) <= Ny // 3) & (jnp.abs(kx) <= Nx // 3) & (jnp.abs(kz) <= Nz // 3)

        # Hou-Li Hermite filter
        filter_cfg = driver_config.get("hermite_filter", {})
        self.use_hermite_filter = filter_cfg.get("enabled", False)
        if self.use_hermite_filter:
            strength = float(filter_cfg.get("strength", 36.0))
            order = int(filter_cfg.get("order", 36))
            cutoff_fraction = float(filter_cfg.get("cutoff_fraction", 1.0))
            self.hermite_filter_electrons = self._houli_filter(Nn_electrons, Nm_electrons, Np_electrons, strength, order, cutoff_fraction)
            self.hermite_filter_ions = self._houli_filter(Nn_ions, Nm_ions, Np_ions, strength, order, cutoff_fraction)
        else:
            self.hermite_filter_electrons = None
            self.hermite_filter_ions = None

        # HermiteFourierODE instances (one per species)
        gq_e = grid_quantities_electrons
        self.ode_e = HermiteFourierODE(
            Nn=Nn_electrons, Nm=Nm_electrons, Np=Np_electrons, Nx=Nx,
            kx_grid=gq_e["kx_grid"], ky_grid=gq_e["ky_grid"], kz_grid=gq_e["kz_grid"],
            k2_grid=gq_e["k2_grid"], Lx=gq_e["Lx"], Ly=gq_e["Ly"], Lz=gq_e["Lz"],
            col=gq_e["col"], sqrt_n_plus=gq_e["sqrt_n_plus"], sqrt_n_minus=gq_e["sqrt_n_minus"],
            sqrt_m_plus=gq_e["sqrt_m_plus"], sqrt_m_minus=gq_e["sqrt_m_minus"],
            sqrt_p_plus=gq_e["sqrt_p_plus"], sqrt_p_minus=gq_e["sqrt_p_minus"],
            mask23=self.mask23,
        )
        gq_i = grid_quantities_ions
        self.ode_i = HermiteFourierODE(
            Nn=Nn_ions, Nm=Nm_ions, Np=Np_ions, Nx=Nx,
            kx_grid=gq_i["kx_grid"], ky_grid=gq_i["ky_grid"], kz_grid=gq_i["kz_grid"],
            k2_grid=gq_i["k2_grid"], Lx=gq_i["Lx"], Ly=gq_i["Ly"], Lz=gq_i["Lz"],
            col=gq_i["col"], sqrt_n_plus=gq_i["sqrt_n_plus"], sqrt_n_minus=gq_i["sqrt_n_minus"],
            sqrt_m_plus=gq_i["sqrt_m_plus"], sqrt_m_minus=gq_i["sqrt_m_minus"],
            sqrt_p_plus=gq_i["sqrt_p_plus"], sqrt_p_minus=gq_i["sqrt_p_minus"],
            mask23=self.mask23,
        )

        self.complex_state_vars = ["Ck_electrons", "Ck_ions", "Fk"]

    @staticmethod
    def _houli_filter(Nn: int, Nm: int, Np: int, strength: float, order: int, cutoff_fraction: float = 1.0) -> Array:
        n = jnp.arange(Nn, dtype=jnp.float64)[None, None, :]
        m = jnp.arange(Nm, dtype=jnp.float64)[None, :, None]
        p = jnp.arange(Np, dtype=jnp.float64)[:, None, None]
        h_max = jnp.sqrt((Nn - 1) ** 2 + (Nm - 1) ** 2 + (Np - 1) ** 2)
        h_cutoff = cutoff_fraction * h_max
        h = jnp.sqrt(n**2 + m**2 + p**2)
        # Always scale by h_max so the exponent reaches `strength` at the highest mode.
        # cutoff_fraction < 1 zeros out modes below the threshold (they see no damping).
        filter_arg = jnp.where(h > h_cutoff, strength * (h / h_max) ** order, 0.0)
        return jnp.exp(-filter_arg)

    def _current(self, Ck: Array, q: float, alpha: Array, u: Array, Nn: int, Nm: int, Np: int) -> Array:
        C0 = Ck[0, 0, 0]
        C100 = Ck[0, 0, 1] if Nn > 1 else jnp.zeros_like(C0)
        C010 = Ck[0, 1, 0] if Nm > 1 else jnp.zeros_like(C0)
        C001 = Ck[1, 0, 0] if Np > 1 else jnp.zeros_like(C0)
        pre = q * alpha[0] * alpha[1] * alpha[2]
        term1 = (1.0 / jnp.sqrt(2.0)) * jnp.stack([alpha[0] * C100, alpha[1] * C010, alpha[2] * C001], axis=0)
        term2 = jnp.stack([u[0] * C0, u[1] * C0, u[2] * C0], axis=0)
        return (term1 + term2) * pre

    def _unpack(self, y: dict) -> dict:
        return {k: y[k].view(jnp.complex128) if k in self.complex_state_vars else y[k] for k in y}

    def _generate_density_noise(self, t: float, Nx: int, alpha: float, amplitude: float, seed: int) -> Array:
        time_seed = (t / self.dt).astype(int) + seed
        key = jax.random.PRNGKey(time_seed)
        phases = 2.0 * jnp.pi * jax.random.uniform(key, (Nx,))
        noise_k = jnp.fft.fft(amplitude * jnp.exp(1j * phases), norm="forward")
        return noise_k / (alpha**3)

    def __call__(self, t: float, y: dict, args: dict) -> dict:
        state = self._unpack(y)
        Ck_e = state["Ck_electrons"]
        Ck_i = state["Ck_ions"]
        Fk = state["Fk"]

        qs = args["qs"]
        nu = args["nu"]
        D = args.get("D", 0.0)
        Omega_ce_tau = args["Omega_ce_tau"]
        mi_me = args["mi_me"]
        alpha_s = args["alpha_s"]
        u_s = args["u_s"]

        F = jnp.fft.ifftn(Fk * self.mask23, axes=(-3, -2, -1), norm="forward")
        C_e = jnp.fft.ifftn(Ck_e * self.mask23, axes=(-3, -2, -1), norm="forward")
        C_i = jnp.fft.ifftn(Ck_i * self.mask23, axes=(-3, -2, -1), norm="forward")

        # Full electron RHS (free-streaming + Lorentz + collisions + diffusion)
        dCk_e_dt = self.ode_e(
            Ck=Ck_e, C=C_e, F=F, nu=nu, D=D,
            alpha=alpha_s[:3], u=u_s[:3], q=qs[0],
            Omega_ce_tau=Omega_ce_tau, m=1.0,
        )

        if self.static_ions:
            dCk_i_dt = jnp.zeros_like(Ck_i)
        else:
            dCk_i_dt = self.ode_i(
                Ck=Ck_i, C=C_i, F=F, nu=nu, D=D,
                alpha=alpha_s[3:], u=u_s[3:], q=qs[1],
                Omega_ce_tau=Omega_ce_tau, m=mi_me,
            )

        if self.use_hermite_filter:
            dCk_e_dt = dCk_e_dt * self.hermite_filter_electrons[:, :, :, None, None, None]
            if not self.static_ions:
                dCk_i_dt = dCk_i_dt * self.hermite_filter_ions[:, :, :, None, None, None]

        if self.noise_enabled:
            if self.noise_electrons_enabled:
                noise_e = self._generate_density_noise(t, self.Nx, alpha_s[0], self.noise_electrons_amplitude, self.noise_electrons_seed)
                dCk_e_dt = dCk_e_dt.at[0, 0, 0, 0, :, 0].add(noise_e)
            if self.noise_ions_enabled and not self.static_ions:
                noise_i = self._generate_density_noise(t, self.Nx, alpha_s[3], self.noise_ions_amplitude, self.noise_ions_seed)
                dCk_i_dt = dCk_i_dt.at[0, 0, 0, 0, :, 0].add(noise_i)

        # Maxwell: curls + plasma current + driver
        dBk_dt = -1j * jnp.cross(self.nabla, Fk[:3], axis=0)

        J_e = self._current(Ck_e, qs[0], alpha_s[:3], u_s[:3], self.Nn_electrons, self.Nm_electrons, self.Np_electrons)
        J = J_e if self.static_ions else J_e + self._current(Ck_i, qs[1], alpha_s[3:], u_s[3:], self.Nn_ions, self.Nm_ions, self.Np_ions)

        driver = jnp.zeros_like(Fk[:3])
        if self.has_driver:
            for drv in self.drivers.values():
                driver = driver + drv(self.driver_config, t)

        dEk_dt = 1j * jnp.cross(self.nabla, Fk[3:], axis=0) - J / Omega_ce_tau + driver

        dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)

        return {
            "Ck_electrons": dCk_e_dt.view(jnp.float64),
            "Ck_ions": dCk_i_dt.view(jnp.float64),
            "Fk": dFk_dt.view(jnp.float64),
        }
