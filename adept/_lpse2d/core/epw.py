import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from adept._lpse2d.core.driver import Driver


class SpectralPotential:
    def __init__(self, cfg) -> None:
        self.background_density = cfg["grid"]["background_density"]
        self.vte_sq = cfg["units"]["derived"]["vte_sq"]
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]
        self.k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2
        self.wp0 = cfg["units"]["derived"]["wp0"]
        self.e = cfg["units"]["derived"]["e"]
        self.me = cfg["units"]["derived"]["me"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.envelope_density = cfg["units"]["envelope density"]
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]
        self.boundary_envelope = cfg["grid"]["absorbing_boundaries"]
        self.dt = cfg["grid"]["dt"]
        self.cfg = cfg
        # self.amp_key, self.phase_key = jax.random.split(jax.random.PRNGKey(np.random.randint(2**20)), 2)
        self.phase_seed = np.random.randint(2**20)
        self.low_pass_filter = cfg["grid"]["low_pass_filter_grid"]
        self.zero_mask = cfg["grid"]["zero_mask"]
        self.nx = cfg["grid"]["nx"]
        self.ny = cfg["grid"]["ny"]
        self.driver = Driver(cfg)
        self.tpd_const = 1j * self.e / (8 * self.wp0 * self.me)
        self.nu_coll = cfg["units"]["derived"]["nu_coll"]
        self.nx_pad = self.nx * 2  # + (self.nx + 1) // 2
        self.ny_pad = self.ny * 2  # + (self.ny + 1) // 2
        self.pad_x = self._compute_pad_width(self.nx_pad, self.nx)
        self.pad_y = self._compute_pad_width(self.ny_pad, self.ny)
        self.trunc_x_start = self.pad_x[0]
        self.trunc_x_end = self.trunc_x_start + self.nx
        self.trunc_y_start = self.pad_y[0]
        self.trunc_y_end = self.trunc_y_start + self.ny
        self.pad_norm = (self.nx_pad / self.nx) * (self.ny_pad / self.ny)

        if cfg["terms"]["epw"]["source"]["srs"]:
            self.w1 = cfg["units"]["derived"]["w1"]
            max_source_k_multiplier = 1.2
            max_k0 = max_source_k_multiplier * np.sqrt(1 - cfg["density"]["min"])
            max_k1 = max_source_k_multiplier * np.sqrt(1 - cfg["density"]["min"] * (self.w0**2) / (self.w1**2))
            is_outside_max_k0 = self.k_sq * (1 / self.w0**2) > max_k0**2
            is_outside_max_k1 = self.k_sq * (1 / self.w1**2) > max_k1**2
            self.E0_filter = jnp.where(is_outside_max_k0, 0.0, 1.0)[..., None]
            self.E1_filter = jnp.where(is_outside_max_k1, 0.0, 1.0)[..., None]

            self.srs_const = self.e * self.wp0 / (4 * self.me * self.w0 * self.w1)

    def calc_fields_from_phi(self, phi: Array) -> tuple[Array, Array]:
        """
        Calculates ex(x, y) and ey(x, y) from phi.

        Args:
            phi (Array): phi(x, y)

        Returns:
            A Tuple containing ex(x, y) and ey(x, y)
        """

        phi_k = jnp.fft.fft2(phi)
        return self.calc_fields_from_phi_k(phi_k)

    def calc_fields_from_phi_k(self, phi_k: Array) -> tuple[Array, Array]:
        """
        Calculates ex(x, y) and ey(x, y) from phi_k.

        Args:
            phi (Array): phi(x, y)

        Returns:
            A Tuple containing ex(x, y) and ey(x, y)
        """

        # Filter is applied before calling this function
        phi_k = phi_k * self.zero_mask
        ex_k = -1j * self.kx[:, None] * phi_k
        ey_k = -1j * self.ky[None, :] * phi_k
        return jnp.fft.ifft2(ex_k), jnp.fft.ifft2(ey_k)

    def calc_phi_from_fields(self, ex: Array, ey: Array) -> Array:
        """
        calculates phi from ex and ey

        Args:
            ex (Array): ex(x, y)
            ey (Array): ey(x, y)

        Returns:
            Array: phi(x, y)

        """

        phi_k = self.calc_phi_k_from_fields(ex, ey)
        # phi_k is already filtered by calc_phi_k_from_fields
        phi = jnp.fft.ifft2(phi_k)

        return phi

    def calc_phi_k_from_fields(self, ex: Array, ey: Array) -> Array:
        """
        calculates phi_k from ex and ey

        Args:
            ex (Array): ex(x, y)
            ey (Array): ey(x, y)

        Returns:
            Array: phi_k(x, y)
        """

        ex_k = jnp.fft.fft2(ex)
        ey_k = jnp.fft.fft2(ey)
        divE_k = 1j * (self.kx[:, None] * ex_k + self.ky[None, :] * ey_k) * self.low_pass_filter

        phi_k = divE_k * self.one_over_ksq
        return phi_k * self.zero_mask

    @staticmethod
    def _compute_pad_width(target: int, original: int) -> tuple[int, int]:
        total = max(target - original, 0)
        before = total // 2
        after = total - before
        return before, after

    def _fft_pad(self, arr_k: Array) -> Array:
        if self.nx_pad == self.nx and self.ny_pad == self.ny:
            return arr_k
        arr_shift = jnp.fft.fftshift(arr_k)
        padded_shift = jnp.pad(
            arr_shift,
            ((self.pad_x[0], self.pad_x[1]), (self.pad_y[0], self.pad_y[1])),
        )
        return jnp.fft.ifftshift(padded_shift)

    def _fft_truncate(self, arr_k_pad: Array) -> Array:
        if self.nx_pad == self.nx and self.ny_pad == self.ny:
            return arr_k_pad
        arr_shift = jnp.fft.fftshift(arr_k_pad)
        truncated_shift = arr_shift[self.trunc_x_start : self.trunc_x_end, self.trunc_y_start : self.trunc_y_end]
        return jnp.fft.ifftshift(truncated_shift)

    def _dealias_fft_product(self, first: Array, second: Array) -> Array:
        first_k = jnp.fft.fft2(first)
        second_k = jnp.fft.fft2(second)
        first_k_pad = self._fft_pad(first_k)
        second_k_pad = self._fft_pad(second_k)
        first_pad = jnp.fft.ifft2(first_k_pad)
        second_pad = jnp.fft.ifft2(second_k_pad)
        prod_pad = first_pad * second_pad
        prod_k_pad = jnp.fft.fft2(prod_pad) * self.pad_norm
        return self._fft_truncate(prod_k_pad)

    def tpd(self, t: float, phi_k: Array, ey: Array, args: dict) -> Array:
        """
        Calculates the two plasmon decay term

        Args:
            t (float): time
            y (Array): phi(x, y)
            args (Dict): dictionary containing E0

        Returns:
            Array: dphi(x, y)

        """
        E0 = args["E0"]
        E0_y = E0[..., 1]
        filtered_E0y = jnp.fft.ifft2(jnp.fft.fft2(E0_y) * self.low_pass_filter)
        tpd1 = self._dealias_fft_product(filtered_E0y, jnp.conj(ey))

        divE_true = jnp.fft.ifft2(self.k_sq * phi_k)
        E0_divE_k = self._dealias_fft_product(filtered_E0y, jnp.conj(divE_true))
        tpd2 = 1j * self.ky[None, :] * self.one_over_ksq * E0_divE_k

        total_tpd = self.tpd_const * jnp.exp(-1j * (self.w0 - 2 * self.wp0) * t) * (tpd1 + tpd2)

        # Source term will be filtered when added to phi_k
        total_tpd *= self.zero_mask

        return total_tpd

    def eval_E0_dot_E1(self, t, y, args):
        E0 = args["E0"]
        E1 = y["E1"]

        # filter E0 and E1
        E0_filtered = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.E0_filter, axes=(0, 1))
        E1_filtered = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.E1_filter, axes=(0, 1))
        E0_x_source, E0_y_source = E0_filtered[..., 0], E0_filtered[..., 1]
        E1_x_source, E1_y_source = E1_filtered[..., 0], E1_filtered[..., 1]

        return E0_x_source * jnp.conj(E1_x_source) + E0_y_source * jnp.conj(E1_y_source)

    def srs(self, t: float, y, args: dict) -> Array:
        E0_dot_E1 = self.eval_E0_dot_E1(t, y, args)
        return jnp.fft.fft2(1j * self.srs_const * self.background_density / self.envelope_density * E0_dot_E1)

    def get_noise(self, t):
        random_amps = 1.0e-12  # jax.random.uniform(self.amp_key, (self.nx, self.ny))
        phase_key = jax.random.PRNGKey((t / self.dt).astype(int) + self.phase_seed)
        random_phases = 2 * np.pi * jax.random.uniform(phase_key, (self.nx, self.ny))
        return random_amps * jnp.exp(1j * random_phases) * self.zero_mask

    def landau_damping(self, phi_k: Array):
        gammaLandauEpw = (
            jnp.sqrt(np.pi / 8)
            * (1.0 + 1.5 * self.k_sq * (self.vte_sq / self.wp0**2))
            * self.wp0**4
            * self.one_over_ksq**1.5
            / self.vte_sq**1.5
            * jnp.exp(-(1.5 + 0.5 * self.wp0**2 * self.one_over_ksq / self.vte_sq))
        ) * self.zero_mask

        # Filter is applied after this function returns
        return phi_k * jnp.exp(-(gammaLandauEpw + self.nu_coll) * self.dt) * self.zero_mask

    def __call__(self, t: float, y: dict[str, Array], args: dict) -> Array:
        phi_k = y["epw"]
        E0 = y["E0"]
        background_density = self.background_density

        # linear propagation - thermal dispersion
        phi_k = phi_k * jnp.exp(-1j * 1.5 * self.vte_sq / self.wp0 * self.k_sq * self.dt)
        # Landau damping
        phi_k = self.landau_damping(phi_k)
        # Apply filter AFTER thermal and damping (matching MATLAB line 1976)
        phi_k = phi_k * self.low_pass_filter

        if self.cfg["terms"]["epw"]["source"]["noise"]:
            phi_k += self.dt * self.get_noise(t)

        ex, ey = self.calc_fields_from_phi_k(phi_k)

        if self.cfg["terms"]["epw"]["source"]["tpd"]:
            tpd_term = self.tpd(t, phi_k, ey, args={"E0": E0})

        if self.cfg["terms"]["epw"]["source"]["srs"]:
            srs_term = self.srs(t, y, args={"E0": E0})

        # density gradient
        if self.cfg["terms"]["epw"]["density_gradient"]:
            background_density_perturbation = background_density / self.envelope_density - 1.0
            phase = jnp.exp(-1j * self.wp0 / 2.0 * background_density_perturbation * self.dt)
            ex = ex * phase
            ey = ey * phase

        ex = ex * self.boundary_envelope
        ey = ey * self.boundary_envelope
        phi_k = self.calc_phi_k_from_fields(ex, ey)

        # tpd
        if self.cfg["terms"]["epw"]["source"]["tpd"]:
            phi_k += self.dt * tpd_term

        if self.cfg["terms"]["epw"]["source"]["srs"]:
            phi_k += self.dt * srs_term

        # add hyperviscosity in k space for phi
        if self.cfg["terms"]["epw"].get("hyperviscosity", {}).get("coeff", 0) > 0:
            if self.cfg["terms"]["epw"]["hyperviscosity"]["order"] % 2 != 0:
                raise ValueError("Hyperviscosity order must be even")
            # hypervisc_coeff * dt < coeff
            # hypervisc_coeff = coeff / (kmax**order * dt)
            coeff = self.cfg["terms"]["epw"]["hyperviscosity"]["coeff"]
            order = self.cfg["terms"]["epw"]["hyperviscosity"]["order"]
            # kmax = kmax * lowpass_filter
            kmax = self.cfg["grid"]["low_pass_filter"] * jnp.sqrt(jnp.max(self.k_sq))
            hypervisc_coeff = coeff / kmax**order / self.dt

            phi_k = phi_k * jnp.exp(-hypervisc_coeff * (self.k_sq ** (order / 2.0)) * self.dt)

        return phi_k


class SpectralEPWSolver:
    """
    Spectral solver for electrostatic plasma waves in k-space.

    Matches MATLAB's spectralEpwUpdate() function (lines 1966-2118).

    State variable: phi_k (electrostatic potential in k-space)
    - MATLAB convention: uses fftshift, so DC is in center
    - JAX convention: uses fftfreq, so DC is at [0,0]

    Key differences from original implementation:
    1. Filter applied at exactly 2 points per timestep
    2. Clear separation between operations (no combined expressions)
    3. Explicit comments matching MATLAB line numbers
    """

    def __init__(self, cfg: dict):
        """
        Initialize the spectral EPW solver.

        Args:
            cfg: Configuration dictionary with grid, units, and physics parameters
        """
        # Grid parameters
        self.nx = cfg["grid"]["nx"]
        self.ny = cfg["grid"]["ny"]
        self.dx = cfg["grid"]["dx"]
        self.dy = cfg["grid"]["dy"]
        self.dt = cfg["grid"]["dt"]

        # K-space grid (JAX uses fftfreq, not fftshift)
        self.kx = cfg["grid"]["kx"]  # Shape: (nx,)
        self.ky = cfg["grid"]["ky"]  # Shape: (ny,)

        # 2D k-space grids
        # Note: MATLAB uses [KX, KY] = meshgrid(kx, ky) with fftshift
        # JAX uses fftfreq which is already in correct order
        self.k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2

        # Avoid division by zero at k=0
        self.one_over_k_sq = jnp.where(self.k_sq > 0, 1.0 / self.k_sq, 0.0)
        self.zero_mask = jnp.where(self.k_sq > 0, 1.0, 0.0)

        # Physics parameters
        self.wp0 = cfg["units"]["derived"]["wp0"]  # Reference plasma frequency
        self.w0 = cfg["units"]["derived"]["w0"]  # Laser frequency
        self.vte_sq = cfg["units"]["derived"]["vte_sq"]  # Thermal velocity squared
        self.e = cfg["units"]["derived"]["e"]  # Elementary charge (normalized)
        self.me = cfg["units"]["derived"]["me"]  # Electron mass (normalized)
        self.nu_coll = cfg["units"]["derived"].get("nu_coll", 0.0)  # Collisional damping

        # Density profile
        self.envelope_density = cfg["units"]["envelope density"]
        self.background_density = cfg["grid"]["background_density"]

        # Boundaries
        self.boundary_envelope = cfg["grid"]["absorbing_boundaries"]

        # Low-pass filter
        # This should be binary (0 or 1) unless taper_fraction > 0
        self.low_pass_filter = cfg["grid"]["low_pass_filter_grid"]

        # TPD parameters
        self.tpd_enabled = cfg["terms"]["epw"]["source"]["tpd"]
        if self.tpd_enabled:
            self.tpd_prefactor = 1j * self.e / (8.0 * self.wp0 * self.me)

        # Noise parameters
        self.noise_enabled = cfg["terms"]["epw"]["source"]["noise"]
        self.noise_amplitude = 1e-12
        self.noise_seed = np.random.randint(2**20)

        # Density gradient
        self.density_gradient_enabled = cfg["terms"]["epw"]["density_gradient"]

        # SRS parameters
        self.srs_enabled = cfg["terms"]["epw"]["source"]["srs"]
        if self.srs_enabled:
            self.w1 = cfg["units"]["derived"]["w1"]
            self.srs_prefactor = 1j * self.e * self.wp0 / (4.0 * self.me * self.w0 * self.w1)

            # Optional: k-space filters for source terms (following MATLAB isSuppressHighKSource)
            max_source_k_multiplier = 1.2
            max_k0 = max_source_k_multiplier * np.sqrt(1 - cfg["density"]["min"])
            max_k1 = max_source_k_multiplier * np.sqrt(1 - cfg["density"]["min"] * (self.w0**2) / (self.w1**2))
            is_outside_max_k0 = self.k_sq * (1 / self.w0**2) > max_k0**2
            is_outside_max_k1 = self.k_sq * (1 / self.w1**2) > max_k1**2
            self.E0_filter = jnp.where(is_outside_max_k0, 0.0, 1.0)
            self.E1_filter = jnp.where(is_outside_max_k1, 0.0, 1.0)

        # Store config for reference
        self.cfg = cfg

    def calc_landau_damping_rate(self) -> Array:
        """
        Calculate Landau damping rate for each k mode.

        Matches MATLAB line 913:
        gammaLandauEpw = sqrt(pi/8) * (1 + 3/2*k^2*vte^2/wp^2) * wp^4/(k^3*vte^3) * exp(...)

        Returns:
            Landau damping rate array (shape: nx, ny)
        """
        # Avoid issues at k=0
        k_sq_safe = jnp.where(self.k_sq > 0, self.k_sq, 1.0)

        damping = (
            jnp.sqrt(np.pi / 8.0)
            * (1.0 + 1.5 * self.k_sq * self.vte_sq / self.wp0**2)
            * self.wp0**4
            / (k_sq_safe**1.5 * self.vte_sq**1.5)
            * jnp.exp(-(1.5 + 0.5 * self.wp0**2 / (k_sq_safe * self.vte_sq)))
        )

        # Zero out k=0 mode
        damping = damping * self.zero_mask

        return damping

    def phi_k_to_e_fields(self, phi_k: Array) -> tuple[Array, Array]:
        """
        Convert phi_k to electric field components in real space.

        Matches MATLAB's calculateFieldsFromDivE() function.
        When isSolveForPotential=true, divE is actually phi_k.

        MATLAB (lines 2458-2502):
          phi = divE  (already in k-space)
          Ex_k = -1i * KX .* phi
          Ey_k = -1i * KY .* phi
          Ex = ifftn(ifftshift(Ex_k))
          Ey = ifftn(ifftshift(Ey_k))

        JAX equivalent (no fftshift needed with fftfreq):
          ex_k = -1j * kx * phi_k
          ey_k = -1j * ky * phi_k
          ex = ifft2(ex_k)
          ey = ifft2(ey_k)

        Args:
            phi_k: Potential in k-space (shape: nx, ny)

        Returns:
            Tuple of (ex, ey) in real space
        """
        # Gradient in k-space: E = -∇φ → E_k = -i*k*φ_k
        ex_k = -1j * self.kx[:, None] * phi_k
        ey_k = -1j * self.ky[None, :] * phi_k

        # Transform to real space
        ex = jnp.fft.ifft2(ex_k)
        ey = jnp.fft.ifft2(ey_k)

        return ex, ey

    def e_fields_to_phi_k(self, ex: Array, ey: Array) -> Array:
        """
        Convert electric field components to phi_k.

        Matches MATLAB's convertFieldsToDivE() function.

        MATLAB (lines 2506-2540):
          Ex_k = fftshift(fftn(Ex))
          Ey_k = fftshift(fftn(Ey))
          divE_k = 1i * (KX.*Ex_k + KY.*Ey_k)
          if isSuppressHighWavenumberModes
              divE_k(isHighWavenumberMode) = 0
          phi_k = divE_k ./ K_sq

        Args:
            ex: Electric field x-component in real space
            ey: Electric field y-component in real space

        Returns:
            phi_k in k-space
        """
        # Transform to k-space
        ex_k = jnp.fft.fft2(ex)
        ey_k = jnp.fft.fft2(ey)

        # Divergence in k-space: ∇·E → i*k·E_k
        div_e_k = 1j * (self.kx[:, None] * ex_k + self.ky[None, :] * ey_k)

        # Apply filter (MATLAB line 2523)
        div_e_k = div_e_k * self.low_pass_filter

        # Poisson equation: ∇²φ = -p → -k²φ = ∇·E → φ = -∇·E/k²
        phi_k = div_e_k * self.one_over_k_sq

        # Zero out k=0 mode (MATLAB line 2529)
        phi_k = phi_k * self.zero_mask

        return phi_k

    def calc_tpd_source(self, t: float, phi_k: Array, ey: Array, E0_y: Array) -> Array:
        """
        Calculate Two Plasmon Decay source term.

        Matches MATLAB lines 1996-2049 for isSolveForPotential=true.

        TPD source has two components:
          TPD1 = FFT(E0_y * conj(Ey))
          TPD2 = 1i * KY/K_sq * FFT(E0_y * conj(divE_true))
          where divE_true = IFFT(K_sq * phi_k)

        Args:
            t: Current time
            phi_k: Potential in k-space
            ey: Electric field y-component in real space
            E0_y: Laser field y-component in real space

        Returns:
            TPD source term in k-space
        """
        # Component 1: E0 * conj(Ey)
        # MATLAB line 2011-2012
        product1 = E0_y * jnp.conj(ey)
        tpd1 = jnp.fft.fft2(product1)

        # Component 2: E0 * conj(divE_true)
        # MATLAB line 2014-2018
        # divE_true is the actual charge density (4π times Poisson)
        div_e_true = jnp.fft.ifft2(self.k_sq * phi_k)
        product2 = E0_y * jnp.conj(div_e_true)
        product2_k = jnp.fft.fft2(product2)
        tpd2 = 1j * self.ky[None, :] * self.one_over_k_sq * product2_k

        # Combine with prefactor
        # MATLAB line 2024
        phase = jnp.exp(-1j * (self.w0 - 2.0 * self.wp0) * t)
        source = self.tpd_prefactor * phase * (tpd1 + tpd2)

        # Apply filter to source (MATLAB line 2032-2033)
        source = source * self.low_pass_filter

        # Zero out k=0 (MATLAB line 2035)
        source = source * self.zero_mask

        return source

    def get_noise(self, t: float) -> Array:
        """
        Generate random noise for plasma waves.

        Args:
            t: Current time

        Returns:
            Random noise in k-space
        """
        # Use time-dependent seed for reproducibility
        seed = (t / self.dt).astype(int) + self.noise_seed
        key = jax.random.PRNGKey(seed)

        # Random phases
        phases = 2.0 * np.pi * jax.random.uniform(key, (self.nx, self.ny))

        # Uniform amplitude with random phase
        noise = self.noise_amplitude * jnp.exp(1j * phases)

        # Zero out k=0
        noise = noise * self.zero_mask

        return noise

    def eval_E0_dot_E1(self, E0: Array, E1: Array) -> Array:
        """
        Calculate scalar product of laser and Raman fields.

        Matches MATLAB's evaluate_E0_dot_E1() function (lines 2302-2352).

        This calculates E0 · conj(E1) with optional filtering to suppress
        high-k modes that are not physically realistic for SRS.

        Args:
            E0: Laser field (shape: nx, ny, 2) where [..., 0] is x, [..., 1] is y
            E1: Raman field (shape: nx, ny, 2) where [..., 0] is x, [..., 1] is y

        Returns:
            E0_dot_E1 in real space (shape: nx, ny)
        """
        # Extract components
        E0_x, E0_y = E0[..., 0], E0[..., 1]
        E1_x, E1_y = E1[..., 0], E1[..., 1]

        # Apply k-space filters if configured (MATLAB isSuppressHighKSource)
        # This removes high-k structure from fields before taking the product
        if hasattr(self, "E0_filter") and hasattr(self, "E1_filter"):
            # Filter E0
            E0_x_k = jnp.fft.fft2(E0_x)
            E0_y_k = jnp.fft.fft2(E0_y)
            E0_x_filtered = jnp.fft.ifft2(E0_x_k * self.E0_filter)
            E0_y_filtered = jnp.fft.ifft2(E0_y_k * self.E0_filter)

            # Filter E1
            E1_x_k = jnp.fft.fft2(E1_x)
            E1_y_k = jnp.fft.fft2(E1_y)
            E1_x_filtered = jnp.fft.ifft2(E1_x_k * self.E1_filter)
            E1_y_filtered = jnp.fft.ifft2(E1_y_k * self.E1_filter)
        else:
            # No filtering
            E0_x_filtered = E0_x
            E0_y_filtered = E0_y
            E1_x_filtered = E1_x
            E1_y_filtered = E1_y

        # Calculate dot product: E0 · conj(E1)
        # MATLAB line 2344 or 2351
        E0_dot_E1 = E0_x_filtered * jnp.conj(E1_x_filtered) + E0_y_filtered * jnp.conj(E1_y_filtered)

        return E0_dot_E1

    def calc_srs_source(self, E0_dot_E1: Array) -> Array:
        """
        Calculate Stimulated Raman Scattering source term.

        Matches MATLAB lines 2052-2078 in spectralEpwUpdate() for isSolveForPotential=true.

        For the potential formulation:
          srsSourceTerm = 1i * e * wp0 / (4*me*w0*w1) * (1 + n_pert) * E0_dot_E1

        Then transformed to k-space.

        Args:
            E0_dot_E1: Scalar product of laser and Raman fields in real space

        Returns:
            SRS source term in k-space
        """
        # MATLAB line 2073:
        # srsSourceTerm = 1i * e * wp0 /(4*me*w0*w1) .* (1 + backgroundDensityPerturbation) .* E0_dot_E1(ixc,iyc);

        # Density perturbation: n/n0 - 1
        density_perturbation = self.background_density / self.envelope_density - 1.0

        # Build source in real space
        source_real = self.srs_prefactor * (1.0 + density_perturbation) * E0_dot_E1

        # Transform to k-space (MATLAB line 2077)
        source_k = jnp.fft.fft2(source_real)

        # Apply filter (matching TPD treatment)
        source_k = source_k * self.low_pass_filter

        # Zero out k=0
        source_k = source_k * self.zero_mask

        return source_k

    def __call__(self, t: float, y, args) -> Array:
        """
        Advance EPW by one timestep using spectral method.

        This matches MATLAB's spectralEpwUpdate() lines 1966-2118.

        Order of operations (matching MATLAB exactly):
        1. Apply thermal dispersion in k-space (line 1975)
        2. Apply Landau damping in k-space (line 1981)
        3. FILTER (line 1976) ← Applied AFTER thermal/damping
        4. Add noise (line 1988)
        5. Calculate E fields from phi_k (line 1992)
        6. Calculate TPD source in k-space (lines 2011-2024)
        6b. Calculate SRS source in k-space (lines 2052-2078)
        7. Apply density gradient to E fields (line 2081-2082)
        8. Apply absorbing boundaries to E fields (line 2088-2100)
        9. Convert E fields back to phi_k (line 2103) ← FILTER applied here too
        10. Add TPD source (line 2109)
        11. Add SRS source (line 2113)

        Args:
            t: Current time
            y: Dictionary containing:
                - "epw": Current EPW potential in k-space
                - "E0": Laser field (shape: nx, ny, 2) where E0[..., 1] is y-component
                - "E1": Raman field (shape: nx, ny, 2), optional for SRS
            args: Additional arguments (not used currently)

        Returns:
            Updated phi_k after one timestep
        """
        phi_k = y["epw"]
        E0 = y["E0"]
        background_density = self.background_density

        # ========================================================================
        # STEP 1-2: Thermal dispersion and Landau damping
        # ========================================================================
        # MATLAB line 1975: divE_k = divE_k .* exp(-1i*3/2*vte_sq/wp0 .* K_sq * DT)
        thermal_phase = jnp.exp(-1j * 1.5 * self.vte_sq / self.wp0 * self.k_sq * self.dt)
        phi_k = phi_k * thermal_phase

        # MATLAB line 1981: divE = divE .* exp(-(gammaLandau + nu_coll) * DT)
        gamma_landau = self.calc_landau_damping_rate()
        damping_factor = jnp.exp(-(gamma_landau + self.nu_coll) * self.dt)
        phi_k = phi_k * damping_factor

        # ========================================================================
        # STEP 3: Apply filter ONCE after thermal + damping
        # ========================================================================
        # MATLAB line 1976: divE_k(isHighWavenumberMode) = 0
        phi_k = phi_k * self.low_pass_filter

        # ========================================================================
        # STEP 4: Add noise
        # ========================================================================
        if self.noise_enabled:
            # MATLAB line 1988: divE = divE + epwNoise * DT
            noise = self.get_noise(t)
            phi_k = phi_k + self.dt * noise

        # ========================================================================
        # STEP 5: Calculate electric fields
        # ========================================================================
        # MATLAB line 1992: [Ex, Ey] = calculateFieldsFromDivE(...)
        ex, ey = self.phi_k_to_e_fields(phi_k)

        # ========================================================================
        # STEP 6: Calculate TPD source (in k-space, before applying density gradient)
        # ========================================================================
        tpd_source = None
        if self.tpd_enabled:
            # MATLAB lines 1996-2035
            E0_y = E0[..., 1]  # y-component of laser field
            tpd_source = self.calc_tpd_source(t, phi_k, ey, E0_y)

        # ========================================================================
        # STEP 6b: Calculate SRS source (in k-space, before applying density gradient)
        # ========================================================================
        srs_source = None
        if self.srs_enabled:
            # MATLAB lines 2052-2078
            E1 = y.get("E1")  # Raman field
            if E1 is not None:
                E0_dot_E1 = self.eval_E0_dot_E1(E0, E1)
                srs_source = self.calc_srs_source(E0_dot_E1)

        # ========================================================================
        # STEP 7: Apply density gradient to E fields (in REAL space)
        # ========================================================================
        if self.density_gradient_enabled:
            # MATLAB line 2081-2082:
            # Ex = Ex .* exp(-1i * wp0/2 * (n/n0 - 1) * DT)
            # Ey = Ey .* exp(-1i * wp0/2 * (n/n0 - 1) * DT)
            density_perturbation = background_density / self.envelope_density - 1.0
            density_phase = jnp.exp(-1j * self.wp0 / 2.0 * density_perturbation * self.dt)
            ex = ex * density_phase
            ey = ey * density_phase

        # ========================================================================
        # STEP 8: Apply absorbing boundaries to E fields (in REAL space)
        # ========================================================================
        # MATLAB line 2088-2100:
        # Ex = Ex .* exp(-DT * boundaryDampingRate)
        # Ey = Ey .* exp(-DT * boundaryDampingRate)
        ex = ex * self.boundary_envelope
        ey = ey * self.boundary_envelope

        # ========================================================================
        # STEP 9: Convert E fields back to phi_k
        # ========================================================================
        # MATLAB line 2103: divE = convertFieldsToDivE(Ex, Ey, ...)
        # This function applies filter at line 2523
        phi_k = self.e_fields_to_phi_k(ex, ey)

        # ========================================================================
        # STEP 10: Add TPD source
        # ========================================================================
        if self.tpd_enabled and tpd_source is not None:
            # MATLAB line 2109: divE = divE + tpdSourceTerm * DT
            phi_k = phi_k + self.dt * tpd_source

        # ========================================================================
        # STEP 11: Add SRS source
        # ========================================================================
        if self.srs_enabled and srs_source is not None:
            # MATLAB line 2113: divE = divE + srsSourceTerm * DT
            phi_k = phi_k + self.dt * srs_source

        return phi_k
