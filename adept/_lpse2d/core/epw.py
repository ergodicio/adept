import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from adept._lpse2d.core.driver import Driver


def landau_damping_rate(k_sq: Array, wp0: float, vte_sq: float, zero_mask: Array) -> Array:
    """
    Landau damping rate for each k mode (amplitude rate, 1/ps).

    Matches MATLAB line 913:
    gammaLandauEpw = sqrt(pi/8) * (1 + 3/2*k^2*vte^2/wp^2) * wp^4/(k^3*vte^3) * exp(...)

    Module-level so the solver (`SpectralEPWSolver`) and the dissipation
    diagnostic (`helpers.get_default_save_func`) use the *same* rates and can
    never drift apart.
    """
    k_sq_safe = jnp.where(k_sq > 0, k_sq, 1.0)

    damping = (
        jnp.sqrt(np.pi / 8.0)
        * (1.0 + 1.5 * k_sq * vte_sq / wp0**2)
        * wp0**4
        / (k_sq_safe**1.5 * vte_sq**1.5)
        * jnp.exp(-(1.5 + 0.5 * wp0**2 / (k_sq_safe * vte_sq)))
    )

    return damping * zero_mask


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

        # SRS parameters
        self.srs_enabled = cfg["terms"]["epw"]["source"].get("srs", False)
        if self.srs_enabled:
            self.w1 = cfg["units"]["derived"]["w1"]
            self.c = cfg["units"]["derived"]["c"]
            # MATLAB line 2073: srsSourceTerm = 1i * e * wp0/(4*me*w0*w1) .* (1 + dn) .* E0_dot_E1
            self.srs_prefactor = 1j * self.e * self.wp0 / (4.0 * self.me * self.w0 * self.w1)
            # high-k filter for the light fields entering the source product
            # (MATLAB isSuppressHighKSource, lines 637-645): only wavevectors near the
            # light-wave envelope produce physically-realistic SRS
            max_source_k_multiplier = 1.2
            n_min = float(np.min(np.array(self.background_density)))
            max_k1_sq = max_source_k_multiplier**2 * max(1.0 - n_min * self.w0**2 / self.w1**2, 0.0)
            is_outside_max_k1 = self.k_sq * (self.c / self.w1) ** 2 > max_k1_sq
            self.E1_filter = jnp.where(is_outside_max_k1, 0.0, 1.0)[..., None]
            # when the pump is evolved (terms.light.pump_depletion) it is filtered too,
            # exactly as MATLAB's evaluate_E0_dot_E1 (lines 2302-2354) does on the
            # dynamic-laser path and skips on the static path (line 2307-2308)
            self.pump_depletion = cfg["terms"].get("light", {}).get("pump_depletion", False)
            if self.pump_depletion:
                max_k0_sq = max_source_k_multiplier**2 * max(1.0 - n_min, 0.0)
                is_outside_max_k0 = self.k_sq * (self.c / self.w0) ** 2 > max_k0_sq
                self.E0_filter = jnp.where(is_outside_max_k0, 0.0, 1.0)[..., None]

        # Noise parameters. Amplitude default matches MATLAB noiseAmp
        # (m201805_matlabLpse_v11.m:49). The seed is resolved (and written back into
        # the cfg, so MLflow logs it) in helpers.get_derived_quantities; the fallback
        # here only fires if that step was skipped.
        self.noise_enabled = cfg["terms"]["epw"]["source"]["noise"]
        self.noise_amplitude = float(cfg["terms"]["epw"]["source"].get("noise_amplitude", 1e-10))
        cfg_seed = cfg["terms"]["epw"]["source"].get("noise_seed")
        self.noise_seed = int(cfg_seed) if cfg_seed is not None else np.random.randint(2**20)

        # Density gradient
        self.density_gradient_enabled = cfg["terms"]["epw"]["density_gradient"]

        # Landau damping flag (previously ignored -- damping was unconditionally on)
        self.landau_enabled = bool(cfg["terms"]["epw"]["damping"].get("landau", True))
        # HPE (Follett-style particle feedback): the damping rate is read from the
        # state (y["gamma_L"], written by HybridParticleEvolution) instead of the
        # static analytic array
        self.hpe_enabled = bool(cfg["terms"].get("hpe", {}).get("active", False))

        # direct EPW driver (drivers.E2), used by the validation/test configs
        self.driver = Driver(cfg)

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
        return landau_damping_rate(self.k_sq, self.wp0, self.vte_sq, self.zero_mask)

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

    def calc_srs_source(self, E0: Array, E1: Array) -> Array:
        """
        Calculate the SRS source term for the EPW potential.

        Matches MATLAB lines 2052-2078 for isSolveForPotential=true:
          E0_dot_E1 = E0 . conj(E1)  (E1 high-k filtered first, evaluate_E0_dot_E1 lines 2302-2354)
          srsSource = 1i * e * wp0/(4*me*w0*w1) * (1 + dn) * E0_dot_E1
          srsSource -> k-space

        The pump is static/prescribed here, so only E1 is filtered (in MATLAB the E0
        filter is skipped on the static-laser path, line 2308).

        Args:
            E0: Pump field (shape: nx, ny, 2)
            E1: Raman field (shape: nx, ny, 2)

        Returns:
            SRS source term in k-space
        """
        E1_filtered = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.E1_filter, axes=(0, 1))
        if self.pump_depletion:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.E0_filter, axes=(0, 1))
        E0_dot_E1 = E0[..., 0] * jnp.conj(E1_filtered[..., 0]) + E0[..., 1] * jnp.conj(E1_filtered[..., 1])

        # (1 + backgroundDensityPerturbation) = n / n_envelope
        source = self.srs_prefactor * self.background_density / self.envelope_density * E0_dot_E1

        return jnp.fft.fft2(source)

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

        # Suppress high-wavenumber modes (MATLAB epwNoise: phi_noise(isHighWavenumberMode) = 0)
        noise = noise * self.low_pass_filter

        # Zero out k=0
        noise = noise * self.zero_mask

        return noise

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
        7. Apply density gradient to E fields (line 2081-2082)
        8. Apply absorbing boundaries to E fields (line 2088-2100)
        9. Convert E fields back to phi_k (line 2103) ← FILTER applied here too
        10. Add TPD source (line 2109)

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
        if self.hpe_enabled:
            gamma_landau = y["gamma_L"]
        elif self.landau_enabled:
            gamma_landau = self.calc_landau_damping_rate()
        else:
            gamma_landau = 0.0
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

        srs_source = None
        if self.srs_enabled:
            # MATLAB lines 2052-2078
            srs_source = self.calc_srs_source(E0, y["E1"])

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
