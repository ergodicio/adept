import jax
import numpy as np
from jax import Array
from jax import numpy as jnp


class SpectralEPWSolver1D:
    """
    1D Spectral solver for electrostatic plasma waves in k-space.

    Matches MATLAB's spectralEpwUpdate() function (lines 1966-2118) adapted for 1D.

    State variable: phi_k (electrostatic potential in k-space)
    - Uses JAX fftfreq convention (DC at [0])

    Key differences from 2D implementation:
    1. Single spatial dimension (x only)
    2. Uses fft/ifft instead of fft2/ifft2
    3. Scalar electric field (only Ey component in MATLAB)
    4. Simpler filtering and k-space operations
    """

    def __init__(self, cfg: dict):
        """
        Initialize the 1D spectral EPW solver.

        Args:
            cfg: Configuration dictionary with grid, units, and physics parameters
        """
        # Grid parameters
        self.nx = cfg["grid"]["nx"]
        self.dx = cfg["grid"]["dx"]
        self.dt = cfg["grid"]["dt"]

        # K-space grid
        self.kx = cfg["grid"]["kx"]  # Shape: (nx,)
        self.k_sq = self.kx**2

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
        self.low_pass_filter = cfg["grid"]["low_pass_filter_grid"]

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
            Landau damping rate array (shape: nx)
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

    def phi_k_to_e_field(self, phi_k: Array) -> Array:
        """
        Convert phi_k to electric field in real space.

        In 1D: E = -dφ/dx → E_k = -i*k*φ_k

        Args:
            phi_k: Potential in k-space (shape: nx)

        Returns:
            E in real space (shape: nx)
        """
        # Gradient in k-space: E = -∇φ → E_k = -i*k*φ_k
        e_k = -1j * self.kx * phi_k

        # Transform to real space
        e = jnp.fft.ifft(e_k)

        return e

    def e_field_to_phi_k(self, e: Array) -> Array:
        """
        Convert electric field to phi_k.

        In 1D: ∇·E = dE/dx → (∇·E)_k = i*k*E_k

        Args:
            e: Electric field in real space (shape: nx)

        Returns:
            phi_k in k-space (shape: nx)
        """
        # Transform to k-space
        e_k = jnp.fft.fft(e)

        # Divergence in k-space: ∇·E → i*k·E_k
        div_e_k = 1j * self.kx * e_k

        # Apply filter (MATLAB line 2523)
        div_e_k = div_e_k * self.low_pass_filter

        # Poisson equation: ∇²φ = -ρ → -k²φ = ∇·E → φ = -∇·E/k²
        phi_k = div_e_k * self.one_over_k_sq

        # Zero out k=0 mode (MATLAB line 2529)
        phi_k = phi_k * self.zero_mask

        return phi_k

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
        phases = 2.0 * np.pi * jax.random.uniform(key, (self.nx,))

        # Uniform amplitude with random phase
        noise = self.noise_amplitude * jnp.exp(1j * phases)

        # Zero out k=0
        noise = noise * self.zero_mask

        return noise

    def eval_E0_dot_E1(self, E0: Array, E1: Array) -> Array:
        """
        Calculate scalar product of laser and Raman fields (1D version).

        Matches MATLAB's evaluate_E0_dot_E1() function (lines 2302-2352) for 1D.

        In 1D, only the y-component exists, so this is just E0_y * conj(E1_y).

        Args:
            E0: Laser field in real space (shape: nx)
            E1: Raman field in real space (shape: nx)

        Returns:
            E0_dot_E1 in real space (shape: nx)
        """
        # Apply k-space filters if configured (MATLAB isSuppressHighKSource)
        if hasattr(self, "E0_filter") and hasattr(self, "E1_filter"):
            # Filter E0
            E0_k = jnp.fft.fft(E0)
            E0_filtered = jnp.fft.ifft(E0_k * self.E0_filter)

            # Filter E1
            E1_k = jnp.fft.fft(E1)
            E1_filtered = jnp.fft.ifft(E1_k * self.E1_filter)
        else:
            # No filtering
            E0_filtered = E0
            E1_filtered = E1

        # Calculate product: E0 * conj(E1)
        # MATLAB line 2349 for nDims=1
        E0_dot_E1 = E0_filtered * jnp.conj(E1_filtered)

        return E0_dot_E1

    def calc_srs_source(self, E0_dot_E1: Array) -> Array:
        """
        Calculate Stimulated Raman Scattering source term (1D version).

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
        source_k = jnp.fft.fft(source_real)

        # Apply filter (matching TPD treatment)
        source_k = source_k * self.low_pass_filter

        # Zero out k=0
        source_k = source_k * self.zero_mask

        return source_k

    def __call__(self, t: float, y, args) -> Array:
        """
        Advance EPW by one timestep using spectral method (1D version).

        This matches MATLAB's spectralEpwUpdate() lines 1966-2118, adapted for 1D.

        Order of operations (matching MATLAB exactly):
        1. Apply thermal dispersion in k-space (line 1975)
        2. Apply Landau damping in k-space (line 1981)
        3. FILTER (line 1976) ← Applied AFTER thermal/damping
        4. Add noise (line 1988)
        5. Calculate E field from phi_k (line 1992)
        6b. Calculate SRS source in k-space (lines 2052-2078)
        7. Apply density gradient to E field (line 2081-2082)
        8. Apply absorbing boundaries to E field (line 2088-2100)
        9. Convert E field back to phi_k (line 2103) ← FILTER applied here too
        11. Add SRS source (line 2113)

        Args:
            t: Current time
            y: Dictionary containing:
                - "epw": Current EPW potential in k-space (shape: nx)
                - "E0": Laser field in real space (shape: nx)
                - "E1": Raman field in real space (shape: nx), optional for SRS
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
        # STEP 5: Calculate electric field
        # ========================================================================
        # MATLAB line 1992: [Ex, Ey] = calculateFieldsFromDivE(...)
        # In 1D, we only have one component
        e = self.phi_k_to_e_field(phi_k)

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
        # STEP 7: Apply density gradient to E field (in REAL space)
        # ========================================================================
        if self.density_gradient_enabled:
            # MATLAB line 2081-2082:
            # Ey = Ey .* exp(-1i * wp0/2 * (n/n0 - 1) * DT)
            density_perturbation = background_density / self.envelope_density - 1.0
            density_phase = jnp.exp(-1j * self.wp0 / 2.0 * density_perturbation * self.dt)
            e = e * density_phase

        # ========================================================================
        # STEP 8: Apply absorbing boundaries to E field (in REAL space)
        # ========================================================================
        # MATLAB line 2088-2100:
        # Ey = Ey .* exp(-DT * boundaryDampingRate)
        e = e * self.boundary_envelope

        # ========================================================================
        # STEP 9: Convert E field back to phi_k
        # ========================================================================
        # MATLAB line 2103: divE = convertFieldsToDivE(...)
        # This function applies filter at line 2523
        phi_k = self.e_field_to_phi_k(e)

        # ========================================================================
        # STEP 11: Add SRS source
        # ========================================================================
        if self.srs_enabled and srs_source is not None:
            # MATLAB line 2113: divE = divE + srsSourceTerm * DT
            phi_k = phi_k + self.dt * srs_source

        return phi_k
