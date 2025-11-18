import numpy as np
from jax import Array
from jax import numpy as jnp


class LaserSolver1D:
    """
    1D solver for laser field evolution (both pump E0 and seed E1).

    Matches MATLAB's evalLaserFieldUpdate() function (lines 1506-1770) for nDims=1.

    Solves the wave equation for electromagnetic fields with ponderomotive coupling to plasma waves.

    For pump (w0):
        ∂E0/∂t = i*c²/(2*w0) * ∂²E0/∂x² + i*w0/2 * (1 - wp0²/w0² * (1 + δn + n_elf)) * E0
                  - i*e/(4*w1*me) * ∇²φ * E1  (SRS coupling)

    For seed (w1):
        ∂E1/∂t = i*c²/(2*w1) * ∂²E1/∂x² + i*w1/2 * (1 - wp0²/w1² * (1 + δn + n_elf)) * E1
                  - i*e/(4*w0*me) * ∇²φ* * E0  (SRS coupling)
    """

    def __init__(self, cfg: dict):
        """
        Initialize the 1D laser solver.

        Args:
            cfg: Configuration dictionary with grid, units, and physics parameters
        """
        # Grid parameters
        self.nx = cfg["grid"]["nx"]
        self.dx = cfg["grid"]["dx"]
        self.dt = cfg["grid"]["dt"]

        # K-space grid
        self.kx = cfg["grid"]["kx"]
        self.k_sq = self.kx**2

        # Physics parameters
        self.c = cfg["units"]["derived"]["c"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.wp0 = cfg["units"]["derived"]["wp0"]
        self.w1 = cfg["units"]["derived"]["w1"]
        self.e = cfg["units"]["derived"]["e"]
        self.me = cfg["units"]["derived"]["me"]

        # Density profile
        self.envelope_density = cfg["units"]["envelope density"]
        self.background_density = cfg["grid"]["background_density"]

        # SRS coupling
        self.srs_enabled = cfg["terms"]["epw"]["source"]["srs"]
        if self.srs_enabled:
            # Coupling coefficients
            # MATLAB lines 1646-1648 and 1686-1688
            self.pump_coupling_coeff = -1j * self.e / (4.0 * self.w1 * self.me)  # For E0 update
            self.seed_coupling_coeff = -1j * self.e / (4.0 * self.w0 * self.me)  # For E1 update

        # Pump depletion
        self.pump_depletion_enabled = cfg["terms"].get("pump_depletion", False)

        # Store config
        self.cfg = cfg

    def calc_pump_update(self, t: float, y: dict) -> Array:
        """
        Calculate time derivative of pump field E0.

        Matches MATLAB lines 1621-1622 for nDims=1, plus SRS coupling (lines 1646-1648).

        Args:
            t: Current time
            y: State dictionary containing E0, background_density, and optionally E1, epw

        Returns:
            dE0/dt (shape: nx)
        """
        E0 = y["E0"]
        background_density = self.background_density

        # Density perturbation: δn = n/n0 - 1
        # MATLAB: backgroundDensityPerturbation
        density_perturbation = background_density / self.envelope_density - 1.0

        # IAW density (if present, otherwise 0)
        n_elf = y.get("n_elf", jnp.zeros_like(E0))

        # ========================================================================
        # Linear propagation: diffraction + dispersion
        # ========================================================================
        # MATLAB line 1621-1622:
        # k_E0.y = 1i * c^2/(2*w0) * (E0.y(ixp) - 2 * E0.y(ixc) + E0.y(ixm))/dx^2
        #          + 1i * w0/2 * (1 - wp0^2/w0^2 * (1 + backgroundDensityPerturbation + Nelf)) .* E0.y

        # Second derivative using finite differences
        # E0[i+1] - 2*E0[i] + E0[i-1]
        E0_shifted_p = jnp.roll(E0, -1)  # E0[i+1]
        E0_shifted_m = jnp.roll(E0, 1)  # E0[i-1]
        d2E0_dx2 = (E0_shifted_p - 2.0 * E0 + E0_shifted_m) / self.dx**2

        # Diffraction term
        diffraction = 1j * self.c**2 / (2.0 * self.w0) * d2E0_dx2

        # Dispersion term
        dispersion_factor = 1.0 - self.wp0**2 / self.w0**2 * (1.0 + density_perturbation + n_elf)
        dispersion = 1j * self.w0 / 2.0 * dispersion_factor * E0

        # Combine linear terms
        k_E0 = diffraction + dispersion

        # ========================================================================
        # SRS coupling term (pump depletion)
        # ========================================================================
        if self.srs_enabled and self.pump_depletion_enabled:
            E1 = y.get("E1")
            phi_k = y.get("epw")

            if E1 is not None and phi_k is not None:
                # MATLAB lines 1644-1648 for isSolveForPotential:
                # k_E0.y = k_E0.y - 1i * e/(4*w1 * me) .* laplacianPhi .* E1.y

                # Calculate laplacian of phi: ∇²φ = IFFT(-k² * φ_k)
                laplacian_phi = jnp.fft.ifft(-self.k_sq * phi_k)

                # Add coupling term
                k_E0 = k_E0 + self.pump_coupling_coeff * laplacian_phi * E1

        return k_E0

    def calc_seed_update(self, t: float, y: dict) -> Array:
        """
        Calculate time derivative of seed field E1.

        Matches MATLAB lines 1661-1662 for nDims=1, plus SRS coupling (lines 1686-1688).

        Args:
            t: Current time
            y: State dictionary containing E1, background_density, and E0, epw for SRS

        Returns:
            dE1/dt (shape: nx)
        """
        E1 = y["E1"]
        background_density = self.background_density

        # Density perturbation: δn = n/n0 - 1
        density_perturbation = background_density / self.envelope_density - 1.0

        # IAW density (if present, otherwise 0)
        n_elf = y.get("n_elf", jnp.zeros_like(E1))

        # ========================================================================
        # Linear propagation: diffraction + dispersion
        # ========================================================================
        # MATLAB line 1661-1662:
        # k_E1.y = 1i * c^2/(2*w1) * (E1.y(ixp) - 2 * E1.y(ixc) + E1.y(ixm))/dx^2
        #          + 1i * w1/2 * (1 - wp0^2/w1^2 * (1 + backgroundDensityPerturbation + Nelf)) .* E1.y

        # Second derivative using finite differences
        E1_shifted_p = jnp.roll(E1, -1)  # E1[i+1]
        E1_shifted_m = jnp.roll(E1, 1)  # E1[i-1]
        d2E1_dx2 = (E1_shifted_p - 2.0 * E1 + E1_shifted_m) / self.dx**2

        # Diffraction term
        diffraction = 1j * self.c**2 / (2.0 * self.w1) * d2E1_dx2

        # Dispersion term
        dispersion_factor = 1.0 - self.wp0**2 / self.w1**2 * (1.0 + density_perturbation + n_elf)
        dispersion = 1j * self.w1 / 2.0 * dispersion_factor * E1

        # Combine linear terms
        k_E1 = diffraction + dispersion

        # ========================================================================
        # SRS coupling term (seed amplification)
        # ========================================================================
        if self.srs_enabled:
            E0 = y.get("E0")
            phi_k = y.get("epw")

            if E0 is not None and phi_k is not None:
                # MATLAB lines 1684-1688 for isSolveForPotential:
                # k_E1.y = k_E1.y - 1i * e/(4*w0 * me) .* conj(laplacianPhi) .* E0.y

                # Calculate laplacian of phi: ∇²φ = IFFT(-k² * φ_k)
                laplacian_phi = jnp.fft.ifft(-self.k_sq * phi_k)

                # Add coupling term (note the conjugate)
                k_E1 = k_E1 + self.seed_coupling_coeff * jnp.conj(laplacian_phi) * E0

        return k_E1

    def __call__(self, t: float, y: dict, field: str) -> Array:
        """
        Calculate time derivative for specified laser field.

        Args:
            t: Current time
            y: State dictionary
            field: Either "E0" for pump or "E1" for seed

        Returns:
            Time derivative of specified field
        """
        if field == "E0":
            return self.calc_pump_update(t, y)
        elif field == "E1":
            return self.calc_seed_update(t, y)
        else:
            raise ValueError(f"Unknown field: {field}. Must be 'E0' or 'E1'")
