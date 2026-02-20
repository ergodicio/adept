"""Base ADEPTModule wrapper for spectrax-based Hermite-Fourier Vlasov-Maxwell solver."""

import os

import pint
import xarray as xr
from diffrax import ConstantStepSize, ODETerm, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import Array
from jax import numpy as jnp

from adept._base_ import ADEPTModule
from adept._spectrax1d.plotting import (
    plot_field_mode_amplitudes,
    plot_fields_lineouts,
    plot_fields_spacetime,
    plot_hermite_modes_at_k,
    plot_moments_lineouts,
    plot_moments_spacetime,
)
from adept._spectrax1d.storage import get_save_quantities, store_scalars, store_species_distribution_timeseries


def fft_index(k_val: int, N: int) -> int:
    """Convert signed k-value to standard FFT array index.

    Args:
        k_val: Signed wavenumber (e.g., -2, -1, 0, 1, 2)
        N: Grid size

    Returns:
        Array index in standard FFT ordering

    Examples:
        fft_index(0, 8) = 0
        fft_index(1, 8) = 1
        fft_index(-1, 8) = 7
        fft_index(-4, 8) = 4
    """
    return k_val if k_val >= 0 else N + k_val


class BaseSpectrax1D(ADEPTModule):
    """
    ADEPTModule wrapper for spectrax-based Landau damping solver.

    This module provides a thin wrapper around the spectrax library's simulation()
    function, enabling MLflow logging and ADEPT configuration management while
    preserving the native spectrax workflow.

    Reference pattern: adept/_vlasov1d/modules.py
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialize the module with configuration.

        Args:
            cfg: Configuration dictionary from YAML file
        """
        super().__init__(cfg)
        self.ureg = pint.UnitRegistry()

    def _compute_hermite_parameters(self, Nn: int, Nm: int, Np: int) -> dict:
        """
        Compute Hermite ladder operators and collision matrix for given mode counts.

        This replicates the spectrax initialization logic for a single species,
        allowing per-species mode counts without calling the full spectrax initialization.

        Args:
            Nn: Number of Hermite modes in x-velocity direction
            Nm: Number of Hermite modes in y-velocity direction
            Np: Number of Hermite modes in z-velocity direction

        Returns:
            dict: Contains:
                - sqrt_n_plus, sqrt_n_minus: Ladder operators for n direction (Nn,)
                - sqrt_m_plus, sqrt_m_minus: Ladder operators for m direction (Nm,)
                - sqrt_p_plus, sqrt_p_minus: Ladder operators for p direction (Np,)
                - col: Hypercollisional damping matrix (Np, Nm, Nn)
        """
        # Ladder operators for Hermite polynomial derivatives
        # sqrt_n_plus[n] = sqrt(n+1) for d/dξ Hₙ(ξ) = 2n Hₙ₋₁(ξ)
        n = jnp.arange(Nn)
        m = jnp.arange(Nm)
        p = jnp.arange(Np)

        sqrt_n_plus = jnp.sqrt(n + 1)
        sqrt_n_minus = jnp.sqrt(n)
        sqrt_m_plus = jnp.sqrt(m + 1)
        sqrt_m_minus = jnp.sqrt(m)
        sqrt_p_plus = jnp.sqrt(p + 1)
        sqrt_p_minus = jnp.sqrt(p)

        # Collision matrix: hypercollisional damping for high modes
        # This prevents unphysical growth at high mode numbers
        # Formula: col[p,m,n] = sum over directions of i*(i-1)*(i-2) / (N-1)(N-2)(N-3)
        p_grid = p[:, None, None]
        m_grid = m[None, :, None]
        n_grid = n[None, None, :]

        def safe_collision_term(N, i):
            """Compute collision term with safe handling for small N."""
            term = i * (i - 1) * (i - 2)
            denom = (N - 1) * (N - 2) * (N - 3)
            return jnp.where(N > 3, term / denom, 0.0)

        col = safe_collision_term(Nn, n_grid) + safe_collision_term(Nm, m_grid) + safe_collision_term(Np, p_grid)

        return {
            "sqrt_n_plus": sqrt_n_plus,
            "sqrt_n_minus": sqrt_n_minus,
            "sqrt_m_plus": sqrt_m_plus,
            "sqrt_m_minus": sqrt_m_minus,
            "sqrt_p_plus": sqrt_p_plus,
            "sqrt_p_minus": sqrt_p_minus,
            "col": col,
        }

    def write_units(self) -> dict:
        """
        Calculate plasma parameters and normalization constants.

        This method is called early in ergoExo.setup() to establish the unit system.
        Results are logged to MLflow as units.yaml artifact.

        Reference: adept/_vlasov1d/modules.py lines 24-71 (pint-based units)
                   adept/vfp1d/base.py lines 21-95 (astropy units example)

        Physical quantities to calculate:
        - wp0: electron plasma frequency [rad/s]
        - tp0: plasma period (2π/wp0) [s]
        - v0: electron thermal velocity [m/s]
        - x0: electron skin depth (c/wp0) [m]
        - c_light: speed of light in normalized units
        - beta: thermal velocity ratio (v0/c)
        - box_length: simulation domain size [m]
        - sim_duration: simulation time [s]
        - lambda_D: Debye length (normalized units)
        - k_norm: Perturbation wavenumber normalized to inverse Debye length

        TODO:
        - Extract normalizing_density and normalizing_temperature from cfg["units"]
        - Use pint to calculate derived quantities
        - Store results in self.cfg["units"]["derived"]
        - Return dict of physical quantities for MLflow logging

        Returns:
            dict: Physical quantities with pint units (for MLflow artifact)
        """
        # Get physics parameters from config
        physics = self.cfg.get("physics", {})

        # Calculate lambda_D and k_norm if we have the necessary parameters
        units_dict = {}

        if "alpha_s" in physics and "mi_me" in physics and "Lx" in physics:
            alpha_s = physics["alpha_s"]
            mi_me = physics["mi_me"]
            Lx = physics["Lx"]

            # Debye length (from _diagnostics.py line 20)
            lambda_D = float(jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2)))))

            # Perturbation wavenumber normalized to inverse Debye length (from _diagnostics.py line 21)
            k_norm = float(jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx)

            units_dict["lambda_D"] = lambda_D
            units_dict["k_norm"] = k_norm

        return units_dict

    def get_derived_quantities(self) -> None:
        """
        Calculate derived grid quantities (scalars and strings only).

        This method is called after write_units() and before MLflow parameter logging.
        Only scalar/string quantities should be added here (no JAX arrays).

        Reference: adept/_vlasov1d/modules.py lines 73-100

        Quantities to calculate:
        - dt: timestep [normalized units] (if not provided, calculate from tmax/nt)
        - tmax: simulation end time [normalized units]
        - nt: number of timesteps (if not provided, calculate from tmax/dt)
        - max_steps: maximum solver steps (usually same as nt, cap at 1e6 if needed)
        """
        cfg_grid = self.cfg["grid"]

        # Get tmax from grid or physics config
        if "tmax" not in cfg_grid:
            cfg_grid["tmax"] = self.cfg["physics"].get("t_max", 50.0)
        tmax = cfg_grid["tmax"]

        # Get dt from config
        dt = cfg_grid.get("dt", None)

        # Calculate nt from tmax and dt
        if dt is None:
            # If dt not provided, try to get nt or default to 501
            nt = cfg_grid.get("nt", 501)
            dt = tmax / nt
            cfg_grid["dt"] = dt
        else:
            # Calculate nt from tmax and dt
            nt = int(tmax / dt) + 1
            cfg_grid["nt"] = nt

        # Ensure nt is set
        if "nt" not in cfg_grid:
            cfg_grid["nt"] = nt

        # Recalculate tmax to be consistent with dt and nt
        cfg_grid["tmax"] = dt * nt

        # Set max_steps (safety limit)
        if nt > 1e6:
            cfg_grid["max_steps"] = int(1e6)
            print(r"Only running $10^6$ steps")
        else:
            cfg_grid["max_steps"] = nt + 4

        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self) -> None:
        """
        Build spectrax input_parameters and solver_parameters dicts.

        This method is called after MLflow param logging, so JAX arrays can be added here.
        Initialize the Fourier and Hermite coefficient arrays that spectrax needs.

        Reference: adept/_vlasov1d/modules.py lines 102-167
        """

        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        # Build input_parameters dict (matches TOML structure)
        input_parameters = {
            "Lx": float(physics["Lx"]),
            "Ly": float(physics["Ly"]),
            "Lz": float(physics["Lz"]),
            "mi_me": float(physics["mi_me"]),
            "Ti_Te": float(physics["Ti_Te"]),
            "qs": physics["qs"],
            "alpha_e": physics["alpha_e"],
            "alpha_s": physics["alpha_s"],
            "u_s": physics["u_s"],
            "Omega_cs": physics["Omega_cs"],
            "nu": float(physics["nu"]),
            "t_max": float(grid["tmax"]),
            "nx": int(physics["nx"]),
            "ny": int(physics["ny"]),
            "nz": int(physics["nz"]),
            "dn1": float(physics["dn1"]),
        }

        # Parse per-species Hermite mode configuration with backward compatibility
        if "hermite_modes" in grid:
            # New per-species format
            Nn_e = int(grid["hermite_modes"]["electrons"]["Nn"])
            Nm_e = int(grid["hermite_modes"]["electrons"]["Nm"])
            Np_e = int(grid["hermite_modes"]["electrons"]["Np"])
            Nn_i = int(grid["hermite_modes"]["ions"]["Nn"])
            Nm_i = int(grid["hermite_modes"]["ions"]["Nm"])
            Np_i = int(grid["hermite_modes"]["ions"]["Np"])
        elif "Nn" in grid and "Nm" in grid and "Np" in grid:
            # Legacy format: same modes for both species
            Nn_e = Nn_i = int(grid["Nn"])
            Nm_e = Nm_i = int(grid["Nm"])
            Np_e = Np_i = int(grid["Np"])
        else:
            raise ValueError("Must specify grid.hermite_modes or grid.Nn/Nm/Np")

        # Store per-species mode counts in grid config for later access
        grid["Nn_electrons"] = Nn_e
        grid["Nm_electrons"] = Nm_e
        grid["Np_electrons"] = Np_e
        grid["Nn_ions"] = Nn_i
        grid["Nm_ions"] = Nm_i
        grid["Np_ions"] = Np_i

        # Build solver_parameters dict
        solver_name = grid.get("solver", "Dopri8")
        adaptive_time_step = grid.get("adaptive_time_step", True)

        solver_parameters = {
            "Nx": int(grid["Nx"]),
            "Ny": int(grid["Ny"]),
            "Nz": int(grid["Nz"]),
            "Nn_electrons": Nn_e,
            "Nm_electrons": Nm_e,
            "Np_electrons": Np_e,
            "Nn_ions": Nn_i,
            "Nm_ions": Nm_i,
            "Np_ions": Np_i,
            "Ns": int(grid["Ns"]),
            "timesteps": int(grid["nt"]),
            "dt": float(grid["dt"]),
            "adaptive_time_step": adaptive_time_step,
            "solver": solver_name,
        }

        # Initialize Fk_0 (electromagnetic field Fourier components)
        # Shape: (6, Ny, Nx, Nz) for [Ex, Ey, Ez, Bx, By, Bz]
        Nx, Ny, Nz = solver_parameters["Nx"], solver_parameters["Ny"], solver_parameters["Nz"]
        Fk_0 = jnp.zeros((6, Ny, Nx, Nz), dtype=jnp.complex128)

        # Calculate derived quantities for field initialization
        nx, ny, nz = input_parameters["nx"], input_parameters["ny"], input_parameters["nz"]
        Lx, Ly, Lz = input_parameters["Lx"], input_parameters["Ly"], input_parameters["Lz"]
        dn = input_parameters["dn1"]
        Omega_cs = input_parameters["Omega_cs"]

        n = nx + ny + nz
        L = Lx * jnp.sign(nx) + Ly * jnp.sign(ny) + Lz * jnp.sign(nz)
        E_field_component = int(jnp.sign(ny) + 2 * jnp.sign(nz))

        # Set field perturbation at ±k modes using standard FFT indexing (k=0 at index 0)
        if n != 0 and Omega_cs[0] != 0:
            amplitude = dn * L / (4 * jnp.pi * n * Omega_cs[0])

            # Set -k mode
            idx_x_minus = fft_index(-nx, Nx)
            idx_y_minus = fft_index(-ny, Ny)
            idx_z_minus = fft_index(-nz, Nz)
            Fk_0 = Fk_0.at[E_field_component, idx_y_minus, idx_x_minus, idx_z_minus].set(amplitude)

            # Set +k mode
            idx_x_plus = fft_index(nx, Nx)
            idx_y_plus = fft_index(ny, Ny)
            idx_z_plus = fft_index(nz, Nz)
            Fk_0 = Fk_0.at[E_field_component, idx_y_plus, idx_x_plus, idx_z_plus].set(amplitude)

        input_parameters["Fk_0"] = Fk_0

        # Initialize Ck_0 (distribution function Hermite-Fourier components)
        # Now separate per species with per-species mode counts
        # Electron shape: (Np_e, Nm_e, Nn_e, Ny, Nx, Nz)
        # Ion shape: (Np_i, Nm_i, Nn_i, Ny, Nx, Nz)
        Nn_e = solver_parameters["Nn_electrons"]
        Nm_e = solver_parameters["Nm_electrons"]
        Np_e = solver_parameters["Np_electrons"]
        Nn_i = solver_parameters["Nn_ions"]
        Nm_i = solver_parameters["Nm_ions"]
        Np_i = solver_parameters["Np_ions"]

        Ck_0_electrons = jnp.zeros((Np_e, Nm_e, Nn_e, Ny, Nx, Nz), dtype=jnp.complex128)
        Ck_0_ions = jnp.zeros((Np_i, Nm_i, Nn_i, Ny, Nx, Nz), dtype=jnp.complex128)

        # Use standard FFT indexing: k=0 at index 0
        # Compute FFT indices for ±k modes
        idx_x_minus = fft_index(-nx, Nx)
        idx_y_minus = fft_index(-ny, Ny)
        idx_z_minus = fft_index(-nz, Nz)
        idx_x_plus = fft_index(nx, Nx)
        idx_y_plus = fft_index(ny, Ny)
        idx_z_plus = fft_index(nz, Nz)

        # Electron distribution (only (0,0,0) Hermite mode)
        # Use alpha_s[0] for electron thermal velocity (in x-direction for 1D)
        alpha_e = input_parameters["alpha_s"][0]
        Ce0_mk = 0 + 1j * (1 / (2 * alpha_e**3)) * dn  # -k mode
        Ce0_0 = 1 / (alpha_e**3) + 0 * 1j  # Equilibrium
        Ce0_k = 0 - 1j * (1 / (2 * alpha_e**3)) * dn  # +k mode

        # Electron modes (p=0, m=0, n=0 Hermite mode)
        Ck_0_electrons = Ck_0_electrons.at[0, 0, 0, 0, 0, 0].set(Ce0_0)  # k=0 equilibrium
        Ck_0_electrons = Ck_0_electrons.at[0, 0, 0, idx_y_minus, idx_x_minus, idx_z_minus].set(Ce0_mk)  # -k
        Ck_0_electrons = Ck_0_electrons.at[0, 0, 0, idx_y_plus, idx_x_plus, idx_z_plus].set(Ce0_k)  # +k

        # Ion distribution (only (0,0,0) Hermite mode)
        # Use alpha_s[3] for ion thermal velocity (first component in ion triplet)
        alpha_i = input_parameters["alpha_s"][3]
        Ci0_0 = 1 / (alpha_i**3) + 0 * 1j

        # Ion distribution (equilibrium only at k=0)
        Ck_0_ions = Ck_0_ions.at[0, 0, 0, 0, 0, 0].set(Ci0_0)

        input_parameters["Ck_0_electrons"] = Ck_0_electrons
        input_parameters["Ck_0_ions"] = Ck_0_ions

        # Store in config
        self.cfg["spectrax_input"] = input_parameters
        self.cfg["spectrax_solver"] = solver_parameters

        # Create time axis for save functions (needed by get_save_quantities)
        # This matches the pattern in Vlasov1D
        tmax = self.cfg["grid"]["tmax"]
        nt = self.cfg["grid"]["nt"]
        self.cfg["grid"]["t"] = jnp.linspace(0, tmax, nt)

    def init_state_and_args(self) -> None:
        """
        Initialize state and runtime arguments.

        Creates simulation parameters, initial conditions, and args tuple for ODE system.
        Follows pattern from adept/_vlasov1d/modules.py and adept/_tf1d/modules.py.

        Reference: adept/_vlasov1d/modules.py lines 238-272
        """
        # Get parameters from config
        input_params = self.cfg["spectrax_input"]
        solver_params = self.cfg["spectrax_solver"]

        # Get dimensions from solver_params
        Nx = solver_params["Nx"]
        Ny = solver_params["Ny"]
        Nz = solver_params["Nz"]
        Nn_e = solver_params["Nn_electrons"]
        Nm_e = solver_params["Nm_electrons"]
        Np_e = solver_params["Np_electrons"]
        Nn_i = solver_params["Nn_ions"]
        Nm_i = solver_params["Nm_ions"]
        Np_i = solver_params["Np_ions"]

        # Compute per-species Hermite parameters (ladder operators and collision matrices)
        hermite_params_e = self._compute_hermite_parameters(Nn_e, Nm_e, Np_e)
        hermite_params_i = self._compute_hermite_parameters(Nn_i, Nm_i, Np_i)

        # Compute k-space grids (shared by both species)
        Lx = input_params["Lx"]
        Ly = input_params["Ly"]
        Lz = input_params["Lz"]

        # Fourier wavenumber grids in standard FFT ordering: [0, 1, ..., N/2-1, -N/2, ..., -1]
        # Grid contains integer mode numbers * 2π (physical wavenumber = k_grid / L)
        kx_simulation = jnp.fft.fftfreq(Nx) * Nx * 2 * jnp.pi
        ky_simulation = jnp.fft.fftfreq(Ny) * Ny * 2 * jnp.pi
        kz_simulation = jnp.fft.fftfreq(Nz) * Nz * 2 * jnp.pi
        ky_grid, kx_grid, kz_grid = jnp.meshgrid(ky_simulation, kx_simulation, kz_simulation, indexing="ij")
        k2_grid = kx_grid**2 + ky_grid**2 + kz_grid**2

        # Normalized gradient operator
        nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])

        # Get initial conditions
        Ck_0_electrons = input_params["Ck_0_electrons"]
        Ck_0_ions = input_params["Ck_0_ions"]

        # Create initial state dictionary with per-species Ck and Fk
        # Store as float64 views for diffrax compatibility (diffrax doesn't fully support complex state)
        self.state = {
            "Ck_electrons": Ck_0_electrons.view(jnp.float64),  # Shape (Np_e, Nm_e, Nn_e, Ny, Nx, Nz)
            "Ck_ions": Ck_0_ions.view(jnp.float64),  # Shape (Np_i, Nm_i, Nn_i, Ny, Nx, Nz)
            "Fk": input_params["Fk_0"].view(jnp.float64),  # Shape (6, Ny, Nx, Nz)
        }

        # Create real-space grid for driver
        self.xax = jnp.linspace(0, Lx, Nx, endpoint=False)

        # Store driver config for VectorField initialization
        self.driver_config = self.cfg.get("drivers", {})

        # Add density noise configuration if present
        density_cfg = self.cfg.get("density", {})
        noise_cfg = density_cfg.get("noise", {})
        if noise_cfg:
            self.driver_config["density_noise"] = noise_cfg

        # Add dt to driver_config for time-dependent noise seeding
        self.driver_config["dt"] = self.cfg["grid"]["dt"]

        # Build args dictionary for the ODE system (only time-varying physical parameters)
        self.args = {
            "qs": jnp.array(input_params["qs"]),
            "nu": float(input_params["nu"]),
            "D": 0.0,  # Diffusion coefficient (typically zero unless specified)
            "Omega_cs": jnp.array(input_params["Omega_cs"]),
            "alpha_s": jnp.array(input_params["alpha_s"]),
            "u_s": jnp.array(input_params["u_s"]),
        }

        # Store per-species grid quantities separately for VectorField initialization
        # These are constants and will be stored as class attributes in VectorField
        self.grid_quantities_electrons = {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "kx_grid": kx_grid,
            "ky_grid": ky_grid,
            "kz_grid": kz_grid,
            "k2_grid": k2_grid,
            "nabla": nabla,
            "col": hermite_params_e["col"],
            "sqrt_n_plus": hermite_params_e["sqrt_n_plus"],
            "sqrt_n_minus": hermite_params_e["sqrt_n_minus"],
            "sqrt_m_plus": hermite_params_e["sqrt_m_plus"],
            "sqrt_m_minus": hermite_params_e["sqrt_m_minus"],
            "sqrt_p_plus": hermite_params_e["sqrt_p_plus"],
            "sqrt_p_minus": hermite_params_e["sqrt_p_minus"],
        }

        self.grid_quantities_ions = {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "kx_grid": kx_grid,
            "ky_grid": ky_grid,
            "kz_grid": kz_grid,
            "k2_grid": k2_grid,
            "nabla": nabla,
            "col": hermite_params_i["col"],
            "sqrt_n_plus": hermite_params_i["sqrt_n_plus"],
            "sqrt_n_minus": hermite_params_i["sqrt_n_minus"],
            "sqrt_m_plus": hermite_params_i["sqrt_m_plus"],
            "sqrt_m_minus": hermite_params_i["sqrt_m_minus"],
            "sqrt_p_plus": hermite_params_i["sqrt_p_plus"],
            "sqrt_p_minus": hermite_params_i["sqrt_p_minus"],
        }

        # Build sponge boundary layer profiles from config.
        # Sponge is applied analytically by SplitStepDampingSolver after each RK step.
        # Quadratic profile: sigma(x) = strength * (normalised_distance_into_sponge)²
        # Supports right boundary only, left boundary only, or both sides.
        boundary_cfg = self.cfg.get("boundary_conditions", {})
        sponge_cfg = boundary_cfg.get("sponge", {})
        if sponge_cfg.get("enabled", False):
            sponge_strength = float(sponge_cfg.get("strength", 10.0))
            sponge_width_frac = float(sponge_cfg.get("width", 0.2))
            sponge_sides = sponge_cfg.get("sides", "right")

            sigma_x = jnp.zeros_like(self.xax)

            # Right-boundary sponge: x > Lx*(1 - width)
            if sponge_sides in ("right", "both"):
                x_r0 = float(Lx) * (1.0 - sponge_width_frac)
                norm_r = jnp.clip((self.xax - x_r0) / (float(Lx) - x_r0 + 1e-10), 0.0, 1.0)
                sigma_x = sigma_x + sponge_strength * norm_r**2

            # Left-boundary sponge: x < Lx*width
            if sponge_sides in ("left", "both"):
                x_l1 = float(Lx) * sponge_width_frac
                norm_l = jnp.clip((x_l1 - self.xax) / (x_l1 + 1e-10), 0.0, 1.0)
                sigma_x = sigma_x + sponge_strength * norm_l**2

            self.grid_quantities_electrons["sponge_fields"] = sigma_x
            self.grid_quantities_electrons["sponge_plasma"] = sigma_x
            self.grid_quantities_ions["sponge_plasma"] = sigma_x

    def init_diffeqsolve(self) -> None:
        """
        Initialize solver configuration with save quantities.

        Sets up time quantities, ODE system, solver, and save configuration.
        Follows pattern from adept/_vlasov1d/modules.py and adept/_tf1d/modules.py.

        Reference: adept/_vlasov1d/modules.py lines 274-281
        """
        # Process save configuration to add time axes and save functions
        self.cfg = get_save_quantities(self.cfg)

        # Set time quantities (matches Vlasov1D pattern)
        self.time_quantities = {"t0": 0.0, "t1": self.cfg["grid"]["tmax"], "max_steps": self.cfg["grid"]["max_steps"]}

        # Get solver configuration
        solver_params = self.cfg["spectrax_solver"]

        # Get dimensions
        Nx = solver_params["Nx"]
        Ny = solver_params["Ny"]
        Nz = solver_params["Nz"]
        Nn_e = solver_params["Nn_electrons"]
        Nm_e = solver_params["Nm_electrons"]
        Np_e = solver_params["Np_electrons"]
        Nn_i = solver_params["Nn_ions"]
        Nm_i = solver_params["Nm_ions"]
        Np_i = solver_params["Np_ions"]
        Ns = solver_params["Ns"]

        # Static ions: freeze ion distribution (no Lorentz force, no current, no free-streaming)
        static_ions = self.cfg["physics"].get("static_ions", False)

        # Only the Lawson-RK4 exponential integrator is supported.
        integrator_type = self.cfg["grid"].get("integrator", "exponential")

        if integrator_type == "exponential":
            # --- Lawson-RK4 exponential integrator path ---
            from adept._spectrax1d.exponential_operators import build_combined_exponential
            from adept._spectrax1d.integrators import LawsonRK4Solver
            from adept._spectrax1d.nonlinear_vector_field import NonlinearVectorField

            # Build exponential operators for all linear terms
            combined_exp = build_combined_exponential(
                grid_quantities_electrons=self.grid_quantities_electrons,
                grid_quantities_ions=self.grid_quantities_ions,
                alpha_s=self.args["alpha_s"],
                u_s=self.args["u_s"],
                nu=self.args["nu"],
                D=self.args.get("D", 0.0),
                Nn_e=Nn_e,
                Nm_e=Nm_e,
                Np_e=Np_e,
                Nn_i=Nn_i,
                Nm_i=Nm_i,
                Np_i=Np_i,
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                static_ions=static_ions,
            )

            # Create nonlinear-only vector field
            vector_field = NonlinearVectorField(
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                Nn_electrons=Nn_e,
                Nm_electrons=Nm_e,
                Np_electrons=Np_e,
                Nn_ions=Nn_i,
                Nm_ions=Nm_i,
                Np_ions=Np_i,
                Ns=Ns,
                xax=self.xax,
                driver_config=self.driver_config,
                grid_quantities_electrons=self.grid_quantities_electrons,
                grid_quantities_ions=self.grid_quantities_ions,
                dt=float(self.cfg["grid"]["dt"]),
                static_ions=static_ions,
            )

            # Create Lawson-RK4 solver
            base_solver = LawsonRK4Solver(combined_exp=combined_exp)

            # Exponential integrator uses fixed timestep (stiffness removed)
            stepsize_controller = ConstantStepSize()
            print(f"Using Lawson-RK4 exponential integrator with fixed dt={self.cfg['grid']['dt']}")

        else:
            raise ValueError(
                f"Unsupported integrator '{integrator_type}'. Only 'exponential' (Lawson-RK4) is supported."
            )

        # Build SubSaveAt dictionary for diagnostics
        subsaves = {}
        for k, v in self.cfg["save"].items():
            if isinstance(v, dict) and "t" in v and "func" in v:
                subsaves[k] = SubSaveAt(ts=v["t"]["ax"], fn=v["func"])

        # Wrap with split-step damping integrator (both paths)
        # This applies sponge boundary damping analytically outside the solver substeps
        from adept._spectrax1d.integrators import SplitStepDampingSolver

        solver = SplitStepDampingSolver(
            wrapped_solver=base_solver,
            sponge_fields=self.grid_quantities_electrons.get("sponge_fields", None),
            sponge_plasma_e=self.grid_quantities_electrons.get("sponge_plasma", None),
            sponge_plasma_i=self.grid_quantities_ions.get("sponge_plasma", None),
            Ny=Ny,
            Nx=Nx,
            Nz=Nz,
        )

        # Store diffeqsolve configuration (matches pattern from other modules)
        self.diffeqsolve_quants = {
            "terms": ODETerm(vector_field),
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "saveat": {"subs": subsaves},
        }

    def __call__(self, trainable_modules: dict, args: dict | None = None) -> dict:
        """
        Run the spectrax simulation.

        Uses pre-initialized quantities from init_state_and_args() and init_diffeqsolve().
        Follows pattern from adept/_vlasov1d/modules.py and adept/_tf1d/modules.py.

        Reference: adept/_vlasov1d/modules.py lines 283-298

        Args:
            trainable_modules: Dict of equinox.Module objects (unused)
            args: Optional runtime arguments (uses self.args if None)

        Returns:
            dict: {"solver result": <diffrax.Solution>}
        """
        # Use pre-initialized args if not provided
        if args is None:
            args = self.args

        # Call diffeqsolve with pre-initialized quantities
        sol = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            stepsize_controller=self.diffeqsolve_quants["stepsize_controller"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            dt0=self.cfg["grid"]["dt"],
            y0=self.state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
            max_steps=self.time_quantities["max_steps"],
            progress_meter=TqdmProgressMeter(),
        )

        return {"solver result": sol}

    def _save_scalar_outputs(self, scalar_outputs: dict, td: str) -> None:
        """Save scalar outputs to a text file."""
        with open(os.path.join(td, "scalar_outputs.txt"), "w") as f:
            f.write("Spectrax Scalar Outputs\n")
            f.write("=" * 50 + "\n\n")
            for key, value in sorted(scalar_outputs.items()):
                f.write(f"{key}: {value}\n")

    def _process_distribution_function(self, Ck, t_array, Nn, Nm, Np, Nx, Ny, Nz, td: str) -> None:
        """Process and save distribution function (Hermite-Fourier coefficients)."""
        import numpy as np

        # Create coordinates
        hermite_modes = np.arange(Ck.shape[1])
        kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0 / Nx))  # Sort kx in increasing order

        # Shift the data to match sorted kx
        Ck_shifted = np.fft.fftshift(Ck, axes=-2)  # Shift along kx dimension

        Ck_ds = xr.Dataset(
            {
                "Ck": (["t", "hermite_mode", "ky", "kx", "kz"], Ck_shifted),
            },
            coords={
                "t": t_array,
                "hermite_mode": hermite_modes,
                "ky": np.arange(Ny),
                "kx": kx,
                "kz": np.arange(Nz),
            },
        )

        Ck_ds.to_netcdf(os.path.join(td, "binary", "distribution.nc"))

        # Create Hermite mode amplitude plots (use original unshifted data)
        plot_hermite_modes_at_k(Ck, t_array, Nn, Nm, Np, Nx, Ny, Nz, td)

    def _process_electromagnetic_fields(self, Fk, t_array, Nx, Ny, Nz, td: str) -> None:
        """Process and save electromagnetic fields."""
        import numpy as np

        kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0 / Nx))  # Sort kx in increasing order

        # Shift the data to match sorted kx
        Fk_shifted = np.fft.fftshift(Fk, axes=-2)  # Shift along kx dimension

        Fk_ds = xr.Dataset(
            {
                "Ex": (["t", "ky", "kx", "kz"], Fk_shifted[:, 0, :, :, :]),
                "Ey": (["t", "ky", "kx", "kz"], Fk_shifted[:, 1, :, :, :]),
                "Ez": (["t", "ky", "kx", "kz"], Fk_shifted[:, 2, :, :, :]),
                "Bx": (["t", "ky", "kx", "kz"], Fk_shifted[:, 3, :, :, :]),
                "By": (["t", "ky", "kx", "kz"], Fk_shifted[:, 4, :, :, :]),
                "Bz": (["t", "ky", "kx", "kz"], Fk_shifted[:, 5, :, :, :]),
            },
            coords={
                "t": t_array,
                "ky": np.arange(Ny),
                "kx": kx,
                "kz": np.arange(Nz),
            },
        )

        Fk_ds.to_netcdf(os.path.join(td, "binary", "fields.nc"))

        # Create field mode amplitude plots (use original unshifted data)
        plot_field_mode_amplitudes(Fk, t_array, Nx, Ny, Nz, td)

    def _process_energy_diagnostics(self, array_outputs: dict, t_array, td: str) -> None:
        """Process and save energy diagnostics."""
        energy_vars = {}
        if "EM_energy" in array_outputs:
            energy_vars["EM_energy"] = (["t"], array_outputs["EM_energy"])
        if "kinetic_energy" in array_outputs:
            energy_vars["kinetic_energy"] = (["t"], array_outputs["kinetic_energy"])

        if energy_vars:
            energy_ds = xr.Dataset(
                energy_vars, coords={"t": t_array[: len(energy_vars[next(iter(energy_vars.keys()))][1])]}
            )
            energy_ds.to_netcdf(os.path.join(td, "binary", "energies.nc"))

    def _process_grid_data(self, array_outputs: dict, td: str) -> None:
        """Process and save grid data arrays."""
        other_arrays = {
            k: array_outputs[k]
            for k in ["k2_grid", "collision_matrix", "alpha_e", "alpha_s", "Dmega_cs"]
            if k in array_outputs
        }
        if other_arrays:
            # Save each array with appropriate dimensions
            data_vars = {}
            for k, v in other_arrays.items():
                # Create dimension names based on the array shape
                # Use unique dimension names for each array to avoid conflicts
                if k == "k2_grid":
                    # Grid in k-space: (Ny, Nx, Nz)
                    data_vars[k] = (["ky", "kx", "kz"], v)
                else:
                    # Use array-specific dimension names to avoid conflicts
                    dims = [f"{k}_dim_{i}" for i in range(len(v.shape))]
                    data_vars[k] = (dims, v)

            other_ds = xr.Dataset(data_vars)
            other_ds.to_netcdf(os.path.join(td, "binary", "grid_data.nc"))

    def _compute_metrics(self, array_outputs: dict, timesteps: int) -> dict:
        """Compute final metrics for MLflow logging."""
        import numpy as np

        metrics = {
            "simulation_completed": True,
            "n_timesteps": timesteps,
        }

        # Add final energy values if available (use abs since energies must be real and positive)
        if "EM_energy" in array_outputs:
            metrics["final_EM_energy"] = float(np.abs(array_outputs["EM_energy"][-1]))
        if "kinetic_energy" in array_outputs:
            metrics["final_kinetic_energy"] = float(np.abs(array_outputs["kinetic_energy"][-1]))

        return metrics

    def _convert_fields_to_real_space(self, Fk_timeseries: Array) -> dict:
        """
        Convert electromagnetic fields from Fourier space to real space.

        Args:
            Fk_timeseries: Array of shape (nt, 6, Ny, Nx, Nz) in Fourier space

        Returns:
            dict: Dictionary with keys Ex, Ey, Ez, Bx, By, Bz containing real-space fields
        """
        import numpy as np

        field_names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
        fields_real = {}

        for i, name in enumerate(field_names):
            # Apply inverse FFT and take real part
            # For 1D case (Ny=1, Nz=1), this is essentially just ifft along x
            field_real = np.real(np.fft.ifftn(Fk_timeseries[:, i, :, :, :], axes=(-3, -2, -1), norm="forward"))
            fields_real[name] = field_real

        return fields_real

    def post_process(self, run_output: dict, td: str) -> dict:
        """
        Post-process spectrax output for visualization and analysis.

        This orchestrates the conversion of SPECTRAX output into xarray datasets,
        generates diagnostic plots, and computes summary metrics for MLflow.

        Plots are organized into three folders:
        - plots/scalars/: Time series of scalar diagnostics (linear + log scale)
        - plots/fields/: Spacetime plots and lineouts of EM fields in real space
        - plots/distributions/: Facet plots of distribution function in (kx, hermite_mode) space

        Args:
            run_output: Dict containing {"solver result": <diffrax Solution>}
            td: Temporary directory path for saving artifacts

        Returns:
            dict: {"metrics": {...}} containing scalar metrics for MLflow logging
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract solution object
        sol = run_output["solver result"]

        # Create directory structure
        os.makedirs(os.path.join(td, "binary"), exist_ok=True)
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        os.makedirs(os.path.join(td, "plots", "scalars"), exist_ok=True)
        os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
        os.makedirs(os.path.join(td, "plots", "fields", "lineouts"), exist_ok=True)
        os.makedirs(os.path.join(td, "plots", "distributions"), exist_ok=True)

        binary_dir = os.path.join(td, "binary")

        # Get grid parameters
        Nx = int(self.cfg["grid"]["Nx"])
        Ny = int(self.cfg["grid"]["Ny"])
        Nz = int(self.cfg["grid"]["Nz"])
        # Support both per-species (hermite_modes) and legacy flat format
        if "Nn_electrons" in self.cfg["grid"]:
            Nn = int(self.cfg["grid"]["Nn_electrons"])
            Nm = int(self.cfg["grid"]["Nm_electrons"])
            Np = int(self.cfg["grid"]["Np_electrons"])
        else:
            Nn = int(self.cfg["grid"]["Nn"])
            Nm = int(self.cfg["grid"]["Nm"])
            Np = int(self.cfg["grid"]["Np"])

        # Create real-space x coordinate
        x = np.linspace(0, self.cfg["physics"]["Lx"], Nx)

        # Process saved time series from SubSaveAt
        saved_datasets = {}
        has_fields_data = False
        for k in sol.ys.keys():
            if k == "default":
                # Default scalar diagnostics
                scalars_xr = store_scalars(self.cfg, sol.ys[k], sol.ts[k], binary_dir)
                saved_datasets["scalars"] = scalars_xr

                # Plot scalar diagnostics in scalars/ folder
                for nm, srs in scalars_xr.items():
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                    srs.plot(ax=ax[0])
                    ax[0].grid()
                    ax[0].set_title(f"{nm}")
                    np.log10(np.abs(srs) + 1e-20).plot(ax=ax[1])
                    ax[1].grid()
                    ax[1].set_ylabel(f"$log_{{10}}$(|{nm}|)")
                    ax[1].set_title("Log scale")
                    fig.savefig(os.path.join(td, "plots", "scalars", f"{nm}.png"), bbox_inches="tight")
                    plt.close()

            elif k == "fields":
                fields_data = sol.ys[k]
                t_array = sol.ts[k]
                has_fields_data = True

                if isinstance(fields_data, dict):
                    # Interpolated fields output (x or kx)
                    if "fields" in self.cfg["save"] and "x" in self.cfg["save"]["fields"]:
                        axis = ("x", self.cfg["save"]["fields"]["x"]["ax"])
                        axis_name = "x"
                    elif "fields" in self.cfg["save"] and "kx" in self.cfg["save"]["fields"]:
                        axis = ("kx", self.cfg["save"]["fields"]["kx"]["ax"])
                        axis_name = "kx"
                    else:
                        axis = ("x", x)
                        axis_name = "x"

                    fields_xr_dict = {
                        name: xr.DataArray(val, coords=[("t", t_array), axis], name=name)
                        for name, val in fields_data.items()
                    }
                    fields_xr = xr.Dataset(fields_xr_dict)
                    fields_xr.to_netcdf(os.path.join(binary_dir, f"fields-{axis_name}-t={round(t_array[-1], 4)}.nc"))
                    saved_datasets[f"fields_{axis_name}"] = fields_xr

                    # Create spacetime and lineout plots (same as Fourier-space path)
                    if axis_name == "x":
                        plot_fields_spacetime(fields_xr, td)
                        plot_fields_lineouts(fields_xr, td)
                else:
                    # Electromagnetic fields in Fourier space - convert to real space and plot
                    Fk_array = np.asarray(fields_data)  # Shape: (nt, 6, Ny, Nx, Nz)

                    # Convert to real space
                    fields_real = self._convert_fields_to_real_space(Fk_array)

                    # For 1D case, squeeze out y and z dimensions and create xarray datasets
                    field_names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
                    fields_xr_dict = {}
                    for name in field_names:
                        # Squeeze to (nt, Nx) for 1D case
                        field_1d = np.squeeze(fields_real[name])
                        fields_xr_dict[name] = xr.DataArray(field_1d, coords=[("t", t_array), ("x", x)], name=name)

                    fields_xr = xr.Dataset(fields_xr_dict)

                    # Save to netCDF
                    fields_xr.to_netcdf(os.path.join(binary_dir, f"fields-t={round(t_array[-1], 4)}.nc"))
                    saved_datasets["fields"] = fields_xr

                    # Create plots
                    plot_fields_spacetime(fields_xr, td)
                    plot_fields_lineouts(fields_xr, td)

            elif k in ["hermite", "distribution"]:
                # Distribution function (Hermite coefficients) - per-species dict
                species_dict = sol.ys[k]

                # Store each species separately
                electrons_xr = store_species_distribution_timeseries(
                    self.cfg, "electrons", np.asarray(species_dict["electrons"]), sol.ts[k], binary_dir
                )
                ions_xr = store_species_distribution_timeseries(
                    self.cfg, "ions", np.asarray(species_dict["ions"]), sol.ts[k], binary_dir
                )

                saved_datasets["distribution_electrons"] = electrons_xr
                saved_datasets["distribution_ions"] = ions_xr

                # Note: Distribution facet plots removed - use EPW1D module for enhanced visualization
            elif k == "moments":
                # Distribution moments in real space
                moments_dict = sol.ys[k]
                t_array = sol.ts[k]

                moments_vars = {}
                for name, arr in moments_dict.items():
                    data = np.asarray(arr)
                    if data.ndim == 4 and Ny == 1 and Nz == 1:
                        data = np.squeeze(data, axis=(1, 3))
                        moments_vars[name] = (["t", "x"], data)
                    else:
                        moments_vars[name] = (["t", "y", "x", "z"], data)

                coords = {"t": t_array, "x": x}
                if Ny != 1 or Nz != 1:
                    coords["y"] = np.arange(Ny)
                    coords["z"] = np.arange(Nz)
                moments_xr = xr.Dataset(moments_vars, coords=coords)

                moments_xr.to_netcdf(os.path.join(binary_dir, f"moments-t={round(t_array[-1], 4)}.nc"))
                saved_datasets["moments"] = moments_xr

                plot_moments_spacetime(moments_xr, td)
                plot_moments_lineouts(moments_xr, td)

        # Compute metrics for MLflow
        metrics = {
            "simulation_completed": True,
        }

        if not has_fields_data:
            print("Warning: no fields data saved; skipping fields plots.")

        # Add timestep info from default save if available
        if "default" in sol.ts:
            metrics["n_timesteps"] = len(sol.ts["default"])

        # Add final values from scalar diagnostics if available
        if "scalars" in saved_datasets:
            for k, v in saved_datasets["scalars"].items():
                metrics[f"final_{k}"] = float(np.abs(v.values[-1]))

        return {"metrics": metrics, "datasets": saved_datasets}
