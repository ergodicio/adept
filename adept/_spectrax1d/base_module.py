"""Base ADEPTModule wrapper for spectrax-based Hermite-Fourier Vlasov-Maxwell solver."""

import os

import pint
import xarray as xr
from diffrax import ConstantStepSize, ODETerm, PIDController, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import Array
from jax import numpy as jnp

from adept._base_ import ADEPTModule
from adept._spectrax1d.storage import (
    get_save_quantities,
    store_scalars,
    store_species_distribution_timeseries,
)
from adept._spectrax1d.vector_field import SpectraxVectorField


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

    def _get_solver_instance(self, solver_name: str, ode_tolerance: float):
        """
        Get a Diffrax solver instance by name.

        This replicates the transformation from spectrax.load_parameters().

        Args:
            solver_name: Name of the solver (e.g., "Dopri8", "Tsit5")
            ode_tolerance: ODE solver tolerance for implicit methods

        Returns:
            Diffrax solver instance
        """
        import inspect

        import diffrax

        # Special case for ImplicitMidpoint (not in standard diffrax)
        if solver_name == "ImplicitMidpoint":
            try:
                from spectrax.helpers import ImplicitMidpoint

                return ImplicitMidpoint(rtol=ode_tolerance, atol=ode_tolerance)
            except ImportError as e:
                raise ValueError("ImplicitMidpoint requires spectrax to be installed") from e

        # Look for solver in diffrax module
        for cls_name, cls in inspect.getmembers(diffrax, inspect.isclass):
            if (
                issubclass(cls, diffrax.AbstractSolver)
                and cls is not diffrax.AbstractSolver
                and cls_name == solver_name
            ):
                return cls()

        raise ValueError(
            f"Solver '{solver_name}' is not supported. Choose from Diffrax solvers "
            "(e.g., Dopri8, Tsit5, Euler, Heun, etc.)"
        )

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
            "ode_tolerance": float(physics.get("ode_tolerance", 1e-8)),
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
        input_params = self.cfg["spectrax_input"]

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

        # Get shard_map configuration from grid config (default: False)
        use_shard_map = self.cfg["grid"].get("use_shard_map", False)

        # Choose integrator: "explicit" (default, Dopri8) or "exponential" (Lawson-RK4)
        integrator_type = self.cfg["grid"].get("integrator", "explicit")

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
            )

            # Build per-step Hermite filter if configured
            filter_config = self.cfg["grid"].get("hermite_filter", {})
            if filter_config.get("enabled", False):
                filter_order = filter_config.get("order", 36)
                filter_strength = filter_config.get("strength", 36.0)
                cutoff_fraction = filter_config.get("cutoff_fraction", 2.0 / 3.0)

                from adept._spectrax1d.exponential_operators import compute_houli_hermite_filter

                hermite_filter_e = compute_houli_hermite_filter(
                    Nn_e, Nm_e, Np_e, cutoff_fraction, filter_strength, filter_order
                )[:, :, :, None, None, None]
                hermite_filter_i = compute_houli_hermite_filter(
                    Nn_i, Nm_i, Np_i, cutoff_fraction, filter_strength, filter_order
                )[:, :, :, None, None, None]
            else:
                hermite_filter_e = None
                hermite_filter_i = None

            # Create Lawson-RK4 solver
            base_solver = LawsonRK4Solver(
                combined_exp=combined_exp,
                hermite_filter_e=hermite_filter_e,
                hermite_filter_i=hermite_filter_i,
            )

            # Exponential integrator uses fixed timestep (stiffness removed)
            stepsize_controller = ConstantStepSize()
            print(f"Using Lawson-RK4 exponential integrator with fixed dt={self.cfg['grid']['dt']}")

        else:
            # --- Standard explicit integrator path (default) ---
            # Create VectorField instance with per-species configuration and grid quantities
            vector_field = SpectraxVectorField(
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
                use_shard_map=use_shard_map,
                hermite_filter_config=self.cfg["grid"].get("hermite_filter", {}),
            )

            # Get stepsize controller
            adaptive_time_step = solver_params["adaptive_time_step"]

            # Compute dtmax from laser period if driver is present
            dtmax = None
            if adaptive_time_step and "drivers" in self.cfg:
                drivers_cfg = self.cfg["drivers"]
                for field_key in ["ex", "ey", "ez"]:
                    if drivers_cfg.get(field_key):
                        for pulse_name, pulse_cfg in drivers_cfg[field_key].items():
                            if isinstance(pulse_cfg, dict) and "w0" in pulse_cfg:
                                w0 = pulse_cfg["w0"]
                                laser_period = 2 * jnp.pi / w0
                                dtmax = laser_period / 20.0
                                print(f"Auto dtmax from laser: {float(dtmax):.6e} (period={float(laser_period):.6e})")
                                break
                    if dtmax is not None:
                        break

            controllers = {
                True: PIDController(
                    rtol=input_params["ode_tolerance"],
                    atol=input_params["ode_tolerance"],
                    dtmax=dtmax,
                ),
                False: ConstantStepSize(),
            }
            stepsize_controller = controllers[adaptive_time_step]

            # Get base solver instance
            base_solver = self._get_solver_instance(solver_params["solver"], input_params["ode_tolerance"])

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

    def _plot_hermite_modes_at_k(self, Ck, t_array, Nn, Nm, Np, Nx, Ny, Nz, td: str) -> None:
        """Plot Hermite mode amplitudes at kx=1 Fourier mode."""
        import matplotlib.pyplot as plt
        import numpy as np

        # k=1 mode at index 1 in standard FFT ordering
        idx_k1 = 1
        idx_k0 = 0  # k=0 for other dimensions (1D problem in x)

        if idx_k1 < Nx:
            # Get kx=1 mode for all time steps and Hermite modes
            kx1_mode = Ck[:, :, idx_k0, idx_k1, idx_k0]

            # Separate electron and ion contributions
            electron_modes = kx1_mode[:, : Nn * Nm * Np]
            ion_modes = kx1_mode[:, Nn * Nm * Np :]

            # Create xarray DataArrays for plotting
            hermite_mode_indices = np.arange(Nn * Nm * Np)

            electron_amp_da = xr.DataArray(
                np.abs(electron_modes),
                coords={"t": t_array, "hermite_mode": hermite_mode_indices},
                dims=["t", "hermite_mode"],
                name="electron_amplitude",
            )

            ion_amp_da = xr.DataArray(
                np.abs(ion_modes),
                coords={"t": t_array, "hermite_mode": hermite_mode_indices},
                dims=["t", "hermite_mode"],
                name="ion_amplitude",
            )

            # Create figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(8, 4), tight_layout=True)
            fig.suptitle("Hermite Mode Amplitudes at kx=1", fontsize=14)

            # Electron - Linear scale (time vertical)
            electron_amp_da.plot(
                ax=axes[0, 0], y="t", x="hermite_mode", cmap="viridis", cbar_kwargs={"label": r"$|C_{n,m,p}|$"}
            )
            axes[0, 0].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
            axes[0, 0].set_xlabel("Hermite Mode Index")
            axes[0, 0].set_title("Electron Species (Linear Scale)")

            # Electron - Log scale (time vertical)
            (np.log10(electron_amp_da + 1e-20)).plot(
                ax=axes[0, 1],
                y="t",
                x="hermite_mode",
                cmap="viridis",
                vmin=-10,
                vmax=0,
                cbar_kwargs={"label": r"$\log_{10}(|C_{n,m,p}|)$"},
            )
            axes[0, 1].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
            axes[0, 1].set_xlabel("Hermite Mode Index")
            axes[0, 1].set_title("Electron Species (Log Scale)")

            # Ion - Linear scale (time vertical)
            ion_amp_da.plot(
                ax=axes[1, 0], y="t", x="hermite_mode", cmap="viridis", cbar_kwargs={"label": r"$|C_{n,m,p}|$"}
            )
            axes[1, 0].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
            axes[1, 0].set_xlabel("Hermite Mode Index")
            axes[1, 0].set_title("Ion Species (Linear Scale)")

            # Ion - Log scale (time vertical)
            (np.log10(ion_amp_da + 1e-20)).plot(
                ax=axes[1, 1],
                y="t",
                x="hermite_mode",
                cmap="viridis",
                vmin=-10,
                vmax=0,
                cbar_kwargs={"label": r"$\log_{10}(|C_{n,m,p}|)$"},
            )
            axes[1, 1].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
            axes[1, 1].set_xlabel("Hermite Mode Index")
            axes[1, 1].set_title("Ion Species (Log Scale)")

            plt.tight_layout()
            plt.savefig(os.path.join(td, "plots", "hermite_mode_amplitudes_kx1.png"), bbox_inches="tight")
            plt.close()

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
        self._plot_hermite_modes_at_k(Ck, t_array, Nn, Nm, Np, Nx, Ny, Nz, td)

    def _plot_field_mode_amplitudes(self, Fk, t_array, Nx, Ny, Nz, td: str) -> None:
        """Plot field mode amplitudes for k=1...5."""
        import matplotlib.pyplot as plt
        import numpy as np

        # k=0 for other dimensions (1D problem in x)
        idx_k0 = 0

        # Plot electric field modes
        fig, axes = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
        fig.suptitle("Electric Field Mode Amplitudes (k=1...5)", fontsize=14)

        field_components = ["Ex", "Ey", "Ez"]
        for comp_idx, (ax, comp_name) in enumerate(zip(axes, field_components, strict=True)):
            for k_mode in range(1, 6):
                if k_mode < Nx:
                    mode_amplitude = np.abs(Fk[:, comp_idx, idx_k0, k_mode, idx_k0])
                    ax.plot(t_array, mode_amplitude, label=f"k={k_mode}", linewidth=2)

            ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
            ax.set_ylabel(f"|{comp_name}|")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{comp_name} Mode Amplitudes")

        plt.tight_layout()
        plt.savefig(os.path.join(td, "plots", "field_mode_amplitudes.png"), bbox_inches="tight")
        plt.close()

        # Plot magnetic field modes
        fig, axes = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
        fig.suptitle("Magnetic Field Mode Amplitudes (k=1...5)", fontsize=14)

        field_components = ["Bx", "By", "Bz"]
        for comp_idx, (ax, comp_name) in enumerate(zip(axes, field_components, strict=True)):
            for k_mode in range(1, 6):
                if k_mode < Nx:
                    mode_amplitude = np.abs(Fk[:, 3 + comp_idx, idx_k0, k_mode, idx_k0])
                    ax.plot(t_array, mode_amplitude, label=f"k={k_mode}", linewidth=2)

            ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
            ax.set_ylabel(f"|{comp_name}|")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{comp_name} Mode Amplitudes")

        plt.tight_layout()
        plt.savefig(os.path.join(td, "plots", "magnetic_field_mode_amplitudes.png"), bbox_inches="tight")
        plt.close()

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
        self._plot_field_mode_amplitudes(Fk, t_array, Nx, Ny, Nz, td)

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

    def _plot_fields_spacetime(self, fields_xr: xr.Dataset, td: str) -> None:
        """
        Create spacetime plots for electromagnetic fields.

        Args:
            fields_xr: xarray Dataset containing field data with (t, x) coordinates
            td: Temporary directory path
        """
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "fields")

        for field_name, field_data in fields_xr.items():
            # Spacetime plot
            field_data.plot()
            plt.title(f"{field_name} Spacetime")
            plt.savefig(os.path.join(plots_dir, f"spacetime-{field_name}.png"), bbox_inches="tight")
            plt.close()

    def _plot_fields_lineouts(self, fields_xr: xr.Dataset, td: str, n_slices: int = 6) -> None:
        """
        Create facet grid plots showing field snapshots at multiple times.

        Args:
            fields_xr: xarray Dataset containing field data with (t, x) coordinates
            td: Temporary directory path
            n_slices: Number of time slices to show (default: 6)
        """
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "fields", "lineouts")

        for field_name, field_data in fields_xr.items():
            # Calculate time slice indices
            nt = field_data.coords["t"].size
            t_skip = max(1, nt // n_slices)
            tslice = slice(0, None, t_skip)

            # Create facet plot
            field_data[tslice].T.plot(col="t", col_wrap=3)
            plt.savefig(os.path.join(plots_dir, f"{field_name}.png"), bbox_inches="tight")
            plt.close()

    def _plot_moments_spacetime(self, moments_xr: xr.Dataset, td: str) -> None:
        """
        Create spacetime plots for distribution moments.

        Args:
            moments_xr: xarray Dataset containing moments with (t, x) coordinates
            td: Temporary directory path
        """
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "moments")
        os.makedirs(plots_dir, exist_ok=True)

        for name, data in moments_xr.items():
            if "x" not in data.dims:
                continue
            data.plot()
            plt.title(f"{name} Spacetime")
            plt.savefig(os.path.join(plots_dir, f"spacetime-{name}.png"), bbox_inches="tight")
            plt.close()

    def _plot_moments_lineouts(self, moments_xr: xr.Dataset, td: str, n_slices: int = 6) -> None:
        """
        Create facet grid plots of moments at multiple times.

        Args:
            moments_xr: xarray Dataset containing moments with (t, x) coordinates
            td: Temporary directory path
            n_slices: Number of time slices to show
        """
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "moments", "lineouts")
        os.makedirs(plots_dir, exist_ok=True)

        for name, data in moments_xr.items():
            if "x" not in data.dims:
                continue
            nt = data.coords["t"].size
            t_skip = max(1, nt // n_slices)
            tslice = slice(0, None, t_skip)

            data[tslice].T.plot(col="t", col_wrap=3)
            plt.savefig(os.path.join(plots_dir, f"{name}.png"), bbox_inches="tight")
            plt.close()

    def _plot_distribution_facets(self, dist_xr: xr.Dataset, td: str, n_timesteps: int = 6) -> None:
        """
        Create facet plots of distribution in (kx, hermite_mode) space for multiple timesteps.

        Args:
            dist_xr: xarray Dataset containing Ck with dimensions (t, hermite_mode, ky, kx, kz)
            td: Temporary directory path
            n_timesteps: Number of timesteps to show (default: 6)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        plots_dir = os.path.join(td, "plots", "distributions")

        Ck = dist_xr["Ck"]

        # Get center indices for y and z (assume 1D in x)
        center_y = int((Ck.coords["ky"].size - 1) / 2)
        center_z = int((Ck.coords["kz"].size - 1) / 2)

        # Slice to get (t, hermite_mode, kx) at center ky, kz
        Ck_slice = Ck[:, :, center_y, :, center_z]

        # Select timesteps
        nt = Ck_slice.coords["t"].size
        t_indices = np.linspace(0, nt - 1, n_timesteps, dtype=int)
        Ck_selected = Ck_slice.isel(t=t_indices)

        # Plot amplitude (log scale)
        amplitude = np.log10(np.abs(Ck_selected) + 1e-20)
        amplitude.plot(
            x="kx", y="hermite_mode", col="t", col_wrap=3, cmap="viridis", cbar_kwargs={"label": r"$\log_{10}(|C_k|)$"}
        )
        plt.savefig(os.path.join(plots_dir, "Ck_amplitude_facets.png"), bbox_inches="tight")
        plt.close()

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

                    print(fields_data)
                    fields_xr_dict = {
                        name: xr.DataArray(val, coords=[("t", t_array), axis], name=name)
                        for name, val in fields_data.items()
                    }
                    fields_xr = xr.Dataset(fields_xr_dict)
                    fields_xr.to_netcdf(os.path.join(binary_dir, f"fields-{axis_name}-t={round(t_array[-1], 4)}.nc"))
                    saved_datasets[f"fields_{axis_name}"] = fields_xr
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
                    self._plot_fields_spacetime(fields_xr, td)
                    self._plot_fields_lineouts(fields_xr, td)

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

                self._plot_moments_spacetime(moments_xr, td)
                self._plot_moments_lineouts(moments_xr, td)

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
