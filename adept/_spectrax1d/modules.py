"""
Spectrax-1D Module: Wrapper for spectrax-based Hermite-Fourier Vlasov-Maxwell solver

This module wraps the external spectrax library to solve the Vlasov-Maxwell equations
using Hermite-Fourier decomposition. It provides ADEPT integration including MLflow
logging, unit management, and configuration handling.

The wrapper approach calls spectrax.simulation() directly rather than reimplementing
the solver in diffrax. This maintains compatibility with the external library while
leveraging ADEPT's experiment tracking infrastructure.
"""

from functools import partial

import pint
import xarray as xr
from diffrax import ConstantStepSize, ODETerm, PIDController, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import Array, jit
from jax import numpy as jnp
from spectrax import Hermite_Fourier_system, cross_product, initialize_simulation_parameters, plasma_current, simulation

from adept._base_ import ADEPTModule
from adept._spectrax1d.storage import (
    get_save_quantities,
    store_distribution_timeseries,
    store_fields_timeseries,
    store_scalars,
)


@partial(jit, static_argnames=["Nx", "Ny", "Nz"])
def _twothirds_mask(Ny: int, Nx: int, Nz: int):
    """Return a boolean mask in fftshifted (zero-centered) k-ordering that keeps |k|<=N//3 in each dim."""

    def centered_modes(N):
        # integer mode numbers in fftshift ordering: [-N//2, ..., -1, 0, 1, ..., N//2-1]
        k = jnp.fft.fftfreq(N) * N
        return jnp.fft.fftshift(k)

    ky = centered_modes(Ny)[:, None, None]
    kx = centered_modes(Nx)[None, :, None]
    kz = centered_modes(Nz)[None, None, :]

    # cutoffs (keep indices with |k| <= floor(N/3)); if N<3 this naturally keeps only k=0
    cy = Ny // 3
    cx = Nx // 3
    cz = Nz // 3

    return (jnp.abs(ky) <= cy) & (jnp.abs(kx) <= cx) & (jnp.abs(kz) <= cz)


@partial(jit, static_argnames=["Nx", "Ny", "Nz", "Nn", "Nm", "Np", "Ns"])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, t, Ck_Fk, args):
    """
    Right-hand side for the coupled Vlasov-Maxwell system expressed in spectral form.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of retained Fourier modes per spatial dimension.
    Nn, Nm, Np : int
        Number of Hermite modes per velocity-space dimension.
    Ns : int
        Number of species.
    t : float
        Integration time (unused but required by Diffrax interface).
    Ck_Fk : jnp.ndarray
        Flattened state vector containing concatenated Hermite coefficients followed
        by electromagnetic field coefficients.
    args : tuple
        Cached parameter tuple produced in `simulation` providing physical constants,
        grids, and helper arrays.

    Returns
    -------
    jnp.ndarray
        Flattened derivative vector matching the shape of `Ck_Fk`.
    """

    (
        qs,
        nu,
        D,
        Omega_cs,
        alpha_s,
        u_s,
        Lx,
        Ly,
        Lz,
        kx_grid,
        ky_grid,
        kz_grid,
        k2_grid,
        nabla,
        col,
        sqrt_n_plus,
        sqrt_n_minus,
        sqrt_m_plus,
        sqrt_m_minus,
        sqrt_p_plus,
        sqrt_p_minus,
    ) = args[7:]

    total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz
    Ck = Ck_Fk[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz)

    # Build the 2/3 mask once per call (JIT will constant-fold it since Nx/Ny/Nz are static)
    mask23 = _twothirds_mask(Ny, Nx, Nz)

    F = jnp.fft.ifftn(jnp.fft.ifftshift(Fk * mask23, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward")
    C = jnp.fft.ifftn(jnp.fft.ifftshift(Ck * mask23, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward")

    dCk_s_dt = Hermite_Fourier_system(
        Ck,
        C,
        F,
        kx_grid,
        ky_grid,
        kz_grid,
        k2_grid,
        col,
        sqrt_n_plus,
        sqrt_n_minus,
        sqrt_m_plus,
        sqrt_m_minus,
        sqrt_p_plus,
        sqrt_p_minus,
        Lx,
        Ly,
        Lz,
        nu,
        D,
        alpha_s,
        u_s,
        qs,
        Omega_cs,
        Nn,
        Nm,
        Np,
        Ns,
        mask23=mask23,
    )

    # nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])
    dBk_dt = -1j * cross_product(nabla, Fk[:3])
    driver = jnp.zeros_like(dBk_dt)
    dex = jnp.exp(-(((t - 35) / 14) ** 8)) * 0.00000 * jnp.exp(-1j * 1.25 * t)
    driver = driver.at[0, 0, 1, 0].set(dex)
    current = plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
    dEk_dt = 1j * cross_product(nabla, Fk[3:]) - current / Omega_cs[0] + driver

    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)
    dy_dt = jnp.concatenate([dCk_s_dt.reshape(-1), dFk_dt.reshape(-1)])
    return dy_dt


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
        import jax.numpy as jnp

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

        Two modes:
        1. If cfg["spectrax_toml_file"] is provided: Load from TOML using spectrax.load_parameters()
        2. Otherwise: Build from YAML config (for ADEPT-native workflow)

        Reference: adept/_vlasov1d/modules.py lines 102-167
        """
        import os

        import jax.numpy as jnp

        # Check if TOML file is specified
        toml_file = self.cfg.get("spectrax_toml_file", None)

        if toml_file is not None:
            # Mode 1: Load from TOML file using spectrax
            from spectrax import load_parameters

            # Resolve path relative to config file if not absolute
            if not os.path.isabs(toml_file):
                # If running from ADEPT root, resolve relative to that
                toml_file = os.path.abspath(toml_file)

            print(f"Loading spectrax parameters from TOML: {toml_file}")
            input_parameters, solver_parameters = load_parameters(toml_file)

        else:
            # Mode 2: Build from YAML config
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

            # Build solver_parameters dict
            solver_name = grid.get("solver", "Dopri8")
            adaptive_time_step = grid.get("adaptive_time_step", True)

            solver_parameters = {
                "Nx": int(grid["Nx"]),
                "Ny": int(grid["Ny"]),
                "Nz": int(grid["Nz"]),
                "Nn": int(grid["Nn"]),
                "Nm": int(grid["Nm"]),
                "Np": int(grid["Np"]),
                "Ns": int(grid["Ns"]),
                "timesteps": int(grid["nt"]),
                "dt": float(grid["dt"]),
                "adaptive_time_step": adaptive_time_step,
                "solver": self._get_solver_instance(solver_name, input_parameters["ode_tolerance"]),
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

        # Set field perturbation at ±k modes
        if n != 0 and Omega_cs[0] != 0:
            amplitude = dn * L / (4 * jnp.pi * n * Omega_cs[0])
            Fk_0 = Fk_0.at[
                E_field_component, int((Ny - 1) / 2 - ny), int((Nx - 1) / 2 - nx), int((Nz - 1) / 2 - nz)
            ].set(amplitude)
            Fk_0 = Fk_0.at[
                E_field_component, int((Ny - 1) / 2 + ny), int((Nx - 1) / 2 + nx), int((Nz - 1) / 2 + nz)
            ].set(amplitude)

        input_parameters["Fk_0"] = Fk_0

        # Initialize Ck_0 (distribution function Hermite-Fourier components)
        # Shape: (2*Nn*Nm*Np, Ny, Nx, Nz) for [electron, ion]
        Nn, Nm, Np = solver_parameters["Nn"], solver_parameters["Nm"], solver_parameters["Np"]
        Ck_0 = jnp.zeros((2 * Nn * Nm * Np, Ny, Nx, Nz), dtype=jnp.complex128)

        # Electron distribution (first Nn*Nm*Np components)
        # Use alpha_s[0] for electron thermal velocity (in x-direction for 1D)
        alpha_e = input_parameters["alpha_s"][0]
        Ce0_mk = 0 + 1j * (1 / (2 * alpha_e**3)) * dn  # -k mode
        Ce0_0 = 1 / (alpha_e**3) + 0 * 1j  # Equilibrium
        Ce0_k = 0 - 1j * (1 / (2 * alpha_e**3)) * dn  # +k mode

        Ck_0 = Ck_0.at[0, int((Ny - 1) / 2 - ny), int((Nx - 1) / 2 - nx), int((Nz - 1) / 2 - nz)].set(Ce0_mk)
        Ck_0 = Ck_0.at[0, int((Ny - 1) / 2), int((Nx - 1) / 2), int((Nz - 1) / 2)].set(Ce0_0)
        Ck_0 = Ck_0.at[0, int((Ny - 1) / 2 + ny), int((Nx - 1) / 2 + nx), int((Nz - 1) / 2 + nz)].set(Ce0_k)

        # Ion distribution (second Nn*Nm*Np components)
        # Use alpha_s[3] for ion thermal velocity (first component in ion triplet)
        alpha_i = input_parameters["alpha_s"][3]
        Ci0_0 = 1 / (alpha_i**3) + 0 * 1j

        Ck_0 = Ck_0.at[Nn * Nm * Np, int((Ny - 1) / 2), int((Nx - 1) / 2), int((Nz - 1) / 2)].set(Ci0_0)

        input_parameters["Ck_0"] = Ck_0

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

        For the wrapper approach, this is minimal since spectrax manages its own state.
        The actual initial conditions are in cfg["spectrax_input"] from get_solver_quantities().

        Reference: adept/_vlasov1d/modules.py lines 169-193

        TODO:
        - Set self.state = {} (empty, spectrax manages state internally)
        - Set self.args = {} (or include any runtime parameters if needed)
        """
        self.state = {}  # Spectrax manages state internally
        self.args = {}

    def init_diffeqsolve(self) -> None:
        """
        Initialize solver configuration with save quantities.

        Sets up time quantities and save configuration for diagnostics.
        Uses get_save_quantities to process save config and create save functions.

        Reference: adept/_vlasov1d/modules.py lines 195-202
        """
        # Process save configuration to add time axes and save functions
        self.cfg = get_save_quantities(self.cfg)

        # Set time quantities (matches Vlasov1D pattern)
        self.time_quantities = {
            "t0": 0.0,
            "t1": self.cfg["grid"]["tmax"],
            "max_steps": self.cfg["grid"]["max_steps"],
        }

        # Store diffeqsolve configuration (now using diffrax for ODE solving)
        # SubSaveAt will be configured in __call__ based on cfg["save"]

    def __call__(self, trainable_modules: dict, args: dict | None = None) -> dict:
        """
        Run the spectrax simulation.

        This is the main execution method. It calls the external spectrax library's
        simulation() function with the parameters prepared in get_solver_quantities().

        Reference: adept/_vlasov1d/modules.py lines 203-219 (diffrax pattern)

        The wrapper pattern differs from native ADEPT modules:
        - Native modules call diffrax.diffeqsolve() with VectorField
        - This wrapper calls spectrax.simulation() directly
        - Output is NOT a diffrax.Solution object, but a spectrax output dict

        Args:
            trainable_modules: Dict of equinox.Module objects (unused in wrapper)
            args: Optional runtime arguments (unused in wrapper)

        Returns:
            dict: {"spectrax output": output_from_spectrax}
        """

        # from jax import block_until_ready

        # Get parameters prepared in get_solver_quantities()
        input_params = self.cfg["spectrax_input"]
        solver_params = self.cfg["spectrax_solver"]

        # Run spectrax simulation
        # output = block_until_ready(simulation(input_params, **solver_params))
        # **Initialize simulation parameters**
        # Get dimensions from solver_params
        Nx = solver_params["Nx"]
        Ny = solver_params["Ny"]
        Nz = solver_params["Nz"]
        Nn = solver_params["Nn"]
        Nm = solver_params["Nm"]
        Np = solver_params["Np"]
        Ns = solver_params["Ns"]
        nt = solver_params["timesteps"]
        dt = solver_params["dt"]

        parameters = initialize_simulation_parameters(
            input_params,
            Nx,
            Ny,
            Nz,
            Nn,
            Nm,
            Np,
            Ns,
            nt,
            dt,
        )

        # Combine initial conditions.
        initial_conditions = jnp.concatenate([parameters["Ck_0"].flatten(), parameters["Fk_0"].flatten()])

        # Arguments for the ODE system.
        args = (
            Nx,
            Ny,
            Nz,
            Nn,
            Nm,
            Np,
            Ns,
            parameters["qs"],
            parameters["nu"],
            parameters["D"],
            parameters["Omega_cs"],
            parameters["alpha_s"],
            parameters["u_s"],
            parameters["Lx"],
            parameters["Ly"],
            parameters["Lz"],
            parameters["kx_grid"],
            parameters["ky_grid"],
            parameters["kz_grid"],
            parameters["k2_grid"],
            parameters["nabla"],
            parameters["collision_matrix"],
            parameters["sqrt_n_plus"],
            parameters["sqrt_n_minus"],
            parameters["sqrt_m_plus"],
            parameters["sqrt_m_minus"],
            parameters["sqrt_p_plus"],
            parameters["sqrt_p_minus"],
        )

        controllers = {
            True: PIDController(
                rtol=parameters["ode_tolerance"],
                atol=parameters["ode_tolerance"],
            ),
            False: ConstantStepSize(),
        }
        adaptive_time_step = solver_params["adaptive_time_step"]
        stepsize_controller = controllers[adaptive_time_step]

        # Solve the ODE system with diagnostic saves
        ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns)

        # Build SubSaveAt dictionary for diagnostics
        # This allows saving interpolated quantities at specified times
        subsaves = {}
        for k, v in self.cfg["save"].items():
            if isinstance(v, dict) and "t" in v and "func" in v:
                subsaves[k] = SubSaveAt(ts=v["t"]["ax"], fn=v["func"])

        sol = diffeqsolve(
            ODETerm(ode_system_partial),
            solver=solver_params["solver"],
            stepsize_controller=stepsize_controller,
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            dt0=self.cfg["grid"]["dt"],
            y0=initial_conditions,
            args=args,
            saveat=SaveAt(subs=subsaves),
            max_steps=self.time_quantities["max_steps"],
            progress_meter=TqdmProgressMeter(),
        )

        # When using only SubSaveAt (no ts), sol.ys is a dict, not an array
        # The diagnostic quantities are saved in sol.ys[key] for each subsave key
        # We return the solution object directly for post-processing
        return {"solver result": sol}

    def _save_scalar_outputs(self, scalar_outputs: dict, td: str) -> None:
        """Save scalar outputs to a text file."""
        import os

        with open(os.path.join(td, "scalar_outputs.txt"), "w") as f:
            f.write("Spectrax Scalar Outputs\n")
            f.write("=" * 50 + "\n\n")
            for key, value in sorted(scalar_outputs.items()):
                f.write(f"{key}: {value}\n")

    def _plot_hermite_modes_at_k(self, Ck, t_array, Nn, Nm, Np, Nx, Ny, Nz, td: str) -> None:
        """Plot Hermite mode amplitudes at kx=1 Fourier mode."""
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        center_x_ck = (Nx - 1) // 2
        center_y_ck = (Ny - 1) // 2
        center_z_ck = (Nz - 1) // 2

        if center_x_ck + 1 < Nx:
            # Get kx=1 mode for all time steps and Hermite modes
            kx1_mode = Ck[:, :, center_y_ck, center_x_ck + 1, center_z_ck]

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
            plt.savefig(os.path.join(td, "plots", "hermite_mode_amplitudes_kx1.png"), dpi=150, bbox_inches="tight")
            plt.close()

    def _process_distribution_function(self, Ck, t_array, Nn, Nm, Np, Nx, Ny, Nz, td: str) -> None:
        """Process and save distribution function (Hermite-Fourier coefficients)."""
        import os

        import numpy as np
        import xarray as xr

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
        import os

        import matplotlib.pyplot as plt
        import numpy as np

        center_x = (Nx - 1) // 2
        center_y = (Ny - 1) // 2
        center_z = (Nz - 1) // 2

        # Plot electric field modes
        fig, axes = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
        fig.suptitle("Electric Field Mode Amplitudes (k=1...5)", fontsize=14)

        field_components = ["Ex", "Ey", "Ez"]
        for comp_idx, (ax, comp_name) in enumerate(zip(axes, field_components, strict=True)):
            for k_mode in range(1, 6):
                if center_x + k_mode < Nx:
                    mode_amplitude = np.abs(Fk[:, comp_idx, center_y, center_x + k_mode, center_z])
                    ax.plot(t_array, mode_amplitude, label=f"k={k_mode}", linewidth=2)

            ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
            ax.set_ylabel(f"|{comp_name}|")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{comp_name} Mode Amplitudes")

        plt.tight_layout()
        plt.savefig(os.path.join(td, "plots", "field_mode_amplitudes.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Plot magnetic field modes
        fig, axes = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
        fig.suptitle("Magnetic Field Mode Amplitudes (k=1...5)", fontsize=14)

        field_components = ["Bx", "By", "Bz"]
        for comp_idx, (ax, comp_name) in enumerate(zip(axes, field_components, strict=True)):
            for k_mode in range(1, 6):
                if center_x + k_mode < Nx:
                    mode_amplitude = np.abs(Fk[:, 3 + comp_idx, center_y, center_x + k_mode, center_z])
                    ax.plot(t_array, mode_amplitude, label=f"k={k_mode}", linewidth=2)

            ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
            ax.set_ylabel(f"|{comp_name}|")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{comp_name} Mode Amplitudes")

        plt.tight_layout()
        plt.savefig(os.path.join(td, "plots", "magnetic_field_mode_amplitudes.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def _process_electromagnetic_fields(self, Fk, t_array, Nx, Ny, Nz, td: str) -> None:
        """Process and save electromagnetic fields."""
        import os

        import numpy as np
        import xarray as xr

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
        import os

        import xarray as xr

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
        import os

        import xarray as xr

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
            field_real = np.real(np.fft.ifftn(Fk_timeseries[:, i, :, :, :], axes=(-3, -2, -1)))
            fields_real[name] = field_real

        return fields_real

    def _plot_fields_spacetime(self, fields_xr: xr.Dataset, td: str) -> None:
        """
        Create spacetime plots for electromagnetic fields.

        Args:
            fields_xr: xarray Dataset containing field data with (t, x) coordinates
            td: Temporary directory path
        """
        import os

        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "fields")

        for field_name, field_data in fields_xr.items():
            # Spacetime plot
            field_data.plot()
            plt.title(f"{field_name} Spacetime")
            plt.savefig(os.path.join(plots_dir, f"spacetime-{field_name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

    def _plot_fields_lineouts(self, fields_xr: xr.Dataset, td: str, n_slices: int = 6) -> None:
        """
        Create facet grid plots showing field snapshots at multiple times.

        Args:
            fields_xr: xarray Dataset containing field data with (t, x) coordinates
            td: Temporary directory path
            n_slices: Number of time slices to show (default: 6)
        """
        import os

        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "fields", "lineouts")

        for field_name, field_data in fields_xr.items():
            # Calculate time slice indices
            nt = field_data.coords["t"].size
            t_skip = max(1, nt // n_slices)
            tslice = slice(0, None, t_skip)

            # Create facet plot
            field_data[tslice].T.plot(col="t", col_wrap=3)
            plt.savefig(os.path.join(plots_dir, f"{field_name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

    def _plot_distribution_facets(self, dist_xr: xr.Dataset, td: str, n_timesteps: int = 6) -> None:
        """
        Create facet plots of distribution in (kx, hermite_mode) space for multiple timesteps.

        Args:
            dist_xr: xarray Dataset containing Ck with dimensions (t, hermite_mode, ky, kx, kz)
            td: Temporary directory path
            n_timesteps: Number of timesteps to show (default: 6)
        """
        import os

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
        plt.savefig(os.path.join(plots_dir, "Ck_amplitude_facets.png"), bbox_inches="tight", dpi=150)
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
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

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

            elif k == "fields_only":
                # Electromagnetic fields in Fourier space - convert to real space and plot
                Fk_array = np.asarray(sol.ys[k])  # Shape: (nt, 6, Ny, Nx, Nz)
                t_array = sol.ts[k]

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
                # Distribution function (Hermite coefficients)
                Ck_array = np.asarray(sol.ys[k])
                dist_xr = store_distribution_timeseries(self.cfg, Ck_array, sol.ts[k], binary_dir)
                saved_datasets["distribution"] = dist_xr

                # Create facet plots for distribution
                self._plot_distribution_facets(dist_xr, td)

        # Compute metrics for MLflow
        metrics = {
            "simulation_completed": True,
        }

        # Add timestep info from default save if available
        if "default" in sol.ts:
            metrics["n_timesteps"] = len(sol.ts["default"])

        # Add final values from scalar diagnostics if available
        if "scalars" in saved_datasets:
            for k, v in saved_datasets["scalars"].items():
                metrics[f"final_{k}"] = float(np.abs(v.values[-1]))

        return {"metrics": metrics, "datasets": saved_datasets}
