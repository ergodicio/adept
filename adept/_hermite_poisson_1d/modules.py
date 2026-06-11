"""Base ADEPTModule for the 1D Hermite-Poisson solver.

Normalization: skin-depth units (x0 = c/wp0, t0 = 1/wp0, v0 = c).
  - alpha_e = vth_e / c,  alpha_i = vth_i / c
  - c_light = 1.0  (speed of light normalized to 1)

State dict:
  Ck_electrons: (Nn_e, Nx) complex, stored as float64 view
  Ck_ions:      (Nn_i, Nx) complex, stored as float64 view
  a:            (Nx+2,)  vector potential with ghost cells
  prev_a:       (Nx+2,)  previous-step a
  e:            (Nx,)    electrostatic field (diagnostics)
  da:           (Nx+2,)  last wave-equation source term

Config keys expected in cfg["physics"]:
  alpha_e (float), alpha_i (float), mi_me (float), Lx (float),
  nu (float), Omega_ce_tau (float), static_ions (bool),
  n0_e (float, default 1.0), n0_i (float, default 1.0), c_light (float, default 1.0)

Config keys expected in cfg["grid"]:
  Nn (int, electron Hermite modes), Ni (int, ion Hermite modes, default 2),
  Nx (int, optional — computed from Lx if missing), tmax (float), dt (float, optional)
"""

import os
import sys

import numpy as np
from diffrax import ConstantStepSize, NoProgressMeter, ODETerm, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import numpy as jnp

from adept._base_ import ADEPTModule, Stepper
from adept._hermite_poisson_1d.storage import get_save_quantities, store_ck_timeseries, store_fields_timeseries
from adept._hermite_poisson_1d.vector_field import (
    CombinedLinearExp1D,
    DiagonalExp1D,
    FreeStreamingExp1D,
    HermitePoisson1DVectorField,
    LinearExp1D,
    PoissonSolver1D,
    TransverseWaveDriver,
)
from adept._vlasov1d.solvers.pushers.field import WaveSolver


def _safe_col_1d(Nn: int) -> jnp.ndarray:
    """Hypercollisional damping vector: col[n] = n*(n-1)*(n-2) / ((Nn-1)(Nn-2)(Nn-3))."""
    n = jnp.arange(Nn, dtype=jnp.float64)
    term = n * (n - 1) * (n - 2)
    denom = (Nn - 1) * (Nn - 2) * (Nn - 3) if Nn > 3 else 1.0
    return jnp.where(Nn > 3, term / denom, jnp.zeros(Nn, dtype=jnp.float64))


class BaseHermitePoisson1D(ADEPTModule):
    """1D Hermite-Poisson base solver in skin-depth units."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Lifecycle: write_units
    # ------------------------------------------------------------------

    def write_units(self) -> dict:
        """Return normalization constants already present in cfg["units"]["derived"]."""
        return self.cfg.get("units", {}).get("derived", {})

    # ------------------------------------------------------------------
    # Lifecycle: get_derived_quantities
    # ------------------------------------------------------------------

    def get_derived_quantities(self) -> None:
        """Compute dt, nt, max_steps from wave-CFL or user dt."""
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        Lx = float(physics["Lx"])
        c_light = float(physics.get("c_light", 1.0))

        # Grid resolution
        if "Nx" not in grid or grid["Nx"] is None:
            # Default: ~1 spatial cell per unit length, minimum 32
            Nx = max(32, int(np.ceil(Lx)))
            grid["Nx"] = Nx
        Nx = int(grid["Nx"])
        dx = Lx / Nx
        grid["dx"] = dx

        tmax = float(grid["tmax"])

        # Timestep: limited by wave CFL if c_light > 0
        dt_cfl = 0.9 * dx / c_light if c_light > 0 else float("inf")
        user_dt = grid.get("dt", None)
        if isinstance(user_dt, (int, float)) and user_dt > 0:
            dt = float(user_dt)
        else:
            dt = dt_cfl
        grid["dt"] = dt

        nt = int(tmax / dt) + 1
        grid["nt"] = nt
        grid["tmax"] = dt * nt  # snap tmax to exact multiple
        grid["max_steps"] = nt + 4

        # Populate save t defaults
        for save_cfg in self.cfg.get("save", {}).values():
            if isinstance(save_cfg, dict) and "t" in save_cfg:
                t_cfg = save_cfg["t"]
                t_cfg.setdefault("tmin", 0.0)
                t_cfg.setdefault("tmax", grid["tmax"])

        self.cfg["grid"] = grid

    # ------------------------------------------------------------------
    # Lifecycle: get_solver_quantities
    # ------------------------------------------------------------------

    def get_solver_quantities(self) -> None:
        """Build k-space arrays, collision matrices, and sponge profiles."""
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        Lx = float(physics["Lx"])
        Nx = int(grid["Nx"])
        dx = float(grid["dx"])

        alpha_e = float(physics["alpha_e"])
        alpha_i = float(physics["alpha_i"])

        Nn_e = int(grid.get("Nn", 128))
        Nn_i = int(grid.get("Ni", 2))

        # Wavenumber arrays (physical units: 1/(c/wp0))
        kx_1d = jnp.fft.fftfreq(Nx) * Nx * 2 * jnp.pi / Lx  # shape (Nx,)
        one_over_kx = np.zeros(Nx, dtype=np.float64)
        one_over_kx[1:] = 1.0 / np.asarray(kx_1d[1:])
        one_over_kx = jnp.array(one_over_kx)
        kx_sq = kx_1d ** 2

        # Ladder operators
        sqrt_n_minus_e = jnp.sqrt(jnp.arange(Nn_e, dtype=jnp.float64))
        sqrt_n_minus_i = jnp.sqrt(jnp.arange(Nn_i, dtype=jnp.float64))

        # Collision vectors (hypercollisional)
        col_e = _safe_col_1d(Nn_e)
        col_i = _safe_col_1d(Nn_i)

        # Real-space x-axis (no ghost cells)
        x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
        # Extended x with ghost cells for wave solver
        x_a = jnp.concatenate([jnp.array([x[0] - dx]), x, jnp.array([x[-1] + dx])])

        grid["kx_1d"] = kx_1d
        grid["one_over_kx"] = one_over_kx
        grid["kx_sq"] = kx_sq
        grid["sqrt_n_minus_e"] = sqrt_n_minus_e
        grid["sqrt_n_minus_i"] = sqrt_n_minus_i
        grid["col_e"] = col_e
        grid["col_i"] = col_i
        grid["x"] = x
        grid["x_a"] = x_a
        grid["Nn_e"] = Nn_e
        grid["Nn_i"] = Nn_i

        self.cfg["grid"] = grid

    # ------------------------------------------------------------------
    # Lifecycle: init_state_and_args
    # ------------------------------------------------------------------

    def init_state_and_args(self) -> None:
        """Initialize state dict and args."""
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        alpha_e = float(physics["alpha_e"])
        alpha_i = float(physics["alpha_i"])
        Nn_e = int(grid["Nn_e"])
        Nn_i = int(grid["Nn_i"])
        Nx = int(grid["Nx"])
        n0_e = float(physics.get("n0_e", 1.0))
        n0_i = float(physics.get("n0_i", 1.0))

        # Initialize Hermite coefficient arrays
        # C_0(kx) = FFT(n(x) / alpha^3); equilibrium: n(x) = n0 → C_0(kx=0) = n0/alpha^3
        Ck_e = jnp.zeros((Nn_e, Nx), dtype=jnp.complex128)
        Ck_i = jnp.zeros((Nn_i, Nx), dtype=jnp.complex128)
        Ck_e = Ck_e.at[0, 0].set(n0_e / (alpha_e ** 3))
        Ck_i = Ck_i.at[0, 0].set(n0_i / (alpha_i ** 3))

        # Optional single-mode electron density perturbation (for dispersion tests):
        # density.perturbation: {mode: int m, amplitude: float eps}
        #   → n_e(x) = n0 * (1 + eps * cos(2*pi*m*x/Lx))
        pert = self.cfg.get("density", {}).get("perturbation", None)
        if pert:
            mode = int(pert["mode"])
            eps = float(pert["amplitude"])
            half = 0.5 * eps * n0_e / (alpha_e ** 3)
            Ck_e = Ck_e.at[0, mode].set(half)
            Ck_e = Ck_e.at[0, -mode].set(half)

        self.state = {
            "Ck_electrons": Ck_e.view(jnp.float64),
            "Ck_ions": Ck_i.view(jnp.float64),
            "a": jnp.zeros(Nx + 2),
            "prev_a": jnp.zeros(Nx + 2),
            "e": jnp.zeros(Nx),
            "da": jnp.zeros(Nx + 2),
        }
        self.args = {}

    # ------------------------------------------------------------------
    # Lifecycle: init_diffeqsolve
    # ------------------------------------------------------------------

    def init_diffeqsolve(self) -> None:
        """Assemble the HermitePoisson1DVectorField and configure diffeqsolve."""
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        alpha_e = float(physics["alpha_e"])
        alpha_i = float(physics["alpha_i"])
        mi_me = float(physics.get("mi_me", 1836.0))
        nu = float(physics.get("nu", 0.0))
        Omega_ce_tau = float(physics.get("Omega_ce_tau", 1.0))
        c_light = float(physics.get("c_light", 1.0))
        static_ions = bool(physics.get("static_ions", False))
        Lx = float(physics["Lx"])

        Nn_e = int(grid["Nn_e"])
        Nn_i = int(grid["Nn_i"])
        Nx = int(grid["Nx"])
        dx = float(grid["dx"])
        dt = float(grid["dt"])

        kx_1d = grid["kx_1d"]
        one_over_kx = grid["one_over_kx"]
        kx_sq = grid["kx_sq"]
        sqrt_n_minus_e = grid["sqrt_n_minus_e"]
        sqrt_n_minus_i = grid["sqrt_n_minus_i"]
        col_e = grid["col_e"]
        col_i = grid["col_i"]
        x_a = grid["x_a"]

        # Sponge profiles (optional)
        sponge_plasma = grid.get("sponge_plasma", None)
        sponge_fields = grid.get("sponge_fields", None)

        # Static ion density (for Poisson when static_ions=True)
        if static_ions:
            Ck_i = self.state["Ck_ions"].view(jnp.complex128)
            n0_i = float(physics.get("n0_i", 1.0))
            # Build ion density profile from initial Ck_i
            n_i_prof = (alpha_i ** 3) * jnp.fft.ifft(Ck_i[0], norm="forward").real
            static_ion_density = n_i_prof
        else:
            static_ion_density = None

        # Build per-species linear exponential operators
        u_e = 0.0  # no drift in base class
        u_i = 0.0
        D = float(physics.get("D", 0.0))

        # Hou-Li filter: exp(-strength * (n/Nn)^order * s) per Lawson substep.
        # Read from drivers.hermite_filter (same key used by kinetic_srs subclass).
        hl_cfg = self.cfg.get("drivers", {}).get("hermite_filter", {})
        if hl_cfg and hl_cfg.get("enabled", True):
            hl_strength = float(hl_cfg.get("strength", 0.0))
            hl_order = float(hl_cfg.get("order", 8))
            n_e_norm = jnp.arange(Nn_e, dtype=jnp.float64) / max(Nn_e - 1, 1)
            n_i_norm = jnp.arange(Nn_i, dtype=jnp.float64) / max(Nn_i - 1, 1)
            hou_li_col_e = n_e_norm ** hl_order
            hou_li_col_i = n_i_norm ** hl_order
        else:
            hl_strength = 0.0
            hou_li_col_e = None
            hou_li_col_i = None

        free_stream_e = FreeStreamingExp1D(Nn_e, alpha_e, u_e, Lx, kx_1d)
        free_stream_i = FreeStreamingExp1D(Nn_i, alpha_i, u_i, Lx, kx_1d)
        diag_e = DiagonalExp1D(nu=nu, col_1d=col_e, D=D, kx_sq_1d=kx_sq,
                               hou_li_strength=hl_strength, hou_li_col_1d=hou_li_col_e)
        diag_i = DiagonalExp1D(nu=nu, col_1d=col_i, D=D, kx_sq_1d=kx_sq,
                               hou_li_strength=hl_strength, hou_li_col_1d=hou_li_col_i)
        linear_e = LinearExp1D(free_stream_e, diag_e)
        linear_i = LinearExp1D(free_stream_i, diag_i)
        combined_exp = CombinedLinearExp1D(linear_e, linear_i, static_ions=static_ions)

        # Poisson solver
        poisson = PoissonSolver1D(
            one_over_kx=one_over_kx,
            alpha_e=alpha_e,
            alpha_i=alpha_i,
            static_ion_density=static_ion_density,
        )

        # Wave solver (c_light > 0 → EM waves; = 0 → frozen a)
        wave_solver = WaveSolver(c=c_light, dx=dx, dt=dt)

        # Transverse wave source driver
        ey_cfg = self.cfg.get("drivers", {}).get("ey", {})
        ey_driver = TransverseWaveDriver(x_a, ey_cfg)

        # Assemble top-level vector field (discrete-map via Stepper)
        vector_field = HermitePoisson1DVectorField(
            combined_exp=combined_exp,
            poisson=poisson,
            wave_solver=wave_solver,
            ey_driver=ey_driver,
            sqrt_n_minus_e=sqrt_n_minus_e,
            sqrt_n_minus_i=sqrt_n_minus_i,
            alpha_e=alpha_e,
            alpha_i=alpha_i,
            q_e=-1.0,
            q_i=1.0,
            mi_me=mi_me,
            Omega_ce_tau=Omega_ce_tau,
            dx=dx,
            dt=dt,
            static_ions=static_ions,
            sponge_plasma=sponge_plasma,
            sponge_fields=sponge_fields,
        )

        # Configure save quantities
        self.cfg = get_save_quantities(self.cfg)

        tmax = float(grid["tmax"])
        max_steps = int(grid["max_steps"])

        subsaves = {}
        for k, v in self.cfg["save"].items():
            if isinstance(v, dict) and "t" in v and "func" in v:
                subsaves[k] = SubSaveAt(ts=v["t"]["ax"], fn=v["func"])

        self.time_quantities = {"t0": 0.0, "t1": tmax, "max_steps": max_steps}
        self.diffeqsolve_quants = {
            "terms": ODETerm(vector_field),
            "solver": Stepper(),
            "saveat": {"subs": subsaves},
        }

    # ------------------------------------------------------------------
    # Lifecycle: __call__
    # ------------------------------------------------------------------

    def __call__(self, trainable_modules: dict, args: dict | None = None) -> dict:
        if args is None:
            args = self.args
        grid = self.cfg["grid"]

        sol = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            stepsize_controller=ConstantStepSize(),
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            dt0=float(grid["dt"]),
            y0=self.state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
            max_steps=self.time_quantities["max_steps"],
            progress_meter=TqdmProgressMeter() if sys.stdout.isatty() else NoProgressMeter(),
        )
        return {"solver result": sol}

    # ------------------------------------------------------------------
    # Lifecycle: post_process
    # ------------------------------------------------------------------

    def post_process(self, run_output: dict, td: str) -> dict:
        """Save outputs to netCDF and produce basic spacetime plots."""
        import matplotlib.pyplot as plt

        sol = run_output["solver result"]
        binary_dir = os.path.join(td, "binary")
        plots_dir = os.path.join(td, "plots")
        os.makedirs(binary_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        grid = self.cfg["grid"]
        x = np.asarray(grid["x"])

        datasets = {}
        metrics = {"simulation_completed": True}

        if hasattr(sol, "stats") and sol.stats is not None:
            for stat_key in ("num_steps", "num_accepted_steps", "num_rejected_steps"):
                if stat_key in sol.stats:
                    metrics[stat_key] = int(sol.stats[stat_key])

        for k in sol.ys.keys():
            t_arr = np.asarray(sol.ts[k])
            data = sol.ys[k]

            if k == "fields":
                fields_dict = {name: np.asarray(arr) for name, arr in data.items()}
                ds = store_fields_timeseries(self.cfg, fields_dict, t_arr, binary_dir, x)
                datasets["fields"] = ds

                # Spacetime plots. Sanitize inf -> nan (matplotlib masks nan but
                # its log tickers crash on inf) and never let a plotting failure
                # take down the netCDF/metric upload for a blown-up run — those
                # are exactly the runs whose diagnostics we need most.
                for field_name, arr in fields_dict.items():
                    if arr.ndim == 2:
                        try:
                            arr_plot = np.where(np.isfinite(arr), arr, np.nan)
                            fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
                            im = ax.pcolormesh(x, t_arr, arr_plot, shading="auto", cmap="RdBu_r")
                            plt.colorbar(im, ax=ax)
                            ax.set_xlabel("x [norm]")
                            ax.set_ylabel("t [norm]")
                            ax.set_title(field_name)
                            fig.savefig(os.path.join(plots_dir, f"spacetime-{field_name}.png"), bbox_inches="tight")
                        except Exception as e:
                            print(f"post_process: spacetime plot for {field_name} failed: {e}")
                        finally:
                            plt.close("all")

            elif k in ("hermite", "distribution"):
                for species, ck_arr in data.items():
                    Ck = np.asarray(ck_arr)
                    ds = store_ck_timeseries(self.cfg, species, Ck, t_arr, binary_dir)
                    datasets[f"hermite_{species}"] = ds

            elif k == "default":
                scalars = {name: np.asarray(arr) for name, arr in data.items()}
                t_arr_def = t_arr

                for name, arr in scalars.items():
                    if arr.ndim == 1:
                        final_val = float(arr[-1])
                        metrics[f"final_{name}"] = final_val if np.isfinite(final_val) else float("nan")
                        try:
                            arr_plot = np.where(np.isfinite(arr), arr, np.nan)
                            fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                            axes[0].plot(t_arr_def, arr_plot)
                            axes[0].set_xlabel("t"); axes[0].set_ylabel(name); axes[0].grid(alpha=0.3)
                            axes[1].semilogy(t_arr_def, np.abs(arr_plot) + 1e-30)
                            axes[1].set_xlabel("t"); axes[1].set_ylabel(f"|{name}|"); axes[1].grid(alpha=0.3)
                            fig.savefig(os.path.join(plots_dir, f"scalar-{name}.png"), bbox_inches="tight")
                        except Exception as e:
                            print(f"post_process: scalar plot for {name} failed: {e}")
                        finally:
                            plt.close("all")

        if "default" in sol.ts:
            metrics["n_timesteps"] = len(sol.ts["default"])

        return {"metrics": metrics, "datasets": datasets}
