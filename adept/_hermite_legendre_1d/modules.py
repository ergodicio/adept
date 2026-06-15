"""Base ADEPTModule for the 1D mixed Hermite-Legendre Vlasov-Poisson solver.

Implements the method of Issan, Delzanno & Roytershteyn (arXiv:2606.12322). The
electron distribution is split f = f0 + df with an AW-Hermite expansion for the
near-Maxwellian bulk f0 and a Legendre expansion (on a bounded velocity window) for
the strongly non-Maxwellian part df. Electrostatic, single electron species with an
immobile neutralizing ion background; periodic in x (Fourier); explicit Lawson-RK4.

Normalization (paper sec 2.1): t*wpe, x/lambda_D, v/vthe.

Config keys in cfg["physics"]:
  Lx (float), alpha (float, Hermite scale), u (float, Hermite shift, default 0),
  v_a, v_b (float, Legendre velocity window), gamma (penalty, default 0.5),
  nu_H, nu_L (artificial collision rates, default 0), enforce_conservation (bool, default True)

Config keys in cfg["grid"]:
  Nx (int), Nh (int, Hermite modes), Nl (int, Legendre modes), tmax (float), dt (float, default 0.01)

Config cfg["initialization"]:
  type: "linear-advection" | "two-stream" | "bump-on-tail" | "custom"  (+ type-specific params)
"""

import os
import sys

import numpy as np
from diffrax import ConstantStepSize, NoProgressMeter, ODETerm, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import numpy as jnp

from adept._base_ import ADEPTModule, Stepper
from adept._hermite_legendre_1d.storage import (
    get_save_quantities,
    store_coeff_timeseries,
    store_fields_timeseries,
)
from adept._hermite_legendre_1d.vector_field import (
    CombinedLinearExp1D,
    DiagonalCollisionExp1D,
    HermiteLegendre1DVectorField,
    PoissonSolver1D,
    StreamingExp1D,
    hermite_legendre_coupling_vector,
    hermite_streaming_matrix,
    legendre_constants,
    safe_col,
)


def _density_profile(x: np.ndarray, Lx: float, base: float, eps: float, mode: int) -> np.ndarray:
    """base * (1 + eps cos(2*pi*mode*x/Lx))."""
    return base * (1.0 + eps * np.cos(2.0 * np.pi * mode * x / Lx))


def _project_legendre(g_v, Nl: int, v_a: float, v_b: float) -> np.ndarray:
    """Project a velocity profile g(v) onto the Legendre basis: B_m = (1/width) int g xi_m dv."""
    from adept._hermite_legendre_1d.vector_field import _legendre_basis_values

    width = v_b - v_a
    deg = max(4 * Nl, 400)
    nodes, weights = np.polynomial.legendre.leggauss(deg)
    v = 0.5 * width * nodes + 0.5 * (v_b + v_a)
    w = 0.5 * width * weights
    xi = _legendre_basis_values(Nl, v, v_a, v_b)  # (Nl, len(v))
    return (xi @ (g_v(v) * w)) / width


class BaseHermiteLegendre1D(ADEPTModule):
    """1D mixed Hermite-Legendre Vlasov-Poisson base solver (normalized units)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------------
    def write_units(self) -> dict:
        """Normalized units throughout; pass through any precomputed derived block."""
        return self.cfg.get("units", {}).get("derived", {})

    # ------------------------------------------------------------------
    def get_derived_quantities(self) -> None:
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        Lx = float(physics["Lx"])
        Nx = int(grid["Nx"])
        grid["dx"] = Lx / Nx

        tmax = float(grid["tmax"])
        dt = float(grid.get("dt", 0.01))
        nt = round(tmax / dt)
        grid["dt"] = dt
        grid["nt"] = nt
        grid["tmax"] = dt * nt  # snap to exact multiple
        grid["max_steps"] = nt + 4

        for save_cfg in self.cfg.get("save", {}).values():
            if isinstance(save_cfg, dict) and "t" in save_cfg:
                t_cfg = save_cfg["t"]
                t_cfg.setdefault("tmin", 0.0)
                t_cfg.setdefault("tmax", grid["tmax"])

        self.cfg["grid"] = grid

    # ------------------------------------------------------------------
    def get_solver_quantities(self) -> None:
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        Lx = float(physics["Lx"])
        Nx = int(grid["Nx"])
        Nh = int(grid["Nh"])
        Nl = int(grid["Nl"])
        alpha = float(physics["alpha"])
        u = float(physics.get("u", 0.0))
        v_a = float(physics["v_a"])
        v_b = float(physics["v_b"])
        width = v_b - v_a

        kx_1d = jnp.fft.fftfreq(Nx) * Nx * 2.0 * jnp.pi / Lx
        one_over_kx = np.zeros(Nx, dtype=np.float64)
        one_over_kx[1:] = 1.0 / np.asarray(kx_1d[1:])
        kx_sq = kx_1d**2

        x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
        modes = jnp.fft.fftfreq(Nx) * Nx
        mask23 = jnp.abs(modes) <= (Nx // 3)

        grid.update(
            {
                "x": x,
                "kx_1d": kx_1d,
                "one_over_kx": jnp.asarray(one_over_kx),
                "kx_sq": kx_sq,
                "mask23": mask23,
                "width": width,
            }
        )
        self.cfg["grid"] = grid

    # ------------------------------------------------------------------
    def init_state_and_args(self) -> None:
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        Lx = float(physics["Lx"])
        Nx = int(grid["Nx"])
        Nh = int(grid["Nh"])
        Nl = int(grid["Nl"])
        alpha = float(physics["alpha"])
        v_a = float(physics["v_a"])
        v_b = float(physics["v_b"])

        x = np.asarray(grid["x"])
        C = np.zeros((Nh, Nx), dtype=np.float64)  # real-space C_n(x)
        B = np.zeros((Nl, Nx), dtype=np.float64)  # real-space B_m(x)

        init = self.cfg.get("initialization", {"type": "linear-advection"})
        itype = init.get("type", "linear-advection")

        if itype == "linear-advection":
            eps = float(init.get("eps", 1.0))
            mode = int(init.get("mode", 1))
            n_bulk = _density_profile(x, Lx, 1.0, eps, mode)
            C[0] = n_bulk / alpha

        elif itype == "two-stream":
            eps = float(init.get("eps", 0.01))
            mode = int(init.get("mode", 1))
            n_bulk = _density_profile(x, Lx, 1.0, eps, mode)
            C[0] = n_bulk / alpha
            # v^2 Maxwellian -> C_2 = sqrt(2) C_0 in the AW-Hermite basis (alpha=sqrt(2))
            if Nh > 2:
                C[2] = np.sqrt(2.0) * C[0]

        elif itype == "bump-on-tail":
            eps = float(init.get("eps", 1e-4))
            mode = int(init.get("mode", 1))
            n_beam = float(init.get("n_beam", 0.01))
            v_drift = float(init.get("v_drift", 10.0))
            v_th = float(init.get("v_th", 1.0))
            n_bulk = _density_profile(x, Lx, 1.0 - n_beam, eps, mode)
            C[0] = n_bulk / alpha
            amp = n_beam / (np.sqrt(2.0 * np.pi) * v_th)
            B_beam = _project_legendre(lambda v: amp * np.exp(-((v - v_drift) ** 2) / (2.0 * v_th**2)), Nl, v_a, v_b)
            B[:] = B_beam[:, None]  # spatially uniform beam

        elif itype == "custom":
            # hermite: {n: {base, eps, mode}} ; df: {beams: [{amp, v_drift, v_th}], eps, mode}
            for n_str, spec in init.get("hermite", {}).items():
                n = int(n_str)
                C[n] = _density_profile(
                    x, Lx, float(spec.get("base", 0.0)), float(spec.get("eps", 0.0)), int(spec.get("mode", 1))
                )
            df = init.get("df", None)
            if df:
                beams = df.get("beams", [])

                def g(v):
                    out = np.zeros_like(v)
                    for b in beams:
                        out = out + float(b["amp"]) * np.exp(
                            -((v - float(b.get("v_drift", 0.0))) ** 2) / (2.0 * float(b.get("v_th", 1.0)) ** 2)
                        )
                    return out

                B_beam = _project_legendre(g, Nl, v_a, v_b)
                spatial = _density_profile(x, Lx, 1.0, float(df.get("eps", 0.0)), int(df.get("mode", 1)))
                B[:] = B_beam[:, None] * spatial[None, :]
        else:
            raise ValueError(f"Unknown initialization.type {itype!r}")

        Ck = jnp.fft.fft(jnp.asarray(C), axis=-1, norm="forward").astype(jnp.complex128)
        Bk = jnp.fft.fft(jnp.asarray(B), axis=-1, norm="forward").astype(jnp.complex128)

        # Seed the field diagnostics with the actual t=0 Poisson field so the energy
        # diagnostic is self-consistent at t=0 (the perturbed initial state carries a
        # nonzero E ~ eps); leaving e=0 would record a spurious O(eps^2) jump at step 1.
        field_on = bool(physics.get("field", True))
        if field_on:
            poisson = PoissonSolver1D(
                one_over_kx=grid["one_over_kx"], kx_sq=grid["kx_sq"], alpha=alpha, width=(v_b - v_a)
            )
            e0 = poisson.electric_field(Ck, Bk)
            phi0 = poisson.potential(Ck, Bk)
        else:
            e0 = jnp.zeros(Nx)
            phi0 = jnp.zeros(Nx)

        self.state = {
            "Ck": Ck.view(jnp.float64),
            "Bk": Bk.view(jnp.float64),
            "e": e0,
            "phi": phi0,
        }
        self.args = {}

    # ------------------------------------------------------------------
    def init_diffeqsolve(self) -> None:
        physics = self.cfg["physics"]
        grid = self.cfg["grid"]

        Nh = int(grid["Nh"])
        Nl = int(grid["Nl"])
        alpha = float(physics["alpha"])
        u = float(physics.get("u", 0.0))
        v_a = float(physics["v_a"])
        v_b = float(physics["v_b"])
        width = v_b - v_a
        gamma = float(physics.get("gamma", 0.5))
        nu_H = float(physics.get("nu_H", 0.0))
        nu_L = float(physics.get("nu_L", 0.0))
        enforce = bool(physics.get("enforce_conservation", True))
        field_on = bool(physics.get("field", True))
        dt = float(grid["dt"])

        kx_1d = grid["kx_1d"]
        one_over_kx = grid["one_over_kx"]
        kx_sq = grid["kx_sq"]
        mask23 = grid["mask23"]

        # Streaming exponentials (exact, prediagonalized symmetric tridiagonals)
        T_H = hermite_streaming_matrix(Nh, u, alpha)
        leg = legendre_constants(Nl, v_a, v_b)
        hermite_stream = StreamingExp1D(T_H, prefactor=-1j * alpha, kx_1d=kx_1d)
        legendre_stream = StreamingExp1D(np.asarray(leg["T_L"]), prefactor=-1j, kx_1d=kx_1d)

        hermite_coll = DiagonalCollisionExp1D(nu_H, safe_col(Nh))
        legendre_coll = DiagonalCollisionExp1D(nu_L, safe_col(Nl))
        combined_exp = CombinedLinearExp1D(hermite_stream, legendre_stream, hermite_coll, legendre_coll)

        poisson = PoissonSolver1D(one_over_kx=one_over_kx, kx_sq=kx_sq, alpha=alpha, width=width)

        # Explicit-term constants
        n = jnp.arange(Nh, dtype=jnp.float64)
        sqrt_2n_over_alpha = jnp.sqrt(2.0 * n) / alpha
        gamma_vec = jnp.where(jnp.arange(Nl) >= 3, gamma, 0.0)

        J = hermite_legendre_coupling_vector(Nh, Nl, alpha, u, v_a, v_b, enforce_conservation=enforce)
        coupling_vec = -(alpha / width) * jnp.sqrt(Nh / 2.0) * J  # folds prefactor into J_{Nh,m}

        vector_field = HermiteLegendre1DVectorField(
            combined_exp=combined_exp,
            poisson=poisson,
            kx_1d=kx_1d,
            sqrt_2n_over_alpha=sqrt_2n_over_alpha,
            deriv=leg["deriv"],
            gamma_vec=gamma_vec,
            xi_a=leg["xi_a"],
            xi_b=leg["xi_b"],
            coupling_vec=coupling_vec,
            alpha=alpha,
            width=width,
            dt=dt,
            mask23=mask23,
            field_on=field_on,
        )

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
    def post_process(self, run_output: dict, td: str) -> dict:
        import matplotlib.pyplot as plt

        sol = run_output["solver result"]
        binary_dir = os.path.join(td, "binary")
        plots_dir = os.path.join(td, "plots")
        os.makedirs(binary_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        x = np.asarray(self.cfg["grid"]["x"])
        datasets = {}
        metrics = {"simulation_completed": True}

        for k in sol.ys.keys():
            t_arr = np.asarray(sol.ts[k])
            data = sol.ys[k]

            if k == "fields":
                fields_dict = {name: np.asarray(arr) for name, arr in data.items()}
                datasets["fields"] = store_fields_timeseries(fields_dict, t_arr, binary_dir, x)
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

            elif k in ("hermite", "legendre"):
                name = "Ck" if k == "hermite" else "Bk"
                arr = np.asarray(data[name])
                datasets[k] = store_coeff_timeseries(name, arr, t_arr, binary_dir)

            elif k == "default":
                scalars = {name: np.asarray(arr) for name, arr in data.items()}
                for name, arr in scalars.items():
                    if arr.ndim == 1:
                        final_val = float(arr[-1])
                        metrics[f"final_{name}"] = final_val if np.isfinite(final_val) else float("nan")
                        # relative drift for conserved invariants
                        if name in ("mass", "momentum", "energy") and np.isfinite(arr[0]) and abs(arr[0]) > 0:
                            metrics[f"reldrift_{name}"] = float(np.max(np.abs(arr - arr[0])) / abs(arr[0]))
                        try:
                            arr_plot = np.where(np.isfinite(arr), arr, np.nan)
                            fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                            axes[0].plot(t_arr, arr_plot)
                            axes[0].set_xlabel("t")
                            axes[0].set_ylabel(name)
                            axes[0].grid(alpha=0.3)
                            axes[1].semilogy(t_arr, np.abs(arr_plot) + 1e-30)
                            axes[1].set_xlabel("t")
                            axes[1].set_ylabel(f"|{name}|")
                            axes[1].grid(alpha=0.3)
                            fig.savefig(os.path.join(plots_dir, f"scalar-{name}.png"), bbox_inches="tight")
                        except Exception as e:
                            print(f"post_process: scalar plot for {name} failed: {e}")
                        finally:
                            plt.close("all")

        if "default" in sol.ts:
            metrics["n_timesteps"] = len(sol.ts["default"])

        return {"metrics": metrics, "datasets": datasets}
