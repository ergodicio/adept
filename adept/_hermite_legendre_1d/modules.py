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
    ExternalExDriver,
    HermiteLegendre1DVectorField,
    PoissonSolver1D,
    StreamingExp1D,
    hermite_force_operator,
    hermite_legendre_coupling_vector,
    hermite_streaming_matrix,
    legendre_constants,
    legendre_force_operator,
    safe_col,
)


def _density_profile(x: np.ndarray, Lx: float, base: float, eps: float, mode: int) -> np.ndarray:
    """base * (1 + eps cos(2*pi*mode*x/Lx))."""
    return base * (1.0 + eps * np.cos(2.0 * np.pi * mode * x / Lx))


def _distribution_diagnostics(cfg: dict, coeff_data: dict, plots_dir: str, binary_dir: str):
    """Facet plots + netCDF of the distribution reconstructed from saved coefficients.

    Reconstructs f(x, v, t) = sum_n C_n(x) psi_n(v) + sum_m B_m(x) xi_m(v) at up to
    six snapshot times and writes (1) a phase-space facet figure f(x, v), (2) a
    times x {Hermite, Legendre} facet figure of the coefficient magnitudes
    |C_n(x)|, |B_m(x)|, and (3) the reconstruction as a netCDF Dataset.
    """
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.colors import LogNorm

    from adept._hermite_legendre_1d.vector_field import _hermite_function_values, _legendre_basis_values

    physics = cfg["physics"]
    alpha = float(physics["alpha"])
    u = float(physics.get("u", 0.0))
    v_a = float(physics["v_a"])
    v_b = float(physics["v_b"])
    x = np.asarray(cfg["grid"]["x"])

    t_arr, Ck = coeff_data["hermite"]  # (nt, Nh, Nx) complex, k-space
    _, Bk = coeff_data["legendre"]  # (nt, Nl, Nx)
    nt = min(Ck.shape[0], Bk.shape[0])
    idx = np.unique(np.linspace(0, nt - 1, min(6, nt)).astype(int))
    Nh, Nl = Ck.shape[1], Bk.shape[1]

    # v grid covering the Hermite bulk and the Legendre window
    v = np.linspace(min(u - 5.0 * alpha, v_a - 1.0), max(u + 5.0 * alpha, v_b + 1.0), 384)
    psi = _hermite_function_values(Nh, v, u, alpha)  # (Nh, Nv)
    xi = _legendre_basis_values(Nl, v, v_a, v_b)
    xi[:, (v < v_a) | (v > v_b)] = 0.0  # window basis has no support outside [v_a, v_b]

    C_sel = [np.fft.ifft(Ck[i], axis=-1, norm="forward").real for i in idx]  # (Nh, Nx) each
    B_sel = [np.fft.ifft(Bk[i], axis=-1, norm="forward").real for i in idx]
    f_sel = np.stack([C.T @ psi + B.T @ xi for C, B in zip(C_sel, B_sel)])  # (nsel, Nx, Nv)

    # facet 1: phase space f(x, v), shared log color scale
    fmax = float(np.nanmax(f_sel))
    floor = max(fmax, 1e-30) * 1e-7
    ncol = min(3, len(idx))
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=True, sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    norm = LogNorm(vmin=floor, vmax=fmax)
    for ax, i, f in zip(axes, idx, f_sel):
        im = ax.pcolormesh(x, v, np.maximum(f, floor).T, norm=norm, cmap="magma", shading="auto", rasterized=True)
        ax.set_title(f"t = {t_arr[i]:.0f}", fontsize=10)
    for ax in axes[len(idx) :]:
        ax.set_visible(False)
    for ax in axes.reshape(nrow, ncol)[:, 0]:
        ax.set_ylabel("v / vth")
    for ax in axes.reshape(nrow, ncol)[-1, :]:
        ax.set_xlabel("x [norm]")
    fig.colorbar(im, ax=axes.tolist(), label=r"$f(x, v)$ (clipped at $10^{-7} f_{max}$)", shrink=0.85)
    fig.savefig(os.path.join(plots_dir, "phase-space-f_xv.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # facet 2: coefficient magnitudes, rows = times, cols = {Hermite, Legendre}
    cmax = max(float(np.max(np.abs(Ck[idx]))), 1e-30)
    bmax = max(float(np.max(np.abs(Bk[idx]))), 1e-30)
    fig, axes = plt.subplots(len(idx), 2, figsize=(9.0, 2.2 * len(idx)), sharex=True, layout="constrained")
    axes = np.atleast_2d(np.asarray(axes))
    for row, (i, C, B) in enumerate(zip(idx, C_sel, B_sel)):
        for col, (arr, N, vmax, name) in enumerate(((C, Nh, cmax, r"$|C_n(x)|$"), (B, Nl, bmax, r"$|B_m(x)|$"))):
            ax = axes[row, col]
            im = ax.pcolormesh(
                x,
                np.arange(N),
                np.maximum(np.abs(arr), vmax * 1e-12),
                norm=LogNorm(vmin=vmax * 1e-12, vmax=vmax),
                cmap="viridis",
                shading="auto",
                rasterized=True,
            )
            ax.set_ylabel(f"t={t_arr[i]:.0f}\n" + ("n" if col == 0 else "m"), fontsize=9)
            if row == 0:
                ax.set_title(name)
            if row == len(idx) - 1:
                ax.set_xlabel("x [norm]")
            fig.colorbar(im, ax=ax, shrink=0.9)
    fig.savefig(os.path.join(plots_dir, "coefficients-facets.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    ds = xr.Dataset({"f": (["t", "x", "v"], f_sel)}, coords={"t": t_arr[idx], "x": x, "v": v})
    ds.to_netcdf(os.path.join(binary_dir, "distribution-f_xv.nc"))
    return ds


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
            "de": jnp.zeros(Nx),  # external Ex driver field (diagnostic)
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
        integrator = str(grid.get("integrator", "lawson")).lower()
        imex = integrator == "imex"
        implicit_mp = integrator == "implicit"
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

        col_e, col_l = safe_col(Nh), safe_col(Nl)
        hermite_coll = DiagonalCollisionExp1D(nu_H, col_e)
        legendre_coll = DiagonalCollisionExp1D(nu_L, col_l)
        combined_exp = CombinedLinearExp1D(hermite_stream, legendre_stream, hermite_coll, legendre_coll)

        poisson = PoissonSolver1D(one_over_kx=one_over_kx, kx_sq=kx_sq, alpha=alpha, width=width)

        # External longitudinal (Ex) driver, e.g. a resonant EPW kick for Landau damping
        ex_cfg = self.cfg.get("drivers", {}).get("ex", {})
        ex_driver = ExternalExDriver(grid["x"], ex_cfg) if ex_cfg else None

        # Explicit-term constants
        n = jnp.arange(Nh, dtype=jnp.float64)
        sqrt_2n_over_alpha = jnp.sqrt(2.0 * n) / alpha
        gamma_vec = jnp.where(jnp.arange(Nl) >= 3, gamma, 0.0)

        J = hermite_legendre_coupling_vector(Nh, Nl, alpha, u, v_a, v_b, enforce_conservation=enforce)
        coupling_vec = -(alpha / width) * jnp.sqrt(Nh / 2.0) * J  # folds prefactor into J_{Nh,m}

        # IMEX force operators (only built when integrator == "imex")
        G_C = jnp.asarray(hermite_force_operator(Nh, alpha)) if imex else None
        G_B = (
            jnp.asarray(legendre_force_operator(leg["deriv"], gamma_vec, leg["xi_a"], leg["xi_b"], width))
            if imex
            else None
        )

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
            ex_driver=ex_driver,
            imex=imex,
            G_C=G_C,
            G_B=G_B,
            implicit=implicit_mp,
            T_H=jnp.asarray(T_H) if implicit_mp else None,
            T_L=jnp.asarray(np.asarray(leg["T_L"])) if implicit_mp else None,
            col_e=col_e if implicit_mp else None,
            col_l=col_l if implicit_mp else None,
            nu_H=nu_H,
            nu_L=nu_L,
            newton_iters=int(grid.get("newton_iters", 3)),
            gmres_restart=int(grid.get("gmres_restart", 20)),
            gmres_maxiter=int(grid.get("gmres_maxiter", 4)),
            gmres_tol=float(grid.get("gmres_tol", 1e-8)),
            precondition=bool(grid.get("precondition", True)),
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
        coeff_data = {}

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
                coeff_data[k] = (t_arr, arr)

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

        if "hermite" in coeff_data and "legendre" in coeff_data:
            try:
                datasets["distribution"] = _distribution_diagnostics(self.cfg, coeff_data, plots_dir, binary_dir)
            except Exception as e:
                print(f"post_process: distribution diagnostics failed: {e}")

        if "default" in sol.ts:
            metrics["n_timesteps"] = len(sol.ts["default"])

        return {"metrics": metrics, "datasets": datasets}
