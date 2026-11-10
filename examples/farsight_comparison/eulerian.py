"""Scoped Eulerian benchmark adapter, executed through the normal ergoExo lifecycle.

The production solver is unchanged. This adapter makes the benchmark IC, uniform
ions, initial electric field, exact save times, and optional softened field model
explicit. Advection is always spectral in x and cubic-spline in v.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from diffrax import SubSaveAt
from scipy.integrate import quad

from adept._vlasov1d.modules import BaseVlasov1D
from adept._vlasov1d.solvers.pushers.field import SpectralPoissonSolver

from .cases import ComparisonCase, get_case


@lru_cache(maxsize=128)
def _softened_multiplier_tuple(nx: int, length: float, epsilon: float) -> tuple[float, ...]:
    """Compute continuum Fourier coefficients, checking independent quadrature tolerances."""
    if nx < 4 or nx % 2 or not np.isfinite(length) or length <= 0 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Require even nx >= 4 and finite positive length and epsilon")
    alpha = epsilon / length

    def kernel(distance):
        r = distance / length
        return 0.5 * r * np.sqrt(1 + 4 * alpha * alpha) / np.sqrt(r * r + alpha * alpha) - r

    # Resolve the narrow regularized core explicitly, even as epsilon tends
    # towards zero; include oscillation-scale partitions for higher modes.
    points = sorted(
        {
            float(x)
            for x in [*np.linspace(0, length / 2, nx + 1), *(epsilon * np.array([0.25, 1, 4, 16]))]
            if 0 < x < length / 2
        }
    )

    coefficients = [0.0]
    for mode in range(1, nx // 2 + 1):
        k = 2 * np.pi * mode / length

        def integrate(tolerance, k=k):
            value, error = quad(
                lambda distance: kernel(distance) * np.sin(k * distance),
                0.0,
                length / 2,
                points=points,
                epsabs=tolerance / (2 * k),
                epsrel=tolerance,
                limit=max(500, 4 * nx),
            )
            return 2 * k * value, 2 * k * error

        coarse, _ = integrate(2e-9)
        fine, error = integrate(2e-12)
        if not np.isfinite(fine) or error > 1e-9 or abs(fine - coarse) > 1e-9:
            raise ArithmeticError(f"Softened Fourier quadrature did not converge for mode {mode}")
        coefficients.append(float(fine))
    modes = np.rint(np.abs(np.fft.fftfreq(nx) * nx)).astype(int)
    return tuple(coefficients[mode] for mode in modes)


def softened_fourier_multiplier(nx: int, length: float, epsilon: float) -> np.ndarray:
    """Return mu(k) such that E_k = -i mu(k) rho_k/k for FARSIGHT's exact periodic kernel.

    μ(k) = 2k ∫_0^(L/2) K_epsilon(x) sin(kx) dx. This is a continuum
    coefficient, not a grid-sampled kernel whose aliases depend on nx.
    The zero mode is projected out; negative modes have the same real μ.
    """
    return np.asarray(_softened_multiplier_tuple(int(nx), float(length), float(epsilon)))


def build_eulerian_config(
    case: ComparisonCase | str,
    *,
    nx: int | None = None,
    nv: int | None = None,
    tmax: float | None = None,
    dt: float | None = None,
    epsilon: float | None = None,
    field_model: str = "poisson",
    output_dir: str | Path | None = None,
    experiment: str = "farsight-eulerian-comparison",
    run_name: str | None = None,
) -> dict:
    """Build a logged Eulerian benchmark config without changing production defaults."""
    case = get_case(case) if isinstance(case, str) else case
    changes = {
        key: value
        for key, value in {"nx": nx, "nv": nv, "tmax": tmax, "dt": dt, "epsilon": epsilon}.items()
        if value is not None
    }
    case = replace(case, **changes)
    if field_model not in {"poisson", "farsight-softened"}:
        raise ValueError("field_model must be 'poisson' or 'farsight-softened'")
    component = {
        "noise_seed": 0,
        "noise_type": "none",
        "noise_val": 0.0,
        "v0": case.drift,
        "T0": case.thermal_speed**2,
        "m": 2.0,
        "basis": "uniform",
        "baseline": 1.0,
    }
    density = {"quasineutrality": False, "species-background": component}
    if case.name == "two-stream":
        density["species-background"] = {**component, "baseline": 0.5}
        density["species-counterstream"] = {**component, "baseline": 0.5, "v0": -case.drift}
    disabled_envelope = {"center": 0.0, "width": 1.0, "rise": 1.0, "baseline": 0.0, "bump_height": 0.0}
    cadence = {"tmin": 0.0, "tmax": case.tmax, "nt": len(case.frame_times)}
    return {
        "solver": "vlasov-1d",
        "units": {"normalizing_temperature": "2000eV", "normalizing_density": "1.5e21/cc"},
        "density": density,
        "grid": {
            "nx": case.nx,
            "nv": case.nv,
            "xmin": 0.0,
            "xmax": case.length,
            "vmin": case.vmin,
            "vmax": case.vmax,
            "tmin": 0.0,
            "tmax": case.tmax,
            "dt": case.dt,
        },
        "save": {"fields": {"t": dict(cadence)}, "electron": {"main": {"t": dict(cadence)}}},
        "drivers": {"ex": {}, "ey": {}},
        "diagnostics": {"diag-vlasov-dfdt": False, "diag-fp-dfdt": False},
        "terms": {
            "field": "poisson",
            "vdfdx": "exponential",
            "edfdv": "cubic-spline",
            "time": "strang",
            "fokker_planck": {
                "is_on": False,
                "type": "Dougherty",
                "time": dict(disabled_envelope),
                "space": dict(disabled_envelope),
            },
            "krook": {"is_on": False, "time": dict(disabled_envelope), "space": dict(disabled_envelope)},
        },
        "mlflow": {"experiment": experiment, "run": run_name or f"{case.name}-eulerian-{field_model}"},
        "benchmark": {
            "case": case.to_dict(),
            "field_model": field_model,
            "initial_condition": "unit-density Gaussian mixture times (1 + amplitude*cos(k*x))",
            "ion_background": "uniform",
            "advection_x": "spectral-exponential",
            "advection_v": "cubic-spline",
            "output_dir": str(Path(output_dir).expanduser().resolve()) if output_dir is not None else None,
        },
    }


class BenchmarkVlasov1D(BaseVlasov1D):
    """Comparison-only uniform-ion setup and optional matched-softened Poisson field."""

    def __init__(self, cfg, *, epsilon=None):
        if not bool(jax.config.jax_enable_x64):
            raise RuntimeError("The kinetic comparison requires JAX_ENABLE_X64=true")
        self.case = ComparisonCase(**cfg["benchmark"]["case"])
        self.field_model = cfg["benchmark"]["field_model"]
        self.epsilon = self.case.epsilon if self.field_model == "farsight-softened" else None
        if epsilon is not None:
            if not np.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("epsilon must be finite and positive")
            self.epsilon = float(epsilon)
            self.field_model = "farsight-softened"
            cfg["benchmark"]["field_model"] = self.field_model
            cfg["benchmark"]["case"]["epsilon"] = self.epsilon
            self.case = replace(self.case, epsilon=self.epsilon)
        if (
            cfg["terms"]["vdfdx"] != "exponential"
            or cfg["terms"]["edfdv"] != "cubic-spline"
            or cfg["terms"]["time"] != "strang"
        ):
            raise ValueError("This benchmark requires spectral x, cubic-spline v, and Strang splitting")
        super().__init__(cfg)
        # The legacy grid adds a step to tmax and uses a linspace with a different
        # spacing. Override only this example's grid with exact integer step times.
        self.simulation.grid = eqx.tree_at(
            lambda grid: (grid.tmax, grid.nt, grid.t, grid.max_steps),
            self.simulation.grid,
            (self.case.tmax, self.case.num_steps + 1, jnp.asarray(self.case.step_times), self.case.num_steps + 4),
        )

    def get_solver_quantities(self):
        super().get_solver_quantities()
        grid = self.simulation.grid
        self.cfg["grid"]["ion_charge"] = jnp.ones(grid.nx)
        multiplier = np.ones(grid.nx)
        multiplier[0] = 0.0
        if self.epsilon is not None:
            multiplier = softened_fourier_multiplier(grid.nx, self.case.length, self.epsilon)
        self.field_multiplier = multiplier
        self.benchmark_field_solver = SpectralPoissonSolver(
            grid.one_over_kx * jnp.asarray(multiplier),
            self.cfg["grid"]["species_grids"],
            self.cfg["grid"]["species_params"],
            static_charge_density=self.cfg["grid"]["ion_charge"],
        )

    def init_state_and_args(self):
        super().init_state_and_args()
        grid = self.simulation.grid
        velocities = self.cfg["grid"]["species_grids"]["electron"]["v"]
        dv = self.cfg["grid"]["species_grids"]["electron"]["dv"]
        distribution = self.case.initial_distribution(np.asarray(grid.x)[:, None], np.asarray(velocities)[None, :])
        distribution *= self.case.length / (grid.dx * dv * np.sum(distribution))
        self.state["electron"] = jnp.asarray(distribution)
        self.state["e"] = self.benchmark_field_solver({"electron": self.state["electron"]}, None, None)

    def _save_scalars(self, t, state, args):
        del t, args
        f = state["electron"]
        velocity_grid = self.cfg["grid"]["species_grids"]["electron"]
        weight = self.simulation.grid.dx * velocity_grid["dv"]
        kinetic = 0.5 * weight * jnp.sum(f * velocity_grid["v"][None, :] ** 2)
        electric = 0.5 * self.simulation.grid.dx * jnp.sum(state["e"] ** 2)
        return {
            "mass": weight * jnp.sum(f),
            "c2": weight * jnp.sum(f * f),
            "momentum": weight * jnp.sum(f * velocity_grid["v"][None, :]),
            "kinetic_energy": kinetic,
            "electric_energy": electric,
            "total_energy": kinetic + electric,
            "min_f": jnp.min(f),
            "negative_mass": weight * jnp.sum(jnp.maximum(-f, 0.0)),
        }

    def init_diffeqsolve(self):
        super().init_diffeqsolve()
        field_solve = self.diffeqsolve_quants["terms"].vector_field.vpfp.vlasov_poisson.field_solve
        field_solve.es_field_solver = self.benchmark_field_solver
        subs = self.diffeqsolve_quants["saveat"]["subs"]
        for name in ("fields", "electron.main"):
            subs[name] = SubSaveAt(ts=jnp.asarray(self.case.frame_times), fn=self.cfg["save"][name]["func"])
        subs["default"] = SubSaveAt(ts=jnp.asarray(self.case.step_times), fn=self._save_scalars)

    def post_process(self, run_output, td):
        """Write comparison arrays through ergoExo's standard artifact-upload path."""
        result = extract_eulerian(run_output, self)
        directories = [Path(td) / "comparison"]
        output_dir = self.cfg["benchmark"].get("output_dir")
        if output_dir is not None:
            directories.append(Path(output_dir))
        metadata = {
            **self.cfg["benchmark"],
            "field_model": self.field_model,
            "fourier_multiplier": self.field_multiplier.tolist(),
            "integrator": "strang",
            "source": "examples.farsight_comparison.eulerian.BenchmarkVlasov1D",
        }
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            filenames = [directory / f"{name}.nc" for name in ("distribution", "scalars", "fields")]
            filenames += [directory / "metrics.json", directory / "benchmark.json"]
            if any(path.exists() for path in filenames):
                raise FileExistsError(f"Refusing to overwrite comparison artifacts in {directory}")
            for name in ("distribution", "scalars", "fields"):
                result[name].to_netcdf(directory / f"{name}.nc", engine="h5netcdf")
            (directory / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")
            (directory / "benchmark.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return result


def extract_eulerian(run_output, module: BenchmarkVlasov1D) -> dict:
    """Normalize retained output to f(t,x,v) and integrated invariant diagnostics."""
    result = run_output["solver result"]
    grid = module.simulation.grid
    attrs = {
        "solver": "vlasov-1d",
        "field_model": module.field_model,
        "normalization": "electron plasma units: n0=m_e=|q_e|=epsilon_0=1",
        "advection_x": "spectral-exponential",
        "advection_v": "cubic-spline",
    }
    distribution = xr.Dataset(
        {"f": (("t", "x", "v"), np.asarray(result.ys["electron.main"]))},
        coords={
            "t": np.asarray(result.ts["electron.main"]),
            "x": np.asarray(grid.x),
            "v": np.asarray(module.cfg["grid"]["species_grids"]["electron"]["v"]),
        },
        attrs=attrs,
    )
    fields = xr.Dataset(
        {"electric_field": (("t", "x"), np.asarray(result.ys["fields"]["e"]))},
        coords={"t": np.asarray(result.ts["fields"]), "x": np.asarray(grid.x)},
        attrs=attrs,
    )
    scalars = xr.Dataset(
        {key: ("t", np.asarray(value)) for key, value in result.ys["default"].items()},
        coords={"t": np.asarray(result.ts["default"])},
        attrs=attrs,
    )
    if module.epsilon is not None:
        for name in ("electric_energy", "total_energy"):
            scalars[name].attrs["description"] = (
                "Physical-grid E^2/2 diagnostic; not the regularized interaction Hamiltonian"
            )
    for dataset in (distribution, fields, scalars):
        if any(not np.isfinite(value).all() for value in dataset.data_vars.values()):
            raise ArithmeticError("Nonfinite Eulerian benchmark output")
    metrics = {f"final_{name}": float(value[-1]) for name, value in scalars.data_vars.items()}
    for name in ("mass", "c2"):
        metrics[f"relative_{name}_change"] = float((scalars[name][-1] - scalars[name][0]) / scalars[name][0])
    return {"distribution": distribution, "fields": fields, "scalars": scalars, "metrics": metrics}


__all__ = ["BenchmarkVlasov1D", "build_eulerian_config", "extract_eulerian", "softened_fourier_multiplier"]
