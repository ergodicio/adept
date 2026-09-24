"""Logging-free preparation and host analysis for the FARSIGHT-inspired solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from adept.core import (
    MetricEvent,
    ObservationPlan,
    ObservationSchedule,
    PreparedSimulation,
    Report,
    RunManifest,
    SimulationSpec,
)
from adept.core.builtin_solvers import FARSIGHT1D_CAPABILITIES
from adept.core.observations_jax import infer_observation_spec
from adept.core.preparation import normalize_key, structural_fingerprint
from adept.core.programs import ScanProgram
from adept.farsight1d.config import Farsight1DConfig, SaveCadence
from adept.farsight1d.numerics import FarsightSystem, diagnose, electric_field, initial_state, make_mesh


class FarsightProgram(ScanProgram):
    """Expose invalid panel geometry or nonfinite evolution as a failed result."""

    def __call__(self, params, state, inputs, key):
        if not isinstance(params, dict) or params or not isinstance(inputs, dict) or inputs:
            raise ValueError("farsight-1d currently requires empty params and inputs; differentiate the initial state")
        result = super().__call__(params, state, inputs, key)
        return result._replace(
            status=diffrax.RESULTS.where(
                result.final_state["valid"], diffrax.RESULTS.successful, diffrax.RESULTS.nonfinite
            )
        )


class FarsightFieldsObservation(eqx.Module):
    """Regularized electric field evaluated at fixed, unique periodic x nodes."""

    x: jax.Array
    weights: jax.Array
    length: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, jax.Array]:
        del t, inputs
        field = electric_field(
            self.x,
            state["x"].reshape(-1),
            (-self.weights * state["f"]).reshape(-1),
            self.length,
            self.epsilon,
            self.chunk_size,
        )
        return {"electric_field": field}


class FarsightScalarsObservation(eqx.Module):
    """Quadrature invariants, remesh error budget, and physical field energy."""

    fields: FarsightFieldsObservation

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, jax.Array]:
        result = diagnose(state, self.fields.weights)
        field = self.fields(t, state, inputs)["electric_field"]
        result["electric_energy"] = 0.5 * self.fields.length * jnp.mean(field**2)
        result["total_energy"] = result["kinetic_energy"] + result["electric_energy"]
        return result


class FarsightDistributionObservation(eqx.Module):
    """Moving phase-space node coordinates and their attached distribution values."""

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, jax.Array]:
        del t, inputs
        return {name: state[name] for name in ("x", "v", "f")}


@dataclass(frozen=True)
class Farsight1DAnalyzer:
    """Convert retained observations to labeled datasets without ambient file IO."""

    def analyze(self, result, manifest):
        import xarray as xr

        if not bool(np.asarray(result.final_state["valid"])):
            raise ArithmeticError(
                "FARSIGHT evolution encountered invalid panel geometry or nonfinite values; "
                "reduce dt/remesh_every and inspect invalid_panels and max_panel_area_error"
            )
        numerical_values = (result.final_state, result.observations, result.times, result.stats)
        if any(not np.all(np.isfinite(np.asarray(value))) for value in jax.tree.leaves(numerical_values)):
            raise ArithmeticError("FARSIGHT result contains nonfinite state, observations, or statistics")

        grid = manifest.resolved_config["grid"]
        datasets = {}
        for name, observations in result.observations.items():
            times = np.asarray(result.times[name])
            if name == "scalars":
                datasets[name] = xr.Dataset(
                    {key: ("t", np.asarray(value)) for key, value in observations.items()},
                    coords={"t": times},
                )
                for key in ("electric_energy", "total_energy"):
                    datasets[name][key].attrs["description"] = (
                        "Physical-grid E^2/2 diagnostic; not the exact regularized interaction Hamiltonian"
                    )
                datasets[name]["c2"].attrs["description"] = (
                    "Reference-quadrature integral of f^2; constant marker values alone do not guarantee "
                    "that the moving panel representation remains resolved"
                )
            elif name == "fields":
                x_axis = np.linspace(grid["xmin"], grid["xmax"], grid["nx"], endpoint=False)
                datasets[name] = xr.Dataset(
                    {key: (("t", "x"), np.asarray(value)) for key, value in observations.items()},
                    coords={"t": times, "x": x_axis},
                )
            else:
                datasets[name] = xr.Dataset(
                    {key: (("t", "x_node", "v_node"), np.asarray(value)) for key, value in observations.items()},
                    coords={"t": times, "x_node": np.arange(grid["nx"] + 1), "v_node": np.arange(grid["nv"] + 1)},
                )
                datasets[name].attrs["description"] = (
                    "x and v are moving marker coordinates, not fixed coordinate axes; "
                    "the x endpoint duplicates the periodic seam with quadrature endpoint weights"
                )
            datasets[name].attrs["solver"] = "farsight-1d"
            datasets[name].attrs["normalization"] = "electron plasma units: n0=m_e=|q_e|=epsilon_0=1"

        scalars = result.observations["scalars"]
        values = {f"final_{name}": float(np.asarray(value)[-1]) for name, value in scalars.items()}
        for name in ("mass", "c2"):
            start, stop = np.asarray(scalars[name])[[0, -1]]
            values[f"relative_{name}_change"] = float((stop - start) / start)
        return Report(result=datasets, metrics=(MetricEvent(values, step=int(np.asarray(result.stats["num_steps"]))),))


def _schedule(cadence: SaveCadence, num_steps: int) -> ObservationSchedule:
    points = tuple(range(0, num_steps + 1, cadence.every_steps))
    if points[-1] != num_steps:
        points += (num_steps,)
    return ObservationSchedule.at_steps(points)


def _initial_distribution(config: Farsight1DConfig, x, v, weights):
    initial = config.initial
    velocity_shape = jnp.exp(-0.5 * ((v - initial.drift) / initial.thermal_speed) ** 2)
    if initial.kind == "two-stream":
        velocity_shape += jnp.exp(-0.5 * ((v + initial.drift) / initial.thermal_speed) ** 2)
    length = config.grid.xmax - config.grid.xmin
    spatial_shape = 1.0 + initial.amplitude * jnp.cos(2 * jnp.pi * initial.mode * (x - config.grid.xmin) / length)
    distribution = spatial_shape * velocity_shape
    mass = jnp.sum(distribution * weights)
    if not np.isfinite(float(mass)) or float(mass) <= 0:
        raise ValueError("Initial distribution is unresolved on the configured velocity domain and mesh")
    return (distribution / mass) * length


class Farsight1DBuilder:
    """Prepare a normalized electron solve using the explicit ADEPT host contracts."""

    def prepare(self, spec: SimulationSpec, *, key: int | jax.Array) -> PreparedSimulation:
        if spec.solver != "farsight-1d":
            raise ValueError(f"Farsight1DBuilder cannot prepare solver {spec.solver!r}")
        if spec.schema_version != "1":
            raise ValueError(f"Unsupported FARSIGHT specification schema version {spec.schema_version!r}")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("farsight-1d requires float64; enable jax_enable_x64 before preparation")
        config = Farsight1DConfig.model_validate(spec.config_dict())
        normalized_key, seed, provenance = normalize_key(key)
        grid, time, numerical = config.grid, config.time, config.numerical
        length = grid.xmax - grid.xmin
        x, v, weights = make_mesh(grid.nx, grid.nv, grid.xmin, grid.xmax, grid.vmin, grid.vmax, numerical.quadrature)
        state = initial_state(x, v, _initial_distribution(config, x, v, weights), weights)
        params, inputs = {}, {}
        system = FarsightSystem(
            x0=x,
            v0=v,
            weights=weights,
            length=length,
            dt=time.dt,
            epsilon=numerical.epsilon,
            charge=-1.0,
            mass=1.0,
            remesh_every=numerical.remesh_every,
            chunk_size=numerical.chunk_size,
        )
        fields = FarsightFieldsObservation(x[:-1, 0], weights, length, numerical.epsilon, numerical.chunk_size)
        functions = {
            "scalars": FarsightScalarsObservation(fields),
            "fields": fields,
            "distribution": FarsightDistributionObservation(),
        }
        plan = ObservationPlan(
            tuple(
                infer_observation_spec(
                    name, functions[name], _schedule(cadence, time.num_steps), t=time.tmin, state=state, inputs=inputs
                )
                for name in functions
                if (cadence := getattr(config.save, name)) is not None
            )
        )
        program = FarsightProgram.from_observation_plan(
            system=system,
            plan=plan,
            state=state,
            inputs=inputs,
            t0=time.tmin,
            dt=time.dt,
            num_steps=time.num_steps,
        )
        resolved = config.model_dump()
        resolved["time"]["num_steps"] = time.num_steps
        units = {
            "normalization": "electron plasma units: n0=m_e=|q_e|=epsilon_0=1",
            "x": "v_ref / omega_pe",
            "v": "v_ref",
            "t": "1 / omega_pe",
            "electric_field": "m_e * v_ref * omega_pe / |q_e|",
            "background": "periodic kernel removes the instantaneous mean charge; no net periodic charge mode",
            "field_energy": "physical E^2/2 quadrature, not the exact regularized Hamiltonian",
        }
        manifest = RunManifest(
            raw_config=spec.config_dict(),
            resolved_config=resolved,
            units=units,
            seed=seed,
            key_provenance=provenance,
            structural_fingerprint=structural_fingerprint(spec, program, params, state, inputs, normalized_key),
        )
        return PreparedSimulation(
            program=program,
            params=params,
            state=state,
            inputs=inputs,
            manifest=manifest,
            analyzer=Farsight1DAnalyzer(),
            capabilities=FARSIGHT1D_CAPABILITIES,
            observation_plan=plan,
        )


__all__ = ["Farsight1DAnalyzer", "Farsight1DBuilder", "FarsightProgram"]
