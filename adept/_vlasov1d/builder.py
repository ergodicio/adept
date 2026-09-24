"""Logging-free preparation and explicit numerical execution for Vlasov-1D."""

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
from jax import numpy as jnp

from adept._vlasov1d import _pytree  # Register the legacy numerical operators.
from adept._vlasov1d.preparation import Vlasov1DSetup
from adept._vlasov1d.storage import distribution_datasets, field_datasets
from adept.core import (
    MetricEvent,
    ObservationPlan,
    ObservationSchedule,
    PreparedSimulation,
    Report,
    RunManifest,
    SimulationSpec,
)
from adept.core.builtin_solvers import VLASOV1D_CAPABILITIES
from adept.core.observations_jax import infer_observation_spec
from adept.core.preparation import normalize_key, structural_fingerprint
from adept.core.programs import ScanProgram


class Vlasov1DSystem(eqx.Module):
    """The existing split-step map with explicit runtime driver controls."""

    operator: Any
    dt: float = eqx.field(static=True)

    def step(self, step, state, params, inputs, key):
        # Random density and OU forcing realizations are prepared from the seeds
        # in the specification; advancing the numerical map is deterministic.
        del key
        return self.operator(step * self.dt, state, eqx.combine(params, inputs))


@dataclass(frozen=True)
class Vlasov1DAnalyzer:
    """Construct the historical datasets without tracking or artifact writes."""

    config: dict

    def analyze(self, result, manifest):
        import xarray as xr

        del manifest
        fields = {}
        for name, values in result.observations.items():
            if name.startswith("fields"):
                fields = field_datasets(self.config, values, result.times[name], name)
        scalars = xr.Dataset(
            {name: ("t", values) for name, values in result.observations["default"].items()},
            coords={"t": result.times["default"]},
        )
        distributions = distribution_datasets(self.config, result.times, result.observations)
        metrics = {name: float(values[-1]) for name, values in result.observations["default"].items()}
        return Report(
            result={"fields": fields, "dists": distributions, "scalars": scalars},
            metrics=(MetricEvent(metrics),),
        )


def _with_defaults(config, defaults):
    """Fill omitted schema defaults without discarding legacy extension fields."""
    result = dict(defaults)
    for name, value in config.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _with_defaults(value, result[name])
        else:
            result[name] = value
    return result


class Vlasov1DBuilder:
    """Prepare Vlasov1D through the same host setup as ``BaseVlasov1D``."""

    def prepare(self, spec: SimulationSpec, *, key: int | jax.Array) -> PreparedSimulation:
        if spec.solver != "vlasov-1d":
            raise ValueError(f"Vlasov1DBuilder cannot prepare solver {spec.solver!r}")
        if spec.schema_version != "1":
            raise ValueError(f"Unsupported Vlasov1D specification schema version {spec.schema_version!r}")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("vlasov-1d requires float64; enable jax_enable_x64 before preparation")
        normalized_key, seed, provenance = normalize_key(key)
        setup = Vlasov1DSetup({**spec.config_dict(), "solver": spec.solver})
        # Resolve optional defaults for new callers without changing the legacy
        # configuration snapshots (the legacy wrapper retains its original cfg).
        setup.cfg = _with_defaults(
            setup.cfg, setup.config_model.model_dump(exclude={"mlflow"}, exclude_none=True, by_alias=True)
        )
        units = setup.write_units()
        setup.get_derived_quantities()
        setup.get_solver_quantities()
        setup.init_state_and_args()
        operator = setup.prepare_step()
        saves = setup.prepare_save_quantities()
        grid = setup.simulation.grid
        if grid.nt > grid.max_steps:
            raise ValueError(f"Requested {grid.nt} steps exceeds the supported limit {grid.max_steps}")
        # Booleans and integers describe driver topology and must remain static.
        runtime = {
            "drivers": jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, float) else x, setup.args["drivers"])
        }
        params, inputs = eqx.partition(runtime, False)
        plan = ObservationPlan(
            tuple(
                infer_observation_spec(
                    name,
                    save["func"],
                    ObservationSchedule.at_times(save["t"]["ax"].tolist()),
                    t=0.0,
                    state=setup.state,
                    inputs=inputs,
                )
                for name, save in saves.items()
            )
        )
        program = ScanProgram.from_observation_plan(
            system=Vlasov1DSystem(operator, grid.dt),
            plan=plan,
            state=setup.state,
            inputs=inputs,
            t0=0.0,  # The legacy Vlasov solve starts at zero, including for nonzero grid.tmin.
            dt=grid.dt,
            num_steps=grid.nt,
            interpolate=True,
        )
        # Observations are owned by the program/plan, never by the host manifest.
        for save in saves.values():
            save.pop("func")
        manifest = RunManifest(
            raw_config=spec.config_dict(),
            resolved_config=setup.cfg,
            units=units,
            seed=seed,
            key_provenance=provenance,
            structural_fingerprint=structural_fingerprint(spec, program, params, setup.state, inputs, normalized_key),
        )
        return PreparedSimulation(
            program=program,
            params=params,
            state=setup.state,
            inputs=inputs,
            manifest=manifest,
            analyzer=Vlasov1DAnalyzer(setup.cfg),
            capabilities=VLASOV1D_CAPABILITIES,
            observation_plan=plan,
        )


__all__ = ["Vlasov1DAnalyzer", "Vlasov1DBuilder", "Vlasov1DSystem"]
