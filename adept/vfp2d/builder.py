"""Logging-free preparation and explicit numerical execution for VFP-2D."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from adept.core import (
    MetricEvent,
    ObservationCollective,
    ObservationPlacement,
    ObservationPlan,
    ObservationSchedule,
    PreparedSimulation,
    Report,
    RunManifest,
    SimulationSpec,
)
from adept.core.builtin_solvers import VFP2D_CAPABILITIES
from adept.core.observations_jax import infer_observation_spec
from adept.core.preparation import normalize_key, structural_fingerprint
from adept.core.programs import ScanProgram
from adept.vfp2d import _pytree  # Register explicit numerical operator trees.
from adept.vfp2d.postprocess import VFP2DPostProcessor
from adept.vfp2d.preparation import VFP2DSetup


class VFP2DSystem(eqx.Module):
    """The existing split-step map with explicit runtime controls."""

    operator: Any
    t0: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    mesh: Any = eqx.field(static=True, default=None)

    def step(self, step, state, params, inputs, key):
        del key  # All currently supported VFP2D terms are deterministic.
        args = eqx.combine(params, inputs)
        time = self.t0 + step * self.dt
        if self.mesh is not None:
            with self.mesh:
                return self.operator(time, state, args)
        return self.operator(time, state, args)


class VFP2DStateObservation(eqx.Module):
    spatial_sharding: Any = eqx.field(static=True, default=None)

    def __call__(self, t, state, inputs):
        del t, inputs
        if self.spatial_sharding is not None:
            return jax.tree.map(self.spatial_sharding.replicate, state)
        return state


class VFP2DProgram(ScanProgram):
    """Retain executed controls so host diagnostics use the same runtime inputs."""

    def __call__(self, params, state, inputs, key):
        result = super().__call__(params, state, inputs, key)
        return result._replace(stats={**result.stats, "runtime_inputs": eqx.combine(params, inputs)})


@dataclass(frozen=True)
class VFP2DAnalyzer:
    """Construct existing datasets and metrics without writing artifacts."""

    postprocessor: VFP2DPostProcessor

    def analyze(self, result, manifest):
        del manifest
        history = SimpleNamespace(ys=result.observations["state"], ts=result.times["state"])
        postprocessor = replace(self.postprocessor, args=result.stats["runtime_inputs"])
        output = postprocessor.post_process({"solver result": history}, "")
        return Report(result={"vfp2d": output["vfp2d"]}, metrics=(MetricEvent(output["metrics"]),))


class VFP2DBuilder:
    """Prepare all VFP2D field modes through the shared host setup."""

    def prepare(self, spec: SimulationSpec, *, key: int | jax.Array) -> PreparedSimulation:
        if spec.solver != "vfp-2d":
            raise ValueError(f"VFP2DBuilder cannot prepare solver {spec.solver!r}")
        if spec.schema_version != "1":
            raise ValueError(f"Unsupported VFP2D specification schema version {spec.schema_version!r}")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("vfp-2d requires float64; enable jax_enable_x64 before preparation")
        normalized_key, seed, provenance = normalize_key(key)
        setup = VFP2DSetup(spec.config_dict())
        units = setup.write_units()
        setup.get_derived_quantities()
        setup.init_state_and_args()
        operator = setup.prepare_step()
        save_times = setup.prepare_save_times()
        # Runtime scalars are arrays so a selected control can be differentiated
        # without also differentiating fixed grids, fields, or collision operators.
        runtime = jax.tree.map(jnp.asarray, setup.args)
        params, inputs = eqx.partition(runtime, False)
        observation = infer_observation_spec(
            "state",
            VFP2DStateObservation(setup.spatial_sharding),
            ObservationSchedule.at_times(save_times.tolist()),
            t=setup.tmin,
            state=setup.state,
            inputs=inputs,
            placement=ObservationPlacement.REPLICATED if setup.spatial_sharding else ObservationPlacement.DEVICE,
            collective=ObservationCollective.ALL_GATHER if setup.spatial_sharding else ObservationCollective.NONE,
        )
        plan = ObservationPlan((observation,))
        program = VFP2DProgram.from_observation_plan(
            system=VFP2DSystem(
                operator,
                setup.tmin,
                setup.grid.dt,
                None if setup.spatial_sharding is None else setup.spatial_sharding.mesh,
            ),
            plan=plan,
            state=setup.state,
            inputs=inputs,
            t0=setup.tmin,
            dt=setup.grid.dt,
            num_steps=setup.nt,
            interpolate=True,
        )
        resolved = setup.cfg
        resolved["units"]["derived"] = units
        manifest = RunManifest(
            raw_config=spec.config_dict(),
            resolved_config=resolved,
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
            analyzer=VFP2DAnalyzer(VFP2DPostProcessor.from_setup(setup)),
            capabilities=VFP2D_CAPABILITIES,
            observation_plan=plan,
        )


__all__ = ["VFP2DAnalyzer", "VFP2DBuilder", "VFP2DProgram", "VFP2DStateObservation", "VFP2DSystem"]
