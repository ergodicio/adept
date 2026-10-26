"""Logging-free builder and explicit discrete program for electrostatic PIC1D."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from adept._pic1d.datamodel import PIC1DConfig
from adept._pic1d.helpers import _initialize_particles_
from adept._pic1d.simulation import sim_from_config
from adept._pic1d.solvers.pushers.shape import deposit
from adept._pic1d.solvers.vector_field import PIC1DVectorField
from adept.core import (
    ObservationPlan,
    ObservationSchedule,
    PassthroughAnalyzer,
    PreparedSimulation,
    RunManifest,
    SimulationSpec,
)
from adept.core.builtin_solvers import PIC1D_CAPABILITIES
from adept.core.observations_jax import infer_observation_spec, with_step_schedule
from adept.core.preparation import normalize_key, structural_fingerprint
from adept.core.programs import ScanProgram


class PIC1DSystem(eqx.Module):
    """Electrostatic PIC1D complete-step map."""

    vector_field: PIC1DVectorField
    t0: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def step(self, step: Any, state: Any, params: Any, inputs: Any, key: jax.Array) -> Any:
        del key
        return self.vector_field(self.t0 + step * self.dt, state, eqx.combine(params, inputs))


class PICFieldsObservation(eqx.Module):
    """Grid fields and deposited particle moments for every species."""

    species_names: tuple[str, ...] = eqx.field(static=True)
    particle_shape: str = eqx.field(static=True)
    nx: int = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    xmin: float = eqx.field(static=True)

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, Any]:
        del t, inputs
        result = {}
        for species_name in self.species_names:
            position = state[f"x_{species_name}"]
            velocity = state[f"v_{species_name}"]
            weight = state[f"w_{species_name}"]
            result[species_name] = {
                "n": deposit(position, weight, self.nx, self.dx, self.xmin, self.particle_shape),
                "j": deposit(
                    position,
                    weight * velocity,
                    self.nx,
                    self.dx,
                    self.xmin,
                    self.particle_shape,
                ),
                "P": deposit(
                    position,
                    weight * velocity * velocity,
                    self.nx,
                    self.dx,
                    self.xmin,
                    self.particle_shape,
                ),
            }
        result.update(
            e=state["e"],
            de=state["de"],
            a=state["a"],
            prev_a=state["prev_a"],
            pond=-0.5 * cast(jax.Array, jnp.gradient(state["a"] ** 2, self.dx))[1:-1],
        )
        return result


class PICScalarsObservation(eqx.Module):
    """Reduced field and per-species particle invariants."""

    species_names: tuple[str, ...] = eqx.field(static=True)
    masses: tuple[float, ...] = eqx.field(static=True)

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, Any]:
        del t, inputs
        result = {}
        for species_name, mass in zip(self.species_names, self.masses, strict=True):
            velocity = state[f"v_{species_name}"]
            weight = state[f"w_{species_name}"]
            result[f"mean_KE_{species_name}"] = 0.5 * mass * jnp.sum(weight * velocity * velocity)
            result[f"mean_p_{species_name}"] = mass * jnp.sum(weight * velocity)
            result[f"sum_w_{species_name}"] = jnp.sum(weight)
        result["mean_e2"] = jnp.mean(state["e"] ** 2)
        result["mean_de2"] = jnp.mean(state["de"] ** 2)
        result["mean_a2"] = jnp.mean(state["a"][1:-1] ** 2)
        return result


class PICDistributionObservation(eqx.Module):
    """Particle phase-space coordinates for one species."""

    species_name: str = eqx.field(static=True)

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, Any]:
        del t, inputs
        return {
            "x": state[f"x_{self.species_name}"],
            "v": state[f"v_{self.species_name}"],
        }


def _write_units(resolved: dict[str, Any], simulation) -> dict[str, str]:
    norm = simulation.plasma_norm
    grid = simulation.grid
    quantities = {
        "wp0": (1 / norm.tau).to("rad/s"),
        "tp0": norm.tau.to("fs"),
        "n0": norm.n0.to("1/cc"),
        "v0": norm.v0.to("m/s"),
        "T0": norm.T0.to("eV"),
        "x0": norm.L0.to("nm"),
        "c_light": norm.speed_of_light_norm(),
        "box_length": ((grid.xmax - grid.xmin) * norm.L0).to("microns"),
        "sim_duration": (grid.tmax * norm.tau).to("ps"),
        "ppc": simulation.ppc,
        "particle_shape": simulation.particle_shape,
    }
    unit_strings = {name: str(value) for name, value in quantities.items()}
    resolved["units"]["derived"] = unit_strings
    resolved["grid"]["beta"] = 1.0
    return unit_strings


def _derive_config(resolved: dict[str, Any], simulation, *, seed: int) -> dict[str, Any]:
    grid = simulation.grid
    resolved_grid = resolved["grid"]
    resolved_grid.update(asdict(grid))
    resolved_grid["ppc"] = int(simulation.ppc)
    resolved_grid["particle_shape"] = simulation.particle_shape

    for save_value in resolved.get("save", {}).values():
        if "t" in save_value:
            save_value["t"].setdefault("tmin", grid.tmin)
            save_value["t"].setdefault("tmax", grid.tmax)
        else:
            for label_config in save_value.values():
                if isinstance(label_config, dict) and "t" in label_config:
                    label_config["t"].setdefault("tmin", grid.tmin)
                    label_config["t"].setdefault("tmax", grid.tmax)

    loaded = _initialize_particles_(resolved, simulation, seed=seed)
    species_params = {}
    total_density = np.zeros(grid.nx)
    for species in simulation.species:
        total_density += loaded[species.name][3]
        species_params[species.name] = {
            "charge": species.charge,
            "mass": species.mass,
            "charge_to_mass": species.charge / species.mass,
        }
    resolved_grid["species_params"] = species_params
    resolved_grid["n_prof_total"] = total_density
    if resolved["density"].get("quasineutrality", True):
        ion_charge = np.zeros_like(total_density)
        for species in simulation.species:
            ion_charge -= species.charge * loaded[species.name][3]
        resolved_grid["ion_charge"] = ion_charge
    else:
        resolved_grid["ion_charge"] = np.zeros_like(total_density)
    resolved["grid"] = resolved_grid
    return loaded


def _initial_state(loaded: dict[str, Any], grid) -> dict[str, jax.Array]:
    state = {}
    for name, (x, velocity, weight, _density, _velocity_axis) in loaded.items():
        state[f"x_{name}"] = jnp.asarray(x)
        state[f"v_{name}"] = jnp.asarray(velocity)
        state[f"w_{name}"] = jnp.asarray(weight)
    state["e"] = jnp.zeros(grid.nx)
    state["de"] = jnp.zeros(grid.nx)
    for field_name in ("a", "prev_a", "da"):
        state[field_name] = jnp.zeros(grid.nx + 2)
    return state


def _runtime_drivers(drivers):
    return jax.tree.map(
        lambda value: jnp.asarray(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value,
        drivers,
    )


def _pic_observation_plan(resolved: dict[str, Any], simulation, state: Any, inputs: Any) -> ObservationPlan:
    grid = simulation.grid
    species_names = tuple(species.name for species in simulation.species)
    fields = PICFieldsObservation(
        species_names=species_names,
        particle_shape=simulation.particle_shape,
        nx=grid.nx,
        dx=grid.dx,
        xmin=grid.xmin,
    )
    specs = []
    for save_name, save_config in resolved.get("save", {}).items():
        if save_name.startswith("fields"):
            function = fields
            entries = ((save_name, save_config),)
        elif save_name in species_names:
            function = PICDistributionObservation(save_name)
            entries = tuple((f"{save_name}.{label}", label_config) for label, label_config in save_config.items())
        else:
            raise ValueError(f"unsupported PIC1D legacy save type: {save_name}")

        for observation_name, observation_config in entries:
            schedule = ObservationSchedule.from_legacy_time_config(observation_config["t"])
            spec = infer_observation_spec(
                observation_name,
                function,
                schedule,
                t=grid.tmin,
                state=state,
                inputs=inputs,
            )
            specs.append(
                with_step_schedule(
                    spec,
                    t0=grid.tmin,
                    dt=grid.dt,
                    num_steps=grid.nt,
                )
            )

    default_schedule = ObservationSchedule.every_steps(1, stop=max(0, grid.nt - 1))
    default_function = PICScalarsObservation(
        species_names=species_names,
        masses=tuple(species.mass for species in simulation.species),
    )
    specs.append(
        infer_observation_spec(
            "default",
            default_function,
            default_schedule,
            t=grid.tmin,
            state=state,
            inputs=inputs,
        )
    )
    return ObservationPlan(specs)


class PIC1DBuilder:
    """Prepare the electrostatic PIC1D program without an ADEPTModule."""

    def prepare(self, spec: SimulationSpec, *, key: int | jax.Array) -> PreparedSimulation:
        if spec.solver != "pic-1d":
            raise ValueError(f"PIC1DBuilder cannot prepare solver {spec.solver!r}")

        normalized_key, seed, key_provenance = normalize_key(key)
        raw_config = spec.config_dict()
        validation_config = {
            **deepcopy(raw_config),
            "solver": spec.solver,
            "mlflow": {"experiment": "unused", "run": "unused"},
        }
        config_model = PIC1DConfig.model_validate(validation_config)
        if config_model.drivers.ey:
            raise ValueError(
                "The new pic-1d builder currently supports electrostatic runs only; "
                "use the legacy ergoExo path for transverse ey drivers"
            )
        if config_model.drivers.ex_stochastic is not None:
            raise ValueError(
                "The new pic-1d builder does not yet support stochastic ex forcing; use the legacy ergoExo path"
            )

        simulation = sim_from_config(config_model)
        resolved: dict[str, Any] = config_model.model_dump(exclude={"mlflow"})
        units = _write_units(resolved, simulation)
        loaded = _derive_config(resolved, simulation, seed=seed)
        state = _initial_state(loaded, simulation.grid)
        params, inputs = eqx.partition({"drivers": _runtime_drivers(simulation.drivers)}, False)

        resolved_grid = cast(dict[str, Any], resolved["grid"])
        program_config = {
            "grid": {
                "species_params": resolved_grid["species_params"],
                "particle_shape": resolved_grid["particle_shape"],
                "ion_charge": resolved_grid["ion_charge"],
                "beta": resolved_grid["beta"],
            },
            "drivers": resolved["drivers"],
            "terms": resolved["terms"],
        }
        vector_field = PIC1DVectorField(program_config, simulation.grid, simulation.drivers)
        observation_plan = _pic_observation_plan(resolved, simulation, state, inputs)
        program = ScanProgram.from_observation_plan(
            system=PIC1DSystem(vector_field, t0=simulation.grid.tmin, dt=simulation.grid.dt),
            plan=observation_plan,
            state=state,
            inputs=inputs,
            t0=simulation.grid.tmin,
            dt=simulation.grid.dt,
            num_steps=simulation.grid.nt,
        )
        manifest = RunManifest(
            raw_config=raw_config,
            resolved_config=resolved,
            units=units,
            seed=seed,
            key_provenance=key_provenance,
            structural_fingerprint=structural_fingerprint(spec, program, params, state, inputs, normalized_key),
        )
        return PreparedSimulation(
            program=program,
            params=params,
            state=state,
            inputs=inputs,
            manifest=manifest,
            analyzer=PassthroughAnalyzer(),
            capabilities=PIC1D_CAPABILITIES,
            observation_plan=observation_plan,
        )


__all__ = [
    "PIC1DBuilder",
    "PIC1DSystem",
    "PICDistributionObservation",
    "PICFieldsObservation",
    "PICScalarsObservation",
]
