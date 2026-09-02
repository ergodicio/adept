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
from adept._pic1d.solvers.vector_field import PIC1DVectorField
from adept.core import (
    ExecutionKind,
    PassthroughAnalyzer,
    Placement,
    Precision,
    PreparedSimulation,
    RunManifest,
    SimulationSpec,
    SolverCapabilities,
)
from adept.core.preparation import normalize_key, structural_fingerprint
from adept.core.programs import ScanProgram


class PIC1DSystem(eqx.Module):
    """Electrostatic PIC1D complete-step map."""

    vector_field: PIC1DVectorField
    t0: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def step(self, step: Any, state: Any, params: Any, inputs: Any, key: jax.Array) -> Any:
        del params, key
        return self.vector_field(self.t0 + step * self.dt, state, inputs)


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
        params: dict[str, Any] = {}
        inputs = {"drivers": _runtime_drivers(simulation.drivers)}

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
        program = ScanProgram(
            system=PIC1DSystem(vector_field, t0=simulation.grid.tmin, dt=simulation.grid.dt),
            t0=simulation.grid.tmin,
            dt=simulation.grid.dt,
            num_steps=simulation.grid.nt,
        )
        capabilities = SolverCapabilities(
            execution_kind=ExecutionKind.DISCRETE,
            precision=Precision.X64,
            differentiable=True,
            batchable=False,
            placements=frozenset({Placement.SINGLE_DEVICE}),
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
            capabilities=capabilities,
        )


__all__ = ["PIC1DBuilder", "PIC1DSystem"]
