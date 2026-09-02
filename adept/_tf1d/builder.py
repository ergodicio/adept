"""Logging-free builder and explicit continuous program for TF1D."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pint

from adept._tf1d.datamodel import ConfigModel
from adept._tf1d.solvers.vector_field import VF
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
from adept.core.programs import DiffraxProgram


class TwoFluid1DSystem(eqx.Module):
    """Continuous TF1D system exposing a true right-hand side."""

    vector_field: VF

    def rhs(self, t: Any, state: Any, params: Any, inputs: Any) -> Any:
        del params
        return self.vector_field(t, state, inputs)


class TwoFluid1DObservation(eqx.Module):
    """Legacy-compatible TF1D in-memory observations."""

    x_input: jax.Array
    x_output: jax.Array | None
    kxr_input: jax.Array
    kx_output: jax.Array | None
    nx: int = eqx.field(static=True)

    def __call__(self, t: Any, state: Any, inputs: Any) -> dict[str, Any]:
        del t, inputs
        observations = {}
        x_output = self.x_output
        if x_output is not None:
            observations["x"] = jax.tree.map(
                lambda field: jnp.interp(x_output, self.x_input, field),
                state,
            )
        kx_output = self.kx_output
        if kx_output is not None:

            def save_kx(field):
                transformed = jnp.fft.rfft(field, axis=0) * 2.0 / self.nx
                interpolated = jnp.interp(kx_output, self.kxr_input, transformed)
                return {"mag": jnp.abs(interpolated), "ang": jnp.angle(interpolated)}

            observations["kx"] = jax.tree.map(save_kx, state)
        return observations


def _write_units(cfg: dict[str, Any]) -> dict[str, str]:
    registry = pint.UnitRegistry()
    quantity = registry.Quantity

    n0 = quantity(cfg["units"]["normalizing_density"]).to("1/cc")
    temperature = quantity(cfg["units"]["normalizing_temperature"]).to("eV")
    wp0 = np.sqrt(n0 * registry.e**2.0 / (registry.m_e * registry.epsilon_0)).to("rad/s")
    tp0 = (1 / wp0).to("fs")
    v0 = np.sqrt(temperature / registry.m_e).to("m/s")
    x0 = (v0 / wp0).to("nm")
    c_light = quantity(1.0 * registry.c).to("m/s") / v0
    beta = (v0 / registry.c).to("dimensionless")
    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * x0).to("microns")
    box_width = "inf"
    if "ymax" in cfg["grid"]:
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * x0).to("microns")
    sim_duration = (cfg["grid"]["tmax"] * tp0).to("ps")

    log_lambda = 23.5 - np.log(n0.magnitude**0.5 * temperature.magnitude**-1.25)
    log_lambda -= (1e-5 + (np.log(temperature.magnitude) - 2) ** 2.0 / 16) ** 0.5
    nuee = quantity(2.91e-6 * n0.magnitude * log_lambda / temperature.magnitude**1.5, "Hz")

    quantities = {
        "wp0": wp0,
        "tp0": tp0,
        "n0": n0,
        "v0": v0,
        "T0": temperature,
        "c_light": c_light,
        "beta": beta,
        "x0": x0,
        "nuee": nuee,
        "logLambda_ee": log_lambda,
        "box_length": box_length,
        "box_width": box_width,
        "sim_duration": sim_duration,
    }
    unit_strings = {name: str(value) for name, value in quantities.items()}
    cfg["units"]["derived"] = unit_strings
    cfg["grid"]["beta"] = beta.magnitude
    return unit_strings


def _derive_config(cfg: dict[str, Any]) -> None:
    grid = cfg["grid"]
    grid["dx"] = grid["xmax"] / grid["nx"]
    grid["dt"] = 0.05 * grid["dx"]
    grid["nt"] = int(grid["tmax"] / grid["dt"] + 1)
    grid["tmax"] = grid["dt"] * grid["nt"]
    grid["max_steps"] = int(1e6) if grid["nt"] > 1e6 else grid["nt"] + 4

    grid["x"] = jnp.linspace(grid["xmin"] + grid["dx"] / 2, grid["xmax"] - grid["dx"] / 2, grid["nx"])
    grid["t"] = jnp.linspace(0, grid["tmax"], grid["nt"])
    grid["kx"] = jnp.fft.fftfreq(grid["nx"], d=grid["dx"]) * 2.0 * np.pi
    grid["kxr"] = jnp.fft.rfftfreq(grid["nx"], d=grid["dx"]) * 2.0 * np.pi

    one_over_kx = np.zeros_like(grid["kx"])
    one_over_kx[1:] = 1.0 / grid["kx"][1:]
    grid["one_over_kx"] = jnp.asarray(one_over_kx)
    one_over_kxr = np.zeros_like(grid["kxr"])
    one_over_kxr[1:] = 1.0 / grid["kxr"][1:]
    grid["one_over_kxr"] = jnp.asarray(one_over_kxr)

    cfg["save"]["t"]["ax"] = jnp.linspace(cfg["save"]["t"]["tmin"], cfg["save"]["t"]["tmax"], cfg["save"]["t"]["nt"])
    if "x" in cfg["save"]:
        save_x = cfg["save"]["x"]
        dx = (save_x["xmax"] - save_x["xmin"]) / save_x["nx"]
        save_x["ax"] = jnp.linspace(save_x["xmin"] + dx / 2, save_x["xmax"] - dx / 2, save_x["nx"])
    if cfg["save"].get("kx") is not None:
        save_kx = cfg["save"]["kx"]
        save_kx["ax"] = jnp.linspace(save_kx["kxmin"], save_kx["kxmax"], save_kx["nkx"])


def _runtime_inputs(drivers: dict[str, Any]) -> dict[str, Any]:
    return {
        "drivers": jax.tree.map(
            lambda value: (
                jnp.asarray(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            ),
            deepcopy(drivers),
        )
    }


class TwoFluid1DBuilder:
    """Prepare TF1D without constructing an ADEPTModule or touching MLflow."""

    def prepare(self, spec: SimulationSpec, *, key: int | jax.Array) -> PreparedSimulation:
        if spec.solver != "tf-1d":
            raise ValueError(f"TwoFluid1DBuilder cannot prepare solver {spec.solver!r}")

        normalized_key, seed, key_provenance = normalize_key(key)
        raw_config = spec.config_dict()
        validation_config = {
            **deepcopy(raw_config),
            "solver": spec.solver,
            "mlflow": {"experiment": "unused", "run": "unused"},
        }
        config_model = ConfigModel.model_validate(validation_config)
        if config_model.physics.ion.trapping.is_on or config_model.physics.electron.trapping.is_on:
            raise ValueError(
                "The new tf-1d builder does not yet support learned trapping closures; use the legacy ergoExo path"
            )

        resolved: dict[str, Any] = config_model.model_dump(exclude={"mlflow"})
        units = _write_units(resolved)
        _derive_config(resolved)
        grid = cast(dict[str, Any], resolved["grid"])
        physics = cast(dict[str, Any], resolved["physics"])
        save = cast(dict[str, Any], resolved["save"])

        state = {
            species: {
                "n": jnp.ones(grid["nx"]),
                "p": jnp.full(grid["nx"], physics[species]["T0"]),
                "u": jnp.zeros(grid["nx"]),
                "delta": jnp.zeros(grid["nx"]),
            }
            for species in ("ion", "electron")
        }
        params: dict[str, Any] = {}
        inputs = _runtime_inputs(cast(dict[str, Any], resolved["drivers"]))
        save_kx = save.get("kx")
        observation = TwoFluid1DObservation(
            x_input=grid["x"],
            x_output=save.get("x", {}).get("ax"),
            kxr_input=grid["kxr"],
            kx_output=save_kx.get("ax") if save_kx is not None else None,
            nx=grid["nx"],
        )
        program = DiffraxProgram(
            system=TwoFluid1DSystem(VF(resolved)),
            solver=diffrax.Tsit5(),
            observation=observation,
            save_times=save["t"]["ax"],
            t0=0.0,
            t1=grid["tmax"],
            dt0=grid["dt"],
            max_steps=grid["max_steps"],
        )
        capabilities = SolverCapabilities(
            execution_kind=ExecutionKind.CONTINUOUS,
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


__all__ = ["TwoFluid1DBuilder", "TwoFluid1DObservation", "TwoFluid1DSystem"]
