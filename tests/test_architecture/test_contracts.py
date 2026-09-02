import json
from dataclasses import FrozenInstanceError

import pytest

from adept.core import (
    ExecutionKind,
    Placement,
    Precision,
    PreparedSimulation,
    RunManifest,
    SimulationSpec,
    SolverCapabilities,
)


def test_legacy_config_adapter_is_defensive_and_excludes_host_settings():
    legacy = {
        "solver": "tf-1d",
        "mlflow": {"experiment": "test", "run": "legacy"},
        "grid": {"nx": 8},
        "drivers": ["first"],
    }

    spec = SimulationSpec.from_legacy_config(legacy)

    legacy["grid"]["nx"] = 16
    assert spec.solver == "tf-1d"
    assert spec.config_dict() == {"grid": {"nx": 8}, "drivers": ["first"]}
    assert "mlflow" not in spec.to_dict()["config"]


def test_specification_views_cannot_change_stored_configuration():
    spec = SimulationSpec("vlasov-1d", {"grid": {"nx": 8}})

    with pytest.raises(TypeError):
        spec.config["grid"] = {"nx": 32}

    mutable_copy = spec.config_dict()
    mutable_copy["grid"]["nx"] = 32
    assert spec.config["grid"]["nx"] == 8


@pytest.mark.parametrize("reserved", ["solver", "mlflow"])
def test_specification_rejects_host_side_reserved_keys(reserved):
    with pytest.raises(ValueError, match="reserved host-side keys"):
        SimulationSpec("tf-1d", {reserved: {}})


def test_legacy_config_requires_a_solver():
    with pytest.raises(ValueError, match="missing required 'solver'"):
        SimulationSpec.from_legacy_config({"grid": {"nx": 8}})


def test_capabilities_are_typed_and_serializable():
    capabilities = SolverCapabilities(
        execution_kind=ExecutionKind.DISCRETE,
        precision=Precision.X64,
        differentiable=True,
        batchable=False,
        placements={Placement.SINGLE_DEVICE, Placement.MULTI_DEVICE},
    )

    assert capabilities.placements == frozenset({Placement.SINGLE_DEVICE, Placement.MULTI_DEVICE})
    assert capabilities.to_dict() == {
        "execution_kind": "discrete",
        "precision": "x64",
        "differentiable": True,
        "batchable": False,
        "placements": ["multi-device", "single-device"],
    }
    assert SolverCapabilities.from_dict(capabilities.to_dict()) == capabilities


def test_capabilities_require_a_supported_placement():
    with pytest.raises(ValueError, match="at least one"):
        SolverCapabilities(ExecutionKind.CONTINUOUS, placements=frozenset())


def test_manifest_is_defensive_and_serializable():
    raw = {"grid": {"nx": 8}}
    manifest = RunManifest(
        raw_config=raw,
        resolved_config={"grid": {"nx": 8, "dx": 0.25}},
        units={"x0": "1 meter"},
        seed=42,
        key_provenance="jax.random.key(42)",
        code={"adept": "abc123"},
        dependencies={"jax": "1.2.3"},
        structural_fingerprint="sha256:example",
    )

    raw["grid"]["nx"] = 16
    returned = manifest.to_dict()
    returned["resolved_config"]["grid"]["nx"] = 32

    assert manifest.raw_config["grid"]["nx"] == 8
    assert manifest.resolved_config["grid"]["nx"] == 8
    assert manifest.to_dict()["structural_fingerprint"] == "sha256:example"
    assert RunManifest.from_dict(manifest.to_dict()) == manifest


def test_specification_round_trips_through_json():
    spec = SimulationSpec("tf-1d", {"grid": {"nx": 8}}, schema_version="2")

    restored = SimulationSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

    assert restored == spec


def test_prepared_simulation_container_is_frozen():
    manifest = RunManifest(raw_config={}, resolved_config={}, structural_fingerprint="sha256:example")
    prepared = PreparedSimulation(
        program="program",
        params={},
        state={"value": 1},
        inputs={},
        manifest=manifest,
        analyzer="analyzer",
        capabilities=SolverCapabilities(ExecutionKind.CONTINUOUS),
    )

    with pytest.raises(FrozenInstanceError):
        prepared.state = {"value": 2}
