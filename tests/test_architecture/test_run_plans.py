from __future__ import annotations

import json

import pytest

from adept.core import (
    AcceleratorKind,
    ExecutionFeature,
    FailurePolicy,
    Placement,
    Precision,
    ResourceRequirements,
    RunHandle,
    RunPlan,
    RunRequest,
    ServiceReference,
    SimulationSpec,
)


def test_run_plan_round_trips_as_canonical_json_and_has_a_stable_fingerprint():
    plan = RunPlan(
        simulation=SimulationSpec("tf-1d", {"grid": {"nx": 8}}),
        seed=42,
        resources=ResourceRequirements(
            precision=Precision.X64,
            accelerator=AcceleratorKind.CPU,
            features={ExecutionFeature.DIFFERENTIABLE},
        ),
        run=RunRequest(
            experiment="scan",
            name="case-1",
            run_id="run-1",
            parent=RunHandle("parent-1", "mlflow"),
            tags={"parameter": "0.5"},
        ),
        tracker=ServiceReference("mlflow", {"tracking_uri": "https://tracking.example"}),
        artifact_sink=ServiceReference("directory", {"root": "./results"}),
        tracking_failure_policy=FailurePolicy.BEST_EFFORT,
    )

    encoded = plan.to_json()
    restored = RunPlan.from_json(encoded)

    assert json.loads(encoded) == plan.to_dict()
    assert restored == plan
    assert restored.fingerprint == plan.fingerprint
    assert ExecutionFeature.ARTIFACT_ACCESS in plan.required_features


def test_run_plan_rejects_non_json_values():
    with pytest.raises(TypeError, match="finite JSON"):
        RunPlan(SimulationSpec("tf-1d", {"value": object()}))


@pytest.mark.parametrize(
    "field",
    [
        "apikey",
        "api-key",
        "api_key",
        "OpenAIAPIKey",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "client_secret_id",
        "databasePassword",
        "auth_token",
        "credential_value",
        "ssh_private_key",
    ],
)
def test_run_plan_rejects_common_separator_insensitive_credential_fields(field):
    with pytest.raises(ValueError, match="credential material"):
        RunPlan(SimulationSpec("tf-1d", {field: "do-not-serialize"}))


def test_service_references_reject_credentials_but_allow_external_profiles():
    with pytest.raises(ValueError, match="credential material"):
        ServiceReference("mlflow", {"password": "do-not-serialize"})

    with pytest.raises(ValueError, match="credential material in a URI"):
        ServiceReference("mlflow", {"tracking_uri": "https://user:password@tracking.example"})

    with pytest.raises(ValueError, match="credential material in a URI"):
        ServiceReference("mlflow", {"tracking_uri": "https://tracking.example?api_token=secret"})

    assert ServiceReference("aws", {"profile": "ergodic-archis"}).config["profile"] == "ergodic-archis"


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        (ResourceRequirements, None),
        (
            lambda: ResourceRequirements(
                placement=Placement.SINGLE_DEVICE,
                devices_per_host=2,
            ),
            "exactly one",
        ),
        (
            lambda: ResourceRequirements(
                placement=Placement.MULTI_DEVICE,
                devices_per_host=1,
            ),
            "at least two",
        ),
        (
            lambda: ResourceRequirements(
                placement=Placement.MULTI_HOST,
                hosts=1,
            ),
            "at least two",
        ),
    ],
)
def test_resource_topologies_are_validated(requirements, message):
    if message is None:
        assert requirements().placement is Placement.SINGLE_DEVICE
    else:
        with pytest.raises(ValueError, match=message):
            requirements()


def test_multi_host_topology_implies_distributed_jax():
    requirements = ResourceRequirements(
        placement=Placement.MULTI_HOST,
        hosts=2,
        devices_per_host=4,
    )

    assert requirements.required_features == frozenset({ExecutionFeature.DISTRIBUTED_JAX})


def test_serialized_plan_rejects_unknown_fields():
    serialized = RunPlan(SimulationSpec("tf-1d", {})).to_dict()
    serialized["typo"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        RunPlan.from_dict(serialized)

    serialized.pop("typo")
    serialized["schema_version"] = "999"
    with pytest.raises(ValueError, match="unsupported RunPlan schema_version"):
        RunPlan.from_dict(serialized)
