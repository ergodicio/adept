from __future__ import annotations

from dataclasses import dataclass

import pytest

from adept.core import (
    AcceleratorKind,
    CapabilityMismatchError,
    ExecutionFeature,
    ExecutionKind,
    ExecutionState,
    ExecutorCapabilities,
    LocalExecutor,
    Placement,
    Precision,
    PreparedSimulation,
    RawResult,
    Report,
    ResourceRequirements,
    RunManifest,
    RunPlan,
    RunRequest,
    ServiceReference,
    SimulationSpec,
    SolverCapabilities,
    SolverRegistry,
)


class Program:
    def __call__(self, params, state, inputs, key):
        del params, inputs, key
        return RawResult(state + 1, {}, {}, "ok", {})


class Analyzer:
    def analyze(self, result, manifest):
        del manifest
        return Report(result=result.final_state)


@dataclass
class FakeBuilder:
    capabilities: SolverCapabilities
    calls: list[str]

    def prepare(self, spec, *, key):
        del key
        self.calls.append(spec.solver)
        return PreparedSimulation(
            program=Program(),
            params={},
            state=spec.config["initial"],
            inputs={},
            manifest=RunManifest(
                raw_config=spec.config,
                resolved_config=spec.config,
                structural_fingerprint="sha256:fake",
            ),
            analyzer=Analyzer(),
            capabilities=self.capabilities,
        )


def _direct(prepared, key):
    return prepared.program(prepared.params, prepared.state, prepared.inputs, key)


def _registry(*, differentiable: bool = True):
    capabilities = SolverCapabilities(
        ExecutionKind.DISCRETE,
        differentiable=differentiable,
        placements={Placement.SINGLE_DEVICE},
    )
    builder = FakeBuilder(capabilities, [])
    registry = SolverRegistry()
    registry.register("fake", builder, capabilities=capabilities)
    return registry, builder


def test_local_executor_runs_a_wire_format_copy_and_exposes_lifecycle(tmp_path):
    registry, builder = _registry()
    plan = RunPlan(
        SimulationSpec("fake", {"initial": 2}),
        seed=7,
        run=RunRequest(run_id="local-run"),
        artifact_sink=ServiceReference("directory", {"root": str(tmp_path / "artifacts")}),
    )

    with LocalExecutor(registry=registry, execute_prepared=_direct) as executor:
        handle = executor.submit(plan)
        completed = executor.result(handle)

        assert executor.status(handle) is ExecutionState.SUCCEEDED
        assert executor.cancel(handle) is False

    assert completed.raw_result.final_state == 3
    assert completed.report.result == 3
    assert completed.handle.run_id == "local-run"
    assert builder.calls == ["fake"]
    assert (tmp_path / "artifacts").is_dir()


def test_preflight_rejects_capability_mismatches_before_loading_builder():
    calls: list[str] = []
    solver_capabilities = SolverCapabilities(ExecutionKind.DISCRETE)
    registry = SolverRegistry()

    def load_builder():
        calls.append("loaded")
        return FakeBuilder(solver_capabilities, [])

    registry.register_lazy("fake", load_builder, capabilities=solver_capabilities)
    cpu_only = ExecutorCapabilities(
        placements={Placement.SINGLE_DEVICE},
        precisions={Precision.DEFAULT},
        accelerators={AcceleratorKind.CPU},
    )
    plan = RunPlan(
        SimulationSpec("fake", {"initial": 0}),
        resources=ResourceRequirements(accelerator=AcceleratorKind.GPU),
    )

    with LocalExecutor(registry=registry, capabilities=cpu_only, execute_prepared=_direct) as executor:
        with pytest.raises(CapabilityMismatchError, match="accelerator 'gpu'"):
            executor.validate(plan)

    assert calls == []


def test_preflight_rejects_solver_and_service_mismatches():
    registry, _ = _registry(differentiable=False)
    plan = RunPlan(
        SimulationSpec("fake", {"initial": 0}),
        resources=ResourceRequirements(features={ExecutionFeature.DIFFERENTIABLE}),
        tracker=ServiceReference("null"),
        artifact_sink=ServiceReference("mlflow"),
    )

    with LocalExecutor(registry=registry, execute_prepared=_direct) as executor:
        with pytest.raises(CapabilityMismatchError) as caught:
            executor.validate(plan)

    assert "solver is not differentiable" in str(caught.value)
    assert "MLflow artifact sink requires an MLflow tracker" in str(caught.value)


def test_failed_local_execution_is_retrievable_by_status():
    registry, builder = _registry()

    def fail(prepared, key):
        del prepared, key
        raise RuntimeError("solver failed")

    with LocalExecutor(registry=registry, execute_prepared=fail) as executor:
        handle = executor.submit(RunPlan(SimulationSpec("fake", {"initial": 0})))
        with pytest.raises(RuntimeError, match="solver failed"):
            executor.result(handle)
        assert executor.status(handle) is ExecutionState.FAILED

    assert builder.calls == ["fake"]
