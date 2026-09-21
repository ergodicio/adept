"""Local host orchestration for prepared simulations and runtime services."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import MaterializedResult, PreparedSimulation
from .tracking import (
    ArtifactReceipt,
    ArtifactSink,
    FailurePolicy,
    MetricEvent,
    NullArtifactSink,
    NullTracker,
    Report,
    RunHandle,
    RunRequest,
    RunStatus,
    Tracker,
)


def _call_program(program: Any, params: Any, state: Any, inputs: Any, key: Any) -> Any:
    return program(params, state, inputs, key)


def _default_execute(prepared: PreparedSimulation, key: Any) -> Any:
    # Equinox is intentionally imported at execution time, not when host-side
    # contracts or run plans are inspected.
    import equinox as eqx

    return eqx.filter_jit(_call_program)(
        prepared.program,
        prepared.params,
        prepared.state,
        prepared.inputs,
        key,
    )


def _synchronize(value: Any, seen: set[int] | None = None) -> None:
    """Wait for asynchronous array leaves without importing JAX."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    block_until_ready = getattr(value, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _synchronize(child, seen)
        return
    if isinstance(value, (tuple, list)):
        for child in value:
            _synchronize(child, seen)


@dataclass(frozen=True, slots=True)
class HostRunResult:
    """Computed and analyzed result plus explicit persistence outcomes."""

    raw_result: Any
    materialized_result: MaterializedResult | None
    report: Report
    handle: RunHandle
    artifacts: tuple[ArtifactReceipt, ...]
    run_time_seconds: float
    analysis_time_seconds: float
    tracking_errors: tuple[str, ...] = ()


def _tracking_error(operation: str, error: Exception) -> str:
    return f"{operation}: {type(error).__name__}: {error}"


def run_prepared(
    prepared: PreparedSimulation,
    *,
    key: Any,
    request: RunRequest | None = None,
    tracker: Tracker | None = None,
    artifact_sink: ArtifactSink | None = None,
    tracking_failure_policy: FailurePolicy = FailurePolicy.STRICT,
    execute: Callable[[PreparedSimulation, Any], Any] | None = None,
) -> HostRunResult:
    """Execute, synchronize, analyze, persist, and terminate one prepared run.

    Artifact operations are always strict: an upload or verification error marks an
    active tracked run failed and is raised to the caller.  Tracker-only errors obey
    ``tracking_failure_policy`` so best-effort telemetry cannot discard a successful
    computed result.
    """

    policy = FailurePolicy(tracking_failure_policy)
    tracker = tracker if tracker is not None else NullTracker()
    artifact_sink = artifact_sink if artifact_sink is not None else NullArtifactSink()
    execute = execute if execute is not None else _default_execute
    request = (request or RunRequest()).with_tags(
        {
            "adept.execution_kind": prepared.capabilities.execution_kind.value,
            "adept.structural_fingerprint": prepared.manifest.structural_fingerprint,
        }
    )

    # Artifact configuration is strict and independent of telemetry policy, so fail
    # its detectable errors before creating a run or launching numerical work.
    artifact_sink.preflight()
    tracking_errors: list[str] = []
    tracking_active = True
    try:
        tracker.preflight(request)
    except Exception as error:
        if policy is FailurePolicy.STRICT:
            raise
        tracking_errors.append(_tracking_error("tracker preflight", error))
        tracking_active = False

    handle: RunHandle
    if tracking_active:
        try:
            handle = tracker.start(request)
        except Exception as error:
            if policy is FailurePolicy.STRICT:
                raise
            tracking_errors.append(_tracking_error("tracker start", error))
            tracking_active = False
            handle = NullTracker().start(request)
    else:
        handle = NullTracker().start(request)

    try:
        artifact_sink.validate(handle)
        started = time.perf_counter()
        raw_result = execute(prepared, key)
        _synchronize(raw_result)
        run_time_seconds = time.perf_counter() - started

        started = time.perf_counter()
        materialized_result = None
        analysis_input = raw_result
        if prepared.observation_plan is not None:
            materialized_result = raw_result.materialize(prepared.observation_plan.materialization)
            if materialized_result is None:
                raise RuntimeError("run_prepared must run analysis on the selected materialization host")
            analysis_input = materialized_result
        report = prepared.analyzer.analyze(analysis_input, prepared.manifest)
        analysis_time_seconds = time.perf_counter() - started
        if not isinstance(report, Report):
            raise TypeError("Prepared analyzers must return adept.Report")

        if tracking_active:
            events = (
                MetricEvent(
                    {
                        "adept.run_time_seconds": run_time_seconds,
                        "adept.analysis_time_seconds": analysis_time_seconds,
                    }
                ),
                *report.metrics,
            )
            try:
                tracker.log_metrics(handle, events)
            except Exception as error:
                if policy is FailurePolicy.STRICT:
                    raise
                tracking_errors.append(_tracking_error("tracker metrics", error))

        receipts: list[ArtifactReceipt] = []
        for artifact in report.artifacts:
            receipt = artifact_sink.put(handle, artifact)
            artifact_sink.verify(handle, receipt)
            receipts.append(receipt)

        if tracking_active:
            try:
                tracker.finish(handle, RunStatus.FINISHED)
            except Exception as error:
                if policy is FailurePolicy.STRICT:
                    raise
                tracking_errors.append(_tracking_error("tracker finish", error))
    except Exception as primary_error:
        if tracking_active:
            try:
                tracker.finish(handle, RunStatus.FAILED, error=str(primary_error))
            except Exception as tracking_error:
                primary_error.add_note(_tracking_error("tracker failure status", tracking_error))
        raise

    return HostRunResult(
        raw_result=raw_result,
        materialized_result=materialized_result,
        report=report,
        handle=handle,
        artifacts=tuple(receipts),
        run_time_seconds=run_time_seconds,
        analysis_time_seconds=analysis_time_seconds,
        tracking_errors=tuple(tracking_errors),
    )


__all__ = ["HostRunResult", "run_prepared"]
