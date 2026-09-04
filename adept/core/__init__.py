"""Public, side-effect-free contracts for preparing ADEPT simulations.

Importing this package deliberately uses only the Python standard library. Numerical
solver implementations, JAX, and tracking backends are loaded by registered builders
only when a simulation is prepared or executed.
"""

from importlib import import_module
from typing import Any

from .builtin_solvers import PIC1D_CAPABILITIES, TF1D_CAPABILITIES
from .contracts import (
    Analyzer,
    ContinuousSystem,
    DiscreteSystem,
    ExecutionKind,
    JaxProgram,
    MaterializedResult,
    Objective,
    ObjectiveResult,
    PassthroughAnalyzer,
    Placement,
    Precision,
    PreparedSimulation,
    RawResult,
    RunManifest,
    SimulationSpec,
    SolverBuilder,
    SolverCapabilities,
)
from .executors import (
    CapabilityMismatchError,
    ExecutionHandle,
    ExecutionState,
    Executor,
    ExecutorCapabilities,
    LocalExecutor,
)
from .observations import (
    MaterializationTarget,
    ObservationCollective,
    ObservationFunction,
    ObservationLeaf,
    ObservationPlacement,
    ObservationPlan,
    ObservationReduction,
    ObservationRetention,
    ObservationSchedule,
    ObservationSchema,
    ObservationSpec,
    ScheduleKind,
)
from .registry import (
    InvalidSolverNameError,
    SolverAlreadyRegisteredError,
    SolverRegistry,
    UnknownSolverError,
    solver_registry,
)
from .run_plans import (
    AcceleratorKind,
    ExecutionFeature,
    ResourceRequirements,
    RunPlan,
    ServiceReference,
)
from .runtime import HostRunResult, run_prepared
from .tracking import (
    Artifact,
    ArtifactReceipt,
    ArtifactSink,
    DirectoryArtifactSink,
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


def _load_two_fluid_1d_builder():
    from adept._tf1d.builder import TwoFluid1DBuilder

    return TwoFluid1DBuilder()


def _load_pic_1d_builder():
    from adept._pic1d.builder import PIC1DBuilder

    return PIC1DBuilder()


solver_registry.register_lazy(
    "tf-1d",
    _load_two_fluid_1d_builder,
    capabilities=TF1D_CAPABILITIES,
)
solver_registry.register_lazy(
    "pic-1d",
    _load_pic_1d_builder,
    capabilities=PIC1D_CAPABILITIES,
)

_LAZY_ATTRIBUTES = {
    "CallableObjective": (".objectives", "CallableObjective"),
    "L2Penalty": (".objectives", "L2Penalty"),
    "LegacyVGAdapter": (".objectives", "LegacyVGAdapter"),
    "MLflowArtifactSink": (".tracking_mlflow", "MLflowArtifactSink"),
    "MLflowTracker": (".tracking_mlflow", "MLflowTracker"),
    "ObjectiveRun": (".objectives", "ObjectiveRun"),
    "ParameterPartition": (".objectives", "ParameterPartition"),
    "ValueAndGradResult": (".objectives", "ValueAndGradResult"),
    "WeightedSumObjective": (".objectives", "WeightedSumObjective"),
    "evaluate_objective": (".objectives", "evaluate_objective"),
    "materialize_result": (".materialization", "materialize_result"),
    "partition_parameters": (".objectives", "partition_parameters"),
    "value_and_grad": (".objectives", "value_and_grad"),
}

__all__ = [
    "AcceleratorKind",
    "Analyzer",
    "Artifact",
    "ArtifactReceipt",
    "ArtifactSink",
    "CallableObjective",
    "CapabilityMismatchError",
    "ContinuousSystem",
    "DirectoryArtifactSink",
    "DiscreteSystem",
    "ExecutionFeature",
    "ExecutionHandle",
    "ExecutionKind",
    "ExecutionState",
    "Executor",
    "ExecutorCapabilities",
    "FailurePolicy",
    "HostRunResult",
    "InvalidSolverNameError",
    "JaxProgram",
    "L2Penalty",
    "LegacyVGAdapter",
    "LocalExecutor",
    "MLflowArtifactSink",
    "MLflowTracker",
    "MaterializationTarget",
    "MaterializedResult",
    "MetricEvent",
    "NullArtifactSink",
    "NullTracker",
    "Objective",
    "ObjectiveResult",
    "ObjectiveRun",
    "ObservationCollective",
    "ObservationFunction",
    "ObservationLeaf",
    "ObservationPlacement",
    "ObservationPlan",
    "ObservationReduction",
    "ObservationRetention",
    "ObservationSchedule",
    "ObservationSchema",
    "ObservationSpec",
    "ParameterPartition",
    "PassthroughAnalyzer",
    "Placement",
    "Precision",
    "PreparedSimulation",
    "RawResult",
    "Report",
    "ResourceRequirements",
    "RunHandle",
    "RunManifest",
    "RunPlan",
    "RunRequest",
    "RunStatus",
    "ScheduleKind",
    "ServiceReference",
    "SimulationSpec",
    "SolverAlreadyRegisteredError",
    "SolverBuilder",
    "SolverCapabilities",
    "SolverRegistry",
    "Tracker",
    "UnknownSolverError",
    "ValueAndGradResult",
    "WeightedSumObjective",
    "evaluate_objective",
    "materialize_result",
    "partition_parameters",
    "run_prepared",
    "solver_registry",
    "value_and_grad",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_ATTRIBUTES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
