"""Public, side-effect-free contracts for preparing ADEPT simulations.

Importing this package deliberately uses only the Python standard library. Numerical
solver implementations, JAX, and tracking backends are loaded by registered builders
only when a simulation is prepared or executed.
"""

from importlib import import_module
from typing import Any

from .contracts import (
    Analyzer,
    ContinuousSystem,
    DiscreteSystem,
    ExecutionKind,
    JaxProgram,
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
from .registry import (
    InvalidSolverNameError,
    SolverAlreadyRegisteredError,
    SolverRegistry,
    UnknownSolverError,
    solver_registry,
)


def _load_two_fluid_1d_builder():
    from adept._tf1d.builder import TwoFluid1DBuilder

    return TwoFluid1DBuilder()


def _load_pic_1d_builder():
    from adept._pic1d.builder import PIC1DBuilder

    return PIC1DBuilder()


solver_registry.register_lazy("tf-1d", _load_two_fluid_1d_builder)
solver_registry.register_lazy("pic-1d", _load_pic_1d_builder)

_LAZY_ATTRIBUTES = {
    "CallableObjective": (".objectives", "CallableObjective"),
    "L2Penalty": (".objectives", "L2Penalty"),
    "LegacyVGAdapter": (".objectives", "LegacyVGAdapter"),
    "ObjectiveRun": (".objectives", "ObjectiveRun"),
    "ParameterPartition": (".objectives", "ParameterPartition"),
    "ValueAndGradResult": (".objectives", "ValueAndGradResult"),
    "WeightedSumObjective": (".objectives", "WeightedSumObjective"),
    "evaluate_objective": (".objectives", "evaluate_objective"),
    "partition_parameters": (".objectives", "partition_parameters"),
    "value_and_grad": (".objectives", "value_and_grad"),
}

__all__ = [
    "Analyzer",
    "CallableObjective",
    "ContinuousSystem",
    "DiscreteSystem",
    "ExecutionKind",
    "InvalidSolverNameError",
    "JaxProgram",
    "L2Penalty",
    "LegacyVGAdapter",
    "Objective",
    "ObjectiveResult",
    "ObjectiveRun",
    "ParameterPartition",
    "PassthroughAnalyzer",
    "Placement",
    "Precision",
    "PreparedSimulation",
    "RawResult",
    "RunManifest",
    "SimulationSpec",
    "SolverAlreadyRegisteredError",
    "SolverBuilder",
    "SolverCapabilities",
    "SolverRegistry",
    "UnknownSolverError",
    "ValueAndGradResult",
    "WeightedSumObjective",
    "evaluate_objective",
    "partition_parameters",
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
