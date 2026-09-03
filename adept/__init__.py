"""ADEPT's public API.

The architecture contracts are safe to import in configuration and submission
processes. Legacy simulation and solver symbols remain available, but are resolved
lazily because they intentionally bring in JAX and MLflow.
"""

from importlib import import_module
from typing import Any

from .core import (
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
    SolverRegistry,
    solver_registry,
)

_LAZY_ATTRIBUTES = {
    "ADEPTModule": ("._base_", "ADEPTModule"),
    "CallableObjective": (".core.objectives", "CallableObjective"),
    "L2Penalty": (".core.objectives", "L2Penalty"),
    "LegacyVGAdapter": (".core.objectives", "LegacyVGAdapter"),
    "MlflowLoggingModule": (".mlflow_logging", "MlflowLoggingModule"),
    "WeightedSumObjective": (".core.objectives", "WeightedSumObjective"),
    "ergoExo": ("._base_", "ergoExo"),
    "evaluate_objective": (".core.objectives", "evaluate_objective"),
    "hermite_legendre_1d": (".hermite_legendre_1d", None),
    "hermite_poisson_1d": (".hermite_poisson_1d", None),
    "lpse2d": (".lpse2d", None),
    "partition_parameters": (".core.objectives", "partition_parameters"),
    "vfp2d": (".vfp2d", None),
    "vlasov1d": (".vlasov1d", None),
    "vlasov2d": (".vlasov2d", None),
    "value_and_grad": (".core.objectives", "value_and_grad"),
}

__all__ = [
    "ADEPTModule",
    "Analyzer",
    "CallableObjective",
    "ContinuousSystem",
    "DiscreteSystem",
    "ExecutionKind",
    "JaxProgram",
    "L2Penalty",
    "LegacyVGAdapter",
    "MlflowLoggingModule",
    "Objective",
    "ObjectiveResult",
    "PassthroughAnalyzer",
    "Placement",
    "Precision",
    "PreparedSimulation",
    "RawResult",
    "RunManifest",
    "SimulationSpec",
    "SolverBuilder",
    "SolverCapabilities",
    "SolverRegistry",
    "WeightedSumObjective",
    "ergoExo",
    "evaluate_objective",
    "hermite_legendre_1d",
    "hermite_poisson_1d",
    "lpse2d",
    "partition_parameters",
    "solver_registry",
    "value_and_grad",
    "vfp2d",
    "vlasov1d",
    "vlasov2d",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_ATTRIBUTES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
