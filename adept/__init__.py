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
    "MlflowLoggingModule": (".mlflow_logging", "MlflowLoggingModule"),
    "ergoExo": ("._base_", "ergoExo"),
    "hermite_legendre_1d": (".hermite_legendre_1d", None),
    "hermite_poisson_1d": (".hermite_poisson_1d", None),
    "lpse2d": (".lpse2d", None),
    "vfp2d": (".vfp2d", None),
    "vlasov1d": (".vlasov1d", None),
    "vlasov2d": (".vlasov2d", None),
}

__all__ = [
    "ADEPTModule",
    "Analyzer",
    "ContinuousSystem",
    "DiscreteSystem",
    "ExecutionKind",
    "JaxProgram",
    "MlflowLoggingModule",
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
    "ergoExo",
    "hermite_legendre_1d",
    "hermite_poisson_1d",
    "lpse2d",
    "solver_registry",
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
