"""Public, side-effect-free contracts for preparing ADEPT simulations.

Importing this package deliberately uses only the Python standard library. Numerical
solver implementations, JAX, and tracking backends are loaded by registered builders
only when a simulation is prepared or executed.
"""

from .contracts import (
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

__all__ = [
    "Analyzer",
    "ContinuousSystem",
    "DiscreteSystem",
    "ExecutionKind",
    "InvalidSolverNameError",
    "JaxProgram",
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
    "solver_registry",
]
