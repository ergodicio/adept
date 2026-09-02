"""Public, side-effect-free contracts for preparing ADEPT simulations.

Importing this package deliberately uses only the Python standard library. Numerical
solver implementations, JAX, and tracking backends are loaded by registered builders
only when a simulation is prepared or executed.
"""

from .contracts import (
    Analyzer,
    ExecutionKind,
    JaxProgram,
    Placement,
    Precision,
    PreparedSimulation,
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

__all__ = [
    "Analyzer",
    "ExecutionKind",
    "InvalidSolverNameError",
    "JaxProgram",
    "Placement",
    "Precision",
    "PreparedSimulation",
    "RunManifest",
    "SimulationSpec",
    "SolverAlreadyRegisteredError",
    "SolverBuilder",
    "SolverCapabilities",
    "SolverRegistry",
    "UnknownSolverError",
    "solver_registry",
]
