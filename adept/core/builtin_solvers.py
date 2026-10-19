"""Import-light capability declarations for ADEPT's built-in solver builders."""

from .contracts import ExecutionKind, Placement, Precision, SolverCapabilities

FARSIGHT1D_CAPABILITIES = SolverCapabilities(
    execution_kind=ExecutionKind.DISCRETE,
    precision=Precision.X64,
    differentiable=True,
    batchable=False,
    placements=frozenset({Placement.SINGLE_DEVICE}),
)

PIC1D_CAPABILITIES = SolverCapabilities(
    execution_kind=ExecutionKind.DISCRETE,
    precision=Precision.X64,
    differentiable=True,
    batchable=False,
    placements=frozenset({Placement.SINGLE_DEVICE}),
)

TF1D_CAPABILITIES = SolverCapabilities(
    execution_kind=ExecutionKind.CONTINUOUS,
    precision=Precision.X64,
    differentiable=True,
    batchable=False,
    placements=frozenset({Placement.SINGLE_DEVICE}),
)

VFP2D_CAPABILITIES = SolverCapabilities(
    execution_kind=ExecutionKind.DISCRETE,
    precision=Precision.X64,
    differentiable=True,
    batchable=False,
    placements=frozenset({Placement.SINGLE_DEVICE, Placement.MULTI_DEVICE}),
)

VLASOV1D_CAPABILITIES = SolverCapabilities(
    execution_kind=ExecutionKind.DISCRETE,
    precision=Precision.X64,
    differentiable=True,
    batchable=False,
    placements=frozenset({Placement.SINGLE_DEVICE, Placement.MULTI_DEVICE}),
)

__all__ = [
    "FARSIGHT1D_CAPABILITIES",
    "PIC1D_CAPABILITIES",
    "TF1D_CAPABILITIES",
    "VFP2D_CAPABILITIES",
    "VLASOV1D_CAPABILITIES",
]
