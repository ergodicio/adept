#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Abstract factory for FP relaxation vector fields.

Provides dependency injection for solver-specific logic, keeping fp_relaxation
agnostic to vlasov1d/vfp1d implementation details.
"""

from abc import abstractmethod

import equinox as eqx
from equinox import AbstractClassVar

from .registry import VelocityGrid


class AbstractFPRelaxationVectorFieldFactory(eqx.Module):
    """
    Abstract factory for creating FP collision vector fields.

    Each solver (vlasov1d, vfp1d) provides a concrete implementation
    that encapsulates model/scheme instantiation and adapter creation.

    Subclasses must:
    1. Define `_spherical` class variable (True for spherical grids, False for Cartesian)
    2. Pass `model_names` and `scheme_names` at initialization
    3. Implement `make_vector_field()`

    Attributes:
        _spherical: Class variable indicating grid geometry (set by subclass).
        model_names: Tuple of model names this factory supports.
        scheme_names: Tuple of scheme names this factory supports.
    """

    _spherical: AbstractClassVar[bool]
    model_names: tuple[str, ...] = eqx.field(static=True)
    scheme_names: tuple[str, ...] = eqx.field(static=True)

    @property
    def spherical(self) -> bool:
        """Whether this factory uses spherical (positive-only) grids."""
        return self._spherical

    @abstractmethod
    def make_vector_field(
        self,
        grid: VelocityGrid,
        model_name: str,
        scheme_name: str,
        dt: float,
        nu: float,
        sc_iterations: int,
    ) -> eqx.Module:
        """
        Create a vector field for the given model/scheme combo.

        Args:
            grid: Velocity grid
            model_name: Name of model (e.g., "LenardBernstein", "FastVFP")
            scheme_name: Name of scheme (e.g., "ChangCooper")
            dt: Time step
            nu: Collision frequency
            sc_iterations: Self-consistency iterations

        Returns:
            Vector field module compatible with diffrax
        """
        ...
