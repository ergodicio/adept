"""Public registry for logging-free ADEPT solver builders."""

from __future__ import annotations

import re
from collections.abc import Callable
from threading import RLock
from typing import Any

from .contracts import PreparedSimulation, SimulationSpec, SolverBuilder, SolverCapabilities

_STABLE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


class InvalidSolverNameError(ValueError):
    """Raised when a registry key is not a stable kebab-case solver name."""


class SolverAlreadyRegisteredError(ValueError):
    """Raised when registration would silently replace an existing builder."""


class UnknownSolverError(LookupError):
    """Raised when no builder is registered for a requested solver name."""


class SolverRegistry:
    """Map stable solver names to builder instances or lazy builder factories."""

    def __init__(self) -> None:
        self._builders: dict[str, SolverBuilder[Any]] = {}
        self._lazy_factories: dict[str, Callable[[], SolverBuilder[Any]]] = {}
        self._capabilities: dict[str, SolverCapabilities] = {}
        self._lock = RLock()

    @staticmethod
    def _validate_name(name: str) -> str:
        name = name.strip()
        if not _STABLE_NAME.fullmatch(name):
            raise InvalidSolverNameError(
                f"Invalid solver name {name!r}; expected lowercase kebab-case such as 'vlasov-1d'"
            )
        return name

    def _ensure_available(self, name: str) -> None:
        if name in self._builders or name in self._lazy_factories:
            raise SolverAlreadyRegisteredError(
                f"Solver {name!r} is already registered; unregister it explicitly before replacing its builder"
            )

    def register(
        self,
        name: str,
        builder: SolverBuilder[Any],
        *,
        capabilities: SolverCapabilities | None = None,
    ) -> SolverBuilder[Any]:
        """Register an already-constructed builder without importing other solvers."""

        name = self._validate_name(name)
        if not isinstance(builder, SolverBuilder):
            raise TypeError(f"Builder for {name!r} must implement prepare(spec, *, key)")
        if capabilities is not None and not isinstance(capabilities, SolverCapabilities):
            raise TypeError("capabilities must be SolverCapabilities or None")
        with self._lock:
            self._ensure_available(name)
            self._builders[name] = builder
            if capabilities is not None:
                self._capabilities[name] = capabilities
        return builder

    def register_lazy(
        self,
        name: str,
        factory: Callable[[], SolverBuilder[Any]],
        *,
        capabilities: SolverCapabilities | None = None,
    ) -> None:
        """Register a factory that imports and constructs its builder on first use."""

        name = self._validate_name(name)
        if not callable(factory):
            raise TypeError(f"Lazy builder factory for {name!r} must be callable")
        if capabilities is not None and not isinstance(capabilities, SolverCapabilities):
            raise TypeError("capabilities must be SolverCapabilities or None")
        with self._lock:
            self._ensure_available(name)
            self._lazy_factories[name] = factory
            if capabilities is not None:
                self._capabilities[name] = capabilities

    def unregister(self, name: str) -> None:
        """Remove a registration, primarily for isolated application/plugin setup."""

        with self._lock:
            if name not in self._builders and name not in self._lazy_factories:
                raise self._unknown_solver(name)
            self._builders.pop(name, None)
            self._lazy_factories.pop(name, None)
            self._capabilities.pop(name, None)

    def names(self) -> tuple[str, ...]:
        """Return all registered names in stable order without loading builders."""

        with self._lock:
            return tuple(sorted(self._builders.keys() | self._lazy_factories.keys()))

    def _unknown_solver(self, name: str) -> UnknownSolverError:
        available = ", ".join(self.names()) or "<none>"
        return UnknownSolverError(f"Unknown solver {name!r}. Registered solvers: {available}")

    def capabilities(self, name: str) -> SolverCapabilities | None:
        """Return declared capabilities without loading a lazy builder."""

        name = self._validate_name(name)
        with self._lock:
            if name not in self._builders and name not in self._lazy_factories:
                raise self._unknown_solver(name)
            return self._capabilities.get(name)

    def resolve(self, name: str) -> SolverBuilder[Any]:
        """Resolve a builder, loading and caching a lazy registration once."""

        name = self._validate_name(name)
        with self._lock:
            if name in self._builders:
                return self._builders[name]
            try:
                factory = self._lazy_factories[name]
            except KeyError as exc:
                raise self._unknown_solver(name) from exc

            builder = factory()
            if not isinstance(builder, SolverBuilder):
                raise TypeError(f"Lazy factory for {name!r} returned an object without prepare(spec, *, key)")
            self._builders[name] = builder
            del self._lazy_factories[name]
            return builder

    def prepare(self, spec: SimulationSpec, *, key: Any) -> PreparedSimulation[Any, Any, Any, Any, Any]:
        """Dispatch deterministic preparation through the builder selected by a spec."""

        return self.resolve(spec.solver).prepare(spec, key=key)


solver_registry = SolverRegistry()


__all__ = [
    "InvalidSolverNameError",
    "SolverAlreadyRegisteredError",
    "SolverRegistry",
    "UnknownSolverError",
    "solver_registry",
]
