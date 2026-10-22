"""Independent FARSIGHT-inspired 1D1V solver using ADEPT's explicit program API.

Numerical modules are loaded lazily so importing this package does not configure JAX.
"""

from importlib import import_module
from typing import Any

_LAZY_ATTRIBUTES = {
    "Farsight1DBuilder": (".builder", "Farsight1DBuilder"),
    "Farsight1DConfig": (".config", "Farsight1DConfig"),
    "AMRConfig": (".config", "AMRConfig"),
    "TreecodeConfig": (".config", "TreecodeConfig"),
    "TreecodeField": (".treecode", "TreecodeField"),
    "FarsightSystem": (".numerics", "FarsightSystem"),
    "AdaptiveFarsightSystem": (".amr", "AdaptiveFarsightSystem"),
    "PanelHierarchy": (".amr", "PanelHierarchy"),
}

__all__ = list(_LAZY_ATTRIBUTES)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_ATTRIBUTES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
