"""adept OSIRIS solver wrapper."""

from __future__ import annotations

__all__ = ["BaseOsiris"]


def __getattr__(name):
    if name == "BaseOsiris":
        from adept.osiris.base import BaseOsiris

        return BaseOsiris
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
