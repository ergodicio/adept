"""adept WarpX solver wrapper."""

from __future__ import annotations

__all__ = ["BaseWarpX"]


def __getattr__(name):
    if name == "BaseWarpX":
        from adept.warpx.base import BaseWarpX

        return BaseWarpX
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
