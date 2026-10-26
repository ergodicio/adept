"""Adaptive box sizing from a density gradient scale length.

OSIRIS decks fix the simulation box (``space.xmax``) and the density ramp
(``profile.fx`` / ``profile.x``) by hand. This module lets a manifest instead
request a target *gradient scale length* ``L`` and have the box scaled so the
linear density ramp realizes that ``L`` at a chosen reference density — mirroring
ADEPT's envelope (``adept/_lpse2d/helpers.py``) and Vlasov (``kinetic_srs``)
solvers, which size their grid the same way.

Geometry. The deck's ``profile`` is a piecewise-linear density ramp
``n(x): nmin -> nmax`` across its interior control points. For a linear ramp the
local scale length is

    L(x) = n(x) / (dn/dx) = n(x) * ramp_span / (nmax - nmin)

so requiring ``L(n_ref) = L_target`` at the reference density ``n_ref`` (default
the quarter-critical surface, ``n_c/4 = 0.25`` in ``n_c`` units) fixes the ramp
span:

    ramp_span = L_target / n_ref * (nmax - nmin)

which is the same relation ADEPT uses (``Lgrid = L / 0.25 * (nmax - nmin)``).

The transform is then a single spatial **scale factor**

    s = ramp_span_target / ramp_span_deck

applied uniformly to every length in the deck: ``space.xmin`` / ``space.xmax``,
every ``profile.x`` array, and every ``diag_species`` phase-space window
(``ps_xmin`` / ``ps_xmax``). ``grid.nx_p`` is scaled by ``s`` as well so the cell
size ``dx`` is held fixed (rounded up to a clean multiple of the per-dimension
node count for even domain decomposition). Time (``dt``, ``tmax``) is untouched,
so the CFL ratio is preserved.

Activation. Only when the manifest carries an ``osiris.density`` block with a
``gradient_scale_length`` (alias ``gradient scale length``). Without it, decks
run with their hand-set box, unchanged.

1D only: raises ``NotImplementedError`` for multi-dimensional decks (the SRS LPI
use case is 1D).
"""

from __future__ import annotations

import math
from typing import Any

from adept.normalization import UREG, skin_depth_normalization, skin_depth_normalization_from_frequency
from adept.osiris.deck import Sections


def apply_gradient_scale_length(sections: Sections, density_cfg: dict | None) -> dict | None:
    """Scale the OSIRIS box so the density ramp has gradient scale length ``L``.

    ``density_cfg`` is the manifest's ``osiris.density`` block. Returns a dict of
    the computed quantities (for logging) or ``None`` when the feature is
    inactive (no ``density_cfg``, or no gradient scale length given).

    Mutates ``sections`` in place. Recognized ``density_cfg`` keys:

    * ``gradient_scale_length`` (alias ``gradient scale length``) — target ``L``,
      a unit string (``"300um"``) or a number already in ``c/wp0`` units.
    * ``min`` / ``max`` — optional ``nmin`` / ``nmax`` in ``n_c`` units; default is
      to read the ramp's interior endpoints from the deck's ``profile.fx``. When
      given, they are also written into the primary ``profile.fx`` endpoints.
    * ``reference_density`` — density (``n_c`` units) at which ``L`` is defined;
      default ``0.25`` (quarter-critical).
    """
    if not density_cfg:
        return None
    L_spec = density_cfg.get("gradient_scale_length", density_cfg.get("gradient scale length"))
    if L_spec is None:
        return None

    reference_density = float(density_cfg.get("reference_density", 0.25))

    simulation = _first_section(sections, "simulation")
    space = _first_section(sections, "space")
    grid = _first_section(sections, "grid")
    profile = _first_section(sections, "profile")
    missing = [name for name, sec in (("space", space), ("grid", grid), ("profile", profile)) if sec is None]
    if missing:
        raise ValueError(
            f"osiris.density (adaptive box sizing) requires the deck to define section(s): {', '.join(missing)}"
        )

    # --- 1D only ----------------------------------------------------------
    nx_key = _array_key(grid, "nx_p")
    if nx_key is None:
        raise ValueError("osiris.density requires grid.nx_p in the deck")
    nx_val = grid[nx_key]
    if isinstance(nx_val, list) and len(nx_val) > 1:
        raise NotImplementedError(
            "osiris.density adaptive box sizing is implemented for 1D decks only "
            f"(grid.{nx_key} has {len(nx_val)} dimensions)"
        )
    nx_old = int(nx_val[0] if isinstance(nx_val, list) else nx_val)

    # --- ramp geometry & densities from the primary profile ---------------
    x_key = _array_key(profile, "x")
    fx_key = _array_key(profile, "fx")
    if x_key is None or fx_key is None:
        raise ValueError("osiris.density requires profile.x and profile.fx in the deck")
    x_old = [float(v) for v in _as_list(profile[x_key])]
    fx = [float(v) for v in _as_list(profile[fx_key])]
    if len(x_old) < 3 or len(fx) < 3:
        raise ValueError(
            f"profile needs >= 3 control points to define a density ramp "
            f"(got {len(x_old)} x-points, {len(fx)} fx-values)"
        )

    given_minmax = "min" in density_cfg or "max" in density_cfg
    if given_minmax:
        nmin = float(density_cfg["min"])
        nmax = float(density_cfg["max"])
    else:
        nmin, nmax = fx[1], fx[-2]  # interior ramp endpoints
    if nmax <= nmin:
        raise ValueError(f"density max ({nmax}) must exceed min ({nmin})")

    ramp_span_old = x_old[-2] - x_old[1]
    if ramp_span_old <= 0:
        raise ValueError(f"profile.{x_key} interior control points must be increasing (got ramp span {ramp_span_old})")

    # --- target ramp span and the resulting scale factor ------------------
    L_norm = _normalized_length(L_spec, simulation)
    ramp_span_new = L_norm * (nmax - nmin) / reference_density
    s = ramp_span_new / ramp_span_old

    xmax_old = _scalar(space, "xmax")
    if xmax_old is None:
        raise ValueError("osiris.density requires space.xmax in the deck")
    xmin_old = _scalar(space, "xmin") or 0.0
    box_new = (xmax_old - xmin_old) * s

    # constant dx -> nx scales with the box; round UP (don't under-resolve)
    # to a clean multiple of the per-dim node count so OSIRIS splits evenly.
    node_count = _node_count(sections)
    nx_new = _round_up_multiple(math.ceil(nx_old * s), node_count)

    # --- apply the rescale in place ---------------------------------------
    if given_minmax:
        fx[1], fx[-2] = nmin, nmax
        profile[fx_key] = fx if isinstance(profile[fx_key], list) else fx[0]

    _scale_array(space, "xmin", s)
    _scale_array(space, "xmax", s)
    for sec_name, params in sections:
        if sec_name == "profile":
            _scale_array(params, "x", s)
        elif sec_name == "diag_species":
            _scale_array(params, "ps_xmin", s)
            _scale_array(params, "ps_xmax", s)
    grid[nx_key] = [nx_new] if isinstance(nx_val, list) else nx_new

    return {
        "gradient_scale_length_requested": str(L_spec),
        "gradient_scale_length_norm": L_norm,
        "reference_density": reference_density,
        "density_min": nmin,
        "density_max": nmax,
        "ramp_span_norm": ramp_span_new,
        "box_norm": box_new,
        "nx": nx_new,
        "scale_factor": s,
    }


# --- helpers --------------------------------------------------------------


def _first_section(sections: Sections, name: str) -> dict | None:
    for sec_name, params in sections:
        if sec_name == name:
            return params
    return None


def _array_key(params: dict, base: str) -> str | None:
    """Key in ``params`` whose base name (before any ``(``) equals ``base``."""
    for k in params:
        if k == base or k.split("(", 1)[0] == base:
            return k
    return None


def _as_list(v: Any) -> list:
    return list(v) if isinstance(v, list) else [v]


def _scalar(params: dict, base: str) -> float | None:
    key = _array_key(params, base)
    if key is None:
        return None
    v = params[key]
    return float(v[0]) if isinstance(v, list) else float(v)


def _scale_array(params: dict, base: str, factor: float) -> bool:
    """Multiply the value(s) at ``base`` by ``factor`` in place; preserve shape."""
    key = _array_key(params, base)
    if key is None:
        return False
    v = params[key]
    if isinstance(v, list):
        params[key] = [round(float(x) * factor, 10) for x in v]
    else:
        params[key] = round(float(v) * factor, 10)
    return True


def _node_count(sections: Sections) -> int:
    node_conf = _first_section(sections, "node_conf")
    if node_conf is None:
        return 1
    key = _array_key(node_conf, "node_number")
    if key is None:
        return 1
    val = node_conf[key]
    first = val[0] if isinstance(val, list) else val
    try:
        return max(1, int(first))
    except (TypeError, ValueError):
        return 1


def _round_up_multiple(n: int, m: int) -> int:
    if m <= 1:
        return int(n)
    return int(math.ceil(n / m) * m)


def _normalized_length(L_spec: Any, simulation: dict | None) -> float:
    """Target ``L`` in skin-depth (``c/wp0``) units.

    A bare number is taken to be already normalized; a unit string is converted
    using the deck's reference density (``simulation.n0`` / ``omega_p0``).
    """
    if isinstance(L_spec, (int, float)) and not isinstance(L_spec, bool):
        return float(L_spec)
    norm = _skin_depth_norm(simulation)
    if norm is None:
        raise ValueError(
            "osiris.density.gradient_scale_length was given with physical units "
            f"({L_spec!r}) but the deck's simulation section has neither n0 nor "
            "omega_p0 to set the c/wp0 length scale"
        )
    return float((UREG.Quantity(str(L_spec)) / norm.L0).to("").magnitude)


def _skin_depth_norm(simulation: dict | None):
    if not simulation:
        return None
    n0 = simulation.get("n0")
    omega_p0 = simulation.get("omega_p0")
    if n0 is not None:
        return skin_depth_normalization(f"{n0} / cc")
    if omega_p0 is not None:
        return skin_depth_normalization_from_frequency(f"{omega_p0} rad/s")
    return None
