"""Honest, common-grid FARSIGHT/Eulerian phase-space comparison artifacts.

``render_comparison`` accepts an Eulerian ``f(t, x, v)`` dataset, the native
fixed-panel FARSIGHT distribution dataset, and two native scalar datasets
containing ``mass(t)`` and ``c2(t)``. Pass ``config={"farsight": resolved_config,
"eulerian": {"field_model": "unsoftened periodic Poisson"}}``. The Eulerian
field-model label is mandatory: changing the force model must not be hidden by
otherwise matched plots. Eulerian coordinates must be uniform cell centers.

Saved, remeshed fixed grids and rectilinear AMR leaves are supported. The
renderer evaluates the actual piecewise biquadratic interpolant, without scatter
interpolation, clipping, renormalization, temporal interpolation, or velocity extrapolation.
Native quadrature invariants and common-grid distribution differences are
reported separately. All output paths are returned for explicit MLflow upload.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _finite(values, name):
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _axis(values, name, *, uniform=False):
    result = _finite(values, name)
    if result.ndim != 1 or result.size < 1 or np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be a nonempty, strictly increasing one-dimensional coordinate")
    if uniform and (result.size < 2 or not np.allclose(np.diff(result), np.diff(result)[0], rtol=1e-10)):
        raise ValueError(f"{name} must contain at least two uniformly spaced cell centers")
    return result


def _lagrange_basis(u):
    """Three nodal polynomials at local coordinates 0, 1, 2."""
    return np.stack(((u - 1) * (u - 2) / 2, u * (2 - u), u * (u - 1) / 2), axis=-1)


def reconstruct_fixed_panels(dataset, x, v, grid):
    """Evaluate fixed, remeshed 3x3 panels at the Cartesian product ``x × v``.

    Returns an array of shape ``(t, len(x), len(v))``. The x axis is periodic,
    including x == xmax; velocity endpoints are included and values strictly
    outside [vmin, vmax] are zero. AMR and moving/deformed panels are rejected.
    """
    expected_dims = ("t", "x_node", "v_node")
    if "active" in dataset or "panel" in dataset.dims:
        raise ValueError("AMR geometry is unsupported; this renderer requires fixed remeshed tensor panels")
    for name in ("x", "v", "f"):
        if name not in dataset or dataset[name].dims != expected_dims:
            raise ValueError(f"FARSIGHT {name!r} must have dimensions {expected_dims}")
    nx, nv = grid["nx"], grid["nv"]
    if isinstance(nx, bool) or isinstance(nv, bool) or int(nx) != nx or int(nv) != nv:
        raise ValueError("FARSIGHT nx and nv must be integer interval counts")
    nx, nv = int(nx), int(nv)
    xmin, xmax, vmin, vmax = (float(grid[key]) for key in ("xmin", "xmax", "vmin", "vmax"))
    if nx < 4 or nv < 2 or nx % 2 or nv % 2 or not np.isfinite([xmin, xmax, vmin, vmax]).all():
        raise ValueError("FARSIGHT requires even nx >= 4 and nv >= 2, with finite bounds")
    if xmax <= xmin or vmax <= vmin:
        raise ValueError("FARSIGHT grid bounds must increase")
    f = _finite(dataset["f"], "FARSIGHT f")
    if f.shape[1:] != (nx + 1, nv + 1):
        raise ValueError("FARSIGHT saved node counts disagree with the configured grid")
    expected_x = np.linspace(xmin, xmax, nx + 1)[None, :, None]
    expected_v = np.linspace(vmin, vmax, nv + 1)[None, None, :]
    for name, expected in (("x", expected_x), ("v", expected_v)):
        coordinates = _finite(dataset[name], f"FARSIGHT {name}")
        tolerance = 1e-11 * max(1.0, float(np.max(np.abs(expected))))
        if not np.allclose(coordinates, expected, rtol=0, atol=tolerance):
            raise ValueError(
                f"FARSIGHT {name} coordinates are moving/deformed; save post-remesh frames or use a genuine "
                "deformed-panel reconstruction (not supported here)"
            )
    seam_tolerance = 1e-12 * max(1.0, float(np.max(np.abs(f))))
    if not np.allclose(f[:, 0], f[:, -1], rtol=1e-11, atol=seam_tolerance):
        raise ValueError("FARSIGHT duplicated periodic endpoints disagree")
    x, v = _finite(x, "target x"), _finite(v, "target v")
    if x.ndim != 1 or v.ndim != 1:
        raise ValueError("Target x and v must be one-dimensional axes")
    dx, dv = (xmax - xmin) / nx, (vmax - vmin) / nv
    x_index = np.mod(x - xmin, xmax - xmin) / dx
    v_index = (v - vmin) / dv
    # Clip before converting to integers; arbitrarily distant finite velocity
    # queries should still produce zero without integer overflow warnings.
    bounded_v_index = np.clip(v_index, 0, nv)
    ix = 2 * np.clip(np.floor(x_index / 2).astype(int), 0, nx // 2 - 1)
    iv = 2 * np.clip(np.floor(bounded_v_index / 2).astype(int), 0, nv // 2 - 1)
    bx = _lagrange_basis(x_index - ix)
    bv = _lagrange_basis(bounded_v_index - iv)
    sampled = np.zeros((f.shape[0], x.size, v.size))
    for i in range(3):
        for j in range(3):
            sampled += f[:, (ix + i)[:, None], (iv + j)[None, :]] * bx[None, :, i, None] * bv[None, None, :, j]
    sampled[:, :, (v < vmin) | (v > vmax)] = 0.0
    return sampled


def reconstruct_amr_panels(dataset, x, v, grid, *, max_level):
    """Reconstruct genuine post-remesh AMR leaves on the Cartesian product x × v.

    Saved ``panel_id`` and ``level`` must identify a complete, nonoverlapping
    reference-hierarchy leaf partition. Node order is x-major/v-minor, matching
    ``make_hierarchy``. On shared edges (including the periodic seam), the finest
    leaf owns the point; equal-level ties use the smallest reference panel ID.
    No interpolation across a missing leaf or advected-panel crack is permitted.
    Inactive slots are ignored, and velocity-exterior queries return zero.
    """
    for name in ("x", "v", "f"):
        if name not in dataset or dataset[name].dims != ("t", "panel", "node"):
            raise ValueError(f"AMR {name!r} must have dimensions (t, panel, node)")
    for name in ("active", "level", "panel_id"):
        if name not in dataset or dataset[name].dims != ("t", "panel"):
            raise ValueError(f"AMR {name!r} must have dimensions (t, panel)")
    if dataset.sizes["node"] != 9:
        raise ValueError("AMR reconstruction requires nine nodes per biquadratic panel")
    nx, nv = grid["nx"], grid["nv"]
    integers = (nx, nv, max_level)
    if any(isinstance(value, bool) or int(value) != value for value in integers):
        raise ValueError("AMR grid interval counts and max_level must be integers")
    nx, nv, max_level = (int(value) for value in integers)
    if nx < 4 or nv < 2 or nx % 2 or nv % 2 or not 0 <= max_level <= 4:
        raise ValueError("AMR requires even nx >= 4, nv >= 2 and 0 <= max_level <= 4")
    xmin, xmax, vmin, vmax = (float(grid[key]) for key in ("xmin", "xmax", "vmin", "vmax"))
    if not np.isfinite([xmin, xmax, vmin, vmax]).all() or xmax <= xmin or vmax <= vmin:
        raise ValueError("AMR bounds must be finite and increasing")
    length, span = xmax - xmin, vmax - vmin
    roots = (nx // 2) * (nv // 2)
    offsets = np.cumsum([0] + [roots * 4**level for level in range(max_level + 1)])
    if offsets[-1] > 32768:
        raise ValueError("AMR reference hierarchy exceeds the production 32768-panel bound")
    raw_active = np.asarray(dataset["active"])
    if not np.all((raw_active == 0) | (raw_active == 1)):
        raise ValueError("AMR active flags must be boolean or 0/1")
    active = raw_active.astype(bool)
    arrays = {name: np.asarray(dataset[name]) for name in ("x", "v", "f", "level", "panel_id")}
    for name, values in arrays.items():
        _finite(values[active], f"active AMR {name}")
    x, v = _finite(x, "target x"), _finite(v, "target v")
    if x.ndim != 1 or v.ndim != 1:
        raise ValueError("Target x and v must be one-dimensional axes")
    npx, npv = (nx // 2) * 2**max_level, (nv // 2) * 2**max_level
    canonical_x = xmin + np.mod(x - xmin, length)
    cell_x = (canonical_x - xmin) / length * npx
    cell_v = np.clip((v - vmin) / span * npv, 0, npv)
    right = np.clip(np.floor(cell_x).astype(int), 0, npx - 1)
    above = np.clip(np.floor(cell_v).astype(int), 0, npv - 1)
    # Include both adjacent finest cells only at an actual shared edge. The
    # lookup map then identifies the coarser/finer containing leaves in O(1).
    ulp_tolerance = 8 * np.finfo(float).eps
    edge_x = np.isclose(cell_x, np.rint(cell_x), rtol=0, atol=ulp_tolerance * np.maximum(1.0, np.abs(cell_x)))
    edge_v = np.isclose(cell_v, np.rint(cell_v), rtol=0, atol=ulp_tolerance * np.maximum(1.0, np.abs(cell_v)))
    right = np.where(edge_x, np.rint(cell_x).astype(int) % npx, right)
    above = np.where(edge_v, np.clip(np.rint(cell_v).astype(int), 0, npv - 1), above)
    left = np.where(edge_x, (np.rint(cell_x).astype(int) - 1) % npx, right)
    below = np.where(edge_v, np.clip(np.rint(cell_v).astype(int) - 1, 0, npv - 1), above)
    output = np.zeros((dataset.sizes["t"], x.size, v.size))
    node_offset = np.arange(3)
    for frame in range(dataset.sizes["t"]):
        slots = np.flatnonzero(active[frame])
        if not slots.size:
            raise ValueError("AMR frame has no active leaves")
        levels = arrays["level"][frame, slots]
        panel_ids = arrays["panel_id"][frame, slots]
        if np.any(levels != np.floor(levels)) or np.any(panel_ids != np.floor(panel_ids)):
            raise ValueError("Active AMR levels and panel IDs must be integers")
        levels, panel_ids = levels.astype(int), panel_ids.astype(int)
        if np.any((levels < 0) | (levels > max_level)):
            raise ValueError("Active AMR levels are outside the configured hierarchy")
        if np.any((panel_ids < offsets[levels]) | (panel_ids >= offsets[levels + 1])):
            raise ValueError("AMR panel IDs disagree with their saved levels")
        if np.unique(panel_ids).size != panel_ids.size:
            raise ValueError("AMR leaf partition contains duplicate panel IDs")
        level_npv = (nv // 2) * 2**levels
        relative_id = panel_ids - offsets[levels]
        ix, iv = relative_id // level_npv, relative_id % level_npv
        dx, dv = length / (nx * 2**levels), span / (nv * 2**levels)
        expected_x = xmin + (2 * ix[:, None] + node_offset) * dx[:, None]
        expected_v = vmin + (2 * iv[:, None] + node_offset) * dv[:, None]
        px = arrays["x"][frame, slots].reshape(-1, 3, 3)
        pv = arrays["v"][frame, slots].reshape(-1, 3, 3)
        for name, actual, expected in (("x", px, expected_x[:, :, None]), ("v", pv, expected_v[:, None, :])):
            tolerance = 1e-11 * max(1.0, float(np.max(np.abs(expected))))
            if not np.allclose(actual, expected, rtol=0, atol=tolerance):
                raise ValueError(
                    f"AMR {name} coordinates are moving/deformed or disagree with panel_id; "
                    "only post-remesh reference leaves are supported"
                )
        owner_map = np.full((npx, npv), -1, dtype=int)
        for leaf, (i, j, level) in enumerate(zip(ix, iv, levels, strict=True)):
            stride = 2 ** (max_level - level)
            region = owner_map[i * stride : (i + 1) * stride, j * stride : (j + 1) * stride]
            if np.any(region != -1):
                raise ValueError("AMR leaf partition has overlapping active interiors")
            region[:] = leaf
        if np.any(owner_map == -1):
            raise ValueError("AMR leaf partition has uncovered reference cells")
        candidates = np.stack(
            [
                owner_map[i[:, None], j[None, :]]
                for i, j in ((right, above), (left, above), (right, below), (left, below))
            ]
        )
        ranks = levels[candidates] * (offsets[-1] + 1) + offsets[-1] - panel_ids[candidates]
        owner = np.take_along_axis(candidates, np.argmax(ranks, axis=0)[None, :, :], axis=0)[0]
        center = px[:, 1, 1][owner]
        displacement = canonical_x[:, None] - center
        displacement -= length * np.floor(displacement / length + 0.5)
        local_x = (center + displacement - px[:, 0, 0][owner]) / dx[owner]
        local_v = (np.clip(v, vmin, vmax)[None, :] - pv[:, 0, 0][owner]) / dv[owner]
        values = arrays["f"][frame, slots].reshape(-1, 3, 3)[owner]
        output[frame] = np.einsum("abij,abi,abj->ab", values, _lagrange_basis(local_x), _lagrange_basis(local_v))
    output[:, :, (v < vmin) | (v > vmax)] = 0.0
    return output


def farsight_method(config):
    """Return the explicit discretization/field-evaluation method identifier."""
    field_solver = config.get("numerical", {}).get("field_solver", "direct")
    if field_solver not in {"direct", "treecode"}:
        raise ValueError("Unknown FARSIGHT field solver; expected direct or treecode")
    mesh = "amr" if config.get("amr", {}).get("enabled", False) else "fixed"
    return f"farsight-{mesh}-{field_solver}"


def iter_validated_amr_frames(dataset, grid, amr):
    """Yield active x/v/f arrays (N, 9) and levels (N,) after geometry validation.

    Empty target axes exercise the complete reference-partition validation
    without constructing a display raster. This is also useful for independent
    exact polynomial quadrature, whose invariant is not the nodal sum.
    """
    reconstruct_amr_panels(dataset, np.empty(0), np.empty(0), grid, max_level=amr.get("max_level", 1))
    for frame in range(dataset.sizes["t"]):
        active = np.asarray(dataset["active"])[frame].astype(bool)
        yield tuple(np.asarray(dataset[name])[frame, active] for name in ("x", "v", "f", "level"))


def _scalar_data(dataset, name, times):
    scalar_times = _axis(dataset["t"], f"{name} scalar t")
    tolerance = 1e-10 * max(1.0, abs(times[-1]))
    if abs(scalar_times[0] - times[0]) > tolerance or scalar_times[-1] < times[-1] - tolerance:
        raise ValueError(f"{name} scalar times must begin with and cover the saved distribution times")
    result = {"t": scalar_times}
    for quantity in ("mass", "c2"):
        if quantity not in dataset or dataset[quantity].dims != ("t",):
            raise ValueError(f"{name} scalar dataset needs {quantity}(t)")
        value = _finite(dataset[quantity], f"{name} {quantity}")
        if value[0] <= 0:
            raise ValueError(f"{name} initial {quantity} must be positive")
        result[quantity] = value
        result[f"relative_{quantity}"] = (value - value[0]) / value[0]
    return result


def _has_h264_encoder(binary):
    """Check codec support rather than assuming every ffmpeg build has x264."""
    try:
        result = subprocess.run(
            [binary, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("V") and fields[1] == "libx264":
            return True
    return False


def _ffmpeg_path():
    """Select an existing H.264-capable executable; never install implicitly."""
    binary = shutil.which("ffmpeg")
    if binary and _has_h264_encoder(binary):
        return binary
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        bundled = None
    if bundled and bundled != binary and _has_h264_encoder(bundled):
        return bundled
    raise RuntimeError(
        "MP4 rendering requires ffmpeg with the libx264 encoder; neither system ffmpeg nor an existing "
        "imageio_ffmpeg installation supplies a usable H.264 encoder"
    )


def render_comparison(
    eulerian_ds,
    farsight_ds,
    eulerian_scalars,
    farsight_scalars,
    config: Mapping,
    output: Path,
    *,
    title="FARSIGHT / Eulerian phase space",
    fps=12,
    make_movie=True,
):
    """Write ``comparison.mp4``, two PNGs, and JSON diagnostics; return paths.

    Pass ``make_movie=False`` for static-only rendering, e.g. in unit tests.
    Each named output must not already exist, so rerendering cannot silently
    overwrite provenance-bearing artifacts. No MLflow or other network IO is
    performed. ``config['eulerian']['field_model']`` is a required human-readable
    description; optional ``label`` overrides the spectral-x/cubic-spline-v label.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    from .representation import representation_integrals

    if isinstance(fps, bool) or int(fps) != fps or fps < 1:
        raise ValueError("fps must be a positive integer")
    try:
        farsight_config = config["farsight"]
        grid = farsight_config["grid"]
        field_model = config["eulerian"]["field_model"]
    except KeyError as exc:
        raise ValueError("config requires farsight resolved config and eulerian.field_model") from exc
    if not isinstance(field_model, str) or not field_model.strip():
        raise ValueError("eulerian.field_model must explicitly describe the force model")
    method = farsight_method(farsight_config)
    amr = farsight_config.get("amr", {})
    is_amr = amr.get("enabled", False)
    if "f" not in eulerian_ds or eulerian_ds["f"].dims != ("t", "x", "v"):
        raise ValueError("Eulerian dataset requires f(t, x, v)")
    times = _axis(eulerian_ds["t"], "Eulerian t")
    farsight_times = _axis(farsight_ds["t"], "FARSIGHT t")
    if times.shape != farsight_times.shape or not np.allclose(times, farsight_times, rtol=1e-10, atol=1e-10):
        raise ValueError("Saved distribution times must match; temporal interpolation is not performed")
    x, v = (_axis(eulerian_ds[name], f"Eulerian {name}", uniform=True) for name in ("x", "v"))
    dx, dv = x[1] - x[0], v[1] - v[0]
    bounds = (x[0] - dx / 2, x[-1] + dx / 2, v[0] - dv / 2, v[-1] + dv / 2)
    expected_bounds = tuple(grid[name] for name in ("xmin", "xmax", "vmin", "vmax"))
    if not np.allclose(bounds, expected_bounds, rtol=1e-10, atol=1e-10):
        raise ValueError("Eulerian cell-center domain must match the configured FARSIGHT domain")
    eulerian = _finite(eulerian_ds["f"], "Eulerian f")
    if is_amr:
        farsight = reconstruct_amr_panels(farsight_ds, x, v, grid, max_level=amr.get("max_level", 1))
        active_counts = np.asarray(farsight_ds["active"]).astype(bool).sum(axis=1)
        capacity = farsight_ds.sizes["panel"]
    else:
        farsight = reconstruct_fixed_panels(farsight_ds, x, v, grid)
        active_counts = None
        capacity = None
    delta = farsight - eulerian
    scalar_data = (
        _scalar_data(eulerian_scalars, "Eulerian", times),
        _scalar_data(farsight_scalars, "FARSIGHT", times),
    )
    representation = representation_integrals(farsight_ds, farsight_config)
    plot_data = (
        *scalar_data,
        {name: np.asarray(representation[name]) for name in ("t", "relative_mass", "relative_c2")},
    )
    encoder = _ffmpeg_path() if make_movie else None
    output = Path(output)
    paths = {
        "contact_sheet": output / "phase_space_contact_sheet.png",
        "conservation": output / "conservation.png",
        "diagnostics": output / "diagnostics.json",
    }
    if make_movie:
        paths["movie"] = output / "comparison.mp4"
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Comparison artifacts already exist: {', '.join(existing)}")
    output.mkdir(parents=True, exist_ok=True)
    epsilon = float(farsight_config["numerical"]["epsilon"])
    quadrature = farsight_config["numerical"].get("quadrature", "trapezoid")
    eulerian_label = config["eulerian"].get("label", "Eulerian: spectral-x / cubic-spline-v")
    field_solver = farsight_config.get("numerical", {}).get("field_solver", "direct")
    mesh_label = "AMR" if is_amr else "fixed panels"
    method_label = f"FARSIGHT: {mesh_label} / {field_solver}"

    def farsight_label(frame):
        resolution = f"{grid['nx']} × {grid['nv']} {'base ' if is_amr else ''}intervals; softened ε = {epsilon:g}"
        if is_amr:
            resolution += f"\nactive leaves: {active_counts[frame]} / {capacity} slots"
        return f"{method_label}\n{resolution}"

    labels = (
        f"{eulerian_label}\n{x.size} × {v.size} cells; {field_model}",
        farsight_label(0),
        "FARSIGHT − Eulerian\non Eulerian cell centers",
    )
    footer = (
        f"Solid: native quadrature ({quadrature} for FARSIGHT). Dashed: exact saved-panel polynomial integral. "
        "Each relative to its own initial value; no clipping/renormalization."
    )
    f_min = min(0.0, float(eulerian.min()), float(farsight.min()))
    f_max = max(float(eulerian.max()), float(farsight.max()))
    if f_max <= f_min:
        f_max = f_min + 1.0
    delta_limit = max(float(np.max(np.abs(delta))), np.finfo(float).eps)
    colors = ("#2765a0", "#d36f24", "#8054a0")

    def phase_image(ax, values, column, *, heading=True):
        limits = (
            {"vmin": -delta_limit, "vmax": delta_limit, "cmap": "RdBu_r"}
            if column == 2
            else {"vmin": f_min, "vmax": f_max, "cmap": "viridis"}
        )
        artist = ax.imshow(values.T, origin="lower", extent=bounds, aspect="auto", interpolation="nearest", **limits)
        if heading:
            ax.set_title(labels[column], fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        return artist

    def conservation_axes(axes):
        lines, cursors = [], []
        for ax, quantity, math_label in zip(axes, ("mass", "c2"), ("M", "C_2"), strict=True):
            quantity_lines = []
            for data, color, label, style in zip(
                plot_data,
                colors,
                ("Eulerian native", "FARSIGHT native", "FARSIGHT polynomial"),
                ("-", "-", "--"),
                strict=True,
            ):
                ax.plot(data["t"], data[f"relative_{quantity}"], color=color, alpha=0.18, linewidth=1, linestyle=style)
                (line,) = ax.plot([], [], color=color, label=label, linewidth=1.6, linestyle=style)
                quantity_lines.append(line)
            ax.axhline(0, color="0.5", linewidth=0.6)
            cursors.append(ax.axvline(times[0], color="0.3", linewidth=0.7, linestyle="--"))
            ax.set_xlabel("t [ωₚ⁻¹]")
            ax.set_ylabel(rf"$({math_label}(t)-{math_label}(0))/{math_label}(0)$")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)
            lines.append(quantity_lines)
        return lines, cursors

    # A separate conservation figure preserves all retained scalar observations,
    # not merely the coarser cadence used for distribution frames.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    try:
        conservation_lines, _ = conservation_axes(axes)
        for quantity, lines in zip(("mass", "c2"), conservation_lines, strict=True):
            for data, line in zip(plot_data, lines, strict=True):
                line.set_data(data["t"], data[f"relative_{quantity}"])
        fig.suptitle(title + " — conservation diagnostics")
        fig.supxlabel(footer, fontsize=8)
        fig.savefig(paths["conservation"], dpi=160)
    finally:
        plt.close(fig)

    indices = [0, len(times) // 2, len(times) - 1]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    try:
        for row, frame in enumerate(indices):
            for column, values in enumerate((eulerian, farsight, delta)):
                artist = phase_image(axes[row, column], values[frame], column, heading=(row == 0))
                if column == 1 and is_amr:
                    axes[row, column].set_title(
                        farsight_label(frame) if row == 0 else f"active leaves: {active_counts[frame]}", fontsize=10
                    )
                axes[row, column].set_ylabel(f"t = {times[frame]:g}\nv")
                fig.colorbar(artist, ax=axes[row, column], label="Δf" if column == 2 else "f", shrink=0.85)
        fig.suptitle(title + " — initial / midpoint / final", fontsize=14)
        fig.supxlabel(
            "Fixed color limits across every movie frame. Δf is a distribution difference, not an error estimator.",
            fontsize=9,
        )
        fig.savefig(paths["contact_sheet"], dpi=160)
    finally:
        plt.close(fig)

    if make_movie:
        fig = plt.figure(figsize=(14, 8), layout="constrained")
        try:
            spec = fig.add_gridspec(2, 1, height_ratios=(1.25, 0.8))
            phase_spec = spec[0].subgridspec(1, 3)
            scalar_spec = spec[1].subgridspec(1, 2)
            phase_axes = [fig.add_subplot(phase_spec[0, index]) for index in range(3)]
            images = []
            for column, (ax, values) in enumerate(zip(phase_axes, (eulerian, farsight, delta), strict=True)):
                artist = phase_image(ax, values[0], column)
                images.append(artist)
                fig.colorbar(artist, ax=ax, label="Δf" if column == 2 else "f", shrink=0.85)
            axes = [fig.add_subplot(scalar_spec[0, index]) for index in range(2)]
            conservation_lines, cursors = conservation_axes(axes)
            heading = fig.suptitle(title, fontsize=14)
            fig.supxlabel(footer, fontsize=8)
            with matplotlib.rc_context({"animation.ffmpeg_path": encoder}):
                writer = animation.FFMpegWriter(
                    fps=int(fps),
                    codec="libx264",
                    bitrate=2500,
                    extra_args=[
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        "-threads",
                        "2",
                        "-filter_threads",
                        "2",
                    ],
                    metadata={
                        "title": title,
                        "comment": "Native FARSIGHT biquadratic reconstruction; fixed frame color limits",
                    },
                )
                with writer.saving(fig, str(paths["movie"]), dpi=100):
                    for frame, time in enumerate(times):
                        if is_amr:
                            phase_axes[1].set_title(farsight_label(frame), fontsize=10)
                        for artist, values in zip(images, (eulerian, farsight, delta), strict=True):
                            artist.set_data(values[frame].T)
                        for quantity, lines, cursor in zip(("mass", "c2"), conservation_lines, cursors, strict=True):
                            cursor.set_xdata([time, time])
                            for data, line in zip(plot_data, lines, strict=True):
                                selected = data["t"] <= time + 1e-10
                                line.set_data(data["t"][selected], data[f"relative_{quantity}"][selected])
                        heading.set_text(f"{title}    t = {time:g} ωₚ⁻¹")
                        writer.grab_frame()
        finally:
            plt.close(fig)

    def native_summary(data):
        return {
            "t": data["t"].tolist(),
            **{name: data[name].tolist() for name in ("mass", "c2", "relative_mass", "relative_c2")},
        }

    denominator = np.sum(eulerian**2, axis=(1, 2))
    relative_l2 = np.sqrt(np.sum(delta**2, axis=(1, 2)) / np.maximum(denominator, np.finfo(float).tiny))
    diagnostics = {
        "title": title,
        "config": config,
        "reconstruction": (
            "piecewise 3x3 biquadratic on post-remesh AMR reference leaves; finest boundary ownership"
            if is_amr
            else "piecewise 3x3 biquadratic on fixed post-remesh FARSIGHT panels"
        ),
        "farsight_method": method,
        "farsight_representation": representation,
        "farsight_amr": {
            "enabled": bool(is_amr),
            "active_panels": active_counts.tolist() if is_amr else None,
            "capacity": capacity,
            "maximum_level": amr.get("max_level", 1) if is_amr else None,
        },
        "comparison_note": (
            "Distribution differences are not accuracy estimates when force models or resolutions differ."
        ),
        "conservation_note": "Native quadrature drifts; the two native quadratures need not be identical.",
        "eulerian_field_model": field_model,
        "farsight_field_model": {"kind": "softened periodic kernel", "epsilon": epsilon},
        "eulerian_cells": [int(x.size), int(v.size)],
        "farsight_intervals": [int(grid["nx"]), int(grid["nv"])],
        "times": times.tolist(),
        "fixed_color_limits": {"f": [f_min, f_max], "delta_f": [-delta_limit, delta_limit]},
        "relative_l2_distribution_difference": relative_l2.tolist(),
        "max_abs_distribution_difference": np.max(np.abs(delta), axis=(1, 2)).tolist(),
        "min_f": {
            "eulerian": np.min(eulerian, axis=(1, 2)).tolist(),
            "farsight_reconstructed": np.min(farsight, axis=(1, 2)).tolist(),
        },
        "common_grid_integrals": {
            "note": "Cell-center display quadrature, not native solver conservation diagnostics",
            **{
                name: {
                    "mass": (dx * dv * values.sum(axis=(1, 2))).tolist(),
                    "c2": (dx * dv * (values**2).sum(axis=(1, 2))).tolist(),
                }
                for name, values in (("eulerian", eulerian), ("farsight_reconstructed", farsight))
            },
        },
        "native_scalars": {"eulerian": native_summary(scalar_data[0]), "farsight": native_summary(scalar_data[1])},
    }
    paths["diagnostics"].write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    return paths
