r"""Canned plots for WarpX runs.

The heavy lifting is deliberately **not** here: :mod:`adept.warpx.io`
converts a run's openPMD output into the same ``binary/<diag>.nc`` contract
(keys, dims, code units, attrs) the OSIRIS wrapper writes, so the OSIRIS
canned-plot driver (:func:`adept.osiris.plots.save_canned_plots`) renders
WarpX runs unchanged — spacetime, log-spacetime, lineouts, ω–k dispersion
per field, current overlays, field-energy and energy-conservation traces.

This module adds the WarpX-specific extras on top: one trace figure per
reduced diagnostic (every column of the ``REDUCED/<name>.nc`` tables vs
time), which have no OSIRIS analog.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from adept.osiris import plots as _osiris_plots

# Re-export the manifest-knob translator: warpx manifests use the same
# ``output:`` block (v_th, omega_k_zoom, overlay_density, bam, ...).
canned_plot_kwargs = _osiris_plots.canned_plot_kwargs


def find_binary_dir(path: str | Path) -> Path:
    """Resolve the NetCDF tree from a run/artifact dir or the dir itself."""
    path = Path(path)
    if (path / "binary").is_dir():
        return path / "binary"
    return path


def plot_reduced_trace(ds: xr.Dataset, *, log: bool = True, title: str | None = None) -> plt.Figure:
    """Every column of one reduced-diagnostic table vs time, on one axes.

    ``ds`` is a ``REDUCED/<name>`` dataset from
    :func:`adept.warpx.io.parse_reduced_diag` (SI values on a ``step``
    coordinate with a ``time`` variable). The y-axis is log by default —
    reduced diags are overwhelmingly energies/fluxes spanning decades — and
    quantities are drawn as their |value| when negative values would
    otherwise vanish from the log view.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = ds["time"].values if "time" in ds else ds["step"].values
    for name, var in ds.data_vars.items():
        if name == "time":
            continue
        v = np.asarray(var.values, dtype="float64")
        label = name
        if log and (v < 0).any():
            v = np.abs(v)
            label = f"|{name}|"
        unit = var.attrs.get("units", "")
        ax.plot(t, v, label=f"{label} [{unit}]" if unit else label, alpha=0.8)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("t [s]" if "time" in ds else "step")
    ax.set_title(title or f"reduced diagnostic: {ds.attrs.get('diag_name', '')}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return fig


def save_canned_plots(run_dir: str | Path, out_dir: str | Path, **kwargs) -> dict[str, Path]:
    """Generate the standard PNG set for a converted WarpX run.

    ``run_dir`` is the ``binary/`` NetCDF tree from
    :func:`adept.warpx.io.save_run_datasets` (or a directory containing
    one). Delegates the shared plot families to the OSIRIS driver — see
    :func:`adept.osiris.plots.save_canned_plots` for the file list and
    keyword knobs (``v_th``, ``omega_p``, ``omega_k_zoom``, ``show_bam``,
    ``dpi``, ``n_panels``) — then adds ``reduced/<name>.png`` per reduced
    diagnostic. Returns the plot-name → path map; every family is
    best-effort.
    """
    binary = find_binary_dir(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = _osiris_plots.save_canned_plots(binary, out_dir, **kwargs)

    dpi = int(kwargs.get("dpi", 120))
    reduced_dir = binary / "REDUCED"
    if reduced_dir.is_dir():
        for p in sorted(reduced_dir.glob("*.nc")):
            try:
                ds = xr.load_dataset(p, engine="h5netcdf")
                fig = plot_reduced_trace(ds)
                dest = out_dir / "reduced" / f"{p.stem}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(dest, bbox_inches="tight", dpi=dpi)
                plt.close(fig)
                written[f"reduced/{p.stem}"] = dest
            except Exception as e:
                print(f"[plots] skipping reduced trace {p.stem}: {e}")
    return written
