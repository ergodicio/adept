r"""Canned matplotlib views over OSIRIS diagnostics.

Each plotter accepts either an already-loaded ``xarray.DataArray`` or a
filesystem path / directory, and returns the ``Axes`` it drew on so the
caller can tweak labels / colorbars / save the figure.

``save_canned_plots(run_dir, out_dir)`` ties them together and writes
PNGs for the standard set:

    out_dir/
      spacetime/<diag>.png                 (t, x) heatmap of every FLD diagnostic
      spacetime_log/<diag>.png             log10|·| of the same
      lineouts/<diag>.png                  value-vs-x snapshots at sampled times
      omega_k/<diag>.png                   2-D FFT (k, ω) dispersion: full range +
                                           equal-aspect square window (ω = k at 45°)
      currents/spacetime.png               j1/j2/j3 (J_x/J_y/J_z) side-by-side spacetime
      currents/lineouts.png                j1/j2/j3 profiles vs x (final + late mean)
      moments/<species>/<q>.png            (t, x) per-species density moments
      moments/<species>/<q>_log.png        log10 of the same
      moments/<species>/lineouts/<q>.png   moment snapshots at sampled times
      profiles/<species>/density.png       number-density profile vs x (initial + final + late mean)
      profiles/<species>/temperature.png   T(x) from a Maxwellian fit to the phase space
                                           (initial + final + late mean); else uth / T_ii moments
      phasespace/<species>/<ps>.png        final-step (x, p) heatmap per species
      phasespace_evolution/<species>/<ps>.png  (x, p) heatmaps at sampled times
      field_decomp/<comp>.png              left/right-going transverse E (Riemann split)
      energy_vs_time.png                   total field energy time-trace
      energy_components_vs_time.png        E-field / B-field / total energy
      total_energy_vs_time.png             field + kinetic conservation (needs HIST/)

All axis / colorbar / title labels are emitted as proper LaTeX (``$\omega$``,
``$c/\omega_p$``, …) via the ``_tex`` helper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Canned plots are written to disk in batch / headless runs (e.g. Perlmutter
# compute nodes), so force a non-interactive backend before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from adept.osiris import io as _io

# --- low-level plotters ----------------------------------------------------


def plot_spacetime(
    series: xr.DataArray | str | Path,
    ax: plt.Axes | None = None,
    *,
    log: bool = False,
    cmap: str = "RdBu_r",
    space_on_x: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """``(t, x)`` heatmap of a 1D-field time-series.

    Accepts a pre-loaded ``DataArray`` (dims must include ``t`` and one
    spatial axis) or a directory of dumps that ``load_series`` will eat.

    By default space is on the horizontal axis and time on the vertical
    (``t`` vs ``x``); set ``space_on_x=False`` to transpose so time is on the
    horizontal axis and space on the vertical (``x`` vs ``t``).
    """
    da = _ensure_series(series)
    if da.ndim != 2:
        raise ValueError(
            f"plot_spacetime expects a 2D (t, x) array; got dims {da.dims}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    data = np.log10(np.abs(da.values) + 1e-30) if log else da.values  # (t, x)
    t = da.coords["t"].values
    xname = next(d for d in da.dims if d != "t")
    x = da.coords[xname].values
    if space_on_x:
        mesh = ax.pcolormesh(x, t, data, shading="auto", cmap=cmap)
        ax.set_xlabel(_axis_label(da, xname))
        ax.set_ylabel(_axis_label(da, "t"))
        orient = r"$t$ vs $x$"
    else:
        mesh = ax.pcolormesh(t, x, data.T, shading="auto", cmap=cmap)
        ax.set_xlabel(_axis_label(da, "t"))
        ax.set_ylabel(_axis_label(da, xname))
        orient = r"$x$ vs $t$"
    plt.colorbar(
        mesh,
        ax=ax,
        label=rf"$\log_{{10}}$ |{_value_label(da)}|" if log else _value_label(da),
    )
    scale = r"$\log_{10}$ " if log else ""
    ax.set_title(title or f"{_display_name(da)}  —  {scale}spacetime ({orient})")
    return ax


def plot_lineouts(
    series: xr.DataArray | str | Path,
    *,
    n_panels: int = 8,
    col_wrap: int = 4,
    title: str | None = None,
) -> plt.Figure:
    """Faceted value-vs-x snapshots of a ``(t, x)`` series at sampled times.

    Subsamples the time axis to roughly ``n_panels`` snapshots (matching the
    ``t_skip = nt // 8`` convention used by the other adept solvers) and lays
    them out in a ``col_wrap`` grid. Returns the ``Figure`` so the caller can
    save / close it.
    """
    da = _decorate(_ensure_series(series))
    if da.ndim != 2:
        raise ValueError(
            f"plot_lineouts expects a 2D (t, x) array; got dims {da.dims}"
        )
    nt = da.coords["t"].size
    t_skip = max(1, nt // n_panels)
    sl = da.isel(t=slice(0, None, t_skip))
    xname = next(d for d in da.dims if d != "t")
    g = sl.plot(x=xname, col="t", col_wrap=min(col_wrap, sl.coords["t"].size))
    g.set_xlabels(_axis_label(da, xname))
    g.set_ylabels(_value_label(da))
    g.fig.suptitle(
        title or rf"{_display_name(da)}  —  lineouts vs $x$ at sampled times", y=1.02
    )
    return g.fig


def plot_phasespace(
    da: xr.DataArray | str | Path,
    ax: plt.Axes | None = None,
    *,
    log: bool = True,
    cmap: str = "viridis",
    title: str | None = None,
) -> plt.Axes:
    """Heatmap of a 2D phase-space density (e.g. ``x1p1``).

    For an x-p phase space the spatial axis is cropped to the physical box
    (``sim.XMAX``; phase-space dumps may pad it out past the box) and drawn on
    the horizontal axis with momentum on the vertical, matching the orientation
    of :func:`plot_phasespace_evolution`.
    """
    if not isinstance(da, xr.DataArray):
        da = _io.load_phasespace_h5(da)
    if da.ndim != 2:
        raise ValueError(
            f"plot_phasespace expects 2D data; got {da.dims} {da.shape}"
        )
    da = _crop_spatial_to_box(da)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    # Space on the horizontal axis, momentum on the vertical; fall back to dim
    # order for momentum-momentum spaces (e.g. p1p2) with no spatial axis.
    spatial = [d for d in da.dims if str(d).startswith("x")]
    moment = [d for d in da.dims if str(d).startswith("p")]
    xdim, ydim = (spatial[0], moment[0]) if spatial and moment else da.dims
    raw = da.transpose(ydim, xdim).values
    vmin = vmax = None
    if log:
        vmin, vmax = _nonzero_log_clim(raw)
        plot_arr = np.log10(np.abs(raw) + 1e-30)
    else:
        plot_arr = raw
    xc, yc = da.coords[xdim].values, da.coords[ydim].values
    mesh = ax.pcolormesh(xc, yc, plot_arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(
        mesh, ax=ax, label=rf"$\log_{{10}}$ {_value_label(da)}" if log else _value_label(da)
    )
    ax.set_xlabel(_axis_label(da, xdim))
    ax.set_ylabel(_axis_label(da, ydim))
    t = da.attrs.get("time", float("nan"))
    scale = r"$\log_{10}$ " if log else ""
    ax.set_title(
        title
        or rf"{_display_name(da)}  —  {scale}phase space  ($t = {t:.3g}\ 1/\omega_p$)"
    )
    return ax


def plot_phasespace_evolution(
    series: xr.DataArray | str | Path,
    *,
    n_panels: int = 8,
    col_wrap: int = 4,
    log: bool = True,
    cmap: str = "viridis",
    title: str | None = None,
) -> plt.Figure:
    """Faceted ``(x, p)`` phase-space heatmaps at sampled times.

    Subsamples a stacked ``(t, p, x)`` phase-space series (as returned by
    ``io.load_series``) to ~``n_panels`` snapshots so the time evolution is
    visible, rather than only the final step. Returns the ``Figure``.
    """
    da = series if isinstance(series, xr.DataArray) else _io.load_series(series)
    da = _decorate(da)
    if da.ndim != 3:
        raise ValueError(
            f"plot_phasespace_evolution expects (t, p, x); got dims {da.dims}"
        )
    da = _crop_spatial_to_box(da)
    nt = da.coords["t"].size
    t_skip = max(1, nt // n_panels)
    sl = da.isel(t=slice(0, None, t_skip))
    plot_da = np.log10(np.abs(sl) + 1e-30) if log else sl
    # Convention: spatial axis horizontal, momentum axis vertical.
    spatial = [d for d in da.dims if d != "t" and str(d).startswith("x")]
    moment = [d for d in da.dims if d != "t" and str(d).startswith("p")]
    facet_kw: dict = {"cmap": cmap}
    if spatial and moment:
        facet_kw.update(x=spatial[0], y=moment[0])
    if log:
        # Floor the shared colour scale at the lowest non-zero value so empty
        # cells (log -> -30) don't crush the contrast across the facets.
        vmin, vmax = _nonzero_log_clim(sl.values)
        facet_kw.update(vmin=vmin, vmax=vmax)
    g = plot_da.plot(col="t", col_wrap=min(col_wrap, sl.coords["t"].size), **facet_kw)
    if spatial and moment:
        g.set_xlabels(_axis_label(da, spatial[0]))
        g.set_ylabels(_axis_label(da, moment[0]))
    scale = r"$\log_{10}$ " if log else ""
    g.fig.suptitle(
        title or rf"{_display_name(da)}  —  {scale}phase space at sampled times", y=1.02
    )
    return g.fig


def field_energy_series(run_dir: str | Path) -> xr.DataArray:
    """Sum ``(|E|^2 + |B|^2) / 2`` over space at every saved step.

    Returns a 1D ``DataArray`` of total field energy in code units vs time.
    ``run_dir`` may be a raw OSIRIS run directory **or** the ``binary/``
    directory of saved NetCDFs (it is resolved through :func:`io.load_series`),
    so the trace is reproducible from the saved artifacts alone.
    """
    ds = field_energy_components(run_dir)
    da = ds["total_field_energy"].rename("field_energy")
    da.attrs.update(long_name="total field energy", units="code")
    return da


def plot_energy_vs_time(
    src: xr.DataArray | str | Path,
    ax: plt.Axes | None = None,
    *,
    log: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """Plot total field energy vs time.

    ``src`` may be a precomputed ``DataArray`` from ``field_energy_series``
    or a ``run_dir`` path (in which case the series is computed on the fly).
    """
    if isinstance(src, xr.DataArray):
        da = src
    else:
        da = field_energy_series(src)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(da.coords["t"].values, da.values)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"t  [$1/\omega_p$]")
    ax.set_ylabel(r"$\int (|E|^2 + |B|^2)/2\ d^Dx$  [code units]")
    ax.set_title(title or "Total field energy vs time")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def field_energy_components(run_dir: str | Path) -> xr.Dataset:
    """E-field, B-field, and total field energy vs time from the FLD diagnostics.

    Like :func:`field_energy_series` but keeps the electric (``e1/e2/e3``) and
    magnetic (``b1/b2/b3``) contributions separate. Returns a ``Dataset`` with
    data variables ``E_energy``, ``B_energy`` and ``total_field_energy`` on a
    shared ``t`` axis. Components not dumped are simply omitted from their sum.

    Sources are resolved via :func:`io.list_diagnostics` / :func:`io.load_series`,
    so ``run_dir`` may be a raw OSIRIS run directory or the ``binary/`` directory
    of saved NetCDFs — the energy is recomputed from the stored ``(t, x)`` field
    series either way.
    """
    diags = _io.list_diagnostics(run_dir)
    groups = {"E_energy": ("e1", "e2", "e3"), "B_energy": ("b1", "b2", "b3")}
    by_iter: dict[int, dict[str, float]] = {}
    times: dict[int, float] = {}
    found = False
    for group, comps in groups.items():
        for comp in comps:
            rel = f"FLD/{comp}"
            if rel not in diags:
                continue
            try:
                ser = _io.load_series(diags[rel])
            except Exception as e:  # a bad component must not sink the rest
                print(f"[plots] skipping field-energy component {comp}: {e}")
                continue
            if ser.ndim != 2:
                continue
            found = True
            e_t = _field_energy_from_series(ser)
            its = (
                np.asarray(ser.coords["iter"].values)
                if "iter" in ser.coords
                else np.arange(ser.sizes["t"])
            )
            ts = np.asarray(ser.coords["t"].values, dtype="float64")
            for k in range(ts.size):
                it = int(its[k])
                rec = by_iter.setdefault(it, {"E_energy": 0.0, "B_energy": 0.0})
                rec[group] += float(e_t[k])
                times.setdefault(it, float(ts[k]))
    if not found or not by_iter:
        raise RuntimeError(f"No field dumps available under {run_dir}")
    iters = sorted(by_iter)
    t = np.array([times[i] for i in iters])
    e_arr = np.array([by_iter[i]["E_energy"] for i in iters])
    b_arr = np.array([by_iter[i]["B_energy"] for i in iters])
    coords = {"t": t, "iter": ("t", np.asarray(iters))}
    ds = xr.Dataset(
        {
            "E_energy": ("t", e_arr),
            "B_energy": ("t", b_arr),
            "total_field_energy": ("t", e_arr + b_arr),
        },
        coords=coords,
    )
    ds["E_energy"].attrs.update(long_name="electric field energy", units="code")
    ds["B_energy"].attrs.update(long_name="magnetic field energy", units="code")
    ds["total_field_energy"].attrs.update(long_name="total field energy", units="code")
    ds["t"].attrs.update(long_name="time", units=r"1/\omega_p")
    return ds


def plot_energy_components(
    src: xr.Dataset | str | Path,
    ax: plt.Axes | None = None,
    *,
    log: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """Overlay E-field, B-field, and total field energy vs time."""
    ds = src if isinstance(src, xr.Dataset) else field_energy_components(src)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    t = ds["t"].values
    labels = {
        "E_energy": r"$\int |E|^2/2$ (electric)",
        "B_energy": r"$\int |B|^2/2$ (magnetic)",
        "total_field_energy": "total field",
    }
    for name, label in labels.items():
        if name in ds:
            ax.plot(t, ds[name].values, label=label)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"t  [$1/\omega_p$]")
    ax.set_ylabel("field energy  [code units]")
    ax.set_title(title or "Field energy components vs time")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_energy_conservation(
    energy: xr.Dataset,
    ax: plt.Axes | None = None,
    *,
    log: bool = False,
    title: str | None = None,
) -> plt.Axes:
    """Plot field, kinetic, and total energy vs time as a conservation check.

    ``energy`` is the ``Dataset`` from :func:`io.load_hist_energy`; it must
    contain a ``total`` variable (field + kinetic). The total-energy drift,
    ``(max - min) / max(|total|)``, is annotated in the title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    t = energy["t"].values
    series_labels = {
        "field_energy": "field",
        "kinetic_total": "kinetic (all species)",
        "total": "total (field + kinetic)",
    }
    for name, label in series_labels.items():
        if name in energy:
            ax.plot(t, energy[name].values, label=label)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"t  [$1/\omega_p$]")
    ax.set_ylabel("energy  [code units]")
    drift = energy.attrs.get("total_drift_frac")
    base = title or "Energy conservation vs time"
    ax.set_title(base if drift is None else f"{base}  (total drift {drift:.2%})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_omega_k(
    series: xr.DataArray | str | Path,
    ax: plt.Axes | None = None,
    *,
    log: bool = True,
    cmap: str = "magma",
    show_em: bool = True,
    show_langmuir: bool = False,
    show_light_line: bool = False,
    v_th: float | None = None,
    omega_p: float = 1.0,
    k_max: float | None = None,
    omega_max: float | None = None,
    equal_aspect: bool = False,
    title: str | None = None,
) -> plt.Axes:
    """2-D FFT of a ``(t, x)`` field series; show power in ``(k, ω)`` space.

    The OSIRIS convention has ω_p = 1, c = 1 in code units, so the
    relativistic-EM dispersion ω² = ω_p² + k² and the Langmuir
    Bohm–Gross dispersion ω² = ω_p² + 3 k² v_th² overlay directly when
    you ask for them. ``show_light_line`` overlays the vacuum light line
    ω = ±k (the ω = k diagonal that the EM branch asymptotes to), which is
    the useful guide when ``k_max`` / ``omega_max`` are set to zoom into the
    low-(k, ω) region where the plasma (Langmuir) waves live.
    """
    da = _ensure_series(series)
    if da.ndim != 2:
        raise ValueError(
            f"plot_omega_k expects (t, x); got dims {da.dims}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    t = da.coords["t"].values
    xname = next(d for d in da.dims if d != "t")
    x = da.coords[xname].values
    dt = float(t[1] - t[0])
    dx = float(x[1] - x[0])
    nt = t.size
    nx = x.size

    # 2-D FFT then shift so DC sits in the middle.
    F = np.fft.fftshift(np.fft.fft2(da.values))
    P = np.abs(F) ** 2
    if log:
        P = np.log10(P + 1e-30)

    omega = np.fft.fftshift(np.fft.fftfreq(nt, d=dt)) * 2 * np.pi
    k = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi

    mesh = ax.pcolormesh(k, omega, P, shading="auto", cmap=cmap)
    plt.colorbar(
        mesh, ax=ax, label=(r"$\log_{10}\,|\tilde{F}|^2$" if log else r"$|\tilde{F}|^2$")
    )

    if k_max is None:
        k_max = float(np.max(np.abs(k)))
    if omega_max is None:
        omega_max = float(np.max(np.abs(omega)))
    ax.set_xlim(-k_max, k_max)
    ax.set_ylim(-omega_max, omega_max)

    # Overlay analytical dispersion lines (sampled across the visible k range).
    k_line = np.linspace(-k_max, k_max, 401)
    if show_light_line:
        ax.plot(k_line, +k_line, "w-", lw=0.8, alpha=0.6,
                label=r"light line: $\omega = \pm k$")
        ax.plot(k_line, -k_line, "w-", lw=0.8, alpha=0.6)
    if show_em:
        w_em = np.sqrt(omega_p ** 2 + k_line ** 2)
        ax.plot(k_line, +w_em, "w--", lw=1, alpha=0.7,
                label=r"EM: $\omega^2 = \omega_p^2 + k^2$")
        ax.plot(k_line, -w_em, "w--", lw=1, alpha=0.7)
    if show_langmuir:
        if v_th is None:
            raise ValueError("show_langmuir=True requires v_th=...")
        w_l = np.sqrt(omega_p ** 2 + 3 * (k_line * v_th) ** 2)
        ax.plot(k_line, +w_l, "c:", lw=1, alpha=0.8,
                label=r"Langmuir: $\omega^2 = \omega_p^2 + 3 k^2 v_{th}^2$")
        ax.plot(k_line, -w_l, "c:", lw=1, alpha=0.8)
    if show_em or show_langmuir or show_light_line:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)

    ax.axhline(0, color="w", lw=0.4, alpha=0.4)
    ax.axvline(0, color="w", lw=0.4, alpha=0.4)
    if equal_aspect:
        # One unit of k displays as one unit of ω, so the light line ω = k is a
        # true 45° slope (lets you read off where power sits relative to it).
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$k\ [\omega_p / c]$")
    ax.set_ylabel(r"$\omega\ [\omega_p]$")
    ax.set_title(title or rf"{_display_name(da)}  —  $(k, \omega)$ power spectrum")
    return ax


def plot_omega_k_figure(
    series: xr.DataArray | str | Path,
    *,
    v_th: float | None = None,
    omega_k_zoom: float | None = 4.0,
    cmap: str = "magma",
    log: bool = True,
) -> plt.Figure:
    """Stacked ``(k, ω)`` spectra: full range on top, equal-aspect below.

    The top panel is the full 2-D FFT over the data's Nyquist range. The bottom
    panel shows the same power with k and ω on the same scale (equal aspect, so
    the light line ``ω = k`` is a true 45° slope), limited to a square
    ``±z`` window with ``z = min(omega_k_zoom, Nyquist)`` so the slope-1 line —
    and any low-frequency power along it — is visible.

    With a coarse dump cadence the ω-Nyquist is small, so that window is only a
    few k-cells wide and the lower panel looks blocky; that is the expected
    consequence of the time-sampling, not a plotting fault.
    """
    da = _ensure_series(series)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 10))
    plot_omega_k(
        da, ax=ax_top, log=log, cmap=cmap,
        show_langmuir=v_th is not None, v_th=v_th,
    )
    z = _omega_k_zoom_window(da, omega_k_zoom)
    plot_omega_k(
        da, ax=ax_bot, log=log, cmap=cmap,
        show_light_line=True, show_langmuir=v_th is not None, v_th=v_th,
        k_max=z, omega_max=z, equal_aspect=True,
        title=rf"{_display_name(da)}  —  $(k, \omega)$ (equal aspect)",
    )
    fig.tight_layout()
    return fig


def _sim_box_bound(da: xr.DataArray, xdim: str, *, upper: bool) -> float:
    """Edge of the *physical* simulation box along ``xdim`` (``sim.XMIN/XMAX``).

    Phase-space diagnostics may be binned over a spatial range wider than the
    field grid (deck ``ps_xmax`` > ``xmax``), so the coordinate axis runs past
    the box. ``sim.XMIN`` / ``sim.XMAX`` (carried on every diagnostic) record
    the true box edges — per spatial dimension, so ``x1`` -> entry 0,
    ``x2`` -> entry 1, …. Falls back to the matching axis end when the attr is
    absent, which makes callers degrade to the full axis exactly when it
    already coincides with the box.
    """
    xv = da.coords[xdim].values
    bound = da.attrs.get("sim.XMAX" if upper else "sim.XMIN")
    if bound is None:
        return float(xv[-1] if upper else xv[0])
    if isinstance(bound, (list, tuple, np.ndarray)):
        digits = "".join(ch for ch in str(xdim) if ch.isdigit())
        i = int(digits) - 1 if digits else 0
        bound = bound[i] if 0 <= i < len(bound) else bound[-1]
    return float(bound)


def _sim_box_xmax(da: xr.DataArray, xdim: str) -> float:
    """Right edge of the physical box along ``xdim`` (see :func:`_sim_box_bound`)."""
    return _sim_box_bound(da, xdim, upper=True)


def _nonzero_log_clim(values) -> tuple[float | None, float | None]:
    """``(vmin, vmax)`` for a ``log10`` heatmap, floored at the lowest non-zero.

    Phase-space dumps are mostly empty (zero) cells; mapping those through
    ``log10`` sends them to a huge negative floor that dominates the colour
    range and washes out the real distribution. Restricting ``vmin`` to the
    lowest *non-zero* magnitude (and ``vmax`` to the largest) keeps the dynamic
    range on the populated cells. Returns ``(None, None)`` for all-zero input.
    """
    mag = np.abs(np.asarray(values))
    nz = mag > 0
    if not nz.any():
        return None, None
    return float(np.log10(mag[nz].min())), float(np.log10(mag.max()))


def _crop_spatial_to_box(da: xr.DataArray) -> xr.DataArray:
    """Trim spatial (``x*``) dims to the physical box ``[sim.XMIN, sim.XMAX]``.

    Phase-space dumps can be binned over a spatial range wider than the box
    (deck ``ps_xmax`` > ``xmax``), padding the high-``x`` end of the axis with
    empty cells. This keeps only the cells within the simulation domain so a
    phase-space plot's spatial axis spans just the box. Returns ``da`` unchanged
    when no spatial dim extends past the box (e.g. ``p1p2`` momentum spaces).
    """
    for d in list(da.dims):
        if not str(d).startswith("x"):
            continue
        xv = da.coords[d].values
        if xv.size < 2:
            continue
        left = int(np.searchsorted(xv, _sim_box_bound(da, d, upper=False), side="left"))
        right = int(np.searchsorted(xv, _sim_box_bound(da, d, upper=True), side="right"))
        right = max(left + 1, min(right, xv.size))
        if left > 0 or right < xv.size:
            da = da.isel({d: slice(left, right)})
    return da


# --- currents (j1/j2/j3) --------------------------------------------------


def _current_components(run_dir: str | Path) -> dict[str, xr.DataArray]:
    """Load whichever of ``FLD/j1``, ``FLD/j2``, ``FLD/j3`` were dumped."""
    diags = _io.list_diagnostics(run_dir)
    out: dict[str, xr.DataArray] = {}
    for comp in ("j1", "j2", "j3"):
        rel = f"FLD/{comp}"
        if rel in diags:
            try:
                ser = _io.load_series(diags[rel])
            except Exception as e:  # skip a bad component
                print(f"[plots] skipping current {comp}: {e}")
                continue
            if ser.ndim == 2:
                out[comp] = ser
    return out


def plot_currents_spacetime(run_dir: str | Path) -> plt.Figure | None:
    """Side-by-side ``(t, x)`` heatmaps of the current components present.

    OSIRIS dumps current density under ``MS/FLD/j1`` … ``j3`` (the labels map
    to ``J_x``, ``J_y``, ``J_z`` for a run aligned with ``x1``). This stacks
    whatever is present into one figure for an at-a-glance comparison;
    returns ``None`` if no current was dumped.
    """
    comps = _current_components(run_dir)
    if not comps:
        return None
    fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 4), squeeze=False)
    for ax, (comp, ser) in zip(axes[0], comps.items(), strict=False):
        plot_spacetime(ser, ax=ax)
    fig.suptitle(r"Current density components  —  spacetime ($t$ vs $x$)", y=1.03)
    fig.tight_layout()
    return fig


def plot_currents_lineouts(run_dir: str | Path, *, n_avg_frac: float = 0.2) -> plt.Figure | None:
    """Overlay the current components vs ``x`` (final snapshot + late-time mean).

    All available ``j1/j2/j3`` are drawn on a single axis so their relative
    magnitudes and spatial structure are directly comparable. Solid = final
    dump, dashed = mean over the last ``n_avg_frac`` of the run.
    """
    comps = _current_components(run_dir)
    if not comps:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for comp, ser in comps.items():
        da = _decorate(ser)
        xdim = next(d for d in da.dims if d != "t")
        x = da.coords[xdim].values
        nt = da.sizes["t"]
        w = max(1, round(n_avg_frac * nt))
        (line,) = ax.plot(x, da.isel(t=-1).values, lw=1.4, label=_tex(_long_name(da)))
        ax.plot(
            x, da.isel(t=slice(nt - w, nt)).mean("t").values,
            lw=1.0, ls="--", alpha=0.7, color=line.get_color(),
        )
    any_ser = _decorate(next(iter(comps.values())))
    xdim = next(d for d in any_ser.dims if d != "t")
    ax.set_xlabel(_axis_label(any_ser, xdim))
    ax.set_ylabel(_value_label(any_ser))
    ax.set_title(r"Current density  —  profile vs $x$ (solid: final, dashed: late mean)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# --- per-species density / temperature profiles ---------------------------


def _species_diags(run_dir: str | Path) -> dict[str, list[tuple[str, str, Path]]]:
    """Group per-species moment diagnostics as ``species -> [(kind, quantity, path)]``.

    Walks ``MS/DENSITY/<species>/<quantity>`` and ``MS/UDIST/<species>/<quantity>``
    (the OSIRIS homes for charge / mass / energy density and for fluid /
    thermal velocity moments respectively).
    """
    diags = _io.list_diagnostics(run_dir)
    out: dict[str, list[tuple[str, str, Path]]] = {}
    for rel, path in diags.items():
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] in ("DENSITY", "UDIST"):
            out.setdefault(parts[1], []).append((parts[0], "/".join(parts[2:]), path))
    return out


def _density_series(entries: list[tuple[str, str, Path]]) -> xr.DataArray | None:
    """Best density-like ``(t, x)`` moment for a species, as a number density.

    Prefers an explicit number-density report; otherwise falls back to the
    charge density and divides out the species charge sign (OSIRIS reports
    charge density ``q*n``, negative for electrons) so the result is a
    non-negative number density ``n``. Returns ``None`` if no spatial density
    moment is present.
    """
    by_q = {q: p for kind, q, p in entries if kind == "DENSITY"}
    for pref in ("n", "n01", "n02", "charge", "m"):
        if pref not in by_q:
            continue
        ser = _io.load_series(by_q[pref])
        if ser.ndim != 2:
            continue
        if pref == "charge":
            q_sign = np.sign(float(np.nansum(ser.values))) or 1.0
            attrs, name = dict(ser.attrs), ser.name
            ser = ser / q_sign
            ser.attrs, ser.name = attrs, name
            ser.attrs["long_name"] = "n"  # now a number density, not charge
            # Charge-density units lead with the charge ``e``; drop it so the
            # label reads as a number density (the magnitudes match in code
            # units where e = 1).
            units = str(ser.attrs.get("units", ""))
            if units.startswith("e "):
                ser.attrs["units"] = units[2:].lstrip()
        return ser
    # Fall back to any DENSITY entry as-is.
    for kind, _q, p in entries:
        if kind == "DENSITY":
            ser = _io.load_series(p)
            if ser.ndim == 2:
                return ser
    return None


def _temperature_series(entries: list[tuple[str, str, Path]]) -> xr.DataArray | None:
    r"""Build a temperature-like ``(t, x)`` profile from thermal moments.

    Two recognised sources, in priority order:

    - thermal-velocity moments ``uth1/uth2/uth3`` (OSIRIS ``UDIST``): summed in
      quadrature, ``T(x) = \sum_i u_{th,i}^2`` (units of ``m c^2``, c = 1);
    - a temperature/pressure tensor diagonal ``T11/T22/T33``: summed directly.

    Returns ``None`` when neither is present (e.g. a cold run that dumps no
    thermal moment), so callers can skip the temperature profile silently.
    """
    by_q = {q: p for _, q, p in entries}
    uth = [by_q[f"uth{i}"] for i in (1, 2, 3) if f"uth{i}" in by_q]
    tens = [by_q[f"T{i}{i}"] for i in (1, 2, 3) if f"T{i}{i}" in by_q]
    if uth:
        comps = [_io.load_series(p) for p in uth]
        comps = [c for c in comps if c.ndim == 2]
        if not comps:
            return None
        total = sum((c ** 2 for c in comps[1:]), comps[0] ** 2)
        long_name = r"T = \sum_i u_{th,i}^2"
        units = r"m_e c^2"
    elif tens:
        comps = [_io.load_series(p) for p in tens]
        comps = [c for c in comps if c.ndim == 2]
        if not comps:
            return None
        total = sum(comps[1:], comps[0])
        long_name = r"T = \mathrm{tr}\,T_{ii}"
        units = comps[0].attrs.get("units", "")
    else:
        return None
    out = total.copy()
    out.attrs = dict(comps[0].attrs)
    out.attrs["long_name"] = long_name
    out.attrs["units"] = units
    out.name = "temperature"
    return out


def _species_phasespace(diags: dict[str, Path], species: str) -> Path | None:
    """Path to an x-p phase space for ``species`` (e.g. ``PHA/p1x1/<species>``)."""
    for rel, path in sorted(diags.items()):
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] == "PHA" and parts[-1] == species:
            ps = parts[1]  # e.g. "p1x1" — needs both a space and a momentum axis
            if "x" in ps and "p" in ps:
                return path
    return None


def _temperature_from_phasespace(series: xr.DataArray | str | Path | None) -> xr.DataArray | None:
    r"""Temperature profile ``T(t, x)`` from an x-p phase space.

    Fits a moment-matched Maxwellian in ``p`` at every ``(t, x)`` and returns
    its temperature ``T = <(p - <p>)^2>`` (the variance of the momentum;
    ``m_e c^2`` units with ``m = 1``) as a ``(t, x)`` DataArray, cropped to the
    physical box. The charge sign is divided out so the weights are
    non-negative. ``None`` if the input is missing or is not an x-p phase space.
    """
    if series is None:
        return None
    ser = _decorate(series if isinstance(series, xr.DataArray) else _io.load_series(series))
    spatial = [d for d in ser.dims if d != "t" and str(d).startswith("x")]
    moment = [d for d in ser.dims if d != "t" and str(d).startswith("p")]
    if not spatial or not moment or "t" not in ser.dims:
        return None
    ser = _crop_spatial_to_box(ser)
    pdim = moment[0]
    pc = ser.coords[pdim]
    q_sign = np.sign(float(np.nansum(ser.values))) or 1.0
    w = (ser / q_sign).clip(min=0.0)  # non-negative weights f(t, p, x)
    weight = w.sum(pdim)
    p0 = (w * pc).sum(pdim) / weight
    var = (w * (pc - p0) ** 2).sum(pdim) / weight  # (t, x) = T
    out = var.where(weight > 0)
    out.attrs = {k: v for k, v in ser.attrs.items() if k != "units"}
    out.attrs["long_name"] = "T"
    out.attrs["units"] = r"m_e c^2"
    out.name = "temperature"
    return out


def plot_profile(
    series: xr.DataArray | str | Path,
    ax: plt.Axes | None = None,
    *,
    abs_value: bool = False,
    n_avg_frac: float = 0.2,
    show_initial: bool = False,
    value_label: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot a ``(t, x)`` moment vs ``x``: final snapshot plus late-time mean.

    The late-time mean (over the last ``n_avg_frac`` of the dumps) smooths
    out the time-dependent fluctuations to show the established profile. With
    ``show_initial`` the ``t = 0`` profile is overlaid too, so the change from
    the initial state is visible.
    """
    da = _decorate(series if isinstance(series, xr.DataArray) else _io.load_series(series))
    if "t" not in da.dims:
        raise ValueError(f"plot_profile expects a (t, x) series; got dims {da.dims}")
    xdim = next(d for d in da.dims if d != "t")
    x = da.coords[xdim].values
    tvals = da.coords["t"].values
    nt = da.sizes["t"]
    w = max(1, round(n_avg_frac * nt))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    final = da.isel(t=-1).values
    mean = da.isel(t=slice(nt - w, nt)).mean("t").values
    if abs_value:
        final, mean = np.abs(final), np.abs(mean)
    if show_initial:
        initial = da.isel(t=0).values
        if abs_value:
            initial = np.abs(initial)
        # Dotted and on top (high zorder) so the initial profile stays visible
        # where the final / mean curves overlap it.
        ax.plot(x, initial, lw=1.6, ls=":", color="k", zorder=3,
                label=f"initial ($t={float(tvals[0]):.3g}$)")
    ax.plot(x, final, lw=1.4, zorder=2, label=f"final ($t={float(tvals[-1]):.3g}$)")
    ax.plot(x, mean, lw=1.1, ls="--", zorder=2.2, label=f"mean of last {w} dumps")
    ax.set_xlabel(_axis_label(da, xdim))
    ax.set_ylabel(value_label or _value_label(da))
    ax.set_title(title or rf"{_display_name(da)}  —  profile vs $x$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# --- left/right-going electric-field decomposition ------------------------


def efield_lr_components(run_dir: str | Path) -> dict[str, dict[str, xr.DataArray]]:
    r"""Split the transverse electric field into left/right-going parts.

    For a 1D run (propagation along ``x1``), vacuum Maxwell makes the
    transverse field pairs into Riemann invariants that advect at ``c`` in a
    single direction:

    - ``(E_y, B_z)`` -> right-going ``E_y^R = (e2 + b3)/2``,
      left-going ``E_y^L = (e2 - b3)/2``;
    - ``(E_z, B_y)`` -> right-going ``E_z^R = (e3 - b2)/2``,
      left-going ``E_z^L = (e3 + b2)/2``.

    (Code units: ``c = 1``, so ``|E| = |B|`` for a pure travelling wave and
    one of the two parts vanishes.) Returns ``{"e2": {"right":…, "left":…},
    "e3": {…}}`` for whichever transverse pairs were both dumped.

    Caveat: the split is *exact only in vacuum / a uniform non-dispersive
    medium*. In a plasma the EM wave is dispersive (``v_φ = ω/k > c``), so
    ``|E| ≠ |B|`` mode-by-mode and the decomposition is approximate — still a
    useful directional diagnostic, but do not read the residual as physical
    counter-propagating power without checking the dispersion. The
    longitudinal ``e1`` is electrostatic and is intentionally left out.
    """
    diags = _io.list_diagnostics(run_dir)

    def _load(comp: str) -> xr.DataArray | None:
        rel = f"FLD/{comp}"
        if rel not in diags:
            return None
        ser = _io.load_series(diags[rel])
        return ser if ser.ndim == 2 else None

    def _pack(values: xr.DataArray, src: xr.DataArray, name: str, long: str) -> xr.DataArray:
        out = values.copy()
        out.attrs = dict(src.attrs)
        out.attrs["long_name"] = long
        out.name = name
        return out

    out: dict[str, dict[str, xr.DataArray]] = {}
    e2, b3 = _load("e2"), _load("b3")
    if e2 is not None and b3 is not None:
        out["e2"] = {
            "right": _pack((e2 + b3) / 2.0, e2, "e2_R", r"E_y^{\rightarrow}"),
            "left": _pack((e2 - b3) / 2.0, e2, "e2_L", r"E_y^{\leftarrow}"),
        }
    e3, b2 = _load("e3"), _load("b2")
    if e3 is not None and b2 is not None:
        out["e3"] = {
            "right": _pack((e3 - b2) / 2.0, e3, "e3_R", r"E_z^{\rightarrow}"),
            "left": _pack((e3 + b2) / 2.0, e3, "e3_L", r"E_z^{\leftarrow}"),
        }
    return out


def plot_field_lr_decomposition(run_dir: str | Path) -> dict[str, plt.Figure]:
    """One figure per transverse component: right/left-going spacetime.

    Two panels side by side — the right-going and left-going parts of the
    transverse E field as spacetime heatmaps with space on the horizontal axis
    and time on the vertical (``t`` vs ``x``).
    """
    figs: dict[str, plt.Figure] = {}
    for comp, parts in efield_lr_components(run_dir).items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, side in zip(axes, ("right", "left"), strict=False):
            da = parts[side]
            plot_spacetime(
                da, ax=ax, space_on_x=True,
                title=f"{_display_name(da)}  —  spacetime",
            )
        fig.suptitle(
            rf"{comp}: left/right-going decomposition (vacuum Riemann split)", y=1.02
        )
        fig.tight_layout()
        figs[comp] = fig
    return figs


# --- driver ---------------------------------------------------------------


def canned_plot_kwargs(output_cfg: dict | None) -> dict:
    """Translate a manifest ``output:`` block into :func:`save_canned_plots` kwargs.

    Shared by the live post-processing path (``post.collect``) and the offline
    regeneration harness (``regen``) so both honour the same knobs:
    ``v_th`` and ``omega_k_zoom`` (which may be explicitly ``null`` to disable
    the zoom). Keys that are absent fall through to the ``save_canned_plots``
    defaults.
    """
    output_cfg = output_cfg or {}
    kwargs: dict = {"v_th": output_cfg.get("v_th")}
    if "omega_k_zoom" in output_cfg:  # may be explicitly null to disable zoom
        kwargs["omega_k_zoom"] = output_cfg["omega_k_zoom"]
    return kwargs


def _omega_k_zoom_window(series: xr.DataArray, requested: float | None) -> float | None:
    """Clamp a requested ``(k, ω)`` zoom half-width to the data's Nyquist range.

    Returns the half-width to use for both axes so the whole ``ω = k`` line is
    visible inside the box, or ``None`` to fall back to the full spectrum when
    the series is too small to define a window.
    """
    t = series.coords["t"].values
    xname = next(d for d in series.dims if d != "t")
    x = series.coords[xname].values
    if t.size < 2 or x.size < 2:
        return None
    k_ny = np.pi / float(x[1] - x[0])
    w_ny = np.pi / float(t[1] - t[0])
    cap = min(k_ny, w_ny)
    if requested is None or requested <= 0:
        return cap
    return float(min(requested, cap))


def save_canned_plots(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    v_th: float | None = None,
    dpi: int = 120,
    n_panels: int = 8,
    omega_k_zoom: float | None = 4.0,
) -> dict[str, Path]:
    """Generate the standard set of PNGs for a finished OSIRIS run.

    Returns a mapping of plot-name → output PNG path. Each diagnostic family is
    best-effort: a failure on one diagnostic logs and is skipped rather than
    aborting the rest.

    ``run_dir`` may be a raw OSIRIS run directory (with an ``MS/`` HDF5 tree and
    optional ``HIST/``) **or** the ``binary/`` directory of saved NetCDFs
    written by :func:`io.save_run_datasets`. The data source is detected
    automatically (every diagnostic is fetched through :func:`io.list_diagnostics`
    / :func:`io.load_series`, which handle both layouts), so the full plot set —
    including the field-energy and energy-conservation traces — can be
    regenerated from the saved NetCDF artifacts alone, with no rerun and no raw
    HDF5 dumps.

    ``omega_k_zoom`` is the ``(k, ω)`` half-width (in ``ω_p`` units) for the
    equal-aspect lower panel of the dispersion plots, where ``ω = k`` is drawn
    at 45° (clamped to the data's Nyquist; ``None`` → full Nyquist window).
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _write(fig: plt.Figure, rel: str) -> Path:
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return p

    diags = _io.list_diagnostics(run_dir)

    # --- Fields (FLD/, incl. currents j1-j3): spacetime, log, lineouts, ω-k ---
    for diag_rel, diag_path in sorted(diags.items()):
        if not diag_rel.startswith("FLD/"):
            continue
        comp = diag_rel.split("/", 1)[1]
        try:
            ser = _io.load_series(diag_path)
        except Exception as e:
            print(f"[plots] skipping series {diag_rel}: {e}")
            continue
        if ser.ndim != 2:
            continue  # 2D-in-space field plots deferred

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_spacetime(ser, ax=ax)
        written[f"spacetime/{comp}"] = _write(fig, f"spacetime/{comp}.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_spacetime(ser, ax=ax, log=True)
        written[f"spacetime_log/{comp}"] = _write(fig, f"spacetime_log/{comp}.png")

        written[f"lineouts/{comp}"] = _write(
            plot_lineouts(ser, n_panels=n_panels), f"lineouts/{comp}.png"
        )

        # Full (k, ω) spectrum on top, equal-aspect square window below.
        written[f"omega_k/{comp}"] = _write(
            plot_omega_k_figure(ser, v_th=v_th, omega_k_zoom=omega_k_zoom),
            f"omega_k/{comp}.png",
        )

    # --- Currents (j1/j2/j3) combined views ---
    try:
        fig = plot_currents_spacetime(run_dir)
        if fig is not None:
            written["currents/spacetime"] = _write(fig, "currents/spacetime.png")
        fig = plot_currents_lineouts(run_dir)
        if fig is not None:
            written["currents/lineouts"] = _write(fig, "currents/lineouts.png")
    except Exception as e:
        print(f"[plots] skipping currents: {e}")

    # --- Per-species moments (DENSITY/<species>/<quantity>) ---
    for diag_rel, diag_path in sorted(diags.items()):
        if not diag_rel.startswith("DENSITY/"):
            continue
        parts = diag_rel.split("/")
        if len(parts) < 3:
            continue
        species, quantity = parts[1], "/".join(parts[2:])
        try:
            ser = _io.load_series(diag_path)
        except Exception as e:
            print(f"[plots] skipping moment {diag_rel}: {e}")
            continue
        if ser.ndim != 2:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_spacetime(ser, ax=ax)
        written[f"moments/{species}/{quantity}"] = _write(
            fig, f"moments/{species}/{quantity}.png"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_spacetime(ser, ax=ax, log=True)
        written[f"moments/{species}/{quantity}_log"] = _write(
            fig, f"moments/{species}/{quantity}_log.png"
        )

        written[f"moments/{species}/lineouts/{quantity}"] = _write(
            plot_lineouts(ser, n_panels=n_panels),
            f"moments/{species}/lineouts/{quantity}.png",
        )

    # --- Per-species density + temperature profiles ---
    for species, entries in sorted(_species_diags(run_dir).items()):
        label = species.replace("_", " ")  # "species_1" -> "species 1" for titles
        try:
            dens = _density_series(entries)
            if dens is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_profile(
                    dens, ax=ax, show_initial=True,
                    title=f"{label}  —  density profile",
                )
                written[f"profiles/{species}/density"] = _write(
                    fig, f"profiles/{species}/density.png"
                )
        except Exception as e:
            print(f"[plots] skipping density profile for {species}: {e}")
        try:
            # Temperature from a Maxwellian fit to the species phase space
            # (preferred); fall back to dumped thermal moments (uth / T_ii).
            ps_path = _species_phasespace(diags, species)
            temp = _temperature_from_phasespace(ps_path)
            if temp is None:
                temp = _temperature_series(entries)
            if temp is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_profile(
                    temp, ax=ax, show_initial=True,
                    title=f"{label}  —  temperature profile",
                )
                written[f"profiles/{species}/temperature"] = _write(
                    fig, f"profiles/{species}/temperature.png"
                )
        except Exception as e:
            print(f"[plots] skipping temperature profile for {species}: {e}")

    # --- Phase space (PHA/<ps>/<species>): final step + time evolution ---
    for diag_rel, diag_path in sorted(diags.items()):
        if not diag_rel.startswith("PHA/"):
            continue
        parts = diag_rel.split("/")
        if len(parts) < 3:
            continue
        ps_name, species = parts[1], parts[2]
        try:
            ser = _io.load_series(diag_path)
        except Exception as e:
            print(f"[plots] skipping phasespace {diag_rel}: {e}")
            continue
        # Final-step heatmap: the last time slice of the series (so it works
        # identically off raw dumps and off the saved NetCDF series).
        if "t" in ser.dims and ser.sizes["t"] > 0:
            final = ser.isel(t=-1)
            final.attrs["time"] = float(np.asarray(final.coords["t"].values))
            if final.ndim == 2:
                fig, ax = plt.subplots(figsize=(5, 4))
                plot_phasespace(final, ax=ax)
                written[f"phasespace/{species}/{ps_name}"] = _write(
                    fig, f"phasespace/{species}/{ps_name}.png"
                )
        try:
            if ser.ndim == 3:
                written[f"phasespace_evolution/{species}/{ps_name}"] = _write(
                    plot_phasespace_evolution(ser, n_panels=n_panels),
                    f"phasespace_evolution/{species}/{ps_name}.png",
                )
        except Exception as e:
            print(f"[plots] skipping phasespace evolution {diag_rel}: {e}")

    # --- Left/right-going transverse E-field decomposition ---
    try:
        for comp, fig in plot_field_lr_decomposition(run_dir).items():
            written[f"field_decomp/{comp}"] = _write(fig, f"field_decomp/{comp}.png")
    except Exception as e:
        print(f"[plots] skipping field decomposition: {e}")

    # --- Energy diagnostics ---
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_energy_vs_time(run_dir, ax=ax)
        written["energy_vs_time"] = _write(fig, "energy_vs_time.png")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[plots] skipping energy_vs_time: {e}")

    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_energy_components(run_dir, ax=ax)
        written["energy_components_vs_time"] = _write(
            fig, "energy_components_vs_time.png"
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[plots] skipping energy_components_vs_time: {e}")

    try:
        energy = _io.load_hist_energy(run_dir)
        if energy is not None and "total" in energy:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_energy_conservation(energy, ax=ax)
            written["total_energy_vs_time"] = _write(fig, "total_energy_vs_time.png")
    except Exception as e:
        print(f"[plots] skipping total_energy_vs_time: {e}")

    return written


# --- helpers --------------------------------------------------------------


def _ensure_series(src) -> xr.DataArray:
    if isinstance(src, xr.DataArray):
        return src
    p = Path(src)
    if p.is_dir():
        return _io.load_series(p)
    raise TypeError(
        f"Expected an xr.DataArray or a directory path; got {type(src).__name__}"
    )


def _tex(s) -> str:
    r"""Wrap an OSIRIS label/unit in math mode so its TeX actually renders.

    OSIRIS stores labels and units as bare TeX fragments (``E_1``,
    ``\omega_p``, ``c / \omega_p``, ``1 / \omega_p``). Matplotlib only
    typesets those inside ``$...$``, so a raw ``\omega`` would otherwise be
    drawn literally. We add the delimiters when the string carries TeX
    markup (a backslash, a sub/superscript, or a fraction slash) and leave
    plain words ("charge", "a.u.") untouched. Idempotent: a string that is
    already ``$``-delimited is returned unchanged.
    """
    s = str(s)
    if not s or (s.startswith("$") and s.endswith("$")):
        return s
    if any(c in s for c in "\\_^/"):
        return f"${s}$"
    return s


def _label_tex(s) -> str:
    r"""Render an OSIRIS *label* that mixes prose words and TeX fragments.

    Diagnostic LABELs like ``species 1 p_1x_1`` interleave plain words with math
    fragments. Wrapping the whole string in ``$...$`` (as :func:`_tex` does for
    units, which are entirely mathematical) would typeset "species" in math
    italics with no word spacing. Instead each whitespace-separated token is
    wrapped on its own: tokens carrying TeX markup (``\``, ``_``, ``^``, ``{``,
    ``}``) become math, while plain words and bare numbers stay as text. For a
    single all-math token (``E_1``, ``\rho``) this matches :func:`_tex`.
    """
    s = str(s)
    if not s or (s.startswith("$") and s.endswith("$")):
        return s
    return " ".join(
        f"${tok}$" if tok and any(c in tok for c in "\\_^{}") else tok
        for tok in s.split(" ")
    )


def _axis_label(da: xr.DataArray, dim: str) -> str:
    if dim == "t":
        u = da.attrs.get("time_units") or r"1/\omega_p"
        return rf"$t$  [{_tex(u)}]" if u else r"$t$"
    units = da.attrs.get("axis_units", {}).get(dim, "")
    long = da.attrs.get("axis_long_names", {}).get(dim, dim)
    if units:
        return rf"{_tex(long)}  [{_tex(units)}]"
    return _tex(long)


def _long_name(da: xr.DataArray) -> str:
    """Human-readable label for a diagnostic: its OSIRIS LABEL, else its name."""
    return str(da.attrs.get("long_name") or da.name or "")


def _display_name(da: xr.DataArray) -> str:
    """``_long_name`` rendered for titles (prose stays text, TeX gets math)."""
    return _label_tex(_long_name(da))


def _value_label(da: xr.DataArray) -> str:
    """Quantity label with units (for colorbars / y-axes)."""
    name = _label_tex(_long_name(da))
    units = da.attrs.get("units", "")
    return rf"{name}  [{_tex(units)}]" if units else name


def _decorate(da: xr.DataArray) -> xr.DataArray:
    """Lift OSIRIS per-axis units / long-names onto coordinate attrs.

    ``io.load_series`` stashes axis metadata in dict-valued DataArray attrs;
    copying it onto the coordinates lets xarray's faceted plotters auto-label
    each panel's axes. Returns a decorated copy (the input is left untouched).
    """
    da = da.copy()
    axis_units = da.attrs.get("axis_units", {}) or {}
    axis_long = da.attrs.get("axis_long_names", {}) or {}
    for dim in da.dims:
        if dim not in da.coords:
            continue
        if dim in axis_long:
            da.coords[dim].attrs.setdefault("long_name", axis_long[dim])
        if dim in axis_units:
            da.coords[dim].attrs.setdefault("units", axis_units[dim])
    if "t" in da.coords:
        da.coords["t"].attrs.setdefault("long_name", "t")
        da.coords["t"].attrs.setdefault("units", da.attrs.get("time_units", r"1/\omega_p"))
    return da


def _cell_volume(da: xr.DataArray) -> float:
    """Product of uniform coordinate spacings over the spatial dims."""
    dvol = 1.0
    for dim in da.dims:
        if dim == "t":
            continue
        c = da.coords[dim].values
        if c.size > 1:
            dvol *= float(c[1] - c[0])
    return dvol


def _field_energy_from_series(ser: xr.DataArray) -> np.ndarray:
    """Per-timestep field energy ``0.5 * ∫ f^2 dx`` for a ``(t, x)`` series.

    Returns a 1-D array aligned with the series' ``t`` axis. Equivalent to
    summing ``0.5 * f^2 * cell_volume`` over space at each dump, but vectorized
    over the whole stacked series so it works identically off the raw HDF5
    dumps and off the saved NetCDFs.
    """
    spatial = [d for d in ser.dims if d != "t"]
    sq = (ser.astype("float64") ** 2).sum(dim=spatial)
    return 0.5 * np.asarray(sq.values, dtype="float64") * _cell_volume(ser)


# --- public helpers for downstream post-processing ------------------------
# Project repos (e.g. osiris-lpi) build their own canned plots on these shared
# label / decoration / box helpers (and on `efield_lr_components`); export
# stable public names so they need not reach into the private (underscore) API.
decorate = _decorate
ensure_series = _ensure_series
axis_label = _axis_label
display_name = _display_name
tex = _tex
sim_box_xmax = _sim_box_xmax
