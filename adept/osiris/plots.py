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
      omega_k/<diag>.png                   2-D FFT (k, ω) dispersion plot
      omega_k_zoom/<diag>.png              same, zoomed to the ω = k line (plasma waves)
      currents/spacetime.png               j1/j2/j3 (J_x/J_y/J_z) side-by-side spacetime
      currents/lineouts.png                j1/j2/j3 profiles vs x (final + late mean)
      moments/<species>/<q>.png            (t, x) per-species density moments
      moments/<species>/<q>_log.png        log10 of the same
      moments/<species>/lineouts/<q>.png   moment snapshots at sampled times
      profiles/<species>/density.png       density profile vs x (final + late mean)
      profiles/<species>/temperature.png   temperature profile vs x (if thermal moments)
      phasespace/<species>/<ps>.png        final-step (x, p) heatmap per species
      phasespace_evolution/<species>/<ps>.png  (x, p) heatmaps at sampled times
      distribution_lineouts/<species>/<ps>.png  f(p) averaged over the right-boundary cells
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
    title: str | None = None,
) -> plt.Axes:
    """``(t, x)`` heatmap of a 1D-field time-series.

    Accepts a pre-loaded ``DataArray`` (dims must include ``t`` and one
    spatial axis) or a directory of dumps that ``load_series`` will eat.
    """
    da = _ensure_series(series)
    if da.ndim != 2:
        raise ValueError(
            f"plot_spacetime expects a 2D (t, x) array; got dims {da.dims}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    data = np.log10(np.abs(da.values) + 1e-30) if log else da.values
    t = da.coords["t"].values
    xname = next(d for d in da.dims if d != "t")
    x = da.coords[xname].values
    mesh = ax.pcolormesh(t, x, data.T, shading="auto", cmap=cmap)
    plt.colorbar(
        mesh,
        ax=ax,
        label=rf"$\log_{{10}}$ |{_value_label(da)}|" if log else _value_label(da),
    )
    ax.set_xlabel(_axis_label(da, "t"))
    ax.set_ylabel(_axis_label(da, xname))
    scale = r"$\log_{10}$ " if log else ""
    ax.set_title(title or f"{_display_name(da)}  —  {scale}spacetime ($x$ vs $t$)")
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
    """Heatmap of a 2D phase-space density (e.g. ``x1p1``)."""
    if not isinstance(da, xr.DataArray):
        da = _io.load_phasespace_h5(da)
    if da.ndim != 2:
        raise ValueError(
            f"plot_phasespace expects 2D data; got {da.dims} {da.shape}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    arr = da.values
    if log:
        arr = np.log10(np.abs(arr) + 1e-30)
    d0, d1 = da.dims
    x0, x1 = da.coords[d0].values, da.coords[d1].values
    mesh = ax.pcolormesh(x0, x1, arr.T, shading="auto", cmap=cmap)
    plt.colorbar(
        mesh, ax=ax, label=rf"$\log_{{10}}$ {_value_label(da)}" if log else _value_label(da)
    )
    ax.set_xlabel(_axis_label(da, d0))
    ax.set_ylabel(_axis_label(da, d1))
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

    Walks every component dir under ``MS/FLD/`` (``e1``, ``e2``, ..., ``b1``,
    ``b2``, ``b3``), aligns by iteration count, and returns a 1D
    ``DataArray`` of total field energy in code units vs time. Components
    that aren't dumped (or that have a different save cadence) are
    skipped.
    """
    run_dir = Path(run_dir)
    fld = run_dir / "MS" / "FLD"
    if not fld.is_dir():
        raise FileNotFoundError(f"No MS/FLD under {run_dir}")
    components = ("e1", "e2", "e3", "b1", "b2", "b3")
    by_iter: dict[int, dict[str, float]] = {}
    times: dict[int, float] = {}
    for comp in components:
        d = fld / comp
        if not d.is_dir():
            continue
        for h5 in sorted(d.iterdir()):
            if h5.suffix != ".h5":
                continue
            da = _io.load_grid_h5(h5)
            e = 0.5 * float((da.values ** 2).sum()) * _cell_volume(da)
            it = int(da.attrs["iter"])
            by_iter.setdefault(it, {})[comp] = e
            times.setdefault(it, float(da.attrs["time"]))
    if not by_iter:
        raise RuntimeError(f"No field dumps under {fld}")
    iters = sorted(by_iter)
    totals = np.array([sum(by_iter[i].values()) for i in iters])
    t = np.array([times[i] for i in iters])
    return xr.DataArray(
        totals,
        coords={"t": t, "iter": ("t", np.asarray(iters))},
        dims=("t",),
        name="field_energy",
        attrs={"long_name": "total field energy", "units": "code"},
    )


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
    """E-field, B-field, and total field energy vs time from ``MS/FLD`` dumps.

    Like :func:`field_energy_series` but keeps the electric (``e1/e2/e3``) and
    magnetic (``b1/b2/b3``) contributions separate. Returns a ``Dataset`` with
    data variables ``E_energy``, ``B_energy`` and ``total_field_energy`` on a
    shared ``t`` axis. Components not dumped are simply omitted from their sum.
    """
    run_dir = Path(run_dir)
    fld = run_dir / "MS" / "FLD"
    if not fld.is_dir():
        raise FileNotFoundError(f"No MS/FLD under {run_dir}")
    groups = {"E_energy": ("e1", "e2", "e3"), "B_energy": ("b1", "b2", "b3")}
    by_iter: dict[int, dict[str, float]] = {}
    times: dict[int, float] = {}
    for group, comps in groups.items():
        for comp in comps:
            d = fld / comp
            if not d.is_dir():
                continue
            for h5 in sorted(d.iterdir()):
                if h5.suffix != ".h5":
                    continue
                da = _io.load_grid_h5(h5)
                e = 0.5 * float((da.values ** 2).sum()) * _cell_volume(da)
                it = int(da.attrs["iter"])
                rec = by_iter.setdefault(it, {"E_energy": 0.0, "B_energy": 0.0})
                rec[group] += e
                times.setdefault(it, float(da.attrs["time"]))
    if not by_iter:
        raise RuntimeError(f"No field dumps under {fld}")
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
    ax.set_xlabel(r"$k\ [\omega_p / c]$")
    ax.set_ylabel(r"$\omega\ [\omega_p]$")
    ax.set_title(title or rf"{_display_name(da)}  —  $(k, \omega)$ power spectrum")
    return ax


def plot_distribution_lineout(
    series: xr.DataArray | str | Path,
    ax: plt.Axes | None = None,
    *,
    n_cells: int = 10,
    n_times: int = 6,
    log: bool = False,
    cmap: str = "viridis",
    title: str | None = None,
) -> plt.Axes:
    """Velocity distribution ``f(p)`` averaged over the rightmost ``n_cells``.

    Takes a phase-space series with a spatial (``x*``) and a momentum
    (``p*``) axis — either a stacked ``(t, p, x)`` series from
    :func:`io.load_series` or a single ``(p, x)`` snapshot — averages it over
    the ``n_cells`` cells nearest the right-hand boundary to get ``f(p)``
    there, and overlays that lineout at ~``n_times`` sampled times (colour =
    time). This is the standard "what does the distribution look like as it
    leaves the box" diagnostic for driven / open-boundary runs.
    """
    da = series if isinstance(series, xr.DataArray) else _io.load_series(series)
    da = _decorate(da)
    spatial = [d for d in da.dims if d != "t" and str(d).startswith("x")]
    moment = [d for d in da.dims if d != "t" and str(d).startswith("p")]
    if not spatial or not moment:
        raise ValueError(
            f"plot_distribution_lineout needs an x* and a p* dim; got {da.dims}"
        )
    xdim, pdim = spatial[0], moment[0]
    nx = da.sizes[xdim]
    k = max(1, min(n_cells, nx))
    # Average over the k rightmost spatial cells -> f over the momentum axis.
    f_xp = da.isel({xdim: slice(nx - k, nx)}).mean(dim=xdim)
    p = f_xp.coords[pdim].values
    xv = da.coords[xdim].values
    x_lo, x_hi = float(xv[nx - k]), float(xv[-1])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if "t" in f_xp.dims:
        nt = f_xp.sizes["t"]
        idx = np.unique(np.linspace(0, nt - 1, min(n_times, nt)).astype(int))
        colours = plt.get_cmap(cmap)(np.linspace(0.15, 0.95, len(idx)))
        tvals = f_xp.coords["t"].values
        for colour, i in zip(colours, idx, strict=False):
            y = f_xp.isel(t=i).values
            y = np.abs(y) if log else y
            ax.plot(p, y, color=colour, lw=1.3, label=f"{float(tvals[i]):.2g}")
        ax.legend(fontsize=7, title=_axis_label(da, "t"), ncol=2, framealpha=0.6)
    else:
        y = np.abs(f_xp.values) if log else f_xp.values
        ax.plot(p, y, lw=1.4)

    if log:
        ax.set_yscale("log")
    units = da.attrs.get("units", "")
    ylabel = rf"$\langle f \rangle_x$  [{_tex(units)}]" if units else r"$\langle f \rangle_x$"
    ax.set_xlabel(_axis_label(da, pdim))
    ax.set_ylabel(ylabel)
    ax.set_title(
        title
        or rf"{_display_name(da)}  —  $f$ over rightmost {k} cells "
        rf"($x \in [{x_lo:.3g}, {x_hi:.3g}]$)"
    )
    ax.grid(True, alpha=0.3)
    return ax


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
    fig.suptitle(r"Current density components  —  spacetime ($x$ vs $t$)", y=1.03)
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
    """Pick the best density-like moment for a species and load it as ``(t, x)``."""
    by_q = {q: p for kind, q, p in entries if kind == "DENSITY"}
    for pref in ("charge", "n", "n01", "m", "ene"):
        if pref in by_q:
            ser = _io.load_series(by_q[pref])
            return ser if ser.ndim == 2 else None
    # Fall back to any DENSITY entry.
    for kind, q, p in entries:
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


def plot_profile(
    series: xr.DataArray | str | Path,
    ax: plt.Axes | None = None,
    *,
    abs_value: bool = False,
    n_avg_frac: float = 0.2,
    value_label: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot a ``(t, x)`` moment vs ``x``: final snapshot plus late-time mean.

    The late-time mean (over the last ``n_avg_frac`` of the dumps) smooths
    out the time-dependent fluctuations to show the established profile.
    """
    da = _decorate(series if isinstance(series, xr.DataArray) else _io.load_series(series))
    if "t" not in da.dims:
        raise ValueError(f"plot_profile expects a (t, x) series; got dims {da.dims}")
    xdim = next(d for d in da.dims if d != "t")
    x = da.coords[xdim].values
    nt = da.sizes["t"]
    w = max(1, round(n_avg_frac * nt))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    final = da.isel(t=-1).values
    mean = da.isel(t=slice(nt - w, nt)).mean("t").values
    if abs_value:
        final, mean = np.abs(final), np.abs(mean)
    tlast = float(da.coords["t"].values[-1])
    ax.plot(x, final, lw=1.4, label=f"final ($t={tlast:.3g}$)")
    ax.plot(x, mean, lw=1.1, ls="--", label=f"mean of last {w} dumps")
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


def plot_field_lr_decomposition(
    run_dir: str | Path,
    *,
    v_th: float | None = None,
    omega_k_zoom: float | None = 4.0,
) -> dict[str, plt.Figure]:
    """One 2x2 figure per transverse component: right/left spacetime + ω-k.

    Top row is the right/left-going ``(t, x)`` spacetime; bottom row is each
    part's ``(k, ω)`` power spectrum (zoomed, with the light line drawn) so
    you can confirm the right-going part really does sit on the ``ω·k > 0``
    branches and the left-going part on ``ω·k < 0``.
    """
    figs: dict[str, plt.Figure] = {}
    for comp, parts in efield_lr_components(run_dir).items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for col, side in enumerate(("right", "left")):
            da = parts[side]
            plot_spacetime(da, ax=axes[0, col], title=f"{_tex(_long_name(da))}  —  spacetime")
            try:
                zoom = _omega_k_zoom_window(da, omega_k_zoom)
                plot_omega_k(
                    da, ax=axes[1, col], show_light_line=True,
                    show_langmuir=v_th is not None, v_th=v_th,
                    k_max=zoom, omega_max=zoom,
                    title=f"{_tex(_long_name(da))}  —  $(k, \\omega)$",
                )
            except Exception as e:  # ω-k overlay is best effort
                axes[1, col].set_visible(False)
                print(f"[plots] ω-k for {comp} {side} skipped: {e}")
        fig.suptitle(
            rf"{comp}: left/right-going decomposition (vacuum Riemann split)", y=1.01
        )
        fig.tight_layout()
        figs[comp] = fig
    return figs


# --- driver ---------------------------------------------------------------


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
    dist_cells: int = 10,
    omega_k_zoom: float | None = 4.0,
) -> dict[str, Path]:
    """Generate the standard set of PNGs for a finished OSIRIS run.

    Returns a mapping of plot-name → output PNG path. Each diagnostic family is
    best-effort: a failure on one diagnostic logs and is skipped rather than
    aborting the rest.

    ``dist_cells`` sets how many right-boundary cells the phase-space
    distribution lineouts average over; ``omega_k_zoom`` is the ``(k, ω)``
    half-width (in ``ω_p`` units) for the zoomed dispersion plots that show the
    whole ``ω = k`` line (``None`` → full spectrum).
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

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_omega_k(ser, ax=ax, show_langmuir=v_th is not None, v_th=v_th)
        written[f"omega_k/{comp}"] = _write(fig, f"omega_k/{comp}.png")

        # Zoomed (k, ω) view sized so the whole ω = k line is visible — the
        # window where the plasma (Langmuir) waves live.
        zoom = _omega_k_zoom_window(ser, omega_k_zoom)
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_omega_k(
            ser, ax=ax, show_light_line=True,
            show_langmuir=v_th is not None, v_th=v_th,
            k_max=zoom, omega_max=zoom,
            title=f"{_display_name(ser)}  —  $(k, \\omega)$ (zoom)",
        )
        written[f"omega_k_zoom/{comp}"] = _write(fig, f"omega_k_zoom/{comp}.png")

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
        try:
            dens = _density_series(entries)
            if dens is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_profile(dens, ax=ax, title=f"{species}  —  density profile")
                written[f"profiles/{species}/density"] = _write(
                    fig, f"profiles/{species}/density.png"
                )
        except Exception as e:
            print(f"[plots] skipping density profile for {species}: {e}")
        try:
            temp = _temperature_series(entries)
            if temp is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_profile(temp, ax=ax, title=f"{species}  —  temperature profile")
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
        last = _latest_h5(diag_path)
        if last is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_phasespace(_io.load_phasespace_h5(last), ax=ax)
            written[f"phasespace/{species}/{ps_name}"] = _write(
                fig, f"phasespace/{species}/{ps_name}.png"
            )
        try:
            ser = _io.load_series(diag_path)
            if ser.ndim == 3:
                written[f"phasespace_evolution/{species}/{ps_name}"] = _write(
                    plot_phasespace_evolution(ser, n_panels=n_panels),
                    f"phasespace_evolution/{species}/{ps_name}.png",
                )
                # f(p) averaged over the right-boundary cells, vs time.
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_distribution_lineout(ser, ax=ax, n_cells=dist_cells)
                written[f"distribution_lineouts/{species}/{ps_name}"] = _write(
                    fig, f"distribution_lineouts/{species}/{ps_name}.png"
                )
        except Exception as e:
            print(f"[plots] skipping phasespace evolution {diag_rel}: {e}")

    # --- Left/right-going transverse E-field decomposition ---
    try:
        for comp, fig in plot_field_lr_decomposition(
            run_dir, v_th=v_th, omega_k_zoom=omega_k_zoom
        ).items():
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
    """``_long_name`` wrapped for math-mode rendering (used in titles)."""
    return _tex(_long_name(da))


def _value_label(da: xr.DataArray) -> str:
    """Quantity label with units (for colorbars / y-axes)."""
    name = _tex(_long_name(da))
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


def _latest_h5(diag_dir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for p in diag_dir.iterdir():
        if p.suffix != ".h5":
            continue
        m = _io._ITER_RE.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        if best is None or it > best[0]:
            best = (it, p)
    return best[1] if best else None
