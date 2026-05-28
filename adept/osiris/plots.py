"""Canned matplotlib views over OSIRIS diagnostics.

Each plotter accepts either an already-loaded ``xarray.DataArray`` or a
filesystem path / directory, and returns the ``Axes`` it drew on so the
caller can tweak labels / colorbars / save the figure.

``save_canned_plots(run_dir, out_dir)`` ties them together and writes
PNGs for the standard set:

    out_dir/
      spacetime/<diag>.png                 (t, x) heatmap of every FLD diagnostic
      spacetime_log/<diag>.png             log10|·| of the same
      lineouts/<diag>.png                  value-vs-x snapshots at sampled times
      moments/<species>/<q>.png            (t, x) per-species density moments
      moments/<species>/<q>_log.png        log10 of the same
      moments/<species>/lineouts/<q>.png   moment snapshots at sampled times
      phasespace/<species>/<ps>.png        final-step (x, p) heatmap per species
      phasespace_evolution/<species>/<ps>.png  (x, p) heatmaps at sampled times
      omega_k/<diag>.png                   2-D FFT (k, ω) dispersion plot
      energy_vs_time.png                   total field energy time-trace
      energy_components_vs_time.png        E-field / B-field / total energy
      total_energy_vs_time.png             field + kinetic conservation (needs HIST/)
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
        label=f"log10 |{_value_label(da)}|" if log else _value_label(da),
    )
    ax.set_xlabel(_axis_label(da, "t"))
    ax.set_ylabel(_axis_label(da, xname))
    scale = "log10 " if log else ""
    ax.set_title(title or f"{_long_name(da)}  —  {scale}spacetime (x vs t)")
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
        title or f"{_long_name(da)}  —  lineouts vs x at sampled times", y=1.02
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
        mesh, ax=ax, label=f"log10 {_value_label(da)}" if log else _value_label(da)
    )
    ax.set_xlabel(_axis_label(da, d0))
    ax.set_ylabel(_axis_label(da, d1))
    t = da.attrs.get("time", float("nan"))
    scale = "log10 " if log else ""
    ax.set_title(title or f"{_long_name(da)}  —  {scale}phase space  (t = {t:.3g} 1/$\\omega_p$)")
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
    scale = "log10 " if log else ""
    g.fig.suptitle(
        title or f"{_long_name(da)}  —  {scale}phase space at sampled times", y=1.02
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
    you ask for them.
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
    plt.colorbar(mesh, ax=ax, label=("log10 |F|^2" if log else "|F|^2"))

    if k_max is None:
        k_max = float(np.max(np.abs(k)))
    if omega_max is None:
        omega_max = float(np.max(np.abs(omega)))
    ax.set_xlim(-k_max, k_max)
    ax.set_ylim(-omega_max, omega_max)

    # Overlay analytical dispersion lines.
    k_line = np.linspace(-k_max, k_max, 401)
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
    if show_em or show_langmuir:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)

    ax.axhline(0, color="w", lw=0.4, alpha=0.4)
    ax.axvline(0, color="w", lw=0.4, alpha=0.4)
    ax.set_xlabel(r"$k\ [\omega_p / c]$")
    ax.set_ylabel(r"$\omega\ [\omega_p]$")
    ax.set_title(title or f"{_long_name(da)}  —  $(k, \\omega)$ power spectrum")
    return ax


# --- driver ---------------------------------------------------------------


def save_canned_plots(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    v_th: float | None = None,
    dpi: int = 120,
    n_panels: int = 8,
) -> dict[str, Path]:
    """Generate the standard set of PNGs for a finished OSIRIS run.

    Returns a mapping of plot-name → output PNG path. Each diagnostic family is
    best-effort: a failure on one diagnostic logs and is skipped rather than
    aborting the rest.
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
        except Exception as e:
            print(f"[plots] skipping phasespace evolution {diag_rel}: {e}")

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


def _axis_label(da: xr.DataArray, dim: str) -> str:
    if dim == "t":
        u = da.attrs.get("time_units", r"$1/\omega_p$")
        return f"t  [{u}]" if u else "t"
    units = da.attrs.get("axis_units", {}).get(dim, "")
    long = da.attrs.get("axis_long_names", {}).get(dim, dim)
    if units:
        # OSIRIS axis units are TeX-ish (e.g. "c / \\omega_p"); wrap in $...$.
        return rf"{long}  [${units}$]"
    return long


def _long_name(da: xr.DataArray) -> str:
    """Human-readable label for a diagnostic: its OSIRIS LABEL, else its name."""
    return str(da.attrs.get("long_name") or da.name or "")


def _value_label(da: xr.DataArray) -> str:
    """Quantity label with units (for colorbars / y-axes)."""
    name = _long_name(da)
    units = da.attrs.get("units", "")
    return rf"{name}  [${units}$]" if units else name


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
