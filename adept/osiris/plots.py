"""Canned matplotlib views over OSIRIS diagnostics.

Each plotter accepts either an already-loaded ``xarray.DataArray`` or a
filesystem path / directory, and returns the ``Axes`` it drew on so the
caller can tweak labels / colorbars / save the figure.

``save_canned_plots(run_dir, out_dir)`` ties them together and writes
PNGs for the standard set:

    out_dir/
      spacetime/<diag>.png            (t, x) heatmaps of every FLD diagnostic
      phasespace/<species>/<ps>.png   final-step (x, p) heatmap per species
      energy_vs_time.png              total field energy time-trace
      omega_k/<diag>.png              2-D FFT (k, ω) dispersion plot
"""

from __future__ import annotations

from pathlib import Path

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
        label=f"log10 |{da.name}|" if log else da.name,
    )
    ax.set_xlabel(_axis_label(da, "t"))
    ax.set_ylabel(_axis_label(da, xname))
    ax.set_title(title or f"{da.name}  spacetime")
    return ax


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
        mesh, ax=ax, label=f"log10 {da.name}" if log else da.name
    )
    ax.set_xlabel(_axis_label(da, d0))
    ax.set_ylabel(_axis_label(da, d1))
    t = da.attrs.get("time", float("nan"))
    ax.set_title(title or f"{da.name}   t = {t:.3g}")
    return ax


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
            # cell volume from coord spacing in each dim
            dvol = 1.0
            for dim in da.dims:
                c = da.coords[dim].values
                if c.size > 1:
                    dvol *= float(c[1] - c[0])
            e = 0.5 * float((da.values ** 2).sum()) * dvol
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
    ax.set_ylabel(r"$\int (|E|^2 + |B|^2)/2\ d^Dx$")
    ax.set_title(title or "field energy")
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
    ax.set_title(title or f"{da.name}  omega-k spectrum")
    return ax


# --- driver ---------------------------------------------------------------


def save_canned_plots(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    v_th: float | None = None,
    dpi: int = 120,
) -> dict[str, Path]:
    """Generate the standard set of PNGs for a finished OSIRIS run.

    Returns a mapping of plot-name → output PNG path.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    diags = _io.list_diagnostics(run_dir)

    # Spacetime + omega-k for each field component (those live under FLD/).
    fld_st = out_dir / "spacetime"
    fld_st.mkdir(exist_ok=True)
    omk_dir = out_dir / "omega_k"
    omk_dir.mkdir(exist_ok=True)
    for diag_rel, diag_path in diags.items():
        if not diag_rel.startswith("FLD/"):
            continue
        comp = diag_rel.split("/", 1)[1]
        try:
            ser = _io.load_series(diag_path)
        except Exception as e:
            print(f"[plots] skipping series {diag_rel}: {e}")
            continue
        if ser.ndim != 2:
            continue  # higher-dim FLD plots skipped for now
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_spacetime(ser, ax=ax)
        p = fld_st / f"{comp}.png"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        written[f"spacetime/{comp}"] = p

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_omega_k(
            ser,
            ax=ax,
            show_langmuir=v_th is not None,
            v_th=v_th,
        )
        p = omk_dir / f"{comp}.png"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        written[f"omega_k/{comp}"] = p

    # Final-step phase space per species.
    ph_dir = out_dir / "phasespace"
    for diag_rel, diag_path in diags.items():
        if not diag_rel.startswith("PHA/"):
            continue
        # PHA/x1p1/beam_pos -> ps_name="x1p1", species="beam_pos"
        parts = diag_rel.split("/")
        if len(parts) < 3:
            continue
        ps_name, species = parts[1], parts[2]
        last = _latest_h5(diag_path)
        if last is None:
            continue
        sp_out = ph_dir / species
        sp_out.mkdir(parents=True, exist_ok=True)
        da = _io.load_phasespace_h5(last)
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_phasespace(da, ax=ax)
        p = sp_out / f"{ps_name}.png"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        written[f"phasespace/{species}/{ps_name}"] = p

    # Energy vs time (uses any/all FLD components present).
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_energy_vs_time(run_dir, ax=ax)
        p = out_dir / "energy_vs_time.png"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        written["energy_vs_time"] = p
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[plots] skipping energy_vs_time: {e}")

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
