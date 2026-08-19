"""Artifact generation and reconnection diagnostics for VFP-2D runs."""

from __future__ import annotations

import os
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.switch_backend("Agg")

COMPONENTS = ("x", "y", "z")


def _selected_indices(nt: int, n_panels: int) -> np.ndarray:
    return np.unique(np.linspace(0, nt - 1, min(nt, n_panels)).round().astype(int))


def _physical_axes(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(ds.x) * float(ds.attrs.get("length_unit_um", 1.0))
    y = np.asarray(ds.y) * float(ds.attrs.get("length_unit_um", 1.0))
    t = np.asarray(ds.t) * float(ds.attrs.get("time_unit_ps", 1.0))
    return x, y, t


def save_xy_facet(
    field: xr.DataArray,
    ds: xr.Dataset,
    path: str,
    *,
    n_panels: int = 9,
    diverging: bool = True,
    title: str | None = None,
) -> None:
    """Save evenly spaced x-y panels with one color scale for all times."""

    x, y, t = _physical_axes(ds)
    indices = _selected_indices(field.sizes["t"], n_panels)
    values = np.asarray(field.isel(t=indices))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif diverging:
        vmax = max(float(np.max(np.abs(finite))), np.finfo(float).tiny)
        vmin = -vmax
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0

    ncols = min(3, indices.size)
    nrows = int(np.ceil(indices.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.25 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for panel, (ax, index) in enumerate(zip(axes.flat, indices, strict=False)):
        image = ax.pcolormesh(
            x,
            y,
            values[panel].T,
            shading="auto",
            cmap="RdBu_r" if diverging else "viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"t = {t[index]:.3g} ps")
        ax.set_xlabel("x [μm]")
        ax.set_ylabel("y [μm]")
        ax.set_aspect("equal")
    for ax in axes.flat[indices.size :]:
        ax.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat[: indices.size]), shrink=0.82)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _vector_potential(bx: np.ndarray, by: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return periodic A_z with B_x=∂_y A_z and B_y=-∂_x A_z."""

    nx, ny = bx.shape[-2:]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx2d, ky2d = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx2d**2 + ky2d**2
    bx_k = np.fft.fftn(bx, axes=(-2, -1))
    by_k = np.fft.fftn(by, axes=(-2, -1))
    numerator = -1j * ky2d * bx_k + 1j * kx2d * by_k
    az_k = np.divide(numerator, k2, out=np.zeros_like(numerator), where=k2 > 0.0)
    return np.fft.ifftn(az_k, axes=(-2, -1)).real


def add_reconnection_diagnostics(ds: xr.Dataset) -> xr.Dataset:
    """Add topology, X-point, rate, and sheet-width diagnostics."""

    bx = np.asarray(ds.b.sel(component="x"))
    by = np.asarray(ds.b.sel(component="y"))
    x = np.asarray(ds.x)
    y = np.asarray(ds.y)
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    az = _vector_potential(bx, by, dx, dy)
    ix0, iy0 = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))

    bx_line = bx[:, ix0, :]
    upstream_iy = np.argmax(np.abs(bx_line), axis=-1)
    upstream_bx = np.take_along_axis(np.abs(bx_line), upstream_iy[:, None], axis=-1)[:, 0]
    upstream_az = np.take_along_axis(az[:, ix0, :], upstream_iy[:, None], axis=-1)[:, 0]
    reconnected_flux = az[:, ix0, iy0] - upstream_az

    vn_y = np.asarray(ds.v_nernst.sel(component="y"))[:, ix0, :]
    upstream_vn = np.max(np.abs(vn_y), axis=-1)
    ez_x = np.asarray(ds.e.sel(component="z"))[:, ix0, iy0]
    scale = upstream_bx * upstream_vn
    normalized_rate = np.divide(ez_x, scale, out=np.zeros_like(ez_x), where=scale > 1e-30)

    jz_line = np.abs(np.asarray(ds.current.sel(component="z"))[:, ix0, :])
    y_from_sheet = y - y[iy0]
    sheet_width = np.sqrt(
        np.divide(
            np.sum(jz_line * y_from_sheet[None, :] ** 2, axis=-1),
            np.sum(jz_line, axis=-1),
            out=np.zeros(jz_line.shape[0]),
            where=np.sum(jz_line, axis=-1) > 1e-30,
        )
    )

    result = ds.assign(
        az=(("t", "x", "y"), az),
        xpoint_ez=("t", ez_x),
        upstream_bx=("t", upstream_bx),
        upstream_v_nernst_y=("t", upstream_vn),
        normalized_reconnection_rate=("t", normalized_rate),
        reconnected_flux=("t", reconnected_flux),
        current_sheet_rms_width=("t", sheet_width),
    )
    result.az.attrs["definition"] = "periodic A_z: B_x=d_y A_z, B_y=-d_x A_z"
    result.normalized_reconnection_rate.attrs["definition"] = "E_z(X)/(max_y|B_x| max_y|v_N,y|) at x=0"
    result.reconnected_flux.attrs["definition"] = "A_z(X)-A_z(x=0,y_at_max_|B_x|)"
    result.current_sheet_rms_width.attrs["definition"] = "|j_z|-weighted RMS y width at x=0"
    for name in (
        "ohm_resistive",
        "ohm_hall",
        "ohm_nernst",
        "ohm_scalar_pressure",
        "ohm_tensor_pressure",
    ):
        if name in result:
            result[f"xpoint_{name}"] = ("t", np.asarray(result[name].sel(component="z"))[:, ix0, iy0])
    return result


def _write_binary(ds: xr.Dataset, binary_dir: str) -> None:
    moments = ds.drop_vars(["flm_real", "flm_imag"], errors="ignore")
    distribution = ds[[name for name in ("flm_real", "flm_imag") if name in ds]]

    def encoding(dataset: xr.Dataset) -> dict:
        return {
            name: {"compression": "gzip", "compression_opts": 1, "shuffle": True}
            for name, value in dataset.data_vars.items()
            if np.issubdtype(value.dtype, np.number) and value.ndim > 0
        }

    moments.to_netcdf(os.path.join(binary_dir, "moments.nc"), engine="h5netcdf", encoding=encoding(moments))
    if distribution.data_vars:
        distribution.to_netcdf(
            os.path.join(binary_dir, "distribution_flm.nc"),
            engine="h5netcdf",
            encoding=encoding(distribution),
        )


def _plot_regular_moments(ds: xr.Dataset, plot_dir: str, n_panels: int) -> None:
    scalar_fields = {
        "density": (ds.ne, False),
        "temperature": (ds.temperature, False),
        "magnetic_field_magnitude": (np.sqrt((ds.b**2).sum("component")), False),
        "current_magnitude": (np.sqrt((ds.current**2).sum("component")), False),
        "nernst_velocity_magnitude": (np.sqrt((ds.v_nernst**2).sum("component")), False),
    }
    for name, (field, diverging) in scalar_fields.items():
        save_xy_facet(
            field,
            ds,
            os.path.join(plot_dir, f"xy_facet_{name}.png"),
            n_panels=n_panels,
            diverging=diverging,
            title=name.replace("_", " "),
        )

    for variable in ("e", "b", "current", "v_nernst"):
        for component in COMPONENTS:
            save_xy_facet(
                ds[variable].sel(component=component),
                ds,
                os.path.join(plot_dir, f"xy_facet_{variable}_{component}.png"),
                n_panels=n_panels,
                title=f"{variable.replace('_', ' ')} {component}",
            )

    pressure_components: Iterable[tuple[str, str]] = (
        ("x", "x"),
        ("x", "y"),
        ("x", "z"),
        ("y", "y"),
        ("y", "z"),
        ("z", "z"),
    )
    for first, second in pressure_components:
        field = ds.pressure_anisotropy.sel(component=first, component_2=second)
        save_xy_facet(
            field,
            ds,
            os.path.join(plot_dir, f"xy_facet_pressure_{first}{second}.png"),
            n_panels=n_panels,
            title=f"pressure anisotropy {first}{second}",
        )


def _plot_xpoint_history(ds: xr.Dataset, path: str) -> None:
    _, _, t = _physical_axes(ds)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True, sharex=True)
    axes[0].plot(t, ds.xpoint_ez, color="black", linewidth=2, label="total E_z")
    for name in ("ohm_resistive", "ohm_hall", "ohm_nernst", "ohm_scalar_pressure", "ohm_tensor_pressure"):
        xpoint_name = f"xpoint_{name}"
        if xpoint_name in ds:
            axes[0].plot(t, ds[xpoint_name], label=name.removeprefix("ohm_").replace("_", " "))
    axes[0].set_ylabel("X-point E_z [norm.] ")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, ds.normalized_reconnection_rate, label="E_z/(B_up v_N,in)")
    axes[1].plot(t, ds.reconnected_flux, label="reconnected flux")
    axes[1].set_ylabel("reconnection diagnostics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    length_um = float(ds.attrs.get("length_unit_um", 1.0))
    axes[2].plot(t, ds.current_sheet_rms_width * length_um, label="current-sheet RMS width")
    axes[2].plot(t, ds.upstream_bx, label="upstream |B_x|")
    axes[2].plot(t, ds.upstream_v_nernst_y, label="inflow |v_N,y|")
    axes[2].set_xlabel("t [ps]")
    axes[2].set_ylabel("width [μm] / normalized amplitude")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ohm_lineouts(ds: xr.Dataset, path: str, n_panels: int) -> None:
    x, y, t = _physical_axes(ds)
    del x
    ix0 = int(np.argmin(np.abs(np.asarray(ds.x))))
    indices = _selected_indices(ds.sizes["t"], min(n_panels, 4))
    fig, axes = plt.subplots(indices.size, 1, figsize=(8, 2.8 * indices.size), constrained_layout=True, squeeze=False)
    for ax, index in zip(axes[:, 0], indices, strict=True):
        ax.plot(y, ds.e.isel(t=index, x=ix0).sel(component="z"), color="black", linewidth=2, label="total E_z")
        for name in ("ohm_resistive", "ohm_hall", "ohm_nernst", "ohm_scalar_pressure", "ohm_tensor_pressure"):
            if name in ds:
                ax.plot(
                    y,
                    ds[name].isel(t=index, x=ix0).sel(component="z"),
                    label=name.removeprefix("ohm_").replace("_", " "),
                )
        ax.set_title(f"x = 0, t = {t[index]:.3g} ps")
        ax.set_xlabel("y [μm]")
        ax.set_ylabel("E_z [norm.]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topology(ds: xr.Dataset, path: str) -> None:
    x, y, t = _physical_axes(ds)
    index = ds.sizes["t"] - 1
    bmag = np.sqrt(np.asarray((ds.b.isel(t=index) ** 2).sum("component")))
    az = np.asarray(ds.az.isel(t=index))
    vx = np.asarray(ds.v_nernst.isel(t=index).sel(component="x"))
    vy = np.asarray(ds.v_nernst.isel(t=index).sel(component="y"))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.pcolormesh(x, y, bmag.T, shading="auto", cmap="magma")
    if np.ptp(az) > 0.0:
        ax.contour(x, y, az.T, colors="white", linewidths=0.7, levels=14, alpha=0.8)
    stride_x, stride_y = max(1, len(x) // 18), max(1, len(y) // 18)
    ax.quiver(
        x[::stride_x],
        y[::stride_y],
        vx[::stride_x, ::stride_y].T,
        vy[::stride_x, ::stride_y].T,
        color="cyan",
        pivot="mid",
        scale=None,
        width=0.003,
    )
    ax.set_title(f"|B|, A_z contours, and Nernst velocity at t={t[index]:.3g} ps")
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label="|B| [norm.]")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_artifacts(ds: xr.Dataset, td: str, *, n_panels: int = 9) -> None:
    """Write compact binaries, standard facets, and reconnection plots."""

    binary_dir = os.path.join(td, "binary")
    moments_dir = os.path.join(td, "plots", "moments")
    reconnection_dir = os.path.join(td, "plots", "reconnection")
    for directory in (binary_dir, moments_dir, reconnection_dir):
        os.makedirs(directory, exist_ok=True)
    _write_binary(ds, binary_dir)
    _plot_regular_moments(ds, moments_dir, n_panels)

    for name in ("az", "xpoint_ez"):
        if name == "az":
            save_xy_facet(
                ds.az,
                ds,
                os.path.join(reconnection_dir, "xy_facet_vector_potential.png"),
                n_panels=n_panels,
                title="reconnection flux A_z",
            )
    for variable, component in (("b", "x"), ("b", "y"), ("e", "z"), ("current", "z"), ("v_nernst", "y")):
        save_xy_facet(
            ds[variable].sel(component=component),
            ds,
            os.path.join(reconnection_dir, f"xy_facet_{variable}_{component}.png"),
            n_panels=n_panels,
            title=f"reconnection: {variable} {component}",
        )
    for name in ("ohm_resistive", "ohm_hall", "ohm_nernst", "ohm_scalar_pressure", "ohm_tensor_pressure"):
        if name in ds:
            save_xy_facet(
                ds[name].sel(component="z"),
                ds,
                os.path.join(reconnection_dir, f"xy_facet_{name}_z.png"),
                n_panels=n_panels,
                title=f"{name.replace('_', ' ')} z",
            )
    _plot_xpoint_history(ds, os.path.join(reconnection_dir, "xpoint_history.png"))
    _plot_ohm_lineouts(ds, os.path.join(reconnection_dir, "ohm_z_lineouts_x0.png"), n_panels)
    _plot_topology(ds, os.path.join(reconnection_dir, "topology_nernst_final.png"))


def reconnection_metrics(ds: xr.Dataset) -> dict[str, float]:
    """Small scalar summary suitable for MLflow comparisons."""

    rate = np.asarray(ds.normalized_reconnection_rate)
    return {
        "vfp2d_peak_abs_b": float(np.max(np.abs(ds.b))),
        "vfp2d_peak_temperature": float(np.max(ds.temperature)),
        "vfp2d_peak_abs_reconnection_rate": float(np.max(np.abs(rate))),
        "vfp2d_final_reconnection_rate": float(rate[-1]),
        "vfp2d_final_reconnected_flux": float(ds.reconnected_flux[-1]),
        "vfp2d_final_current_sheet_rms_width": float(ds.current_sheet_rms_width[-1]),
    }
