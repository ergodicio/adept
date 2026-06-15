"""Tests for the OSIRIS plotting additions.

Synthesizes tiny OSIRIS-shaped runs (no real solver needed) to exercise:

  - proper-LaTeX axis/value labels (``_tex`` wrapping)
  - the zoomed (k, ω) dispersion view with the light line
  - combined J_x/J_y/J_z (j1/j2/j3) current figures
  - per-species density + temperature profiles
  - the left/right-going transverse E-field decomposition (incl. correctness)
  - that ``save_canned_plots`` emits all of the above
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import numpy as np

from adept.osiris import io as oio
from adept.osiris import plots as oplt


def _write_dump(path: Path, name: str, data: np.ndarray, t: float, it: int, axes) -> None:
    """Write one OSIRIS-style HDF5 grid/phasespace dump.

    ``axes`` is a list of ``(name, long_name, units, min, max)`` in OSIRIS
    (Fortran) order; the last entry is the fastest-varying numpy axis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["TIME"] = np.array([t])
        f.attrs["ITER"] = np.array([it], dtype="int32")
        f.attrs["NAME"] = np.array([name.encode()], dtype="S256")
        f.attrs["LABEL"] = np.array([name.encode()], dtype="S256")
        f.attrs["UNITS"] = np.array([b"a.u."], dtype="S256")
        f.attrs["TIME UNITS"] = np.array([rb"1 / \omega_p"], dtype="S256")
        ax = f.create_group("AXIS")
        for i, (an, ln, un, lo, hi) in enumerate(axes, start=1):
            d = ax.create_dataset(f"AXIS{i}", data=np.array([lo, hi], dtype="float64"))
            d.attrs["NAME"] = np.array([an.encode()], dtype="S256")
            d.attrs["LONG_NAME"] = np.array([ln.encode()], dtype="S256")
            d.attrs["UNITS"] = np.array([un.encode()], dtype="S256")
        sim = f.create_group("SIMULATION")
        sim.attrs["DT"] = np.array([0.05])
        sim.attrs["NDIMS"] = np.array([1], dtype="int32")
        sim.attrs["NX"] = np.array([data.shape[-1]], dtype="int32")
        sim.attrs["XMIN"] = np.array([axes[-1][3]])
        sim.attrs["XMAX"] = np.array([axes[-1][4]])
        f.create_dataset(name, data=data.astype("float32"))


X_AX = ("x1", "x_1", r"c / \omega_p", 0.0, 10.0)
P_AX = ("p1", "p_1", r"m_e c", -2.0, 2.0)


def _make_rich_run(root: Path, n_steps: int = 6, nx: int = 16, npx: int = 12) -> Path:
    """Run with EM fields, currents, density+thermal moments, and phase space."""
    run_dir = root / "run"
    rng = np.random.default_rng(0)
    x = np.linspace(X_AX[3], X_AX[4], nx)

    for k in range(n_steps):
        it, t = k * 10, k * 0.5
        # Right-going transverse pairs: e2 = b3, e3 = -b2 (so left part ~ 0).
        wave = np.sin(2 * np.pi * (x - t) / 10.0)
        _write_dump(run_dir / "MS/FLD/e2" / f"e2-{it:06d}.h5", "e2", wave, t, it, [X_AX])
        _write_dump(run_dir / "MS/FLD/b3" / f"b3-{it:06d}.h5", "b3", wave, t, it, [X_AX])
        _write_dump(run_dir / "MS/FLD/e3" / f"e3-{it:06d}.h5", "e3", wave, t, it, [X_AX])
        _write_dump(run_dir / "MS/FLD/b2" / f"b2-{it:06d}.h5", "b2", -wave, t, it, [X_AX])
        # Currents j1/j2/j3.
        for j in ("j1", "j2", "j3"):
            _write_dump(run_dir / f"MS/FLD/{j}" / f"{j}-{it:06d}.h5", j,
                        rng.standard_normal(nx), t, it, [X_AX])
        # Density + thermal-velocity moments for a species.
        _write_dump(run_dir / "MS/DENSITY/electron/charge" / f"charge-electron-{it:06d}.h5",
                    "charge", -np.abs(rng.standard_normal(nx)) - 1.0, t, it, [X_AX])
        for u in ("uth1", "uth2", "uth3"):
            _write_dump(run_dir / f"MS/UDIST/electron/{u}" / f"{u}-electron-{it:06d}.h5",
                        u, 0.1 + 0.01 * rng.standard_normal(nx), t, it, [X_AX])
        # Phase space (p, x).
        _write_dump(run_dir / "MS/PHA/x1p1/electron" / f"x1p1-electron-{it:06d}.h5",
                    "x1p1", rng.random((npx, nx)), t, it, [X_AX, P_AX])
    return run_dir


# --- LaTeX labels ---------------------------------------------------------


def test_tex_wraps_only_tex_fragments() -> None:
    assert oplt._tex(r"\omega_p") == r"$\omega_p$"
    assert oplt._tex("x_1") == "$x_1$"
    assert oplt._tex(r"c / \omega_p") == r"$c / \omega_p$"
    assert oplt._tex("charge") == "charge"  # plain word, left alone
    assert oplt._tex("a.u.") == "a.u."
    assert oplt._tex(r"$E_1$") == r"$E_1$"  # idempotent


def test_axis_label_is_math_mode(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path, n_steps=3)
    da = oio.load_series(run_dir / "MS/FLD/e2")
    # Time-axis units wrapped in $...$ instead of rendered literally.
    assert oplt._axis_label(da, "t") == r"$t$  [$1 / \omega_p$]"
    # Spatial axis: both long-name and units in math mode.
    assert oplt._axis_label(da, "x1") == r"$x_1$  [$c / \omega_p$]"


# --- zoomed omega-k -------------------------------------------------------


def test_omega_k_zoom_window_clamps_to_nyquist(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path)
    ser = oio.load_series(run_dir / "MS/FLD/e2")
    big = oplt._omega_k_zoom_window(ser, requested=1e6)  # huge -> clamp to Nyquist
    small = oplt._omega_k_zoom_window(ser, requested=2.0)
    assert small == 2.0
    assert 0 < big < 1e6


def test_omega_k_light_line_drawn(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path)
    ser = oio.load_series(run_dir / "MS/FLD/e2")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    oplt.plot_omega_k(ser, ax=ax, show_em=False, show_light_line=True, k_max=4, omega_max=4)
    labels = [ln.get_label() for ln in ax.lines]
    assert any("light line" in str(lbl) for lbl in labels)
    plt.close("all")


# --- currents -------------------------------------------------------------


def test_current_components_loaded(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path)
    comps = oplt._current_components(run_dir)
    assert set(comps) == {"j1", "j2", "j3"}
    assert oplt.plot_currents_spacetime(run_dir) is not None
    assert oplt.plot_currents_lineouts(run_dir) is not None


def test_currents_absent_returns_none(tmp_path: Path) -> None:
    (tmp_path / "run" / "MS").mkdir(parents=True)
    assert oplt.plot_currents_spacetime(tmp_path / "run") is None


# --- density / temperature profiles ---------------------------------------


def test_temperature_series_from_uth(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path)
    entries = oplt._species_diags(run_dir)["electron"]
    temp = oplt._temperature_series(entries)
    assert temp is not None
    # T = uth1^2 + uth2^2 + uth3^2, so ~ 3 * 0.1^2 = 0.03, and positive.
    assert float(temp.isel(t=-1).mean()) > 0
    assert temp.attrs["long_name"].startswith("T =")
    dens = oplt._density_series(entries)
    assert dens is not None and dens.name == "charge"


def test_temperature_series_absent_returns_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_dump(run_dir / "MS/DENSITY/ion/charge" / "charge-ion-000000.h5",
                "charge", np.ones(8), 0.0, 0, [X_AX])
    entries = oplt._species_diags(run_dir)["ion"]
    assert oplt._temperature_series(entries) is None


# --- left/right-going E decomposition -------------------------------------


def test_efield_decomposition_isolates_right_going(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path)
    parts = oplt.efield_lr_components(run_dir)
    assert set(parts) == {"e2", "e3"}
    e2 = oio.load_series(run_dir / "MS/FLD/e2")
    # e2 == b3 is a pure right-going wave: right == e2, left == 0.
    np.testing.assert_allclose(parts["e2"]["right"].values, e2.values, atol=1e-6)
    np.testing.assert_allclose(parts["e2"]["left"].values, 0.0, atol=1e-6)
    # e3 == -b2 is also pure right-going.
    e3 = oio.load_series(run_dir / "MS/FLD/e3")
    np.testing.assert_allclose(parts["e3"]["right"].values, e3.values, atol=1e-6)
    np.testing.assert_allclose(parts["e3"]["left"].values, 0.0, atol=1e-6)


def test_field_decomposition_figures(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path)
    figs = oplt.plot_field_lr_decomposition(run_dir)
    assert set(figs) == {"e2", "e3"}
    # Each figure is now a single row of two spacetime panels (no omega-k row).
    for fig in figs.values():
        assert len(fig.axes) == 4  # 2 heatmaps + their 2 colorbars


# --- end-to-end driver ----------------------------------------------------


def test_save_canned_plots_emits_new_views(tmp_path: Path) -> None:
    run_dir = _make_rich_run(tmp_path)
    written = oplt.save_canned_plots(run_dir, tmp_path / "plots", v_th=0.1)
    expected = {
        "omega_k/e2",
        "currents/spacetime",
        "currents/lineouts",
        "profiles/electron/density",
        "profiles/electron/temperature",
        "field_decomp/e2",
        "field_decomp/e3",
    }
    assert expected <= set(written)
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0

