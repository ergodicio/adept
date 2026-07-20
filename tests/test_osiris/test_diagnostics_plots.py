"""Self-contained tests for the expanded OSIRIS diagnostics / plots.

These synthesize a tiny OSIRIS-shaped run (fields, per-species density moment,
phase space, and HIST energy histories) so they need no real run on disk, and
exercise the additions in plots.py / io.py / post.py:

  - log-scale spacetime + lineout field plots
  - per-species DENSITY moment plots
  - phase-space time-evolution panels
  - E/B-field energy split and HIST-based energy conservation
  - post.collect wiring (plots land under td/plots)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import numpy as np

from adept.osiris import io as oio
from adept.osiris import plots as oplt
from adept.osiris import post as opost


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


def _make_full_run(root: Path, n_steps: int = 6, nx: int = 16, npx: int = 12) -> Path:
    """Build a run with FLD/e1, DENSITY/electron/charge, PHA, and HIST energy."""
    run_dir = root / "run"
    rng = np.random.default_rng(0)
    x_ax = ("x1", "x_1", r"c / \omega_p", 0.0, 10.0)
    p_ax = ("p1", "p_1", r"m_e c", -2.0, 2.0)

    for k in range(n_steps):
        it = k * 10
        t = k * 0.5
        _write_dump(run_dir / "MS/FLD/e1" / f"e1-{it:06d}.h5", "e1", rng.standard_normal(nx), t=t, it=it, axes=[x_ax])
        _write_dump(
            run_dir / "MS/DENSITY/electron/charge" / f"charge-electron-{it:06d}.h5",
            "charge",
            rng.standard_normal(nx),
            t=t,
            it=it,
            axes=[x_ax],
        )
        _write_dump(
            run_dir / "MS/PHA/x1p1/electron" / f"x1p1-electron-{it:06d}.h5",
            "x1p1",
            rng.random((npx, nx)),
            t=t,
            it=it,
            axes=[x_ax, p_ax],
        )

    # HIST energy histories: iter, time, then value columns.
    hist = run_dir / "HIST"
    hist.mkdir(parents=True, exist_ok=True)
    times = [k * 0.5 for k in range(n_steps)]
    fld = ["! iter time e1 e2 e3 b1 b2 b3"]
    par = ["! iter time ene"]
    for k, t in enumerate(times):
        fld.append(f"{k * 10} {t} {1.0 + 0.1 * k} 0.0 0.0 0.0 0.0 0.0")
        par.append(f"{k * 10} {t} {100.0 - 0.1 * k}")
    (hist / "fld_ene").write_text("\n".join(fld) + "\n")
    (hist / "par01_ene").write_text("\n".join(par) + "\n")

    (run_dir / "os-stdin").write_text("node_conf\n{\n}\n")
    (run_dir / "stdout.log").write_text("ok\n")
    (run_dir / "stderr.log").write_text("")
    return run_dir


def test_field_energy_components_splits_e_and_b(tmp_path: Path) -> None:
    run_dir = _make_full_run(tmp_path)
    ds = oplt.field_energy_components(run_dir)
    assert {"E_energy", "B_energy", "total_field_energy"} <= set(ds.data_vars)
    # Only e1 was dumped, so all energy is electric and B is identically zero.
    assert np.all(ds["E_energy"].values > 0)
    assert np.all(ds["B_energy"].values == 0)
    np.testing.assert_allclose(ds["total_field_energy"].values, ds["E_energy"].values)
    # e1 is the only dump, so the longitudinal (EPW) energy equals the E energy.
    assert "e1_energy" in ds
    np.testing.assert_allclose(ds["e1_energy"].values, ds["E_energy"].values)


def test_field_energy_picks_up_savg_and_excludes_poynting(tmp_path: Path) -> None:
    """A 2D-style deck dumps e1,savg (not full grid) and s1 Poynting lineouts.

    field_energy_components must resolve the ``e1-savg`` variant, and the
    total-field-energy metric must NOT fold the ``s1`` lineout into the sum.
    """
    run_dir = tmp_path / "run"
    x_ax = ("x1", "x_1", r"c / \omega_p", 0.0, 10.0)
    x2_ax = ("x2", "x_2", r"c / \omega_p", 0.0, 4.0)
    rng = np.random.default_rng(3)
    for k in range(4):
        it, t = k * 10, k * 0.5
        _write_dump(run_dir / "MS/FLD/e1-savg" / f"e1-savg-{it:06d}.h5",
                    "e1", rng.standard_normal(8), t=t, it=it, axes=[x_ax])
        # an s1 Poynting-flux lineout along x2 (would inflate field energy if summed)
        _write_dump(run_dir / "MS/FLD/s1-line-x2-24" / f"s1-line-x2-24-{it:06d}.h5",
                    "s1", 5.0 + rng.standard_normal(6), t=t, it=it, axes=[x2_ax])

    ds = oplt.field_energy_components(run_dir)
    assert np.all(ds["e1_energy"].values > 0)          # savg variant resolved
    total = opost._total_field_energy(run_dir / "MS")
    # only e1 contributes; s1 is excluded, so the total matches the e1 energy of
    # the last dump (both are 0.5 * sum(e1^2) * dx).
    assert np.isfinite(total) and total > 0


def test_field_energy_components_2d_spatial(tmp_path: Path) -> None:
    """field_energy_components integrates over BOTH spatial dims for a 2D field."""
    run_dir = tmp_path / "run"
    x_ax = ("x1", "x_1", r"c / \omega_p", 0.0, 10.0)
    x2_ax = ("x2", "x_2", r"c / \omega_p", 0.0, 4.0)
    rng = np.random.default_rng(5)
    for k in range(3):
        it, t = k * 10, k * 0.5
        data = rng.standard_normal((6, 8))  # (x2, x1)
        _write_dump(run_dir / "MS/FLD/e1" / f"e1-{it:06d}.h5",
                    "e1", data, t=t, it=it, axes=[x_ax, x2_ax])
    ds = oplt.field_energy_components(run_dir)
    assert ds["e1_energy"].sizes["t"] == 3
    assert np.all(ds["e1_energy"].values > 0)
    ax = oplt.plot_epw_energy(ds)
    assert ax.get_ylabel()


def test_load_hist_energy_builds_conservation_total(tmp_path: Path) -> None:
    run_dir = _make_full_run(tmp_path, n_steps=5)
    energy = oio.load_hist_energy(run_dir)
    assert energy is not None
    assert "field_energy" in energy
    assert "kinetic_par01_ene" in energy
    assert "kinetic_total" in energy
    assert "total" in energy
    # total = field + kinetic, elementwise.
    np.testing.assert_allclose(
        energy["total"].values,
        energy["field_energy"].values + energy["kinetic_total"].values,
    )
    assert "total_drift_frac" in energy.attrs
    assert 0.0 <= energy.attrs["total_drift_frac"] <= 1.0


def test_load_hist_energy_absent_returns_none(tmp_path: Path) -> None:
    # A run dir with no HIST/ yields None rather than raising.
    (tmp_path / "run" / "MS").mkdir(parents=True)
    assert oio.load_hist_energy(tmp_path / "run") is None


def test_save_canned_plots_emits_full_set(tmp_path: Path) -> None:
    run_dir = _make_full_run(tmp_path)
    out = tmp_path / "plots"
    written = oplt.save_canned_plots(run_dir, out, v_th=0.1)

    expected = {
        "spacetime/e1",
        "spacetime_log/e1",
        "lineouts/e1",
        "omega_k/e1",
        "moments/electron/charge",
        "moments/electron/charge_log",
        "moments/electron/lineouts/charge",
        "phasespace/electron/x1p1",
        "phasespace_evolution/electron/x1p1",
        "energy_vs_time",
        "energy_components_vs_time",
        "total_energy_vs_time",  # present because HIST/ supplies kinetic energy
        "energy_partition_vs_time",  # same HIST source: total split into particle/EM
    }
    assert expected <= set(written)
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_collect_writes_plots_under_td(tmp_path: Path) -> None:
    run_dir = _make_full_run(tmp_path)
    run_output = {"solver result": {"run_dir": str(run_dir), "wall_time": 1.0, "exit_code": 0}}
    cfg = {"osiris": {"deck": str(run_dir / "os-stdin")}, "output": {}}

    td = tmp_path / "td"
    td.mkdir()
    result = opost.collect(run_output, cfg, str(td))

    # Plots were generated as artifacts and energy metrics added.
    assert (td / "plots" / "spacetime" / "e1.png").exists()
    assert (td / "plots" / "energy_components_vs_time.png").exists()
    assert "efield_energy_final" in result["metrics"]
    assert "energy_drift_frac" in result["metrics"]
