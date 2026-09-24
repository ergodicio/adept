"""The parity harness (adept/_lpse2d/parity): reference-run location and the comparison
metrics that reproduce the 2026-09-14 ``log_lpse_reference.py`` baseline on MLflow."""

import os
from pathlib import Path

import numpy as np
import pytest

from adept._lpse2d.parity import (
    DEFAULT_WINDOWS,
    compare,
    deck_path,
    energy_floor_crossings,
    growth_rate,
    lpse_root,
    parse_windows,
    read_flux,
    reference_run_dir,
)


def test_parse_windows_and_growth_rate():
    assert parse_windows("0.3-0.9,0.9-1.5") == ((0.3, 0.9), (0.9, 1.5))
    t = np.linspace(0.0, 2.0, 201)
    e = 1e-3 * np.exp(4.0 * t)
    assert growth_rate(t, e, 0.5, 1.5) == pytest.approx(4.0, rel=1e-10)
    # fewer than four positive samples in the window -> nan, as the reference script
    assert np.isnan(growth_rate(t, e, 0.5, 0.52))
    assert np.isnan(growth_rate(t, np.zeros_like(t), 0.5, 1.5))


def _synthetic(gamma_adept=3.0, gamma_lpse=2.5, gamma_raman=4.0):
    t = np.linspace(0.0, 2.0, 401)
    tl = np.linspace(0.0, 2.0, 201)
    series = {
        "t (ps)": t,
        "epw_energy": 1e-4 * np.exp(gamma_adept * t),
        "e1_sq": 1e-6 * np.exp(gamma_raman * t),
        "iaw_density_abs_max": 1e-3 * (1.0 + t),
    }
    lpse = {
        "time": tl,
        "EPW_energy": 1e-4 * np.exp(gamma_lpse * tl),
        "E1_energy": 1e-6 * np.exp(gamma_raman * tl),
        "Nelf_max": 2e-3 * (1.0 + tl),
    }
    return series, lpse


def test_compare_keys_and_values_match_the_reference_script():
    series, lpse = _synthetic()
    m = compare(series, lpse, "0.5-1.0,1.0-1.5")
    assert set(m) == {
        "epw_energy_growth_adept_0.5_1ps",
        "epw_energy_growth_lpse_0.5_1ps",
        "epw_energy_growth_ratio_0.5_1ps",
        "raman_energy_growth_adept_0.5_1ps",
        "raman_energy_growth_lpse_0.5_1ps",
        "raman_energy_growth_ratio_0.5_1ps",
        "epw_energy_growth_adept_1_1.5ps",
        "epw_energy_growth_lpse_1_1.5ps",
        "epw_energy_growth_ratio_1_1.5ps",
        "raman_energy_growth_adept_1_1.5ps",
        "raman_energy_growth_lpse_1_1.5ps",
        "raman_energy_growth_ratio_1_1.5ps",
        "epw_half_max_time_adept_ps",
        "epw_half_max_time_lpse_ps",
        "iaw_max_adept",
        "iaw_max_lpse",
    }
    assert m["epw_energy_growth_ratio_0.5_1ps"] == pytest.approx(3.0 / 2.5, rel=1e-9)
    assert m["raman_energy_growth_ratio_1_1.5ps"] == pytest.approx(1.0, rel=1e-9)
    # half max of exp(g t) on [0, 2]: t = 2 - ln 2 / g
    assert m["epw_half_max_time_adept_ps"] == pytest.approx(2.0 - np.log(2.0) / 3.0, abs=0.01)
    assert m["epw_half_max_time_lpse_ps"] == pytest.approx(2.0 - np.log(2.0) / 2.5, abs=0.01)
    assert m["iaw_max_adept"] == pytest.approx(3e-3) and m["iaw_max_lpse"] == pytest.approx(6e-3)
    # windows as tuples give the same result
    assert compare(series, lpse, ((0.5, 1.0), (1.0, 1.5))) == m


def test_compare_without_raman_or_iaw_columns():
    series, lpse = _synthetic()
    del series["e1_sq"], lpse["Nelf_max"]
    m = compare(series, lpse, ((0.5, 1.0),))
    assert not any(k.startswith(("raman", "iaw")) for k in m)
    crossings = energy_floor_crossings(series, lpse, floor_window=(0.0, 0.2), factors=(10,))
    # floor = median of exp(g t) on (0, 0.2) ~ exp(0.1 g); 10x floor at t = 0.1 + ln 10 / g
    assert crossings["epw_10x_floor_time_adept_ps"] == pytest.approx(0.1 + np.log(10.0) / 3.0, abs=0.02)


def test_default_windows_cover_the_baseline_decks():
    assert set(DEFAULT_WINDOWS) == {
        "hom_srs_020", "hom_tpd_023", "test_006", "test_025", "test_029", "test_010",
        "test_032", "test_013", "test_022", "test_036", "test_037",
    }  # fmt: skip
    for windows in DEFAULT_WINDOWS.values():
        assert all(lo < hi for lo, hi in windows)


def test_lpse_root_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LPSE_ROOT", str(tmp_path))
    assert lpse_root() == tmp_path
    monkeypatch.setenv("LPSE_ROOT", str(tmp_path / "missing"))
    assert lpse_root() is None
    assert deck_path("test_006") is None
    monkeypatch.delenv("LPSE_ROOT")
    root = lpse_root()
    assert root is None or (root / "examples").is_dir() or (root / "runs").is_dir()


def test_deck_path_falls_back_to_the_reference_run_copy(tmp_path):
    root = tmp_path / "lpse"
    (root / "examples" / "testRuns" / "test_006").mkdir(parents=True)
    (root / "examples" / "testRuns" / "test_006" / "lpse.parms").write_text("x")
    (root / "runs" / "hom_srs_020").mkdir(parents=True)
    (root / "runs" / "hom_srs_020" / "lpse.parms").write_text("x")
    assert deck_path("test_006", root) == root / "examples" / "testRuns" / "test_006" / "lpse.parms"
    assert deck_path("hom_srs_020", root) == root / "runs" / "hom_srs_020" / "lpse.parms"
    assert deck_path("nope", root) is None


def test_reference_run_dir_without_download(monkeypatch, tmp_path):
    monkeypatch.setenv("LPSE_ROOT", str(tmp_path))
    monkeypatch.setenv("LPSE_REFERENCE_CACHE", str(tmp_path / "cache"))
    assert reference_run_dir("test_025", download=False) is None
    run = tmp_path / "runs" / "test_025" / "data"
    run.mkdir(parents=True)
    (run / "lpse.metrics").write_text("# (1) time\n0.0\n")
    assert reference_run_dir("test_025", download=False) == tmp_path / "runs" / "test_025"


RUN_036 = reference_run_dir("test_036", download=False)


@pytest.mark.skipif(
    RUN_036 is None or not (RUN_036 / "data" / "lpse.flux").is_file(), reason="no test_036 reference run"
)
def test_read_flux_test_036():
    t, flux = read_flux(RUN_036 / "data" / "lpse.flux")
    assert flux.shape[1:] == (7, 6) and t[0] == pytest.approx(0.12)
    assert np.all(flux[:, :, 4:] == 0.0)  # no z faces in 2-D
    assert np.all(flux[:, :4, :4] > 0.0)


RUN_006 = reference_run_dir("test_006", download=False)


@pytest.mark.skipif(RUN_006 is None, reason="no test_006 reference run")
def test_compare_on_the_test_006_reference_against_itself():
    """Feeding LPSE's own energies back as the adept series gives ratios of exactly 1."""
    from adept._lpse2d.lpse_deck import read_metrics

    d = read_metrics(RUN_006 / "data" / "lpse.metrics")
    series = {"t (ps)": d["time"], "epw_energy": d["EPW_energy"], "e1_sq": d["E1_energy"]}
    m = compare(series, d, DEFAULT_WINDOWS["test_006"])
    for k, v in m.items():
        if "ratio" in k:
            assert v == pytest.approx(1.0, rel=1e-12), k
    assert m["epw_half_max_time_adept_ps"] == m["epw_half_max_time_lpse_ps"]
    # the stored baseline value for this deck (MLflow lpse-parity bf56ee6c, 2026-09-14)
    assert m["epw_energy_growth_lpse_2.5_3.5ps"] == pytest.approx(5.707053668690972, rel=1e-9)


@pytest.mark.skipif(not os.environ.get("MLFLOW_TRACKING_URI"), reason="no MLflow tracking environment")
def test_verify_reproduces_the_stored_baseline_metrics(tmp_path):
    """``verify`` on the test_006 cross-check run (bf56ee6c…): the ratio metrics logged on
    2026-09-14 by ``log_lpse_reference.py`` are reproduced from the run's own artifacts."""
    from adept._lpse2d.parity.__main__ import main

    rc = main(["verify", "bf56ee6c3072421d80bdcff0080a5c84", "--dest", str(tmp_path), "--rtol", "1e-6"])
    assert rc == 0
    assert (Path(tmp_path) / "binary" / "series.xr").is_file()


def test_run_deck_refuses_float32():
    """The comparison baseline is float64; a float32 run would be silently different."""
    import jax

    from adept._lpse2d.parity import run_deck

    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            run_deck("unused.parms")
    finally:
        jax.config.update("jax_enable_x64", True)


def test_batch_jobs_merge_defaults_and_build_run_commands(tmp_path):
    """A batch file's ``defaults`` merge under each job (``overrides`` / ``tags``
    recursively), ``job_command`` is the ``parity run`` argument vector, and the driver runs the
    jobs on N workers, writes one log per job and a summary it can resume from."""
    import json

    from adept._lpse2d.parity import job_command, load_jobs, run_batch
    from adept._lpse2d.parity.batch import parse_log, slug

    spec = tmp_path / "jobs.yaml"
    spec.write_text(
        "defaults:\n  experiment: lpse-parity\n  tags: {plan2_item: N.1}\n"
        "  overrides: {terms: {epw: {source: {noise_seed: 1}}}}\n"
        "jobs:\n"
        "  - {deck: test_010, run: test_010/n1-seed/2, overrides: {terms: {epw: {source: {noise_seed: 2}}}},"
        " tags: {variable: seed}}\n"
        "  - {deck: hom_tpd_023, run: hom_tpd_023/n1-angle/16.7, windows: 0.5-0.8, no_log: true}\n"
    )
    jobs = load_jobs(spec)
    assert [j["run"] for j in jobs] == ["test_010/n1-seed/2", "hom_tpd_023/n1-angle/16.7"]
    assert jobs[0]["overrides"] == {"terms": {"epw": {"source": {"noise_seed": 2}}}}
    assert jobs[0]["tags"] == {"plan2_item": "N.1", "variable": "seed"}
    assert jobs[1]["overrides"] == {"terms": {"epw": {"source": {"noise_seed": 1}}}}
    cmd = job_command(jobs[0], tmp_path / "out", python="py")
    assert cmd[:5] == ["py", "-m", "adept._lpse2d.parity", "run", "test_010"]
    assert cmd[cmd.index("--run") + 1] == "test_010/n1-seed/2"
    assert json.loads(cmd[cmd.index("--overrides") + 1]) == jobs[0]["overrides"]
    assert "--tag" in cmd and "plan2_item=N.1" in cmd and "variable=seed" in cmd
    assert cmd[-1] == str(tmp_path / "out" / slug("test_010/n1-seed/2"))
    cmd1 = job_command(jobs[1], None, python="py")
    assert "--windows" in cmd1 and "0.5-0.8" in cmd1 and "--no-log" in cmd1 and "--out" not in cmd1

    # the shipped matrices load
    root = Path(__file__).resolve().parents[2] / "adept" / "_lpse2d" / "parity" / "batches"
    for name in ("relog.yaml", "n1_test_010.yaml", "n2_hpe.yaml", "f3_spol.yaml"):
        assert load_jobs(root / name)

    # run the driver with a stand-in interpreter: each job prints a run id and a metric line
    fake = tmp_path / "fake.sh"
    fake.write_text(
        '#!/bin/sh\nname=\'\'\nwhile [ $# -gt 0 ]; do [ "$1" = --run ] && name="$2"; shift; done\n'
        "echo 'mlflow run 0123456789abcdef0123456789abcdef'\necho '  epw_energy_growth_ratio_0.3_0.9ps: 0.75'\n"
        'case "$name" in *fail*) exit 1;; esac\n'
    )
    fake.chmod(0o755)
    jobs = [{"deck": "d", "run": "a/one"}, {"deck": "d", "run": "a/two"}, {"deck": "d", "run": "a/fail"}]
    summary = run_batch(jobs, tmp_path / "logs", workers=2, python=str(fake), env={})
    assert {k: v["rc"] for k, v in summary.items()} == {"a/one": 0, "a/two": 0, "a/fail": 1}
    assert summary["a/one"]["run_id"] == "0123456789abcdef" * 2
    assert summary["a/one"]["metrics"] == {"epw_energy_growth_ratio_0.3_0.9ps": 0.75}
    assert parse_log(tmp_path / "logs" / "a--two.log")["metrics"]["epw_energy_growth_ratio_0.3_0.9ps"] == 0.75
    assert json.loads((tmp_path / "logs" / "summary.json").read_text())["a/fail"]["rc"] == 1
    # --skip-done: finished jobs are not re-run, the failed one is
    (tmp_path / "logs" / "a--one.log").unlink()
    summary = run_batch(jobs, tmp_path / "logs", workers=1, python=str(fake), env={}, skip_done=True)
    assert not (tmp_path / "logs" / "a--one.log").exists() and summary["a/fail"]["rc"] == 1
