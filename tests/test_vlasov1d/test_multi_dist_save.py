"""Test multiple distribution function saves per species.

Species distribution saves use a nested YAML structure::

    save:
      electron:
        main:
          t: {nt: 11}
        full:
          t: {nt: 5}
          x: {xmin: 0.0, xmax: 20.94, nx: 32}
          v: {vmin: -6.4, vmax: 6.4, nv: 512}

Each ``<species>/<label>`` pair becomes a flat key in ``result.ys`` /
``result.ts``, e.g. ``"electron.main"``, ``"electron.full"``.
"""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept import ergoExo


@pytest.fixture
def base_config():
    with open("tests/test_vlasov1d/configs/resonance.yaml") as f:
        cfg = yaml.safe_load(f)
    # resonance.yaml already uses the nested format with save.electron.main
    return cfg


def test_single_label_basic(base_config):
    """save.electron.main produces result key 'electron/main'."""
    exo = ergoExo()
    exo.setup(base_config)
    result, datasets, _ = exo(None)

    solver_result = result["solver result"]
    assert "electron.main" in solver_result.ys
    assert "electron.main" in solver_result.ts

    # Shape: (nt, nx_save, nv_save) matching resonance.yaml values
    f = solver_result.ys["electron.main"]
    assert f.shape == (9, 24, 384)

    # Dataset variable named after the internal key
    f_result = datasets["dists"]
    assert "electron.main" in f_result
    assert "electron.main" in f_result["electron.main"].data_vars


def test_two_labels_same_species(base_config):
    """Two labels under one species produce independent save keys."""
    cfg = deepcopy(base_config)
    tmax = cfg["grid"]["tmax"]

    # Replace single 'main' with two named saves
    cfg["save"]["electron"] = {
        "full": {
            "t": {"tmin": 0.0, "tmax": tmax, "nt": 3},
            "x": {"xmin": 0.0, "xmax": cfg["grid"]["xmax"], "nx": 16},
            "v": {"vmin": -6.4, "vmax": 6.4, "nv": 128},
        },
        "monitor": {
            "t": {"tmin": 0.0, "tmax": tmax, "nt": 9},
            "x": {"xmin": 0.0, "xmax": cfg["grid"]["xmax"], "nx": 8},
            "v": {"vmin": -3.0, "vmax": 3.0, "nv": 64},
        },
    }

    exo = ergoExo()
    exo.setup(cfg)
    result, datasets, _ = exo(None)
    solver_result = result["solver result"]

    assert "electron.full" in solver_result.ys
    assert "electron.monitor" in solver_result.ys

    assert solver_result.ys["electron.full"].shape == (3, 16, 128)
    assert solver_result.ys["electron.monitor"].shape == (9, 8, 64)

    ts_full = np.asarray(solver_result.ts["electron.full"])
    ts_monitor = np.asarray(solver_result.ts["electron.monitor"])
    np.testing.assert_allclose([ts_full[0], ts_full[-1]], [0.0, tmax], atol=1e-6)
    np.testing.assert_allclose([ts_monitor[0], ts_monitor[-1]], [0.0, tmax], atol=1e-6)


def test_dist_save_files_and_datasets(base_config):
    """post_process writes dist-<key>.nc and returns a dict keyed by label."""
    cfg = deepcopy(base_config)
    tmax = cfg["grid"]["tmax"]

    cfg["save"]["electron"] = {
        "full": {
            "t": {"tmin": 0.0, "tmax": tmax, "nt": 3},
            "x": {"xmin": 0.0, "xmax": cfg["grid"]["xmax"], "nx": 8},
            "v": {"vmin": -6.4, "vmax": 6.4, "nv": 64},
        },
        "monitor": {
            "t": {"tmin": 0.0, "tmax": tmax, "nt": 5},
            "x": {"xmin": 0.0, "xmax": cfg["grid"]["xmax"], "nx": 4},
            "v": {"vmin": -3.0, "vmax": 3.0, "nv": 32},
        },
    }

    exo = ergoExo()
    exo.setup(cfg)
    _, datasets, _ = exo(None)

    f_result = datasets["dists"]
    assert set(f_result.keys()) == {"electron.full", "electron.monitor"}

    # Each dataset carries the internal key as its variable name
    assert "electron.full" in f_result["electron.full"].data_vars
    assert "electron.monitor" in f_result["electron.monitor"].data_vars

    # Velocity coordinate named after the species, not the label
    assert "v_electron" in f_result["electron.full"]["electron.full"].dims
    assert "v_electron" in f_result["electron.monitor"]["electron.monitor"].dims


def test_full_resolution_save(base_config):
    """A label with only 't' (no x/v) returns the full simulation grid."""
    cfg = deepcopy(base_config)
    tmax = cfg["grid"]["tmax"]

    cfg["save"]["electron"] = {
        "raw": {"t": {"tmin": 0.0, "tmax": tmax, "nt": 2}},
    }

    exo = ergoExo()
    exo.setup(cfg)
    result, _, _ = exo(None)

    raw = result["solver result"].ys["electron.raw"]
    assert raw.shape == (2, cfg["grid"]["nx"], cfg["grid"]["nv"])
