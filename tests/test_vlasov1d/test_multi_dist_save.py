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
from jax import numpy as jnp

from adept import ergoExo
from adept._vlasov1d.storage import _add_dim_axes, get_dist_save_func


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

    # resonance.yaml saves v over the solver's full [-vmax, vmax], which is exactly
    # the case that used to leave the first and last v column NaN.
    saved = f_result["electron.main"]["electron.main"].values
    assert np.all(np.isfinite(saved)), f"{int((~np.isfinite(saved)).sum())} non-finite cells in the save"


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


def _solver_axis(lo: float, hi: float, n: int) -> jnp.ndarray:
    """Cell-centred axis, matching how the solver builds its own x and v grids."""
    d = (hi - lo) / n
    return jnp.array(np.linspace(lo + d / 2.0, hi - d / 2.0, n))


def _edge_save_setup(grid_nv: int, save_nv: int = 384):
    """An x-v save spanning the solver's full domain, plus the solver axes it samples."""
    vmax, xmax, grid_nx, save_nx = 6.4, 20.94, 32, 24

    axes = {
        "x": _solver_axis(0.0, xmax, grid_nx),
        "v": _solver_axis(-vmax, vmax, grid_nv),
        "kx": jnp.fft.rfftfreq(grid_nx, d=xmax / grid_nx) * 2.0 * np.pi,
    }
    save_cfg = {
        "t": {"tmin": 0.0, "tmax": 1.0, "nt": 2},
        "x": {"xmin": 0.0, "xmax": xmax, "nx": save_nx},
        "v": {"vmin": -vmax, "vmax": vmax, "nv": save_nv},
    }
    _add_dim_axes(save_cfg)
    return axes, save_cfg, (save_nx, save_nv), vmax


@pytest.mark.parametrize("grid_nv", [384, 512, 2048])
def test_full_domain_v_save_has_no_nan_edges(grid_nv):
    """A save spanning [-vmax, vmax] stays finite regardless of solver resolution.

    The save v axis is endpoint-inclusive while the solver's is cell-centred, so
    the save always queries half a solver cell beyond the outermost cell centre.
    That overhang does not shrink with resolution, so neither does the bug it used
    to cause -- hence the sweep over grid_nv both below and above the save's nv.
    """
    axes, save_cfg, out_shape, _ = _edge_save_setup(grid_nv)

    # Precondition: the save really is asking for points outside the solver axis.
    assert save_cfg["v"]["ax"][0] < float(axes["v"][0])
    assert save_cfg["v"]["ax"][-1] > float(axes["v"][-1])

    func = get_dist_save_func(axes=axes, dist_save_config=save_cfg, dist_key="electron")
    f = jnp.exp(-(axes["v"][None, :] ** 2) / 2.0) * jnp.ones((len(axes["x"]), 1))
    out = np.asarray(func(0.0, {"electron": f}, None))

    assert out.shape == out_shape
    assert np.all(np.isfinite(out)), f"{int((~np.isfinite(out)).sum())} non-finite cells"


def test_edge_v_columns_take_nearest_solver_value():
    """Edge cells are clamped to the outermost solver cell, not extrapolated.

    The reported v coordinate stays exactly vmin/vmax, so consumers that assume
    the save axis spans the configured range are unaffected.
    """
    grid_nv = 512
    axes, save_cfg, _, vmax = _edge_save_setup(grid_nv)

    # A separable f, linear in x so that linear interpolation in x is exact.
    g = 1.0 + 0.5 * axes["x"] / float(axes["x"][-1])
    h = jnp.exp(-(axes["v"] ** 2) / 2.0)
    f = g[:, None] * h[None, :]

    func = get_dist_save_func(axes=axes, dist_save_config=save_cfg, dist_key="electron")
    out = np.asarray(func(0.0, {"electron": f}, None))

    g_at_save_x = np.interp(save_cfg["x"]["ax"], np.asarray(axes["x"]), np.asarray(g))
    np.testing.assert_allclose(out[:, 0], g_at_save_x * float(h[0]), rtol=1e-6)
    np.testing.assert_allclose(out[:, -1], g_at_save_x * float(h[-1]), rtol=1e-6)

    # The coordinate the file will carry is untouched by the clamp.
    np.testing.assert_allclose([save_cfg["v"]["ax"][0], save_cfg["v"]["ax"][-1]], [-vmax, vmax])


def test_save_x_finer_than_solver_grid_has_no_nan_edges():
    """A save x axis finer than the solver's also overhangs, and must stay finite."""
    xmax, grid_nx, save_nx, vmax, grid_nv = 500.0, 64, 256, 15.0, 2048

    axes = {
        "x": _solver_axis(0.0, xmax, grid_nx),
        "v": _solver_axis(-vmax, vmax, grid_nv),
        "kx": jnp.fft.rfftfreq(grid_nx, d=xmax / grid_nx) * 2.0 * np.pi,
    }
    save_cfg = {
        "t": {"tmin": 0.0, "tmax": 1.0, "nt": 2},
        "x": {"xmin": 0.0, "xmax": xmax, "nx": save_nx},
        "v": {"vmin": -vmax, "vmax": vmax, "nv": 512},
    }
    _add_dim_axes(save_cfg)

    # Precondition: a 256-point save on a 64-point solver grid pokes out in x too.
    assert save_cfg["x"]["ax"][0] < float(axes["x"][0])
    assert save_cfg["x"]["ax"][-1] > float(axes["x"][-1])

    func = get_dist_save_func(axes=axes, dist_save_config=save_cfg, dist_key="electron")
    f = jnp.exp(-(axes["v"][None, :] ** 2) / 2.0) * jnp.ones((grid_nx, 1))
    out = np.asarray(func(0.0, {"electron": f}, None))

    assert np.all(np.isfinite(out)), f"{int((~np.isfinite(out)).sum())} non-finite cells"


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
