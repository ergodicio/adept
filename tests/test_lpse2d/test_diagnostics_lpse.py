"""LPSE-style diagnostics: Poynting-flux fields (save.fields.poynting) and Thomson probes (save.thomson)."""

from copy import deepcopy

import numpy as np
import yaml
from jax import numpy as jnp


def _finish(cfg):
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_save_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    cfg = get_save_quantities(cfg)
    return cfg


def _cfg(**save):
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": 0.2}
    cfg["grid"].update({"dx": "0.1um", "xmax": "12.8um", "ymax": "3.2um", "ymin": "-3.2um", "dt": "1fs", "tmax": "2fs"})
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": False})
    cfg["save"].update(save)
    return _finish(cfg)


def test_poynting_of_a_plane_wave_is_group_velocity_times_intensity(tmp_path):
    from adept._lpse2d.helpers import make_field_xarrays

    cfg = _cfg()
    cfg["save"]["fields"]["poynting"] = True
    d = cfg["units"]["derived"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])
    k0 = kx[6]  # a resolved grid mode
    x = np.asarray(cfg["grid"]["x"])
    amp = 2.0e-3
    e0 = np.zeros((1, nx, ny, 2), dtype=np.complex128)
    e0[0, :, :, 1] = amp * np.exp(1j * k0 * x)[:, None]
    state = {
        "epw": np.zeros((1, nx, ny), dtype=np.complex128).view(np.float64),
        "E0": e0.view(np.float64),
        "E1": np.zeros((1, nx, ny, 2), dtype=np.complex128).view(np.float64),
    }
    (tmp_path / "binary").mkdir()
    _, fields = make_field_xarrays(cfg, np.array([0.0]), state, str(tmp_path))
    v_g = d["c"] ** 2 * k0 / d["w0"]
    s0x = fields["s0_x"].values[0, 4:-4, :]  # away from the one-sided gradient at the edges
    np.testing.assert_allclose(s0x, v_g * amp**2, rtol=2e-2)
    assert np.max(np.abs(fields["s0_y"].values)) < 1e-9 * v_g * amp**2
    assert np.max(np.abs(fields["s1_x"].values)) == 0.0


def test_thomson_probe_picks_the_mode_in_its_window():
    cfg = _cfg(thomson=[{"k": [0.5, 0.0], "bandwidth": 0.05}, {"k": [-0.5, 0.0], "bandwidth": 0.05}])
    save_func = cfg["save"]["default"]["func"]
    d = cfg["units"]["derived"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])
    k_probe = 0.5 * d["w0"] / d["c"]
    ix = int(np.argmin(np.abs(kx - k_probe)))
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128).at[ix, 0].set(3.0 + 4.0j)
    y = {
        "epw": phi_k,
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
    }
    out = save_func(0.0, y, {"drivers": {k: v["derived"] for k, v in cfg["drivers"].items()}})
    np.testing.assert_allclose(float(out["thomson_0_re"]), 3.0)
    np.testing.assert_allclose(float(out["thomson_0_im"]), 4.0)
    np.testing.assert_allclose(float(out["thomson_0_power"]), 25.0)
    assert float(out["thomson_1_power"]) == 0.0


def _combined_cfg():
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["density"] = {"basis": "uniform", "val": 0.2}
    cfg["grid"].update({"dx": "0.1um", "xmax": "12.8um", "ymax": "3.2um", "ymin": "-3.2um", "dt": "1fs", "tmax": "2fs"})
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False}
    cfg["save"]["fields"]["poynting"] = True
    # the full-grid save path (both save paths take E1 through the same transverse projection,
    # helpers._raman_light_part, before any interpolation)
    cfg["save"]["fields"].pop("x", None)
    cfg["save"]["fields"].pop("y", None)
    return _finish(cfg)


def _saved_fields(cfg, e1, tmp_path):
    """Run the field save function on one state and build the xarrays from its output."""
    from adept._lpse2d.helpers import make_field_xarrays

    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    y = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.asarray(e1).view(jnp.float64),
    }
    saved = cfg["save"]["fields"]["func"](0.0, y, None)
    state = {k: np.asarray(v)[None] for k, v in saved.items()}
    (tmp_path / "binary").mkdir(exist_ok=True)
    _, fields = make_field_xarrays(cfg, np.array([0.0]), state, str(tmp_path))
    return saved, fields


def test_combined_raman_flux_maps_exclude_the_epw(tmp_path):
    """The combined solver's E1 carries the EPW as its longitudinal part; the saved E1 fields and
    the s1 flux maps are the Raman light (transverse part) only, as the default series. Tolerances
    fixed before the run: a pure longitudinal mode leaves < 1e-9 of v_g |A|^2 in s1, and a mixed
    field's s1 equals its transverse part's to 1e-9."""
    cfg = _combined_cfg()
    d = cfg["units"]["derived"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])
    k = kx[6]
    x = np.asarray(cfg["grid"]["x"])
    amp = 2.0e-3
    wave = amp * np.exp(1j * k * x)[:, None] * np.ones((1, ny))
    longitudinal = np.zeros((nx, ny, 2), dtype=np.complex128)
    longitudinal[..., 0] = wave  # E parallel to k: a travelling plasma wave, no light
    transverse = np.zeros((nx, ny, 2), dtype=np.complex128)
    transverse[..., 1] = 0.5 * wave

    scale = d["c"] ** 2 * k / d["w1"] * amp**2
    saved, fields = _saved_fields(cfg, longitudinal, tmp_path)
    assert np.max(np.abs(np.asarray(saved["E1"]).view(np.complex128))) < 1e-9 * amp
    assert np.max(np.abs(fields["s1_x"].values)) < 1e-9 * scale
    assert np.max(np.abs(fields["s1_y"].values)) < 1e-9 * scale

    _, fields_mixed = _saved_fields(cfg, longitudinal + transverse, tmp_path)
    _, fields_t = _saved_fields(cfg, transverse, tmp_path)
    np.testing.assert_allclose(fields_mixed["s1_x"].values, fields_t["s1_x"].values, rtol=0, atol=1e-9 * scale)
    assert np.max(np.abs(fields_t["s1_x"].values)) > 0.1 * scale
