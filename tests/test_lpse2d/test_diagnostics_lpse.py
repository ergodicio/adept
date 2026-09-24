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
