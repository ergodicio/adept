"""Small end-to-end tests for the VFP-2D ADEPTModule."""

import copy

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.sharding import NamedSharding

from adept.vfp2d import BaseVFP2D
from adept.vfp2d.plotting import add_reconnection_diagnostics


def _config(collisions=False):
    return {
        "solver": "vfp-2d",
        "mlflow": {"experiment": "vfp2d-test", "run": "uniform"},
        "units": {
            "laser_wavelength": "351nm",
            "reference electron temperature": "3000eV",
            "reference ion temperature": "300eV",
            "reference electron density": "2.275e21/cm^3",
            "Z": 6,
            "Ion": "Au+",
            "logLambda": "nrl",
        },
        "density": {
            "quasineutrality": True,
            "species-electron": {
                "m": 2.0,
                "n": {"basis": "uniform", "baseline": 1.0},
                "T": {"basis": "uniform", "baseline": 1.0},
            },
        },
        "grid": {
            "xmin": "0um",
            "xmax": "2um",
            "nx": 2,
            "ymin": "0um",
            "ymax": "2um",
            "ny": 2,
            "tmin": "0fs",
            "tmax": "0.02fs",
            "dt": "0.01fs",
            "nv": 8,
            "vmax": 5.0,
            "lmax": 2,
            "mmax": 2,
        },
        "terms": {
            "fokker_planck": {
                "active": collisions,
                "flm": {"ee": False},
                "f00": {"model": "CoulombianKernel", "scheme": "central"},
            }
        },
        "drivers": {},
        "save": {"t": {"tmin": "0fs", "tmax": "0.02fs", "nt": 3}},
    }


def _setup_and_run(cfg):
    module = BaseVFP2D(copy.deepcopy(cfg))
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module, module(None, None)


def test_uniform_collisionless_state_is_stationary_and_uses_real_diffrax_storage():
    module, output = _setup_and_run(_config(collisions=False))
    saved = output["solver result"].ys
    assert saved["flm"].shape == (3, 2, 2, module.layout.size, 8, 2)
    assert saved["flm"].dtype == jnp.float64
    np.testing.assert_allclose(saved["flm"][0], saved["flm"][-1], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(saved["e"], 0.0, atol=1e-13)
    np.testing.assert_allclose(saved["b"], 0.0, atol=1e-13)

    dataset = module.post_process(output, "")["vfp2d"]
    assert dataset.flm_real.dims == ("t", "x", "y", "harmonic", "v")
    np.testing.assert_array_equal(dataset.ell, [0, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(dataset.m, [0, 0, 1, 0, 1, 2])


def test_collisional_end_to_end_step_is_finite():
    _module, output = _setup_and_run(_config(collisions=True))
    for value in output["solver result"].ys.values():
        assert jnp.all(jnp.isfinite(value))


def test_x_sharding_places_spatial_state_and_args():
    cfg = _config(collisions=True)
    cfg["grid"]["nx"] = 4
    cfg["grid"]["sharding"] = {"enabled": True, "axis": "x"}
    cfg["terms"].update(
        {
            "field_solver": {"mode": "kinetic-ohm"},
            "hou_li_filter": {"is_on": True, "dimensions": ["y"]},
        }
    )
    module = BaseVFP2D(copy.deepcopy(cfg))
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()

    assert isinstance(module.state["flm"].sharding, NamedSharding)
    assert module.state["flm"].sharding.spec[0] == "x"
    assert module.state["e"].sharding.spec[0] == "x"
    assert module.args["ni"].sharding.spec[0] == "x"
    output = module(None, None)
    assert jnp.all(jnp.isfinite(output["solver result"].ys["flm"]))


def test_chang_cooper_collisions_preserve_uniform_maxwellian():
    cfg = _config(collisions=True)
    cfg["terms"]["fokker_planck"]["f00"]["scheme"] = "chang_cooper"
    cfg["grid"].update({"dt": "2fs", "tmax": "20fs"})
    cfg["save"]["t"].update({"tmax": "20fs", "nt": 2})
    module, output = _setup_and_run(cfg)
    temperature = module.post_process(output, "")["vfp2d"].temperature
    np.testing.assert_allclose(temperature[-1], temperature[0], rtol=2e-12, atol=2e-12)


def test_spatial_ib_driver_builds_two_gaussian_hotspots():
    cfg = _config(collisions=True)
    cfg["grid"].update({"nx": 12, "ny": 16, "xmin": "-3um", "xmax": "3um", "ymin": "-4um", "ymax": "4um"})
    cfg["drivers"] = {
        "ib": {
            "intensity_1e15_Wcm2": 0.25,
            "polarisation": "linear",
            "switch_off": "1fs",
            "switch_width": "0.1fs",
            "profile": {
                "basis": "gaussian_spots",
                "x_center": "0um",
                "x_radius": "1um",
                "y_centers": ["-2um", "2um"],
                "y_radius": "0.75um",
            },
        }
    }
    module = BaseVFP2D(copy.deepcopy(cfg))
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()

    heating = module.args["ib_vosc2"]
    assert heating.shape == (12, 16)
    assert jnp.all(heating >= 0.0)
    center_x = int(jnp.argmin(jnp.abs(module.grid.x)))
    hot_y = int(jnp.argmax(heating[center_x]))
    expected_y = float(2e-6 / module.plasma_norm.L0.to("m").magnitude)
    assert abs(abs(float(module.grid.y[hot_y])) - expected_y) < 2 * module.grid.dy
    assert heating[center_x, hot_y] > 10.0 * heating[0, module.grid.ny // 2]
    assert module.args["ib_t_off"] > module.args["ib_switch_width"] > 0.0
    derived = module.cfg["units"]["derived"]
    expected_ratio = (
        derived["nuee_coeff"] * derived["logLam_ratio"] * module.args["Z"] ** 2 * module.args["ni"] / derived["w0_norm"]
    )
    np.testing.assert_allclose(module.args["ib_Z2ni_w0"], expected_ratio)


def test_isothermal_uniform_hidden_gradient_has_no_biermann_or_temperature_structure():
    cfg = _config(collisions=False)
    cfg["grid"].update({"nx": 4, "ny": 4})
    cfg["terms"]["field_solver"] = {
        "mode": "kinetic-ohm",
        "hidden_density_gradient": {
            "active": True,
            "scale_length": "2um",
            "profile": {"basis": "uniform", "baseline": 1.0},
        },
    }

    module, output = _setup_and_run(cfg)
    saved = output["solver result"].ys
    dataset = module.post_process(output, "")["vfp2d"]

    np.testing.assert_allclose(saved["b"], 0.0, atol=2e-13)
    final_temperature = np.asarray(dataset.temperature[-1])
    np.testing.assert_allclose(
        final_temperature,
        np.asarray(dataset.temperature[0]),
        rtol=2e-8,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        final_temperature,
        np.mean(final_temperature),
        rtol=2e-12,
        atol=2e-12,
    )


def test_kinetic_ohm_mode_runs_without_explicit_maxwell_evolution(tmp_path):
    cfg = _config(collisions=False)
    cfg["terms"]["field_solver"] = {
        "mode": "kinetic-ohm",
        "hidden_density_gradient": {
            "active": True,
            "scale_length": "2um",
            "switch_off": "1fs",
            "profile": {
                "basis": "gaussian_spots",
                "x_radius": "0.5um",
                "y_centers": ["0um"],
                "y_radius": "0.5um",
            },
        },
    }
    module, output = _setup_and_run(cfg)
    saved = output["solver result"].ys
    assert module.field_mode == "kinetic-ohm"
    assert jnp.all(jnp.isfinite(saved["flm"]))
    assert jnp.all(jnp.isfinite(saved["e"]))
    assert jnp.all(jnp.isfinite(saved["b"]))
    assert jnp.max(jnp.abs(saved["e"][-1, ..., 2])) > 0.0
    postprocessed = module.post_process(output, str(tmp_path))
    dataset = postprocessed["vfp2d"]
    for name in (
        "ohm_resistive",
        "ohm_hall",
        "ohm_nernst",
        "ohm_scalar_pressure",
        "ohm_tensor_pressure",
    ):
        assert dataset[name].dims == ("t", "x", "y", "component")
    reconstructed = sum(
        dataset[name]
        for name in (
            "ohm_resistive",
            "ohm_hall",
            "ohm_nernst",
            "ohm_scalar_pressure",
            "ohm_tensor_pressure",
        )
    )
    np.testing.assert_allclose(reconstructed, dataset.e, rtol=2e-12, atol=2e-12)
    for diagnostic in (
        "az",
        "xpoint_ez",
        "normalized_reconnection_rate",
        "reconnected_flux",
        "current_sheet_rms_width",
        "reconnection_valid",
        "bz_quadrupole_purity",
    ):
        assert diagnostic in dataset
    assert postprocessed["metrics"]["vfp2d_peak_abs_reconnection_rate"] >= 0.0
    for artifact in (
        "binary/moments.nc",
        "binary/distribution_flm.nc",
        "plots/moments/xy_facet_temperature.png",
        "plots/reconnection/xpoint_history.png",
        "plots/reconnection/ohm_z_lineouts_x0.png",
        "plots/reconnection/bx_jz_sheet_lineouts_x0.png",
        "plots/reconnection/xy_facet_b_z.png",
        "plots/reconnection/topology_nernst_final.png",
    ):
        assert (tmp_path / artifact).is_file()


def test_reconnection_gate_and_bz_quadrupole_detect_a_central_sheet():
    n = 33
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    y = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    bx = np.tanh(3.0 * np.sin(yy))
    by = 0.25 * np.sin(xx)
    bz = np.sin(xx) * np.sin(yy)
    current_z = 0.25 * np.cos(xx) - 3.0 * np.cos(yy) / np.cosh(3.0 * np.sin(yy)) ** 2
    b = np.stack((bx, by, bz), axis=-1)[None, ...]
    current_field = np.zeros_like(b)
    current_field[0, ..., 2] = current_z
    e = np.zeros_like(b)
    e[0, ..., 2] = 0.1
    vn = np.zeros_like(b)
    vn[0, ..., 1] = -np.sign(yy)
    ds = xr.Dataset(
        {
            "b": (("t", "x", "y", "component"), b),
            "e": (("t", "x", "y", "component"), e),
            "current": (("t", "x", "y", "component"), current_field),
            "v_nernst": (("t", "x", "y", "component"), vn),
        },
        coords={"t": [0.0], "x": x, "y": y, "component": ["x", "y", "z"]},
    )

    diagnosed = add_reconnection_diagnostics(ds)
    assert bool(diagnosed.reconnection_valid.item())
    assert np.isfinite(diagnosed.normalized_reconnection_rate.item())
    assert diagnosed.current_sheet_dominance.item() > 0.7
    assert diagnosed.bz_quadrupole_purity.item() > 0.9
