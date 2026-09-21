"""Initial-unit, import, topology and quasistatic-current checks."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from adept.normalization import UREG, laser_normalization
from adept.vfp2d import BaseVFP2D, Grid, Maxwell2D, conserved_to_primitive, current, density, real_to_complex
from adept.vfp2d.geometry import initial_magnetic_field, profile_2d


def _small_config():
    path = Path(__file__).parents[2] / "configs/vfp-2d/magpie-carbon-periodic.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["grid"].update(nx=8, ny=16, nv=16, ymin="-2mm", ymax="2mm")
    cfg["initial_conditions"]["ion_velocity"][1]["wavelength"] = "4mm"
    cfg["terms"]["fokker_planck"]["active"] = False
    return cfg


def _initialize(cfg):
    module = BaseVFP2D(cfg)
    module.write_units()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module


def _grid():
    norm = laser_normalization("351nm", "15eV")
    length = float((1 * UREG.mm / norm.L0).to("").magnitude)
    grid = Grid(
        xmin=-length, xmax=length, nx=16, ymin=-2 * length, ymax=2 * length, ny=24, vmax=0.04, nv=16, dt=1.0, l_max=2
    )
    return grid, norm


def test_physical_counterstreams_have_symmetry_quasineutrality_and_ampere_current():
    module = _initialize(_small_config())
    primitive = conserved_to_primitive(module.state["ions"], module.ion_gamma)
    velocity = np.asarray(primitive[..., 1:4])
    np.testing.assert_allclose(velocity[..., [0, 2]], 0.0, atol=0)
    np.testing.assert_allclose(velocity[..., 1], -velocity[:, ::-1, 1], rtol=1e-13, atol=1e-17)
    assert np.all(velocity[:, module.grid.ny // 2 :, 1] < 0)
    assert np.max(abs(velocity)) <= float((50 * UREG.km / UREG.s / module.plasma_norm.v0).to(""))
    ion_density = module.state["ions"][..., 0] / module.ion_mass
    ti = primitive[..., 4] / ion_density
    np.testing.assert_allclose(ti, float((50 * UREG.eV / (UREG.m_e * UREG.c**2)).to("")), rtol=1e-13)
    flm = real_to_complex(module.state["flm"])
    ne = density(flm, module.layout, module.grid.v, module.grid.dv)
    np.testing.assert_allclose(ne, 4 * ion_density, rtol=1e-14)
    b = module.state["b"]
    np.testing.assert_allclose(b[..., 0], -b[:, ::-1, 0], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(b[..., 1], -b[::-1, :, 1], rtol=1e-12, atol=1e-18)
    target = module._maxwell.c2 * module._maxwell.curl(b)
    assert np.max(abs(target)) > 0
    np.testing.assert_allclose(
        current(flm, module.layout, module.grid.v, module.grid.dv), target, rtol=1e-12, atol=1e-25
    )
    np.testing.assert_array_equal(module.state["current_projection_energy"], 0.0)
    np.testing.assert_allclose(module._maxwell.ddx(b[..., 0]) + module._maxwell.ddy(b[..., 1]), 0.0, atol=1e-23)


@pytest.mark.parametrize("finite_difference", [False, True])
def test_vector_potential_has_zero_discrete_divergence_and_physical_units(finite_difference):
    grid, norm = _grid()
    amplitude = "0.3T*mm"
    spec = {
        "vector_potential": {
            "z": {
                "scale": amplitude,
                "profile": {
                    "x": {"basis": "cosine", "baseline": 1.0, "amplitude": 0.7, "wavelength": "2mm"},
                    "y": {"basis": "cosine", "baseline": 1.0, "amplitude": 0.4, "wavelength": "4mm"},
                },
            }
        },
        "uniform": ["0T", "0T", "2T"],
    }
    b = initial_magnetic_field(spec, grid, norm, finite_difference=finite_difference)
    derivatives = Maxwell2D(
        grid.kx, grid.ky, c=1.0, dx=grid.dx if finite_difference else None, dy=grid.dy if finite_difference else None
    )
    np.testing.assert_allclose(derivatives.ddx(b[..., 0]) + derivatives.ddy(b[..., 1]), 0.0, atol=1e-22)
    b0 = norm.m0 / (norm.q0 * norm.tau)
    np.testing.assert_allclose(b[..., 2], float((2 * UREG.tesla / b0).to("")), rtol=1e-14)
    if not finite_difference:
        length_x = grid.xmax - grid.xmin
        length_y = grid.ymax - grid.ymin
        a = float((UREG.Quantity(amplitude) / (b0 * norm.L0)).to(""))
        expected_bx = (
            -a
            * (1 + 0.7 * jnp.cos(2 * jnp.pi * grid.x / length_x))[:, None]
            * (0.4 * 2 * jnp.pi / length_y * jnp.sin(2 * jnp.pi * grid.y / length_y))[None, :]
        )
        np.testing.assert_allclose(b[..., 0], expected_bx, rtol=1e-12, atol=1e-19)


def test_legacy_normalized_velocity_and_reference_ion_temperature_are_preserved():
    cfg = _small_config()
    cfg.pop("initial_conditions")
    cfg["terms"]["ion_fluid"]["initial_velocity"] = [1e-5, -2e-5, 3e-5]
    module = _initialize(cfg)
    primitive = conserved_to_primitive(module.state["ions"], module.ion_gamma)
    np.testing.assert_allclose(
        primitive[..., 1:4], np.broadcast_to([1e-5, -2e-5, 3e-5], primitive[..., 1:4].shape), rtol=1e-14
    )
    np.testing.assert_array_equal(module.state["b"], 0.0)


def test_xy_import_interpolates_bilinear_data_in_explicit_units(tmp_path):
    grid, norm = _grid()
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-0.2, 0.0, 0.2])  # cm
    values = 20 + 2 * x[:, None] + 3 * (10 * y[None, :]) + x[:, None] * (10 * y[None, :])
    path = tmp_path / "snapshot.npz"
    np.savez(path, x=x, y=y, temperature=values)
    profile = {
        "basis": "file_xy",
        "path": str(path),
        "x_unit": "mm",
        "y_unit": "cm",
        "value_unit": "eV",
        "value_key": "temperature",
    }
    result = profile_2d(profile, grid, norm, reference=norm.T0)
    xt = np.asarray(grid.x) * float((norm.L0 / UREG.mm).to(""))
    yt = np.asarray(grid.y) * float((norm.L0 / UREG.mm).to(""))
    expected = (20 + 2 * xt[:, None] + 3 * yt[None, :] + xt[:, None] * yt[None, :]) / 15
    np.testing.assert_allclose(result, expected, rtol=2e-14)


@pytest.mark.parametrize("failure", ["unsorted", "nonfinite", "transposed", "extrapolation", "missing_units"])
def test_xy_import_rejects_ambiguous_or_invalid_data(tmp_path, failure):
    grid, norm = _grid()
    x, y = np.linspace(-1, 1, 3), np.linspace(-2, 2, 5)
    values = np.ones((3, 5))
    if failure == "unsorted":
        x = x[::-1]
    elif failure == "nonfinite":
        values[0, 0] = np.nan
    elif failure == "transposed":
        values = values.T
    elif failure == "extrapolation":
        x *= 0.1
    path = tmp_path / "snapshot.npz"
    np.savez(path, x=x, y=y, values=values)
    profile = {"basis": "file_xy", "path": str(path), "x_unit": "mm", "y_unit": "mm", "value_unit": "eV"}
    if failure == "missing_units":
        profile.pop("value_unit")
    with pytest.raises(ValueError):
        profile_2d(profile, grid, norm, reference=norm.T0)


def test_spatial_ion_temperature_and_electron_profiles_use_reference_units(tmp_path):
    cfg = _small_config()
    x = np.linspace(-14, 14, 3)
    y = np.linspace(-2, 2, 5)
    temperature = 50 + y[None, :] * np.ones((x.size, 1))
    path = tmp_path / "plane.npz"
    np.savez(path, x=x, y=y, Ti=temperature, ne=np.full((3, 5), 3e17), Te=np.full((3, 5), 15.0))
    template = {"basis": "file_xy", "path": str(path), "x_unit": "mm", "y_unit": "mm"}
    cfg["initial_conditions"]["ion_temperature"] = {**template, "value_unit": "eV", "value_key": "Ti"}
    cfg["density"]["species-electron"]["n"] = {**template, "value_unit": "cm^-3", "value_key": "ne"}
    cfg["density"]["species-electron"]["T"] = {**template, "value_unit": "eV", "value_key": "Te"}
    module = _initialize(cfg)
    primitive = conserved_to_primitive(module.state["ions"], module.ion_gamma)
    ni = module.state["ions"][..., 0] / module.ion_mass
    measured = np.asarray(primitive[..., 4] / ni) * float((1 * UREG.m_e * UREG.c**2 / UREG.eV).to(""))
    y_mm = np.asarray(module.grid.y) * float((module.plasma_norm.L0 / UREG.mm).to(""))
    np.testing.assert_allclose(measured, np.broadcast_to(50 + y_mm, measured.shape), rtol=1e-13)


@pytest.mark.parametrize("temperature", ["0eV", "-1eV", np.nan])
def test_invalid_ion_temperature_is_rejected(temperature):
    cfg = _small_config()
    cfg["initial_conditions"]["ion_temperature"] = temperature
    with pytest.raises(ValueError):
        _initialize(cfg)


def test_unresolved_sheet_and_conflicting_magnetic_inputs_are_rejected():
    grid, norm = _grid()
    with pytest.raises(ValueError, match="two y cells"):
        initial_magnetic_field({"periodic_sheet": {"field": "3T", "width": "0.01mm"}}, grid, norm)
    with pytest.raises(ValueError, match="not both"):
        initial_magnetic_field({"vector_potential": [0, 0, 0], "periodic_sheet": {}}, grid, norm)


def test_seeded_magnetic_counterflow_runs_two_finite_coupled_steps():
    module = _initialize(_small_config())
    initial_b = np.asarray(module.state["b"])
    output = module(None, None)
    saved = output["solver result"].ys
    for value in saved.values():
        assert np.all(np.isfinite(np.asarray(value)))
    assert np.max(abs(np.asarray(saved["b"][-1]) - initial_b)) > 0
    assert np.max(abs(np.asarray(saved["ions"][-1]) - np.asarray(saved["ions"][0]))) > 0


def test_current_sheet_requires_transverse_current_harmonic():
    cfg = _small_config()
    cfg["grid"]["mmax"] = 0
    with pytest.raises(ValueError, match="transverse Ampere current"):
        _initialize(cfg)


@pytest.mark.parametrize("guide_only", [False, True])
def test_axisymmetric_harmonics_accept_compatible_magnetic_current(guide_only):
    cfg = _small_config()
    cfg["grid"]["mmax"] = 0
    field = {"uniform": ["0T", "0T", "1T"]}
    if not guide_only:
        field["vector_potential"] = {
            "x": {
                "scale": "0.03T*mm",
                "profile": {
                    "basis": "cosine",
                    "axis": "y",
                    "baseline": 1.0,
                    "amplitude": 1.0,
                    "wavelength": "4mm",
                },
            }
        }
    cfg["initial_conditions"]["magnetic_field"] = field
    module = _initialize(cfg)
    flm = real_to_complex(module.state["flm"])
    required = module._maxwell.c2 * module._maxwell.curl(module.state["b"])
    measured = current(flm, module.layout, module.grid.v, module.grid.dv)
    np.testing.assert_allclose(measured, required, rtol=1e-12, atol=1e-22)
    if not guide_only:
        assert float(jnp.max(jnp.abs(required[..., 0]))) > 0


def test_dimensional_separable_profiles_require_a_single_physical_scale():
    grid, norm = _grid()
    spec = {
        "x": {"basis": "uniform", "baseline": "30eV"},
        "y": {"basis": "uniform", "baseline": "45eV"},
    }
    with pytest.raises(ValueError, match="cannot both have physical amplitudes"):
        profile_2d(spec, grid, norm, reference=norm.T0)
    spec["y"]["baseline"] = 3.0
    np.testing.assert_allclose(profile_2d(spec, grid, norm, reference=norm.T0), 6.0)
    spec["x"]["baseline"] = 2.0
    np.testing.assert_allclose(profile_2d({"scale": "15eV", "profile": spec}, grid, norm, reference=norm.T0), 6.0)
