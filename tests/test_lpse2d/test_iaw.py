"""Tests for the ion-acoustic-wave path ported from lpse-matlab."""

import numpy as np
import yaml
from jax import numpy as jnp


def _make_cfg(*, landau=0.0, collisions=0.0, max_density_perturbation=None, with_pump=False):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    if with_pump:
        with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
            cfg["drivers"]["E0"] = yaml.safe_load(fi)["drivers"]["E0"]

    cfg["density"] = {"basis": "uniform", "val": 0.25}
    cfg["grid"].update(
        {
            "boundary_width": "0.2um",
            "dt": "2fs",
            "dx": "0.1um",
            "xmax": "6.4um",
            "tmax": "20fs",
            "ymax": "0.05um",
            "ymin": "-0.05um",
            "light_substeps": 1,
        }
    )
    cfg["terms"]["zero_mask"] = True
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
    cfg["terms"]["epw"]["source"] = {"noise": False, "tpd": False, "srs": False}
    cfg["terms"]["epw"]["density_gradient"] = False
    cfg["terms"]["iaw"] = {
        "active": True,
        "boundary": {"x": "periodic", "y": "periodic"},
        "damping": {"collisions": collisions, "landau": landau},
        "max_density_perturbation": max_density_perturbation,
    }

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _zero_fields(cfg):
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    return {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
    }


def test_iaw_split_step_matches_matlab_order():
    """W is kicked by -laplacian(PP) before Nelf is drifted by the new W."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    cfg = _make_cfg()
    solver = IonAcousticWave(cfg)
    x = np.asarray(cfg["grid"]["x"])
    k = np.asarray(cfg["grid"]["kx"])[3]
    density = jnp.asarray(0.02 * np.cos(k * (x - x[0])))[:, None]
    state = {
        **_zero_fields(cfg),
        "iaw_density": density,
        "iaw_velocity_divergence": jnp.zeros_like(density),
    }

    expected_w = -solver.dt * solver.laplacian(solver.cs**2 * density)
    expected_n = density - solver.dt * expected_w
    out = solver(state)

    np.testing.assert_allclose(out["iaw_velocity_divergence"], expected_w, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(out["iaw_density"], expected_n, rtol=1e-11, atol=1e-13)


def test_iaw_landau_damps_velocity_at_twice_the_amplitude_rate():
    """Damping W at 2*gamma makes the coupled acoustic-mode amplitude decay at gamma."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    landau_factor = 0.2
    cfg = _make_cfg(landau=landau_factor)
    solver = IonAcousticWave(cfg)
    x = np.asarray(cfg["grid"]["x"])
    k = np.asarray(cfg["grid"]["kx"])[4]
    velocity = jnp.asarray(np.cos(k * (x - x[0])))[:, None]
    state = {
        **_zero_fields(cfg),
        "iaw_density": jnp.zeros_like(velocity),
        "iaw_velocity_divergence": velocity,
    }

    out = solver(state)
    expected = velocity * np.exp(-2.0 * landau_factor * solver.cs * abs(k) * solver.dt)
    np.testing.assert_allclose(out["iaw_velocity_divergence"], expected, rtol=1e-11, atol=1e-13)


def test_light_intensity_drives_iaw_and_density_limiter():
    """The pump ponderomotive pressure drives W; the optional Nelf limiter is applied last."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    cfg = _make_cfg(max_density_perturbation=1.0e-8)
    solver = IonAcousticWave(cfg)
    x = np.asarray(cfg["grid"]["x"])
    k = np.asarray(cfg["grid"]["kx"])[2]
    E0 = jnp.zeros_like(_zero_fields(cfg)["E0"])
    E0 = E0.at[..., 1].set(jnp.asarray(1.0 + 0.25 * np.cos(k * (x - x[0])))[:, None])
    density = jnp.zeros(E0.shape[:2])
    state = {
        **_zero_fields(cfg),
        "E0": E0,
        "iaw_density": density,
        "iaw_velocity_divergence": jnp.zeros_like(density),
    }

    potential = solver.ponderomotive_prefactor * jnp.sum(jnp.abs(E0) ** 2, axis=-1) / solver.w0**2
    expected_w = -solver.dt * solver.laplacian(potential)
    out = solver(state)

    np.testing.assert_allclose(out["iaw_velocity_divergence"], expected_w, rtol=1e-11, atol=1e-13)
    assert np.max(np.abs(np.asarray(out["iaw_density"]))) <= 1.0e-8


def test_iaw_density_detunes_epw_by_matlab_phase():
    """A uniform Nelf multiplies the EPW envelope by exp(-i wp0 Nelf dt / 2)."""
    from adept._lpse2d.core.epw import SpectralEPWSolver

    cfg = _make_cfg()
    solver = SpectralEPWSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128).at[3, 0].set(1.0 + 0.5j)
    common = {
        "epw": phi_k,
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
    }
    dn = 0.03
    out_zero = solver(jnp.asarray(0.0), {**common, "iaw_density": jnp.zeros((nx, ny))}, None)
    out_dn = solver(jnp.asarray(0.0), {**common, "iaw_density": jnp.full((nx, ny), dn)}, None)

    phase = np.exp(-1j * solver.wp0 * dn * solver.dt / 2.0)
    np.testing.assert_allclose(out_dn[3, 0], out_zero[3, 0] * phase, rtol=1e-11, atol=1e-13)


def test_iaw_density_detunes_raman_rhs():
    """The Raman light sees the same Nelf addition to n/n_env as the MATLAB solver."""
    from adept._lpse2d.core.raman import RamanLight

    cfg = _make_cfg()
    solver = RamanLight(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E1 = jnp.ones((nx, ny, 2), dtype=jnp.complex128)
    E0 = jnp.zeros_like(E1)
    laplacian_phi = jnp.zeros((nx, ny), dtype=jnp.complex128)
    dn = jnp.full((nx, ny), 0.04)

    base = solver.rhs(0.0, E1, E0, laplacian_phi, None)
    with_iaw = solver.rhs(0.0, E1, E0, laplacian_phi, None, dn)
    expected = -1j * solver.wp0**2 / (2.0 * solver.w1) * dn[..., None] * E1
    np.testing.assert_allclose(with_iaw - base, expected, rtol=1e-11, atol=1e-13)


def test_iaw_density_detunes_evolved_pump_rhs():
    """The dynamic pump sees Nelf in the same density detuning as MATLAB."""
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg(with_pump=True)
    cfg["drivers"]["E0"]["derived"].update({"offset": 0.4, "turn_on_time": 0.01})
    solver = CoupledLight(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E0 = jnp.ones((nx, ny, 2), dtype=jnp.complex128)
    E1 = jnp.zeros_like(E0)
    laplacian_phi = jnp.zeros((nx, ny), dtype=jnp.complex128)
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    dn = jnp.full((nx, ny), 0.04)

    base = solver.pump_rhs(0.0, E0, E1, laplacian_phi, pump_args)
    with_iaw = solver.pump_rhs(0.0, E0, E1, laplacian_phi, pump_args, dn)
    expected = -1j * solver.wp0**2 / (2.0 * solver.w0) * dn[..., None] * E0
    np.testing.assert_allclose(with_iaw - base, expected, rtol=1e-11, atol=1e-13)


def test_iaw_configuration_defaults_to_epw_boundaries():
    # Re-run only the pre-array derivation from a fresh raw deck because the helper
    # converts unit strings in place.
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        fresh = yaml.safe_load(fi)
    fresh["density"] = {"basis": "uniform", "val": 0.25}
    fresh["grid"]["ymax"] = "0.02um"
    fresh["grid"]["ymin"] = "-0.02um"
    fresh["terms"]["iaw"] = {"active": True, "boundary": None}

    from adept._lpse2d.helpers import get_derived_quantities, write_units

    write_units(fresh)
    fresh = get_derived_quantities(fresh)
    assert fresh["terms"]["iaw"]["boundary"] == fresh["terms"]["epw"]["boundary"]


def test_iaw_composes_with_quasi_1d_particle_feedback():
    """The existing HPE tracker and the new IAW state can run in the same 1D model."""
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("configs/envelope-2d/srs-hpe.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["terms"]["hpe"]["n_particles"] = 64
    cfg["terms"]["iaw"] = {"active": True}

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    step = SplitStep(cfg)

    assert step.hpe is not None and step.iaw is not None
