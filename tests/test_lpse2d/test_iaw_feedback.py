"""How the IAW density feeds back into the EPW and the light (``terms.iaw.feedback``).

``iaw_density`` is the local fraction ``delta n / n_b`` (the IAW drive is density-independent),
so the waves see the density ``n_b (1 + iaw_density)``: in units of the envelope density the
perturbation is ``iaw_density * n_b / n_env`` (LPSE ``Nelf * backgroundDensity``). The MATLAB
prototype added ``Nelf`` to ``n_b / n_env`` directly (``feedback: envelope``), exact only where
``n_b = n_env``. Every coupling site is checked at ``n_b = 0.1, n_env = 0.25`` -- test_001's
CBET regime, where the prototype form couples 2.5x too strongly.
"""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

N_B, N_ENV = 0.1, 0.25
FEEDBACK = [("local", N_B / N_ENV), ("envelope", 1.0)]


def _finish(cfg):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _iaw(feedback, **extra):
    return {
        "active": True,
        "boundary": {"x": "periodic", "y": "periodic"},
        "damping": {"collisions": 0.0, "landau": 0.1},
        "feedback": feedback,
        **extra,
    }


def _fd_cfg(feedback, with_pump=False):
    """The test_iaw.py box (uniform, periodic, one light sub-step) at n_b = 0.1, n_env = 0.25."""
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    if with_pump:
        with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
            cfg["drivers"]["E0"] = yaml.safe_load(fi)["drivers"]["E0"]
    cfg["density"] = {"basis": "uniform", "val": N_B}
    cfg["units"]["envelope density"] = N_ENV
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
    cfg["terms"]["iaw"] = _iaw(feedback)
    return _finish(cfg)


def _spectral_cfg(feedback, absorption=False):
    """Spectral pump + Raman light with pump depletion, uniform n_b = 0.1."""
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["density"] = {"basis": "uniform", "val": N_B}
    cfg["units"]["envelope density"] = N_ENV
    cfg["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "0.4um", "ymin": "-0.4um", "dx": "0.1um"})
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": True, "absorption": absorption}
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["iaw"] = _iaw(feedback, solver="spectral", boundary=None)
    return _finish(cfg)


def _combined_cfg(feedback):
    """The combined TPD + SRS solver (test_combined.py's box) at n_b = 0.1."""
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["density"] = {"basis": "uniform", "val": N_B}
    cfg["units"]["envelope density"] = N_ENV
    cfg["grid"].update(
        {
            "boundary_width": "0.6um",
            "dt": "1fs",
            "dx": "0.1um",
            "xmax": "6.4um",
            "tmax": "10fs",
            "ymax": "1.6um",
            "ymin": "-1.6um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["density_gradient"] = True
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False}
    cfg["terms"]["iaw"] = _iaw(feedback, solver="spectral")
    return _finish(cfg)


@pytest.mark.parametrize("feedback, factor", FEEDBACK)
def test_epw_detuning_sees_the_local_density(feedback, factor):
    """A uniform iaw_density dn rotates the EPW by exp(-i wp0/2 dn n_b/n_env dt) (LPSE
    densityUpdateOfE_fft: wpe/(2 No) * Nelf * n_b)."""
    from adept._lpse2d.core.epw import SpectralEPWSolver

    cfg = _fd_cfg(feedback)
    solver = SpectralEPWSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128).at[3, 0].set(1.0 + 0.5j)
    common = {"epw": phi_k, "E0": jnp.zeros((nx, ny, 2), jnp.complex128), "E1": jnp.zeros((nx, ny, 2), jnp.complex128)}
    dn = 0.03
    out_zero = solver(jnp.asarray(0.0), {**common, "iaw_density": jnp.zeros((nx, ny))}, None)
    out_dn = solver(jnp.asarray(0.0), {**common, "iaw_density": jnp.full((nx, ny), dn)}, None)
    phase = np.exp(-1j * solver.wp0 * dn * factor * solver.dt / 2.0)
    np.testing.assert_allclose(out_dn[3, 0], out_zero[3, 0] * phase, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("feedback, factor", FEEDBACK)
def test_fd_raman_and_pump_detuning_see_the_local_density(feedback, factor):
    """The FD light RHS: -i wp0^2/(2 w) dn n_b/n_env E, i.e. -i w0/2 (n_b dn) (w0/w) E -- the
    local density times the local fraction (LPSE LightSolver: Nelf * backgroundDensity)."""
    from adept._lpse2d.core.light import CoupledLight
    from adept._lpse2d.core.raman import RamanLight

    cfg = _fd_cfg(feedback)
    raman = RamanLight(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E1 = jnp.ones((nx, ny, 2), dtype=jnp.complex128)
    zeros = jnp.zeros_like(E1)
    lap_phi = jnp.zeros((nx, ny), dtype=jnp.complex128)
    dn = jnp.full((nx, ny), 0.04)
    delta = raman.rhs(0.0, E1, zeros, lap_phi, None, dn) - raman.rhs(0.0, E1, zeros, lap_phi, None)
    expected = -1j * raman.wp0**2 / (2.0 * raman.w1) * dn[..., None] * factor * E1
    np.testing.assert_allclose(delta, expected, rtol=1e-11, atol=1e-13)

    cfg = _fd_cfg(feedback, with_pump=True)
    cfg["drivers"]["E0"]["derived"].update({"offset": 0.4, "turn_on_time": 0.01})
    pump = CoupledLight(cfg)
    E0 = jnp.ones((nx, ny, 2), dtype=jnp.complex128)
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    delta = pump.pump_rhs(0.0, E0, zeros, lap_phi, pump_args, dn) - pump.pump_rhs(0.0, E0, zeros, lap_phi, pump_args)
    expected = -1j * pump.wp0**2 / (2.0 * pump.w0) * dn[..., None] * factor * E0
    np.testing.assert_allclose(delta, expected, rtol=1e-11, atol=1e-13)
    # the same density shift as the prescribed pump's own background term: wp0^2 n_b/n_env = w0^2 n_b
    if feedback == "local":
        np.testing.assert_allclose(pump.wp0**2 * factor, pump.w0**2 * N_B, rtol=1e-12)  # i.e. -i w0/2 * n_b * dn


@pytest.mark.parametrize("feedback, factor", FEEDBACK)
def test_spectral_light_detuning_sees_the_local_density(feedback, factor):
    """Spectral pump and Raman light over one EPW step with a uniform dn: every operation is
    linear, so the IAW term is a uniform phase exp(-i wp0^2/(2 w) dn n_b/n_env dt) on each field."""
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight

    cfg = _spectral_cfg(feedback)
    light = SpectralCoupledLight(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    x = np.asarray(cfg["grid"]["x"])
    kx = 2.0 * np.pi * 20 / (nx * cfg["grid"]["dx"])
    wave = np.exp(1j * kx * x)[:, None] * np.ones((1, ny))
    E0 = jnp.zeros((nx, ny, 3), jnp.complex128).at[..., 1].set(jnp.asarray(wave))
    E1 = jnp.zeros((nx, ny, 3), jnp.complex128).at[..., 1].set(jnp.asarray(0.3 * np.conj(wave)))
    phi_k = jnp.zeros((nx, ny), jnp.complex128)
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.zeros((1, ny)),  # no injection: only the free evolution is compared
        "phases": jnp.zeros((1, ny)),
    }
    dn = 0.04
    a0, a1 = light(0.0, E0, E1, phi_k, pump_args, None)
    b0, b1 = light(0.0, E0, E1, phi_k, pump_args, None, jnp.full((nx, ny), dn))
    dt = cfg["grid"]["dt"]
    for a, b, w in ((a0, b0, light.w0), (a1, b1, light.w1)):
        a, b = np.asarray(a)[..., 1], np.asarray(b)[..., 1]
        mask = np.abs(a) > 1e-3 * np.abs(a).max()
        expected = np.exp(-1j * light.wp0**2 / (2.0 * w) * dn * factor * dt)
        np.testing.assert_allclose(b[mask] / a[mask], expected, rtol=1e-10)


@pytest.mark.parametrize("feedback, factor", FEEDBACK)
def test_collisional_absorption_sees_the_perturbed_density(feedback, factor):
    """The inverse-bremsstrahlung rate is taken at the total density n_b (1 + dn) (local) --
    n_b + n_env dn with the prototype form."""
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight

    cfg = _spectral_cfg(feedback, absorption=2.0)
    light = SpectralCoupledLight(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    ratio = jnp.full((nx, ny), 0.3)
    dn = 0.1
    got = np.asarray(light.absorption_factor(2.0, ratio, jnp.full((nx, ny), dn)))[..., 0]
    n_total = 0.3 * (1.0 + dn * factor / (N_B / N_ENV))
    np.testing.assert_allclose(got, np.exp(-2.0 * light.dt_l * n_total**2), rtol=1e-12)
    if feedback == "local":
        np.testing.assert_allclose(n_total, 0.3 * (1.0 + dn), rtol=1e-12)


@pytest.mark.parametrize("feedback, factor", FEEDBACK)
def test_combined_field_sees_the_local_density(feedback, factor):
    """The combined wp0-enveloped field: the IAW term exp(-i wp0/2 dn n_b/n_env dt) on the whole
    field (EPW and Raman light alike), a uniform phase on the free evolution."""
    from adept._lpse2d.core.combined import CombinedSolver

    cfg = _combined_cfg(feedback)
    combined = CombinedSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(3)
    mask = np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    field = np.stack(
        [np.fft.ifft2((rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * mask) for _ in range(2)], axis=-1
    )
    E1 = jnp.asarray(field)
    E0 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    y = {"E0": E0, "E1": E1, "epw": combined.potential(E1)}
    dn = 0.03
    _, a1, _ = combined(0.0, y, {}, lambda t: E0)
    _, b1, _ = combined(0.0, {**y, "iaw_density": jnp.full((nx, ny), dn)}, {}, lambda t: E0)
    a1, b1 = np.asarray(a1), np.asarray(b1)
    big = np.abs(a1) > 1e-3 * np.abs(a1).max()
    expected = np.exp(-1j * combined.wp0 / 2.0 * dn * factor * cfg["grid"]["dt"])
    np.testing.assert_allclose(b1[big] / a1[big], expected, rtol=1e-9)


def test_feedback_option_is_validated():
    from adept._lpse2d.core.iaw import iaw_feedback_factor

    cfg = {"terms": {"iaw": {"active": True, "feedback": "bogus"}}, "grid": {}, "units": {}}
    with pytest.raises(ValueError, match="feedback"):
        iaw_feedback_factor(cfg)
    # inactive IAW or the prototype form: no factor
    assert iaw_feedback_factor({"terms": {}, "grid": {}, "units": {}}) == 1.0
    assert iaw_feedback_factor({"terms": {"iaw": {"active": True, "feedback": "envelope"}}}) == 1.0
