"""Resonance absorption of the pump at the critical surface (terms.light.resonance_absorption;
LPSE laser.evolution.resonanceAbsorption) -- plan 2 L.3."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept._lpse2d.parity import deck_path


def _cfg(angle, ramp, ra, tmax=0.4):
    """A p-polarised super-Gaussian beam (1 um, order 4) at `angle` on a linear ramp n = 0.1 at
    2 um -> 1.9 at 10 um (n_c at 6 um, L_n = 4.4 um, (k0 L_n)^(1/3) = 4.3), FD order 4 at 6 cells per wavelength."""
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {
            "ymax": "2um",
            "ymin": "-2um",
            "xmax": "10um",
            "dx": "0.0625um",
            "dt": "1fs",
            "tmax": f"{tmax}ps",
            "boundary_width": "1.5um",
        }
    )
    cfg["terms"]["light"] = {"pump_depletion": True, "solver": "fd", "fd_order": 4}
    if ra is not False and ra is not None:
        cfg["terms"]["light"]["resonance_absorption"] = ra
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"].update({"noise": False, "srs": False, "tpd": True})
    cfg["units"]["envelope density"] = 0.25
    noise = {"max": 1e-9, "min": 1e-10, "type": "uniform"}
    if ramp:
        cfg["density"] = {
            "basis": "lpse-linear",
            "min": 0.1,
            "max": 1.9,
            "min_location": "2um",
            "max_location": "10um",
            "noise": noise,
        }
    else:
        cfg["density"] = {"basis": "uniform", "val": 0.1, "noise": noise}
    cfg["drivers"]["E0"].update({"angle": angle, "beam_width": "1.0um", "beam_sg_order": 4, "delta_omega_max": 0.0})
    cfg["save"]["fields"]["t"].update({"tmax": f"{tmax}ps", "dt": "0.1ps"})
    cfg["mlflow"]["run"] = f"ra-{angle}-{ramp}-{ra}"
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, _, _ = exo(modules)
    result = sol["solver result"]
    dcfg = exo.adept_module.cfg
    t = np.asarray(result.ts["default"])
    net = np.asarray(result.ys["default"]["incident_flux"])
    x = np.asarray(dcfg["grid"]["x"])
    raw = np.asarray(result.ys["fields"]["E0"])
    e0 = raw.view(np.complex64 if raw.dtype == np.float32 else np.complex128)
    band = (x > 5.5) & (x < 6.5)  # about the critical surface
    energy_nc = np.sum(np.abs(e0[:, band]) ** 2, axis=(1, 2, 3))
    return t, net, np.asarray(result.ts["fields"]), energy_nc


def test_resonance_absorption_at_the_denisov_peak_and_beyond():
    tau = 4.3 * np.sin(np.deg2rad([11.0, 25.0]))
    assert 0.75 < tau[0] < 0.9 and tau[1] > 1.7
    # the launched flux: the same beam in a uniform sub-critical box
    _, net_free, _, _ = _run(_cfg(11.0, ramp=False, ra=False))
    f_inc = np.mean(net_free[-40:])
    assert f_inc > 0.3

    t, net, tf, energy = _run(_cfg(11.0, ramp=True, ra=True))
    steady = t > 0.25
    absorbed = np.mean(net[steady]) / f_inc
    # net flux into the ramp = the absorbed fraction; the Denisov curve peaks (0.5-0.7 across
    # the cold / warm models) near tau 0.8 -- the envelope model with LPSE's warm and Landau
    # terms gives ~0.42 here
    assert 0.3 < absorbed < 0.7
    # a true steady state: the flux and the energy at n_c stop changing
    np.testing.assert_allclose(np.mean(net[t > 0.35]), np.mean(net[(t > 0.25) & (t < 0.3)]), rtol=2e-2)
    assert energy[-1] < 1.05 * energy[np.argmin(np.abs(tf - 0.2))]

    # (without RA the cold-plasma resonance at n_c -- epsilon = 0 with nothing to limit the
    # longitudinal field -- has no steady state: on the 0.05 um grid the energy there grew
    # 6x over 0.1-0.4 ps; the growth is grid-dependent, so it is not asserted here)

    # far past the peak (tau ~ 1.8) the absorption is small; the launched x-flux of the 25 deg
    # beam is the 11 deg one times cos(25)/cos(11) (measured 0.922 vs 0.923)
    t25, net25, _, _ = _run(_cfg(25.0, ramp=True, ra=True))
    absorbed25 = np.mean(net25[t25 > 0.25]) / (f_inc * np.cos(np.deg2rad(25.0)) / np.cos(np.deg2rad(11.0)))
    assert absorbed25 < 0.15


def test_validation_and_translator():
    from adept._lpse2d.helpers import get_derived_quantities, write_units

    cfg = _cfg(11.0, ramp=True, ra=True)
    cfg["terms"]["light"]["solver"] = "spectral"
    write_units(cfg)
    with pytest.raises(ValueError, match="fd light solver"):
        get_derived_quantities(cfg)

    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    if deck_path("test_011") is None:
        pytest.skip("no LPSE decks present")
    cfg, report = translate_parms(parse_parms(deck_path("test_011")), run="test_011")
    ra = cfg["terms"]["light"]["resonance_absorption"]
    assert cfg["terms"]["light"]["solver"] == "fd" and ra["landau_update"] == 10 and ra["filter"] is True
    assert (
        cfg["drivers"]["E0"]["beam_width"] == f"{3.0 / np.sqrt(2.0)}um" and cfg["drivers"]["E0"]["beam_sg_order"] == 4.0
    )
    assert abs(cfg["drivers"]["E0"]["angle"] - np.degrees(np.arctan2(0.1676, 0.98586))) < 1e-6
    assert not any("resonanceAbsorption" in u for u in report["unsupported"])


def _finish(cfg):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def test_an_empty_mapping_enables_resonance_absorption_like_true():
    """``resonance_absorption: {}`` is on, as ``true`` (PR #378 review): setup builds the same finite
    Landau-rate table for both, and a short run with ``{}`` stays finite (before the fix the
    light solver built the table only for a truthy option and handed the enabled RA no rates)."""
    from adept._lpse2d.core.light import CoupledLight

    tables = []
    for ra in (True, {}):
        light = CoupledLight(_finish(_cfg(11.0, ramp=True, ra=ra, tmax=0.02)))
        assert light.resonance.enabled
        tables.append(np.asarray(light.resonance.damping_over_k_sq))
    assert np.all(np.isfinite(tables[1])) and np.max(tables[1]) > 0.0
    np.testing.assert_array_equal(tables[1], tables[0])
    short = _cfg(11.0, ramp=True, ra={}, tmax=0.02)
    short["save"]["fields"]["t"]["dt"] = "0.01ps"  # a field frame inside the short run
    _, net, _, energy = _run(short)
    assert np.all(np.isfinite(net)) and np.all(np.isfinite(energy))


def _resonance(landau_update, dt_l, rate=None, filter_on=False):
    """A ResonanceAbsorption on the RA test box with its own update interval, sub-step and rate."""
    from adept._lpse2d.core.epw import analytic_landau_rate
    from adept._lpse2d.core.light import CoupledLight
    from adept._lpse2d.core.resonance import ResonanceAbsorption

    cfg = _cfg(11.0, ramp=False, ra={"landau_update": landau_update, "filter": filter_on})
    cfg = _finish(cfg)
    light = CoupledLight(cfg)
    k_shape = (cfg["grid"]["nx"], cfg["grid"]["ny"])
    rates = np.asarray(analytic_landau_rate(cfg)) if rate is None else np.full(k_shape, float(rate))
    return cfg, ResonanceAbsorption(cfg, light, dt_l, rates)


@pytest.mark.parametrize(("landau_update", "n_sub", "n_epw"), [(10, 1, 20), (5, 3, 5)])
def test_resonance_landau_cadence_runs_over_absolute_light_steps(landau_update, n_sub, n_epw):
    """The k-space term acts on every ``landau_update``-th light sub-step counted across EPW steps
    (LPSE raLdCounter; PR #378 review), also when the interval exceeds and does not divide the
    sub-steps per EPW step. With the warm term zeroed and the k-space step marked by +1, the count
    over n_epw EPW steps of n_sub sub-steps is len(range(0, n_epw n_sub, landau_update)) exactly
    (before the fix the local sub-step index restarted every EPW step: 20 and 5 here, not 2 and 3)."""
    from jax import numpy as jnp

    dt = 1.0e-3  # the RA test box's EPW step (ps)
    dt_l = dt / n_sub
    _, ra = _resonance(landau_update, dt_l)
    ra.warm_term = lambda E: jnp.zeros_like(E)
    ra.landau_step = lambda E: E + 1.0
    E = jnp.zeros((4, 1, 2), dtype=jnp.complex128)
    for n in range(n_epw):
        for i in range(n_sub):
            E = ra(n * dt + i * dt_l, E)
    assert float(jnp.real(E[0, 0, 0])) == len(range(0, n_epw * n_sub, landau_update))


def test_resonance_landau_damping_is_bounded_over_the_update_interval():
    """Over the update interval T = landau_update dt_l the k-space term removes at most the
    longitudinal part (PR #378 review): the rate is capped at 1/T, not 1/dt_l. A longitudinal plane
    wave in the unmasked interior with landau_update = 10 is removed (|E| < 1e-12 A) at a huge rate,
    where the old cap multiplied it by 1 - 10 = -9, and halved at gamma T = 0.5 (rtol 1e-12)."""
    from jax import numpy as jnp

    dt_l = 1.0e-4
    for rate, factor in ((1.0e9, 0.0), (0.5 / (10 * dt_l), 0.5)):
        cfg, ra = _resonance(10, dt_l, rate=rate)
        x = np.asarray(cfg["grid"]["x"])
        k = float(np.asarray(cfg["grid"]["kx"])[5])
        ny = cfg["grid"]["ny"]
        amp = 1.0e-3
        E = np.zeros((x.size, ny, 2), dtype=np.complex128)
        E[..., 0] = amp * np.exp(1j * k * x)[:, None]
        out = np.asarray(ra.landau_step(jnp.asarray(E)))
        interior = np.asarray(ra.landau_mask)[:, 0, 0] == 1.0
        assert interior.sum() > 10
        got = np.abs(out[interior, :, 0])
        if factor == 0.0:
            assert np.max(got) < 1e-12 * amp
        else:
            np.testing.assert_allclose(got, factor * amp, rtol=1e-12)


def test_the_quasi_1d_resonance_filter_keeps_the_retained_band():
    """In quasi-1D (ny == 1) the RA filter's transition has zero width; it is a sharp passband
    |k| <= filter_width K_nyq, not an all-zero mask (PR #378 review: the zero-width s-curve made the
    default filter erase the pump on every damping update). A transverse (y-polarised) mode along x
    passes landau_step with the default filter unchanged (rtol 1e-12, fixed before the run), and a
    narrower filter_width keeps exactly the modes with |k| <= filter_width K_nyq."""
    from jax import numpy as jnp

    from adept._lpse2d.core.epw import analytic_landau_rate
    from adept._lpse2d.core.light import CoupledLight
    from adept._lpse2d.core.resonance import ResonanceAbsorption

    for filter_width in (1.0, 0.5):
        cfg = _cfg(0.0, ramp=False, ra={"filter_width": filter_width})
        cfg["grid"].update({"ymax": "0.02um", "ymin": "-0.02um"})
        cfg["drivers"]["E0"].pop("beam_width")
        cfg = _finish(cfg)
        assert cfg["grid"]["ny"] == 1
        light = CoupledLight(cfg)
        ra = ResonanceAbsorption(cfg, light, light.dt_l, np.asarray(analytic_landau_rate(cfg)))
        x = np.asarray(cfg["grid"]["x"])
        kx = np.asarray(cfg["grid"]["kx"])
        k_nyq = np.abs(kx).max()
        for ik in (3, len(kx) // 3):
            k = float(kx[ik])
            E = np.zeros((x.size, 1, 2), dtype=np.complex128)
            E[:, 0, 1] = 1.0e-3 * np.exp(1j * k * x)
            out = np.asarray(ra.landau_step(jnp.asarray(E)))
            expected = E if abs(k) <= filter_width * k_nyq else 0.0 * E
            np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12 * 1.0e-3)
