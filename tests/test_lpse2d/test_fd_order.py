"""FD light stencil order (terms.light.fd_order; LPSE evolution.solverOrder 2 / 4 / 6): the
stencils, the plane-wave injector that follows from them, the dt limit and the translator."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept._lpse2d.core import stencils
from adept._lpse2d.parity import deck_path


def _load_cfg():
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        return yaml.safe_load(fi)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_stencil_symbol_converges_at_its_order(order):
    """sigma(theta) / (-theta^2) - 1 falls as theta^order: halving theta divides it by 2^order."""
    e1 = abs(stencils.symbol(order, 0.4) / (-(0.4**2)) - 1.0)
    e2 = abs(stencils.symbol(order, 0.2) / (-(0.2**2)) - 1.0)
    assert 0.8 * 2**order < e1 / e2 < 1.25 * 2**order
    # the first-difference stencil is exact on x^order
    d = stencils.first_derivative(order)
    m = order // 2
    for power in range(order):
        deriv = sum(d[j + m] * float(j) ** power for j in range(-m, m + 1))
        assert abs(deriv - (1.0 if power == 1 else 0.0)) < 1e-12


def test_stencil_symbol_and_eigenvalue_factor_match_lpse():
    """The coefficients are LPSE's (SchrodingerSolver3::step_2d, /12 and /180 denominators)."""
    np.testing.assert_allclose(stencils.second_derivative(4) * 12, [-1, 16, -30, 16, -1])
    np.testing.assert_allclose(stencils.second_derivative(6) * 180, [2, -27, 270, -490, 270, -27, 2])
    assert stencils.max_eigenvalue_factor(2) == 1.0
    np.testing.assert_allclose([stencils.max_eigenvalue_factor(4), stencils.max_eigenvalue_factor(6)], [4 / 3, 68 / 45])
    with pytest.raises(ValueError):
        stencils.check_order(3)


def test_second_order_injector_is_the_two_point_source():
    assert stencils.injector_weights(2, +1) == {0: [(1, 1.0)], 1: [(-1, -1.0)]}
    assert stencils.injector_weights(2, -1) == {0: [(1, -1.0)], 1: [(-1, 1.0)]}
    assert stencils.injector_offsets(4, +1) == [-1, 0, 1, 2]
    assert stencils.injector_offsets(6, +1) == [-2, -1, 0, 1, 2, 3]


@pytest.mark.parametrize("order", [2, 4, 6])
def test_commutator_injector_launches_a_clean_wave_in_a_discrete_steady_state(order):
    """Solve the 1-D discrete Helmholtz problem of the stencil with the injector source, damped
    only in two end layers: the field is a single rightward wave above the plane with the
    amplitude and grid wavenumber ``stencils`` predict, and the backward leak -- the
    analytic wave's dispersion mismatch with the stencil -- falls with the order."""
    n, i_plane, k_dx, layer, nu_max = 600, 250, 0.8, 120, 0.3
    c = stencils.second_derivative(order)
    m = order // 2
    r = np.arange(n)
    nu = nu_max * (np.clip((layer - r) / layer, 0, 1) ** 2 + np.clip((r - (n - 1 - layer)) / layer, 0, 1) ** 2)
    # operator A f = sum_j c_j f_{r+j} + (k dx)^2 f + i nu f: with this sign e^{+i k_g r} is the
    # rightward wave, absorbed in the end layers (periodic wrap-around suppressed)
    A = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(-m, m + 1):
            A[i, (i + j) % n] += c[j + m]
        A[i, i] += k_dx**2 + 1j * nu[i]
    # source S = D[H V] - H D[V]: A E = -S has E = H V when V is an exact discrete wave
    rhs = np.zeros(n, dtype=complex)
    for rr, pairs in stencils.injector_weights(order, +1).items():
        for j, w in pairs:
            rhs[i_plane + rr] += w * np.exp(1j * k_dx * (i_plane + rr + j))
    E = np.linalg.solve(A, -rhs)
    below = np.abs(E[layer + 20 : i_plane - 2 * m])
    above = E[i_plane + 2 * m + 5 : n - layer - 20]
    amp = np.abs(above)
    assert amp.std() / amp.mean() < 1e-4  # one wave, no standing-wave ripple
    theta = np.angle(above[1:] / above[:-1])
    np.testing.assert_allclose(theta.mean(), stencils.grid_wavenumber(k_dx, order), rtol=1e-6)
    predicted = stencils.launched_amplitude_ratio(k_dx, order)
    np.testing.assert_allclose(amp.mean(), predicted, rtol=1e-5)
    # the dispersion deficit and the backward leak both fall with the order
    assert abs(1.0 - predicted) < {2: 0.03, 4: 0.003, 6: 0.0003}[order]
    assert below.max() / amp.mean() < {2: 0.03, 4: 0.003, 6: 0.0003}[order]


def _pump_cfg(order):
    cfg = deepcopy(_load_cfg())
    cfg["grid"]["ymax"] = "0.02um"
    cfg["grid"]["ymin"] = "-0.02um"
    cfg["terms"]["light"] = {"pump_depletion": True, "fd_order": order}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["grid"]["xmax"] = "20um"
    cfg["grid"]["tmax"] = "0.3ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.3ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.05ps"
    cfg["mlflow"]["run"] = f"fd-order-{order}-injector"
    return cfg


@pytest.mark.parametrize("order", [4, 6])
def test_higher_order_pump_injector_launches_the_nominal_amplitude(order):
    """The evolved pump at order 4 / 6 fills the box with the nominal (swelled) amplitude and
    the physical wavenumber to a few 1e-3 -- the second-order run of test_srs.py carries the
    compact stencil's 2-3 % dispersion deficit at 8 cells per wavelength."""
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(_pump_cfg(order))
    sol, _, _ = exo(modules)
    result = sol["solver result"]
    dcfg = exo.adept_module.cfg
    derived = dcfg["units"]["derived"]
    n = dcfg["units"]["envelope density"]
    x = np.array(dcfg["grid"]["x"])
    e0_raw = np.array(result.ys["fields"]["E0"])
    e0 = e0_raw.view(np.complex64 if e0_raw.dtype == np.float32 else np.complex128)
    e0y = e0[-1, :, 0, 1]
    bulk = slice(np.argmin(np.abs(x - 8.0)), np.argmin(np.abs(x - 14.0)))
    # the single colour sits at the driver's delta_omega (-0.015 here): the injected wavenumber
    # is k0(n, delta_omega) as calc_pump_source launches it
    from adept._lpse2d.modules.driver import UniformDriver

    _, args = UniformDriver(dcfg)({}, {"drivers": {}})
    dw = float(np.asarray(args["drivers"]["E0"]["delta_omega"])[0])
    k0_dx = derived["w0"] / derived["c"] * np.sqrt((1.0 + dw) ** 2 - n) * dcfg["grid"]["dx"]
    ratio = stencils.launched_amplitude_ratio(k0_dx, order)
    assert abs(1.0 - ratio) < 3e-3
    expected_amp = derived["E0_source"] * (1.0 - n) ** -0.25
    np.testing.assert_allclose(np.mean(np.abs(e0y[bulk])), expected_amp * ratio, rtol=5e-3)
    # phase advance per cell: the stencil's grid wavenumber, within 0.3 % of k0 dx
    theta = np.angle(e0y[bulk][1:] / e0y[bulk][:-1])
    np.testing.assert_allclose(np.mean(theta), stencils.grid_wavenumber(k0_dx, order), rtol=1e-3)
    assert abs(np.mean(theta) / k0_dx - 1.0) < 3e-3
    # the probes see the nominal flux (their dispersion correction uses the stencil's k_g at
    # delta_omega = 0, hence the same 2 % tolerance as the second-order test in test_srs.py)
    t = np.array(result.ts["default"])
    steady = t > 0.2
    incident = np.array(result.ys["default"]["incident_flux"])
    transmitted = np.array(result.ys["default"]["transmitted_flux"])
    np.testing.assert_allclose(np.mean(incident[steady]), ratio**2, rtol=2e-2)
    np.testing.assert_allclose(np.mean(transmitted[steady]), np.mean(incident[steady]), rtol=1e-2)
    # nothing is launched behind the injector (scattered-field side; the rows sit at
    # xmin + 2 boundary_width = 6 um, the absorber's skirt ends near 4.5 um)
    i0 = int(np.argmin(np.abs(x - 6.0)))
    behind = slice(np.argmin(np.abs(x - 4.8)), i0 - order // 2 - 1)
    assert np.max(np.abs(e0y[behind])) < 0.02 * expected_amp


def test_higher_order_tightens_the_light_dt_limit():
    from adept._lpse2d.helpers import get_derived_quantities, get_solver_quantities, write_units

    subs = {}
    for order in (2, 4, 6):
        cfg = _pump_cfg(order)
        cfg["grid"]["dt"] = "0.5fs"
        write_units(cfg)
        cfg = get_derived_quantities(cfg)
        subs[order] = get_solver_quantities(cfg)["light_substeps"]
    assert subs[2] <= subs[4] <= subs[6]
    assert subs[6] > subs[2]


def test_injector_source_mask_covers_the_stencil_rows():
    from adept._lpse2d.helpers import _injector_rows

    x = np.arange(40, dtype=float)
    for order, expected in ((2, {10, 11}), (4, {9, 10, 11, 12}), (6, {8, 9, 10, 11, 12, 13})):
        rows = _injector_rows(x, 10.0, None, 1.0, {"solver": "fd", "fd_order": order})
        assert set(np.flatnonzero(rows == 0.0)) == expected


@pytest.mark.skipif(deck_path("test_016") is None, reason="no LPSE decks present")
def test_translator_maps_solver_order():
    """test_016 (1-D SRS absolute threshold, laser.solver = fd, solverOrder = 4). test_022's
    ``solverOrder = 4`` is on the combined solver, which adept runs spectrally, so it is not
    mapped there."""
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    parms = parse_parms(deck_path("test_016"))
    assert parms["laser.evolution.solverOrder"].strip() == "4"
    cfg, report = translate_parms(parms, run="test_016")
    assert cfg["terms"]["light"]["solver"] == "fd" and cfg["terms"]["light"]["fd_order"] == 4
    assert not any("solverOrder" in u for u in report["unsupported"])
    # an inactive Raman field's order is ignored; an active one with a different order is noted
    parms["raman.evolution.solverOrder"] = "6"
    cfg, report = translate_parms(parms, run="test_016")
    assert cfg["terms"]["light"]["fd_order"] == 4 and not any("solverOrder" in n for n in report["notes"])
    parms.update({"raman.enable": "true", "raman.solver": "fd"})
    cfg, report = translate_parms(parms, run="test_016")
    assert cfg["terms"]["light"]["fd_order"] == 6
    assert any("solverOrder differ" in n for n in report["notes"])
    parms["raman.evolution.solverOrder"] = "8"
    cfg, report = translate_parms(parms, run="test_016")
    assert "fd_order" not in cfg["terms"]["light"] and any("solverOrder = 8" in u for u in report["unsupported"])
