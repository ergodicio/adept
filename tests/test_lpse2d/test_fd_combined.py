"""The combined solver on the FD light propagator (inventory A15; LPSE lw.solver = combined with
{laser|raman}.solver = fd: SchrodingerSolver3::step_2d with the Bohm-Gross grad-div coefficient,
LightSolver::calculateSources with FD derivatives, LwSolver::advanceLW_combinedTPDandSRS_FD).

Each check compares with a closed form of the discrete operator (the stencil's symbol), so the
tolerances are round-off (1e-10 relative), fixed in advance.
"""

from copy import deepcopy

import numpy as np
import yaml
from jax import numpy as jnp


def _cfg(source_window=None, srs_k_filter=False, sources=True):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["density"] = {"basis": "uniform", "val": 0.2}
    cfg["units"]["envelope density"] = 0.2
    cfg["grid"].update(
        {
            "boundary_width": "0.6um",
            "dt": "1fs",
            "dx": "0.1um",
            "xmax": "6.4um",
            "tmax": "10fs",
            "ymax": "0.05um",
            "ymin": "-0.05um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["density_gradient"] = True
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": sources, "srs": sources, "srs_k_filter": srs_k_filter})
    if source_window is not None:
        cfg["terms"]["epw"]["source_window"] = source_window
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "fd", "pump_depletion": True}
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _solver():
    from adept._lpse2d.core.fd_combined import FDCombinedSolver

    cfg = _cfg()
    return cfg, FDCombinedSolver(cfg)


def _wave(cfg, m, component):
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    x = np.asarray(cfg["grid"]["x"])
    k = 2.0 * np.pi * m / (nx * cfg["grid"]["dx"])
    field = np.zeros((nx, ny, 3), dtype=np.complex128)
    field[..., component] = np.exp(1j * k * x)[:, None]
    return k, jnp.asarray(field)


def _second_symbol(solver, k):
    """-(second-derivative stencil applied to exp(i k x)) / exp(i k x), i.e. the discrete k^2."""
    h = solver.dx
    return -sum(w * np.cos(j * k * h) for j, w in solver._second) / h**2


def _first_symbol(solver, k):
    """(first-derivative stencil applied to exp(i k x)) / (i exp(i k x)), the discrete k."""
    h = solver.dx
    return sum(w * np.sin(j * k * h) for j, w in solver._first) / h


def test_combined_field_operator_is_light_transverse_and_bohm_gross_longitudinal():
    cfg, solver = _solver()
    solver.sources_on = False
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E0 = jnp.zeros((nx, ny, 3), dtype=jnp.complex128)
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.zeros((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    derived = cfg["units"]["derived"]
    for m in (3, 11):
        # E_y(x): transverse -> the light's c^2 k_h^2 / (2 wp0); E_x(x): longitudinal -> 3 vte^2 k_h^2 / (2 wp0)
        k, e_t = _wave(cfg, m, 1)
        _, k_e1 = solver.coupled_rhs(0.0, E0, e_t, None, pump_args, None)
        k2 = _second_symbol(solver, k)
        want = -1j * derived["c"] ** 2 / (2.0 * derived["wp0"]) * k2 * np.asarray(e_t)
        np.testing.assert_allclose(np.asarray(k_e1), want, rtol=1e-10, atol=1e-10 * np.abs(want).max())
        _, e_l = _wave(cfg, m, 0)
        _, k_e1 = solver.coupled_rhs(0.0, E0, e_l, None, pump_args, None)
        want = -1j * 3.0 * derived["vte_sq"] / (2.0 * derived["wp0"]) * k2 * np.asarray(e_l)
        np.testing.assert_allclose(np.asarray(k_e1), want, rtol=1e-10, atol=1e-10 * np.abs(want).max())


def test_unified_couplings_use_the_fd_derivatives():
    """E0 = y e^{i k0 x}, E1 = x e^{i k1 x}: grad(E0 . E1*) = 0 (orthogonal), div E1 = i k1_h E1_x, so the
    source is -i e/(4 me w0) (1 - w0/wp0) E0 (i k1_h)^* e^{-i k1 x} and the depletion
    i e/(2 me w0) E1 (i k1_h e^{i k1 x}); with E1 along y instead, only the gradient term
    i (k0 - k1)_h (E0 . E1*) survives."""
    cfg, solver = _solver()
    k0, e0 = _wave(cfg, 7, 1)
    k1, e1_long = _wave(cfg, 3, 0)
    d = cfg["units"]["derived"]
    k1h = _first_symbol(solver, k1)
    src = np.asarray(solver.unified_source_fd(0.0, e0, e1_long))
    want = -1j * d["e"] / (4.0 * d["me"] * d["w0"]) * (1.0 - d["w0"] / d["wp0"]) * np.asarray(e0)
    want = want * np.conj(1j * k1h * np.asarray(e1_long)[..., 0])[..., None]
    np.testing.assert_allclose(src, want, rtol=1e-10, atol=1e-10 * np.abs(want).max())
    dep = np.asarray(solver.unified_depletion_fd(0.0, e1_long))
    want = (
        1j
        * d["e"]
        / (2.0 * d["me"] * d["w0"])
        * np.asarray(e1_long)
        * (1j * k1h * np.asarray(e1_long)[..., 0])[..., None]
    )
    np.testing.assert_allclose(dep, want, rtol=1e-10, atol=1e-10 * np.abs(want).max())
    _, e1_tr = _wave(cfg, 3, 1)
    src = np.asarray(solver.unified_source_fd(0.0, e0, e1_tr))
    kd = _first_symbol(solver, k0 - k1)
    scalar = np.sum(np.asarray(e0) * np.conj(np.asarray(e1_tr)), axis=-1)
    want = np.zeros_like(src)
    want[..., 0] = -1j * d["e"] / (4.0 * d["me"] * d["w0"]) * 1j * kd * scalar
    np.testing.assert_allclose(src, want, rtol=1e-10, atol=1e-10 * np.abs(want).max())


def test_fd_unified_source_takes_the_epw_window_and_the_pump_k_filter():
    """terms.epw.source_window and terms.epw.source.srs_k_filter act on the FD combined unified source
    as on CombinedSolver.unified_source (PR #378 review; before the fix the FD path read neither): a
    window outside the box leaves no drive in the coupled RHS, a partial window multiplies the source
    by the EPW mask, and the filter removes the pump's modes above its cutoff before the product.
    Identities on random fields, 1e-12, fixed in advance."""
    from adept._lpse2d.core.fd_combined import FDCombinedSolver
    from adept._lpse2d.core.vector import fft2c, ifft2c

    cfg, base = _solver()
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(11)
    E0 = jnp.asarray(rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3)))
    E1 = jnp.asarray(rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3)))
    t = 0.37 * cfg["grid"]["dt"]
    src = np.asarray(base.unified_source_fd(t, E0, E1))
    assert base.epw_source_mask is None and base.E0_filter is None

    partial = _cfg(source_window={"width": ["2um", "0um"], "center": ["3.2um", "0um"], "edge_width": "0.5um"})
    mask = np.asarray(partial["grid"]["epw_source_mask"])
    assert 0.1 < mask.mean() < 0.9 and np.any(mask == 0.0) and np.any((mask > 0.0) & (mask < 1.0))
    got = np.asarray(FDCombinedSolver(partial).unified_source_fd(t, E0, E1))
    np.testing.assert_allclose(got, mask[..., None] * src, rtol=1e-12, atol=1e-12 * np.abs(src).max())

    outside = _cfg(source_window={"width": ["1um", "0um"], "center": ["50um", "0um"], "edge_width": "0um"})
    assert not np.any(np.asarray(outside["grid"]["epw_source_mask"]))
    solver = FDCombinedSolver(outside)
    assert float(jnp.max(jnp.abs(solver.unified_source_fd(t, E0, E1)))) == 0.0
    pump_args = {
        **outside["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    _, k_windowed = solver.coupled_rhs(t, E0, E1, None, pump_args, None)
    solver.sources_on = False
    _, k_no_source = solver.coupled_rhs(t, E0, E1, None, pump_args, None)
    np.testing.assert_array_equal(np.asarray(k_windowed), np.asarray(k_no_source))

    filtered = FDCombinedSolver(_cfg(srs_k_filter=True))
    keep = np.asarray(filtered.E0_filter)
    assert 0.0 < keep.mean() < 1.0
    E0f = ifft2c(fft2c(E0) * jnp.asarray(keep)[..., None])
    want = np.asarray(base.unified_source_fd(t, E0f, E1))
    got = np.asarray(filtered.unified_source_fd(t, E0, E1))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_langmuir_step_damps_only_the_longitudinal_part():
    """LwSolver::advanceLW_combinedTPDandSRS_FD: E_L -= E_L (1 - exp(-gamma dt)) on the band; the
    transverse part is untouched (1e-12)."""
    cfg, solver = _solver()
    k, e_l = _wave(cfg, 5, 0)
    _, e_t = _wave(cfg, 5, 1)
    gamma = jnp.full(solver.spectral.k_sq.shape, 50.0)
    out_l = np.asarray(solver.langmuir_step(0.0, e_l, gamma))
    out_t = np.asarray(solver.langmuir_step(0.0, e_t, gamma))
    np.testing.assert_allclose(out_l, np.exp(-50.0 * solver.dt) * np.asarray(e_l), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(out_t, np.asarray(e_t), rtol=1e-12, atol=1e-14)


def test_fd_combined_light_step_respects_the_combined_field_bound_with_the_sources_off():
    """The FD combined solver evolves E1 (light operator + Bohm-Gross grad-div) with TPD and SRS both
    off too, so setup must sub-cycle the light to the combined field's stability bound then as well
    (PR #378 review; before the fix the bound entered only with SRS on and the pump's alone set the
    sub-step). Setup: the sources-off sub-step count equals the sources-on one, whose bound includes
    the combined field's. Propagation, criterion fixed before the run: full-spectrum noise in E1 with
    the pump off stays bounded, max |E1| after 20 EPW steps <= 2 x its initial value."""
    import jax

    from adept._lpse2d.core.vector_field import SplitStep

    on, off = _cfg(), _cfg(sources=False)
    assert off["grid"]["light_substeps"] == on["grid"]["light_substeps"]
    step = SplitStep(off)
    assert not step.combined.sources_on
    nx, ny = off["grid"]["nx"], off["grid"]["ny"]
    rng = np.random.default_rng(23)
    e1 = 1e-6 * (rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3)))
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 3), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.asarray(e1).view(jnp.float64),
    }
    pump_args = {
        **off["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.zeros((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    advance = jax.jit(lambda t, y: step(t, y, {"drivers": {"E0": pump_args}}))
    for i in range(20):
        state = advance(jnp.asarray(i * off["grid"]["dt"]), dict(state))
    e1_end = np.asarray(state["E1"].view(jnp.complex128))
    assert np.all(np.isfinite(e1_end))
    assert np.abs(e1_end).max() <= 2.0 * np.abs(e1).max()


def test_split_step_runs_the_fd_combined_solver():
    """Ten EPW steps of the FD combined solver inside the split step: the pump is launched by its
    FD injector, the fields stay finite and the stored potential is the combined field's."""
    from adept._lpse2d.core.fd_combined import FDCombinedSolver
    from adept._lpse2d.core.vector_field import SplitStep

    cfg = _cfg()
    step = SplitStep(cfg)
    assert isinstance(step.combined, FDCombinedSolver)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 3), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 3), dtype=jnp.complex128).view(jnp.float64),
    }
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    for i in range(10):
        state = step(jnp.asarray(i * cfg["grid"]["dt"]), dict(state), {"drivers": {"E0": pump_args}})
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in state.values())
    assert float(jnp.max(jnp.abs(state["E0"].view(jnp.complex128)))) > 0.0
    E1 = state["E1"].view(jnp.complex128)
    np.testing.assert_allclose(
        np.asarray(state["epw"].view(jnp.complex128)), np.asarray(step.combined.spectral.potential(E1)), rtol=1e-12
    )


def test_fd_combined_seed_injector_launches_the_seed_at_the_combined_carrier():
    """drivers.E1 seeds the FD combined field through the separate FD Raman solver's injector rows
    at the combined carrier wp0 (LPSE's Raman class injects at W1 = wpe in combined mode;
    PR #378 review). Pump off (zero beam intensity), noise off, subcritical injector at
    n = 0.15 nc, n_env = 0.2 (w1 = 1.24 wp0, so a w1-built injector misses the operator by ~20 %
    and launches at k1 = 0.88 instead of 0.50 wp0/c). Tolerances fixed before the run (k dx = 0.2,
    FD dispersion < 1 %): in the bulk |E1_y| = A / eps^(1/4), eps = 1 - n/n_env, to 3 %, ripple
    under 3 %, a leftward wave at k = wp0 sqrt(eps) / c to 2 %, and the transverse seed leaves the
    derived potential at zero. Before the fix the FD combined path dropped the seed (E1 = 0)."""
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["density"] = {"basis": "uniform", "val": 0.15}
    cfg["units"]["envelope density"] = 0.2
    cfg["grid"].update(
        {"boundary_width": "3um", "dt": "1fs", "dx": "0.05um", "xmax": "20um", "tmax": "0.2ps"}
        | {"ymax": "0.02um", "ymin": "-0.02um", "low_pass_filter": 0.6}
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "fd", "pump_depletion": True}
    cfg["drivers"]["E1"] = {"intensity": "1.0e+12W/cm^2", "delta_omega": 0.0, "turn_on_time": "10fs", "offset": "7um"}
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)

    step = SplitStep(cfg)
    assert step.combined.seed_enabled and step.combined.w_seed == cfg["units"]["derived"]["wp0"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 3), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 3), dtype=jnp.complex128).view(jnp.float64),
    }
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.zeros((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    args = {"drivers": {"E0": pump_args, "E1": cfg["drivers"]["E1"]["derived"]}}
    for i in range(180):
        state = step(jnp.asarray(i * cfg["grid"]["dt"]), dict(state), args)

    e1y = np.asarray(state["E1"].view(jnp.complex128))[:, 0, 1]
    x = np.asarray(cfg["grid"]["x"])
    derived = cfg["units"]["derived"]
    eps = 1.0 - 0.15 / 0.2
    k_seed = derived["wp0"] * np.sqrt(eps) / derived["c"]
    bulk = slice(int(np.argmin(np.abs(x - 5.0))), int(np.argmin(np.abs(x - 10.0))))
    expected = cfg["drivers"]["E1"]["derived"]["amplitude"] / eps**0.25
    # the pump stays off: only the depletion of the seed's round-off divergence reaches it
    assert float(jnp.max(jnp.abs(state["E0"].view(jnp.complex128)))) < 1e-12 * expected
    np.testing.assert_allclose(np.mean(np.abs(e1y[bulk])), expected, rtol=0.03)
    assert np.std(np.abs(e1y[bulk])) < 0.03 * expected
    k_measured = np.polyfit(x[bulk], np.unwrap(np.angle(e1y[bulk])), 1)[0]
    assert k_measured < 0
    np.testing.assert_allclose(abs(k_measured), k_seed, rtol=0.02)
    assert float(jnp.max(jnp.abs(state["epw"]))) < 1e-12 * expected
