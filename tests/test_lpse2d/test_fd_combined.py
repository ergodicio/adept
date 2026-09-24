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


def _cfg():
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
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
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
