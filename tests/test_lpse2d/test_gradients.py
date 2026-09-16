"""Reverse-mode gradients through every lpse2d solver path (plan 2, N.5).

Differentiability is why these solvers exist in adept, and none of the paths the parity
branch added was exercised by a gradient. Each case below runs a short (64 x 16 cells,
``N_STEPS`` steps) box through ``SplitStep`` and differentiates the final EPW energy with
respect to the pump intensity multiplier -- the ``args["drivers"]["E0"]["intensities"]``
array that ``UniformDriver`` exposes as its trainable control -- with ``jax.grad``, and
checks it against a central finite difference of the same jitted function. The pump enters
as ``E0_source * sqrt(intensity)``, so the sensitivity is order one in every case.

Paths (``terms.epw.solver`` / ``terms.light.solver`` and extras):

- ``separate/fd`` and ``separate/spectral``: SRS with pump depletion, the two coupled
  light solvers;
- ``separate/static``: the prescribed (non-evolved) pump, the MATLAB path;
- ``combined/spectral``: TPD + SRS on the wp0-enveloped combined field, pump depletion;
- ``combined/spectral+iaw``: plus the spectral ion-acoustic solver;
- ``separate/spectral+thermal-noise``: plus the per-step thermal noise source (a fixed
  seed; the kicks do not depend on the intensity, the response does);
- ``separate/spectral+hpe``: the hybrid particle push and its Landau feedback.

The EPW is seeded with a small band-limited random potential so the parametric SRS / TPD
response is non-zero from the first step.
"""

from copy import deepcopy

import jax
import numpy as np
import pytest
import yaml
from jax import lax
from jax import numpy as jnp

N_STEPS = 50
FD_STEP = 1.0e-3  # relative central-difference step on the intensity multiplier
RTOL = 1.0e-2  # plan 2 N.5


def _finish(cfg):
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _cfg(
    *,
    epw_solver="separate",
    light_solver="spectral",
    pump_depletion=True,
    tpd=False,
    srs=True,
    noise=False,
    iaw=False,
    hpe=False,
    density=0.2,
):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": density}
    cfg["units"]["envelope density"] = 0.25 if tpd else density
    # pump fully on from t = 0 (srs.yaml's envelope would still be on its rising edge)
    cfg["drivers"]["E0"]["envelope"]["tc"] = "19.95ps"
    # 64 x 16 cells of 40 nm: the retained band reaches k lambda_D ~ 0.37 at n = 0.2 n_c, so the
    # SRS-resonant EPW (k ~ 32/um, k lambda_D ~ 0.25) is resolved and Landau damped
    cfg["grid"].update(
        {
            "boundary_width": "0.4um",
            "dt": "1fs",
            "dx": "0.04um",
            "xmax": "2.56um",
            "tmax": "0.1ps",
            "ymax": "0.32um",
            "ymin": "-0.32um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["density_gradient"] = True
    cfg["terms"]["epw"]["solver"] = epw_solver
    cfg["terms"]["epw"]["source"].update(
        {"noise": noise, "noise_model": "thermal", "noise_seed": 5, "noise_amplitude": 1.0e-12, "tpd": tpd, "srs": srs}
    )  # 1e-12: the noise-driven EPW energy after N_STEPS is ~2x the seeded one, not 1e16x
    cfg["terms"]["light"] = {"solver": light_solver, "pump_depletion": pump_depletion}
    if iaw:
        cfg["terms"]["iaw"] = {"active": True, "solver": "spectral"}
    if hpe:
        cfg["terms"]["hpe"] = {"active": True, "n_particles": 4000, "seed": 3, "n_angles": 8, "gather_refine": 1}
    return _finish(cfg)


def _seed_state(cfg, seed=11, relative_amplitude=1.0e-3):
    """Zero light, a random band-limited EPW potential (k-space, as the state stores it)
    whose field peaks at ``relative_amplitude`` x the pump amplitude ``E0_source``."""
    from adept._lpse2d.core.hpe import load_particles

    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k *= np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    field = np.stack([np.fft.ifft2(-1j * kx * phi_k), np.fft.ifft2(-1j * ky * phi_k)], -1)
    scale = relative_amplitude * cfg["units"]["derived"]["E0_source"] / np.abs(field).max()
    phi_k, field = phi_k * scale, field * scale
    state = {
        "epw": jnp.asarray(phi_k),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
    }
    if cfg["terms"]["epw"].get("solver") == "combined":
        # the combined field carries the EPW as its longitudinal part: E1 = -grad phi
        state["E1"] = jnp.asarray(field)
    if cfg["terms"].get("iaw", {}).get("active", False):
        state["iaw_density"] = jnp.zeros((nx, ny))
        state["iaw_velocity_divergence"] = jnp.zeros((nx, ny))
    if cfg["terms"].get("hpe", {}).get("active", False):
        state = state | {k: jnp.asarray(v) for k, v in load_particles(cfg).items()}
    return {k: v.view(jnp.float64) if jnp.iscomplexobj(v) else v for k, v in state.items()}


def _objective(cfg, n_steps=N_STEPS):
    """``f(s)``: final EPW energy after ``n_steps`` with the pump intensity multiplied by ``s``."""
    from adept._lpse2d.core.vector_field import SplitStep

    step = SplitStep(cfg)
    state0 = _seed_state(cfg)
    ny = cfg["grid"]["ny"]
    dt = cfg["grid"]["dt"]
    pump = {**cfg["drivers"]["E0"]["derived"], "delta_omega": jnp.zeros(1), "phases": jnp.zeros((1, ny))}

    def f(s):
        args = {"drivers": {"E0": {**pump, "intensities": s * jnp.ones((1, ny))}}}

        def body(i, state):
            return step(i * dt, state, args)

        state = lax.fori_loop(0, n_steps, body, dict(state0))
        return step.epw.energy(state["epw"].view(jnp.complex128))

    return jax.jit(f)


def _check_gradient(cfg, s0=1.0):
    f = _objective(cfg)
    value = float(f(s0))
    assert np.isfinite(value) and value > 0.0
    grad = float(jax.grad(f)(s0))
    h = FD_STEP * s0
    fd = (float(f(s0 + h)) - float(f(s0 - h))) / (2.0 * h)
    # the objective must respond to the control for the comparison to mean anything
    assert abs(fd) * s0 > 1.0e-6 * value, (value, fd)
    assert np.isfinite(grad)
    assert grad == pytest.approx(fd, rel=RTOL), (grad, fd, value)
    return value, grad, fd


PATHS = {
    "separate/fd": dict(light_solver="fd"),
    "separate/spectral": dict(light_solver="spectral"),
    "separate/static": dict(light_solver="fd", pump_depletion=False),
    "combined/spectral": dict(epw_solver="combined", tpd=True, srs=True, density=0.22),
    "combined/spectral+iaw": dict(epw_solver="combined", tpd=True, srs=True, density=0.22, iaw=True),
    "separate/spectral+thermal-noise": dict(light_solver="spectral", noise=True),
}


@pytest.mark.parametrize("path", list(PATHS))
def test_gradient_wrt_pump_intensity_matches_finite_difference(path):
    _check_gradient(_cfg(**PATHS[path]))


def test_gradient_through_the_hpe_feedback():
    """With HPE on, the Landau rate the EPW step applies is the one the particle histogram
    produces, so the objective differs from the analytic-damping run (the feedback is in the
    graph). Its derivative with respect to the fields is identically zero -- ``jnp.histogram``
    and the wall / energy bins are piecewise constant -- so the gradient is the one with
    ``gamma_L`` frozen at the value the push produced, and the push itself must still be
    differentiable (no ``stop_gradient`` is needed, none is used)."""
    cfg = _cfg(light_solver="spectral", hpe=True)
    reference = float(_objective(_cfg(light_solver="spectral"))(1.0))
    value, grad, fd = _check_gradient(cfg)
    assert abs(value - reference) > 1.0e-3 * reference, (value, reference)
