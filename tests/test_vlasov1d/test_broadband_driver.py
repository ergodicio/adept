"""Unit tests for the broadband (multi-color) ey driver.

Architecture under test (see BroadbandDriver in simulation.py): ``EMDriver.from_config``
returns ONE ``BroadbandDriver`` eqx.Module carrying per-line array leaves
(``amplitudes``/``delta_omega``/``phases``, each (N,)); the transverse source pusher
reads drivers from ``args["drivers"].ey`` and evaluates all lines vectorized. No full
solve is run here.

Covered:
  * a uniform N-line comb carries the same time-averaged power as the monochromatic
    driver at ``base_intensity`` (``sum_j a_j^2 == a0^2``),
  * ``delta_omega`` is the comb HALF-width; ``num_colors: 1`` sits exactly at ``w0``
    and reproduces the monochromatic driver's source to machine precision at the
    ``TransverseCurrentSourceDriver`` level,
  * seeded line sets are reproducible across independent builds,
  * config validation (``init: random`` without a seed, missing ``base_intensity``),
  * gradients w.r.t. ``amplitudes`` and ``phases`` flow through the args route
    (regression guard: fails if a construction-time precompute or a disconnected
    driver copy is ever reintroduced).
"""

import math

import numpy as np
import pytest
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import equinox as eqx
import jax
from jax import numpy as jnp
from pydantic import ValidationError

from adept._vlasov1d.datamodel import EMDriverConfig
from adept._vlasov1d.simulation import BroadbandDriver, EMDriver, EMDriverSet
from adept._vlasov1d.solvers.pushers.field import TransverseCurrentSourceDriver
from adept.normalization import electron_debye_normalization

NORM = electron_debye_normalization("9.05e21/cc", "4000eV")
C_NORM = NORM.speed_of_light_norm()
BASE_INTENSITY = "2.378e+14 W/cm^2"
WAVELENGTH = "351nm"


def _envelope(source_type):
    # micron-scale antenna at 5 um for the point source; box-wide for extended
    width = "1.0um" if source_type == "point" else "50.0um"
    return {
        "time": {"center": "20500.0fs", "rise": "100.0fs", "width": "42000.0fs"},
        "space": {"center": "5.0um" if source_type == "point" else "25.0um", "rise": "0.1um", "width": width},
    }


def _mono(source_type="point"):
    cfg = EMDriverConfig(
        params={"intensity": BASE_INTENSITY, "wavelength": WAVELENGTH},
        envelope=_envelope(source_type),
        source_type=source_type,
    )
    (d,) = EMDriver.from_config(cfg, NORM)
    return d


def _comb(num_colors, delta_omega, intensities=None, phases=None, source_type="point"):
    cfg = EMDriverConfig(
        params={
            "num_colors": num_colors,
            "delta_omega": delta_omega,
            "wavelength": WAVELENGTH,
            "intensities": intensities or {"base_intensity": BASE_INTENSITY, "init": "uniform"},
            "phases": phases or {"init": "random", "seed": 1},
        },
        envelope=_envelope(source_type),
        source_type=source_type,
    )
    bb = EMDriver.from_config(cfg, NORM)
    assert isinstance(bb, BroadbandDriver)
    return bb


def _source(driver, t, xax):
    """Evaluate the transverse current source through the args route the solver uses."""
    pusher = TransverseCurrentSourceDriver(xax, drivers=[driver], c=C_NORM)
    return pusher(t, {"drivers": EMDriverSet(ex=[], ey=[driver])})


XAX = jnp.linspace(0.0, 50.0 * 213.0, 1024)  # ~50 um in Debye-length units, coarse


def test_uniform_comb_has_monochromatic_power():
    """sum_j a_j^2 == a0^2 for a uniform comb (a_j = a0 sqrt(w_j / sum w))."""
    mono = _mono()
    bb = _comb(50, 0.0025)
    assert bb.amplitudes.shape == (50,)
    assert np.allclose(np.asarray(bb.amplitudes), float(bb.amplitudes[0]))  # uniform weights
    assert math.isclose(float(jnp.sum(bb.amplitudes**2)), float(mono.a0) ** 2, rel_tol=1e-12)
    # the comb shares the monochromatic carrier
    assert float(bb.k0) == float(mono.k0) and float(bb.w0) == float(mono.w0)


def test_delta_omega_is_half_width():
    """delta_omega (N,) spans [-d, +d] * w0 uniformly (full width 2d)."""
    bb = _comb(3, 0.0025)
    rel = np.asarray(bb.delta_omega) / float(bb.w0)
    assert np.allclose(rel, [-0.0025, 0.0, 0.0025], atol=1e-15)
    bb = _comb(2, 0.0025)
    rel = np.asarray(bb.delta_omega) / float(bb.w0)
    assert np.allclose(rel, [-0.0025, 0.0025], atol=1e-15)


@pytest.mark.parametrize("source_type", ["point", "extended"])
def test_single_line_is_monochromatic(source_type):
    """num_colors: 1 IS the monochromatic driver: dw = 0, same a0, and the evaluated
    source matches the plain EMDriver's to machine precision at several times."""
    mono = _mono(source_type)
    bb = _comb(1, 0.0025, phases={"init": "uniform", "base_phase": 0.0}, source_type=source_type)
    assert bb.delta_omega.shape == (1,) and float(bb.delta_omega[0]) == 0.0  # not -delta_omega * w0
    assert float(bb.amplitudes[0]) == float(mono.a0)
    for t in (0.0, 313.7, 20500.0 * 1.88, 41000.0 * 1.88):  # spread across the envelope
        s_mono = _source(mono, t, XAX)
        s_bb = _source(bb, t, XAX)
        np.testing.assert_allclose(np.asarray(s_bb), np.asarray(s_mono), rtol=1e-12, atol=0.0)


def test_seeded_line_sets_are_reproducible():
    """Same seeds -> identical arrays across two builds; phase seed only moves phases."""
    kw = dict(
        intensities={"base_intensity": BASE_INTENSITY, "init": "random", "seed": 7},
        phases={"init": "random", "seed": 3},
    )
    a, b = _comb(20, 0.001, **kw), _comb(20, 0.001, **kw)
    np.testing.assert_array_equal(np.asarray(a.amplitudes), np.asarray(b.amplitudes))
    np.testing.assert_array_equal(np.asarray(a.phases), np.asarray(b.phases))
    np.testing.assert_array_equal(np.asarray(a.delta_omega), np.asarray(b.delta_omega))
    c = _comb(20, 0.001, intensities=kw["intensities"], phases={"init": "random", "seed": 4})
    np.testing.assert_array_equal(np.asarray(a.amplitudes), np.asarray(c.amplitudes))
    assert np.any(np.asarray(a.phases) != np.asarray(c.phases))
    # random weights still carry the monochromatic power
    assert math.isclose(float(jnp.sum(a.amplitudes**2)), float(_mono().a0) ** 2, rel_tol=1e-12)


def test_random_init_requires_seed():
    """init: random without a seed is a validation error, not a KeyError at build time."""
    with pytest.raises(ValidationError, match=r"phases\.seed"):
        _comb(4, 0.001, phases={"init": "random"})
    with pytest.raises(ValidationError, match=r"intensities\.seed"):
        _comb(4, 0.001, intensities={"base_intensity": BASE_INTENSITY, "init": "random"})
    with pytest.raises(ValidationError):  # base_intensity is required
        _comb(4, 0.001, intensities={"init": "uniform"})


@pytest.mark.parametrize("source_type", ["point", "extended"])
def test_gradients_flow_through_args_route(source_type):
    """d(source)/d(amplitudes) and d(source)/d(phases) are finite and nonzero.

    Regression guard for the differentiable-driver design: the pusher must read the
    line parameters from args["drivers"] and compute point-source scales in-trace.
    A construction-time precompute or a fallback to a disconnected self.drivers copy
    zeroes these gradients, failing this test."""
    bb = _comb(8, 0.01, source_type=source_type)
    pusher = TransverseCurrentSourceDriver(XAX, drivers=[bb], c=C_NORM)
    t = 20500.0 * 1.88  # mid-envelope so the time envelope is ~1

    def loss(amps, phases):
        d = eqx.tree_at(lambda m: (m.amplitudes, m.phases), bb, (amps, phases))
        return jnp.sum(pusher(t, {"drivers": EMDriverSet(ex=[], ey=[d])}) ** 2)

    g_amp, g_phase = jax.grad(loss, argnums=(0, 1))(bb.amplitudes, bb.phases)
    for g in (g_amp, g_phase):
        assert g.shape == (8,)
        assert np.all(np.isfinite(np.asarray(g)))
        assert np.max(np.abs(np.asarray(g))) > 0.0
