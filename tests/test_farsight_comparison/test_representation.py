"""Analytic checks for representation diagnostics, independent of native weights."""

import numpy as np
import pytest
import xarray as xr

from examples.farsight_comparison.representation import (
    integrate_rectangles,
    positivity_bounds,
    representation_integrals,
)


def test_exact_square_of_biquadratic():
    # On [0,2]^2, f=x²v² has mass64/9 and C2=1024/25.
    x, v = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    mass, c2 = integrate_rectangles((x**2 * v**2)[None], np.array([4.0]))
    assert mass == pytest.approx(64 / 9, rel=2e-15)
    assert c2 == pytest.approx(1024 / 25, rel=2e-15)


def test_signed_values_are_not_clipped():
    f = np.full((2, 3, 3), -2.0)
    f[1] = 3
    mass, c2 = integrate_rectangles(f, [0.25, 0.5])
    assert mass == pytest.approx(1)
    assert c2 == pytest.approx(5.5)


def test_reject_invalid_geometry():
    with pytest.raises(ValueError, match="positive areas"):
        integrate_rectangles(np.ones((1, 3, 3)), [-1])


def test_positive_nodes_do_not_certify_polynomial():
    # q(v)=2v²-v on [0,1] has nonnegative nodes but minimum -1/8.
    f = np.broadcast_to([0.0, 0.0, 1.0], (1, 3, 3))
    result = positivity_bounds(f)
    assert result["min_bernstein_coefficient"] == -0.5
    assert result["uncertified_panels"] == 1
    assert result["negative_mean_panels"] == 0
    assert result["min_panel_mean"] == pytest.approx(1 / 6)


def test_negative_bernstein_bound_is_not_proof_of_negative_f():
    # q(v)=(v-.5)² is nonnegative everywhere but has a negative Bernstein coefficient.
    result = positivity_bounds(np.broadcast_to([0.25, 0.0, 0.25], (1, 3, 3)))
    assert result["min_bernstein_coefficient"] == -0.25
    assert result["uncertified_panels"] == 1


def test_bernstein_certificate_and_scale_relative_tolerance():
    f = np.stack([np.ones((3, 3)), np.zeros((3, 3)), -np.full((3, 3), 1e-200)])
    result = positivity_bounds(f)
    assert result["active_panels"] == 3
    assert result["uncertified_panels"] == 1
    assert result["negative_mean_panels"] == 1
    with pytest.raises(ValueError, match="nonempty finite"):
        positivity_bounds(np.zeros((0, 3, 3)))


def test_fixed_representation_differs_from_nodal_trapezoid():
    # f(v)=1+v² on x[0,4],v[-1,1], independent of periodic x.
    x, v = np.meshgrid(np.linspace(0, 4, 5), np.linspace(-1, 1, 3), indexing="ij")
    f = 1 + v**2
    ds = xr.Dataset(
        {name: (("t", "x_node", "v_node"), value[None]) for name, value in (("x", x), ("v", v), ("f", f))},
        coords={"t": [0.0]},
    )
    config = {"grid": {"nx": 4, "nv": 2, "xmin": 0.0, "xmax": 4.0, "vmin": -1.0, "vmax": 1.0}}
    result = representation_integrals(ds, config)
    assert result["mass"] == pytest.approx([32 / 3])
    assert result["c2"] == pytest.approx([224 / 15])
    assert result["relative_c2"] == [0]
    ds["x"] = ds.x + 0.1
    with pytest.raises(ValueError, match="moving/deformed"):
        representation_integrals(ds, config)
