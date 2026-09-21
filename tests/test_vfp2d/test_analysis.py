"""Dimensional, signed-flow and Faraday checks for reconnection analysis."""

import numpy as np
import pytest
import xarray as xr
from scipy.constants import e, mu_0

from adept.vfp2d.analysis import MagpieReference
from adept.vfp2d.plotting import add_reconnection_diagnostics, reconnection_metrics


def _sheet_dataset(nt=1):
    x = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    y = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    b = np.zeros((nt, x.size, y.size, 3))
    b[..., 0] = np.tanh(3.0 * np.sin(yy))
    b[..., 1] = 0.25 * np.sin(xx)
    current = np.zeros_like(b)
    current[..., 2] = 0.25 * np.cos(xx) - 3.0 * np.cos(yy) / np.cosh(3.0 * np.sin(yy)) ** 2
    electric = np.zeros_like(b)
    electric[..., 2] = 0.1
    velocity = np.zeros_like(b)
    velocity[..., 1] = -2.0 * np.sign(yy)
    velocity[..., 0] = 3.0 * np.sign(xx)
    ions = np.zeros((*b.shape[:-1], 5))
    ions[..., 0] = 4.0
    scalars = np.ones(b.shape[:-1])
    return xr.Dataset(
        {
            "b": (("t", "x", "y", "component"), b),
            "current": (("t", "x", "y", "component"), current),
            "e": (("t", "x", "y", "component"), electric),
            "v_nernst": (("t", "x", "y", "component"), np.zeros_like(b)),
            "ion_velocity": (("t", "x", "y", "component"), velocity),
            "ions": (("t", "x", "y", "ion_conserved"), ions),
            "ne": (("t", "x", "y"), scalars * 2.0),
            "temperature": (("t", "x", "y"), scalars * 5.0),
            "ion_pressure": (("t", "x", "y"), scalars * 0.5),
        },
        coords={
            "t": np.linspace(0.0, 0.2, nt),
            "x": x,
            "y": y,
            "component": ["x", "y", "z"],
            "ion_conserved": ["rho", "rho_ux", "rho_uy", "rho_uz", "energy"],
        },
        attrs={"light_speed_normalized": 3.0, "temperature_energy_normalized": 0.2},
    )


def test_bulk_and_alfven_rates_work_with_zero_nernst_and_correct_magnetic_units():
    ds = add_reconnection_diagnostics(_sheet_dataset())
    assert ds.reconnection_valid.item()
    assert not ds.rate_normalization_valid.item()
    assert np.isnan(ds.normalized_reconnection_rate.item())
    assert ds.bulk_rate_normalization_valid.item()
    assert ds.alfven_rate_normalization_valid.item()
    b_up = np.tanh(3.0)
    va = 3.0 * b_up / 2.0
    np.testing.assert_allclose(ds.upstream_ion_inflow_y, 2.0)
    np.testing.assert_allclose(ds.upstream_alfven_speed, va)
    np.testing.assert_allclose(ds.normalized_reconnection_rate_bulk, 0.1 / (b_up * 2.0))
    np.testing.assert_allclose(ds.normalized_reconnection_rate_alfven, 0.1 / (b_up * va))
    np.testing.assert_allclose(ds.upstream_magnetic_pressure, 0.5 * 9.0 * b_up**2)
    np.testing.assert_allclose(ds.upstream_ram_pressure, 16.0)
    np.testing.assert_allclose(ds.upstream_thermal_pressure, 2.5)
    np.testing.assert_allclose(ds.centerline_ion_outflow_speed, 3.0)
    np.testing.assert_allclose(ds.upstream_bulk_flux_inflow, 2.0 * b_up)
    np.testing.assert_allclose(ds.upstream_electric_flux_inflow, -0.1)
    metrics = reconnection_metrics(ds)
    assert metrics["vfp2d_alfven_rate_valid_fraction"] == 1.0
    assert "vfp2d_peak_abs_alfven_reconnection_rate" in metrics


def test_one_sided_outward_flow_is_signed_and_disables_bulk_normalization():
    ds = _sheet_dataset()
    ds.ion_velocity.loc[{"component": "y"}] = 2.0
    ds = add_reconnection_diagnostics(ds)
    np.testing.assert_allclose(ds.upstream_ion_inflow_y, 0.0)
    np.testing.assert_allclose(ds.upstream_ion_inflow_y_upper, -2.0)
    assert not ds.bulk_rate_normalization_valid.item()
    assert np.isnan(ds.normalized_reconnection_rate_bulk.item())
    assert ds.alfven_rate_normalization_valid.item()


def test_topology_gate_still_rejects_an_unreversed_field():
    ds = _sheet_dataset()
    ds.b.loc[{"component": "x"}] = 1.0
    ds = add_reconnection_diagnostics(ds)
    assert not ds.reconnection_valid.item()
    assert not ds.alfven_rate_normalization_valid.item()
    assert not ds.bulk_rate_normalization_valid.item()
    assert np.isnan(ds.normalized_reconnection_rate_alfven.item())


def test_missing_or_invalid_normalization_metadata_cannot_silently_assume_c_equals_one():
    ds = _sheet_dataset()
    ds.attrs = {}
    ds = add_reconnection_diagnostics(ds)
    assert np.isnan(ds.upstream_alfven_speed.item())
    assert np.isnan(ds.upstream_magnetic_pressure.item())
    assert np.isnan(ds.upstream_thermal_pressure.item())
    assert not ds.alfven_rate_normalization_valid.item()
    assert ds.bulk_rate_normalization_valid.item()
    ds = _sheet_dataset()
    ds.attrs["light_speed_normalized"] = -3.0
    assert not add_reconnection_diagnostics(ds).alfven_rate_normalization_valid.item()


def test_asymmetric_sides_use_mean_of_b_times_alfven_not_product_of_means():
    ds = _sheet_dataset()
    upper = ds.y > 0.0
    ds.b.loc[{"component": "x"}] = ds.b.sel(component="x") * xr.where(upper, 2.0, 1.0)
    ds.ions.loc[{"ion_conserved": "rho"}] = xr.where(upper, 9.0, 4.0)
    diagnosed = add_reconnection_diagnostics(ds)
    # Feed a prescribed valid topology to isolate asymmetric normalization from
    # the deliberately stronger discontinuous field profile's center null score.
    from adept.vfp2d.analysis import add_flow_diagnostics

    diagnosed["reconnection_valid"][:] = True
    lower = np.array([np.argmin(np.abs(np.asarray(ds.y) + np.pi / 2.0))])
    upper = np.array([np.argmin(np.abs(np.asarray(ds.y) - np.pi / 2.0))])
    ix0 = int(np.argmin(np.abs(np.asarray(ds.x))))
    iy0 = int(np.argmin(np.abs(np.asarray(ds.y))))
    diagnosed = add_flow_diagnostics(diagnosed, ix0=ix0, iy0=iy0, lower_iy=lower, upper_iy=upper)
    b = np.tanh(3.0)
    np.testing.assert_allclose(diagnosed.normalized_reconnection_rate_alfven, 0.1 / (2.75 * b**2))


def test_fixed_line_flux_budget_obeys_faraday_with_mean_magnetic_flux():
    ds = _sheet_dataset(nt=3)
    for it, t in enumerate(ds.t):
        ds.b[it, :, :, 0] -= float(t)
    ds.e.loc[{"component": "z"}] = ds.y
    ds = add_reconnection_diagnostics(ds)
    np.testing.assert_allclose(ds.centerline_lower_faraday_residual, 0.0, atol=2e-14)
    np.testing.assert_allclose(ds.centerline_upper_faraday_residual, 0.0, atol=2e-14)
    np.testing.assert_allclose(ds.centerline_lower_faraday_rate, -np.pi)
    np.testing.assert_allclose(ds.centerline_upper_faraday_rate, -float(ds.y[-1]))
    single = add_reconnection_diagnostics(_sheet_dataset())
    assert np.isnan(single.centerline_lower_faraday_residual.item())


def test_harris_gradient_width_converges_and_is_not_rms_width():
    ds = add_reconnection_diagnostics(_sheet_dataset())
    np.testing.assert_allclose(ds.current_sheet_gradient_half_width, 1.0 / 3.0, rtol=0.04)
    assert ds.current_sheet_gradient_half_width.item() != ds.current_sheet_rms_width.item()


def test_magpie_reference_scales_match_si_and_documented_regime():
    reference = MagpieReference()
    scales = reference.scales()
    assert 68e3 < scales["alfven_speed_m_s"] < 71e3
    assert 0.7 < scales["alfven_mach"] < 0.75
    assert 1.0 < scales["dynamic_beta"] < 1.1
    assert 0.35 < scales["thermal_beta"] < 0.4
    assert 0.7e-3 < scales["ion_inertial_length_m"] < 0.73e-3
    np.testing.assert_allclose(scales["inflow_crossing_time_s"], 12e-9)
    np.testing.assert_allclose(scales["dynamic_beta"], 2.0 * scales["alfven_mach"] ** 2)
    np.testing.assert_allclose(scales["thermal_pressure_pa"], (3e23 * 15.0 + 7.5e22 * 50.0) * e)
    np.testing.assert_allclose(scales["magnetic_pressure_pa"], 9.0 / (2.0 * mu_0))


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_magpie_reference_rejects_unphysical_inputs(value):
    with pytest.raises(ValueError, match="mean_charge"):
        MagpieReference(mean_charge=value).scales()
