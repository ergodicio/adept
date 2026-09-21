"""Flow and flux diagnostics for a prescribed, centered reconnection geometry.

These functions sample the sheet center and upstream positions selected by
``plotting.add_reconnection_diagnostics``. They do not locate arbitrary X-points.
Fields and moments retain ADEPT's normalization; notably magnetic pressure is
``c_normalized**2 * B**2 / 2``, not just ``B**2 / 2``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.constants import atomic_mass, e, m_e, mu_0


def _ratio(numerator: np.ndarray, denominator: np.ndarray, valid=True) -> np.ndarray:
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=valid & np.isfinite(denominator) & (denominator > 0.0),
    )


def add_flow_diagnostics(
    ds: xr.Dataset, *, ix0: int, iy0: int, lower_iy: np.ndarray, upper_iy: np.ndarray
) -> xr.Dataset:
    """Add bulk-ion, Alfvén, pressure and fixed-line Faraday diagnostics.

    ``normalized_reconnection_rate`` remains the legacy Nernst normalization.
    New rate variables use independent validity masks and the same topology gate.
    Missing normalization metadata leaves the affected diagnostics NaN.
    """

    nt = ds.sizes["t"]
    time_indices = np.arange(nt)

    def sides(field):
        values = np.asarray(field)
        return np.stack((values[time_indices, ix0, lower_iy], values[time_indices, ix0, upper_iy]), axis=-1)

    side_sign = np.asarray([1.0, -1.0])
    bx = sides(ds.b.sel(component="x"))
    ez = sides(ds.e.sel(component="z"))
    # Signed flux transported towards the mid-plane; negative means flux leaves.
    electric_flux = np.mean(side_sign * np.sign(bx) * ez, axis=-1)
    data = {
        "upstream_y_lower": np.asarray(ds.y)[lower_iy],
        "upstream_y_upper": np.asarray(ds.y)[upper_iy],
        "upstream_electric_flux_inflow": electric_flux,
    }
    descriptions = {
        "upstream_electric_flux_inflow": (
            "mean(s E_z sign(B_x)), s=+1 below and -1 above; signed total flux transport into sheet; "
            "includes every Ohm term and is not a unique magnetic-field advection velocity"
        ),
    }

    # Fixed integration limits matter: differentiating a flux to a moving B-peak
    # would require an additional moving-boundary term.
    y = np.asarray(ds.y)
    bx_line = np.asarray(ds.b.sel(component="x"))[:, ix0, :]
    ez_line = np.asarray(ds.e.sel(component="z"))[:, ix0, :]
    center_gradient = np.gradient(bx_line, y, axis=-1)[:, iy0]
    data["current_sheet_gradient_half_width"] = _ratio(np.asarray(ds.upstream_bx), np.abs(center_gradient))
    descriptions["current_sheet_gradient_half_width"] = (
        "mean upstream |Bx| / |d_y Bx| at fixed sheet center; equals Harris delta only for a resolved Harris profile; "
        "ungated geometry diagnostic, not the current-weighted RMS width"
    )
    for label, indices, rhs in (
        ("lower", slice(None, iy0 + 1), ez_line[:, 0] - ez_line[:, iy0]),
        ("upper", slice(iy0, None), ez_line[:, iy0] - ez_line[:, -1]),
    ):
        flux = np.trapezoid(bx_line[:, indices], y[indices], axis=-1)
        prefix = f"centerline_{label}"
        data[f"{prefix}_bx_flux"] = flux
        data[f"{prefix}_faraday_rate"] = rhs
        derivative = np.full(nt, np.nan)
        times = np.asarray(ds.t)
        if nt >= 2 and np.all(np.diff(times) > 0.0):
            derivative = np.gradient(flux, times, edge_order=2 if nt >= 3 else 1)
        data[f"{prefix}_faraday_residual"] = derivative - rhs
        descriptions[f"{prefix}_bx_flux"] = (
            "signed integral B_x dy at fixed diagnostic x, from lower saved cell center to sheet center"
            if label == "lower"
            else "signed integral B_x dy at fixed diagnostic x, from sheet center to upper saved cell center"
        )
        descriptions[f"{prefix}_faraday_residual"] = (
            "d/dt integral B_x dy - (E_z at lower integration limit - E_z at upper integration limit); "
            "includes saved-time and spatial quadrature errors; not a topology-gated reconnection rate"
        )

    if "ion_velocity" in ds and "ions" in ds:
        uy = sides(ds.ion_velocity.sel(component="y"))
        ux = sides(ds.ion_velocity.sel(component="x"))
        by = sides(ds.b.sel(component="y"))
        signed_inflow = side_sign * uy
        mean_inflow = np.mean(signed_inflow, axis=-1)
        rho = sides(ds.ions.sel(ion_conserved="rho"))
        c_norm = float(ds.attrs.get("light_speed_normalized", np.nan))
        if not np.isfinite(c_norm) or c_norm <= 0.0:
            c_norm = np.nan
        alfven = c_norm * np.abs(bx) / np.sqrt(np.where(rho > 0.0, rho, np.nan))
        mean_alfven = np.mean(alfven, axis=-1)
        topology = np.asarray(ds.reconnection_valid, dtype=bool)
        center_ez = np.asarray(ds.xpoint_ez)
        physical_samples = np.all(np.isfinite(rho) & (rho > 0.0), axis=-1) & np.isfinite(center_ez)
        # No run-maximum or Nernst threshold: finite, inward flow on both sides.
        bulk_scale = np.mean(np.abs(bx) * signed_inflow, axis=-1)
        bulk_valid = (
            topology
            & physical_samples
            & np.all(signed_inflow > 0.0, axis=-1)
            & np.isfinite(bulk_scale)
            & (bulk_scale > 0.0)
        )
        alfven_scale = np.mean(np.abs(bx) * alfven, axis=-1)
        alfven_valid = topology & physical_samples & np.isfinite(alfven_scale) & (alfven_scale > 0.0)
        outflow_x = np.asarray(ds.ion_velocity.sel(component="x"))[:, :, iy0]
        x_from_center = np.asarray(ds.x) - float(ds.x[ix0])
        left = x_from_center < 0.0
        right = x_from_center > 0.0
        # Window covers the saved x line. This is a peak speed, not flux weighting.
        outflow_left = np.max(np.maximum(-outflow_x[:, left], 0.0), axis=-1, initial=0.0)
        outflow_right = np.max(np.maximum(outflow_x[:, right], 0.0), axis=-1, initial=0.0)
        mean_outflow = 0.5 * (outflow_left + outflow_right)
        b_squared = sides((ds.b**2).sum("component"))
        magnetic_pressure = np.mean(0.5 * c_norm**2 * b_squared, axis=-1)
        ram_pressure = np.mean(rho * uy**2, axis=-1)
        data.update(
            upstream_ion_inflow_y=mean_inflow,
            upstream_ion_inflow_y_lower=signed_inflow[:, 0],
            upstream_ion_inflow_y_upper=signed_inflow[:, 1],
            upstream_alfven_speed=mean_alfven,
            upstream_alfven_mach=_ratio(mean_inflow, mean_alfven),
            bulk_rate_normalization_valid=bulk_valid,
            alfven_rate_normalization_valid=alfven_valid,
            normalized_reconnection_rate_bulk=_ratio(center_ez, bulk_scale, bulk_valid),
            normalized_reconnection_rate_alfven=_ratio(center_ez, alfven_scale, alfven_valid),
            centerline_ion_outflow_speed=mean_outflow,
            centerline_ion_outflow_alfven_mach=_ratio(mean_outflow, mean_alfven),
            upstream_bulk_flux_inflow=np.mean(side_sign * np.sign(bx) * (uy * bx - ux * by), axis=-1),
            upstream_magnetic_pressure=magnetic_pressure,
            upstream_ram_pressure=ram_pressure,
            upstream_dynamic_beta=_ratio(ram_pressure, magnetic_pressure),
        )
        theta = float(ds.attrs.get("temperature_energy_normalized", np.nan))
        if not np.isfinite(theta) or theta <= 0.0:
            theta = np.nan
        if "ion_pressure" in ds and "ne" in ds and "temperature" in ds:
            thermal_pressure = np.mean(sides(ds.ion_pressure + theta * ds.ne * ds.temperature), axis=-1)
            data["upstream_thermal_pressure"] = thermal_pressure
            data["upstream_thermal_beta"] = _ratio(thermal_pressure, magnetic_pressure)
        descriptions.update(
            upstream_ion_inflow_y="mean signed inward u_i,y at the two upstream B_x peaks; outward flow is negative",
            upstream_alfven_speed="mean(c_normalized |B_x| / sqrt(rho_i)) at the two upstream positions",
            normalized_reconnection_rate_bulk=(
                "central E_z / mean(|B_x| signed inward u_i,y); NaN unless centered topology gate and both sides inward"
            ),
            normalized_reconnection_rate_alfven=(
                "central E_z / mean(|B_x| v_A,reconnecting); NaN unless centered topology gate and positive scale"
            ),
            centerline_ion_outflow_speed=(
                "mean of peak outward u_i,x on each side of center along the full saved x line at sheet y; "
                "not a boundary or mass-flux-weighted measurement"
            ),
            upstream_bulk_flux_inflow="mean(s sign(B_x) (u_i,y B_x - u_i,x B_y)); same sampling as total electric flux",
            upstream_magnetic_pressure="mean(c_normalized^2 |B|^2 / 2), including guide field",
            upstream_ram_pressure="mean(rho_i u_i,y^2); ram pressure, twice directed kinetic energy density",
            upstream_thermal_pressure=(
                "mean(p_i + n_e T_e T0/(m_e v0^2)); electron scalar second moment in ion frame, "
                "a weak-drift thermal-pressure proxy"
            ),
        )
    result = ds.assign({name: ("t", values) for name, values in data.items()})
    for name, description in descriptions.items():
        if name in result:
            result[name].attrs["definition"] = description
    result.attrs["reconnection_sampling"] = (
        f"fixed center x={float(ds.x[ix0])}, y={float(ds.y[iy0])}; upstream |Bx| peaks on this x line; "
        "assumes x outflow / y inflow and centered sheet; not general X-point detection"
    )
    return result


@dataclass(frozen=True)
class MagpieReference:
    """Carbon upstream reference, not a complete simulation input deck.

    Hare et al., PRL 118, 085001 (2017), Table I; L=7 mm from
    Hare et al., PoP 25, 055703 (2018), section III. Ti=50 eV is an upper bound.
    """

    electron_density_m3: float = 3.0e23
    mean_charge: float = 4.0
    ion_mass_u: float = 12.0
    electron_temperature_ev: float = 15.0
    ion_temperature_ev: float = 50.0
    magnetic_field_t: float = 3.0
    inflow_speed_m_s: float = 5.0e4
    sheet_half_width_m: float = 0.6e-3
    sheet_half_length_m: float = 7.0e-3

    def scales(self) -> dict[str, float]:
        """Derived SI and dimensionless scales without a transport closure.

        Isothermal ion-acoustic speed uses (Z Te + Ti)/mi; the ion fluid's
        adiabatic characteristic speed is a different quantity. No resistivity,
        cooling or ionization law is inferred from these inputs.
        """
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        ni = self.electron_density_m3 / self.mean_charge
        mi = self.ion_mass_u * atomic_mass
        rho = ni * mi
        b = self.magnetic_field_t
        va = b / np.sqrt(mu_0 * rho)
        cs = np.sqrt(e * (self.mean_charge * self.electron_temperature_ev + self.ion_temperature_ev) / mi)
        pm = b**2 / (2.0 * mu_0)
        pr = rho * self.inflow_speed_m_s**2
        pt = e * (self.electron_density_m3 * self.electron_temperature_ev + ni * self.ion_temperature_ev)
        di = np.sqrt(mi / (mu_0 * ni * (self.mean_charge * e) ** 2))
        return {
            "ion_density_m3": ni,
            "ion_mass_ratio": mi / m_e,
            "mass_density_kg_m3": rho,
            "alfven_speed_m_s": va,
            "isothermal_sound_speed_m_s": cs,
            "alfven_mach": self.inflow_speed_m_s / va,
            "isothermal_sonic_mach": self.inflow_speed_m_s / cs,
            "magnetic_pressure_pa": pm,
            "ram_pressure_pa": pr,
            "thermal_pressure_pa": pt,
            "dynamic_beta": pr / pm,
            "thermal_beta": pt / pm,
            "ion_inertial_length_m": di,
            "ion_inertial_length_over_half_width": di / self.sheet_half_width_m,
            "sheet_aspect_ratio": self.sheet_half_length_m / self.sheet_half_width_m,
            "inflow_crossing_time_s": self.sheet_half_width_m / self.inflow_speed_m_s,
            "alfven_transit_time_s": self.sheet_half_length_m / va,
        }


def main() -> None:
    """Print a reference scale report, optionally overriding inputs with JSON."""
    parser = argparse.ArgumentParser(
        description="Carbon MAGPIE reference scales; not a simulation or experiment predictor"
    )
    parser.add_argument("--parameters", type=Path, help="JSON object overriding MagpieReference SI/eV fields")
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text()) if args.parameters else {}
    reference = MagpieReference(**parameters)
    print(json.dumps({"parameters": asdict(reference), "scales": reference.scales()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
