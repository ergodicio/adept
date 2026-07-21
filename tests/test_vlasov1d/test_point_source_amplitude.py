# Copyright (c) Ergodic LLC 2023
# research@ergodic.io
"""Test that a point-source ey driver produces waves with the expected amplitude.

Launches a single ey driver in point-source mode in a near-vacuum plasma
(n_e = 0.001 * n_crit). The wave propagates rightward from a source placed
near the left boundary. After the wave fills the analysis region, the
amplitude is measured via a matched filter at (k, ω) and compared to a0.

In vacuum the Green's function calibration is exact. At 0.1% of critical
density, the plasma correction k_vac/k_plasma is ~1.00005, so the measured
amplitude should match a0 to well within a few percent.
"""

import math

import numpy as np

from adept import ergoExo
from adept.normalization import electron_debye_normalization


def build_config():
    normalizing_density = "1e20/cc"
    normalizing_temperature = "2000eV"

    norm = electron_debye_normalization(normalizing_density, normalizing_temperature)
    c_norm = norm.speed_of_light_norm()

    # Near-vacuum: density = 0.001 * n_crit → ωpe² = 0.001
    n_e = 0.001

    # EM wave well above plasma frequency
    w0 = 10.0
    # Vacuum wavenumber
    k0 = w0 / c_norm
    em_wavelength = 2 * math.pi / k0

    a0 = 1e-4

    # Domain: many wavelengths so the wave has room to propagate
    xmax = 40 * em_wavelength
    nx = 512
    dx = xmax / nx
    dt = 0.5 * dx / c_norm

    # Enough time for the wave to cross the domain
    transit_time = xmax / c_norm
    tmax = 3 * transit_time
    # Time envelope rise must be fast so the wave is at full amplitude
    # well before the analysis window starts (at 1.5 * transit_time)
    time_rise = 0.1 * transit_time

    config = {
        "solver": "vlasov-1d",
        "units": {
            "laser_wavelength": "351nm",
            "normalizing_temperature": normalizing_temperature,
            "normalizing_density": normalizing_density,
            "Z": 1,
            "Zp": 1,
        },
        "density": {
            "quasineutrality": True,
            "species-background": {
                "noise_seed": 42,
                "noise_type": "uniform",
                "noise_val": 0.0,
                "v0": 0.0,
                "T0": 1.0,
                "m": 2.0,
                "basis": "uniform",
                "rise": float(n_e),
            },
        },
        "grid": {
            "dt": float(dt),
            "nv": 64,
            "nx": nx,
            "tmin": 0.0,
            "tmax": float(tmax),
            "vmax": 6.0,
            "xmin": 0.0,
            "xmax": float(xmax),
        },
        "save": {
            "fields": {
                "t": {"tmin": 0.0, "tmax": float(tmax), "nt": 301},
            },
            "electron": {
                "main": {
                    "t": {"tmin": 0.0, "tmax": float(tmax), "nt": 4},
                },
            },
        },
        "mlflow": {
            "experiment": "test-point-source-amplitude",
            "run": "test",
        },
        "drivers": {
            "ex": {},
            "ey": {
                "0": {
                    "params": {
                        "a0": a0,
                        "k0": float(k0),
                        "w0": float(w0),
                        "dw0": 0.0,
                    },
                    "source_type": "point",
                    "envelope": {
                        "time": {
                            "center": float(10 * tmax),
                            "rise": float(time_rise),
                            "width": float(20 * tmax),
                        },
                        "space": {
                            "center": float(xmax * 0.05),
                            "rise": float(em_wavelength),
                            "width": float(em_wavelength),
                        },
                    },
                },
            },
        },
        "diagnostics": {
            "diag-vlasov-dfdt": False,
            "diag-fp-dfdt": False,
        },
        "terms": {
            "field": "poisson",
            "edfdv": "exponential",
            "time": "leapfrog",
            "fokker_planck": {
                "is_on": False,
                "type": "Dougherty",
                "time": {
                    "baseline": 0.0,
                    "bump_or_trough": "bump",
                    "center": 0.0,
                    "rise": 1.0,
                    "slope": 0.0,
                    "bump_height": 0.0,
                    "width": 1.0,
                },
                "space": {
                    "baseline": 0.0,
                    "bump_or_trough": "bump",
                    "center": 0.0,
                    "rise": 1.0,
                    "slope": 0.0,
                    "bump_height": 0.0,
                    "width": 1.0,
                },
            },
            "krook": {
                "is_on": False,
                "time": {
                    "baseline": 0.0,
                    "bump_or_trough": "bump",
                    "center": 0.0,
                    "rise": 1.0,
                    "bump_height": 0.0,
                    "width": 1.0,
                },
                "space": {
                    "baseline": 0.0,
                    "bump_or_trough": "bump",
                    "center": 0.0,
                    "rise": 1.0,
                    "bump_height": 0.0,
                    "width": 1.0,
                },
            },
        },
    }

    return config, {
        "a0": a0,
        "w0": w0,
        "k0": k0,
        "c_norm": c_norm,
        "n_e": n_e,
        "transit_time": transit_time,
    }


def test_point_source_amplitude():
    """Verify that a point-source ey driver produces wave amplitude matching a0."""
    config, params = build_config()

    a0 = params["a0"]
    w0 = params["w0"]
    k0 = params["k0"]
    transit_time = params["transit_time"]

    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    nx = config["grid"]["nx"]
    xmax = config["grid"]["xmax"]
    dx = xmax / nx
    x = np.linspace(dx / 2, xmax - dx / 2, nx)

    a_field = np.array(solver_result.ys["fields"]["a"])  # (nt, nx+2) with ghost cells
    t_fields = np.array(solver_result.ts["fields"])
    a_interior = a_field[:, 1:-1]  # (nt, nx)

    # Analyze in the middle of the domain, away from source and right boundary
    x_start = int(nx * 0.3)
    x_end = int(nx * 0.7)
    x_sub = x[x_start:x_end]
    a_sub = a_interior[:, x_start:x_end]

    # Late-time window: after the wave has had time to fill the analysis region
    t_mask = t_fields > 1.5 * transit_time
    a_late = a_sub[t_mask]
    t_late = t_fields[t_mask]

    # Matched filter: extract amplitude at (k0, w0)
    a_k_t = np.mean(a_late * np.exp(-1j * k0 * x_sub[None, :]), axis=1) * 2.0
    corr = np.mean(a_k_t * np.exp(1j * w0 * t_late))
    measured_a0 = np.abs(corr)

    print("\nPoint Source Amplitude Test")
    print(f"  Specified a0 = {a0:.6e}")
    print(f"  Measured a0  = {measured_a0:.6e}")
    if a0 > 0:
        rel_err = abs(measured_a0 - a0) / a0
        print(f"  Relative error: {rel_err:.4f}")

    assert measured_a0 > 0, "No wave signal detected"
    rel_err = abs(measured_a0 - a0) / a0
    assert rel_err < 0.10, (
        f"Point source wave amplitude {measured_a0:.6e} does not match "
        f"specified a0={a0:.6e} (relative error {rel_err:.4f})"
    )


if __name__ == "__main__":
    test_point_source_amplitude()
