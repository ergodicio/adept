# Copyright (c) Ergodic LLC 2023
# research@ergodic.io
"""Test ex driver quiver velocity normalization.

An electron in a uniform oscillating electric field E = ω·a0·sin(kx - ωt)
acquires a quiver velocity v_quiver = a0·cos(kx - ωt).

The quiver velocity amplitude should be exactly a0 (in normalized units),
because the ω in the E-field amplitude cancels with the 1/ω from time integration.
"""

import numpy as np

from adept import ergoExo


# FIXME: move the config to a .yaml file in configs/ -- then manually set some of the values in python code. So build_config() should load the yaml and then modify the dict.
def build_config():
    """Build config for quiver velocity test."""

    # FIXME: the test config should use the intensity and wavelength specification, and calculate expected quiver velocity from intensity using the pint units library.
    a0 = 1e-3
    k0 = 0.3
    w0 = 10.0  # Well above ωpe = 1 to minimize self-consistent plasma response
    xmax = 2 * np.pi / k0
    tmax = 20.0  # Several oscillation periods at ω = 10 (period ≈ 0.63)

    config = {
        "solver": "vlasov-1d",
        "units": {
            "laser_wavelength": "351nm",
            "normalizing_temperature": "2000eV",
            "normalizing_density": "1e18/cc",  # Very low density to minimize self-consistent response
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
            },
        },
        "grid": {
            "dt": 0.01,
            "nv": 128,
            "nx": 32,
            "tmin": 0.0,
            "tmax": tmax,
            "vmax": 6.4,
            "xmin": 0.0,
            "xmax": float(xmax),
        },
        "save": {
            "fields": {
                "t": {"tmin": 0.0, "tmax": tmax, "nt": 501},
            },
            "electron": {
                "main": {
                    "t": {"tmin": 0.0, "tmax": tmax, "nt": 501},
                },
            },
        },
        "mlflow": {
            "experiment": "test-ex-driver-quiver",
            "run": "test",
        },
        "drivers": {
            "ex": {
                "0": {
                    "params": {
                        "a0": a0,
                        "k0": k0,
                        "w0": w0,
                        "dw0": 0.0,
                    },
                    "envelope": {
                        "time": {"center": 1000.0, "rise": 5.0, "width": 2000.0},
                        "space": {"center": 0.0, "rise": 10.0, "width": 1e6},
                    },
                }
            },
            "ey": {},
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
    return config, a0, k0, w0


def test_ex_driver_quiver():
    """Verify ex driver produces correct quiver velocity amplitude.

    Measures the k0 Fourier mode of the mean velocity <v>(x,t) and checks
    that its amplitude matches a0.
    """
    config, a0, k0, w0 = build_config()

    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    # Get grid info
    sim = exo.adept_module.simulation
    grid = sim.grid
    nx = config["grid"]["nx"]
    nv = config["grid"]["nv"]
    vmax = config["grid"]["vmax"]
    dv = 2.0 * vmax / nv
    v = np.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    # Extract distribution function: shape (nt, nx, nv) — saved at full resolution
    f = np.array(solver_result.ys["electron.main"])
    t = np.array(solver_result.ts["electron.main"])

    # Compute mean velocity <v>(x, t) = ∫v·f dv / ∫f dv
    density = np.sum(f, axis=2) * dv  # (nt, nx)
    momentum = np.sum(f * v[None, None, :], axis=2) * dv  # (nt, nx)
    mean_v = momentum / density  # (nt, nx)

    # Extract the k0 Fourier mode of mean velocity (mode index 1 since xmax = 2π/k0)
    mean_v_k1 = np.fft.fft(mean_v, axis=1)[:, 1] * (2.0 / nx)  # complex amplitude vs time

    # Use late-time window where the driver envelope is fully on
    t_start = 2.0  # Skip initial transient
    t_mask = t > t_start
    mean_v_k1_late = mean_v_k1[t_mask]
    t_late = t[t_mask]

    # Matched filter at driver frequency w0 to isolate driven response
    # from self-consistent EPW oscillation at a different frequency.
    # corr = <v_k1(t) * exp(+i*w0*t)> extracts the w0 component exactly,
    # avoiding FFT bin alignment issues.
    corr = np.mean(mean_v_k1_late * np.exp(1j * w0 * t_late))
    measured_amplitude = np.abs(corr)

    print("\nEx Driver Quiver Velocity Test")
    print(f"  a0 = {a0}")
    print(f"  k0 = {k0}, w0 = {w0}")
    print(f"  Expected quiver amplitude: {a0:.6e}")
    print(f"  Measured quiver amplitude: {measured_amplitude:.6e}")
    print(f"  Relative error: {abs(measured_amplitude - a0) / a0:.4f}")

    assert abs(measured_amplitude - a0) / a0 < 0.05, (
        f"Quiver velocity amplitude {measured_amplitude:.6e} does not match "
        f"expected a0 = {a0:.6e} (relative error {abs(measured_amplitude - a0) / a0:.4f})"
    )


if __name__ == "__main__":
    test_ex_driver_quiver()
