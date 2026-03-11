# Copyright (c) Ergodic LLC 2023
# research@ergodic.io
"""Test ponderomotive force from ey driver beat wave.

Two co-propagating EM waves with slightly different frequencies create a beat
pattern in a²(x,t). The slowly-varying ponderomotive force from the beat
drives a low-frequency electron oscillation.

The test:
1. Launches two ey waves at (k₁,ω₁) and (k₂,ω₂) from the EM dispersion relation
2. Measures the actual a²(x,t) beat amplitude from the simulation
3. Computes the expected ponderomotive electron velocity from that amplitude
4. Measures the actual electron mean velocity at the beat (Δk, Δω)
5. Compares measured vs expected

This tests the full ponderomotive pipeline: a → a² → pond → force → accel → velocity.
"""

import math

import numpy as np

from adept import ergoExo
from adept.normalization import electron_debye_normalization


def build_config():
    """Build config for ponderomotive beat wave test."""
    normalizing_density = "1e20/cc"
    normalizing_temperature = "2000eV"

    norm = electron_debye_normalization(normalizing_density, normalizing_temperature)
    c_norm = norm.speed_of_light_norm()

    # Two EM waves well above ωpe = 1, with large Δω so Δω >> ωpe
    # to avoid dielectric shielding of the ponderomotive response.
    # ε(Δk, Δω) = 1 - ωpe²/Δω² ≈ 1 - 1/25 = 0.96 ≈ 1
    w1 = 7.0
    w2 = 2.0
    delta_w = w1 - w2  # 5.0
    k1 = math.sqrt(w1**2 - 1.0) / c_norm
    k2 = math.sqrt(w2**2 - 1.0) / c_norm
    delta_k = k1 - k2

    a0 = 1e-4  # Small to stay in linear regime (resonant source amplifies)

    # Domain: 4 beat wavelengths, 6 beat periods
    beat_wavelength = 2 * math.pi / abs(delta_k)
    beat_period = 2 * math.pi / delta_w
    em_wavelength = 2 * math.pi / k1  # EM wavelength for source sizing
    xmax = 4 * beat_wavelength
    tmax = 6 * beat_period

    nx = 512
    dx = xmax / nx
    dt = 0.5 * dx / c_norm  # CFL condition

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
            },
        },
        "grid": {
            "dt": float(dt),
            "nv": 128,
            "nx": nx,
            "tmin": 0.0,
            "tmax": float(tmax),
            "vmax": 6.0,
            "xmin": 0.0,
            "xmax": float(xmax),
        },
        "save": {
            "fields": {
                "t": {"tmin": 0.0, "tmax": float(tmax), "nt": 501},
            },
            "electron": {
                "main": {
                    "t": {"tmin": 0.0, "tmax": float(tmax), "nt": 501},
                },
            },
        },
        "mlflow": {
            "experiment": "test-ey-ponderomotive-quiver",
            "run": "test",
        },
        "drivers": {
            "ex": {},
            "ey": {
                "0": {
                    "params": {
                        "a0": a0,
                        "k0": float(k1),
                        "w0": float(w1),
                        "dw0": 0.0,
                    },
                    "envelope": {
                        "time": {"center": float(10 * tmax), "rise": float(beat_period), "width": float(20 * tmax)},
                        "space": {"center": float(xmax * 0.1), "rise": float(em_wavelength), "width": float(3 * em_wavelength)},
                    },
                },
                "1": {
                    "params": {
                        "a0": a0,
                        "k0": float(k2),
                        "w0": float(w2),
                        "dw0": 0.0,
                    },
                    "envelope": {
                        "time": {"center": float(10 * tmax), "rise": float(beat_period), "width": float(20 * tmax)},
                        "space": {"center": float(xmax * 0.1), "rise": float(em_wavelength), "width": float(3 * em_wavelength)},
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

    params = {
        "a0": a0,
        "w1": w1,
        "w2": w2,
        "k1": k1,
        "k2": k2,
        "delta_w": delta_w,
        "delta_k": delta_k,
        "c_norm": c_norm,
        "beat_period": beat_period,
    }

    return config, params


def test_ey_ponderomotive_quiver():
    """Verify ponderomotive force from ey driver beat wave.

    Measures the (Δk, Δω) Fourier component of the mean electron velocity
    and checks consistency with the a² beat amplitude from the simulation.

    The ponderomotive acceleration for electrons (q=-1, m=1) from the beat
    component of a²:
        pond = -0.5 * ∂(a²)/∂x
        accel = pond  (since q²/m = 1 for electrons)
        v_pond = 0.5 * Δk * A_beat / Δω

    where A_beat is the amplitude of a² at (Δk, Δω).
    """
    config, params = build_config()

    delta_w = params["delta_w"]
    delta_k = params["delta_k"]

    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    nx = config["grid"]["nx"]
    nv = config["grid"]["nv"]
    vmax = config["grid"]["vmax"]
    dv = 2.0 * vmax / nv
    v = np.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
    xmax = config["grid"]["xmax"]
    dx = xmax / nx

    # ---- Restrict analysis to propagation region away from source ----
    # Source is at x ~ 0.15*xmax on the left; both waves have k>0 so they
    # propagate rightward. Analyze in the right half where wave amplitude is uniform.
    x = np.linspace(dx / 2, xmax - dx / 2, nx)
    x_analysis_start = int(nx * 0.5)
    x_analysis_end = int(nx * 0.9)  # avoid right boundary
    x_sub = x[x_analysis_start:x_analysis_end]

    # ---- Measure actual a² beat amplitude from the vector potential ----
    a_field = np.array(solver_result.ys["fields"]["a"])  # (nt, nx+2) with ghost cells
    t_fields = np.array(solver_result.ts["fields"])
    a_interior = a_field[:, 1:-1]  # (nt, nx)
    a_sub = a_interior[:, x_analysis_start:x_analysis_end]

    a_sq = a_sub**2

    # Late-time window to skip transient (waves need time to propagate from source)
    t_start = 2 * params["beat_period"]
    t_mask_f = t_fields > t_start
    a_sq_late = a_sq[t_mask_f]
    t_fields_late = t_fields[t_mask_f]

    # Measure individual wave amplitudes A₁, A₂ from a(x,t) at their carrier
    # wavenumbers k₁, k₂. These are well separated from the low-k envelope
    # modulation, so extraction is clean.
    a_late = a_sub[t_mask_f]
    k1 = params["k1"]
    k2 = params["k2"]

    # Spatial matched filter at k1, then temporal matched filter at w1
    a_k1_t = np.mean(a_late * np.exp(-1j * k1 * x_sub[None, :]), axis=1) * 2.0
    corr_a1 = np.mean(a_k1_t * np.exp(1j * params["w1"] * t_fields_late))
    A1 = np.abs(corr_a1)

    a_k2_t = np.mean(a_late * np.exp(-1j * k2 * x_sub[None, :]), axis=1) * 2.0
    corr_a2 = np.mean(a_k2_t * np.exp(1j * params["w2"] * t_fields_late))
    A2 = np.abs(corr_a2)

    # The beat component of a² from two waves A₁sin(k₁x-ω₁t) + A₂sin(k₂x-ω₂t):
    # Cross term: 2·A₁·A₂·sin(...)·sin(...) → A₁·A₂·cos(Δk·x - Δω·t) (slowly varying)
    a_sq_beat_amplitude = A1 * A2

    # Expected ponderomotive velocity:
    # pond = -0.5 * grad(a²), for the beat component:
    #   pond_beat = 0.5 * Δk * A₁A₂ * sin(Δk*x - Δω*t)
    # For electrons (q=-1, m=1): accel = pond, v = accel/Δω
    # Include dielectric correction: ε(Δk,Δω) = 1 - ωpe²/Δω² screens the response
    epsilon = 1.0 - 1.0 / delta_w**2  # cold-plasma dielectric at beat frequency
    v_pond_expected = 0.5 * abs(delta_k) * a_sq_beat_amplitude / (delta_w * epsilon)

    # ---- Measure actual electron velocity at (Δk, Δω) in same sub-region ----
    f = np.array(solver_result.ys["electron.main"])  # (nt, nx, nv)
    t = np.array(solver_result.ts["electron.main"])

    f_sub = f[:, x_analysis_start:x_analysis_end, :]

    density = np.sum(f_sub, axis=2) * dv
    momentum = np.sum(f_sub * v[None, None, :], axis=2) * dv
    mean_v = momentum / density

    t_mask = t > t_start
    mean_v_late = mean_v[t_mask]
    t_late = t[t_mask]

    # Same matched filter approach for velocity
    mean_v_k_t = np.mean(mean_v_late * np.exp(-1j * delta_k * x_sub[None, :]), axis=1) * 2.0
    corr_v = np.mean(mean_v_k_t * np.exp(1j * delta_w * t_late))
    measured_v_pond = np.abs(corr_v)

    print("\nEy Ponderomotive Quiver Test")
    print(f"  a0 = {params['a0']}")
    print(f"  c_norm = {params['c_norm']:.2f}")
    print(f"  ω₁ = {params['w1']:.4f}, k₁ = {params['k1']:.6f}")
    print(f"  ω₂ = {params['w2']:.4f}, k₂ = {params['k2']:.6f}")
    print(f"  Δω = {delta_w:.4f}, Δk = {delta_k:.6f}")
    print(f"  Wave amplitudes: A₁ = {A1:.6e}, A₂ = {A2:.6e}")
    print(f"  a² beat amplitude (A₁·A₂) = {a_sq_beat_amplitude:.6e}")
    print(f"  Expected v_pond (from a²) = {v_pond_expected:.6e}")
    print(f"  Measured v_pond = {measured_v_pond:.6e}")
    if v_pond_expected > 0:
        rel_err = abs(measured_v_pond - v_pond_expected) / v_pond_expected
        print(f"  Relative error: {rel_err:.4f}")

    assert a_sq_beat_amplitude > 0, "No beat signal detected in a² field"
    assert v_pond_expected > 0, "Expected ponderomotive velocity is zero"
    rel_err = abs(measured_v_pond - v_pond_expected) / v_pond_expected
    assert rel_err < 0.25, (
        f"Ponderomotive velocity {measured_v_pond:.6e} does not match "
        f"expected {v_pond_expected:.6e} from a² beat (relative error {rel_err:.4f})"
    )


if __name__ == "__main__":
    test_ey_ponderomotive_quiver()
