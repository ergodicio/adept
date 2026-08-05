"""Shared config builder and direct-module runner for Vlasov-1D2V tests."""

import copy

import numpy as np


def base_config(nx=32, nv=256, nvperp=16, tmax=100.0, dt=0.25, nu=1.0e-3, fp_type="dougherty", a0=1.0e-2):
    """Return a small pawl-style config dict (modern vlasov-1d schema + 2V grid keys)."""
    envelope_time = {
        "baseline": 1.0,
        "bump_or_trough": "bump",
        "center": 0.0,
        "rise": 25.0,
        "slope": 0.0,
        "bump_height": 0.0,
        "width": 1.0e5,
    }
    cfg = {
        "units": {"normalizing_temperature": "2000eV", "normalizing_density": "1.5e21/cc"},
        "density": {
            "quasineutrality": True,
            "species-background": {
                "noise_seed": 420,
                "noise_type": "gaussian",
                "noise_val": 0.0,
                "v0": 0.0,
                "T0": 1.0,
                "m": 2.0,
                "basis": "uniform",
                "baseline": 1.0,
                "bump_or_trough": "bump",
                "center": 0.0,
                "rise": 25.0,
                "bump_height": 0.0,
                "width": 1.0e5,
            },
        },
        "grid": {
            "dt": dt,
            "nv": nv,
            "nx": nx,
            "tmin": 0.0,
            "tmax": tmax,
            "vmax": 6.4,
            "xmax": 20.94,
            "xmin": 0.0,
            "nvperp": nvperp,
            "vperp_max": 6.4,
        },
        "save": {
            "fields": {"t": {"tmin": 0.0, "tmax": tmax, "nt": 51}},
            "electron": {"main": {"t": {"tmin": 0.0, "tmax": tmax, "nt": 11}}},
        },
        "solver": "vlasov-1d2v",
        "mlflow": {"experiment": "test-vlasov1d2v", "run": "test"},
        "drivers": {
            "ex": {
                "0": {
                    "params": {"a0": a0, "k0": 0.3, "w0": 1.1598, "dw0": 0.0},
                    "envelope": {
                        "time": {"center": 40.0, "rise": 5.0, "width": 30.0},
                        "space": {"center": 0.0, "rise": 10.0, "width": 4.0e6},
                    },
                }
            },
            "ey": {},
        },
        "diagnostics": {"diag-vlasov-dfdt": True, "diag-fp-dfdt": True},
        "terms": {
            "field": "poisson",
            "edfdv": "exponential",
            "time": "sixth",
            "fokker_planck": {
                "is_on": nu > 0.0,
                "type": fp_type,
                "self_consistent_beta": {"enabled": True, "max_steps": 3},
                "time": dict(envelope_time, baseline=nu),
                "space": dict(envelope_time, baseline=1.0),
            },
            "krook": {
                "is_on": False,
                "time": dict(envelope_time),
                "space": dict(envelope_time),
            },
        },
    }
    return copy.deepcopy(cfg)


def run_module(module_class, cfg):
    """Drive a module through the ergoExo setup sequence and one solve, sans MLflow."""
    mod = module_class(cfg)
    mod.write_units()
    mod.get_derived_quantities()
    mod.get_solver_quantities()
    mod.init_state_and_args()
    mod.init_diffeqsolve()
    out = mod({})
    return mod, out["solver result"]


def marginal(f, wperp):
    """Return the v_perp marginal of f(..., v_par, v_perp)."""
    return np.einsum("...vp,p->...v", np.asarray(f), np.asarray(wperp))
