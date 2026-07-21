"""Shared helpers for PIC-1D integrated physics tests."""

from __future__ import annotations

import copy

import numpy as np

from adept.pic1d import BasePIC1D


def run_sim(cfg: dict):
    """Run a PIC-1D simulation through the full ADEPTModule lifecycle.

    Returns ``(diffrax_solution, module)``.
    """
    m = BasePIC1D(copy.deepcopy(cfg))
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    out = m({})
    return out["solver result"], m


def loglinear_fit(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of a least-squares fit to ``log|y|`` vs ``t``.

    Robust to small/zero ``y`` via a tiny floor.
    """
    log_y = np.log(np.abs(y) + 1e-300)
    coeffs = np.polyfit(t, log_y, 1)
    return float(coeffs[0]), float(coeffs[1])


def mode_amplitude(E_xt: np.ndarray, m_mode: int) -> np.ndarray:
    """Return ``|E_k|(t)`` for spatial Fourier mode ``m_mode``."""
    Ek = np.fft.fft(E_xt, axis=1)
    return np.abs(Ek[:, m_mode])
