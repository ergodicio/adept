#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""
Units-boundary regression tests for the Vlasov-1D normalization layer.

Every physics test in this suite works in dimensionless code units, which
validates the engine but not the dimensional dictionary: numeric inputs bypass
``normalize()`` entirely. These tests cross the units boundary — dimensional
input in, dimensional quantity out — and compare against *independent* physical
references (CODATA constants via scipy, and the NRL formulary), so a convention
error in ``normalization.py`` cannot hide.

The engine convention under test: v0 = sqrt(T0/m_e) (RMS / standard-deviation
thermal speed), L0 = v0/wp0 = lambda_De, Maxwellian exp(-v^2/2) at T=1.
"""

import numpy as np
from scipy import constants as csts

from adept.normalization import electron_debye_normalization, normalize

N0_STR, T0_STR = "1.5e21/cc", "2000eV"
N0_SI = 1.5e21 * 1e6  # 1/m^3
T0_J = 2000.0 * csts.e


def test_debye_normalization_against_codata():
    """v0, L0, tau, and c_hat must match hand-computed CODATA values."""
    norm = electron_debye_normalization(N0_STR, T0_STR)

    v_th = np.sqrt(T0_J / csts.m_e)  # sigma convention: sqrt(T/m)
    wp0 = np.sqrt(N0_SI * csts.e**2 / (csts.epsilon_0 * csts.m_e))
    lambda_de = v_th / wp0

    np.testing.assert_allclose(norm.v0.to("m/s").magnitude, v_th, rtol=1e-6)
    np.testing.assert_allclose(norm.L0.to("m").magnitude, lambda_de, rtol=1e-6)
    np.testing.assert_allclose(norm.tau.to("s").magnitude, 1.0 / wp0, rtol=1e-6)
    np.testing.assert_allclose(norm.speed_of_light_norm(), csts.c / v_th, rtol=1e-6)


def test_debye_length_against_nrl_formulary():
    """lambda_De = 7.43e2 sqrt(T_eV/n_cc) cm (NRL formulary) — a truly external reference."""
    norm = electron_debye_normalization(N0_STR, T0_STR)
    lambda_de_nrl_m = 7.43e2 * np.sqrt(2000.0 / 1.5e21) * 1e-2
    np.testing.assert_allclose(norm.L0.to("m").magnitude, lambda_de_nrl_m, rtol=1e-3)


def test_dimensional_string_round_trip():
    """String inputs must convert with L0 = lambda_De (not sqrt(2) lambda_De)."""
    norm = electron_debye_normalization(N0_STR, T0_STR)

    v_th = np.sqrt(T0_J / csts.m_e)
    wp0 = np.sqrt(N0_SI * csts.e**2 / (csts.epsilon_0 * csts.m_e))
    lambda_de = v_th / wp0

    # x: 100 um -> 100e-6 / lambda_De code units
    np.testing.assert_allclose(normalize("100um", norm, dim="x"), 100e-6 / lambda_de, rtol=1e-6)
    # k: 1/um -> lambda_De / 1e-6 code units (k lambda_De)
    np.testing.assert_allclose(normalize("1/um", norm, dim="k"), lambda_de / 1e-6, rtol=1e-6)
    # t: 1 ps -> wp0 * 1e-12
    np.testing.assert_allclose(normalize("1ps", norm, dim="t"), wp0 * 1e-12, rtol=1e-6)
    # numeric inputs pass through untouched
    assert normalize(3.25, norm, dim="x") == 3.25


def test_logged_collision_frequency_is_physical():
    """logLambda and nuee must be positive and match the NRL expression."""
    norm = electron_debye_normalization(N0_STR, T0_STR)
    log_lambda = float(norm.logLambda_ee())
    # NRL: 23.5 - ln(n^1/2 T^-5/4) - [1e-5 + (ln T - 2)^2/16]^1/2
    expected = 23.5 - np.log(np.sqrt(1.5e21) * 2000.0**-1.25) - np.sqrt(1e-5 + (np.log(2000.0) - 2.0) ** 2 / 16.0)
    np.testing.assert_allclose(log_lambda, expected, rtol=1e-10)
    assert log_lambda > 0
    assert norm.approximate_ee_collision_frequency().to("Hz").magnitude > 0
