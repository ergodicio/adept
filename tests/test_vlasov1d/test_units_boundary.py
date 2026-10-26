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

from pathlib import Path

import numpy as np
import yaml
from scipy import constants as csts

from adept.normalization import electron_debye_normalization, ion_debye_normalization, normalize

N0_STR, T0_STR = "1.5e21/cc", "2000eV"
N0_SI = 1.5e21 * 1e6  # 1/m^3
T0_J = 2000.0 * csts.e

# Ion reference: deuterium at ion density n_i and ion temperature T_i
A_ION, Z_ION = 2.0, 1.0
NI_STR, TI_STR = "1.0e20/cc", "100eV"
NI_SI = 1.0e20 * 1e6  # 1/m^3
TI_J = 100.0 * csts.e
M_ION = A_ION * csts.m_p


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


def test_ion_debye_normalization_against_codata():
    """Ion v0, L0, tau, and c_hat must match hand-computed CODATA values."""
    norm = ion_debye_normalization(NI_STR, TI_STR, A=A_ION, Z=Z_ION)

    v_ti = np.sqrt(TI_J / M_ION)  # sigma convention: sqrt(T/m_i)
    wpi = np.sqrt(NI_SI * (Z_ION * csts.e) ** 2 / (csts.epsilon_0 * M_ION))
    lambda_di = v_ti / wpi

    np.testing.assert_allclose(norm.v0.to("m/s").magnitude, v_ti, rtol=1e-6)
    np.testing.assert_allclose(norm.L0.to("m").magnitude, lambda_di, rtol=1e-6)
    np.testing.assert_allclose(norm.tau.to("s").magnitude, 1.0 / wpi, rtol=1e-6)
    np.testing.assert_allclose(norm.speed_of_light_norm(), csts.c / v_ti, rtol=1e-6)
    # A Z=1 ion normalization at (n, T) shares the Debye length with the electron
    # one but the plasma frequency is sqrt(m_e/m_i) smaller
    e_norm = electron_debye_normalization(NI_STR, TI_STR)
    np.testing.assert_allclose(norm.L0.to("m").magnitude, e_norm.L0.to("m").magnitude, rtol=1e-6)
    np.testing.assert_allclose((norm.tau / e_norm.tau).to("").magnitude, np.sqrt(M_ION / csts.m_e), rtol=1e-6)


def test_ion_dimensional_string_round_trip():
    """String inputs must convert with the ion units (L0 = lambda_Di, tau = 1/wpi)."""
    norm = ion_debye_normalization(NI_STR, TI_STR, A=A_ION, Z=Z_ION)

    v_ti = np.sqrt(TI_J / M_ION)
    wpi = np.sqrt(NI_SI * (Z_ION * csts.e) ** 2 / (csts.epsilon_0 * M_ION))
    lambda_di = v_ti / wpi

    np.testing.assert_allclose(normalize("300um", norm, dim="x"), 300e-6 / lambda_di, rtol=1e-6)
    np.testing.assert_allclose(normalize("50ps", norm, dim="t"), wpi * 50e-12, rtol=1e-6)
    np.testing.assert_allclose(normalize("1/um", norm, dim="k"), lambda_di / 1e-6, rtol=1e-6)
    np.testing.assert_allclose(normalize("100km/s", norm, dim="v"), 100e3 / v_ti, rtol=1e-6)
    assert normalize(3.25, norm, dim="x") == 3.25


def test_ion_collision_frequency_against_nrl():
    """logLambda_ii and nu_ii must match the NRL formulary expressions."""
    norm = ion_debye_normalization(NI_STR, TI_STR, A=A_ION, Z=Z_ION)

    # NRL single-species ii Coulomb log: 23 - ln(Z^3 sqrt(2 n_i) / T_i^3/2)
    expected_log = 23.0 - np.log(Z_ION**3 * np.sqrt(2.0 * 1.0e20) * 100.0**-1.5)
    np.testing.assert_allclose(float(norm.logLambda_ii()), expected_log, rtol=1e-10)
    assert expected_log > 0

    # NRL ion collision rate: 4.80e-8 Z^4 mu^-1/2 n_i lambda_ii T_i^-3/2 Hz
    expected_nu = 4.80e-8 * Z_ION**4 / np.sqrt(A_ION) * 1.0e20 * expected_log * 100.0**-1.5
    np.testing.assert_allclose(norm.approximate_ii_collision_frequency().to("Hz").magnitude, expected_nu, rtol=1e-10)


def test_ion_reference_config_round_trip():
    """A full ion-reference config converts dimensional grid inputs with ion units,
    and write_units() reports ion-referenced quantities."""
    from adept._vlasov1d.modules import BaseVlasov1D

    with open(Path(__file__).parent / "configs" / "boltzmann_iaw.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["units"] = {
        "normalizing_temperature": TI_STR,
        "normalizing_density": NI_STR,
        "reference": "ion",
        "A": A_ION,
        "Z": Z_ION,
    }
    cfg["grid"]["xmax"] = "300um"
    cfg["grid"]["tmax"] = "50ps"

    module = BaseVlasov1D(cfg)
    norm = module.simulation.plasma_norm
    grid = module.simulation.grid

    v_ti = np.sqrt(TI_J / M_ION)
    wpi = np.sqrt(NI_SI * (Z_ION * csts.e) ** 2 / (csts.epsilon_0 * M_ION))

    # x round trip: grid.xmax (code units) * L0 must be the 300 um we asked for
    np.testing.assert_allclose(grid.xmax * norm.L0.to("m").magnitude, 300e-6, rtol=1e-6)
    # t: tmax is realigned to a whole number of steps, so compare to within one dt
    assert abs(grid.tmax - wpi * 50e-12) <= grid.dt

    units = module.write_units()
    np.testing.assert_allclose(units["wp0"].to("rad/s").magnitude, wpi, rtol=1e-6)
    np.testing.assert_allclose(units["v0"].to("m/s").magnitude, v_ti, rtol=1e-6)
    np.testing.assert_allclose(units["x0"].to("m").magnitude, v_ti / wpi, rtol=1e-6)
    np.testing.assert_allclose(units["box_length"].to("m").magnitude, 300e-6, rtol=1e-6)
    np.testing.assert_allclose(units["sim_duration"].to("s").magnitude, grid.tmax / wpi, rtol=1e-6)
    # Electron-specific collision entries are replaced by ion-ion ones
    assert units["reference_species"] == "ion"
    assert "nuii" in units and "nuii_norm" in units and "logLambda_ii" in units
    assert "nuee" not in units and "nuee_norm" not in units and "logLambda_ee" not in units
