"""Tests for the linearized Boltzmann electron closure and stochastic forcing.

The ``poisson-boltzmann`` field solver evolves kinetic ions only; electrons
respond adiabatically through the screened Poisson solve

    e phi_k / T_e = (delta n_k / n_0) / (1 + k^2 lambda_De^2)

Ion acoustic dispersion in this system (weak ion Landau damping, T_e >> T_i):

    omega^2 = k^2 c_s^2 / (1 + k^2 lambda_De^2) + 3 k^2 v_ti^2

with c_s^2 = Z T_e / m_i.
"""

import numpy as np
import pytest
import yaml

from adept import ergoExo
from adept._vlasov1d.datamodel import StochasticDriverConfig
from adept._vlasov1d.grid import Grid
from adept._vlasov1d.simulation import StochasticDriver
from adept._vlasov1d.solvers.pushers.field import BoltzmannPoissonSolver


def _load_config():
    with open("tests/test_vlasov1d/configs/boltzmann_iaw.yaml") as file:
        return yaml.safe_load(file)


def _measure_frequency(signal, time_axis, expected_omega):
    """Measure the dominant oscillation frequency of the k=1 box mode."""
    nx = signal.shape[1]
    mode = 2.0 / nx * np.fft.fft(signal, axis=1)[:, 1]

    nt = len(time_axis)
    mode_late = mode[nt // 4 :]
    dt = time_axis[1] - time_axis[0]
    omega_axis = 2 * np.pi * np.fft.fftfreq(len(mode_late), dt)
    spectrum = np.abs(np.fft.fft(mode_late))

    search = (omega_axis > expected_omega / 5) & (omega_axis < 5 * expected_omega)
    return omega_axis[search][np.argmax(spectrum[search])]


def test_boltzmann_field_solver_kernel():
    """The field solve must match the analytic screened-Poisson response.

    For n_i(x) = n_0 (1 + eps cos(kx)) the linearized Boltzmann closure gives
    E(x) = T_e * eps * k / (1 + k^2 lambda_De^2) * sin(kx).
    """
    nx, nv = 64, 256
    length = 2 * np.pi / 0.1
    dx = length / nx
    x = np.linspace(dx / 2, length - dx / 2, nx)
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi

    vmax, k, eps, Te = 0.64, 0.1, 1e-3, 1.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
    T_i = 0.01
    maxwellian = np.exp(-(v**2) / (2 * T_i)) / np.sqrt(2 * np.pi * T_i)
    f = (1 + eps * np.cos(k * x))[:, None] * maxwellian[None, :]

    for lambda_De, expected_screening in [(1.0, 1 + k**2), (0.0, 1.0), (None, 1 + k**2 * Te)]:
        solver = BoltzmannPoissonSolver(
            kx=kx,
            species_grids={"ion": {"dv": dv}},
            species_params={"ion": {"charge": 1.0}},
            Te=Te,
            lambda_De=lambda_De,
        )
        e_field = np.array(solver({"ion": f}, prev_ex=None, dt=None))
        expected = Te * eps * k / expected_screening * np.sin(k * x)
        np.testing.assert_allclose(e_field, expected, atol=1e-8 * eps * k)


def test_stochastic_driver_statistics():
    """The OU driver must be reproducible and have the configured stationary RMS."""
    cfg = StochasticDriverConfig(modes=[1, 2], amplitude=0.01, tau=100.0, seed=7)
    grid = Grid(
        xmin=0.0,
        xmax=100.0,
        nx=64,
        tmin=0.0,
        tmax_requested=100000.0,
        dt_requested=0.5,
        should_override_dt_for_em_waves=False,
        beta=1.0,
    )
    driver = StochasticDriver(cfg, grid)

    amps = np.array(driver.amp_real) + 1j * np.array(driver.amp_imag)
    rms = np.sqrt(np.mean(np.abs(amps) ** 2, axis=0))
    # 10000 correlation times of samples: the stationary RMS should be well converged
    np.testing.assert_allclose(rms, cfg.amplitude, rtol=0.1)

    # Same seed reproduces the same realization
    driver2 = StochasticDriver(cfg, grid)
    np.testing.assert_array_equal(np.array(driver.amp_real), np.array(driver2.amp_real))

    # Evaluation returns a zero-mean field on the grid containing only driven modes
    field = np.array(driver(50.0, np.array(grid.x)))
    field_k = np.fft.rfft(field) / len(field)
    assert abs(field_k[0]) < 1e-12
    assert np.all(np.abs(field_k[3:]) < 1e-12)


def test_iaw_dispersion_boltzmann():
    """Ion acoustic frequency with Boltzmann electrons must match the dispersion relation."""
    config = _load_config()

    k = config["density"]["species-ion-background"]["wavenumber"]
    Te = config["terms"]["boltzmann_electrons"]["Te"]
    lambda_De = config["terms"]["boltzmann_electrons"]["lambda_De"]
    Z = config["terms"]["species"][0]["charge"]
    m_i = config["terms"]["species"][0]["mass"]
    T_i = config["density"]["species-ion-background"]["T0"]

    cs_squared = Z * Te / m_i
    vti_squared = T_i / m_i
    expected_omega = np.sqrt(k**2 * cs_squared / (1 + k**2 * lambda_De**2) + 3 * k**2 * vti_squared)

    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    n_ion = solver_result.ys["fields"]["ion"]["n"]
    time_axis = solver_result.ts["fields"]

    measured_omega = _measure_frequency(np.array(n_ion), np.array(time_axis), expected_omega)

    print(f"\nBoltzmann-electron IAW: expected omega = {expected_omega:.6f}, measured = {measured_omega:.6f}")
    np.testing.assert_allclose(measured_omega, expected_omega, rtol=0.05)


@pytest.mark.parametrize("edfdv", ["exponential", "cubic-spline"])
def test_stochastic_forcing_smoke(edfdv):
    """Short driven run: forcing must inject energy and the solve must stay finite."""
    config = _load_config()

    config["grid"]["tmax"] = 200.0
    config["save"]["fields"]["t"]["nt"] = 101
    config["save"]["ion"]["main"]["t"]["tmax"] = 200.0
    config["density"]["species-ion-background"]["amplitude"] = 0.0
    config["terms"]["edfdv"] = edfdv
    config["drivers"]["ex_stochastic"] = {
        "modes": [1, 2],
        "amplitude": 1.0e-3,
        "tau": 50.0,
        "seed": 11,
    }
    config["mlflow"]["run"] = f"stochastic-smoke-{edfdv}"

    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    e_field = np.array(solver_result.ys["fields"]["e"])
    n_ion = np.array(solver_result.ys["fields"]["ion"]["n"])

    assert np.all(np.isfinite(e_field)), "Electric field must stay finite under stochastic forcing"
    assert np.all(np.isfinite(n_ion)), "Ion density must stay finite under stochastic forcing"
    # The initial state is unperturbed, so any late-time density fluctuation comes from the forcing
    assert np.std(n_ion[len(n_ion) // 2 :]) > 0, "Stochastic forcing should drive density fluctuations"


if __name__ == "__main__":
    test_iaw_dispersion_boltzmann()
