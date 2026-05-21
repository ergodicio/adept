"""2D Landau damping test for the Vlasov-2D solver.

A longitudinal current source Jx(x, y, t) drives an electron plasma wave at
(kx0, omega0 = Re(root)). When the driver envelope turns off, the resulting
Ex(kx0) decays at the Landau rate gamma = Im(root). We measure that
late-time slope and compare to the analytic 1D Landau prediction (1D theory
applies because the driver is k = (kx0, 0) and f is initialized as an
unperturbed bi-Maxwellian).

The test goes through the full ergoExo flow so it logs to MLflow.
"""

import os

import numpy as np
import pytest
import yaml

from adept import electrostatic, ergoExo

_HERE = os.path.dirname(__file__)


def _load_cfg() -> dict:
    with open(os.path.join(_HERE, "configs", "landau_damping.yaml")) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("kx0", [0.3, 0.4])
def test_landau_damping_driven(kx0):
    """Drive the EPW at (kx0, omega0); measure the post-driver decay rate."""
    cfg = _load_cfg()

    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, kx0)
    omega = float(np.real(root))
    gamma_analytic = float(np.imag(root))

    cfg["grid"]["xmax"] = float(2.0 * np.pi / kx0)
    cfg["drivers"]["ex"]["0"]["params"]["k0x"] = float(kx0)
    cfg["drivers"]["ex"]["0"]["params"]["w0"] = omega
    cfg["mlflow"]["run"] = f"kx-{kx0}"

    exo = ergoExo()
    exo.setup(cfg)
    result, _, _ = exo(None)
    sol = result["solver result"]

    # Ex shape (nt, nx, ny); take k = (kx0, 0) by FFTing in x and selecting kx index 1
    ex_t = np.asarray(sol.ys["fields"]["ex"])  # (nt, nx, ny)
    t_ax = np.asarray(sol.ts["fields"])
    nx = ex_t.shape[1]
    ex_k = (2.0 / nx) * np.fft.fft(ex_t, axis=1)[:, 1, :].mean(axis=-1)
    ex_mag = np.abs(ex_k)

    # Fit log|Ex(k)| vs t between driver-off (~t=80) and recurrence/end margin.
    # With nvx>=256 and kx ~ 0.3, the Landau recurrence time is well past tmax.
    t0_fit = 90.0
    t1_fit = 150.0
    mask = (t_ax >= t0_fit) & (t_ax <= t1_fit) & (ex_mag > 0)
    if mask.sum() < 5:
        raise RuntimeError(f"Not enough fit points in t=[{t0_fit}, {t1_fit}]")
    slope, _ = np.polyfit(t_ax[mask], np.log(ex_mag[mask]), 1)
    gamma_measured = float(slope)

    print(
        f"\n[kx={kx0}] driven Landau: measured γ={gamma_measured:.5f}, "
        f"analytic γ={gamma_analytic:.5f}, ω={omega:.5f}"
    )
    np.testing.assert_allclose(gamma_measured, gamma_analytic, rtol=0.10, atol=2e-3)


if __name__ == "__main__":
    test_landau_damping_driven(0.3)
