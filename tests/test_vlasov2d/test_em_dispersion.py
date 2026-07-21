"""EM dispersion test for the Vlasov-2D solver.

A transverse current source Jy(x, t) at (kx0, omega0) drives the (Ey, Bz)
EM wave. In the cold-plasma limit the dispersion is

    omega^2 = omega_p^2 + c^2 kx^2

With normalized omega_p = 1 and c = c_norm (read from the config's plasma
normalization), we drive at the predicted resonance and verify the Bz
oscillation frequency matches omega0 within a few percent.
"""

import os

import numpy as np
import pytest
import yaml

from adept import ergoExo
from adept.normalization import electron_debye_normalization

_HERE = os.path.dirname(__file__)


def _load_cfg() -> dict:
    with open(os.path.join(_HERE, "configs", "em_dispersion.yaml")) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("kx0", [0.5, 0.7])
def test_em_dispersion(kx0):
    cfg = _load_cfg()

    norm = electron_debye_normalization(cfg["units"]["normalizing_density"], cfg["units"]["normalizing_temperature"])
    c_norm = float(norm.speed_of_light_norm())
    omega0 = float(np.sqrt(1.0 + (c_norm * kx0) ** 2))

    cfg["grid"]["xmax"] = float(2.0 * np.pi / kx0)
    cfg["drivers"]["ey"]["0"]["params"]["k0x"] = float(kx0)
    cfg["drivers"]["ey"]["0"]["params"]["w0"] = omega0
    cfg["mlflow"]["run"] = f"kx-{kx0}"

    exo = ergoExo()
    exo.setup(cfg)
    result, _, _ = exo(None)
    sol = result["solver result"]

    # Bz shape (nt, nx, ny); take the (kx0, 0) Fourier component
    bz_t = np.asarray(sol.ys["fields"]["bz"])  # (nt, nx, ny)
    t_ax = np.asarray(sol.ts["fields"])
    nx = bz_t.shape[1]
    bz_k = (2.0 / nx) * np.fft.fft(bz_t, axis=1)[:, 1, :].mean(axis=-1)
    # real part oscillates at omega0; analyze a window centred under the driver
    t0_win = 4.0
    t1_win = float(t_ax[-1])
    m = (t_ax >= t0_win) & (t_ax <= t1_win)
    sig = np.real(bz_k[m])
    t_win = t_ax[m]

    # FFT-based dominant-frequency estimate
    dt = float(np.mean(np.diff(t_win)))
    freqs = np.fft.rfftfreq(len(t_win), d=dt) * 2 * np.pi
    amp = np.abs(np.fft.rfft(sig - sig.mean()))
    omega_meas = float(freqs[np.argmax(amp)])

    print(f"\n[kx={kx0}] EM dispersion: measured ω={omega_meas:.4f}, analytic ω={omega0:.4f} (c_norm={c_norm:.3f})")
    # Frequency resolution of the FFT is 2π/T ≈ 2π/16 ≈ 0.39, so allow 5% rel
    # tolerance with a small absolute floor matching ~one bin.
    np.testing.assert_allclose(omega_meas, omega0, rtol=0.05, atol=0.5)


if __name__ == "__main__":
    test_em_dispersion(0.5)
