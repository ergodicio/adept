"""Gyrorotation test for the Vlasov-2D solver.

Initialise f as a Maxwellian drifted by v0x in vx, override Bz to a uniform
value, and freeze Maxwell so the only dynamics is the kinetic magnetic
rotation. The mean velocity should rotate at omega_c = q*Bz / m.

For an electron (q=-1, m=1) with Bz0 > 0, omega_c = -Bz0 — i.e. clockwise
rotation in the (vx, vy) plane.  We test:

    <vx>(t) = v0x cos(|omega_c| t)
    <vy>(t) = v0x sin(omega_c t) =  -v0x sin(|omega_c| t)   (since omega_c < 0)

Fit a sinusoid to <vx>(t) and verify the frequency matches |omega_c|.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from adept import ergoExo

_HERE = os.path.dirname(__file__)


def _load_cfg() -> dict:
    with open(os.path.join(_HERE, "configs", "gyrorotation.yaml")) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("bz0", [0.5, 1.0])
def test_gyrofrequency(bz0):
    cfg = _load_cfg()
    cfg["mlflow"]["run"] = f"bz-{bz0}"

    # tmax = ~2 gyroperiods
    omega_c = bz0  # magnitude; q/m = -1 for electron
    cfg["grid"]["tmax"] = float(4.0 * 2.0 * np.pi / omega_c)
    cfg["save"]["fields"]["t"]["nt"] = 400

    exo = ergoExo()
    exo.setup(cfg)
    # Override Bz to a uniform value (config can't initialize fields)
    nx = cfg["grid"]["nx"]
    ny = cfg["grid"]["ny"]
    exo.adept_module.state["bz"] = jnp.ones((nx, ny)) * float(bz0)

    result, _, _ = exo(None)
    sol = result["solver result"]

    # <vx>(t) is stored as "ux" under each species
    ux_field = np.asarray(sol.ys["fields"]["electron"]["ux"])  # (nt, nx, ny)
    t_ax = np.asarray(sol.ts["fields"])
    ux = ux_field.mean(axis=(1, 2))

    # FFT dominant frequency
    dt = float(np.mean(np.diff(t_ax)))
    freqs = np.fft.rfftfreq(len(t_ax), d=dt) * 2 * np.pi
    amp = np.abs(np.fft.rfft(ux - ux.mean()))
    omega_meas = float(freqs[np.argmax(amp[1:]) + 1])  # skip DC bin

    v0x = cfg["density"]["species-background"]["v0x"]
    print(
        f"\n[Bz={bz0}] gyrorotation: measured |ω_c|={omega_meas:.4f}, "
        f"analytic |ω_c|={omega_c:.4f}, v0x={v0x}, <ux>(0)={ux[0]:.4f}"
    )

    # The FFT frequency resolution is 2π/T ≈ 2π/12 ≈ 0.52 rad/t for Bz=1; the
    # spectral B-rotation is exact, so we expect agreement within the bin width.
    np.testing.assert_allclose(omega_meas, omega_c, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    test_gyrofrequency(1.0)
