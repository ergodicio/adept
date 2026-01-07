"""Test multi-species pushers using method of manufactured solutions.

These tests verify convergence to exact solutions obtained by characteristic tracing,
since the pushers implement linear advection operators in space.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import yaml

from adept._vlasov1d.solvers.pushers.vlasov import VelocityExponential, VelocityCubicSpline, SpaceExponential


def test_space_exponential_convergence():
    """Test SpaceExponential achieves machine precision using method of manufactured solutions.

    The space pusher solves: ∂f/∂t + v ∂f/∂x = 0
    Exact characteristic solution: f(x, v, t+dt) = f(x - v*dt, v, t)

    The spectral method achieves machine precision even at low resolution.
    """
    Lx = 2 * np.pi
    nv = 1  # Only testing spatial dimension
    v = jnp.array([0.5])  # Single velocity value

    # Test at two resolutions to verify machine precision
    nx_values = [16, 32]
    errors = []

    for nx in nx_values:
        dx = Lx / nx
        x = jnp.linspace(0, Lx - dx, nx)

        species_grids = {
            "electron": {"v": v, "nv": nv},
        }

        pusher = SpaceExponential(x, species_grids)

        # Manufactured solution: sinusoidal in space with single velocity
        k = 2  # wavenumber
        f_init = jnp.sin(k * x)[:, None]  # Shape (nx, 1)

        dt = 0.01

        # Apply pusher
        f_dict = {"electron": f_init}
        result = pusher(f_dict, dt)
        f_numerical = result["electron"]

        # Exact solution: shift in x by -v*dt
        # For sinusoidal initial condition: sin(k*x) -> sin(k*(x - v*dt)) = sin(k*x - k*v*dt)
        f_exact = jnp.sin(k * x - k * v[0] * dt)[:, None]  # Shape (nx, 1)

        # Compute error (L2 norm)
        error = jnp.sqrt(jnp.mean((f_numerical - f_exact)**2))
        errors.append(float(error))

    # Spectral method should achieve machine precision
    assert all(err < 1e-12 for err in errors), \
        f"SpaceExponential should achieve machine precision. Errors: {[f'{e:.2e}' for e in errors]}"


def test_velocity_exponential_convergence():
    """Test VelocityExponential achieves machine precision for multiple species.

    The velocity pusher solves: ∂f/∂t + (q/m)E ∂f/∂v = 0
    Exact characteristic solution: f(x, v, t+dt) = f(x, v - (q/m)*E*dt, t)

    Tests that each species is pushed by its respective q/m factor.
    The spectral method achieves machine precision even at low resolution.
    """
    nx = 1  # Only testing velocity dimension
    vmax = 2 * np.pi  # Periodic domain in velocity
    qm_electron = -1.0  # electron charge-to-mass ratio
    qm_ion = 1.0 / 1836.0  # ion charge-to-mass ratio (proton)
    e = jnp.array([0.5])  # Constant electric field
    dt = 0.01

    # Test at two resolutions to verify machine precision
    nv_values = [16, 32]
    errors_electron = []
    errors_ion = []

    for nv in nv_values:
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv/2, vmax - dv/2, nv)
        kvr = jnp.fft.rfftfreq(nv, d=dv) * 2.0 * np.pi

        species_grids = {
            "electron": {"kvr": kvr, "nv": nv, "v": v},
            "ion": {"kvr": kvr, "nv": nv, "v": v},
        }

        species_params = {
            "electron": {"charge": -1.0, "mass": 1.0, "charge_to_mass": qm_electron},
            "ion": {"charge": 1.0, "mass": 1836.0, "charge_to_mass": qm_ion},
        }

        pusher = VelocityExponential(species_grids, species_params)

        # Manufactured solution: sinusoidal in velocity
        # k must be chosen so function is periodic over [-vmax, vmax]
        # Period = 2*vmax, so k = 2*pi*n / (2*vmax) = pi*n/vmax
        n_mode = 1  # Choose mode number
        k = np.pi * n_mode / vmax  # This ensures periodicity
        f_init = jnp.sin(k * v)[None, :]  # Shape (1, nv)

        # Apply pusher to both species with same initial condition
        f_dict = {"electron": f_init, "ion": f_init}
        result = pusher(f_dict, e, dt)
        f_electron_numerical = result["electron"]
        f_ion_numerical = result["ion"]

        # Exact solution using characteristic solution: f(v, t+dt) = f(v - (q/m)*E*dt, t)
        # Each species should be shifted by its own q/m factor
        v_shift_electron = qm_electron * e[0] * dt
        v_shift_ion = qm_ion * e[0] * dt

        f_electron_exact = jnp.sin(k * (v - v_shift_electron))[None, :]
        f_ion_exact = jnp.sin(k * (v - v_shift_ion))[None, :]

        # Compute errors (L2 norm)
        error_electron = jnp.sqrt(jnp.mean((f_electron_numerical - f_electron_exact)**2))
        error_ion = jnp.sqrt(jnp.mean((f_ion_numerical - f_ion_exact)**2))

        errors_electron.append(float(error_electron))
        errors_ion.append(float(error_ion))

    # Spectral method should achieve machine precision for both species
    assert all(err < 1e-12 for err in errors_electron), \
        f"VelocityExponential should achieve machine precision for electrons. Errors: {[f'{e:.2e}' for e in errors_electron]}"
    assert all(err < 1e-12 for err in errors_ion), \
        f"VelocityExponential should achieve machine precision for ions. Errors: {[f'{e:.2e}' for e in errors_ion]}"


if __name__ == "__main__":
    test_space_exponential_convergence()
    test_velocity_exponential_convergence()
    print("All pusher tests passed!")
