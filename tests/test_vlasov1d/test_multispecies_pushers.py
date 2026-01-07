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
    """Test SpaceExponential convergence using method of manufactured solutions.

    The space pusher solves: ∂f/∂t + v ∂f/∂x = 0
    Exact characteristic solution: f(x, v, t+dt) = f(x - v*dt, v, t)

    Tests convergence by refining nx and verifying error decreases at 2nd order or better.
    """
    Lx = 2 * np.pi
    nv = 1  # Only testing spatial dimension
    v = jnp.array([0.5])  # Single velocity value

    # Test multiple resolutions
    nx_values = [16, 32, 64, 128]
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

    # Verify 2nd order convergence or better (or already at machine precision)
    # For doubling resolution (factor of 2), error should decrease by at least factor of 4 (2nd order)
    convergence_rates = []
    for i in range(len(errors) - 1):
        ratio = nx_values[i+1] / nx_values[i]
        error_ratio = errors[i] / errors[i+1]
        # Convergence order: log(error_ratio) / log(ratio)
        if errors[i+1] > 1e-14 and errors[i] > 1e-14:  # Above machine precision
            order = np.log(error_ratio) / np.log(ratio)
            convergence_rates.append(order)

    # Check that we have good convergence or are at machine precision
    has_good_convergence = any(order >= 2.0 for order in convergence_rates) if convergence_rates else False
    at_machine_precision = all(err < 1e-12 for err in errors)

    assert has_good_convergence or at_machine_precision, \
        f"No 2nd order convergence and not at machine precision. " \
        f"Convergence orders: {[f'{o:.2f}' for o in convergence_rates]}, " \
        f"Errors: {[f'{e:.2e}' for e in errors]}"


def test_velocity_exponential_multispecies_qm_ratio():
    """Test that VelocityExponential correctly applies different q/m ratios to different species.

    With same initial conditions and E field, the velocity shift should be proportional to q/m.
    """
    nx = 32
    nv = 64
    vmax = 6.0
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv/2, vmax - dv/2, nv)
    kvr = jnp.fft.rfftfreq(nv, d=dv/(2*np.pi))

    species_grids = {
        "electron": {"kvr": kvr, "nv": nv, "v": v},
        "ion": {"kvr": kvr, "nv": nv, "v": v},
    }

    qm_electron = -1.0
    qm_ion = 1.0 / 1836.0

    species_params = {
        "electron": {"charge": -1.0, "mass": 1.0, "charge_to_mass": qm_electron},
        "ion": {"charge": 1.0, "mass": 1836.0, "charge_to_mass": qm_ion},
    }

    pusher = VelocityExponential(species_grids, species_params)

    # Same initial condition for both species (Gaussian)
    v_center = 0.0
    v_width = 1.0
    f_init_v = jnp.exp(-((v - v_center) / v_width)**2)
    f_init = jnp.ones((nx, 1)) * f_init_v[None, :]

    e = jnp.ones(nx) * 0.5
    dt = 0.01

    f_dict = {"electron": f_init, "ion": f_init}
    result = pusher(f_dict, e, dt)

    # Compute center of mass in velocity for each species
    def velocity_moment(f):
        return jnp.sum(v[None, :] * f, axis=1) / jnp.sum(f, axis=1)

    v_cm_electron = jnp.mean(velocity_moment(result["electron"]))
    v_cm_ion = jnp.mean(velocity_moment(result["ion"]))

    # Expected shifts
    shift_electron = -qm_electron * e[0] * dt
    shift_ion = -qm_ion * e[0] * dt

    # Check that the velocity shifts have the correct ratio
    ratio_expected = shift_electron / shift_ion
    ratio_actual = (v_cm_electron - v_center) / (v_cm_ion - v_center)

    # Ratio should match q/m ratio (with some tolerance for numerical error)
    assert jnp.abs(ratio_actual - ratio_expected) / jnp.abs(ratio_expected) < 0.01, \
        f"q/m ratio not correctly applied: expected {ratio_expected}, got {ratio_actual}"


if __name__ == "__main__":
    test_space_exponential_convergence()
    test_velocity_exponential_multispecies_qm_ratio()
    print("All pusher tests passed!")
