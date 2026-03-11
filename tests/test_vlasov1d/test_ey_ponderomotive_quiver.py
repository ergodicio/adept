# Copyright (c) Ergodic LLC 2023
# research@ergodic.io
"""Test ponderomotive force handling in velocity pushers.

Verifies that the velocity pushers correctly compute the species-dependent
force from the ponderomotive potential gradient:

    force_s = q_s * E + (q_s² / m_s) * pond
    accel_s = force_s / m_s

where pond = -(1/2) * ∂(a²)/∂x is species-independent.

For electrons (q=-1, m=1): accel = -E + pond
For ions    (q=Z, m=M):   accel = (Z/M)*E + (Z²/M²)*pond

This test creates a uniform pond field and verifies each species is shifted
by the correct amount, independently of the electrostatic field E.
"""

import numpy as np
from jax import numpy as jnp

from adept._vlasov1d.solvers.pushers.vlasov import VelocityCubicSpline, VelocityExponential


def test_ponderomotive_velocity_push_exponential():
    """Test VelocityExponential correctly applies species-dependent ponderomotive force.

    Uses a sinusoidal distribution in velocity and verifies each species
    is shifted by the correct amount from both E and pond contributions.
    """
    nx = 1
    nv = 64
    vmax = 2 * np.pi
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
    kvr = jnp.fft.rfftfreq(nv, d=dv) * 2.0 * np.pi

    # Two species: electron (q=-1, m=1) and ion (q=10, m=18360)
    q_e, m_e = -1.0, 1.0
    q_i, m_i = 10.0, 18360.0

    species_grids = {
        "electron": {"kvr": kvr, "nv": nv, "v": v},
        "ion": {"kvr": kvr, "nv": nv, "v": v},
    }
    species_params = {
        "electron": {"charge": q_e, "mass": m_e, "charge_to_mass": q_e / m_e},
        "ion": {"charge": q_i, "mass": m_i, "charge_to_mass": q_i / m_i},
    }

    pusher = VelocityExponential(species_grids, species_params)

    # Known fields: uniform E and pond
    e = jnp.array([0.5])  # electrostatic field
    pond = jnp.array([-0.3])  # ponderomotive potential gradient = -(1/2)*grad(a²)
    dt = 0.01

    # Expected accelerations:
    # electron: force = q*E + (q²/m)*pond = -0.5 + 1*(-0.3) = -0.8
    #           accel = force/m = -0.8
    # ion:      force = 10*0.5 + (100/18360)*(-0.3) = 5.0 - 0.001634...
    #           accel = force/18360 = 2.724e-4 - 8.9e-8
    accel_e = (q_e * 0.5 + (q_e**2 / m_e) * (-0.3)) / m_e
    accel_i = (q_i * 0.5 + (q_i**2 / m_i) * (-0.3)) / m_i

    # Sinusoidal initial condition (periodic over [-vmax, vmax])
    n_mode = 1
    k = np.pi * n_mode / vmax
    f_init = jnp.sin(k * v)[None, :]  # shape (1, nv)

    f_dict = {"electron": f_init, "ion": f_init}
    result = pusher(f_dict, e, pond, dt)

    # Exact solution: f(v, t+dt) = f(v - accel*dt, t)
    v_shift_e = accel_e * dt
    v_shift_i = accel_i * dt
    f_electron_exact = jnp.sin(k * (v - v_shift_e))[None, :]
    f_ion_exact = jnp.sin(k * (v - v_shift_i))[None, :]

    error_e = float(jnp.sqrt(jnp.mean((result["electron"] - f_electron_exact) ** 2)))
    error_i = float(jnp.sqrt(jnp.mean((result["ion"] - f_ion_exact) ** 2)))

    print("\nPonderomotive Velocity Push Test (Exponential)")
    print(f"  E = {float(e[0])}, pond = {float(pond[0])}")
    print(f"  Electron: accel = {accel_e:.6f}, v_shift = {v_shift_e:.8f}, error = {error_e:.2e}")
    print(f"  Ion:      accel = {accel_i:.6f}, v_shift = {v_shift_i:.8f}, error = {error_i:.2e}")

    assert error_e < 1e-12, f"Electron ponderomotive push error {error_e:.2e} exceeds tolerance"
    assert error_i < 1e-12, f"Ion ponderomotive push error {error_i:.2e} exceeds tolerance"


def test_ponderomotive_velocity_push_cubic_spline():
    """Test VelocityCubicSpline correctly applies species-dependent ponderomotive force."""
    nx = 1
    nv = 128
    vmax = 2 * np.pi
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    q_e, m_e = -1.0, 1.0
    q_i, m_i = 10.0, 18360.0

    species_grids = {
        "electron": {"v": v, "nv": nv},
        "ion": {"v": v, "nv": nv},
    }
    species_params = {
        "electron": {"charge": q_e, "mass": m_e, "charge_to_mass": q_e / m_e},
        "ion": {"charge": q_i, "mass": m_i, "charge_to_mass": q_i / m_i},
    }

    pusher = VelocityCubicSpline(species_grids, species_params)

    e = jnp.array([0.5])
    pond = jnp.array([-0.3])
    dt = 0.001  # Small dt for cubic spline accuracy

    accel_e = (q_e * 0.5 + (q_e**2 / m_e) * (-0.3)) / m_e
    accel_i = (q_i * 0.5 + (q_i**2 / m_i) * (-0.3)) / m_i

    # Use a smooth Gaussian-like function that's well-resolved
    f_init = jnp.exp(-0.5 * (v / 1.0) ** 2)[None, :]
    f_dict = {"electron": f_init, "ion": f_init}
    result = pusher(f_dict, e, pond, dt)

    # Exact: f(v, t+dt) = f(v - accel*dt, t) = exp(-0.5 * ((v - accel*dt)/1.0)²)
    v_shift_e = accel_e * dt
    v_shift_i = accel_i * dt
    f_electron_exact = jnp.exp(-0.5 * ((v - v_shift_e) / 1.0) ** 2)[None, :]
    f_ion_exact = jnp.exp(-0.5 * ((v - v_shift_i) / 1.0) ** 2)[None, :]

    error_e = float(jnp.sqrt(jnp.mean((result["electron"] - f_electron_exact) ** 2)))
    error_i = float(jnp.sqrt(jnp.mean((result["ion"] - f_ion_exact) ** 2)))

    print("\nPonderomotive Velocity Push Test (Cubic Spline)")
    print(f"  E = {float(e[0])}, pond = {float(pond[0])}")
    print(f"  Electron: accel = {accel_e:.6f}, v_shift = {v_shift_e:.8f}, error = {error_e:.2e}")
    print(f"  Ion:      accel = {accel_i:.6f}, v_shift = {v_shift_i:.8f}, error = {error_i:.2e}")

    # Cubic spline has higher error than spectral but should still be very accurate
    assert error_e < 1e-6, f"Electron cubic spline ponderomotive error {error_e:.2e} exceeds tolerance"
    assert error_i < 1e-6, f"Ion cubic spline ponderomotive error {error_i:.2e} exceeds tolerance"


def test_pond_zero_recovers_original():
    """Verify that pond=0 gives the same result as the old q/m * E behavior."""
    nx = 1
    nv = 32
    vmax = 2 * np.pi
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
    kvr = jnp.fft.rfftfreq(nv, d=dv) * 2.0 * np.pi

    species_grids = {"electron": {"kvr": kvr, "nv": nv, "v": v}}
    species_params = {"electron": {"charge": -1.0, "mass": 1.0, "charge_to_mass": -1.0}}

    pusher = VelocityExponential(species_grids, species_params)

    e = jnp.array([0.5])
    pond = jnp.zeros_like(e)  # No ponderomotive force
    dt = 0.01

    f_init = jnp.sin(np.pi / vmax * v)[None, :]
    f_dict = {"electron": f_init}
    result = pusher(f_dict, e, pond, dt)

    # With pond=0: accel = q*E/m = -0.5, same as old q/m * E
    qm = -1.0
    v_shift = qm * float(e[0]) * dt
    k = np.pi / vmax
    f_exact = jnp.sin(k * (v - v_shift))[None, :]

    error = float(jnp.sqrt(jnp.mean((result["electron"] - f_exact) ** 2)))
    print(f"\nPond=0 recovery test: error = {error:.2e}")
    assert error < 1e-12, f"pond=0 should recover old q/m*E behavior, error = {error:.2e}"


if __name__ == "__main__":
    test_ponderomotive_velocity_push_exponential()
    test_ponderomotive_velocity_push_cubic_spline()
    test_pond_zero_recovers_original()
    print("\nAll ponderomotive pusher tests passed!")
