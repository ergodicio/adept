"""Self-consistent kinetic verification against independent linear Fourier IVPs."""

import jax
import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, root
from scipy.special import wofz

from adept import SimulationSpec, run_prepared, solver_registry


def _linear_reference(times, *, wave_number, epsilon, drift, thermal_speed, amplitude, vmax):
    """Continuum Fourier-mode Vlasov IVP with the same softened Poisson kernel.

    For electrons, E_k = i mu/k integral(g_k dv) and dg_k/dt =
    -i k v g_k + E_k f0'. The smooth velocity integral is independently
    resolved on 513 nodes, and SciPy DOP853 integrates the complex mode.
    """
    length = 2 * np.pi / wave_number
    alpha = epsilon / length

    def kernel(distance):
        r = distance / length
        return 0.5 * r * np.sqrt(1 + 4 * alpha**2) / np.sqrt(r**2 + alpha**2) - r

    multiplier = (
        2
        * wave_number
        * quad(
            lambda distance: kernel(distance) * np.sin(wave_number * distance),
            0.0,
            length / 2,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
    )
    velocity = np.linspace(-vmax, vmax, 513)
    weights = np.full(velocity.shape, velocity[1] - velocity[0])
    weights[[0, -1]] *= 0.5
    plus = np.exp(-0.5 * ((velocity - drift) / thermal_speed) ** 2) / (thermal_speed * np.sqrt(2 * np.pi))
    minus = np.exp(-0.5 * ((velocity + drift) / thermal_speed) ** 2) / (thermal_speed * np.sqrt(2 * np.pi))
    distribution = 0.5 * (plus + minus)
    derivative = -0.5 * ((velocity - drift) * plus + (velocity + drift) * minus) / thermal_speed**2

    def rhs(_time, mode):
        field = 1j * multiplier * np.sum(weights * mode) / wave_number
        return -1j * wave_number * velocity * mode + field * derivative

    solution = solve_ivp(
        rhs,
        (0.0, times[-1]),
        (0.5 * amplitude * distribution).astype(complex),
        t_eval=times,
        method="DOP853",
        rtol=1e-11,
        atol=1e-14,
    )
    assert solution.success
    fields = 1j * multiplier * np.sum(weights[:, None] * solution.y, axis=0) / wave_number

    def dielectric(frequency):
        zeta = (frequency / wave_number + np.array([-drift, drift])) / (np.sqrt(2) * thermal_speed)
        plasma_dispersion = 1j * np.sqrt(np.pi) * wofz(zeta)
        return 1 + multiplier * np.sum(1 + zeta * plasma_dispersion) / (2 * wave_number**2 * thermal_speed**2)

    if drift:
        frequency = 1j * brentq(lambda growth: dielectric(1j * growth).real, 0.01, 0.6, xtol=1e-13)
    else:

        def root_function(parts):
            value = dielectric(parts[0] + 1j * parts[1])
            return np.array([value.real, value.imag])

        pole = root(root_function, [1.3, -0.2], tol=1e-11)
        assert pole.success
        frequency = pole.x[0] + 1j * pole.x[1]
        assert abs(dielectric(frequency)) < 1e-10
    return fields, frequency, multiplier


@pytest.mark.parametrize("kind", ["two-stream", "maxwellian"])
def test_electric_field_matches_linear_fourier_initial_value_problem(kind):
    """Compare the complex field history, including the initial stable transient.

    This is an intentionally resolved regularized model, not evidence for
    epsilon->0 or turbulent late-time convergence. A narrow kernel on this
    coarse x lattice would produce large quadrature aliases between nodes.
    """
    if kind == "two-stream":
        wave_number, epsilon, drift, thermal_speed, vmax = 0.3, 1.5, 2.0, 0.3, 4.0
    else:
        wave_number, epsilon, drift, thermal_speed, vmax = 0.5, 0.9, 0.0, 1.0, 6.0
    amplitude = 1e-4
    config = {
        "grid": {"nx": 32, "nv": 64, "xmin": 0.0, "xmax": 2 * np.pi / wave_number, "vmin": -vmax, "vmax": vmax},
        "time": {"tmin": 0.0, "tmax": 12.0, "dt": 0.05},
        "initial": {"kind": kind, "thermal_speed": thermal_speed, "drift": drift, "amplitude": amplitude, "mode": 1},
        "numerical": {"epsilon": epsilon, "quadrature": "trapezoid", "remesh_every": 1, "chunk_size": 64},
        "save": {"scalars": {"every_steps": 2}, "fields": {"every_steps": 2}, "distribution": None},
    }
    prepared = solver_registry.prepare(SimulationSpec("farsight-1d", config), key=0)
    completed = run_prepared(prepared, key=jax.random.key(0))
    fields = completed.report.result["fields"]
    mode = np.fft.rfft(fields.electric_field, axis=1)[:, 1] / config["grid"]["nx"]
    reference, frequency, multiplier = _linear_reference(
        np.asarray(fields.t),
        wave_number=wave_number,
        epsilon=epsilon,
        drift=drift,
        thermal_speed=thermal_speed,
        amplitude=amplitude,
        vmax=vmax,
    )
    relative_error = np.linalg.norm(mode - reference) / np.linalg.norm(reference)
    scalars = completed.report.result["scalars"]
    relative_mass_change = float((scalars.mass[-1] - scalars.mass[0]) / scalars.mass[0])
    relative_c2_change = float((scalars.c2[-1] - scalars.c2[0]) / scalars.c2[0])
    relative_negative_mass = float((scalars.negative_mass / scalars.mass).max())
    print(
        f"{kind}: kernel multiplier={multiplier:.9g}, asymptotic omega={frequency:.9g}, "
        f"field-history relative L2={relative_error:.9g}, mass change={relative_mass_change:.9g}, "
        f"C2 change={relative_c2_change:.9g}, max negative fraction={relative_negative_mass:.9g}, "
        f"final/initial mode={abs(mode[-1]) / abs(mode[0]):.9g}, run seconds={completed.run_time_seconds:.3f}"
    )
    assert bool(scalars.valid.all())
    assert int(scalars.invalid_panels.max()) == 0
    assert relative_error < 0.02
    if kind == "two-stream":
        assert abs(mode[-1]) > 3 * abs(mode[0])
    else:
        assert max(abs(mode[np.asarray(fields.t) >= 9.0])) < 0.4 * abs(mode[0])
    assert abs(relative_mass_change) < 1e-4
    assert abs(relative_c2_change) < 1e-3
    np.testing.assert_allclose(scalars.remap_c2_change[-1], scalars.c2[-1] - scalars.c2[0], rtol=1e-9, atol=2e-13)
