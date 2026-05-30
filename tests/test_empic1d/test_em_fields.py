"""Validation of the transverse Yee field solver (Inc 5a), particles aside.

A standing EM wave in vacuum rings at ω = c·k (up to the Yee numerical
dispersion, negligible for a well-resolved mode). This isolates the
Faraday/Ampère updates from the particle coupling.
"""

import jax
import numpy as np
from jax import numpy as jnp

from adept._empic1d.laser import soft_source_jy
from adept._empic1d.solvers.pushers.field import advance_bz_faces, advance_ey_nodes
from adept._empic1d.solvers.vector_field import em_step


def _dominant_frequency(signal, dt):
    sig = signal - signal.mean()
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=dt) * 2.0 * np.pi
    k = int(np.argmax(spec[1:])) + 1
    a, b, cc = spec[k - 1], spec[k], spec[k + 1]
    denom = a - 2.0 * b + cc
    offset = 0.5 * (a - cc) / denom if denom != 0 else 0.0
    return freqs[k] + offset * (freqs[1] - freqs[0])


def test_vacuum_em_wave_dispersion():
    c = 1.0
    L = 2.0 * np.pi
    nx = 128
    dx = L / nx
    dt = 0.5 * dx / c  # CFL number 0.5
    mode = 2
    k = 2.0 * np.pi * mode / L

    x_node = (jnp.arange(nx)) * dx  # nodes at i·dx
    e_y = jnp.sin(k * x_node)
    b_z = jnp.zeros(nx)

    def step(carry, _):
        e, b = carry
        b = advance_bz_faces(b, e, dx, dt)  # B^{n+½} from E^n
        e = advance_ey_nodes(e, b, 0.0, c, dx, dt)  # E^{n+1} from B^{n+½}
        return (e, b), jnp.fft.rfft(e)[mode]

    n_steps = 4000
    _, mode_hist = jax.lax.scan(step, (e_y, b_z), None, length=n_steps)

    omega = _dominant_frequency(np.real(np.asarray(mode_hist)), dt)
    assert abs(omega - c * k) / (c * k) < 0.01, f"omega={omega:.4f} vs ck={c * k:.4f}"


def test_em_wave_in_plasma_dispersion():
    """A transverse EM wave in cold plasma rings at ω² = ω_pe² + c²k².

    This validates the self-consistent transverse current j_y coupling: the
    plasma response supplies the ω_pe² offset above the vacuum branch.
    """
    c = 2.0
    L = 2.0 * np.pi
    nx = 64
    dx = L / nx
    dt = 0.02
    mode = 1
    k = 2.0 * np.pi * mode / L
    omega_expected = np.sqrt(1.0 + (c * k) ** 2)  # ω_pe = 1

    # Cold quiet plasma, density 1 ⇒ ω_pe = 1.
    ppc = 100
    n_p = nx * ppc
    xp = jnp.array((np.arange(n_p) + 0.5) * (L / n_p))
    wp = jnp.full((n_p,), L / n_p)
    up = jnp.zeros((n_p, 3))

    x_node = jnp.arange(nx) * dx
    state = {
        "species": {"electron": {"x": xp, "u": up, "w": wp}},
        "E": jnp.zeros(nx),
        "Ey": 1e-3 * jnp.sin(k * x_node),
        "Bz": jnp.zeros(nx),
    }
    params = dict(
        species_params={"electron": {"charge": -1.0, "qm": -1.0}},
        dt=dt,
        c=c,
        nx=nx,
        dx=dx,
        xmin=0.0,
        length=L,
        shape="tsc",
    )

    def step(s, _):
        s = em_step(s, **params)
        return s, jnp.fft.rfft(s["Ey"])[mode]

    _, mode_hist = jax.lax.scan(step, state, None, length=4000)
    omega = _dominant_frequency(np.real(np.asarray(mode_hist)), dt)
    assert abs(omega - omega_expected) / omega_expected < 0.02, f"omega={omega:.4f} vs {omega_expected:.4f}"


def test_laser_soft_source_propagates_at_c():
    """A soft-source laser pulse launched in vacuum travels at c."""
    c = 1.0
    L = 120.0
    nx = 600
    dx = L / nx
    dt = 0.5 * dx / c
    x_node = jnp.arange(nx) * dx
    x0 = 20.0
    omega0 = 3.0
    src = dict(x0=x0, sigma_x=1.0, omega0=omega0, amplitude=1.0, t0=2.5, tau=2.0)

    state = {"species": {}, "E": jnp.zeros(nx), "Ey": jnp.zeros(nx), "Bz": jnp.zeros(nx)}
    params = dict(species_params={}, dt=dt, c=c, nx=nx, dx=dx, xmin=0.0, length=L, shape="tsc")

    times = jnp.arange(400) * dt

    def step(s, t):
        s = em_step(s, **params, j_y_source=soft_source_jy(x_node, t, **src))
        return s, s["Ey"]

    _, ey_hist = jax.lax.scan(step, state, times)
    ey_hist = np.asarray(ey_hist)
    xn = np.asarray(x_node)

    # Centroid of the right-going pulse energy in a window ahead of the antenna.
    window = (xn > x0 + 3.0) & (xn < x0 + 70.0)

    def centroid(step_idx):
        w = (ey_hist[step_idx] ** 2) * window
        return float(np.sum(xn * w) / np.sum(w))

    i1, i2 = 120, 320
    speed = (centroid(i2) - centroid(i1)) / ((i2 - i1) * dt)
    assert abs(speed - c) / c < 0.03, f"pulse speed {speed:.4f} vs c={c}"
