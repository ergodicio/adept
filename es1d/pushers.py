from jax import numpy as jnp


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))


def calc_driver(this_pulse, current_time):
    kk = this_pulse["k0"]
    ww = this_pulse["w0"]
    dw = this_pulse["dw0"]
    t_L = this_pulse["t_c"] - this_pulse["t_w"] * 0.5
    t_R = this_pulse["t_c"] + this_pulse["t_w"] * 0.5
    t_wL = this_pulse["t_r"]
    t_wR = this_pulse["t_r"]
    x_L = this_pulse["x_c"] - this_pulse["x_w"] * 0.5
    x_R = this_pulse["x_c"] + this_pulse["x_w"] * 0.5
    x_wL = this_pulse["x_r"]
    x_wR = this_pulse["x_r"]
    envelope_t = get_envelope(t_wL, t_wR, t_L, t_R, current_time)
    envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, xax)

    return envelope_t * envelope_x * jnp.abs(kk) * this_pulse["a0"] * jnp.sin(kk * xax - (ww + dw) * current_time)


def step_e_ampere(n, u):
    return n * u


def calc_e(n):
    ef = jnp.real(jnp.fft.ifft(1j * one_over_kx * jnp.fft.fft(1 - n)))
    return ef


def gradient(arr, dx):
    return jnp.fft.ifft(1j * kx * jnp.fft.fft(arr))


def step_n(n, u):
    return -u * gradient(n, dx) - n * gradient(u, dx)


def step_u(n, u, p, ef):
    corr = w0_corr * 1.25
    return -u * gradient(u, dx) - corr * gradient(p, dx) / n + ef + 2 * wi * u


def step_p(n, u, p, ef):
    T = p / n
    q = -2.0655 * n * jnp.sqrt(T) * gradient(T, dx)
    return -u * gradient(p, dx) - gamma * p * gradient(u, dx) - 0 * gradient(q, dx) + 2 * n * u * ef
