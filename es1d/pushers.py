from typing import Dict, Tuple, Callable

import jax
from jax import numpy as jnp
from jax.nn import tanh
import numpy as np
import equinox as eqx

from theory.electrostatic import get_complex_frequency_table


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))


class Driver(eqx.Module):
    xax: jax.Array

    def __init__(self, xax):
        self.xax = xax

    def __call__(self, this_pulse: Dict, current_time: jnp.float64):
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
        envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, self.xax)

        return (
            envelope_t * envelope_x * jnp.abs(kk) * this_pulse["a0"] * jnp.sin(kk * self.xax - (ww + dw) * current_time)
        )


class PoissonSolver(eqx.Module):
    one_over_kx: jax.Array

    def __init__(self, one_over_kx):
        self.one_over_kx = one_over_kx

    def __call__(self, dn):
        return jnp.real(jnp.fft.ifft(1j * self.one_over_kx * jnp.fft.fft(dn)))


class StepAmpere(eqx.Module):
    def __call__(self, n, u):
        return n * u


def gradient(arr, kx):
    return jnp.real(jnp.fft.ifft(1j * kx * jnp.fft.fft(arr)))


class DensityStepper(eqx.Module):
    kx: jax.Array

    def __init__(self, kx):
        self.kx = kx

    def __call__(self, n, u):
        return -u * gradient(n, self.kx) - n * gradient(u, self.kx)


class VelocityStepper(eqx.Module):
    kx: jax.Array
    wr_corr: jax.Array
    wis: jax.Array

    def __init__(self, kx, kxr, one_over_kxr, physics):
        self.kx = kx

        wrs, wis, klds = get_complex_frequency_table(1024, True if physics["gamma"] == "kinetic" else False)
        wrs = jnp.array(jnp.interp(kxr, klds, wrs, left=1.0, right=wrs[-1]))

        if physics["gamma"] == "kinetic":
            self.wr_corr = (jnp.square(wrs) - 1.0) * one_over_kxr**2.0
        else:
            self.wr_corr = 1.0

        if physics["landau_damping"]:
            self.wis = jnp.array(jnp.interp(kxr, klds, wis, left=0.0, right=wis[-1]))
        else:
            self.wis = jnp.zeros_like(kxr)

    def landau_damping_term(self, u):
        return 2 * jnp.real(jnp.fft.irfft(self.wis * jnp.fft.rfft(u)))

    def restoring_force_term(self, gradp_over_nm):
        return jnp.real(jnp.fft.irfft(self.wr_corr * jnp.fft.rfft(gradp_over_nm)))

    def __call__(self, n, u, p_over_m, q_over_m_times_e, delta):
        return (
            -u * gradient(u, self.kx)
            - self.restoring_force_term(gradient(p_over_m, self.kx) / n)
            - q_over_m_times_e
            + self.landau_damping_term(u) / (1.0 + delta**2)
        )


class EnergyStepper(eqx.Module):
    kx: jax.Array
    gamma: float

    def __init__(self, kx, physics):
        self.kx = kx
        if physics["gamma"] == "kinetic":
            self.gamma = 1.0
        else:
            self.gamma = physics["gamma"]

    def __call__(self, n, u, p_over_m, q_over_m_times_e):
        return (
            -u * gradient(p_over_m, self.kx)
            - self.gamma * p_over_m * gradient(u, self.kx)
            - 2 * n * u * q_over_m_times_e
        )


class ParticleTrapper(eqx.Module):
    kxr: jax.Array
    kx: jax.Array
    model_kld: float
    wrs: jax.Array
    wis: jax.Array
    table_klds: jax.Array
    norm_kld: jnp.float64
    norm_nuee: jnp.float64
    vph: jnp.float64
    nu_g_model: Callable
    nu_d_model: Callable

    def __init__(self, cfg, species="electron", train=False):  # kld, nuee, kxr, kx, kinetic_real_epw):
        nuee = cfg["physics"][species]["trapping"]["nuee"]
        nn = cfg["physics"][species]["trapping"]["nn"]
        kxr = np.array(cfg["grid"]["kxr"])
        one_over_kxr = np.zeros(kxr.size)
        one_over_kxr[1:] = 1.0 / kxr[1:]
        kx = cfg["grid"]["kx"]
        if cfg["physics"][species]["gamma"] == "kinetic":
            kinetic_real_epw = True
        else:
            kinetic_real_epw = False

        self.kxr = kxr
        self.kx = kx
        table_wrs, table_wis, table_klds = get_complex_frequency_table(1024, kinetic_real_epw)
        self.model_kld = cfg["physics"][species]["trapping"]["kld"]
        self.wrs = jnp.interp(kxr, table_klds, table_wrs, left=1.0, right=table_wrs[-1])
        self.wis = jnp.interp(kxr, table_klds, table_wis, left=0.0, right=0.0)
        self.table_klds = table_klds
        self.norm_kld = (self.model_kld - 0.26) / 0.14
        self.norm_nuee = (jnp.log10(nuee) + 7.0) / -4.0
        self.vph = jnp.interp(self.model_kld, table_klds, table_wrs, left=1.0, right=table_wrs[-1]) / self.model_kld

        # Make models
        if train:
            nu_g_model = []
            for i, width in enumerate(nn.split("|")):
                nu_g_model += [hk.Linear(int(width), name=f"nu_g_model_linear_{i}"), tanh]
            nu_g_model += [hk.Linear(1, name=f"nu_g_model_linear_last"), tanh]
            self.nu_g_model = hk.Sequential(nu_g_model)
        else:
            self.nu_g_model = lambda x: 1e-3

        if train:
            nu_d_model = []
            for i, width in enumerate(nn.split("|")):
                nu_d_model += [hk.Linear(int(width), name=f"nu_d_model_linear_{i}"), tanh]
            nu_d_model += [hk.Linear(1, name=f"nu_d_model_linear_last"), tanh]
            self.nu_d_model = hk.Sequential(nu_d_model)
        else:
            self.nu_d_model = lambda x: 1e-3

    def __call__(self, e, delta, args):
        ek = jnp.fft.rfft(e, axis=0) * 2.0 / self.kx.size
        norm_e = (jnp.log10(jnp.interp(self.model_kld, self.kxr, jnp.abs(ek)) + 1e-10) + 10.0) / -10.0
        func_inputs = jnp.stack([norm_e, self.norm_kld, self.norm_nuee], axis=-1)
        growth_rates = 10 ** (3 * jnp.squeeze(self.nu_g_model(func_inputs)))
        damping_rates = 10 ** (3 * jnp.squeeze(self.nu_d_model(func_inputs)))

        return (
            -self.vph * gradient(delta, self.kx)
            + growth_rates * jnp.abs(jnp.fft.irfft(ek * self.kx.size / 2.0 * self.wis)) / (1.0 + delta**2.0)
            - damping_rates * delta
        )
