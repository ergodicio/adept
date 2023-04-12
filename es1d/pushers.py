from typing import Dict, Tuple

from jax import numpy as jnp
from jax.nn import tanh
import haiku as hk
import numpy as np

from theory.electrostatic import get_roots_to_electrostatic_dispersion


def get_complex_frequency_table(num: int, kinetic_real_epw: bool) -> Tuple[np.array, np.array, np.array]:
    """
    This function creates a table of the complex plasma frequency for $0.2 < k \lambda_D < 0.4$ in `num` steps

    :param kinetic_real_epw:
    :param num:
    :return:
    """
    klds = np.linspace(0.2, 0.4, num)
    wrs = np.zeros(num)
    wis = np.zeros(num)

    for i, kld in enumerate(klds):
        ww = get_roots_to_electrostatic_dispersion(1.0, 1.0, kld)
        if kinetic_real_epw:
            wrs[i] = np.real(ww)
        else:
            wrs[i] = np.sqrt(1.0 + 3.0 * kld**2.0)
        wis[i] = np.imag(ww)

    return wrs, wis, klds


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))


class Driver(hk.Module):
    def __init__(self, xax):
        super().__init__()
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


class PoissonSolver(hk.Module):
    def __init__(self, one_over_kx):
        super().__init__()
        self.one_over_kx = one_over_kx

    def __call__(self, dn):
        return jnp.real(jnp.fft.ifft(1j * self.one_over_kx * jnp.fft.fft(dn)))


class StepAmpere(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, n, u):
        return n * u


def gradient(arr, kx):
    return jnp.real(jnp.fft.ifft(1j * kx * jnp.fft.fft(arr)))


class DensityStepper(hk.Module):
    def __init__(self, kx):
        super().__init__()
        self.kx = kx

    def __call__(self, n, u):
        return -u * gradient(n, self.kx) - n * gradient(u, self.kx)


class VelocityStepper(hk.Module):
    def __init__(self, kx, kxr, physics):
        super().__init__()
        self.kx = kx

        wrs, wis, klds = get_complex_frequency_table(128, physics["kinetic_real_wepw"])
        wrs = jnp.array(jnp.interp(kxr, klds, wrs, left=0.0, right=wrs[-1]))
        self.wr_corr = wrs / jnp.sqrt(1 + 3 * kxr**2.0)
        self.absorption_coeff = 1.0  # 0.85

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
            - self.absorption_coeff * q_over_m_times_e
            + self.landau_damping_term(u) / (1.0 + delta**2)
        )


class EnergyStepper(hk.Module):
    def __init__(self, kx, gamma):
        super().__init__()
        self.kx = kx
        self.gamma = gamma

    def __call__(self, n, u, p_over_m, q_over_m_times_e):
        return (
            -u * gradient(p_over_m, self.kx)
            - self.gamma * p_over_m * gradient(u, self.kx)
            - 2 * n * u * q_over_m_times_e
        )


class ParticleTrapper(hk.Module):
    def __init__(self, cfg, species="electron", train=True):  # kld, nuee, kxr, kx, kinetic_real_epw):
        super().__init__()
        nuee = cfg["physics"][species]["trapping"]["nuee"]
        nn = cfg["physics"][species]["trapping"]["nn"]
        kxr = np.array(cfg["grid"]["kxr"])
        one_over_kxr = np.zeros(kxr.size)
        one_over_kxr[1:] = 1.0 / kxr[1:]
        kx = cfg["grid"]["kx"]
        kinetic_real_epw = cfg["physics"]["kinetic_real_wepw"]

        self.kxr = kxr
        self.kx = kx
        table_wrs, table_wis, table_klds = get_complex_frequency_table(128, kinetic_real_epw)
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
