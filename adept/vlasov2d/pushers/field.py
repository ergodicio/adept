#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from typing import Dict, Callable
from functools import partial

from jax import numpy as jnp
import equinox as eqx


class SpectralPoissonSolver(eqx.Module):
    ion_charge: jnp.array
    one_over_kx: jnp.array
    one_over_ky: jnp.array
    dvx: float
    dvy: float

    def __init__(self, ion_charge, one_over_kx, one_over_ky, dvx, dvy):
        super(SpectralPoissonSolver, self).__init__()
        self.ion_charge = jnp.array(ion_charge)
        self.one_over_kx = jnp.array(one_over_kx)
        self.one_over_ky = jnp.array(one_over_ky)
        self.dvx = dvx
        self.dvy = dvy

    def compute_charges(self, f):
        return jnp.trapz(jnp.trapz(f, dx=self.dvy, axis=3), dx=self.dvx, axis=2)

    def __call__(self, f: jnp.ndarray, prev_force: jnp.ndarray, dt: jnp.float64):
        dcharge = self.ion_charge - self.compute_charges(f)
        return jnp.concatenate(
            [
                jnp.real(jnp.fft.ifft(1j * self.one_over_kx[:, None] * jnp.fft.fft(dcharge, axis=0), axis=0))[
                    ..., None
                ],
                jnp.real(jnp.fft.ifft(1j * self.one_over_ky[None, :] * jnp.fft.fft(dcharge, axis=1), axis=1))[
                    ..., None
                ],
            ],
            axis=-1,
        )


class AmpereSolver(eqx.Module):
    vx: jnp.array
    vy: jnp.array
    moment_x: Callable
    moment_y: Callable

    def __init__(self, cfg):
        super(AmpereSolver, self).__init__()
        self.vx = cfg["grid"]["vx"]
        self.vy = cfg["grid"]["vy"]
        self.moment_x = partial(jnp.trapz, dx=cfg["grid"]["dvx"], axis=2)
        self.moment_y = partial(jnp.trapz, dx=cfg["grid"]["dvy"], axis=-1)

    def __call__(self, f: jnp.ndarray, prev_force: jnp.ndarray, dt: jnp.float64):
        jx = self.moment_x(self.vx[None, None, :] * self.moment_y(f))[..., None]
        jy = self.moment_y(self.vy[None, None, :] * self.moment_x(f))[..., None]

        return prev_force - dt * jnp.concatenate([jx, jy], axis=-1)


class ElectricFieldSolver(eqx.Module):
    es_field_solver: eqx.Module

    def __init__(self, cfg):
        super(ElectricFieldSolver, self).__init__()

        if cfg["solver"]["field"] == "poisson":
            self.es_field_solver = SpectralPoissonSolver(
                ion_charge=cfg["grid"]["ion_charge"],
                one_over_kx=cfg["grid"]["one_over_kx"],
                one_over_ky=cfg["grid"]["one_over_ky"],
                dvx=cfg["grid"]["dvx"],
                dvy=cfg["grid"]["dvy"],
            )
        elif cfg["solver"]["field"] == "ampere":
            if cfg["solver"]["dfdt"] == "leapfrog":
                self.es_field_solver = AmpereSolver(cfg)
            else:
                raise NotImplementedError(f"ampere + {cfg['solver']['dfdt']} has not yet been implemented")
        else:
            raise NotImplementedError("Field Solver: <" + cfg["solver"]["field"] + "> has not yet been implemented")
        # self.dx = cfg["derived"]["dx"]

    def __call__(self, prev_force: jnp.ndarray, f: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        This returns the total electrostatic field that is used in the Vlasov equation
        The total field is a sum of the driver field and the
        self-consistent electrostatic field from a Poisson or Ampere solve

        :param f: distribution function
        :param a:
        :return:
        """
        # ponderomotive_force = -0.5 * jnp.gradient(jnp.square(a), self.dx)[1:-1]
        self_consistent_e = self.es_field_solver(f, prev_force, dt)
        return self_consistent_e


class Driver(eqx.Module):
    xax: jnp.ndarray
    yax: jnp.ndarray

    def __init__(self, xax, yax):
        self.xax = xax
        self.yax = yax

    def get_this_pulse(self, this_pulse: Dict, current_time: jnp.float64):
        kk = this_pulse["k0"]
        ww = this_pulse["w0"]
        dw = this_pulse["dw0"]
        t_L = this_pulse["t_center"] - this_pulse["t_width"] * 0.5
        t_R = this_pulse["t_center"] + this_pulse["t_width"] * 0.5
        t_wL = this_pulse["t_rise"]
        t_wR = this_pulse["t_rise"]
        x_L = this_pulse["x_center"] - this_pulse["x_width"] * 0.5
        x_R = this_pulse["x_center"] + this_pulse["x_width"] * 0.5
        x_wL = this_pulse["x_rise"]
        x_wR = this_pulse["x_rise"]

        y_L = this_pulse["y_center"] - this_pulse["y_width"] * 0.5
        y_R = this_pulse["y_center"] + this_pulse["y_width"] * 0.5
        y_wL = this_pulse["y_rise"]
        y_wR = this_pulse["y_rise"]

        envelope_t = get_envelope(t_wL, t_wR, t_L, t_R, current_time)
        envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, self.xax)
        envelope_y = get_envelope(y_wL, y_wR, y_L, y_R, self.yax)

        return (
            envelope_t
            * envelope_x[:, None]
            * envelope_y[None, :]
            * jnp.abs(kk)
            * this_pulse["a0"]
            * jnp.sin(kk * self.xax[:, None] - (ww + dw) * current_time)
        )

    def __call__(self, current_time, args):
        """
        Applies the driver function

        P.S. This needs names because its going through a jitted JAX call

        :param current_time:
        :param pulses:
        :return:
        """

        total_dex = jnp.zeros(current_time.shape + self.xax.shape + self.yax.shape)
        # total_djy = np.zeros(current_time.shape + xs[0].shape + xs[1].shape)

        for key, pulse in args["driver"]["ex"].items():
            total_dex += self.get_this_pulse(pulse, current_time)

        # for key, pulse in pulses["ey"].items():
        #     total_djy += get_this_pulse(pulse, current_time)

        total_dey = jnp.zeros((self.xax.size, self.yax.size))

        return jnp.concatenate([total_dex[..., None], total_dey[..., None]], axis=-1)


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))
