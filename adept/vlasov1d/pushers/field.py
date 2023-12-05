#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from typing import Dict
from functools import partial


from jax import numpy as jnp

from adept.tf1d.pushers import get_envelope


class Driver:
    def __init__(self, xax):
        self.xax = xax

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
        envelope_t = get_envelope(t_wL, t_wR, t_L, t_R, current_time)
        envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, self.xax)

        return (
            envelope_t * envelope_x * jnp.abs(kk) * this_pulse["a0"] * jnp.sin(kk * self.xax - (ww + dw) * current_time)
        )

    def __call__(self, t, args):
        total_dex = jnp.zeros_like(self.xax)
        total_djy = jnp.zeros_like(self.xax)

        for key, pulse in args["drivers"]["ex"].items():
            total_dex += self.get_this_pulse(pulse, t)

        for key, pulse in args["drivers"]["ey"].items():
            total_djy += self.get_this_pulse(pulse, t)

        return total_dex, total_djy


class WaveSolver:
    def __init__(self, c: jnp.float64, dx: jnp.float64, dt: jnp.float64):
        super(WaveSolver, self).__init__()
        self.dx = dx
        self.c = c
        self.c_sq = c**2.0
        c_over_dx = c / dx
        self.dt = dt
        self.const = c_over_dx * dt
        # abc_const = (const - 1.0) / (const + 1.0)
        self.one_over_const = 1.0 / dt / c_over_dx

    def apply_2nd_order_abc(self, aold, a, anew):
        """
        Second order absorbing boundary conditions

        :param aold:
        :param a:
        :param anew:
        :param dt:
        :return:
        """

        coeff = -1.0 / (self.one_over_const + 2.0 + self.const)

        # # 2nd order ABC
        a_left = (self.one_over_const - 2.0 + self.const) * (anew[1] + aold[0])
        a_left += 2.0 * (self.const - self.one_over_const) * (a[0] + a[2] - anew[0] - aold[1])
        a_left -= 4.0 * (self.one_over_const + self.const) * a[1]
        a_left *= coeff
        a_left -= aold[2]
        a_left = jnp.array([a_left])

        a_right = (self.one_over_const - 2.0 + self.const) * (anew[-2] + aold[-1])
        a_right += 2.0 * (self.const - self.one_over_const) * (a[-1] + a[-3] - anew[-1] - aold[-2])
        a_right -= 4.0 * (self.one_over_const + self.const) * a[-2]
        a_right *= coeff
        a_right -= aold[-3]
        a_right = jnp.array([a_right])

        # commenting out first order damping
        # a_left = jnp.array([a[1] + abc_const * (anew[0] - a[0])])
        # a_right = jnp.array([a[-2] + abc_const * (anew[-1] - a[-1])])

        return jnp.concatenate([a_left, anew, a_right])

    def __call__(self, a: jnp.ndarray, aold: jnp.ndarray, djy_array: jnp.ndarray, electron_charge: jnp.ndarray):
        # if self.c > 0:
        #     # d2dx2 = (a[:-2] - 2.0 * a[1:-1] + a[2:]) / self.dx**2.0
        #     padded_a = jnp.concatenate([a[-1:], a, a[:1]])
        #     d2dx2 = (padded_a[:-2] - 2.0 * padded_a[1:-1] + padded_a[2:]) / self.dx**2.0
        #     # anew = (
        #     #     2.0 * a[1:-1]
        #     #     - aold[1:-1]
        #     #     + self.dt**2.0 * (self.c_sq * d2dx2 - electron_charge * a[1:-1] + djy_array)
        #     # )
        #     anew = 2.0 * a - aold + self.dt**2.0 * (self.c_sq * d2dx2 - electron_charge * a + djy_array)
        #     return self.apply_2nd_order_abc(aold, a, anew), a
        # else:
        return a, aold


class SpectralPoissonSolver:
    def __init__(self, ion_charge, one_over_kx, dv):
        super(SpectralPoissonSolver, self).__init__()
        self.ion_charge = ion_charge
        self.one_over_kx = one_over_kx
        self.compute_charges = partial(jnp.trapz, dx=dv, axis=1)

    def __call__(self, f: jnp.ndarray, prev_force: jnp.ndarray, dt: jnp.float64):
        return jnp.real(jnp.fft.ifft(1j * self.one_over_kx * jnp.fft.fft(self.ion_charge - self.compute_charges(f))))


class AmpereSolver:
    def __init__(self, cfg):
        super(AmpereSolver, self).__init__()
        self.vx = cfg["grid"]["v"]
        self.vx_moment = partial(jnp.trapz, dx=cfg["grid"]["dv"], axis=1)

    def __call__(self, f: jnp.ndarray, prev_force: jnp.ndarray, dt: jnp.float64):
        return prev_force - dt * self.vx_moment(self.vx[None, :] * f)


class ElectricFieldSolver:
    def __init__(self, cfg):
        super(ElectricFieldSolver, self).__init__()

        # if cfg["solver"]["field"] == "poisson":
        self.es_field_solver = SpectralPoissonSolver(
            ion_charge=cfg["grid"]["ion_charge"], one_over_kx=cfg["grid"]["one_over_kx"], dv=cfg["grid"]["dv"]
        )
        # elif cfg["solver"]["field"] == "ampere":
        #     if cfg["solver"]["dfdt"] == "leapfrog":
        #         self.es_field_solver = AmpereSolver(cfg)
        #     else:
        #         raise NotImplementedError(f"ampere + {cfg['solver']['dfdt']} has not yet been implemented")
        # else:
        #     raise NotImplementedError("Field Solver: <" + cfg["solver"]["field"] + "> has not yet been implemented")
        self.dx = cfg["grid"]["dx"]

    def __call__(
        self, dex_array: jnp.ndarray, f: jnp.ndarray, a: jnp.ndarray, total_force: jnp.ndarray, dt: jnp.float64
    ):
        """
        This returns the total electrostatic field that is used in the Vlasov equation
        The total field is a sum of the ponderomotive force from `E_y`, the driver field, and the
        self-consistent electrostatic field from a Poisson or Ampere solve

        :param dex_array: an electric field
        :param f: distribution function
        :param a:
        :return:
        """
        ponderomotive_force = -0.5 * jnp.gradient(a**2.0, self.dx)  # [1:-1]
        self_consistent_ex = self.es_field_solver(f, total_force, dt)
        return ponderomotive_force + dex_array + self_consistent_ex, ponderomotive_force, self_consistent_ex
