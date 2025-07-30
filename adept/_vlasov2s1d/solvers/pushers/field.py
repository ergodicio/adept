#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from jax import numpy as jnp

from adept._base_ import get_envelope


class Driver:
    def __init__(self, xax, driver_key="ex"):
        self.xax = xax
        self.driver_key = driver_key

    def get_this_pulse(self, this_pulse: dict, current_time: float):
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
        total_de = jnp.zeros_like(self.xax)

        for _, pulse in args["drivers"][self.driver_key].items():
            total_de += self.get_this_pulse(pulse, t)

        return total_de


class WaveSolver:
    def __init__(self, c: float, dx: float, dt: float):
        super().__init__()
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
        if self.c > 0:
            d2dx2 = (a[:-2] - 2.0 * a[1:-1] + a[2:]) / self.dx**2.0
            # padded_a = jnp.concatenate([a[-1:], a, a[:1]])
            # d2dx2 = (padded_a[:-2] - 2.0 * padded_a[1:-1] + padded_a[2:]) / self.dx**2.0
            anew = (
                2.0 * a[1:-1]
                - aold[1:-1]
                + self.dt**2.0 * (self.c_sq * d2dx2 - electron_charge * a[1:-1] + djy_array[1:-1])
            )
            # anew = 2.0 * a - aold + self.dt**2.0 * (self.c_sq * d2dx2 - electron_charge * a + djy_array)
            return {"a": self.apply_2nd_order_abc(aold, a, anew), "prev_a": a}
        else:
            return {"a": a, "prev_a": aold}


class SpectralPoissonSolver2S:
    def __init__(self, one_over_kx, dv_e, dv_i):
        super().__init__()
        self.one_over_kx = one_over_kx
        self.dv_e = dv_e
        self.dv_i = dv_i

    def compute_charges(self, f_e, f_i):
        electron_density = jnp.sum(f_e, axis=1) * self.dv_e
        ion_density = jnp.sum(f_i, axis=1) * self.dv_i
        return electron_density, ion_density

    def __call__(self, f_e: jnp.ndarray, f_i: jnp.ndarray, prev_ex: jnp.ndarray, dt: float):
        electron_density, ion_density = self.compute_charges(f_e, f_i)
        # Total charge density: ions positive, electrons negative
        total_charge = ion_density - electron_density
        return jnp.real(jnp.fft.ifft(1j * self.one_over_kx * jnp.fft.fft(total_charge)))


class AmpereSolver2S:
    def __init__(self, cfg):
        super().__init__()
        # This would need to be implemented similar to AmpereSolver but handling 2 species
        # For now, just use the Poisson solver as a placeholder
        self.poisson_solver = SpectralPoissonSolver2S(
            one_over_kx=cfg["grid"]["one_over_kx"], dv_e=cfg["grid"]["dv"], dv_i=cfg["grid"]["dv_i"]
        )

    def __call__(self, f_e: jnp.ndarray, f_i: jnp.ndarray, prev_ex: jnp.ndarray, dt: float):
        # Placeholder - use Poisson solver for now
        return self.poisson_solver(f_e, f_i, prev_ex, dt)


class HampereSolver2S:
    def __init__(self, cfg):
        super().__init__()
        # This would need to be implemented similar to HampereSolver but handling 2 species
        # For now, just use the Poisson solver as a placeholder
        self.poisson_solver = SpectralPoissonSolver2S(
            one_over_kx=cfg["grid"]["one_over_kx"], dv_e=cfg["grid"]["dv"], dv_i=cfg["grid"]["dv_i"]
        )

    def __call__(self, f_e: jnp.ndarray, f_i: jnp.ndarray, prev_ex: jnp.ndarray, dt: float):
        # Placeholder - use Poisson solver for now
        return self.poisson_solver(f_e, f_i, prev_ex, dt)


class ElectricFieldSolver:
    def __init__(self, cfg):
        super().__init__()

        if cfg["terms"]["field"] == "poisson":
            self.es_field_solver = SpectralPoissonSolver2S(
                one_over_kx=cfg["grid"]["one_over_kx"], dv_e=cfg["grid"]["dv"], dv_i=cfg["grid"]["dv_i"]
            )
            self.hampere = False
        elif cfg["terms"]["field"] == "ampere":
            if cfg["terms"]["time"] == "leapfrog":
                self.es_field_solver = AmpereSolver2S(cfg)
                self.hampere = False
            else:
                raise NotImplementedError(f"ampere + {cfg['terms']['time']} has not yet been implemented")
        elif cfg["terms"]["field"] == "hampere":
            if cfg["terms"]["time"] == "leapfrog":
                self.es_field_solver = HampereSolver2S(cfg)
                self.hampere = True
            else:
                raise NotImplementedError(f"ampere + {cfg['terms']['time']} has not yet been implemented")
        else:
            raise NotImplementedError("Field Solver: <" + cfg["solver"]["field"] + "> has not yet been implemented")
        self.dx = cfg["grid"]["dx"]

    def __call__(self, f_e: jnp.ndarray, f_i: jnp.ndarray, a: jnp.ndarray, prev_ex: jnp.ndarray, dt: float):
        """
        This returns the total electrostatic field that is used in the Vlasov equation
        The total field is a sum of the ponderomotive force from `E_y`, the driver field, and the
        self-consistent electrostatic field from a Poisson or Ampere solve

        :param f_e: electron distribution function
        :param f_i: ion distribution function
        :param a:
        :return:
        """
        ponderomotive_force = -0.5 * jnp.gradient(a**2.0, self.dx)[1:-1]
        self_consistent_ex = self.es_field_solver(f_e, f_i, prev_ex, dt)
        return ponderomotive_force, self_consistent_ex
