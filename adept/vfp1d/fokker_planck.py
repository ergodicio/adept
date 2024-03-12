from jax import numpy as jnp
from jax import vmap
import numpy as np
import lineax as lx
import diffrax


class LenardBernstein:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]

        # these are twice as large for the solver
        self.ones = jnp.ones(2 * self.cfg["grid"]["nv"])
        self.refl_v = jnp.concatenate([-self.v[::-1], self.v])
        self.midpt = self.cfg["grid"]["nv"]
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        self.nuee_coeff = 4.0 * np.pi / 3 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

    def _solve_one_vslice_(self, nu: jnp.float64, f0: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        operator = self._get_operator_(nu, f0, dt)
        refl_f0 = jnp.concatenate([f0[::-1], f0])
        return lx.linear_solve(operator, refl_f0, solver=lx.Tridiagonal()).value[self.midpt :]

    def _get_operator_(self, nu, f0, dt):
        half_vth_sq = (
            jnp.sum(f0 * (self.v[None, :]) ** 4.0, axis=1) / jnp.sum(f0 * (self.v[None, :]) ** 2.0, axis=1) / 3.0
        )

        lower_diagonal = self.nuee_coeff * dt * (-half_vth_sq / self.dv**2.0 + (self.refl_v[:-1]) / 2.0 / self.dv)
        diagonal = 1.0 + self.nuee_coeff * dt * self.ones * (2.0 * half_vth_sq / self.dv**2.0)
        upper_diagonal = self.nuee_coeff * dt * (-half_vth_sq / self.dv**2.0 - (self.refl_v[1:]) / 2.0 / self.dv)
        return lx.TridiagonalLinearOperator(
            diagonal=diagonal, upper_diagonal=upper_diagonal, lower_diagonal=lower_diagonal
        )

    def __call__(self, nu: jnp.float64, f0x: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """

        :param nu:
        :param f_vxvy:
        :param dt:
        :return:
        """

        return vmap(self._solve_one_vslice_, in_axes=(None, 0, None))(nu, f0x, dt)


class FLMCollisions:
    def __init__(self, cfg):
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.Z = cfg["units"]["Z"]

        r_e = 2.8179402894e-13
        kp = np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        kpre = r_e * kp
        self.nuee_coeff = kpre * cfg["units"]["derived"]["logLambda_ee"]
        self.nuei_coeff = (
            kpre * self.Z**2.0 * cfg["units"]["derived"]["logLambda_ei"]
        )  # will be multiplied by ni = ne / Z

        self.nl = cfg["grid"]["nl"]
        self.ee = cfg["terms"]["fokker_planck"]["flm"]["ee"]

        self.Z_nuei_scaling = (cfg["units"]["Z"] + 4.2) / (cfg["units"]["Z"] + 0.24)

        self.a1, self.a2, self.b1, self.b2, self.b3, self.b4 = (
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
        )

        for il in range(1, self.nl + 1):
            self.a1[il] = (il + 1) * (il + 2) / (2 * il + 1) / (2 * il + 3)
            self.a2[il] = -(il - 1) * il / (2 * il + 1) / (2 * il - 1)
            self.b1[il] = (-il * (il + 1) / 2 - (il + 1)) / (2 * il + 1) / (2 * il + 3)
            self.b2[il] = (il * (il + 1) / 2 + (il + 2)) / (2 * il + 1) / (2 * il + 3)
            self.b3[il] = (il * (il + 1) / 2 + (il - 1)) / (2 * il + 1) / (2 * il - 1)
            self.b4[il] = (il * (il + 1) / 2 - il) / (2 * il + 1) / (2 * il - 1)

    def calc_ros_i(self, flm, power):
        return 4 * jnp.pi * self.v**-power * jnp.cumsum(self.v[None, :] ** (2.0 + power) * flm, axis=1) * self.dv

    def calc_ros_j(self, flm, power):
        return (
            4
            * jnp.pi
            * self.v[None, :] ** -power
            * jnp.cumsum((self.v[None, :] ** (2.0 + power) * flm)[:, ::-1], axis=1)[:, ::-1]
            * self.dv
        )

    def get_ee_offdiagonal_contrib(self, t, y, args):
        ddv = args["ddvf0"]
        d2dv2 = args["d2dv2f0"]
        il = args["il"]
        flm = y

        contrib = (self.a1[il] * d2dv2 + self.b1[il] * ddv) * self.calc_ros_i(flm, power=2.0 + il)
        contrib += (self.a1[il] * d2dv2 + self.b2[il] * ddv) * self.calc_ros_j(flm, power=-il - 1.0)
        contrib += (self.a2[il] * d2dv2 + self.b3[il] * ddv) * self.calc_ros_i(flm, power=il)
        if il > 1:
            contrib += (self.a2[il] * d2dv2 + self.b4[il] * ddv) * self.calc_ros_j(flm, power=1.0 - il)
        # i31 = self.calc_ros_i(flm, power=3.0)
        # jm21 = self.calc_ros_j(flm, power=-2.0)
        # i11 = self.calc_ros_i(flm, power=1.0)

        # term1 = 1 / 5.0 / self.v[None, :] * (i31 + jm21) * d2dv2
        # term2 = 1 / 30.0 / self.v[None, :] ** 2.0 * (-6 * i31 + 4 * jm21 + 10 * i11) * ddv

        return contrib

    def get_ee_diagonal_contrib(self, f0):
        i0 = self.calc_ros_i(f0, power=0.0)
        jm1 = self.calc_ros_j(f0, power=-1.0)
        i2 = self.calc_ros_i(f0, power=2.0)

        diag_term1 = 8 * jnp.pi * f0

        lower_d2dv2 = (i2 + jm1) / (3.0 * self.v[None, :]) / self.dv**2.0
        diag_d2dv2 = (i2 + jm1) / (3.0 * self.v[None, :]) / self.dv**2.0
        upper_d2dv2 = (i2 + jm1) / (3.0 * self.v[None, :]) / self.dv**2.0

        diag_angular = -(-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v[None, :] ** 3.0)

        lower_ddv = (-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v[None, :] ** 2.0) / 2 / self.dv
        upper_ddv = (-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v[None, :] ** 2.0) / 2 / self.dv

        # adding spatial differencing coefficients here
        # 1  -2  1  for d2dv2
        # 1  -1     for ddv
        lower = lower_d2dv2 - lower_ddv
        diag = diag_term1 - 2.0 * diag_d2dv2 + diag_angular
        upper = upper_d2dv2 + upper_ddv

        return diag, lower[:, :-1], upper[:, 1:]

    def _solve_one_x_tridiag_(self, diag, upper, lower, f10):
        op = lx.TridiagonalLinearOperator(diagonal=diag, upper_diagonal=upper, lower_diagonal=lower)
        return lx.linear_solve(op, f10, solver=lx.Tridiagonal()).value

    def _solve_one_x_dense_(self, f0, f10, Z, ni, dt):
        il = 1
        i0 = 4 * jnp.pi * jnp.cumsum(self.v**2.0 * f0) * self.dv
        jm1 = 4 * jnp.pi / self.v * jnp.cumsum((self.v * f0)[::-1])[::-1] * self.dv
        i2 = 4 * jnp.pi * self.v**2.0 * jnp.cumsum(self.v**4.0 * f0) * self.dv
        term1 = 8 * jnp.pi * f0
        c_d2dv2 = (i2 + jm1) / (3.0 * self.v)
        c_ang = -(-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v**3.0)
        c_ddv = (-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v**2.0)

        pad_f0 = jnp.concatenate([f0[1::-1], f0])
        d2dv2f0 = 0.5 / self.v * jnp.gradient(jnp.gradient(pad_f0, self.dv), self.dv)[2:]
        ddvf0 = self.v**-2.0 * jnp.gradient(pad_f0, self.dv)[2:]

        c_o1 = self.a1[il] * d2dv2f0 + self.b1[il] * ddvf0
        c_o2 = self.a1[il] * d2dv2f0 + self.b2[il] * ddvf0
        c_o3 = self.a2[il] * d2dv2f0 + self.b3[il] * ddvf0
        c_o4 = self.a2[il] * d2dv2f0 + self.b4[il] * ddvf0

        c_ang_ei = -il * (il + 1) / 2.0 * (Z**2.0) * ni / self.v**3.0

        def __implicit_ee__(this_flm):
            d2dv2flm = c_d2dv2 * jnp.gradient(jnp.gradient(this_flm, self.dv), self.dv)
            angular = c_ang * this_flm
            ddvflm = c_ddv * jnp.gradient(this_flm, self.dv)

            offdiag_term1 = c_o1 * self.calc_ros_i(this_flm, power=2.0 + il)
            offdiag_term2 = c_o2 * self.calc_ros_j(this_flm, power=-il - 1.0)
            offdiag_term3 = c_o3 * self.calc_ros_i(this_flm, power=il)
            offdiag_term4 = c_o4 * self.calc_ros_j(this_flm, power=1.0 - il)

            ang_ei = c_ang_ei * this_flm

            dflmdt_ee = self.nuee_coeff * (
                term1 + angular + ddvflm + d2dv2flm + offdiag_term1 + offdiag_term2 + offdiag_term3 + offdiag_term4
            )

            dflmdt_ei = self.nuei_coeff * ang_ei

            return this_flm - dt * (dflmdt_ee + dflmdt_ei)

        op = lx.FunctionLinearOperator(__implicit_ee__, input_structure=f10)
        return lx.linear_solve(op, f10, solver=lx.SVD()).value

    def __call__(self, Z, ni, f0, f10, dt):
        ee_diag, ee_lower, ee_upper = self.get_ee_diagonal_contrib(f0)

        for il in range(1, self.nl + 1):
            ei_diag = -il * (il + 1) / 2.0 * (Z[:, None] ** 2.0) * ni[:, None] / self.v[None, :] ** 3.0

            if self.ee:
                pad_f0 = jnp.concatenate([f0[:, 1::-1], f0], axis=1)
                #
                d2dv2 = (
                    0.5 / self.v[None, :] * jnp.gradient(jnp.gradient(pad_f0, self.dv, axis=1), self.dv, axis=1)[:, 2:]
                )

                ddv = self.v[None, :] ** -2.0 * jnp.gradient(pad_f0, self.dv, axis=1)[:, 2:]

                diag = 1 - dt * (self.nuei_coeff * ei_diag + self.nuee_coeff * ee_diag)
                lower = -dt * self.nuee_coeff * ee_lower
                upper = -dt * self.nuee_coeff * ee_upper

                new_f10 = vmap(self._solve_one_x_tridiag_, in_axes=(0, 0, 0, 0))(diag, upper, lower, f10)

                new_f10 = new_f10 + dt * self.nuee_coeff * self.get_ee_offdiagonal_contrib(
                    None, f10, {"ddvf0": ddv, "d2dv2f0": d2dv2, "il": il}
                )

                # new_f10 = diffrax.diffeqsolve(
                #     diffrax.ODETerm(self.get_ee_offdiagonal_contrib),
                #     solver=diffrax.Tsit5(),
                #     t0=0.0,
                #     t1=dt,
                #     dt0=dt,
                #     y0=new_f10,
                #     args={"ddvf0": ddv, "d2dv2f0": d2dv2, "il": il},
                # ).ys[-1]

            else:
                # only uses the Z* epperlein haines scaling instead of solving the ee collisions

                new_f10 = f10 / (1 - dt * self.nuei_coeff * self.Z_nuei_scaling * ei_diag)

        return new_f10
