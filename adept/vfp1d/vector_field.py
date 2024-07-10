from typing import Dict
from jax import numpy as jnp, Array
import optimistix as optx
import diffrax
from adept.vfp1d.fokker_planck import LenardBernstein, FLMCollisions


class OSHUN1D:
    """
    This is the OSHUN1D solver for f0, f1, and e in 1D. It uses the Lenard-Bernstein collision operator and the
    FLM collision operator. It can solve for the electric field using the "perturbed charge" method.

    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]

        self.dx = cfg["grid"]["dx"]
        self.dt = cfg["grid"]["dt"]
        self.nx = cfg["grid"]["nx"]
        # self.c_squared = cfg["units"]["derived"]["c_light"].magnitude ** 2.0
        self.e_solver = cfg["terms"]["e_solver"]

        self.ampere_coeff = 1e-6
        self.lb = LenardBernstein(cfg)
        self.ei = FLMCollisions(cfg)

        self.large_eps = 1e-6
        self.eps = 1e-12

    def ddv(self, f: Array) -> Array:
        """
        Calculates the derivative of f with respect to v

        Args:
            f (Array): f(x, v)

        Returns:
            Array: df/dv
        """
        temp = jnp.concatenate([f[:, :1], f], axis=1)
        return jnp.gradient(temp, self.dv, axis=1)[:, 1:]

    def ddv_f1(self, f: Array) -> Array:
        """
        Calculates the derivative of f1 with respect to v

        Args:
            f (Array): f(x, v)

        Returns:
            Array: df1/dv
        """
        temp = jnp.concatenate([-f[:, :1], f], axis=1)
        return jnp.gradient(temp, self.dv, axis=1)[:, 1:]

    def ddx(self, f: Array) -> Array:
        """
        Calculates the derivative of f with respect to x

        Args:
            f (Array): f(x, v)

        Returns:
            Array: df/dx
        """
        periodic_f = jnp.concatenate([f[-1:], f, f[:1:]], axis=0)
        return jnp.gradient(periodic_f, self.dx, axis=0)[1:-1]

    def calc_j(self, f1: Array) -> Array:
        """
        Calculates the current density

        Args:
            f1 (Array): f1(x, v)

        Returns:
            Array: j(x)

        """
        return -4 * jnp.pi / 3.0 * jnp.sum(f1 * self.v[None, :] ** 3.0, axis=1) * self.dv

    def implicit_e_solve(self, Z: Array, ni: Array, f0: Array, f10: Array, e: Array) -> Array:
        """
        This is the implicit solve for the electric field. It uses the "perturbed charge" method and is a direct solve.

        Args:
            Z (Array): charge
            ni (Array): number density
            f0 (Array): f0(x, v)
            f10 (Array): f10(x, v)
            e (Array): e(x)

        Returns:
            Array: new_e(x)

        """

        # calculate j without any e field
        f10_after_coll = self.ei(Z=Z, ni=ni, f0=f0, f10=f10, dt=self.dt)
        j0 = self.calc_j(f10_after_coll)

        # get perturbation
        de = jnp.abs(e) * self.large_eps + self.eps

        # calculate effect of dex
        _, f10_after_dex = self.push_edfdv(f0, f10, de)
        f10_after_dex = self.ei(Z=Z, ni=ni, f0=f0, f10=f10_after_dex, dt=self.dt)
        jx_dx = self.calc_j(f10_after_dex)
        # jy_dx = 0.0
        # jz_dx = 0.0

        # f10_after_dey = self.step_f10_coll(self.apply_dey(f10))
        # jx_dy = -4 * jnp.pi / 3.0 * jnp.sum(f10_after_dey * self.v[None, :] ** 3.0, axis=1) * self.dv
        # jy_dy = 0.0
        # jz_dy = 0.0

        # f10_after_dez = self.step_f10_coll(self.apply_dez(f10))
        # jx_dz = -4 * jnp.pi / 3.0 * jnp.sum(f10_after_dez * self.v[None, :] ** 3.0, axis=1) * self.dv
        # jy_dz = 0.0
        # jz_dz = 0.0

        # directly solve for ex
        new_e = -j0 * de / (jx_dx - j0)

        return new_e

    def linear_implicit_e_f0_f1_operator(self, this_y):
        """
        UNUSED

        """
        f0, f1, e = this_y["f0"], this_y["f1"], this_y["e"]

        prev_f0_approx = f0 + self.dt * (-e[:, None] / 3 * (self.ddv_f1(f1) + 2 / self.v * f1))
        # C_f1 = self.step_f10_coll(f1)
        prev_f1_approx = f1 + self.dt * (-e[:, None] * self.ddv(f0) + self.ei.nuei_coeff * f1 / self.v[None, :] ** 3.0)

        j = -4 * jnp.pi / 3.0 * jnp.sum(f1 * self.v[None, :] ** 3.0, axis=1) * self.dv
        prev_e_approx = e + self.dt * j

        return {"f0": prev_f0_approx, "f1": prev_f1_approx, "e": prev_e_approx}

    def nonlinear_implicit_e_f0_f1_operator(self, y, args):
        """
        UNUSED

        """
        new_f0, new_f1, new_e = y["f0"], y["f1"], y["e"]
        old_f0, old_f1, old_e = args["f0"], args["f1"], args["e"]

        res_f0 = (new_f0 - old_f0) / self.dt - new_e[:, None] / 3 * (self.ddv_f1(new_f1) + 2 / self.v * new_f1)
        # C_f1 = self.step_f10_coll(f1)
        res_f1 = (
            (new_f1 - old_f1) / self.dt - new_e[:, None] * self.ddv(new_f0) + 1e-4 * new_f1 / self.v[None, :] ** 3.0
        )

        new_j = -4 * jnp.pi / 3.0 * jnp.sum(new_f1 * self.v[None, :] ** 3.0, axis=1) * self.dv
        res_e = (new_e - old_e) / self.dt + new_j

        # return {"f0": res_f0, "f1": res_f1, "e": res_e}

        return jnp.sum(jnp.square(res_f0)) + jnp.sum(jnp.square(res_f1)) + jnp.sum(jnp.square(res_e))

    def implicit_e_f0_f1_solve(self, f0, f1, e):
        """
        UNUSED

        """

        sol = optx.minimise(
            fn=self.nonlinear_implicit_e_f0_f1_operator,
            solver=optx.NonlinearCG(rtol=1e-6, atol=1e-6, norm=optx.rms_norm),
            # OptaxMinimiser(
            # optim=optax.adam(learning_rate=1e-2), rtol=1e-3, atol=1e-4, verbose=frozenset({"step", "loss"})
            # ),
            y0={"f0": f0, "f1": f1, "e": e},
            args={"f0": f0, "f1": f1, "e": e},
            max_steps=4096,
            throw=True,
        )

        # sol = optx.least_squares(
        #     fn=self.nonlinear_implicit_e_f0_f1_operator,
        #     solver=optx.LevenbergMarquardt(rtol=1e-3, atol=1e-4),
        #     y0={"f0": f0, "f1": f1, "e": e},
        #     args={"f0": f0, "f1": f1, "e": e},
        #     max_steps=4096,
        #     throw=True,
        # )
        # operator = lx.FunctionLinearOperator(
        # self.implicit_e_f0_f1_operator, input_structure={"f0": f0, "f1": f1, "e": e}
        # )
        # rhs = {"f0": f0, "f1": f1, "e": e}
        # solver = lx.GMRES(
        #     rtol=1e-3, atol=1e-4, max_steps=16384, norm=lx.internal.max_norm  # restart=128, stagnation_iters=128
        # )
        # sol = lx.linear_solve(
        #     operator, rhs, solver=solver, options={"y0": {"f0": f0, "f1": f1, "e": jnp.zeros_like(e)}}
        # )

        return sol.value["f0"], sol.value["f1"], sol.value["e"]

    def _edfdv_(self, t: float, y: Dict, args: Dict) -> Dict:
        """
        This is the edfdv solve for f0 and f1. It uses the `t, y, args` formulation for diffeqsolve
        because we step it using a 5th order integrator (but can also use Euler etc.)

        Args:
            t (float): time
            y (Dict): y0 -- just distribution functions here
            args (Dict): args -- in this case there is just the electric field

        Returns:
            Dict: new y
        """
        f0 = y["f0"]
        f10 = y["f10"]
        e_field = args["e"]

        g00 = self.ddv(f0)
        h10 = 2.0 / self.v * f10 + self.ddv_f1(f10)

        df0dt_e = e_field[:, None] / 3.0 * h10
        df10dt_e = e_field[:, None] * g00

        return {"f0": df0dt_e, "f10": df10dt_e}

    def push_edfdv(self, f0, f10, e):
        """
        This is the explicit solve for f0 and f1 given the electric field.

        Args:
            f0 (Array): f0(x, v)
            f10 (Array): f10(x, v)
            e (Array): e(x)

        Returns:
            Tuple[Array, Array]: new f0, new f10
        """
        result = diffrax.diffeqsolve(
            diffrax.ODETerm(self._edfdv_),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=self.dt,
            dt0=self.dt,
            y0={"f0": f0, "f10": f10},
            args={"e": e},
        )
        return result.ys["f0"][-1], result.ys["f10"][-1]

    def _vdfdx_(self, t: float, y: Dict, args: Dict) -> Dict:
        """
        This is the vdfdx solve for f0 and f1. It uses the `t, y, args` formulation for diffeqsolve
        because we step it using a 5th order integrator (but can also use Euler etc.)

        Args:
            t (float): time
            y (Dict): y0 -- just distribution functions here
            args (Dict): args -- in this case there are no args needed or used

        Returns:
            Dict: new y

        """
        f0 = y["f0"]
        f10 = y["f10"]

        df0dt_sa = -self.v[None, :] / 3.0 * self.ddx(f10)
        df10dt_sa = -self.v[None, :] * self.ddx(f0)

        return {"f0": df0dt_sa, "f10": df10dt_sa}

    def push_vdfdx(self, f0: Array, f10: Array) -> Array:
        """
        This is the explicit solve for f0 and f1 given the electric field.

        Args:
            f0 (Array): f0(x, v)
            f10 (Array): f10(x, v)

        Returns:
            Tuple[Array, Array]: new f0, new f10
        """

        result = diffrax.diffeqsolve(
            diffrax.ODETerm(self._vdfdx_),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=self.dt,
            dt0=self.dt,
            y0={"f0": f0, "f10": f10},
        )
        return result.ys["f0"][-1], result.ys["f10"][-1]

    def __call__(self, t, y, args) -> Dict:
        """
        This is the main function that is called by the solver. It steps the distribution functions and the electric
        field.

        Args:
            t (float): time
            y (Dict): y0 -- all variables
            args (Dict): args -- in this case there are no args needed or used

        Returns:
            Dict: new y

        """

        f0 = y["f0"]
        f10 = y["f10"]
        Z = y["Z"]
        ni = y["ni"]

        # explicit push for v df/dx
        f0_star, f10_star = self.push_vdfdx(f0, f10)
        # implicit solve f00 coll
        f0_star = self.lb(None, f0_star, self.dt)

        # implicit solve for E
        if self.e_solver == "oshun":  # implicit E, explicit f0, f1 with this Taylor expansion of J method
            # taylor expansion of j(E) method from Tzoufras 2013
            new_e = self.implicit_e_solve(Z, ni, f0_star, f10_star, y["e"])
            # push e
            new_f0, new_f10 = self.push_edfdv(f0_star, f10_star, new_e)
            # solve f10 coll
            new_f10 = self.ei(Z=Z, ni=ni, f0=f0_star, f10=new_f10, dt=self.dt)

        elif self.e_solver == "edfdv-ampere-implicit":  # implicit E, f0, f1 using a nonlinear iterative inversion
            new_f0, new_f10, new_e = self.implicit_e_f0_f1_solve(f0=f0_star, f1=f10_star, e=y["e"])

        elif self.e_solver == "ampere":
            new_e = y["e"] + self.dt * self.ampere_coeff * self.calc_j(f10_star)
            # push e
            new_f0, new_f10 = self.push_edfdv(f0_star, f10_star, new_e)
            # solve f10 coll
            new_f10 = self.ei(Z=Z, ni=ni, f0=new_f0, f10=new_f10, dt=self.dt)

        else:
            raise NotImplementedError

        return {"f0": new_f0, "f10": new_f10, "f11": y["f11"], "e": new_e, "b": y["b"], "Z": y["Z"], "ni": y["ni"]}
