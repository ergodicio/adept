from typing import Tuple

from functools import partial

from jax import numpy as jnp
import diffrax

from adept.vlasov1d2v.pushers import field, fokker_planck, vlasov
from adept._base_ import get_envelope


class Stepper(diffrax.Euler):
    """

    :param cfg:
    """

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful


class TimeIntegrator:
    """

    :param cfg:
    """

    def __init__(self, cfg):
        self.field_solve = field.ElectricFieldSolver(cfg)
        self.edfdv = self.get_edfdv(cfg)
        self.vdfdx = vlasov.SpaceExponential(cfg)

    def get_edfdv(self, cfg):
        if cfg["terms"]["edfdv"] == "exponential":
            return vlasov.VelocityExponential(cfg)
        elif cfg["terms"]["edfdv"] == "cubic-spline":
            return vlasov.VelocityCubicSpline(cfg)
        else:
            raise NotImplementedError(f"{cfg['terms']['edfdv']} has not been implemented")


class LeapfrogIntegrator(TimeIntegrator):
    """

    :param cfg:
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dt = cfg["grid"]["dt"]
        self.dt_array = self.dt * jnp.array([0.0, 1.0])

    def __call__(self, f, a, dex_array, prev_ex) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        f_after_v = self.vdfdx(f=f, dt=self.dt)
        if self.field_solve.hampere:
            f_for_field = f
        else:
            f_for_field = f_after_v
        pond, e = self.field_solve(f=f_for_field, a=a, prev_ex=prev_ex, dt=self.dt)
        f = self.edfdv(f=f_after_v, e=pond + e + dex_array[0], dt=self.dt)

        return e, f


class SixthOrderHamIntegrator(TimeIntegrator):
    """

    :param cfg:
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dt = cfg["grid"]["dt"]

        self.a1 = 0.168735950563437422448196
        self.a2 = 0.377851589220928303880766
        self.a3 = -0.093175079568731452657924
        b1 = 0.049086460976116245491441
        b2 = 0.264177609888976700200146
        b3 = 0.186735929134907054308413
        c1 = -0.000069728715055305084099
        c2 = -0.000625704827430047189169
        c3 = -0.002213085124045325561636
        d2 = -2.916600457689847816445691e-6
        d3 = 3.048480261700038788680723e-5
        e3 = 4.985549387875068121593988e-7

        self.D1 = b1 + 2.0 * c1 * self.dt**2.0
        self.D2 = b2 + 2.0 * c2 * self.dt**2.0 + 4.0 * d2 * self.dt**4.0
        self.D3 = b3 + 2.0 * c3 * self.dt**2.0 + 4.0 * d3 * self.dt**4.0 - 8.0 * e3 * self.dt**6.0

        self.dt_array = self.dt * jnp.array(
            [
                0.0,
                self.a1,
                self.a1 + self.a2,
                self.a1 + self.a2 + self.a3,
                self.a1 + self.a2 + self.a3 + self.a2,
                self.a1 + self.a2 + self.a3 + self.a2 + self.a1,
            ]
        )

    def __call__(self, f, a, dex_array, prev_ex) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[0] + self_consistent_ex
        f = self.edfdv(f=f, e=force, dt=self.D1 * self.dt)

        f = self.vdfdx(f=f, dt=self.a1 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[1] + self_consistent_ex

        f = self.edfdv(f=f, e=force, dt=self.D2 * self.dt)

        f = self.vdfdx(f=f, dt=self.a2 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[2] + self_consistent_ex

        f = self.edfdv(f=f, e=force, dt=self.D3 * self.dt)

        f = self.vdfdx(f=f, dt=self.a3 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[3] + self_consistent_ex

        f = self.edfdv(f=f, e=force, dt=self.D3 * self.dt)

        f = self.vdfdx(f=f, dt=self.a2 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[4] + self_consistent_ex

        f = self.edfdv(f=f, e=force, dt=self.D2 * self.dt)

        f = self.vdfdx(f=f, dt=self.a1 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[5] + self_consistent_ex

        f = self.edfdv(f=f, e=force, dt=self.D1 * self.dt)

        return self_consistent_ex, f


class VlasovPoissonFokkerPlanck:
    """

    :param cfg:
    """

    def __init__(self, cfg):
        self.dt = cfg["grid"]["dt"]
        self.v = cfg["grid"]["v"]
        if cfg["terms"]["time"] == "sixth":
            self.vlasov_poisson = SixthOrderHamIntegrator(cfg)
            self.dex_save = 3
        elif cfg["terms"]["time"] == "leapfrog":
            self.vlasov_poisson = LeapfrogIntegrator(cfg)
            self.dex_save = 0
        else:
            raise NotImplementedError
        self.fp = fokker_planck.Collisions(cfg=cfg)

    def __call__(
        self,
        f: jnp.ndarray,
        a: jnp.ndarray,
        prev_ex: jnp.ndarray,
        dex_array: jnp.ndarray,
        nu_ee: jnp.ndarray,
        nu_ei: jnp.ndarray,
        nu_K: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        e, f = self.vlasov_poisson(f, a, dex_array, prev_ex)
        f = self.fp(nu_ee, nu_ei, nu_K, f, dt=self.dt)

        return e, f


class VlasovMaxwell:
    """

    :param cfg:
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.vpfp = VlasovPoissonFokkerPlanck(cfg)
        self.wave_solver = field.WaveSolver(c=1.0 / cfg["grid"]["beta"], dx=cfg["grid"]["dx"], dt=cfg["grid"]["dt"])
        self.dt = self.cfg["grid"]["dt"]
        self.ey_driver = field.Driver(cfg["grid"]["x_a"], driver_key="ey")
        self.ex_driver = field.Driver(cfg["grid"]["x"], driver_key="ex")

    def compute_charges(self, f):
        return jnp.sum(jnp.sum(f, axis=2), axis=1) * self.cfg["grid"]["dv"] * self.cfg["grid"]["dv"]

    def nu_prof(self, t, nu_args):
        t_L = nu_args["time"]["center"] - nu_args["time"]["width"] * 0.5
        t_R = nu_args["time"]["center"] + nu_args["time"]["width"] * 0.5
        t_wL = nu_args["time"]["rise"]
        t_wR = nu_args["time"]["rise"]
        x_L = nu_args["space"]["center"] - nu_args["space"]["width"] * 0.5
        x_R = nu_args["space"]["center"] + nu_args["space"]["width"] * 0.5
        x_wL = nu_args["space"]["rise"]
        x_wR = nu_args["space"]["rise"]

        nu_time = get_envelope(t_wL, t_wR, t_L, t_R, t)
        if nu_args["time"]["bump_or_trough"] == "trough":
            nu_time = 1 - nu_time
        nu_time = nu_args["time"]["baseline"] + nu_args["time"]["bump_height"] * nu_time

        nu_prof = get_envelope(x_wL, x_wR, x_L, x_R, self.cfg["grid"]["x"])
        if nu_args["space"]["bump_or_trough"] == "trough":
            nu_prof = 1 - nu_prof
        nu_prof = nu_args["space"]["baseline"] + nu_args["space"]["bump_height"] * nu_prof

        return nu_time * nu_prof

    def __call__(self, t, y, args):
        """
        This is just a wrapper around a Vlasov-Poisson + Fokker-Planck timestep

        :param t:
        :param y:
        :param args:

        :return:
        """

        dex = [self.ex_driver(t + dt, args) for dt in self.vpfp.vlasov_poisson.dt_array]
        djy = self.ey_driver(t + self.vpfp.vlasov_poisson.dt_array[1], args)

        if self.cfg["terms"]["fokker_planck"]["nu_ee"]["is_on"]:
            nu_ee_prof = self.nu_prof(t=t, nu_args=args["terms"]["fokker_planck"]["nu_ee"])
        else:
            nu_ee_prof = None

        if self.cfg["terms"]["fokker_planck"]["nu_ei"]["is_on"]:
            nu_ei_prof = self.nu_prof(t=t, nu_args=args["terms"]["fokker_planck"]["nu_ee"])
        else:
            nu_ei_prof = None

        # if self.cfg["terms"]["krook"]["is_on"]:
        #     nu_K_prof = self.nu_prof(t=t, nu_args=args["terms"]["krook"])
        # else:
        nu_K_prof = None

        electron_density_n = self.compute_charges(y["electron"])
        e, f = self.vpfp(
            f=y["electron"], a=y["a"], prev_ex=y["e"], dex_array=dex, nu_ee=nu_ee_prof, nu_ei=nu_ei_prof, nu_K=nu_K_prof
        )
        electron_density_np1 = self.compute_charges(f)

        a = self.wave_solver(
            a=y["a"], aold=y["prev_a"], djy_array=djy, electron_charge=0.5 * (electron_density_n + electron_density_np1)
        )

        return {"electron": f, "a": a["a"], "prev_a": a["prev_a"], "da": djy, "de": dex[self.vpfp.dex_save], "e": e}
