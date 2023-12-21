from typing import Tuple

from functools import partial

from jax import numpy as jnp
import diffrax

from adept.vlasov1d.pushers import field, fokker_planck, vlasov
from adept.tf1d.pushers import get_envelope


class Stepper(diffrax.Euler):
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful


# class VectorField:
#     """
#     This class contains the function that updates the state
#
#     All the pushers are chosen and initialized here and a single time-step is defined here.
#
#     :param cfg:
#     :return:
#     """
#
#     def __init__(self, cfg):
#         self.vlasov = Vlasov(cfg)
#
#     def __call__(self, t, y, args):
#         return self.vlasov(t, y, args)
#


class SixthOrderHamIntegrator:
    def __init__(self, cfg):
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
        self.field_solve = field.ElectricFieldSolver(cfg)
        self.edfdv = vlasov.VelocityExponential(cfg)
        self.vdfdx = vlasov.SpaceExponential(cfg)

    def __call__(self, f, a, dex_array, total_force):
        force, _, _ = self.field_solve(dex_array=dex_array[0], f=f, a=a, total_force=None, dt=None)
        f = self.edfdv(f=f, e=force, dt=self.D1 * self.dt)

        f = self.vdfdx(f=f, dt=self.a1 * self.dt)
        force, _, _ = self.field_solve(dex_array=dex_array[1], f=f, a=a, total_force=None, dt=None)

        f = self.edfdv(f=f, e=force, dt=self.D2 * self.dt)

        f = self.vdfdx(f=f, dt=self.a2 * self.dt)
        force, _, _ = self.field_solve(dex_array=dex_array[2], f=f, a=a, total_force=None, dt=None)

        f = self.edfdv(f=f, e=force, dt=self.D3 * self.dt)

        f = self.vdfdx(f=f, dt=self.a3 * self.dt)
        force_save, pond_save, e_save = self.field_solve(dex_array=dex_array[3], f=f, a=a, total_force=None, dt=None)

        f = self.edfdv(f=f, e=force_save, dt=self.D3 * self.dt)

        f = self.vdfdx(f=f, dt=self.a2 * self.dt)
        force, _, _ = self.field_solve(dex_array=dex_array[4], f=f, a=a, total_force=None, dt=None)

        f = self.edfdv(f=f, e=force, dt=self.D2 * self.dt)

        f = self.vdfdx(f=f, dt=self.a1 * self.dt)
        force, _, _ = self.field_solve(dex_array=dex_array[5], f=f, a=a, total_force=None, dt=None)

        f = self.edfdv(f=f, e=force, dt=self.D1 * self.dt)

        return e_save, f, force_save, pond_save


class VlasovPoissonFokkerPlanck:
    def __init__(self, cfg):
        self.dt = cfg["grid"]["dt"]
        self.v = cfg["grid"]["v"]
        self.vlasov_poisson = SixthOrderHamIntegrator(cfg)
        self.fp = fokker_planck.Collisions(cfg=cfg)

    def __call__(
        self, f, a, total_force, dex_array, nu_fp, nu_K
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        e, f, force, pond = self.vlasov_poisson(f, a, dex_array, total_force)
        f = self.fp(nu_fp, nu_K, f, dt=self.dt)

        return e, f, force, pond


class VlasovMaxwell:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vpfp = VlasovPoissonFokkerPlanck(cfg)
        self.wave_solver = field.WaveSolver(c=cfg["grid"]["c_light"], dx=cfg["grid"]["dx"], dt=cfg["grid"]["dt"])
        self.compute_charges = partial(jnp.trapz, dx=cfg["grid"]["dv"], axis=1)
        self.dt = self.cfg["grid"]["dt"]
        self.driver = field.Driver(cfg["grid"]["x"])

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

        de = [self.driver(t + dt, args) for dt in self.vpfp.vlasov_poisson.dt_array]
        dex = [val[0] for val in de]
        djy = de[0][1]

        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            nu_fp_prof = self.nu_prof(t=t, nu_args=args["terms"]["fokker_planck"])
        else:
            nu_fp_prof = None

        if self.cfg["terms"]["krook"]["is_on"]:
            nu_K_prof = self.nu_prof(t=t, nu_args=args["terms"]["krook"])
        else:
            nu_K_prof = None

        electron_density_n = self.compute_charges(y["electron"])
        e, f, force, pond = self.vpfp(y["electron"], y["a"], y["e"], dex, nu_fp_prof, nu_K_prof)
        electron_density_np1 = self.compute_charges(f)

        a = self.wave_solver(
            a=y["a"],
            aold=y["prev_a"],
            djy_array=djy[2],
            electron_charge=0.5 * (electron_density_n + electron_density_np1),
        )

        return {"electron": f, "a": a[0], "prev_a": a[1], "da": djy, "de": dex[3], "e": e}
