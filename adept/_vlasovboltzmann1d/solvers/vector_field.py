from jax import Array
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._vlasov1d.solvers.pushers import field, fokker_planck, vlasov


class TimeIntegrator:
    """
    This is the base class for all time integrators. This makes it so that we dont have to
    load the electric field solver and the Vlasov pushers in every time integrator

    The available solvers for E df/dv are "exponential" and "cubic-spline"
    The only solver for v df/dx is "exponential"

    :param cfg: Dict

    """

    def __init__(self, cfg: dict):
        # self.field_solve = field.ElectricFieldSolver(cfg)
        self.edfdv = self.get_edfdv(cfg)
        self.vdfdx = vlasov.SpaceExponential(cfg)

    def field_solve(self, f: Array) -> Array:
        ni = jnp.sum(f, axis=1) * self.cfg["grid"]["dv"]
        phi = ni / self.n0 - 1.0
        return jnp.gradient(phi) / self.cfg["grid"]["dx"]

    def get_edfdv(self, cfg: dict):
        if cfg["terms"]["edfdv"] == "exponential":
            return vlasov.VelocityExponential(cfg)
        elif cfg["terms"]["edfdv"] == "cubic-spline":
            return vlasov.VelocityCubicSpline(cfg)
        else:
            raise NotImplementedError(f"{cfg['terms']['edfdv']} has not been implemented")


class LeapfrogIntegrator(TimeIntegrator):
    """
    This is a leapfrog integrator

    :param cfg:
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dt = cfg["grid"]["dt"]
        self.dt_array = self.dt * jnp.array([0.0, 1.0])

    def __call__(self, f: Array, dex_array: Array) -> tuple[Array, Array]:
        f_after_v = self.vdfdx(f=f, dt=self.dt)
        e = self.field_solve(f=f_after_v)
        f = self.edfdv(f=f_after_v, e=e + dex_array[0], dt=self.dt)

        return e, f


class VlasovFokkerPlanckBoltzmann:
    """
    This class contains the Vlasov-Poisson + Fokker-Planck timestep

    :param cfg: Configuration dictionary

    :return: Tuple of the electric field and the distribution function
    """

    def __init__(self, cfg: dict):
        self.dt = cfg["grid"]["dt"]
        self.v = cfg["grid"]["v"]
        self.vlasov_boltzmann = LeapfrogIntegrator(cfg)
        self.fp = fokker_planck.Collisions(cfg=cfg)

    def __call__(
        self, f: Array, dex_array: Array, nu_fp: Array, nu_K: Array
    ) -> tuple[Array, Array]:
        e, f_vlasov = self.vlasov_boltzmann(f, dex_array)
        f_fp = self.fp(nu_fp, nu_K, f_vlasov, dt=self.dt)

        return e, f_fp


class VFPBoltzmann:
    """
    This class contains the Vlasov-Poisson + Fokker-Planck timestep and the wave equation solver

    :param cfg:
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.vpfp = VlasovFokkerPlanckBoltzmann(cfg)

        self.dt = self.cfg["grid"]["dt"]
        self.ex_driver = field.Driver(cfg["grid"]["x"], driver_key="ex")

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

        dex = [self.ex_driver(t + dt, args) for dt in [0.0, self.dt]]

        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            nu_fp_prof = self.nu_prof(t=t, nu_args=args["terms"]["fokker_planck"])
        else:
            nu_fp_prof = None

        if self.cfg["terms"]["krook"]["is_on"]:
            nu_K_prof = self.nu_prof(t=t, nu_args=args["terms"]["krook"])
        else:
            nu_K_prof = None

        e, f = self.vpfp(f=y["ion"], dex_array=dex, nu_fp=nu_fp_prof, nu_K=nu_K_prof)

        return {"ion": f, "de": dex[0], "e": e}
