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
        self.field_solve = field.ElectricFieldSolver(cfg)
        self.species_grids = cfg["grid"]["species_grids"]
        self.species_params = cfg["grid"]["species_params"]
        self.edfdv = self.get_edfdv(cfg)
        self.vdfdx = vlasov.SpaceExponential(cfg["grid"]["x"], self.species_grids)

    def get_edfdv(self, cfg: dict):
        if cfg["terms"]["edfdv"] == "exponential":
            return vlasov.VelocityExponential(self.species_grids, self.species_params)
        elif cfg["terms"]["edfdv"] == "cubic-spline":
            return vlasov.VelocityCubicSpline(self.species_grids, self.species_params)
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

    def __call__(self, f: dict, a: Array, dex_array: Array, prev_ex: Array) -> tuple[Array, dict]:
        f_after_v = self.vdfdx(f, dt=self.dt)
        if self.field_solve.hampere:
            f_for_field = f
        else:
            f_for_field = f_after_v
        pond, e = self.field_solve(f=f_for_field, a=a, prev_ex=prev_ex, dt=self.dt)
        f = self.edfdv(f_after_v, e=pond + e + dex_array[0], dt=self.dt)

        return e, f


class SixthOrderHamIntegrator(TimeIntegrator):
    """
    This class contains the 6th order Hamiltonian integrator from Crousseilles

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

    def __call__(self, f: dict, a: Array, dex_array: Array, prev_ex: Array) -> tuple[Array, dict]:
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[0] + self_consistent_ex
        f = self.edfdv(f, e=force, dt=self.D1 * self.dt)

        f = self.vdfdx(f, dt=self.a1 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[1] + self_consistent_ex

        f = self.edfdv(f, e=force, dt=self.D2 * self.dt)

        f = self.vdfdx(f, dt=self.a2 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[2] + self_consistent_ex

        f = self.edfdv(f, e=force, dt=self.D3 * self.dt)

        f = self.vdfdx(f, dt=self.a3 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[3] + self_consistent_ex

        f = self.edfdv(f, e=force, dt=self.D3 * self.dt)

        f = self.vdfdx(f, dt=self.a2 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[4] + self_consistent_ex

        f = self.edfdv(f, e=force, dt=self.D2 * self.dt)

        f = self.vdfdx(f, dt=self.a1 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f=f, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[5] + self_consistent_ex

        f = self.edfdv(f, e=force, dt=self.D1 * self.dt)

        return self_consistent_ex, f


class VlasovPoissonFokkerPlanck:
    """
    This class contains the Vlasov-Poisson + Fokker-Planck timestep

    :param cfg: Configuration dictionary

    :return: Tuple of the electric field and the distribution function
    """

    def __init__(self, cfg: dict):
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
        self.vlasov_dfdt = cfg["diagnostics"]["diag-vlasov-dfdt"]
        self.fp_dfdt = cfg["diagnostics"]["diag-fp-dfdt"]

    def __call__(
        self, f: dict, a: Array, prev_ex: Array, dex_array: Array, nu_fp: Array, nu_K: Array
    ) -> tuple[Array, dict, dict]:
        e, f_vlasov = self.vlasov_poisson(f, a, dex_array, prev_ex)

        f_fp = self.fp(nu_fp, nu_K, f_vlasov, dt=self.dt)
        diags = {}

        if self.vlasov_dfdt:
            # Compute diagnostics for each species
            for species_name in f.keys():
                diags[f"diag-vlasov-dfdt-{species_name}"] = (f_vlasov[species_name] - f[species_name]) / self.dt
        if self.fp_dfdt:
            # Compute diagnostics for each species
            for species_name in f.keys():
                diags[f"diag-fp-dfdt-{species_name}"] = (f_fp[species_name] - f_vlasov[species_name]) / self.dt

        return e, f_fp, diags


class VlasovMaxwell:
    """
    This class contains the Vlasov-Poisson + Fokker-Planck timestep and the wave equation solver

    :param cfg:
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.vpfp = VlasovPoissonFokkerPlanck(cfg)
        self.wave_solver = field.WaveSolver(c=1.0 / cfg["grid"]["beta"], dx=cfg["grid"]["dx"], dt=cfg["grid"]["dt"])

        self.dt = self.cfg["grid"]["dt"]
        self.ey_driver = field.Driver(cfg["grid"]["x_a"], driver_key="ey")
        self.ex_driver = field.Driver(cfg["grid"]["x"], driver_key="ex")

    def compute_charges(self, f_dict):
        """Compute charge density from distribution functions.

        For a dict of species distributions, sum over all species with their charge-to-mass ratios.
        """
        charge_density = jnp.zeros_like(self.cfg["grid"]["x"])
        for species_name, f in f_dict.items():
            dv = self.cfg["grid"]["species_grids"][species_name]["dv"]
            charge = self.cfg["grid"]["species_params"][species_name]["charge"]
            # Sum over velocity axis (axis=1) to get spatial density, then multiply by charge
            charge_density += charge * jnp.sum(f, axis=1) * dv
        return charge_density

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

        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            nu_fp_prof = self.nu_prof(t=t, nu_args=args["terms"]["fokker_planck"])
        else:
            nu_fp_prof = None

        if self.cfg["terms"]["krook"]["is_on"]:
            nu_K_prof = self.nu_prof(t=t, nu_args=args["terms"]["krook"])
        else:
            nu_K_prof = None

        # Extract all species distributions from state
        f_dict = {k: v for k, v in y.items() if k in self.cfg["grid"]["species_grids"]}

        electron_density_n = self.compute_charges(f_dict)
        e, f, diags = self.vpfp(
            f=f_dict, a=y["a"], prev_ex=y["e"], dex_array=dex, nu_fp=nu_fp_prof, nu_K=nu_K_prof
        )
        electron_density_np1 = self.compute_charges(f)

        a = self.wave_solver(
            a=y["a"], aold=y["prev_a"], djy_array=djy, electron_charge=0.5 * (electron_density_n + electron_density_np1)
        )

        # Build result dict with all species distributions
        result = {
            "a": a["a"],
            "prev_a": a["prev_a"],
            "da": djy,
            "de": dex[self.vpfp.dex_save],
            "e": e,
        }
        # Add all species distributions
        result.update(f)
        result.update(diags)

        return result
