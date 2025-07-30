from jax import Array
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._vlasov2s1d.solvers.pushers import field, fokker_planck, vlasov


class TimeIntegrator:
    """
    This is the base class for all time integrators for 2-species Vlasov solver.
    This handles both electrons and ions with their respective velocity grids.

    The available solvers for E df/dv are "exponential" and "cubic-spline"
    The only solver for v df/dx is "exponential"

    :param cfg: Dict

    """

    def __init__(self, cfg: dict):
        self.field_solve = field.ElectricFieldSolver(cfg)
        # Electron pushers
        self.edfdv_e = self.get_edfdv(cfg, species="electron")
        self.vdfdx_e = vlasov.SpaceExponential(cfg)
        # Ion pushers
        self.edfdv_i = self.get_edfdv(cfg, species="ion")
        self.vdfdx_i = vlasov.IonSpaceExponential(cfg)

    def get_edfdv(self, cfg: dict, species: str = "electron"):
        if species == "electron":
            if cfg["terms"]["edfdv"] == "exponential":
                return vlasov.VelocityExponential(cfg)
            elif cfg["terms"]["edfdv"] == "cubic-spline":
                return vlasov.VelocityCubicSpline(cfg)
            else:
                raise NotImplementedError(f"{cfg['terms']['edfdv']} has not been implemented for electrons")
        elif species == "ion":
            if cfg["terms"]["edfdv"] == "exponential":
                return vlasov.IonVelocityExponential(cfg)
            elif cfg["terms"]["edfdv"] == "cubic-spline":
                return vlasov.IonVelocityCubicSpline(cfg)
            else:
                raise NotImplementedError(f"{cfg['terms']['edfdv']} has not been implemented for ions")
        else:
            raise ValueError(f"Unknown species: {species}")


class Leapfrog2SIntegrator(TimeIntegrator):
    """
    This is a leapfrog integrator for 2-species plasma (electrons + ions)

    :param cfg:
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dt = cfg["grid"]["dt"]
        self.dt_array = self.dt * jnp.array([0.0, 1.0])

    def __call__(
        self, f_e: Array, f_i: Array, a: Array, dex_array: Array, prev_ex: Array
    ) -> tuple[Array, Array, Array]:
        # Step 1: Advance both species in space (v df/dx)
        f_e_after_v = self.vdfdx_e(f=f_e, dt=self.dt)
        f_i_after_v = self.vdfdx_i(f=f_i, dt=self.dt)

        # Step 2: Compute electric field using total charge density
        if self.field_solve.hampere:
            f_e_for_field = f_e
            f_i_for_field = f_i
        else:
            f_e_for_field = f_e_after_v
            f_i_for_field = f_i_after_v

        pond, e = self.field_solve(f_e=f_e_for_field, f_i=f_i_for_field, a=a, prev_ex=prev_ex, dt=self.dt)

        # Step 3: Advance both species in velocity (E df/dv)
        total_e_field = pond + e + dex_array[0]
        f_e_final = self.edfdv_e(f=f_e_after_v, e=total_e_field, dt=self.dt)
        f_i_final = self.edfdv_i(f=f_i_after_v, e=total_e_field, dt=self.dt)

        return e, f_e_final, f_i_final


class Vlasov2SPoissonFokkerPlanck:
    """
    This class contains the 2-species Vlasov-Poisson + Fokker-Planck timestep

    :param cfg: Configuration dictionary

    :return: Tuple of the electric field and the distribution functions
    """

    def __init__(self, cfg: dict):
        self.dt = cfg["grid"]["dt"]
        self.v = cfg["grid"]["v"]
        self.v_i = cfg["grid"]["v_i"]
        if cfg["terms"]["time"] == "leapfrog":
            self.vlasov_poisson = Leapfrog2SIntegrator(cfg)
            self.dex_save = 0
        else:
            raise NotImplementedError("Only leapfrog integrator implemented for 2-species solver")

        # Collision operators for both species
        self.fp_e = fokker_planck.Collisions(cfg=cfg)
        self.fp_i = fokker_planck.IonCollisions(cfg=cfg)

        self.vlasov_dfdt = cfg["diagnostics"]["diag-vlasov-dfdt"]
        self.fp_dfdt = cfg["diagnostics"]["diag-fp-dfdt"]

    def __call__(
        self,
        f_e: Array,
        f_i: Array,
        a: Array,
        prev_ex: Array,
        dex_array: Array,
        nu_fp_e: Array,
        nu_K_e: Array,
        nu_fp_i: Array,
        nu_K_i: Array,
    ) -> tuple[Array, Array, Array]:
        e, f_e_vlasov, f_i_vlasov = self.vlasov_poisson(f_e, f_i, a, dex_array, prev_ex)

        # Apply collisions to both species
        # f_e_fp = self.fp_e(nu_fp_e, nu_K_e, f_e_vlasov, dt=self.dt)
        # f_i_fp = self.fp_i(nu_fp_i, nu_K_i, f_i_vlasov, dt=self.dt)

        diags = {}

        f_e_fp = f_e_vlasov
        f_i_fp = f_i_vlasov

        # if self.vlasov_dfdt:
        #     diags["diag-vlasov-dfdt-e"] = (f_e_vlasov - f_e) / self.dt
        #     diags["diag-vlasov-dfdt-i"] = (f_i_vlasov - f_i) / self.dt
        # if self.fp_dfdt:
        #     diags["diag-fp-dfdt-e"] = (f_e_fp - f_e_vlasov) / self.dt
        #     diags["diag-fp-dfdt-i"] = (f_i_fp - f_i_vlasov) / self.dt

        return e, f_e_fp, f_i_fp, diags


class Vlasov2SMaxwell:
    """
    This class contains the 2-species Vlasov-Maxwell solver with Fokker-Planck collisions

    :param cfg:
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.v2spfp = Vlasov2SPoissonFokkerPlanck(cfg)
        self.wave_solver = field.WaveSolver(c=1.0 / cfg["grid"]["beta"], dx=cfg["grid"]["dx"], dt=cfg["grid"]["dt"])

        self.dt = self.cfg["grid"]["dt"]
        self.ey_driver = field.Driver(cfg["grid"]["x_a"], driver_key="ey")
        self.ex_driver = field.Driver(cfg["grid"]["x"], driver_key="ex")

    def compute_charges(self, f_e, f_i):
        """Compute charge densities for both species"""
        electron_density = jnp.sum(f_e, axis=1) * self.cfg["grid"]["dv"]
        ion_density = jnp.sum(f_i, axis=1) * self.cfg["grid"]["dv_i"]
        return electron_density, ion_density

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
        This is the main timestep for 2-species Vlasov-Maxwell + Fokker-Planck

        :param t:
        :param y:
        :param args:

        :return:
        """

        dex = [self.ex_driver(t + dt, args) for dt in self.v2spfp.vlasov_poisson.dt_array]
        djy = self.ey_driver(t + self.v2spfp.vlasov_poisson.dt_array[1], args)

        # Collision frequencies for both species
        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            nu_fp_prof_e = self.nu_prof(t=t, nu_args=args["terms"]["fokker_planck"])
            nu_fp_prof_i = self.nu_prof(t=t, nu_args=args["terms"]["fokker_planck"])  # Can be different
        else:
            nu_fp_prof_e = None
            nu_fp_prof_i = None

        if self.cfg["terms"]["krook"]["is_on"]:
            nu_K_prof_e = self.nu_prof(t=t, nu_args=args["terms"]["krook"])
            nu_K_prof_i = self.nu_prof(t=t, nu_args=args["terms"]["krook"])  # Can be different
        else:
            nu_K_prof_e = None
            nu_K_prof_i = None

        electron_density_n, ion_density_n = self.compute_charges(y["electron"], y["ion"])
        e, f_e, f_i, diags = self.v2spfp(
            f_e=y["electron"],
            f_i=y["ion"],
            a=y["a"],
            prev_ex=y["e"],
            dex_array=dex,
            nu_fp_e=nu_fp_prof_e,
            nu_K_e=nu_K_prof_e,
            nu_fp_i=nu_fp_prof_i,
            nu_K_i=nu_K_prof_i,
        )
        electron_density_np1, ion_density_np1 = self.compute_charges(f_e, f_i)

        # Total charge density for wave solver (electrons negative, ions positive)
        total_charge_n = -electron_density_n + ion_density_n  # electrons: -1, ions: +1
        total_charge_np1 = -electron_density_np1 + ion_density_np1

        a = self.wave_solver(
            a=y["a"], aold=y["prev_a"], djy_array=djy, electron_charge=0.5 * (total_charge_n + total_charge_np1)
        )

        return {
            "electron": f_e,
            "ion": f_i,
            "a": a["a"],
            "prev_a": a["prev_a"],
            "da": djy,
            "de": dex[self.v2spfp.dex_save],
            "e": e,
        } | diags
