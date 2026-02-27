from jax import Array
from jax import numpy as jnp

from adept._vlasov1d.grid import Grid
from adept._vlasov1d.simulation import EMDriverSet
from adept._vlasov1d.solvers.pushers import field, fokker_planck, vlasov
from adept.functions import SpaceTimeEnvelopeFunction


def _is_parallel(parallel, axis: str) -> bool:
    """Return True if `axis` is requested in the parallel config tuple."""
    if not parallel:
        return False
    return axis in parallel


class TimeIntegrator:
    """
    This is the base class for all time integrators. This makes it so that we dont have to
    load the electric field solver and the Vlasov pushers in every time integrator

    The available solvers for E df/dv are "exponential" and "cubic-spline"
    The only solver for v df/dx is "exponential"

    :param cfg: Dict
    :param grid: Grid object for configuration space

    """

    def __init__(self, cfg: dict, grid: Grid):
        self.field_solve = field.ElectricFieldSolver(cfg, grid)
        self.species_grids = cfg["grid"]["species_grids"]
        self.species_params = cfg["grid"]["species_params"]
        parallel = cfg["grid"].get("parallel", False)
        self.edfdv = self.get_edfdv(cfg, parallel)
        self.vdfdx = vlasov.SpaceExponential(grid.x, self.species_grids, parallel=_is_parallel(parallel, "v"))

    def get_edfdv(self, cfg: dict, parallel):
        if cfg["terms"]["edfdv"] == "exponential":
            return vlasov.VelocityExponential(
                self.species_grids, self.species_params, parallel=_is_parallel(parallel, "x")
            )
        elif cfg["terms"]["edfdv"] == "cubic-spline":
            return vlasov.VelocityCubicSpline(
                self.species_grids, self.species_params, parallel=_is_parallel(parallel, "x")
            )
        else:
            raise NotImplementedError(f"{cfg['terms']['edfdv']} has not been implemented")


class LeapfrogIntegrator(TimeIntegrator):
    """Leapfrog integrator for Vlasov-Poisson system.

    Implements a standard leapfrog scheme where all species are pushed
    synchronously under the same self-consistent electric field.

    The field solve uses the total charge density from all species:
    rho = sum_s q_s * integral(f_s dv_s)

    Args:
        cfg: Configuration dictionary
        grid: Grid object for configuration space
    """

    def __init__(self, cfg: dict, grid: Grid):
        super().__init__(cfg, grid)
        self.dt = grid.dt
        self.dt_array = self.dt * jnp.array([0.0, 1.0])

    def __call__(self, f_dict: dict, a: Array, dex_array: Array, prev_ex: Array) -> tuple[Array, dict]:
        """Perform one leapfrog timestep for all species.

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution
            a: vector potential a[nx+2]
            dex_array: external driver field at substep times
            prev_ex: previous electric field E[nx]

        Returns:
            Tuple of (electric_field[nx], updated_f_dict)
        """
        f_after_v = self.vdfdx(f_dict, dt=self.dt)
        if self.field_solve.hampere:
            f_for_field = f_dict
        else:
            f_for_field = f_after_v
        pond, e = self.field_solve(f_dict=f_for_field, a=a, prev_ex=prev_ex, dt=self.dt)
        f_dict = self.edfdv(f_after_v, e=pond + e + dex_array[0], dt=self.dt)

        return e, f_dict


class SixthOrderHamIntegrator(TimeIntegrator):
    """6th-order Hamiltonian integrator for Vlasov-Poisson system.

    Implements the 6th-order symplectic integrator from Crouseilles et al.
    All species are pushed synchronously under the same self-consistent
    electric field at each substep.

    The field solve uses the total charge density from all species:
    rho = sum_s q_s * integral(f_s dv_s)

    Args:
        cfg: Configuration dictionary
        grid: Grid object for configuration space
    """

    def __init__(self, cfg: dict, grid: Grid):
        super().__init__(cfg, grid)
        self.dt = grid.dt

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

    def __call__(self, f_dict: dict, a: Array, dex_array: Array, prev_ex: Array) -> tuple[Array, dict]:
        """Perform one 6th-order timestep for all species.

        Args:
            f_dict: dict mapping species_name -> f[nx, nv] distribution
            a: vector potential a[nx+2]
            dex_array: external driver field at substep times
            prev_ex: previous electric field (unused, kept for interface consistency)

        Returns:
            Tuple of (electric_field[nx], updated_f_dict)
        """
        ponderomotive_force, self_consistent_ex = self.field_solve(f_dict=f_dict, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[0] + self_consistent_ex
        f_dict = self.edfdv(f_dict, e=force, dt=self.D1 * self.dt)

        f_dict = self.vdfdx(f_dict, dt=self.a1 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f_dict=f_dict, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[1] + self_consistent_ex

        f_dict = self.edfdv(f_dict, e=force, dt=self.D2 * self.dt)

        f_dict = self.vdfdx(f_dict, dt=self.a2 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f_dict=f_dict, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[2] + self_consistent_ex

        f_dict = self.edfdv(f_dict, e=force, dt=self.D3 * self.dt)

        f_dict = self.vdfdx(f_dict, dt=self.a3 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f_dict=f_dict, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[3] + self_consistent_ex

        f_dict = self.edfdv(f_dict, e=force, dt=self.D3 * self.dt)

        f_dict = self.vdfdx(f_dict, dt=self.a2 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f_dict=f_dict, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[4] + self_consistent_ex

        f_dict = self.edfdv(f_dict, e=force, dt=self.D2 * self.dt)

        f_dict = self.vdfdx(f_dict, dt=self.a1 * self.dt)
        ponderomotive_force, self_consistent_ex = self.field_solve(f_dict=f_dict, a=a, prev_ex=None, dt=None)
        force = ponderomotive_force + dex_array[5] + self_consistent_ex

        f_dict = self.edfdv(f_dict, e=force, dt=self.D1 * self.dt)

        return self_consistent_ex, f_dict


class VlasovPoissonFokkerPlanck:
    """Vlasov-Poisson + Fokker-Planck timestep for multi-species simulations.

    Combines a Vlasov-Poisson integrator (leapfrog or 6th-order Hamiltonian)
    with optional Fokker-Planck collisions. Handles dict-based multi-species
    distributions where each species evolves under the same self-consistent
    electric field.

    Args:
        cfg: Configuration dictionary
        grid: Grid object for configuration space

    Returns:
        Tuple of (electric_field, f_dict, diagnostics_dict)
    """

    def __init__(self, cfg: dict, grid: Grid):
        self.dt = grid.dt
        if cfg["terms"]["time"] == "sixth":
            self.vlasov_poisson = SixthOrderHamIntegrator(cfg, grid)
            self.dex_save = 3
        elif cfg["terms"]["time"] == "leapfrog":
            self.vlasov_poisson = LeapfrogIntegrator(cfg, grid)
            self.dex_save = 0
        else:
            raise NotImplementedError
        self.fp = fokker_planck.Collisions(cfg=cfg)
        self.vlasov_dfdt = cfg["diagnostics"]["diag-vlasov-dfdt"]
        self.fp_dfdt = cfg["diagnostics"]["diag-fp-dfdt"]

        if "hou_li_filter" in cfg["terms"]:
            hl_cfg = cfg["terms"]["hou_li_filter"]
            self.hou_li_filter_on = hl_cfg["is_on"]
            if self.hou_li_filter_on:
                self.hou_li_filter = vlasov.HouLiFilter(
                    species_grids=cfg["grid"]["species_grids"],
                    nx=cfg["grid"]["nx"],
                    alpha=hl_cfg["alpha"],
                    order=hl_cfg["order"],
                    dimensions=hl_cfg["dimensions"],
                )
        else:
            self.hou_li_filter_on = False

    def __call__(
        self, f_dict: dict, a: Array, prev_ex: Array, dex_array: Array, nu_fp: Array, nu_K: Array
    ) -> tuple[Array, dict, dict]:
        e, f_vlasov = self.vlasov_poisson(f_dict, a, dex_array, prev_ex)

        f_fp = self.fp(nu_fp, nu_K, f_vlasov, dt=self.dt)

        if self.hou_li_filter_on:
            f_fp = self.hou_li_filter(f_fp)

        diags = {}

        if self.vlasov_dfdt:
            # Compute diagnostics for each species
            for species_name in f_dict.keys():
                diags[f"diag-vlasov-dfdt-{species_name}"] = (f_vlasov[species_name] - f_dict[species_name]) / self.dt
        if self.fp_dfdt:
            # Compute diagnostics for each species
            for species_name in f_dict.keys():
                diags[f"diag-fp-dfdt-{species_name}"] = (f_fp[species_name] - f_vlasov[species_name]) / self.dt

        return e, f_fp, diags


class VlasovMaxwell:
    """
    This class contains the Vlasov-Poisson + Fokker-Planck timestep and the wave equation solver

    :param cfg:
    :param grid: Grid object for configuration space
    :param nu_fp_prof: Optional SpaceTimeEnvelopeFunction for Fokker-Planck collision frequency
    :param nu_K_prof: Optional SpaceTimeEnvelopeFunction for Krook collision frequency
    """

    def __init__(
        self,
        cfg: dict,
        grid: Grid,
        drivers: EMDriverSet,
        nu_fp_prof: SpaceTimeEnvelopeFunction | None = None,
        nu_K_prof: SpaceTimeEnvelopeFunction | None = None,
    ):
        self.cfg = cfg
        self.grid = grid
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof
        self.vpfp = VlasovPoissonFokkerPlanck(cfg, grid)
        # beta = 1/c_norm is still read from cfg["grid"] for now
        beta = cfg["grid"]["beta"]
        self.wave_solver = field.WaveSolver(c=1.0 / beta, dx=grid.dx, dt=grid.dt)

        self.dt = grid.dt
        self.ey_driver = field.Driver(grid.x_a, drivers=drivers.ey)
        self.ex_driver = field.Driver(grid.x, drivers=drivers.ex)

    def compute_electron_charge_density(self, f_dict):
        """Compute charge density from the electron distribution function.
        """
        charge_density = jnp.zeros_like(self.grid.x)
        if "electron" in f_dict:
            dv = self.cfg["grid"]["species_grids"]["electron"]["dv"]
            charge = self.cfg["grid"]["species_params"]["electron"]["charge"]
            # Sum over velocity axis (axis=1) to get spatial density, then multiply by charge
            f = f_dict["electron"]
            charge_density += charge * jnp.sum(f, axis=1) * dv
        return charge_density

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

        # Evaluate collision frequency profiles at current time
        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            nu_fp_val = self.nu_fp_prof(self.grid.x, t)
        else:
            nu_fp_val = None

        if self.cfg["terms"]["krook"]["is_on"]:
            nu_K_val = self.nu_K_prof(self.grid.x, t)
        else:
            nu_K_val = None

        # Extract all species distributions from state
        f_dict = {k: v for k, v in y.items() if k in self.cfg["grid"]["species_grids"]}

        electron_charge_density_n = self.compute_electron_charge_density(f_dict)
        e, f_dict_new, diags = self.vpfp(
            f_dict=f_dict, a=y["a"], prev_ex=y["e"], dex_array=dex, nu_fp=nu_fp_val, nu_K=nu_K_val
        )
        electron_charge_density_np1 = self.compute_electron_charge_density(f_dict_new)

        a = self.wave_solver(
            a=y["a"], aold=y["prev_a"], djy_array=djy, electron_density=-0.5 * (electron_charge_density_n + electron_charge_density_np1)
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
        result.update(f_dict_new)
        result.update(diags)

        return result
