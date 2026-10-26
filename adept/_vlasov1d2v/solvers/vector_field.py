"""Vector-field composition for Vlasov-1D2V Diffrax solves.

The electrostatic/EM field machinery is reused from _vlasov1d verbatim by
feeding it MARGINALS F(x, v_par) = int f w_perp dv_perp — the field solve
only ever consumes velocity moments, and the moments of f under the
cylindrical weight equal the 1D moments of the marginal.
"""

from jax import Array
from jax import numpy as jnp

from adept._vlasov1d.grid import Grid
from adept._vlasov1d.simulation import EMDriverSet
from adept._vlasov1d.solvers.pushers import field
from adept._vlasov1d2v.solvers.pushers import fokker_planck, vlasov
from adept.functions import SpaceTimeEnvelopeFunction


class TimeIntegrator:
    """Shared field solver and Vlasov pushers for the 2V integrators."""

    def __init__(self, cfg: dict, grid: Grid):
        """Construct the (1D, marginal-fed) field solver and 2V pushers."""
        self.field_solve = field.ElectricFieldSolver(cfg, grid)
        self.species_grids = cfg["grid"]["species_grids"]
        self.species_params = cfg["grid"]["species_params"]
        self.wperp = {name: g["wperp"] for name, g in self.species_grids.items()}
        if cfg["terms"]["edfdv"] != "exponential":
            raise NotImplementedError("vlasov-1d2v supports edfdv: exponential only")
        self.edfdv = vlasov.VelocityExponential2V(self.species_grids, self.species_params)
        self.vdfdx = vlasov.SpaceExponential2V(grid.x, self.species_grids)

    def marginals(self, f_dict: dict) -> dict:
        """Return the marginal F(x, v_par) of every species distribution."""
        return {name: jnp.einsum("xvp,p->xv", f, self.wperp[name]) for name, f in f_dict.items()}


class LeapfrogIntegrator(TimeIntegrator):
    """Leapfrog integrator for the 2V Vlasov-Poisson system."""

    def __init__(self, cfg: dict, grid: Grid):
        """Initialize leapfrog coefficients and shared pusher state."""
        super().__init__(cfg, grid)
        self.dt = grid.dt
        self.dt_array = self.dt * jnp.array([0.0, 1.0])

    def __call__(self, f_dict: dict, a: Array, dex_array: Array, prev_ex: Array) -> tuple[Array, dict]:
        """Perform one leapfrog timestep for all species."""
        f_after_v = self.vdfdx(f_dict, dt=self.dt)
        f_for_field = f_dict if self.field_solve.hampere else f_after_v
        pond, e = self.field_solve(f_dict=self.marginals(f_for_field), a=a, prev_ex=prev_ex, dt=self.dt)
        f_dict = self.edfdv(f_after_v, e=e + dex_array[0], pond=pond, dt=self.dt)

        return e, f_dict


class SixthOrderHamIntegrator(TimeIntegrator):
    """6th-order Hamiltonian integrator (Crouseilles et al.) for the 2V system."""

    def __init__(self, cfg: dict, grid: Grid):
        """Initialize sixth-order Hamiltonian splitting coefficients."""
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
        """Perform one 6th-order timestep for all species."""
        drift_steps = [self.a1, self.a2, self.a3, self.a2, self.a1]
        kick_steps = [self.D1, self.D2, self.D3, self.D3, self.D2, self.D1]

        pond, self_consistent_ex = self.field_solve(f_dict=self.marginals(f_dict), a=a, prev_ex=None, dt=None)
        f_dict = self.edfdv(f_dict, e=dex_array[0] + self_consistent_ex, pond=pond, dt=kick_steps[0] * self.dt)

        for i, drift in enumerate(drift_steps):
            f_dict = self.vdfdx(f_dict, dt=drift * self.dt)
            pond, self_consistent_ex = self.field_solve(f_dict=self.marginals(f_dict), a=a, prev_ex=None, dt=None)
            f_dict = self.edfdv(
                f_dict, e=dex_array[i + 1] + self_consistent_ex, pond=pond, dt=kick_steps[i + 1] * self.dt
            )

        return self_consistent_ex, f_dict


class VlasovPoissonFokkerPlanck:
    """Vlasov-Poisson + Fokker-Planck timestep with marginal dfdt diagnostics."""

    def __init__(self, cfg: dict, grid: Grid):
        """Build the Vlasov-Poisson integrator, collisions, and diagnostics."""
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
        self.vlasov_cumulative = cfg["diagnostics"]["diag-vlasov-cumulative"]
        self.fp_cumulative = cfg["diagnostics"]["diag-fp-cumulative"]

    def __call__(
        self, f_dict: dict, a: Array, prev_ex: Array, dex_array: Array, nu_fp: Array, nu_K: Array
    ) -> tuple[Array, dict, dict]:
        """Advance Vlasov, collision, and diagnostic terms for one timestep."""
        e, f_vlasov = self.vlasov_poisson(f_dict, a, dex_array, prev_ex)

        f_fp = self.fp(nu_fp, nu_K, f_vlasov, dt=self.dt)

        diags = {}

        # Diagnostics are emitted on the MARGINAL F(x, v_par): that is what the
        # electrostatic dielectric (and the Delta-eps decomposition of the NLEPW
        # analysis) sees, and it keeps the save arrays 1D-shaped so the vlasov-1d
        # storage/postprocess machinery applies verbatim.
        #
        # These are per-step INCREMENTS, which VlasovMaxwell2V accumulates into
        # running time-integrals. Saving the instantaneous rate instead would
        # alias: the x-averaged wave-particle exchange carries 2*omega content
        # (period ~2.7 at k lambda_D = 0.3) while affordable save cadences are
        # dt_save ~ 10. Differencing the accumulated integral between two save
        # points instead returns the EXACT interval-averaged rate -- a box
        # filter applied before decimation -- for one extra array and no
        # filter design.
        ref_species = "electron" if "electron" in f_dict else next(iter(f_dict))
        if self.vlasov_cumulative or self.fp_cumulative:
            marg = self.vlasov_poisson.marginals
            if self.vlasov_cumulative:
                diags["diag-vlasov-cumulative"] = (
                    marg({ref_species: f_vlasov[ref_species]})[ref_species]
                    - marg({ref_species: f_dict[ref_species]})[ref_species]
                )
            if self.fp_cumulative:
                diags["diag-fp-cumulative"] = (
                    marg({ref_species: f_fp[ref_species]})[ref_species]
                    - marg({ref_species: f_vlasov[ref_species]})[ref_species]
                )

        return e, f_fp, diags


class VlasovMaxwell2V:
    """Vlasov-Poisson + Fokker-Planck timestep coupled to the transverse wave solver."""

    def __init__(
        self,
        cfg: dict,
        grid: Grid,
        drivers: EMDriverSet,
        nu_fp_prof: SpaceTimeEnvelopeFunction | None = None,
        nu_K_prof: SpaceTimeEnvelopeFunction | None = None,
    ):
        """Assemble the coupled electrostatic, transverse-wave, and driver operators."""
        self.cfg = cfg
        self.grid = grid
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof
        self.vpfp = VlasovPoissonFokkerPlanck(cfg, grid)
        beta = cfg["grid"]["beta"]
        c = 1.0 / beta
        self.wave_solver = field.WaveSolver(c=c, dx=grid.dx, dt=grid.dt)

        self.dt = grid.dt
        self.ey_driver = field.TransverseCurrentSourceDriver(grid.x_a, drivers=drivers.ey, c=c)
        self.ex_driver = field.LongitudinalElectricFieldDriver(grid.x, drivers=drivers.ex)

    def compute_electron_charge_density(self, f_dict):
        """Compute charge density from the electron distribution function."""
        charge_density = jnp.zeros_like(self.grid.x)
        if "electron" in f_dict:
            g = self.cfg["grid"]["species_grids"]["electron"]
            charge = self.cfg["grid"]["species_params"]["electron"]["charge"]
            F = jnp.einsum("xvp,p->xv", f_dict["electron"], g["wperp"])
            charge_density += charge * jnp.sum(F, axis=1) * g["dv"]
        return charge_density

    def __call__(self, t, y, args):
        """One Vlasov-Poisson-Fokker-Planck + wave-equation timestep."""
        dex = [self.ex_driver(t + dt, args) for dt in self.vpfp.vlasov_poisson.dt_array]
        djy = self.ey_driver(t + self.vpfp.vlasov_poisson.dt_array[1], args)

        nu_fp_val = self.nu_fp_prof(self.grid.x, t) if self.cfg["terms"]["fokker_planck"]["is_on"] else None
        nu_K_val = self.nu_K_prof(self.grid.x, t) if self.cfg["terms"]["krook"]["is_on"] else None

        f_dict = {k: v for k, v in y.items() if k in self.cfg["grid"]["species_grids"]}

        electron_charge_density_n = self.compute_electron_charge_density(f_dict)
        e, f_dict_new, diags = self.vpfp(
            f_dict=f_dict, a=y["a"], prev_ex=y["e"], dex_array=dex, nu_fp=nu_fp_val, nu_K=nu_K_val
        )
        electron_charge_density_np1 = self.compute_electron_charge_density(f_dict_new)

        a = self.wave_solver(
            a=y["a"],
            aold=y["prev_a"],
            djy_array=djy,
            electron_density=-0.5 * (electron_charge_density_n + electron_charge_density_np1),
        )

        result = {
            "a": a["a"],
            "prev_a": a["prev_a"],
            "da": djy,
            "de": dex[self.vpfp.dex_save],
            "e": e,
        }
        result.update(f_dict_new)
        # accumulate the per-step increments into running time-integrals
        for key, increment in diags.items():
            result[key] = y[key] + increment

        return result
