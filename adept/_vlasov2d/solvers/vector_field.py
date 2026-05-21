"""Time integrator for the Vlasov-2D solver.

One full timestep, Strang-split end-to-end:

    Phase A (configuration streaming, 1/2 dt):
        ½ x-stream  →  ½ y-stream

    Phase B (velocity push, full dt):
        ½ Ex  →  ½ Ey  →  full B-rotation  →  ½ Ey  →  ½ Ex
        (E shifts commute, so the inner pair can also be ¼ + ¼ + ¼ + ¼.)

    Phase C (configuration streaming, 1/2 dt):
        ½ y-stream  →  ½ x-stream

    Phase D (Maxwell update, full dt):
        spectral Strang-split (½ B, full E with current J, ½ B)

    Phase E (collisions, full dt):
        Dougherty FP + Krook + optional Hou-Li filter

The current J fed to Maxwell is computed from f AFTER the kinetic push but
BEFORE the Maxwell update, plus prescribed driver currents at t + dt/2.
"""

from jax import Array
from jax import numpy as jnp

from adept._vlasov2d.grid import Grid
from adept._vlasov2d.simulation import EMDriverSet, Vlasov2DSimulation
from adept._vlasov2d.solvers.pushers import field, fokker_planck, vlasov


class TimeIntegrator:
    """A single Strang-split kinetic + Maxwell timestep."""

    def __init__(self, cfg: dict, grid: Grid, simulation: Vlasov2DSimulation):
        species_grids = cfg["grid"]["species_grids"]
        species_params = cfg["grid"]["species_params"]

        self.dt = grid.dt
        self.cfg = cfg

        # Streaming
        self.stream_x = vlasov.SpaceExponentialX(grid.kxr, species_grids)
        self.stream_y = vlasov.SpaceExponentialY(grid.kyr, species_grids)

        # Velocity push
        if cfg["terms"]["edfdv"] == "exponential":
            self.e_push = vlasov.VelocityExponentialE(species_grids, species_params)
        else:
            self.e_push = vlasov.VelocitySL2DE(species_grids, species_params)
        self.b_rot = vlasov.VelocityRotateB(species_grids, species_params)

        # Maxwell
        beta = cfg["grid"]["beta"]
        c = 1.0 / beta
        self.maxwell = field.MaxwellSpectral(grid.kx, grid.ky, c=c)
        self.currents = field.CurrentMoments(species_grids, species_params)

        # Drivers (always present; lists may be empty). Treated as Maxwell current sources.
        self.driver_jx = field.EMDriverFieldSource(grid.x, grid.y, simulation.drivers.ex)
        self.driver_jy = field.EMDriverFieldSource(grid.x, grid.y, simulation.drivers.ey)

        # Collisions
        self.collisions = fokker_planck.Collisions(cfg)

        # Filter
        hl = cfg["terms"].get("hou_li_filter", {"is_on": False})
        if hl.get("is_on", False):
            self.filter = vlasov.HouLiFilter(
                species_grids=species_grids,
                nx=cfg["grid"]["nx"],
                ny=cfg["grid"]["ny"],
                alpha=hl["alpha"],
                order=hl["order"],
                dimensions=hl["dimensions"],
            )
            self.filter_on = True
        else:
            self.filter_on = False

    def kinetic_substep(self, f_dict: dict, ex: Array, ey: Array, bz: Array) -> dict:
        """Strang-split kinetic push for one full dt (no Maxwell update)."""
        dt = self.dt
        f_dict = self.stream_x(f_dict, dt=0.5 * dt)
        f_dict = self.stream_y(f_dict, dt=0.5 * dt)

        f_dict = self.e_push.push_vx(f_dict, ex, dt=0.5 * dt)
        f_dict = self.e_push.push_vy(f_dict, ey, dt=0.5 * dt)
        f_dict = self.b_rot(f_dict, bz, dt=dt)
        f_dict = self.e_push.push_vy(f_dict, ey, dt=0.5 * dt)
        f_dict = self.e_push.push_vx(f_dict, ex, dt=0.5 * dt)

        f_dict = self.stream_y(f_dict, dt=0.5 * dt)
        f_dict = self.stream_x(f_dict, dt=0.5 * dt)
        return f_dict


class VlasovMaxwell:
    """Top-level RHS for diffrax: takes y at time t, returns y at time t+dt."""

    def __init__(
        self,
        cfg: dict,
        grid: Grid,
        simulation: Vlasov2DSimulation,
    ):
        self.cfg = cfg
        self.grid = grid
        self.simulation = simulation
        self.integrator = TimeIntegrator(cfg, grid, simulation)
        self.dt = grid.dt

        # cached for diagnostics
        self.species_grids = cfg["grid"]["species_grids"]

        # collision profiles
        self.nu_fp_prof = simulation.nu_fp_prof
        self.nu_K_prof = simulation.nu_K_prof
        self.fp_on = cfg["terms"]["fokker_planck"]["is_on"]
        self.krook_on = cfg["terms"]["krook"]["is_on"]
        self.skip_maxwell = bool(cfg["terms"].get("skip_maxwell", False))

    def _compute_nus(self, t: float):
        nu_fp = self.nu_fp_prof(self.grid.x, self.grid.y, t) if (self.fp_on and self.nu_fp_prof is not None) else None
        nu_K = self.nu_K_prof(self.grid.x, self.grid.y, t) if (self.krook_on and self.nu_K_prof is not None) else None
        return nu_fp, nu_K

    def __call__(self, t, y, args):
        f_dict = {name: y[name] for name in self.species_grids.keys()}
        ex = y["ex"]
        ey = y["ey"]
        bz = y["bz"]

        # Kinetic push uses fields at time t throughout the substeps
        f_dict = self.integrator.kinetic_substep(f_dict, ex, ey, bz)

        # Self-consistent + driver currents at the midpoint
        t_mid = t + 0.5 * self.dt
        jx_drv = self.integrator.driver_jx(t_mid)
        jy_drv = self.integrator.driver_jy(t_mid)
        if not self.skip_maxwell:
            jx_self, jy_self = self.integrator.currents(f_dict)
            ex, ey, bz = self.integrator.maxwell(ex, ey, bz, jx_self + jx_drv, jy_self + jy_drv, self.dt)

        # Collisions
        nu_fp, nu_K = self._compute_nus(t)
        f_dict = self.integrator.collisions(nu_fp, nu_K, f_dict, self.dt)

        if self.integrator.filter_on:
            f_dict = self.integrator.filter(f_dict)

        out = {"ex": ex, "ey": ey, "bz": bz}
        out.update(f_dict)
        out["jx_driver"] = jx_drv
        out["jy_driver"] = jy_drv
        return out
