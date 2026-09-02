"""Top-level PIC-1D vector field.

The ``ergoExo`` infrastructure wraps this in a ``diffrax.ODETerm`` and steps it
with the dummy Euler stepper. Each call performs one full symplectic step on
all species and returns the updated state dict (including the most recent
self-consistent ``E`` and external driver field for diagnostics).

When transverse EM drivers (``drivers.ey``) are configured the same step also
advances a transverse vector potential ``a(x)`` via a 2nd-order leapfrog wave
solver — identical to the one used by Vlasov-1D's :class:`VlasovMaxwell` —
and adds a ponderomotive force ``-½ ∂_x(a²)`` to the longitudinal particle
kick. The particles themselves remain 1D1V.
"""

from typing import cast

import equinox as eqx
from jax import numpy as jnp

from adept._pic1d.solvers.pushers.field import (
    ElectronChargeDensity,
    LongitudinalElectricFieldDriver,
)
from adept._pic1d.solvers.pushers.push import (
    LeapfrogIntegrator,
    Yoshida4Integrator,
    _Kicker,
)
from adept._vlasov1d.grid import Grid
from adept._vlasov1d.simulation import EMDriverSet
from adept._vlasov1d.solvers.pushers.field import (
    TransverseCurrentSourceDriver,
    WaveSolver,
)


class PIC1DVectorField(eqx.Module):
    grid: Grid
    species_names: list[str]
    stepper: LeapfrogIntegrator | Yoshida4Integrator
    diagnostic_driver: LongitudinalElectricFieldDriver
    has_ey: bool
    electron_charge_density: ElectronChargeDensity | None
    wave_solver: WaveSolver | None
    wave_solver_active: bool
    ey_driver: TransverseCurrentSourceDriver | None

    def __init__(self, cfg: dict, grid: Grid, drivers: EMDriverSet):
        self.grid = grid
        self.species_names = list(cfg["grid"]["species_params"].keys())

        shape = cfg["grid"]["particle_shape"]
        static_bg = cfg["grid"].get("ion_charge")

        kicker = cast(
            _Kicker,
            _Kicker(
                nx=grid.nx,
                dx=grid.dx,
                xmin=grid.xmin,
                species_params=cfg["grid"]["species_params"],
                one_over_kx=grid.one_over_kx,
                xax=grid.x,
                ex_drivers=drivers.ex,
                static_charge_density=static_bg,
                shape=shape,
            ),
        )

        time_scheme = cfg["terms"]["time"]
        if time_scheme == "leapfrog":
            self.stepper = cast(
                LeapfrogIntegrator,
                LeapfrogIntegrator(kicker, grid.xmin, grid.xmax, grid.dt),
            )
        elif time_scheme == "yoshida4":
            self.stepper = cast(
                Yoshida4Integrator,
                Yoshida4Integrator(kicker, grid.xmin, grid.xmax, grid.dt),
            )
        else:
            raise NotImplementedError(f"PIC time integrator '{time_scheme}' is not implemented")

        # We need an independent driver evaluator for diagnostic ``de`` output.
        self.diagnostic_driver = kicker.driver

        # Transverse EM components are constructed only when ``drivers.ey`` is
        # non-empty. The state schema remains stable in both branches.
        beta = cfg["grid"]["beta"]
        c = 1.0 / beta if beta > 0 else 1.0
        self.has_ey = len(drivers.ey) > 0
        self.wave_solver_active = self.has_ey
        if self.wave_solver_active:
            self.electron_charge_density = cast(
                ElectronChargeDensity,
                ElectronChargeDensity(
                    nx=grid.nx,
                    dx=grid.dx,
                    xmin=grid.xmin,
                    species_params=cfg["grid"]["species_params"],
                    shape=shape,
                ),
            )
            self.wave_solver = WaveSolver(c=c, dx=grid.dx, dt=grid.dt)
            self.ey_driver = TransverseCurrentSourceDriver(grid.x_a, drivers=drivers.ey, c=c)
        else:
            self.electron_charge_density = None
            self.wave_solver = None
            self.ey_driver = None

    def _pack(self, y: dict) -> dict:
        return {name: {"x": y[f"x_{name}"], "v": y[f"v_{name}"], "w": y[f"w_{name}"]} for name in self.species_names}

    def _unpack(self, particles: dict, e_sc: jnp.ndarray, de: jnp.ndarray, a, prev_a, da) -> dict:
        out = {"e": e_sc, "de": de, "a": a, "prev_a": prev_a, "da": da}
        for name, p in particles.items():
            out[f"x_{name}"] = p["x"]
            out[f"v_{name}"] = p["v"]
            out[f"w_{name}"] = p["w"]
        return out

    def __call__(self, t, y, args):
        particles = self._pack(y)
        a_n = y["a"]

        # 1. Charge density of electrons at the start of the step (needed for
        #    the wave equation's ``n_e a`` source after the particles move).
        ne_charge_n = self.electron_charge_density(particles) if self.electron_charge_density is not None else None

        # 2. Symplectic particle push under (E + ponderomotive(a^n)).
        a_for_push = a_n if self.has_ey else None
        particles, e_sc, _e_drv_end = self.stepper(particles, t, a_for_push, args["drivers"].ex)

        # 3. Report the driver field at the *new* step time so consecutive saves
        #    see consistent (E, dE) at the same temporal grid point.
        de = self.diagnostic_driver(t + self.grid.dt, args["drivers"].ex)

        # 4. Advance the wave equation using density averaged across n and n+1
        #    — matches Vlasov-1D's :class:`VlasovMaxwell.__call__`. Skip the
        #    wave solve entirely when no transverse driver is configured, so
        #    pure ES runs don't pay the (small) extra cost.
        if self.wave_solver_active:
            electron_charge_density = self.electron_charge_density
            ey_driver = self.ey_driver
            wave_solver = self.wave_solver
            assert electron_charge_density is not None
            assert ey_driver is not None
            assert wave_solver is not None
            assert ne_charge_n is not None
            ne_charge_np1 = electron_charge_density(particles)
            djy = ey_driver(t + self.grid.dt, args)
            a_new = wave_solver(
                a=a_n,
                aold=y["prev_a"],
                djy_array=djy,
                electron_density=-0.5 * (ne_charge_n + ne_charge_np1),
            )
            a_out, prev_a_out = a_new["a"], a_new["prev_a"]
        else:
            djy = jnp.zeros_like(a_n)
            a_out, prev_a_out = a_n, y["prev_a"]

        return self._unpack(particles, e_sc, de, a_out, prev_a_out, djy)
