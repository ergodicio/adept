from typing import Dict
import equinox as eqx
from jax import numpy as jnp

from adept.vlasov2d.pushers import space, velocity, field


class VlasovPoissonBase(eqx.Module):
    dt: float
    vdfdx: eqx.Module
    edfdv: eqx.Module
    field_solve: eqx.Module

    def __init__(self, cfg):
        super(VlasovPoissonBase, self).__init__()
        self.dt = cfg["grid"]["dt"]
        # self.vx = cfg["derived"]["vx"]
        # self.vy = cfg["derived"]["vy"]
        self.vdfdx = self.get_vdfdx(cfg)
        self.edfdv = self.get_edfdv(cfg)
        self.field_solve = field.ElectricFieldSolver(cfg=cfg)

    def get_vdfdx(self, cfg):
        if cfg["solver"]["vdfdx"] == "exponential":
            vdfdx = space.Exponential(cfg)
        # elif cfg["solver"]["vdfdx"] == "semilagrangian":
        #     vdfdx = space.SemiLagrangian(cfg)
        else:
            raise NotImplementedError("v df/dx: <" + cfg["solver"]["vdfdx"] + "> has not yet been implemented in JAX")

        return vdfdx

    def get_edfdv(self, cfg):
        if cfg["solver"]["edfdv"] == "exponential":
            edfdv = velocity.Exponential(cfg)
        # elif cfg["solver"]["edfdv"] == "center_difference":
        #     edfdv = velocity.CD2(cfg)
        # elif cfg["solver"]["edfdv"] == "weno":
        #     edfdv = velocity.WENO(cfg)
        # elif cfg["solver"]["edfdv"] == "semilagrangian":
        #     edfdv = velocity.SemiLagrangian(cfg)
        else:
            raise NotImplementedError("e df/dv: <" + cfg["solver"]["edfdv"] + "> has not yet been implemented in JAX")

        return edfdv


class LeapfrogIntegrator(VlasovPoissonBase):
    b: jnp.ndarray
    driver: eqx.Module

    def __init__(self, cfg):
        super(LeapfrogIntegrator, self).__init__(cfg)
        self.b = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"], 2))
        self.driver = field.Driver(cfg["grid"]["x"], cfg["grid"]["y"])

    def __call__(self, t, y, args):
        new_state = {}
        e = y["e"]
        for species in ["electron"]:
            f = self.vdfdx(f=y["electron"], dt=0.5 * self.dt)
            de = self.driver(t, args)
            e = self.field_solve(prev_force=e, f=f, dt=0.5 * self.dt)
            f = self.edfdv(f=f, e=e + de, dt=self.dt)
            f = self.vdfdx(f=f, dt=0.5 * self.dt)
            e = self.field_solve(prev_force=e, f=f, dt=0.5 * self.dt)

            new_state[species] = f

        for nm, fld in zip(["e", "b", "de"], [e, self.b, de]):
            new_state[nm] = fld

        return new_state
