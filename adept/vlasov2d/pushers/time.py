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
        self.dt = cfg["derived"]["dt"]
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


def get_vp_timestep(cfg):
    if cfg["solver"]["dfdt"] == "leapfrog":
        return LeapfrogIntegrator(cfg)
    # elif cfg["solver"]["dfdt"] == "hamiltonian_sixth":
    #     return SixthOrderHamIntegrator(cfg)
    else:
        raise NotImplementedError


class LeapfrogIntegrator(VlasovPoissonBase):
    dt_array: jnp.ndarray

    def __init__(self, cfg):
        super(LeapfrogIntegrator, self).__init__(cfg)
        self.dt_array = self.dt * jnp.array([0.0, 1.0])

    def __call__(self, f, de_array):
        f = self.vdfdx(f=f, dt=0.5 * self.dt)
        force, e = self.field_solve(de_array=de_array[0], f=f)
        f = self.edfdv(f=f, e=force, dt=self.dt)
        f = self.vdfdx(f=f, dt=0.5 * self.dt)

        return f, {"total_ex": force[0], "total_ey": force[1], "ex": e[0], "ey": e[1]}
