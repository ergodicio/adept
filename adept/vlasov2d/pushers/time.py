from typing import Dict, Tuple

from jax import numpy as jnp
import diffrax

from adept.vlasov2d.pushers import vlasov, field


class Stepper(diffrax.Euler):
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful


class VlasovFieldBase:
    def __init__(self, cfg):
        self.dt = cfg["grid"]["dt"]
        self.vdfdx = self.get_vdfdx(cfg)
        self.velocity_pusher = self.get_edfdv(cfg)
        self.field_solve = field.FieldSolver(cfg=cfg)
        self.driver = field.Driver(cfg["grid"]["x"], cfg["grid"]["y"])

    def get_vdfdx(self, cfg):
        if cfg["solver"]["vdfdx"] == "exponential":
            vdfdx = vlasov.ExponentialSpatialAdvection(cfg)
        else:
            raise NotImplementedError("v df/dx: <" + cfg["solver"]["vdfdx"] + "> has not yet been implemented in JAX")

        return vdfdx

    def get_edfdv(self, cfg):
        if cfg["solver"]["edfdv"] == "exponential":
            edfdv = vlasov.ExponentialVelocityAdvection(cfg)
        elif cfg["solver"]["edfdv"] == "center_difference":
            edfdv = vlasov.CD2VelocityAdvection(cfg)
        # elif cfg["solver"]["edfdv"] == "weno":
        #     edfdv = velocity.WENO(cfg)
        # elif cfg["solver"]["edfdv"] == "semilagrangian":
        #     edfdv = velocity.SemiLagrangian(cfg)
        else:
            raise NotImplementedError("e df/dv: <" + cfg["solver"]["edfdv"] + "> has not yet been implemented in JAX")

        return edfdv


class ChargeConservingMaxwell(VlasovFieldBase):
    """
    This class contains the function that updates the state

    All the pushers are chosen and initialized here and a single time-step is defined here.

    1. Li, Y. et al. Solving the Vlasov–Maxwell equations using Hamiltonian splitting. Journal of Computational Physics 396, 381–399 (2019).



    :param cfg:
    :return:
    """

    def __init__(self, cfg: Dict):
        super(ChargeConservingMaxwell, self).__init__(cfg)
        self.push = cfg["solver"]["push_f"]
        self.dth = 0.5 * self.dt
        self.kx = cfg["grid"]["kx"]
        self.kvx = cfg["grid"]["kvx"][None, None, :, None]
        self.vx = cfg["grid"]["vx"][None, None, :, None]
        self.ky = cfg["grid"]["ky"]
        self.kvy = cfg["grid"]["kvy"][None, None, None, :]
        self.vy = cfg["grid"]["vy"][None, None, None, :]

    def step_vxB_1(self, bz, f, dt):
        fxykvx = jnp.fft.fft(f, axis=2)
        new_fxykvx = fxykvx * jnp.exp(-1j * dt * self.kvx * self.vy * bz[..., None, None])
        return jnp.fft.ifft(new_fxykvx, axis=2)

    def step_vxB_2(self, bz, f, dt):
        fxykvy = jnp.fft.fft(f, axis=3)
        new_fxykvy = fxykvy * jnp.exp(1j * dt * self.kvy * self.vx * bz[..., None, None])
        return jnp.fft.ifft(new_fxykvy, axis=3)

    def __call__(self, t: float, y: Dict, args: Dict) -> Dict:
        ex, ey, bz, f = y["ex"], y["ey"], y["bz"], y["electron"]

        dex, dey = self.driver(t, args)

        # H_E
        # update e df/dv
        fhe_xy = self.velocity_pusher.edfdv(fxy=f, ex=ex + dex, ey=ey + dey, dt=self.dt)

        exk, eyk, bzk = jnp.fft.fft2(ex), jnp.fft.fft2(ey), jnp.fft.fft2(bz)

        # update b
        bzkp = self.field_solve.faraday(bzk=bzk, exk=exk, eyk=eyk, dt=self.dt)
        bzn = jnp.real(jnp.fft.ifft2(bzkp))

        # H_1f
        # update vxB df/dv
        fb1_xy = self.step_vxB_1(bzn, fhe_xy, dt=self.dt)

        # update v1 df/dx1
        f1_xy = self.vdfdx.step_x(fb1_xy, dt=self.dt)
        # update e1
        e1p = self.field_solve.hampere_e1(exk=exk, fxy=fb1_xy, dt=self.dt)

        # H_2f
        # update vxB df/dv
        fb2_xy = self.step_vxB_2(bzn, f1_xy, dt=self.dt)

        # update v2 df/dx2
        f2_xy = self.vdfdx.step_y(fb2_xy, dt=self.dt)
        # update e2
        e2p = self.field_solve.hampere_e2(eyk=eyk, fxy=fb2_xy, dt=self.dt)

        # H_B
        # update E -> dE/dt = curl B
        exkp, eykp = self.field_solve.ampere(exk=e1p, eyk=e2p, bzk=bzkp, dt=self.dt)

        e1n, e2n = jnp.real(jnp.fft.ifft2(exkp)), jnp.real(jnp.fft.ifft2(eykp))

        return {"electron": f2_xy, "ex": e1n, "ey": e2n, "bz": bzn, "dex": dex, "dey": dey}
