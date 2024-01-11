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


#
# class LeapfrogIntegrator(VlasovFieldBase):
#     def __init__(self, cfg):
#         super(LeapfrogIntegrator, self).__init__(cfg)
#         self.bz = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]))
#
#     def __call__(self, t, y, args):
#
#         f = y["electron"].view(dtype=jnp.complex128)
#         bz = y["bz"].view(dtype=jnp.complex128)
#
#
#
#         new_state = {}
#         for species in ["electron"]:
#             f = self.vdfdx(f=f, dt=0.5 * self.dt)
#             dex, dey = self.driver(t, args)
#             ex, ey = self.field_solve.poisson(f=f)
#             f = self.velocity_pusher(fk=f, ex=ex + dex, ey=ey + dey, bz=self.bz, dt=self.dt)
#             f = self.vdfdx(f=f, dt=0.5 * self.dt)
#             ex, ey = self.field_solve.poisson(f=f)
#
#             new_state[species] = f
#
#         for nm, fld in zip(["ex", "ey", "bz", "dex", "dey", "dbz"], [ex, ey, self.bz, dex, dey]):
#             new_state[nm] = fld
#
#         return new_state
#


class ChargeConservingMaxwell(VlasovFieldBase):
    """
    This class contains the function that updates the state

    All the pushers are chosen and initialized here and a single time-step is defined here.

    1. Crouseilles, N., Navaro, P. & Sonnendrücker, É. Charge-conserving grid based methods for the Vlasov–Maxwell equations. Comptes Rendus Mécanique 342, 636–646 (2014).


    :param cfg:
    :return:
    """

    def __init__(self, cfg: Dict):
        super(ChargeConservingMaxwell, self).__init__(cfg)
        self.push = cfg["solver"]["push_f"]
        self.dth = 0.5 * self.dt
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]

    # def step_ampere_faraday(
    #     self, ex: jnp.ndarray, ey: jnp.ndarray, bz: jnp.ndarray, f: jnp.ndarray
    # ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    #     """
    #     Performs a Faraday full timestep and Ampere half timestep
    #     This is the first split component of the Vlasov-Maxwell solve
    #
    #     :param ex:
    #     :param ey:
    #     :param bz:
    #     :param f:
    #     :return:
    #     """
    #     bznph = self.field_solve.faraday(exk=ex, eyk=ey, bzk=bz, dt=self.dt)
    #     jx = self.field_solve.compute_jx(f=f)
    #     jy = self.field_solve.compute_jy(f=f)
    #     exnph, eynph = self.field_solve.ampere(exk=ex, eyk=ey, bzk=bz, jxk=jx, jyk=jy, dt=0.5 * self.dt)
    #
    #     return bznph, exnph, eynph
    #
    # def step_v(self, exnph: jnp.ndarray, eynph: jnp.ndarray, bznph: jnp.ndarray, fn2: jnp.ndarray) -> jnp.ndarray:
    #     if self.push:
    #         return self.velocity_pusher(fk=fn2, ex=exnph, ey=eynph, bz=bznph, dt=self.dt)
    #     else:
    #         return fn2
    #
    # def step_y(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     if self.push:
    #         fh = self.vdfdx.fyh(f=f, dt=self.dth)
    #         f1 = f - 1j * self.ky[None, :, None, None] * self.dth * fh
    #     else:
    #         fh = f
    #         f1 = f
    #     return f1, self.field_solve.compute_jy(f=fh)
    #
    # def step_x(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     if self.push:
    #         fh = self.vdfdx.fxh(f=f, dt=self.dth)
    #         f1 = f - 1j * self.kx[:, None, None, None] * self.dth * fh
    #     else:
    #         fh = f
    #         f1 = f
    #     return f1, self.field_solve.compute_jx(fh)
    #
    # def step_ampere(
    #     self,
    #     ex: jnp.ndarray,
    #     ey: jnp.ndarray,
    #     bz: jnp.ndarray,
    #     jxn12: jnp.ndarray,
    #     jxn92: jnp.ndarray,
    #     jyn32: jnp.ndarray,
    #     jyn72: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     jxnph = 0.5 * (jxn12 + jxn92)
    #     jynph = 0.5 * (jyn32 + jyn72)
    #
    #     return self.field_solve.ampere(exk=ex, eyk=ey, jxk=jxnph, jyk=jynph, bzk=bz, dt=0.5 * self.dt)
    #
    # # def __call__(self, t: float, y: Dict, args: Dict) -> Dict:
    # #     ex, ey, bz, f = (
    # #         y["ex"].view(dtype=jnp.complex128),
    # #         y["ey"].view(dtype=jnp.complex128),
    # #         y["bz"].view(dtype=jnp.complex128),
    # #         y["electron"].view(dtype=jnp.complex128),
    # #     )
    # #
    # #     dexk, deyk = self.driver(t, args)
    # #
    # #     bznph, exnph, eynph = self.step_ampere_faraday(ex, ey, bz, f)
    # #     fn1, jxn12 = self.step_x(f)
    # #     fn2, jyn32 = self.step_y(fn1)
    # #     fn3 = self.step_v(exnph + dexk, eynph + deyk, bznph, fn2)
    # #     fn4, jyn72 = self.step_y(fn3)
    # #     fnp1, jxn92 = self.step_x(fn4)
    # #     exnp1, eynp1 = self.step_ampere(exnph, eynph, bznph, jxn12, jxn92, jyn32, jyn72)
    # #
    # #     return {
    # #         "electron": fnp1.view(dtype=jnp.float64),
    # #         "ex": exnp1.view(dtype=jnp.float64),
    # #         "ey": eynp1.view(dtype=jnp.float64),
    # #         "bz": bznph.view(dtype=jnp.float64),
    # #         "dex": dexk.view(dtype=jnp.float64),
    # #         "dey": deyk.view(dtype=jnp.float64),
    # #     }

    def step_vxB(self, bz, f):
        return f

    def __call__(self, t: float, y: Dict, args: Dict) -> Dict:
        ex, ey, bz, f = y["ex"], y["ey"], y["bz"], y["electron"]

        dex, dey = self.driver(t, args)

        # ex += dex
        # ey += dey

        # H_E
        # update e df/dv
        fhe_xy = self.velocity_pusher.edfdv(fxy=f, ex=ex + dex, ey=ey + dey, dt=self.dt)

        exk, eyk, bzk = jnp.fft.fft2(ex), jnp.fft.fft2(ey), jnp.fft.fft2(bz)

        # update b
        bzkp = self.field_solve.faraday(bzk=bzk, exk=exk, eyk=eyk, dt=self.dt)

        # H_1f
        # update vxB df/dv
        fb1_xy = self.step_vxB(bzkp, fhe_xy)

        # update v1 df/dx1
        f1_xy = self.vdfdx.step_x(fb1_xy, dt=self.dt)
        # update e1
        e1p = self.field_solve.hampere_e1(exk=exk, fxy=fb1_xy, dt=self.dt)

        # H_2f
        # update vxB df/dv
        fb2_xy = self.step_vxB(bzkp, f1_xy)

        # update v2 df/dx2
        f2_xy = self.vdfdx.step_y(fb2_xy, dt=self.dt)
        # update e2
        e2p = self.field_solve.hampere_e2(eyk=eyk, fxy=fb2_xy, dt=self.dt)

        e1n, e2n, bn = jnp.real(jnp.fft.ifft2(e1p)), jnp.real(jnp.fft.ifft2(e2p)), jnp.real(jnp.fft.ifft2(bzkp))

        return {"electron": f2_xy, "ex": e1n, "ey": e2n, "bz": bn, "dex": dex, "dey": dey}
