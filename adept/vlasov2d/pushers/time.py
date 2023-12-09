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
        # elif cfg["solver"]["edfdv"] == "center_difference":
        #     edfdv = velocity.CD2(cfg)
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

    def __init__(self, cfg):
        super(ChargeConservingMaxwell, self).__init__(cfg)

    def step_1(self, ex, ey, bz, f):
        bznph = self.field_solve.faraday(ex, ey, bz, self.dt)
        jx = self.field_solve.compute_jx(f)
        jy = self.field_solve.compute_jy(f)
        exnph, eynph = self.field_solve.ampere(ex, ey, bz, jx, jy, 0.5 * self.dt)

        return bznph, exnph, eynph

    def step_2(self, f):
        fn1 = self.vdfdx.step_x(f, 0.5 * self.dt)
        return fn1, self.field_solve.compute_jx(fn1)

    def step_3(self, fn1):
        fn2 = self.vdfdx.step_y(fn1, 0.5 * self.dt)
        return fn2, self.field_solve.compute_jy(fn2)

    def step_4(self, exnph, eynph, bznph, fn2):
        return self.velocity_pusher(fk=fn2, ex=exnph, ey=eynph, bz=bznph, dt=self.dt)

    def step_5(self, fn3):
        fn4 = self.vdfdx.step_y(fn3, 0.5 * self.dt)
        return fn4, self.field_solve.compute_jy(fn4)

    def step_6(self, fn4):
        fn5 = self.vdfdx.step_x(fn4, 0.5 * self.dt)
        return fn5, self.field_solve.compute_jx(fn5)

    def step_7(self, ex, ey, bz, jxn12, jxn92, jyn32, jyn72):
        jxnph = 0.5 * (jxn12 + jxn92)
        jynph = 0.5 * (jyn32 + jyn72)

        return self.field_solve.ampere(ex, ey, jxnph, jynph, bz, 0.5 * self.dt)

    def __call__(self, t, y, args):
        ex, ey, bz, f = (
            y["ex"].view(dtype=jnp.complex128),
            y["ey"].view(dtype=jnp.complex128),
            y["bz"].view(dtype=jnp.complex128),
            y["electron"].view(dtype=jnp.complex128),
        )

        dex, dey = self.driver(t, args)

        bznph, exnph, eynph = self.step_1(ex, ey, bz, f)
        fn1, jxn12 = self.step_2(f)
        fn2, jyn32 = self.step_3(fn1)
        fn3 = self.step_4(exnph, eynph, bznph, fn2)
        fn4, jyn72 = self.step_5(fn3)
        fnp1, jxn92 = self.step_6(fn4)
        exnp1, eynp1 = self.step_7(exnph, eynph, bznph, jxn12, jxn92, jyn32, jyn72)

        return {
            "electron": fnp1.view(dtype=jnp.float64),
            "ex": exnp1.view(dtype=jnp.float64),
            "ey": eynp1.view(dtype=jnp.float64),
            "bz": bznph.view(dtype=jnp.float64),
            "dex": dex.view(dtype=jnp.float64),
            "dey": dey.view(dtype=jnp.float64),
        }
