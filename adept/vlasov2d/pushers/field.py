#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from jax import numpy as jnp
import equinox as eqx


class SpectralPoissonSolver(eqx.Module):
    ion_charge: jnp.array
    one_over_kx: jnp.array
    one_over_ky: jnp.array
    dvx: float
    dvy: float

    def __init__(self, ion_charge, one_over_kx, one_over_ky, dvx, dvy):
        super(SpectralPoissonSolver, self).__init__()
        self.ion_charge = jnp.array(ion_charge)
        self.one_over_kx = jnp.array(one_over_kx)
        self.one_over_ky = jnp.array(one_over_ky)
        self.dvx = dvx
        self.dvy = dvy

    def compute_charges(self, f):
        return jnp.trapz(jnp.trapz(f, dx=self.dvy, axis=3), dx=self.dvx, axis=2)

    def __call__(self, f: jnp.ndarray):
        return jnp.real(
            jnp.fft.ifft(
                1j * self.one_over_kx[:, None] * jnp.fft.fft(self.ion_charge - self.compute_charges(f), axis=0),
                axis=0,
            )
        ), jnp.real(
            jnp.fft.ifft(
                1j * self.one_over_ky[None, :] * jnp.fft.fft(self.ion_charge - self.compute_charges(f), axis=1),
                axis=1,
            )
        )


# class AmpereSolver(eqx.Module):
#     def __init__(self, cfg):
#         super(AmpereSolver, self).__init__()
#         self.vx = cfg["derived"]["v"]
#         self.vx_moment = partial(jnp.trapz, dx=cfg["derived"]["dv"], axis=1)
#
#     def __call__(self, f: jnp.ndarray, prev_force: jnp.ndarray, dt: jnp.float64):
#         return prev_force - dt * self.vx_moment(self.vx[None, :] * f)
#


class ElectricFieldSolver(eqx.Module):
    es_field_solver: eqx.Module

    def __init__(self, cfg):
        super(ElectricFieldSolver, self).__init__()

        if cfg["solver"]["field"] == "poisson":
            self.es_field_solver = SpectralPoissonSolver(
                ion_charge=cfg["derived"]["iprof"],
                one_over_kx=cfg["derived"]["one_over_kx"],
                one_over_ky=cfg["derived"]["one_over_ky"],
                dvx=cfg["derived"]["dvx"],
                dvy=cfg["derived"]["dvy"],
            )
        # elif cfg["solver"]["field"] == "ampere":
        #     if cfg["solver"]["dfdt"] == "leapfrog":
        #         self.es_field_solver = AmpereSolver(cfg)
        #     else:
        #         raise NotImplementedError(f"ampere + {cfg['solver']['dfdt']} has not yet been implemented")
        else:
            raise NotImplementedError("Field Solver: <" + cfg["solver"]["field"] + "> has not yet been implemented")
        # self.dx = cfg["derived"]["dx"]

    def __call__(self, de_array: jnp.ndarray, f: jnp.ndarray):
        """
        This returns the total electrostatic field that is used in the Vlasov equation
        The total field is a sum of the driver field and the
        self-consistent electrostatic field from a Poisson or Ampere solve

        :param de_array: an electric field
        :param f: distribution function
        :param a:
        :return:
        """
        # ponderomotive_force = -0.5 * jnp.gradient(jnp.square(a), self.dx)[1:-1]
        self_consistent_ex, self_consistent_ey = self.es_field_solver(f)
        return (de_array + self_consistent_ex, self_consistent_ey), (self_consistent_ex, self_consistent_ey)
