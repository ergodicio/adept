from typing import Dict, Tuple
from functools import partial

import diffrax
import jax
from jax import numpy as jnp, Array
import equinox as eqx
import numpy as np
from adept.theory import electrostatic
from adept.lpse2d.core.driver import Driver
from adept.lpse2d.core.trapper import ParticleTrapper


class SpectralPotential:
    def __init__(self, cfg) -> None:

        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]
        self.k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2
        self.wp0 = cfg["units"]["derived"]["wp0"]
        self.e = cfg["units"]["derived"]["e"]
        self.me = cfg["units"]["derived"]["me"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.envelope_density = cfg["units"]["envelope density"]
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]
        self.boundary_envelope = cfg["grid"]["absorbing_boundaries"]
        self.dt = cfg["grid"]["dt"]
        self.cfg = cfg
        self.amp_key, self.phase_key = jax.random.split(jax.random.PRNGKey(np.random.randint(2**20)), 2)
        self.low_pass_filter = cfg["grid"]["low_pass_filter"]
        zero_mask = cfg["grid"]["zero_mask"]
        self.low_pass_filter = self.low_pass_filter * zero_mask
        self.nx = cfg["grid"]["nx"]
        self.ny = cfg["grid"]["ny"]
        self.driver = Driver(cfg)
        # self.step_tpd = partial(
        #     diffrax.diffeqsolve,
        #     terms=diffrax.ODETerm(self.tpd),
        #     solver=diffrax.Tsit5(),
        #     t0=0.0,
        #     t1=self.dt,
        #     dt0=self.dt,
        # )
        self.tpd_const = 1j * self.e / (8 * self.wp0 * self.me)

    def calc_fields_from_phi(self, phi: Array) -> Tuple[Array, Array]:
        """
        Calculates ex(x, y) and ey(x, y) from phi.

        Args:
            phi (Array): phi(x, y)

        Returns:
            A Tuple containing ex(x, y) and ey(x, y)
        """

        phi_k = jnp.fft.fft2(phi)
        phi_k *= self.low_pass_filter

        ex_k = self.kx[:, None] * phi_k * self.low_pass_filter
        ey_k = self.ky[None, :] * phi_k * self.low_pass_filter
        return -1j * jnp.fft.ifft2(ex_k), -1j * jnp.fft.ifft2(ey_k)

    def calc_phi_from_fields(self, ex: Array, ey: Array) -> Array:
        """
        calculates phi from ex and ey

        Args:
            ex (Array): ex(x, y)
            ey (Array): ey(x, y)

        Returns:
            Array: phi(x, y)

        """

        ex_k = jnp.fft.fft2(ex)
        ey_k = jnp.fft.fft2(ey)
        divE_k = 1j * (self.kx[:, None] * ex_k + self.ky[None, :] * ey_k)

        phi_k = divE_k * self.one_over_ksq
        phi = jnp.fft.ifft2(phi_k * self.low_pass_filter)

        return phi

    def tpd(self, t: float, y: Array, args: Dict) -> Array:
        """
        Calculates the two plasmon decay term

        Args:
            t (float): time
            y (Array): phi(x, y)
            args (Dict): dictionary containing E0

        Returns:
            Array: dphi(x, y)

        """
        E0 = args["E0"]
        phi = y
        _, ey = self.calc_fields_from_phi(phi)

        tpd1 = E0[..., 1] * jnp.conj(ey)
        tpd1 = jnp.fft.ifft2(jnp.fft.fft2(tpd1) * self.low_pass_filter)
        # tpd1 = E0_Ey

        divE_true = jnp.fft.ifft2(self.k_sq * jnp.fft.fft2(phi))
        E0_divE_k = jnp.fft.fft2(E0[..., 1] * jnp.conj(divE_true))
        tpd2 = 1j * self.ky[None, :] * self.one_over_ksq * E0_divE_k
        tpd2 = jnp.fft.ifft2(tpd2 * self.low_pass_filter)

        total_tpd = self.tpd_const * jnp.exp(-1j * (self.w0 - 2 * self.wp0) * t) * (tpd1 + tpd2)

        dphi = total_tpd

        return dphi

    def calc_tpd1(self, t: float, y: Array, args: Dict) -> Array:
        """
        Calculates the first term of the two plasmon decay

        Args:
            t (float): time
            y (Array): phi(x, y)
            args (Dict): dictionary containing E0

        Returns:
            Array: dphi(x, y)
        """
        E0 = args["E0"]
        phi = y

        _, ey = self.calc_fields_from_phi(phi)

        tpd1 = E0[..., 1] * jnp.conj(ey)
        return self.tpd_const * tpd1

    def calc_tpd2(self, t: float, y: Array, args: Dict) -> Array:
        """
        Calculates the second term of the two plasmon decay

        Args:
            t (float): time
            y (Array): phi(x, y)
            args (Dict): dictionary containing E0

        Returns:
            Array: dphi(x, y)

        """
        phi = y
        E0 = args["E0"]

        divE_true = jnp.fft.ifft2(self.k_sq * jnp.fft.fft2(phi))
        E0_divE_k = jnp.fft.fft2(E0[..., 1] * jnp.conj(divE_true))

        tpd2 = 1j * self.ky[None, :] * self.one_over_ksq * E0_divE_k
        tpd2 = jnp.fft.ifft2(tpd2)
        return self.tpd_const * tpd2

    def get_noise(self):
        random_amps = 1000.0  # jax.random.uniform(self.amp_key, (self.nx, self.ny))
        random_phases = 2 * np.pi * jax.random.uniform(self.phase_key, (self.nx, self.ny))
        return jnp.fft.ifft2(random_amps * jnp.exp(1j * random_phases) * self.low_pass_filter)

    def __call__(self, t: float, y: Dict[str, Array], args: Dict) -> Array:
        phi = y["epw"]
        E0 = y["E0"]
        background_density = y["background_density"]
        vte_sq = y["vte_sq"]

        if self.cfg["terms"]["epw"]["linear"]:
            # linear propagation
            phi = jnp.fft.ifft2(jnp.fft.fft2(phi) * jnp.exp(-1j * 1.5 * vte_sq[0, 0] / self.wp0 * self.k_sq * self.dt))

        # tpd
        if self.cfg["terms"]["epw"]["source"]["tpd"]:
            phi = phi + self.dt * self.tpd(t, phi, args={"E0": E0})

        # density gradient
        if self.cfg["terms"]["epw"]["density_gradient"]:
            ex, ey = self.calc_fields_from_phi(phi)
            ex *= jnp.exp(-1j * self.wp0 / 2.0 * (1 - background_density / self.envelope_density) * self.dt)
            ey *= jnp.exp(-1j * self.wp0 / 2.0 * (1 - background_density / self.envelope_density) * self.dt)
            phi = self.calc_phi_from_fields(ex, ey)

        if self.cfg["terms"]["epw"]["source"]["noise"]:
            phi += self.dt * self.get_noise()

        return phi
