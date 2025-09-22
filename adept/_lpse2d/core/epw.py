import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from adept._lpse2d.core.driver import Driver


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
        self.zero_mask = cfg["grid"]["zero_mask"]
        self.nx = cfg["grid"]["nx"]
        self.ny = cfg["grid"]["ny"]
        self.driver = Driver(cfg)
        self.tpd_const = 1j * self.e / (8 * self.wp0 * self.me)
        self.nu_coll = cfg["units"]["derived"]["nu_coll"]

        if cfg["terms"]["epw"]["source"]["srs"]:
            self.w1 = cfg["units"]["derived"]["w1"]
            max_source_k_multiplier = 1.2
            max_k0 = max_source_k_multiplier * np.sqrt(1 - cfg["density"]["min"])
            max_k1 = max_source_k_multiplier * np.sqrt(1 - cfg["density"]["min"] * (self.w0**2) / (self.w1**2))
            is_outside_max_k0 = self.k_sq * (1 / self.w0**2) > max_k0**2
            is_outside_max_k1 = self.k_sq * (1 / self.w1**2) > max_k1**2
            self.E0_filter = jnp.where(is_outside_max_k0, 0.0, 1.0)[..., None]
            self.E1_filter = jnp.where(is_outside_max_k1, 0.0, 1.0)[..., None]

            self.srs_const = self.e * self.wp0 / (4 * self.me * self.w0 * self.w1)

    def calc_fields_from_phi(self, phi: Array) -> tuple[Array, Array]:
        """
        Calculates ex(x, y) and ey(x, y) from phi.

        Args:
            phi (Array): phi(x, y)

        Returns:
            A Tuple containing ex(x, y) and ey(x, y)
        """

        phi_k = jnp.fft.fft2(phi)
        return self.calc_fields_from_phi_k(phi_k)

    def calc_fields_from_phi_k(self, phi_k: Array) -> tuple[Array, Array]:
        """
        Calculates ex(x, y) and ey(x, y) from phi_k.

        Args:
            phi (Array): phi(x, y)

        Returns:
            A Tuple containing ex(x, y) and ey(x, y)
        """

        phi_k = phi_k * self.low_pass_filter * self.zero_mask
        ex_k = -1j * self.kx[:, None] * phi_k
        ey_k = -1j * self.ky[None, :] * phi_k
        return jnp.fft.ifft2(ex_k), jnp.fft.ifft2(ey_k)

    def calc_phi_from_fields(self, ex: Array, ey: Array) -> Array:
        """
        calculates phi from ex and ey

        Args:
            ex (Array): ex(x, y)
            ey (Array): ey(x, y)

        Returns:
            Array: phi(x, y)

        """

        phi_k = self.calc_phi_k_from_fields(ex, ey)
        phi = jnp.fft.ifft2(phi_k * self.low_pass_filter)

        return phi

    def calc_phi_k_from_fields(self, ex: Array, ey: Array) -> Array:
        """
        calculates phi_k from ex and ey

        Args:
            ex (Array): ex(x, y)
            ey (Array): ey(x, y)

        Returns:
            Array: phi_k(x, y)
        """

        ex_k = jnp.fft.fft2(ex)
        ey_k = jnp.fft.fft2(ey)
        divE_k = 1j * (self.kx[:, None] * ex_k + self.ky[None, :] * ey_k) * self.low_pass_filter

        phi_k = divE_k * self.one_over_ksq
        return phi_k * self.zero_mask

    def tpd(self, t: float, phi_k: Array, ey: Array, args: dict) -> Array:
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
        tpd1 = E0[..., 1] * jnp.conj(ey)
        tpd1 = jnp.fft.fft2(tpd1)

        divE_true = jnp.fft.ifft2(self.k_sq * phi_k)
        E0_divE_k = jnp.fft.fft2(E0[..., 1] * jnp.conj(divE_true))
        tpd2 = 1j * self.ky[None, :] * self.one_over_ksq * E0_divE_k
        # tpd2 = jnp.fft.ifft2(tpd2 * self.low_pass_filter)

        total_tpd = self.tpd_const * jnp.exp(-1j * (self.w0 - 2 * self.wp0) * t) * (tpd1 + tpd2)

        total_tpd *= self.zero_mask * self.low_pass_filter

        return total_tpd

    def eval_E0_dot_E1(self, t, y, args):
        E0 = args["E0"]
        E1 = y["E1"]

        # filter E0 and E1
        E0_filtered = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.E0_filter, axes=(0, 1))
        E1_filtered = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.E1_filter, axes=(0, 1))
        E0_x_source, E0_y_source = E0_filtered[..., 0], E0_filtered[..., 1]
        E1_x_source, E1_y_source = E1_filtered[..., 0], E1_filtered[..., 1]

        return E0_x_source * jnp.conj(E1_x_source) + E0_y_source * jnp.conj(E1_y_source)

    def srs(self, t: float, y, args: dict) -> Array:
        E0_dot_E1 = self.eval_E0_dot_E1(t, y, args)
        return jnp.fft.fft2(1j * self.srs_const * y["background_density"] / self.envelope_density * E0_dot_E1)

    def get_noise(self):
        random_amps = 1.0e-16  # jax.random.uniform(self.amp_key, (self.nx, self.ny))
        random_phases = 2 * np.pi * jax.random.uniform(self.phase_key, (self.nx, self.ny))
        return random_amps * jnp.exp(1j * random_phases) * self.zero_mask

    def landau_damping(self, phi_k: Array, vte_sq: float):
        gammaLandauEpw = (
            jnp.sqrt(np.pi / 8)
            * (1.0 + 1.5 * self.k_sq * (vte_sq / self.wp0**2))
            * self.wp0**4
            * self.one_over_ksq**1.5
            / vte_sq**1.5
            * jnp.exp(-(1.5 + 0.5 * self.wp0**2 * self.one_over_ksq / vte_sq))
        ) * self.zero_mask

        return phi_k * jnp.exp(-(gammaLandauEpw + self.nu_coll) * self.dt) * self.zero_mask * self.low_pass_filter

    def __call__(self, t: float, y: dict[str, Array], args: dict) -> Array:
        phi_k = y["epw"]
        E0 = y["E0"]
        background_density = y["background_density"]
        vte_sq = y["vte_sq"]

        # linear propagation
        phi_k = phi_k * jnp.exp(-1j * 1.5 * vte_sq[0, 0] / self.wp0 * self.k_sq * self.dt) * self.low_pass_filter
        phi_k = self.landau_damping(phi_k, vte_sq[0, 0])

        if self.cfg["terms"]["epw"]["source"]["noise"]:
            phi_k += self.dt * self.get_noise()

        ex, ey = self.calc_fields_from_phi_k(phi_k)

        if self.cfg["terms"]["epw"]["source"]["tpd"]:
            tpd_term = self.tpd(t, phi_k, ey, args={"E0": E0})

        if self.cfg["terms"]["epw"]["source"]["srs"]:
            srs_term = self.srs(t, y, args={"E0": E0})

        # density gradient
        if self.cfg["terms"]["epw"]["density_gradient"]:
            background_density_perturbation = background_density / self.envelope_density - 1.0
            phase = jnp.exp(-1j * self.wp0 / 2.0 * background_density_perturbation * self.dt)
            ex = ex * phase
            ey = ey * phase

        # if boundary_damping:
        ex = ex * self.boundary_envelope
        ey = ey * self.boundary_envelope
        phi_k = self.calc_phi_k_from_fields(ex, ey)

        # tpd
        if self.cfg["terms"]["epw"]["source"]["tpd"]:
            phi_k += self.dt * tpd_term

        if self.cfg["terms"]["epw"]["source"]["srs"]:
            phi_k += self.dt * srs_term

        return phi_k
