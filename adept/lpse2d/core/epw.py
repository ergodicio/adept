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


class SpectralPotential_old(eqx.Module):
    wp0: float
    ld_rates: jax.Array
    n0: float
    w0: float
    nuei: float
    dx: float
    dy: float
    trapper: eqx.Module
    driver: eqx.Module
    kx: jax.Array
    ky: jax.Array
    kax_sq: jax.Array
    one_over_ksq: jax.Array
    dt: float
    cfg: Dict
    wr_corr: jax.Array
    num_substeps: int
    dt_substeps: float

    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg["grid"]["dt"]
        self.wp0 = cfg["plasma"]["wp0"]
        self.n0 = np.sqrt(self.wp0)
        self.w0 = cfg["drivers"]["E0"]["w0"]
        self.nuei = 0.0  # -cfg["units"]["derived"]["nuei_norm"]
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]
        self.dx = cfg["grid"]["dx"]
        self.dy = cfg["grid"]["dy"]
        self.kax_sq = self.kx[:, None] ** 2.0 + self.ky[None, :] ** 2.0
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]

        self.driver = Driver(cfg)
        self.trapper = ParticleTrapper(cfg)

        self.dt_substeps = 0.005
        self.num_substeps = int(self.dt / self.dt_substeps) + 4

        table_wrs, table_wis, table_klds = electrostatic.get_complex_frequency_table(
            1024, cfg["terms"]["epw"]["kinetic real part"]
        )
        all_ks = jnp.sqrt(self.kax_sq).flatten()
        self.ld_rates = jnp.interp(all_ks, table_klds, table_wis, left=0.0, right=table_wis[-1]).reshape(
            self.kax_sq.shape
        )
        wrs = jnp.interp(all_ks, table_klds, table_wrs, left=1.0, right=table_wrs[-1]).reshape(self.kax_sq.shape)
        if self.cfg["terms"]["epw"]["kinetic real part"]:
            self.wr_corr = (wrs - 1.0) * self.one_over_ksq / 1.5
        else:
            self.wr_corr = 1.0

    def get_eh_x(self, phi: jax.Array) -> jax.Array:
        ehx = -jnp.fft.ifft2(1j * self.kx[:, None] * phi)
        ehy = -jnp.fft.ifft2(1j * self.ky[None, :] * phi)

        return jnp.concatenate([ehx[..., None], ehy[..., None]], axis=-1) * self.kx.size * self.ky.size / 4

    def get_phi_from_eh(self, eh: jax.Array) -> jax.Array:
        ekx = jnp.fft.fft2(eh[..., 0])
        eky = jnp.fft.fft2(eh[..., 1])
        phi = (
            (1j * self.kx[:, None] * ekx + 1j * self.ky[None, :] * eky)
            * self.one_over_ksq
            * 4
            / self.kx.size
            / self.ky.size
        )
        return phi

    def calc_linear_step(self, temperature: jax.Array) -> Tuple[jax.Array, jax.Array]:
        damping_term = self.nuei / 2.0 + self.ld_rates
        osc_term = self.wr_corr * 1j * 1.5 * temperature * self.kax_sq / self.wp0
        return osc_term, damping_term

    def calc_density_pert_step(self, nb: jax.Array, dn: jax.Array) -> jax.Array:
        return -1j / 2.0 * self.wp0 * (1.0 - nb / self.n0 - dn / self.n0)

    def _calc_div_(self, arr):
        arrk = jnp.fft.fft2(arr, axes=[0, 1])
        divk = self.kx[:, None] * arrk[..., 0] + self.ky[None, :] * arrk[..., 1]
        return jnp.fft.ifft2(divk, axes=[0, 1])

    def calc_tpd_source_step(self, phi: jax.Array, e0: jax.Array, nb: jax.Array, t: float) -> jax.Array:
        """
        Returns i*e/(4 m_e w_0) *exp(i(2w_p0 - w_0)t) *
        (
        F(n_b / n_0 * E_h^* dot E_0 ) -
        (1 - w_0/w_{p0})*ik/k^2 dot (F(n_b / n_0) E_0 div E_h^*)
        )


        :param phi:
        :param e0:
        :param nb:
        :param t:
        :return:
        """
        eh = self.get_eh_x(phi)
        eh_star = jnp.conj(eh)
        div_ehstar = self._calc_div_(eh_star)
        coeff = -1j / 4.0 / self.w0 * jnp.exp(1j * (2.0 * self.wp0 - self.w0) * t)
        term1 = jnp.fft.fft2(nb / self.n0 * jnp.sum(eh_star * e0, axis=-1))  # 2d array

        term2_temp = jnp.fft.fft2(nb[..., None] / self.n0 * (e0 * div_ehstar[..., None]), axes=[0, 1])  # 3d array
        term2 = (
            (1.0 - self.wp0 / self.w0)
            * 1j
            * self.one_over_ksq
            * (self.kx[:, None] * term2_temp[..., 0] + self.ky[None, :] * term2_temp[..., 1])
        )
        return coeff * (term1 + term2)

    def update_potential(self, t, y):
        # do the equation first --- this is equation 54 so far
        if self.cfg["terms"]["epw"]["linear"]:
            osc_term, damping_term = self.calc_linear_step(y["temperature"])
            new_phi = y["phi"] * jnp.exp(osc_term * self.dt)
            new_phi = new_phi + self.dt * jnp.fft.fft2(jnp.fft.ifft2(damping_term * new_phi) / (1 + y["delta"] ** 2.0))

            if self.cfg["terms"]["epw"]["density_gradient"]:
                eh_x = self.get_eh_x(new_phi)
                density_step = self.calc_density_pert_step(y["nb"], y["dn"])[..., None]
                eh_x = eh_x * jnp.exp(density_step * self.dt)
                new_phi = self.get_phi_from_eh(eh_x)

            if self.cfg["terms"]["epw"]["source"]["tpd"]:
                new_phi += self.dt * self.calc_tpd_source_step(y["phi"], y["e0"], y["nb"], t)

        else:
            raise NotImplementedError("The linear term is necessary to run the code")

        return new_phi

    def update_delta(self, t, y, args):
        # return y["delta"] + self.dt * self.trapper(t, y["delta"], args)
        return diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.trapper),
            solver=diffrax.Tsit5(),
            t0=t,
            t1=t + self.dt,
            max_steps=self.num_substeps,
            dt0=self.dt_substeps,
            y0=y["delta"],
            args=args,
        ).ys[0]

    def __call__(self, t, y, args):
        # push the equation of motion for the potential
        y["phi"] = self.update_potential(t, y)

        if ("E2" in self.cfg["drivers"].keys()) or self.cfg["terms"]["epw"]["trapping"]["active"]:
            eh = self.get_eh_x(y["phi"])

        # then push the trapper if desired
        if self.cfg["terms"]["epw"]["trapping"]["active"]:
            y["delta"] = self.update_delta(t, y, args={"eh": eh, "nu_g": args["nu_g"]})

        # then push the driver if any
        if "E2" in self.cfg["drivers"].keys():
            eh += jnp.concatenate(
                [temp := self.driver(args["drivers"]["E2"], t)[..., None], jnp.zeros_like(temp)], axis=-1
            )
            y["phi"] = self.get_phi_from_eh(eh)

        if (
            self.cfg["terms"]["epw"]["boundary"]["x"] == "absorbing"
            or self.cfg["terms"]["epw"]["boundary"]["y"] == "absorbing"
        ):
            y["phi"] = self.get_phi_from_eh(
                self.get_eh_x(y["phi"]) * self.cfg["grid"]["absorbing_boundaries"][..., None]
            )

        return y


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
        self.low_pass_filter = np.where(np.sqrt(self.k_sq) < 2.0 / 3.0 * np.amax(self.kx), 1, 0)
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

    def calc_fields_from_phi(self, phi):
        """
        Calculates ex(x, y) and ey(x, y) from phi.

        checked

        Args:
            phi (jnp.array): phi(x, y)

        Returns:
            ex_ey (jnp.array): Vector field e(x, y, dir)
        """
        phi_k = jnp.fft.fft2(phi)
        phi_k *= self.low_pass_filter

        ex_k = self.kx[:, None] * phi_k * self.low_pass_filter
        ey_k = self.ky[None, :] * phi_k * self.low_pass_filter
        return -1j * jnp.fft.ifft2(ex_k), -1j * jnp.fft.ifft2(ey_k)

    def calc_phi_from_fields(self, ex, ey):
        """

        checked

        """
        ex_k = jnp.fft.fft2(ex)
        ey_k = jnp.fft.fft2(ey)
        divE_k = 1j * (self.kx[:, None] * ex_k + self.ky[None, :] * ey_k)

        phi_k = divE_k * self.one_over_ksq
        phi = jnp.fft.ifft2(phi_k * self.low_pass_filter)

        return phi

    def tpd(self, t, y, args):
        """
        checked

        """
        E0 = args["E0"]  # .view(jnp.complex128)
        phi = y  # .view(jnp.complex128)
        _, ey = self.calc_fields_from_phi(phi)

        tpd1 = E0[..., 1] * jnp.conj(ey)
        tpd1 = jnp.fft.ifft2(jnp.fft.fft2(tpd1) * self.low_pass_filter)
        # tpd1 = E0_Ey

        divE_true = jnp.fft.ifft2(self.k_sq * jnp.fft.fft2(phi))
        E0_divE_k = jnp.fft.fft2(E0[..., 1] * jnp.conj(divE_true))
        tpd2 = 1j * self.ky[None, :] * self.one_over_ksq * E0_divE_k
        tpd2 = jnp.fft.ifft2(tpd2 * self.low_pass_filter)

        total_tpd = self.tpd_const * jnp.exp(-1j * (self.w0 - 2 * self.wp0) * t) * (tpd1 + tpd2)

        dphi = total_tpd  # .view(jnp.float64)

        return dphi

    def calc_tpd1(self, t, y, args):
        E0 = args["E0"]
        phi = y

        _, ey = self.calc_fields_from_phi(phi)

        tpd1 = E0[..., 1] * jnp.conj(ey)
        return self.tpd_const * tpd1

    def calc_tpd2(self, t, y, args):
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

    def drive_epw(self, t) -> Array:
        return self.epw_driver(t)

    def __call__(self, t, y, args):
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
