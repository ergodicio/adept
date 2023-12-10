from typing import Dict, Tuple

import diffrax
import jax
from jax import numpy as jnp
import equinox as eqx
import numpy as np
from theory import electrostatic
from adept.lpse2d.core.driver import Driver


class ParticleTrapper(eqx.Module):
    kx: np.ndarray
    kax_sq: jax.Array

    model_kld: float
    wis: jax.Array
    norm_kld: jnp.float64
    norm_nuee: jnp.float64
    vph: jnp.float64
    fft_norm: float
    dx: float

    def __init__(self, cfg, species="electron"):
        self.kx = cfg["grid"]["kx"]
        self.dx = cfg["grid"]["dx"]
        self.kax_sq = cfg["grid"]["kx"][:, None] ** 2 + cfg["grid"]["ky"][None, :] ** 2
        table_wrs, table_wis, table_klds = electrostatic.get_complex_frequency_table(
            1024, cfg["terms"]["epw"]["kinetic real part"]
        )
        all_ks = jnp.sqrt(self.kax_sq).flatten()
        self.model_kld = cfg["terms"]["epw"]["trapping"]["kld"]
        self.wis = jnp.interp(all_ks, table_klds, table_wis, left=0.0, right=0.0).reshape(self.kax_sq.shape)
        self.norm_kld = (self.model_kld - 0.26) / 0.14
        self.norm_nuee = (jnp.log10(1.0e-7) + 7.0) / -4.0

        this_wr = jnp.interp(self.model_kld, table_klds, table_wrs, left=1.0, right=table_wrs[-1])
        self.vph = this_wr / self.model_kld
        self.fft_norm = cfg["grid"]["nx"] * cfg["grid"]["ny"] / 4.0
        # Make models
        # if models is not None:
        #     self.nu_g_model = models["nu_g"]
        # else:
        #     self.nu_g_model = lambda x: -32

    def __call__(self, t, delta, args):
        e = args["eh"]
        ek = jnp.fft.fft2(e[..., 0]) / self.fft_norm

        # this is where a specific k is chosen for the growth rate and where the identity of this delta object is given
        chosen_ek = jnp.interp(self.model_kld, self.kx, jnp.mean(jnp.abs(ek), axis=1))
        norm_e = (jnp.log10(chosen_ek + 1e-10) + 10.0) / -10.0
        func_inputs = jnp.stack([norm_e, self.norm_kld, self.norm_nuee], axis=-1)
        growth_rates = 10 ** jnp.squeeze(3 * args["nu_g"](func_inputs))

        return -self.vph * jnp.gradient(jnp.pad(delta, pad_width=1, mode="wrap"), axis=0)[
            1:-1, 1:-1
        ] / self.dx + growth_rates * jnp.abs(jnp.fft.ifft2(ek * self.fft_norm * self.wis)) / (1.0 + delta**2.0)


class EPW2D(eqx.Module):
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
        self.nuei = cfg["plasma"]["nu_ei"]
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
        arrk = jnp.fft.fft2(arr)
        divk = self.kx[:, None] * arrk[..., 0] + self.ky[None, :] * arrk[..., 1]
        return jnp.fft.ifft2(divk)

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

    def update_potential(self, y):
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
                new_phi += self.dt * self.calc_tpd_source_step(y["phi"], y["e0"], y["nb"], y["t"])

        else:
            raise NotImplementedError("The linear term is necessary to run the code")

        return new_phi

    def update_delta(self, t, y, args):
        # return y["delta"] + self.dt * self.trapper(t, y["delta"], args)
        return diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.trapper),
            solver=diffrax.Dopri8(),
            t0=t,
            t1=t + self.dt,
            max_steps=self.num_substeps,
            dt0=self.dt_substeps,
            y0=y["delta"],
            args=args,
        ).ys[0]

    def __call__(self, t, y, args):
        # push the equation of motion for the potential
        y["phi"] = self.update_potential(y)

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
