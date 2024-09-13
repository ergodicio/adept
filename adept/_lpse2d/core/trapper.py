import numpy as np
from jax import numpy as jnp, Array
import equinox as eqx
from adept import electrostatic


class ParticleTrapper(eqx.Module):
    kx: np.ndarray
    kax_sq: Array
    model_kld: float
    wis: Array
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
