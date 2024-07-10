from typing import Dict

import numpy as np
from jax import random as jr, numpy as jnp, Array
import equinox as eqx


class GenerativeDriver(eqx.Module):
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(self, decoder_width: int, decoder_depth: int, input_width: int, output_width: int, key: int):
        super().__init__()
        da_k, dp_k = jr.split(jr.PRNGKey(key), 2)
        self.amp_decoder = eqx.nn.MLP(
            input_width, output_width, width_size=decoder_width, depth=decoder_depth, key=da_k, activation=jnp.tanh
        )
        self.phase_decoder = eqx.nn.MLP(
            input_width, output_width, width_size=decoder_width, depth=decoder_depth, key=dp_k, activation=jnp.tanh
        )

    def __call__(self, x: Array) -> Dict:
        amps = self.amp_decoder(x)
        phases = self.phase_decoder(x)
        return {"amps": amps, "phases": phases}


class DriverModel(eqx.Module):
    encoder: eqx.Module
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(
        self,
        encoder_width: int,
        encoder_depth: int,
        decoder_width: int,
        decoder_depth: int,
        input_width: int,
        output_width: int,
        latent_width: int,
        key: int,
    ):
        super().__init__()
        e_k, da_k, dp_k = jr.split(jr.PRNGKey(key), 3)
        self.encoder = eqx.nn.MLP(
            input_width, latent_width, width_size=encoder_width, depth=encoder_depth, key=e_k, activation=jnp.tanh
        )
        self.amp_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=da_k, activation=jnp.tanh
        )
        self.phase_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=dp_k, activation=jnp.tanh
        )

    def __call__(self, x: Array) -> Dict:
        encoded = self.encoder(x)
        amps = self.amp_decoder(encoded)
        phases = self.phase_decoder(encoded)
        return {"amps": amps, "phases": phases}


class DriverVAE(eqx.Module):
    gen_k: jr.PRNGKey
    encoder: eqx.Module
    mu: eqx.Module
    sigma: eqx.Module
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(self, input_width: int, output_width: int, latent_width: int, key: int):
        super().__init__()
        e_k, mu_k, sigma_k, da_k, dp_k = jr.split(jr.PRNGKey(key), 5)
        self.gen_k = jr.PRNGKey(np.random.randint(0, 2**20))
        self.encoder = eqx.nn.Linear(input_width, latent_width, key=e_k)
        self.mu = eqx.nn.Linear(latent_width, latent_width, key=mu_k)
        self.sigma = eqx.nn.Linear(latent_width, latent_width, key=sigma_k)
        self.amp_decoder = eqx.nn.Linear(latent_width, output_width, key=da_k)
        self.phase_decoder = eqx.nn.Linear(latent_width, output_width, key=dp_k)

    def __call__(self, x: Array) -> Dict:
        latent = jnp.tanh(self.encoder(x))
        encoded_mu = jnp.tanh(self.mu(latent))
        encoded_sigma = jnp.tanh(self.sigma(latent))
        encoded_var = encoded_sigma**2.0
        log_var = jnp.log(encoded_var)
        encoded = encoded_mu + encoded_sigma * jr.normal(self.gen_k, encoded_mu.shape)

        amps = self.amp_decoder(encoded)
        phases = self.phase_decoder(encoded)
        kl_loss = -0.5 * jnp.sum(1 + log_var - jnp.square(encoded_mu) - encoded_var)

        return {"amps": amps, "phases": phases, "kl_loss": kl_loss}
