import jax
import equinox as eqx
import jax.numpy as jnp
import json
import numpy as np


class GenerativeDriver(eqx.Module):
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(self, decoder_width, decoder_depth, input_width, output_width, key):
        super().__init__()
        da_k, dp_k = jax.random.split(jax.random.PRNGKey(key), 2)
        self.amp_decoder = eqx.nn.MLP(
            input_width, output_width, width_size=decoder_width, depth=decoder_depth, key=da_k, activation=jnp.tanh
        )
        self.phase_decoder = eqx.nn.MLP(
            input_width, output_width, width_size=decoder_width, depth=decoder_depth, key=dp_k, activation=jnp.tanh
        )

    def __call__(self, x):
        amps = self.amp_decoder(x)
        phases = self.phase_decoder(x)
        return {"amps": amps, "phases": phases}


class DriverModel(eqx.Module):
    encoder: eqx.Module
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(
        self, encoder_width, encoder_depth, decoder_width, decoder_depth, input_width, output_width, latent_width, key
    ):
        super().__init__()
        e_k, da_k, dp_k = jax.random.split(jax.random.PRNGKey(key), 3)
        self.encoder = eqx.nn.MLP(
            input_width, latent_width, width_size=encoder_width, depth=encoder_depth, key=e_k, activation=jnp.tanh
        )
        self.amp_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=da_k, activation=jnp.tanh
        )
        self.phase_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=dp_k, activation=jnp.tanh
        )

    def __call__(self, x):
        encoded = self.encoder(x)
        amps = self.amp_decoder(encoded)
        phases = self.phase_decoder(encoded)
        return {"amps": amps, "phases": phases}


class DriverVAE(eqx.Module):
    gen_k: jax.random.PRNGKey
    encoder: eqx.Module
    mu: eqx.Module
    sigma: eqx.Module
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(self, input_width, output_width, latent_width, key):
        super().__init__()
        e_k, mu_k, sigma_k, da_k, dp_k = jax.random.split(jax.random.PRNGKey(key), 5)
        self.gen_k = jax.random.PRNGKey(np.random.randint(0, 2**20))
        self.encoder = eqx.nn.Linear(input_width, latent_width, key=e_k)
        self.mu = eqx.nn.Linear(latent_width, latent_width, key=mu_k)
        self.sigma = eqx.nn.Linear(latent_width, latent_width, key=sigma_k)
        self.amp_decoder = eqx.nn.Linear(latent_width, output_width, key=da_k)
        self.phase_decoder = eqx.nn.Linear(latent_width, output_width, key=dp_k)

    def __call__(self, x):
        latent = jnp.tanh(self.encoder(x))
        encoded_mu = jnp.tanh(self.mu(latent))
        encoded_sigma = jnp.tanh(self.sigma(latent))
        encoded_var = encoded_sigma**2.0
        log_var = jnp.log(encoded_var)
        encoded = encoded_mu + encoded_sigma * jax.random.normal(self.gen_k, encoded_mu.shape)

        amps = self.amp_decoder(encoded)
        phases = self.phase_decoder(encoded)
        kl_loss = -0.5 * jnp.sum(1 + log_var - jnp.square(encoded_mu) - encoded_var)

        return {"amps": amps, "phases": phases, "kl_loss": kl_loss}


def save(filename, model_cfg, model):
    with open(filename, "wb") as f:
        model_cfg_str = json.dumps(model_cfg)
        f.write((model_cfg_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename):
    with open(filename, "rb") as f:
        model_cfg = json.loads(f.readline().decode())
        if "type" in model_cfg:
            hyperparams = model_cfg["hyperparams"]
            if model_cfg["type"] == "VAE":
                model = DriverVAE(**hyperparams)
            elif model_cfg["type"] == "MLP":
                model = DriverModel(**hyperparams)
            elif model_cfg["type"] == "GEN":
                model = GenerativeDriver(**hyperparams)
            else:
                raise NotImplementedError(f"Model type {model_cfg['type']} not implemented")
        else:
            model = DriverVAE(**model_cfg)
            print("Model type not specified, defaulting to VAE. dangerous (and often probably just doesn't load)")
            hyperparams = model_cfg

        return eqx.tree_deserialise_leaves(f, model), hyperparams
