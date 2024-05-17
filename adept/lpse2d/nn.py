import jax
import equinox as eqx
import jax.numpy as jnp
import json


class DriverModel(eqx.Module):
    encoder: eqx.Module
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(
        self, encoder_width, encoder_depth, decoder_width, decoder_depth, input_width, output_width, latent_width, key
    ):
        super().__init__()
        self.encoder = eqx.nn.MLP(
            input_width,
            latent_width,
            width_size=encoder_width,
            depth=encoder_depth,
            key=jax.random.PRNGKey(key),
            activation=jnp.tanh,
        )
        self.amp_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=jax.random.PRNGKey(key + 1)
        )
        self.phase_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=jax.random.PRNGKey(key + 2)
        )

    def __call__(self, x):
        encoded = self.encoder(x)
        amps = self.amp_decoder(encoded)
        phases = self.phase_decoder(encoded)
        return {"amps": amps, "phases": phases}


class DriverVAE(eqx.Module):
    encoder: eqx.Module
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module

    def __init__(
        self, encoder_width, encoder_depth, decoder_width, decoder_depth, input_width, output_width, latent_width, key
    ):
        super().__init__()
        self.encoder = eqx.nn.MLP(
            input_width,
            2 * latent_width,
            width_size=encoder_width,
            depth=encoder_depth,
            key=jax.random.PRNGKey(key),
            activation=jnp.tanh,
        )
        self.amp_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=jax.random.PRNGKey(key + 1)
        )
        self.phase_decoder = eqx.nn.MLP(
            latent_width, output_width, width_size=decoder_width, depth=decoder_depth, key=jax.random.PRNGKey(key + 2)
        )

    def __call__(self, x):
        encoded_mu_and_sigma = self.encoder(x)
        encoded_mu = encoded_mu_and_sigma[:, : encoded_mu_and_sigma.shape[-1] // 2]
        encoded_sigma = encoded_mu_and_sigma[:, encoded_mu_and_sigma.shape[-1] // 2 :]
        encoded_var = encoded_sigma**2.0
        encoded = encoded_mu + jnp.exp(0.5 * encoded_sigma) * jax.random.normal(jax.random.PRNGKey(0), encoded_mu.shape)

        amps = self.amp_decoder(encoded)
        phases = self.phase_decoder(encoded)
        kl_loss = 0.5 * jnp.sum(-jnp.log(encoded_var) - 1.0 + encoded_var + jnp.square(encoded_mu), axis=-1)

        return {"amps": amps, "phases": phases, "kl_loss": kl_loss}


def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())

        if hyperparams["model"]["type"] == "VAE":
            model = DriverVAE(**hyperparams)
        elif hyperparams["model"]["type"] == "MLP":
            model = DriverModel(**hyperparams)
        else:
            raise NotImplementedError(f"Model type {hyperparams['model']['type']} not implemented")
        return eqx.tree_deserialise_leaves(f, model), hyperparams
