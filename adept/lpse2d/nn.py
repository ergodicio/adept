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
        return amps, phases


def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = DriverModel(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model), hyperparams
