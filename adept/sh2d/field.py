import jax
from jax import numpy as jnp
import equinox as eqx


class SpectralPoissonSolver(eqx.Module):
    ion_charge: jnp.array
    one_over_kx: jnp.array
    one_over_ky: jnp.array
    dv: float
    v: jax.Array

    def __init__(self, ion_charge, one_over_kx, one_over_ky, dv, v):
        super(SpectralPoissonSolver, self).__init__()
        self.ion_charge = jnp.array(ion_charge)
        self.one_over_kx = jnp.array(one_over_kx)
        self.one_over_ky = jnp.array(one_over_ky)
        self.dv = dv
        self.v = v

    def compute_charges(self, f):
        return jnp.sum(f * self.v**2.0, axis=2) * self.dv

    def __call__(self, f: jnp.ndarray):
        return jnp.concatenate(
            [
                jnp.real(
                    jnp.fft.ifft(
                        1j * self.one_over_kx[:, None] * jnp.fft.fft(self.ion_charge - self.compute_charges(f), axis=0),
                        axis=0,
                    )
                )[..., None],
                jnp.real(
                    jnp.fft.ifft(
                        1j * self.one_over_ky[None, :] * jnp.fft.fft(self.ion_charge - self.compute_charges(f), axis=1),
                        axis=1,
                    )
                )[..., None],
                jnp.zeros((self.one_over_kx.size, self.one_over_ky.size, 1)),
            ],
            axis=-1,
        )
