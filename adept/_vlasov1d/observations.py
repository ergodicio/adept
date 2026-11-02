"""Explicit numerical observations shared by the legacy and prepared solvers."""

import equinox as eqx
import jax
from interpax import interp2d
from jax import numpy as jnp
from jax.scipy.special import xlogy


class FieldsObservation(eqx.Module):
    species_grids: dict
    dx: float

    def __call__(self, t, y, args):
        result = {}
        for name, grid in self.species_grids.items():
            v, dv = grid["v"], grid["dv"]
            f = y[name]
            n = jnp.sum(f * dv, axis=1)
            current = jnp.sum(f * v[None, :] * dv, axis=1)
            velocity = current / n
            centered_v = v[None, :] - velocity[:, None]
            result[name] = {
                "n": n,
                "j": current,
                "v": velocity,
                "p": jnp.sum(f * centered_v**2.0 * dv, axis=1),
                "q": jnp.sum(f * centered_v**3.0 * dv, axis=1),
                "-flogf": jnp.sum(-xlogy(jnp.abs(f), jnp.abs(f)) * dv, axis=1),
                "f^2": jnp.sum(f * f * dv, axis=1),
            }
        result.update({name: y[name] for name in ("e", "de", "a", "prev_a")})
        result["pond"] = -0.5 * jnp.gradient(y["a"] ** 2.0, self.dx)[1:-1]
        return result


class DistributionObservation(eqx.Module):
    name: str = eqx.field(static=True)

    def __call__(self, t, y, args):
        return y[self.name]


class InterpolatedDistributionObservation(eqx.Module):
    name: str = eqx.field(static=True)
    spectral: bool = eqx.field(static=True)
    shape: tuple[int, int] = eqx.field(static=True)
    spatial_axis: jax.Array
    velocity_axis: jax.Array
    spatial_queries: jax.Array
    velocity_queries: jax.Array

    def __call__(self, t, y, args):
        f = jnp.abs(jnp.fft.rfft(y[self.name], axis=0)) if self.spectral else y[self.name]
        return interp2d(
            self.spatial_queries,
            self.velocity_queries,
            self.spatial_axis,
            self.velocity_axis,
            f,
            method="linear",
        ).reshape(self.shape)


class ScalarsObservation(eqx.Module):
    species_grids: dict
    species_params: dict
    dx: float

    def __call__(self, t, y, args):
        scalars = {}
        mean_kinetic_energy = 0.0
        for name, grid in self.species_grids.items():
            v, dv = grid["v"][None, :], grid["dv"]
            f = y[name]

            def moment(value, spacing=dv):
                return jnp.mean(jnp.sum(value * spacing, axis=1))

            scalars[f"mean_P_{name}"] = moment(f * v**2.0)
            scalars[f"mean_j_{name}"] = moment(f * v)
            scalars[f"mean_n_{name}"] = moment(f)
            scalars[f"mean_q_{name}"] = moment(f * v**3.0)
            scalars[f"mean_-flogf_{name}"] = moment(-xlogy(jnp.abs(f), jnp.abs(f)))
            scalars[f"mean_f2_{name}"] = moment(f * f)
            mean_kinetic_energy += 0.5 * self.species_params[name]["mass"] * scalars[f"mean_P_{name}"]
        scalars["mean_de2"] = jnp.mean(y["de"] ** 2.0)
        scalars["mean_e2"] = jnp.mean(y["e"] ** 2.0)
        scalars["mean_pond"] = jnp.mean(-0.5 * jnp.gradient(y["a"] ** 2.0, self.dx)[1:-1])
        scalars["mean_kinetic_energy"] = mean_kinetic_energy
        scalars["mean_field_energy"] = 0.5 * scalars["mean_e2"]
        scalars["mean_total_energy"] = mean_kinetic_energy + 0.5 * scalars["mean_e2"]
        return scalars
