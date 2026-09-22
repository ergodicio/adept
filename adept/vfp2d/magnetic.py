"""Magnetic force and mechanical work for the quasistatic ion fluid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from adept.vfp2d.vector_field import Maxwell2D


class IonMagneticCoupling:
    """Return ``J x B`` and ``u_i . (J x B)`` in the solver normalization.

    Ampere's law is ``J = c**2 curl(B)`` and magnetic energy is ``c**2 B**2/2``.
    Reusing the induction operator's discrete curl makes ideal magnetic work
    cancel globally on the periodic grid. The energy source belongs to the ion
    total energy; its reservoir is the magnetic field advanced by Faraday's law.
    There is deliberately no opposite electron thermal-energy source.
    """

    def __init__(self, maxwell: Maxwell2D):
        self.maxwell = maxwell

    def force(self, magnetic_field: Array) -> Array:
        return jnp.cross(self.maxwell.c2 * self.maxwell.curl(magnetic_field), magnetic_field)

    def __call__(self, magnetic_field: Array, ions: Array) -> tuple[Array, dict[str, Array]]:
        if ions.shape != (*magnetic_field.shape[:-1], 5) or magnetic_field.shape[-1] != 3:
            raise ValueError("ions and magnetic_field must share the spatial grid and have 5 and 3 components")
        force = self.force(magnetic_field)
        velocity = ions[..., 1:4] / ions[..., :1]
        work = jnp.sum(velocity * force, axis=-1)
        rate = jnp.zeros_like(ions).at[..., 1:4].set(force).at[..., 4].set(work)
        return rate, {"ion_magnetic_force": force, "ion_magnetic_work": work}
