"""Long-timescale kinetic Ohm-law utilities for VFP-2D.

The diagnostic closure follows Joglekar et al., PRL 112, 105004 (2014),
Eq. (2).  It intentionally neglects electron inertia.  This makes it useful
for collisional transport and the PRL benchmark, but distinct from the future
fully implicit kinetic-current response solve.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from adept.vfp2d.harmonics import (
    HarmonicLayout,
    current,
    density,
    nernst_velocity,
    scalar_velocity_moment,
    tensor_velocity_moment,
)


class KineticOhm2D:
    """Evaluate the moment-resolved, inertia-free generalized Ohm law."""

    def __init__(
        self,
        layout: HarmonicLayout,
        v: Array,
        dv: float,
        kx: Array,
        ky: Array,
        *,
        resistivity_coefficient: float = 0.0,
    ):
        self.layout = layout
        self.v = jnp.asarray(v)
        self.dv = float(dv)
        self.kx = jnp.asarray(kx)
        self.ky = jnp.asarray(ky)
        self.resistivity_coefficient = float(resistivity_coefficient)

    def ddx(self, value: Array) -> Array:
        shape = (self.kx.size,) + (1,) * (value.ndim - 1)
        return jnp.fft.ifft(1j * self.kx.reshape(shape) * jnp.fft.fft(value, axis=0), axis=0).real

    def ddy(self, value: Array) -> Array:
        shape = (1, self.ky.size) + (1,) * (value.ndim - 2)
        return jnp.fft.ifft(1j * self.ky.reshape(shape) * jnp.fft.fft(value, axis=1), axis=1).real

    def __call__(
        self,
        flm: Array,
        magnetic_field: Array,
        *,
        plasma_current: Array | None = None,
        hidden_dndz: Array | float = 0.0,
    ) -> tuple[Array, dict[str, Array]]:
        """Return ``E`` and its five PRL Eq. (2) contributions.

        ``hidden_dndz`` represents the prescribed density derivative in the
        unresolved z direction. The kinetic step supplies the matching
        ``df/dz = (dndz/ne) f`` streaming term, so an isothermal Maxwellian is
        in pressure balance instead of being spuriously accelerated by Ez.
        """

        ne = density(flm, self.layout, self.v, self.dv)
        safe_ne = jnp.maximum(ne, jnp.finfo(ne.dtype).tiny)
        if plasma_current is None:
            plasma_current = current(flm, self.layout, self.v, self.dv)

        v3 = scalar_velocity_moment(flm, self.layout, self.v, self.dv, power=3)
        v5 = scalar_velocity_moment(flm, self.layout, self.v, self.dv, power=5)
        tensor_v3 = tensor_velocity_moment(flm, self.layout, self.v, self.dv, power=3)
        v_nernst = nernst_velocity(
            flm,
            self.layout,
            self.v,
            self.dv,
            plasma_current=plasma_current,
        )
        safe_v3 = jnp.maximum(v3, jnp.finfo(v3.dtype).tiny)
        hidden_dndz = jnp.broadcast_to(jnp.asarray(hidden_dndz), ne.shape)

        eta = self.resistivity_coefficient / safe_v3
        resistive = eta[..., None] * plasma_current
        hall = jnp.cross(plasma_current, magnetic_field) / safe_ne[..., None]
        nernst = -jnp.cross(v_nernst, magnetic_field)

        scalar_flux = ne * v5
        scalar_gradient = jnp.stack((self.ddx(scalar_flux), self.ddy(scalar_flux), hidden_dndz * v5), axis=-1)
        scalar_pressure = -scalar_gradient / (6.0 * safe_ne * safe_v3)[..., None]

        tensor_flux = ne[..., None, None] * tensor_v3
        tensor_divergence = self.ddx(tensor_flux[..., :, 0]) + self.ddy(tensor_flux[..., :, 1])
        tensor_divergence = tensor_divergence + hidden_dndz[..., None] * tensor_v3[..., :, 2]
        tensor_pressure = -tensor_divergence / (2.0 * safe_ne * safe_v3)[..., None]

        terms = {
            "resistive": resistive,
            "hall": hall,
            "nernst": nernst,
            "scalar_pressure": scalar_pressure,
            "tensor_pressure": tensor_pressure,
        }
        electric_field = sum(terms.values(), start=jnp.zeros_like(magnetic_field))
        return electric_field, terms


def project_current_moment(
    flm: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    target_current: Array,
) -> Array:
    """Project only the bulk-current moment of ``f1`` onto ``target_current``.

    The correction is proportional to the local ``f00`` radial shape. It leaves
    density and every ``l != 1`` harmonic unchanged, while retaining the
    non-Maxwellian part of ``f1`` that carries the heat flux.
    """

    i10, i11 = layout.index(1, 0), layout.index(1, 1)
    if i10 < 0:
        raise ValueError("current projection requires l_max >= 1")
    measured = current(flm, layout, v, dv)
    correction = target_current - measured
    f00 = jnp.real(flm[..., layout.index(0, 0), :])
    response = (4.0 * jnp.pi / 3.0) * jnp.sum(f00 * v**3, axis=-1) * dv
    response = jnp.maximum(response, jnp.finfo(response.dtype).tiny)
    # current = -response * [a_x, 2 Re(a_1), -2 Im(a_1)]
    ax = -correction[..., 0] / response
    a1 = (-correction[..., 1] + 1j * correction[..., 2]) / (2.0 * response)
    result = flm.at[..., i10, :].add(ax[..., None] * f00)
    if i11 >= 0:
        result = result.at[..., i11, :].add(a1[..., None] * f00)
    return result
