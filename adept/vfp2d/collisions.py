"""Collision adapters for packed arbitrary-``f_lm`` VFP-2D states."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from adept.vfp1d.fokker_planck import F0Collisions, FLMCollisions
from adept.vfp2d.harmonics import HarmonicLayout


class AnisotropicCollisions:
    """Apply the Tzoufras linearized anisotropic operator to every packed mode.

    The radial operator depends on ``l`` but is diagonal in ``m``. Spatial axes
    are flattened into a single batch for the existing JAX/Lineax tridiagonal
    solve, then restored. The isotropic ``f00`` mode is intentionally unchanged;
    it is advanced by the conservative isotropic collision solver.
    """

    def __init__(self, operator: FLMCollisions, layout: HarmonicLayout):
        if operator.grid.nl < layout.l_max:
            raise ValueError("FLMCollisions grid.nl must be at least layout.l_max")
        self.operator = operator
        self.layout = layout

    @staticmethod
    def _spatial_field(value: Array | float, shape: tuple[int, ...], dtype) -> Array:
        return jnp.broadcast_to(jnp.asarray(value, dtype=dtype), shape).reshape(-1)

    def __call__(self, flm: Array, Z: Array | float, ni: Array | float, dt: float) -> Array:
        spatial_shape = flm.shape[:-2]
        nv = flm.shape[-1]
        f0 = jnp.real(flm[..., self.layout.index(0, 0), :]).reshape((-1, nv))
        flat_Z = self._spatial_field(Z, spatial_shape, f0.dtype)
        flat_ni = self._spatial_field(ni, spatial_shape, f0.dtype)
        result = flm

        for i, (ell, _m) in enumerate(self.layout.pairs):
            if ell == 0:
                continue
            mode = flm[..., i, :].reshape((-1, nv))
            updated = self.operator.solve_harmonic(flat_Z, flat_ni, f0, mode, dt, il=ell)
            result = result.at[..., i, :].set(updated.reshape((*spatial_shape, nv)))
        return result


class CollisionStep:
    """Conservative ``f00`` plus arbitrary-``f_lm`` implicit collision step."""

    def __init__(
        self,
        layout: HarmonicLayout,
        isotropic: F0Collisions | None = None,
        anisotropic: AnisotropicCollisions | None = None,
    ):
        self.layout = layout
        self.isotropic = isotropic
        self.anisotropic = anisotropic

    def __call__(
        self,
        flm: Array,
        Z: Array | float,
        ni: Array | float,
        dt: float,
        *,
        D0_heating: Array | float | None = None,
        ib_vosc2: Array | float | None = None,
        ib_Z2ni_w0: Array | float | None = None,
    ) -> Array:
        spatial_shape = flm.shape[:-2]
        nv = flm.shape[-1]
        result = flm
        if self.isotropic is not None:
            f00 = jnp.real(result[..., self.layout.index(0, 0), :]).reshape((-1, nv))
            heating = {}
            if D0_heating is not None:
                heating["D0_heating"] = jnp.broadcast_to(
                    jnp.asarray(D0_heating), spatial_shape
                ).reshape(-1)
            if ib_vosc2 is not None:
                if ib_Z2ni_w0 is None:
                    raise ValueError("ib_Z2ni_w0 is required when inverse-bremsstrahlung heating is enabled")
                heating["ib_vosc2"] = jnp.broadcast_to(
                    jnp.asarray(ib_vosc2), spatial_shape
                ).reshape(-1)
                heating["ib_Z2ni_w0"] = jnp.broadcast_to(
                    jnp.asarray(ib_Z2ni_w0), spatial_shape
                ).reshape(-1)
            f00 = self.isotropic(None, f00, dt, **heating).reshape((*spatial_shape, nv))
            result = result.at[..., self.layout.index(0, 0), :].set(f00)
        if self.anisotropic is not None:
            result = self.anisotropic(result, Z=Z, ni=ni, dt=dt)
        return result
