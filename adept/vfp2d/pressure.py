"""Kinetic electron-pressure feedback for the VFP-2D ion fluid."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from adept.vfp2d.harmonics import HarmonicLayout, density, scalar_velocity_moment, tensor_velocity_moment
from adept.vfp2d.moving_frame import IonFrameVlasov


def electron_pressure_tensor(
    f: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    electron_mass: float = 1.0,
) -> Array:
    """Return the full peculiar-frame pressure tensor from ``f0 + f2``."""

    if electron_mass <= 0.0:
        raise ValueError("electron_mass must be positive")
    electron_density = density(f, layout, v, dv)
    mean_square_speed = scalar_velocity_moment(f, layout, v, dv, power=2)
    scalar_pressure = float(electron_mass) * electron_density * mean_square_speed / 3.0
    anisotropic_pressure = (
        float(electron_mass) * electron_density[..., None, None] * tensor_velocity_moment(f, layout, v, dv, power=0)
    )
    identity = jnp.eye(3, dtype=scalar_pressure.dtype)
    return scalar_pressure[..., None, None] * identity + anisotropic_pressure


class ElectronPressureCoupling:
    """Map kinetic electron pressure into conservative ion sources.

    The ion momentum source is ``-div(P_e)``. Its total-energy source is the
    resolved mechanical work ``-u_i . div(P_e)``. Electron peculiar energy
    receives ``-P_e : grad(u_i)`` from the moving-frame deformation operator,
    so this source must not add another f00 energy correction. Their sum is
    the conservative pressure-flux divergence ``-div(P_e . u_i)``; the two
    work terms cancel globally on a periodic domain, not pointwise.
    """

    def __init__(self, ion_frame: IonFrameVlasov, *, electron_mass: float = 1.0):
        if electron_mass <= 0.0:
            raise ValueError("electron_mass must be positive")
        self.ion_frame = ion_frame
        self.layout = ion_frame.layout
        self.v = ion_frame.vlasov.v
        self.dv = ion_frame.vlasov.dv
        self.electron_mass = float(electron_mass)

    def pressure_tensor(self, f: Array) -> Array:
        return electron_pressure_tensor(
            f,
            self.layout,
            self.v,
            self.dv,
            self.electron_mass,
        )

    def pressure_divergence(self, f: Array) -> Array:
        pressure = self.pressure_tensor(f)
        derivative_x = jnp.real(self.ion_frame.vlasov.spatial_derivative(pressure[..., :, 0], axis=0))
        derivative_y = jnp.real(self.ion_frame.vlasov.spatial_derivative(pressure[..., :, 1], axis=1))
        return derivative_x + derivative_y

    def __call__(self, f: Array, ion_conserved: Array) -> tuple[Array, Array, dict[str, Array]]:
        if ion_conserved.shape != (*f.shape[:-2], 5):
            raise ValueError("ion_conserved must match the spatial VFP shape and have five conserved variables")
        pressure = self.pressure_tensor(f)
        force = -self.pressure_divergence(f)
        ion_velocity = ion_conserved[..., 1:4] / ion_conserved[..., :1]
        ion_rate = jnp.zeros_like(ion_conserved)
        ion_rate = ion_rate.at[..., 1:4].set(force)
        ion_rate = ion_rate.at[..., 4].set(jnp.sum(ion_velocity * force, axis=-1))
        electron_rate = jnp.zeros_like(f)
        electron_work = -jnp.einsum(
            "...ij,...ij->...",
            pressure,
            self.ion_frame.velocity_gradient(ion_velocity),
        )
        pressure_flux = jnp.einsum("...ij,...i->...j", pressure, ion_velocity)
        flux_divergence = jnp.real(self.ion_frame.vlasov.spatial_derivative(pressure_flux[..., 0], axis=0))
        flux_divergence += jnp.real(self.ion_frame.vlasov.spatial_derivative(pressure_flux[..., 1], axis=1))
        diagnostics = {
            "electron_pressure": pressure,
            "ion_pressure_force": force,
            "ion_pressure_work": ion_rate[..., 4],
            "electron_pressure_work": electron_work,
            "electron_deformation_work": electron_work,
            "pressure_flux_divergence": flux_divergence,
            "local_pressure_work_residual": ion_rate[..., 4] + electron_work + flux_divergence,
        }
        return electron_rate, ion_rate, diagnostics
