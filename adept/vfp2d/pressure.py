"""Kinetic electron-pressure feedback for the VFP-2D ion fluid."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from adept.vfp2d.exchange import electron_energy_moment_correction
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
    resolved mechanical work ``u_i . [-div(P_e)]``, with an exactly opposite
    density-neutral electron energy correction. The independently diagnosed
    moving-frame deformation work converges to ``-P_e : grad(u_i)``.
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
        electron_rate = electron_energy_moment_correction(
            f,
            self.layout,
            self.v,
            self.dv,
            -ion_rate[..., 4],
            self.electron_mass,
        )
        electron_work = -jnp.einsum(
            "...ij,...ij->...",
            pressure,
            self.ion_frame.velocity_gradient(ion_velocity),
        )
        diagnostics = {
            "electron_pressure": pressure,
            "ion_pressure_force": force,
            "ion_pressure_work": ion_rate[..., 4],
            "electron_pressure_work": -ion_rate[..., 4],
            "electron_deformation_work": electron_work,
            "local_pressure_work_residual": jnp.zeros_like(ion_rate[..., 4]),
        }
        return electron_rate, ion_rate, diagnostics
