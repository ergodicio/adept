"""Conservative ion-fluid building blocks for kinetic-electron coupling.

The ion state uses cell averages of ``(rho, rho*u_x, rho*u_y, rho*u_z, E)``.
Only the two configuration-space directions carry fluxes, but all three momentum
components are retained for the eventual 2D3P VFP coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

Boundary = Literal["periodic", "outflow"]

N_HYDRO_VARS = 5
DENSITY = 0
ENERGY = 4


def _check_state_shape(state: Array) -> None:
    if state.ndim < 1 or state.shape[-1] != N_HYDRO_VARS:
        raise ValueError("hydro states must have a final axis of length 5")


def _check_gamma(gamma: float) -> None:
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")


def primitive_to_conserved(primitive: Array, gamma: float = 5.0 / 3.0) -> Array:
    """Convert ``(rho, u_x, u_y, u_z, p)`` to conservative variables."""

    primitive = jnp.asarray(primitive)
    _check_state_shape(primitive)
    _check_gamma(gamma)
    rho = primitive[..., DENSITY]
    velocity = primitive[..., 1:4]
    pressure = primitive[..., ENERGY]
    momentum = rho[..., None] * velocity
    total_energy = pressure / (gamma - 1.0) + 0.5 * rho * jnp.sum(velocity**2, axis=-1)
    return jnp.concatenate((rho[..., None], momentum, total_energy[..., None]), axis=-1)


def conserved_to_primitive(
    conserved: Array,
    gamma: float = 5.0 / 3.0,
    *,
    density_floor: float = 1.0e-12,
    pressure_floor: float = 1.0e-12,
) -> Array:
    """Convert conservative variables to ``(rho, u_x, u_y, u_z, p)``.

    Floors keep wave-speed and reconstruction calculations finite if an
    intermediate explicit stage approaches vacuum. Valid states are unchanged.
    """

    conserved = jnp.asarray(conserved)
    _check_state_shape(conserved)
    _check_gamma(gamma)
    if density_floor <= 0.0 or pressure_floor <= 0.0:
        raise ValueError("density_floor and pressure_floor must be positive")

    rho = jnp.maximum(conserved[..., DENSITY], density_floor)
    momentum = conserved[..., 1:4]
    velocity = momentum / rho[..., None]
    kinetic_energy = 0.5 * jnp.sum(momentum**2, axis=-1) / rho
    pressure = jnp.maximum((gamma - 1.0) * (conserved[..., ENERGY] - kinetic_energy), pressure_floor)
    return jnp.concatenate((rho[..., None], velocity, pressure[..., None]), axis=-1)


def euler_flux(primitive: Array, normal_axis: int, gamma: float = 5.0 / 3.0) -> Array:
    """Return the physical Euler flux normal to ``x`` (0) or ``y`` (1)."""

    primitive = jnp.asarray(primitive)
    _check_state_shape(primitive)
    _check_gamma(gamma)
    if normal_axis not in (0, 1):
        raise ValueError("normal_axis must be 0 (x) or 1 (y)")

    conserved = primitive_to_conserved(primitive, gamma)
    pressure = primitive[..., ENERGY]
    normal_velocity = primitive[..., 1 + normal_axis]
    flux = conserved * normal_velocity[..., None]
    flux = flux.at[..., 1 + normal_axis].add(pressure)
    return flux.at[..., ENERGY].set((conserved[..., ENERGY] + pressure) * normal_velocity)


def _nonzero(value: Array) -> Array:
    eps = jnp.finfo(value.dtype).eps
    sign = jnp.where(value < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(value) < eps, sign * eps, value)


def _hllc_star_state(
    primitive: Array,
    wave_speed: Array,
    contact_speed: Array,
    normal_axis: int,
    gamma: float,
) -> Array:
    rho = primitive[..., DENSITY]
    pressure = primitive[..., ENERGY]
    velocity = primitive[..., 1:4]
    normal_velocity = velocity[..., normal_axis]
    conserved = primitive_to_conserved(primitive, gamma)
    denominator = _nonzero(wave_speed - contact_speed)
    star_density = rho * (wave_speed - normal_velocity) / denominator
    star_velocity = velocity.at[..., normal_axis].set(contact_speed)
    star_pressure = pressure + rho * (wave_speed - normal_velocity) * (contact_speed - normal_velocity)
    star_energy = (
        (wave_speed - normal_velocity) * conserved[..., ENERGY]
        - pressure * normal_velocity
        + star_pressure * contact_speed
    ) / denominator
    return jnp.concatenate(
        (
            star_density[..., None],
            star_density[..., None] * star_velocity,
            star_energy[..., None],
        ),
        axis=-1,
    )


def hllc_flux(left: Array, right: Array, normal_axis: int, gamma: float = 5.0 / 3.0) -> Array:
    """Compute the HLLC flux for left/right primitive states.

    The same implementation is used in both spatial directions. The third
    velocity component is transported as tangential momentum.
    """

    left = jnp.asarray(left)
    right = jnp.asarray(right)
    _check_state_shape(left)
    _check_state_shape(right)
    _check_gamma(gamma)
    if left.shape != right.shape:
        raise ValueError("left and right states must have matching shapes")
    if normal_axis not in (0, 1):
        raise ValueError("normal_axis must be 0 (x) or 1 (y)")

    rho_left, rho_right = left[..., DENSITY], right[..., DENSITY]
    pressure_left, pressure_right = left[..., ENERGY], right[..., ENERGY]
    velocity_left = left[..., 1 + normal_axis]
    velocity_right = right[..., 1 + normal_axis]
    sound_left = jnp.sqrt(gamma * pressure_left / rho_left)
    sound_right = jnp.sqrt(gamma * pressure_right / rho_right)
    wave_left = jnp.minimum(velocity_left - sound_left, velocity_right - sound_right)
    wave_right = jnp.maximum(velocity_left + sound_left, velocity_right + sound_right)
    contact_numerator = (
        pressure_right
        - pressure_left
        + rho_left * velocity_left * (wave_left - velocity_left)
        - rho_right * velocity_right * (wave_right - velocity_right)
    )
    contact_denominator = _nonzero(rho_left * (wave_left - velocity_left) - rho_right * (wave_right - velocity_right))
    contact = contact_numerator / contact_denominator

    conserved_left = primitive_to_conserved(left, gamma)
    conserved_right = primitive_to_conserved(right, gamma)
    flux_left = euler_flux(left, normal_axis, gamma)
    flux_right = euler_flux(right, normal_axis, gamma)
    star_left = _hllc_star_state(left, wave_left, contact, normal_axis, gamma)
    star_right = _hllc_star_state(right, wave_right, contact, normal_axis, gamma)
    left_star_flux = flux_left + wave_left[..., None] * (star_left - conserved_left)
    right_star_flux = flux_right + wave_right[..., None] * (star_right - conserved_right)

    return jnp.where(
        (wave_left >= 0.0)[..., None],
        flux_left,
        jnp.where(
            (contact >= 0.0)[..., None],
            left_star_flux,
            jnp.where((wave_right > 0.0)[..., None], right_star_flux, flux_right),
        ),
    )


def _minmod3(first: Array, second: Array, third: Array) -> Array:
    all_positive = (first > 0.0) & (second > 0.0) & (third > 0.0)
    all_negative = (first < 0.0) & (second < 0.0) & (third < 0.0)
    magnitude = jnp.minimum(jnp.abs(first), jnp.minimum(jnp.abs(second), jnp.abs(third)))
    return jnp.where(all_positive, magnitude, jnp.where(all_negative, -magnitude, 0.0))


def _limited_slope(primitive: Array, axis: int, boundary: Boundary, theta: float) -> Array:
    values = jnp.moveaxis(primitive, axis, 0)
    if boundary == "periodic":
        before = jnp.roll(values, 1, axis=0)
        after = jnp.roll(values, -1, axis=0)
    elif boundary == "outflow":
        before = jnp.concatenate((values[:1], values[:-1]), axis=0)
        after = jnp.concatenate((values[1:], values[-1:]), axis=0)
    else:
        raise ValueError(f"unsupported hydro boundary: {boundary}")
    slope = _minmod3(theta * (values - before), 0.5 * (after - before), theta * (after - values))
    return jnp.moveaxis(slope, 0, axis)


def _interface_states(
    primitive: Array,
    axis: int,
    boundary: Boundary,
    theta: float,
    density_floor: float,
    pressure_floor: float,
) -> tuple[Array, Array]:
    """Reconstruct primitive states on all ``n + 1`` faces along one axis."""

    values = jnp.moveaxis(primitive, axis, 0)
    slopes = jnp.moveaxis(_limited_slope(primitive, axis, boundary, theta), axis, 0)
    high_left_interior = values[:-1] + 0.5 * slopes[:-1]
    high_right_interior = values[1:] - 0.5 * slopes[1:]
    low_left_interior = values[:-1]
    low_right_interior = values[1:]

    if boundary == "periodic":
        high_left_boundary = values[-1] + 0.5 * slopes[-1]
        high_right_boundary = values[0] - 0.5 * slopes[0]
        low_left_boundary = values[-1]
        low_right_boundary = values[0]
    else:
        high_left_boundary = values[0]
        high_right_boundary = values[0] - 0.5 * slopes[0]
        low_left_boundary = values[0]
        low_right_boundary = values[0]

    high_left = jnp.concatenate(
        (high_left_boundary[None, ...], high_left_interior, (values[-1] + 0.5 * slopes[-1])[None, ...]),
        axis=0,
    )
    high_right = jnp.concatenate(
        (high_right_boundary[None, ...], high_right_interior, high_right_boundary[None, ...]),
        axis=0,
    )
    low_left = jnp.concatenate(
        (low_left_boundary[None, ...], low_left_interior, values[-1][None, ...]),
        axis=0,
    )
    low_right = jnp.concatenate(
        (low_right_boundary[None, ...], low_right_interior, low_right_boundary[None, ...]),
        axis=0,
    )

    if boundary == "outflow":
        high_right = high_right.at[-1].set(values[-1])
        low_right = low_right.at[-1].set(values[-1])

    valid_left = (
        jnp.all(jnp.isfinite(high_left), axis=-1)
        & (high_left[..., DENSITY] > density_floor)
        & (high_left[..., ENERGY] > pressure_floor)
    )
    valid_right = (
        jnp.all(jnp.isfinite(high_right), axis=-1)
        & (high_right[..., DENSITY] > density_floor)
        & (high_right[..., ENERGY] > pressure_floor)
    )
    use_high_order = (valid_left & valid_right)[..., None]
    left = jnp.where(use_high_order, high_left, low_left)
    right = jnp.where(use_high_order, high_right, low_right)
    return jnp.moveaxis(left, 0, axis), jnp.moveaxis(right, 0, axis)


@dataclass(frozen=True)
class IonEuler2D:
    """MUSCL-HLLC finite-volume operator for an ideal ion fluid."""

    dx: float
    dy: float
    gamma: float = 5.0 / 3.0
    boundaries: tuple[Boundary, Boundary] = ("periodic", "periodic")
    limiter_theta: float = 1.5
    density_floor: float = 1.0e-12
    pressure_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        _check_gamma(self.gamma)
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("dx and dy must be positive")
        if len(self.boundaries) != 2 or any(boundary not in ("periodic", "outflow") for boundary in self.boundaries):
            raise ValueError("boundaries must contain periodic/outflow choices for x and y")
        if not 1.0 <= self.limiter_theta <= 2.0:
            raise ValueError("limiter_theta must lie between 1 and 2")
        if self.density_floor <= 0.0 or self.pressure_floor <= 0.0:
            raise ValueError("density_floor and pressure_floor must be positive")

    def fluxes(self, conserved: Array, axis: int) -> Array:
        """Return numerical fluxes on all faces along ``axis``."""

        conserved = jnp.asarray(conserved)
        _check_state_shape(conserved)
        if conserved.ndim != 3:
            raise ValueError("IonEuler2D expects state shape (nx, ny, 5)")
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (x) or 1 (y)")
        primitive = conserved_to_primitive(
            conserved,
            self.gamma,
            density_floor=self.density_floor,
            pressure_floor=self.pressure_floor,
        )
        left, right = _interface_states(
            primitive,
            axis,
            self.boundaries[axis],
            self.limiter_theta,
            self.density_floor,
            self.pressure_floor,
        )
        return hllc_flux(left, right, axis, self.gamma)

    def rhs(self, conserved: Array) -> Array:
        """Return the conservative semidiscrete spatial operator."""

        flux_x = self.fluxes(conserved, axis=0)
        flux_y = self.fluxes(conserved, axis=1)
        return -(flux_x[1:] - flux_x[:-1]) / self.dx - (flux_y[:, 1:] - flux_y[:, :-1]) / self.dy

    def step(self, conserved: Array, dt: float | Array) -> Array:
        """Advance one second-order strong-stability-preserving Runge--Kutta step."""

        first_stage = conserved + dt * self.rhs(conserved)
        return 0.5 * (conserved + first_stage + dt * self.rhs(first_stage))

    def cfl_timestep(self, conserved: Array, cfl: float = 0.4) -> Array:
        """Return the unsplit acoustic/advection CFL timestep."""

        if not 0.0 < cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")
        primitive = conserved_to_primitive(
            conserved,
            self.gamma,
            density_floor=self.density_floor,
            pressure_floor=self.pressure_floor,
        )
        sound_speed = jnp.sqrt(self.gamma * primitive[..., ENERGY] / primitive[..., DENSITY])
        rate = (jnp.abs(primitive[..., 1]) + sound_speed) / self.dx
        rate = rate + (jnp.abs(primitive[..., 2]) + sound_speed) / self.dy
        maximum_rate = jnp.max(rate)
        return jnp.where(maximum_rate > 0.0, cfl / maximum_rate, jnp.inf)
