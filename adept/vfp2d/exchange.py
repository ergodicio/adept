"""Conservative finite-mass electron--ion moment exchange for VFP-2D.

This module supplies a differential, weak-anisotropy relaxation reference.  It
changes only the electron ``f0`` energy moment and ``f1`` bulk-momentum moment,
then applies their measured discrete opposites to the ion conserved state.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from adept.vfp2d.harmonics import HarmonicLayout, current, density, scalar_velocity_moment
from adept.vfp2d.moving_frame import IonFrameVlasov
from adept.vfp2d.ohm import project_current_moment


def electron_momentum_density(
    f: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    electron_mass: float = 1.0,
) -> Array:
    """Return ``m_e integral(c f d^3c)`` in the local ion frame."""

    return float(electron_mass) * current(f, layout, v, dv, charge=1.0)


def electron_kinetic_energy_density(
    f: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    electron_mass: float = 1.0,
) -> Array:
    """Return ``m_e/2 integral(c^2 f d^3c)``."""

    i00 = layout.index(0, 0)
    f00 = jnp.real(f[..., i00, :])
    return 2.0 * jnp.pi * float(electron_mass) * jnp.sum(f00 * jnp.asarray(v) ** 4, axis=-1) * float(dv)


def electron_energy_moment_correction(
    f: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    energy_correction: Array,
    electron_mass: float,
) -> Array:
    """Return a density-neutral ``f00`` correction with a prescribed energy."""

    i00 = layout.index(0, 0)
    f00 = jnp.real(f[..., i00, :])
    mean_square_speed = scalar_velocity_moment(f, layout, v, dv, power=2)
    density_neutral_basis = (v**2 - mean_square_speed[..., None]) * f00
    response = 2.0 * jnp.pi * electron_mass * jnp.sum(density_neutral_basis * v**4, axis=-1) * dv
    tiny = jnp.finfo(response.dtype).tiny
    safe_response = jnp.where(jnp.abs(response) > tiny, response, 1.0)
    amplitude = jnp.where(jnp.abs(response) > tiny, energy_correction / safe_response, 0.0)
    return jnp.zeros_like(f).at[..., i00, :].set(amplitude[..., None] * density_neutral_basis)


class VelocityFrameRemap:
    """Translate ``f(c)`` between finite-mass ion velocity frames.

    A second-order velocity-space translation retains the resolved distribution
    shape. Discrete projections then enforce the exact Galilean density,
    momentum, and kinetic-energy transforms, including radial-grid boundary
    defects from the differential translation operator.
    """

    def __init__(self, ion_frame: IonFrameVlasov, *, electron_mass: float = 1.0):
        if electron_mass <= 0.0:
            raise ValueError("electron_mass must be positive")
        self.ion_frame = ion_frame
        self.layout = ion_frame.layout
        self.v = ion_frame.vlasov.v
        self.dv = ion_frame.vlasov.dv
        self.electron_mass = float(electron_mass)

    def __call__(self, f: Array, frame_velocity_change: Array) -> Array:
        expected_shape = (*f.shape[:-2], 3)
        if frame_velocity_change.shape != expected_shape:
            raise ValueError(f"frame_velocity_change must have shape {expected_shape}")

        electron_density = density(f, self.layout, self.v, self.dv)
        old_momentum = electron_momentum_density(f, self.layout, self.v, self.dv, self.electron_mass)
        old_energy = electron_kinetic_energy_density(f, self.layout, self.v, self.dv, self.electron_mass)

        first = self.ion_frame.frame_acceleration(f, frame_velocity_change)
        midpoint = f + 0.5 * first
        translated = f + self.ion_frame.frame_acceleration(midpoint, frame_velocity_change)

        # The differential operator is conservative in the continuum. Restore
        # exact discrete density before correcting its first two moments.
        i00 = self.layout.index(0, 0)
        translated_density = density(translated, self.layout, self.v, self.dv)
        tiny = jnp.finfo(translated_density.dtype).tiny
        density_scale = jnp.where(jnp.abs(translated_density) > tiny, electron_density / translated_density, 1.0)
        translated = translated.at[..., i00, :].set(density_scale[..., None] * jnp.real(translated[..., i00, :]))

        target_momentum = old_momentum - self.electron_mass * electron_density[..., None] * frame_velocity_change
        target_current = -target_momentum / self.electron_mass
        translated = project_current_moment(
            translated,
            self.layout,
            self.v,
            self.dv,
            target_current,
        )

        target_energy = old_energy - jnp.sum(frame_velocity_change * old_momentum, axis=-1)
        target_energy += 0.5 * self.electron_mass * electron_density * jnp.sum(frame_velocity_change**2, axis=-1)
        measured_energy = electron_kinetic_energy_density(
            translated,
            self.layout,
            self.v,
            self.dv,
            self.electron_mass,
        )
        translated += electron_energy_moment_correction(
            translated,
            self.layout,
            self.v,
            self.dv,
            target_energy - measured_energy,
            self.electron_mass,
        )
        return translated


class ElectronIonExchange:
    """Discrete equal-and-opposite electron--ion relaxation rates.

    ``momentum_relaxation_rate`` defines ``dP_e/dt = -nu_m P_e`` in the ion
    frame. ``temperature_relaxation_rate`` defines
    ``dT_e/dt = -nu_T (T_e - T_i)`` for the isotropic weak-drift temperature.
    The returned ion source has exactly opposite measured momentum and energy.
    """

    def __init__(
        self,
        layout: HarmonicLayout,
        v: Array,
        dv: float,
        *,
        electron_mass: float = 1.0,
        ion_mass: float = 1836.0,
        ion_gamma: float = 5.0 / 3.0,
    ):
        if layout.l_max < 1:
            raise ValueError("electron-ion momentum exchange requires l_max >= 1")
        if electron_mass <= 0.0 or ion_mass <= 0.0:
            raise ValueError("electron and ion masses must be positive")
        if ion_gamma <= 1.0:
            raise ValueError("ion_gamma must be greater than one")
        self.layout = layout
        self.v = jnp.asarray(v)
        self.dv = float(dv)
        self.electron_mass = float(electron_mass)
        self.ion_mass = float(ion_mass)
        self.ion_gamma = float(ion_gamma)

    def _temperatures(self, f: Array, ion_conserved: Array) -> tuple[Array, Array, Array]:
        electron_density = density(f, self.layout, self.v, self.dv)
        mean_square_speed = scalar_velocity_moment(f, self.layout, self.v, self.dv, power=2)
        electron_temperature = self.electron_mass * mean_square_speed / 3.0

        ion_density = ion_conserved[..., 0] / self.ion_mass
        ion_velocity = ion_conserved[..., 1:4] / ion_conserved[..., :1]
        ion_internal_energy = ion_conserved[..., 4] - 0.5 * ion_conserved[..., 0] * jnp.sum(ion_velocity**2, axis=-1)
        ion_pressure = (self.ion_gamma - 1.0) * ion_internal_energy
        ion_temperature = ion_pressure / jnp.maximum(ion_density, jnp.finfo(ion_density.dtype).tiny)
        return electron_density, electron_temperature, ion_temperature

    def _momentum_rate(self, f: Array, rate: Array) -> Array:
        measured_current = current(f, self.layout, self.v, self.dv)
        target_current = (1.0 - rate[..., None]) * measured_current
        return project_current_moment(f, self.layout, self.v, self.dv, target_current) - f

    def _thermal_rate(self, f: Array, energy_rate: Array) -> Array:
        return electron_energy_moment_correction(
            f,
            self.layout,
            self.v,
            self.dv,
            energy_rate,
            self.electron_mass,
        )

    def __call__(
        self,
        f: Array,
        ion_conserved: Array,
        *,
        momentum_relaxation_rate: float | Array,
        temperature_relaxation_rate: float | Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        if ion_conserved.shape != (*f.shape[:-2], 5):
            raise ValueError("ion_conserved must match the spatial VFP shape and have five conserved variables")
        spatial_shape = f.shape[:-2]
        momentum_rate = jnp.broadcast_to(jnp.asarray(momentum_relaxation_rate), spatial_shape)
        temperature_rate = jnp.broadcast_to(jnp.asarray(temperature_relaxation_rate), spatial_shape)
        electron_density, electron_temperature, ion_temperature = self._temperatures(f, ion_conserved)

        requested_energy_rate = -1.5 * electron_density * temperature_rate * (electron_temperature - ion_temperature)
        dfdt = self._momentum_rate(f, momentum_rate) + self._thermal_rate(f, requested_energy_rate)
        measured_momentum_rate = electron_momentum_density(
            dfdt,
            self.layout,
            self.v,
            self.dv,
            self.electron_mass,
        )
        measured_energy_rate = electron_kinetic_energy_density(
            dfdt,
            self.layout,
            self.v,
            self.dv,
            self.electron_mass,
        )

        ion_rate = jnp.zeros_like(ion_conserved)
        ion_rate = ion_rate.at[..., 1:4].set(-measured_momentum_rate)
        ion_velocity = ion_conserved[..., 1:4] / ion_conserved[..., :1]
        measured_lab_energy_rate = measured_energy_rate + jnp.sum(
            ion_velocity * measured_momentum_rate,
            axis=-1,
        )
        ion_rate = ion_rate.at[..., 4].set(-measured_lab_energy_rate)
        diagnostics = {
            "electron_temperature": electron_temperature,
            "ion_temperature": ion_temperature,
            "electron_momentum_rate": measured_momentum_rate,
            "ion_momentum_rate": ion_rate[..., 1:4],
            "electron_energy_rate": measured_energy_rate,
            "electron_lab_energy_rate": measured_lab_energy_rate,
            "ion_energy_rate": ion_rate[..., 4],
            "momentum_residual": measured_momentum_rate + ion_rate[..., 1:4],
            "energy_residual": measured_lab_energy_rate + ion_rate[..., 4],
        }
        return dfdt, ion_rate, diagnostics
