"""Time-centered kinetic-electron / ion-fluid coupling for VFP-2D.

The split composes ion-frame electrons, ideal-ion Euler transport, kinetic
electron-pressure feedback, and finite-mass electron--ion exchange. It remains
explicit and auditable: hydro half-step, coupled-source half-step, full kinetic
step, the second coupled-source half-step, then the second hydro half-step.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from adept.vfp2d.exchange import (
    ElectronIonExchange,
    VelocityFrameRemap,
    electron_kinetic_energy_density,
    electron_momentum_density,
)
from adept.vfp2d.harmonics import HarmonicLayout, complex_to_real, density, real_to_complex
from adept.vfp2d.hydro import IonEuler2D, conserved_to_primitive
from adept.vfp2d.pressure import ElectronPressureCoupling
from adept.vfp2d.vector_field import KineticOhmStep


class CoupledIonKineticStep:
    """Strang-style map step for kinetic electrons and ideal-fluid ions.

    The electron step must be a ``KineticOhmStep`` configured with an ion-frame
    operator. Ion kinematics are held at the hydro midpoint during its RK stages.
    This production coupling includes ideal-ion transport and local finite-mass
    thermal and momentum exchange together with electron-pressure feedback.
    """

    def __init__(
        self,
        electron_step: KineticOhmStep,
        hydro: IonEuler2D,
        dt: float,
        *,
        exchange: ElectronIonExchange | None = None,
        pressure: ElectronPressureCoupling | None = None,
        evolve_ions: bool = True,
    ):
        if electron_step.ion_frame is None:
            raise ValueError("coupled kinetic-ion stepping requires an ion-frame electron operator")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.electron_step = electron_step
        self.hydro = hydro
        self.dt = float(dt)
        self.exchange = exchange
        self.pressure = pressure
        electron_mass = 1.0
        if exchange is not None:
            electron_mass = exchange.electron_mass
        if pressure is not None:
            if exchange is not None and pressure.electron_mass != electron_mass:
                raise ValueError("pressure and exchange electron masses must match")
            electron_mass = pressure.electron_mass
        self.frame_remap = VelocityFrameRemap(electron_step.ion_frame, electron_mass=electron_mass)
        self.evolve_ions = bool(evolve_ions)

    def _hydro_half_step(self, ions: Array) -> Array:
        return self.hydro.step(ions, 0.5 * self.dt) if self.evolve_ions else ions

    def ion_kinematics(self, ions: Array) -> dict[str, Array]:
        """Return midpoint velocity, gradient, and material acceleration."""

        primitive = conserved_to_primitive(
            ions,
            self.hydro.gamma,
            density_floor=self.hydro.density_floor,
            pressure_floor=self.hydro.pressure_floor,
        )
        velocity = primitive[..., 1:4]
        gradient = self.electron_step.ion_frame.velocity_gradient(velocity)
        if self.evolve_ions:
            hydro_rate = self.hydro.rhs(ions)
            rho = ions[..., 0]
            safe_rho = jnp.maximum(rho, self.hydro.density_floor)
            velocity_rate = (hydro_rate[..., 1:4] - velocity * hydro_rate[..., :1]) / safe_rho[..., None]
            advective_rate = jnp.einsum("...j,...ij->...i", velocity, gradient)
            material_acceleration = velocity_rate + advective_rate
        else:
            material_acceleration = jnp.zeros_like(velocity)
        return {
            "ion_velocity": velocity,
            "ion_velocity_gradient": gradient,
            "ion_material_acceleration": material_acceleration,
        }

    @staticmethod
    def _rate(args: dict | None, key: str, t: float, template: Array) -> Array:
        if not args or key not in args:
            return jnp.zeros_like(template)
        value = args[key]
        value = value(t) if callable(value) else value
        return jnp.broadcast_to(jnp.asarray(value), template.shape)

    def _exchange(
        self,
        t: float,
        flm: Array,
        ions: Array,
        args: dict | None,
        source_dt: float | None = None,
    ) -> tuple[Array, Array]:
        if self.exchange is None and self.pressure is None:
            return flm, ions
        source_dt = self.dt if source_dt is None else float(source_dt)
        template = ions[..., 0]
        momentum_rate = self._rate(args, "ei_momentum_relaxation_rate", t, template)
        temperature_rate = self._rate(args, "ei_temperature_relaxation_rate", t, template)
        df1 = jnp.zeros_like(flm)
        di1 = jnp.zeros_like(ions)
        if self.exchange is not None:
            exchange_df1, exchange_di1, _diagnostics = self.exchange(
                flm,
                ions,
                momentum_relaxation_rate=momentum_rate,
                temperature_relaxation_rate=temperature_rate,
            )
            df1 += exchange_df1
            di1 += exchange_di1
        if self.pressure is not None:
            pressure_df1, pressure_di1, _diagnostics = self.pressure(flm, ions)
            df1 += pressure_df1
            di1 += pressure_di1
        midpoint_i = ions + 0.5 * source_dt * di1
        initial_velocity = ions[..., 1:4] / ions[..., :1]
        midpoint_velocity = midpoint_i[..., 1:4] / midpoint_i[..., :1]
        half_frame_change = midpoint_velocity - initial_velocity
        midpoint_f = self.frame_remap(flm + 0.5 * source_dt * df1, half_frame_change)
        df2 = jnp.zeros_like(flm)
        di2 = jnp.zeros_like(ions)
        if self.exchange is not None:
            exchange_df2, exchange_di2, _diagnostics = self.exchange(
                midpoint_f,
                midpoint_i,
                momentum_relaxation_rate=momentum_rate,
                temperature_relaxation_rate=temperature_rate,
            )
            df2 += exchange_df2
            di2 += exchange_di2
        if self.pressure is not None:
            pressure_df2, pressure_di2, _diagnostics = self.pressure(midpoint_f, midpoint_i)
            df2 += pressure_df2
            di2 += pressure_di2
        final_i = ions + source_dt * di2
        final_velocity = final_i[..., 1:4] / final_i[..., :1]
        base_in_midpoint_frame = self.frame_remap(flm, half_frame_change)
        final_f_in_midpoint_frame = base_in_midpoint_frame + source_dt * df2
        final_f = self.frame_remap(final_f_in_midpoint_frame, final_velocity - midpoint_velocity)
        return final_f, final_i

    def __call__(self, t: float, state: dict[str, Array], args: dict | None = None) -> dict[str, Array]:
        if "ions" not in state:
            raise ValueError("coupled state must contain the ion conserved array under 'ions'")
        flm = real_to_complex(state["flm"]) if self.electron_step.real_storage else state["flm"]
        midpoint_ions = self._hydro_half_step(state["ions"])
        flm, midpoint_ions = self._exchange(t + 0.25 * self.dt, flm, midpoint_ions, args, 0.5 * self.dt)
        electron_args = {**({} if args is None else args), **self.ion_kinematics(midpoint_ions)}
        electron_state = {
            "flm": complex_to_real(flm) if self.electron_step.real_storage else flm,
            "e": state["e"],
            "b": state["b"],
        }
        if "current_projection_energy" in state:
            electron_state["current_projection_energy"] = state["current_projection_energy"]
        advanced_electrons = self.electron_step(t, electron_state, electron_args)

        flm = (
            real_to_complex(advanced_electrons["flm"]) if self.electron_step.real_storage else advanced_electrons["flm"]
        )
        flm, midpoint_ions = self._exchange(t + 0.75 * self.dt, flm, midpoint_ions, args, 0.5 * self.dt)
        final_ions = self._hydro_half_step(midpoint_ions)
        final_args = {**({} if args is None else args), **self.ion_kinematics(final_ions)}
        hidden_dndz = self.electron_step._hidden_dndz(
            t + self.dt,
            final_args,
            advanced_electrons["b"][..., 0],
        )
        advanced_electrons["e"], _terms = self.electron_step.electric_field(
            flm,
            advanced_electrons["b"],
            final_args,
            hidden_dndz=hidden_dndz,
        )
        advanced_electrons["flm"] = complex_to_real(flm) if self.electron_step.real_storage else flm
        return {**advanced_electrons, "ions": final_ions}


def coupled_invariants(
    flm: Array,
    ions: Array,
    magnetic_field: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    *,
    dx: float,
    dy: float,
    ion_mass: float,
    ion_charge: float,
    light_speed: float,
    electron_mass: float = 1.0,
    current_projection_energy: Array | None = None,
) -> dict[str, Array]:
    """Return global number, momentum, and energy histories.

    ``flm`` is expressed in peculiar velocity. Electron lab-frame momentum and
    kinetic energy therefore include the ion-frame translation terms.
    ``current_projection_energy`` records work introduced by enforcing the
    quasistatic Ampere moment; subtracting it exposes the conservative energy
    defect of the coupled evolution itself.
    Leading batch dimensions (normally time) are preserved.
    """

    electron_density = density(flm, layout, v, dv)
    ion_density = ions[..., 0] / float(ion_mass)
    ion_velocity = ions[..., 1:4] / ions[..., :1]
    relative_momentum = electron_momentum_density(flm, layout, v, dv, electron_mass)
    electron_momentum = relative_momentum + float(electron_mass) * electron_density[..., None] * ion_velocity
    electron_energy = electron_kinetic_energy_density(flm, layout, v, dv, electron_mass)
    electron_energy += jnp.sum(ion_velocity * relative_momentum, axis=-1)
    electron_energy += 0.5 * float(electron_mass) * electron_density * jnp.sum(ion_velocity**2, axis=-1)

    cell_area = float(dx) * float(dy)
    scalar_axes = (-2, -1)
    vector_axes = (-3, -2)
    electron_number = cell_area * jnp.sum(electron_density, axis=scalar_axes)
    ion_number = cell_area * jnp.sum(ion_density, axis=scalar_axes)
    total_momentum = cell_area * jnp.sum(electron_momentum + ions[..., 1:4], axis=vector_axes)
    electron_energy = cell_area * jnp.sum(electron_energy, axis=scalar_axes)
    ion_energy = cell_area * jnp.sum(ions[..., 4], axis=scalar_axes)
    magnetic_energy = (
        0.5
        * float(light_speed) ** 2
        * cell_area
        * jnp.sum(
            magnetic_field**2,
            axis=(-3, -2, -1),
        )
    )
    quasineutrality_linf = jnp.max(
        jnp.abs(electron_density - float(ion_charge) * ion_density),
        axis=scalar_axes,
    )
    total_energy = electron_energy + ion_energy + magnetic_energy
    if current_projection_energy is None:
        projection_energy = jnp.zeros_like(total_energy)
    else:
        projection_energy = cell_area * jnp.sum(current_projection_energy, axis=scalar_axes)
    return {
        "electron_number": electron_number,
        "ion_number": ion_number,
        "quasineutrality_linf": quasineutrality_linf,
        "total_momentum": total_momentum,
        "electron_energy": electron_energy,
        "ion_energy": ion_energy,
        "magnetic_energy": magnetic_energy,
        "total_energy": total_energy,
        "current_projection_energy": projection_energy,
        "accounted_total_energy": total_energy - projection_energy,
    }
