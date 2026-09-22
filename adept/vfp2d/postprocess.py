"""Host-side VFP2D datasets and diagnostics, shared by both execution APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from adept.normalization import PlasmaNormalization
from adept.vfp2d.coupling import CoupledIonKineticStep, coupled_invariants
from adept.vfp2d.distributed import SpatialSharding
from adept.vfp2d.grid import Grid
from adept.vfp2d.harmonics import (
    HarmonicLayout,
    current,
    density,
    nernst_velocity,
    real_to_complex,
    scalar_velocity_moment,
    tensor_velocity_moment,
)
from adept.vfp2d.hydro import conserved_to_primitive
from adept.vfp2d.ohm import KineticOhm2D
from adept.vfp2d.reservoir import SOURCE_INVARIANTS, DrivenReservoirStep
from adept.vfp2d.vector_field import KineticOhmStep, Maxwell2D

if TYPE_CHECKING:
    from jax import Array

    from adept.vfp2d.preparation import VFP2DSetup


@dataclass(frozen=True)
class VFP2DPostProcessor:
    """Host metadata and operators needed to analyze an observed state history."""

    grid: Grid
    layout: HarmonicLayout
    plasma_norm: PlasmaNormalization
    args: dict[str, Any]
    _streaming_speed: Array
    _maxwell: Maxwell2D
    _kinetic_ohm: KineticOhm2D | None
    _kinetic_step: KineticOhmStep | None
    _coupled_step: CoupledIonKineticStep | None
    _reservoir_step: DrivenReservoirStep | None
    spatial_sharding: SpatialSharding | None
    field_mode: str
    ion_fluid_active: bool
    ion_mass: float
    ion_gamma: float
    ion_charge: float
    n_panels: int

    @classmethod
    def from_setup(cls, setup: VFP2DSetup) -> VFP2DPostProcessor:
        return cls(
            grid=setup.grid,
            layout=setup.layout,
            plasma_norm=setup.plasma_norm,
            args=setup.args,
            _streaming_speed=setup._streaming_speed,
            _maxwell=setup._maxwell,
            _kinetic_ohm=setup._kinetic_ohm,
            _kinetic_step=setup._kinetic_step,
            _coupled_step=setup._coupled_step,
            _reservoir_step=setup._reservoir_step,
            spatial_sharding=setup.spatial_sharding,
            field_mode=setup.field_mode,
            ion_fluid_active=setup.ion_fluid_active,
            ion_mass=setup.ion_mass,
            ion_gamma=setup.ion_gamma,
            ion_charge=float(setup.cfg["units"]["Z"]),
            n_panels=int(setup.cfg.get("output", {}).get("n_panels", 9)),
        )

    def post_process(self, run_output: dict, td: str) -> dict:
        from adept.vfp2d.plotting import add_reconnection_diagnostics, reconnection_metrics, save_artifacts

        result = run_output["solver result"]
        flm_jax = real_to_complex(result.ys["flm"])
        flm = np.asarray(flm_jax)
        ne = density(flm_jax, self.layout, self.grid.v, self.grid.dv)
        plasma_current = current(flm_jax, self.layout, self.grid.v, self.grid.dv, streaming_speed=self._streaming_speed)
        mean_v2 = scalar_velocity_moment(flm_jax, self.layout, self.grid.v, self.grid.dv, power=2)
        temperature_normalized = (2.0 / 3.0) * mean_v2 / self.plasma_norm.vth_norm() ** 2
        pressure_anisotropy = tensor_velocity_moment(flm_jax, self.layout, self.grid.v, self.grid.dv, power=0)
        v_nernst = nernst_velocity(flm_jax, self.layout, self.grid.v, self.grid.dv, plasma_current=plasma_current)
        coords = {
            "t": np.asarray(result.ts),
            "x": np.asarray(self.grid.x),
            "y": np.asarray(self.grid.y),
            "harmonic": np.arange(self.layout.size),
            "v": np.asarray(self.grid.v),
            "component": ["x", "y", "z"],
            "ell": ("harmonic", self.layout.ell),
            "m": ("harmonic", self.layout.m),
        }
        data_vars = {
            "flm_real": (("t", "x", "y", "harmonic", "v"), flm.real),
            "flm_imag": (("t", "x", "y", "harmonic", "v"), flm.imag),
            "e": (("t", "x", "y", "component"), np.asarray(result.ys["e"])),
            "b": (("t", "x", "y", "component"), np.asarray(result.ys["b"])),
            "ne": (("t", "x", "y"), np.asarray(ne)),
            "temperature": (("t", "x", "y"), np.asarray(temperature_normalized)),
            "current": (("t", "x", "y", "component"), np.asarray(plasma_current)),
            "v_nernst": (("t", "x", "y", "component"), np.asarray(v_nernst)),
            "pressure_anisotropy": (
                ("t", "x", "y", "component", "component_2"),
                np.asarray(pressure_anisotropy),
            ),
        }
        ampere_target_frames = []
        for frame in result.ys["b"]:
            ax, ay, az = frame[..., 0], frame[..., 1], frame[..., 2]
            ampere_target_frames.append(
                self._maxwell.c2
                * np.stack(
                    (
                        np.asarray(self._maxwell.ddy(az)),
                        -np.asarray(self._maxwell.ddx(az)),
                        np.asarray(self._maxwell.ddx(ay)) - np.asarray(self._maxwell.ddy(ax)),
                    ),
                    axis=-1,
                )
            )
        ampere_target = np.stack(ampere_target_frames)
        ampere_residual = np.asarray(plasma_current) - ampere_target
        magnetic_field_energy = (
            0.5 * self._maxwell.c2 * self.grid.dx * self.grid.dy * jnp.sum(result.ys["b"] ** 2, axis=(1, 2, 3))
        )
        data_vars.update(
            {
                "ampere_target_current": (
                    ("t", "x", "y", "component"),
                    ampere_target,
                ),
                "ampere_residual": (
                    ("t", "x", "y", "component"),
                    ampere_residual,
                ),
                "ampere_residual_linf": (
                    ("t",),
                    np.max(np.abs(ampere_residual), axis=(1, 2, 3)),
                ),
                "magnetic_field_energy": (("t",), np.asarray(magnetic_field_energy)),
            }
        )
        if self.field_mode in ("maxwell", "ampere"):
            electric_field_energy = (
                0.5
                * self._maxwell.relative_permittivity
                * self.grid.dx
                * self.grid.dy
                * jnp.sum(result.ys["e"] ** 2, axis=(1, 2, 3))
            )
            data_vars.update(
                {
                    "electric_field_energy": (("t",), np.asarray(electric_field_energy)),
                    "electromagnetic_field_energy": (
                        ("t",),
                        np.asarray(electric_field_energy + magnetic_field_energy),
                    ),
                }
            )
        if self.ion_fluid_active:
            ions_jax = result.ys["ions"]
            ions = np.asarray(ions_jax)
            ion_primitive = conserved_to_primitive(ions_jax, self.ion_gamma)
            ion_number_density = ions_jax[..., 0] / self.ion_mass
            ion_temperature_scale = 0.5 * self.plasma_norm.vth_norm() ** 2
            ion_temperature = ion_primitive[..., 4] / jnp.maximum(
                ion_number_density * ion_temperature_scale,
                jnp.finfo(ion_number_density.dtype).tiny,
            )
            invariants = coupled_invariants(
                flm_jax,
                ions_jax,
                result.ys["b"],
                self.layout,
                self.grid.v,
                self.grid.dv,
                dx=self.grid.dx,
                dy=self.grid.dy,
                ion_mass=self.ion_mass,
                ion_charge=self.ion_charge,
                light_speed=self.plasma_norm.speed_of_light_norm(),
                current_projection_energy=result.ys["current_projection_energy"],
            )
            f00 = jnp.real(flm_jax[..., self.layout.index(0, 0), :])
            negative_f00_mass = (
                4.0
                * jnp.pi
                * self.grid.dx
                * self.grid.dy
                * jnp.sum(jnp.maximum(-f00, 0.0) * self.grid.v**2, axis=(1, 2, 3))
                * self.grid.dv
            )
            harmonic_free_energy = (
                self.grid.dx
                * self.grid.dy
                * jnp.sum(jnp.abs(flm_jax) ** 2 * self.grid.v**2, axis=(1, 2, 4))
                * self.grid.dv
            )
            div_b_linf = jnp.stack(
                [
                    jnp.max(jnp.abs(self._maxwell.ddx(frame[..., 0]) + self._maxwell.ddy(frame[..., 1])))
                    for frame in result.ys["b"]
                ]
            )
            coords["ion_conserved"] = ["rho", "rho_ux", "rho_uy", "rho_uz", "energy"]
            data_vars.update(
                {
                    "ions": (("t", "x", "y", "ion_conserved"), ions),
                    "ni": (("t", "x", "y"), np.asarray(ion_number_density)),
                    "ion_velocity": (("t", "x", "y", "component"), np.asarray(ion_primitive[..., 1:4])),
                    "ion_pressure": (("t", "x", "y"), np.asarray(ion_primitive[..., 4])),
                    "ion_temperature": (("t", "x", "y"), np.asarray(ion_temperature)),
                    "electron_number": (("t",), np.asarray(invariants["electron_number"])),
                    "ion_number": (("t",), np.asarray(invariants["ion_number"])),
                    "quasineutrality_linf": (("t",), np.asarray(invariants["quasineutrality_linf"])),
                    "total_momentum": (("t", "component"), np.asarray(invariants["total_momentum"])),
                    "electron_energy": (("t",), np.asarray(invariants["electron_energy"])),
                    "ion_energy": (("t",), np.asarray(invariants["ion_energy"])),
                    "magnetic_energy": (("t",), np.asarray(invariants["magnetic_energy"])),
                    "total_energy": (("t",), np.asarray(invariants["total_energy"])),
                    "current_projection_energy": (
                        ("t",),
                        np.asarray(invariants["current_projection_energy"]),
                    ),
                    "accounted_total_energy": (("t",), np.asarray(invariants["accounted_total_energy"])),
                    "div_b_linf": (("t",), np.asarray(div_b_linf)),
                    "negative_f00_mass": (("t",), np.asarray(negative_f00_mass)),
                    "harmonic_free_energy": (("t", "harmonic"), np.asarray(harmonic_free_energy)),
                }
            )
        if self._reservoir_step is not None:
            data_vars["reservoir_magnetic_field_change"] = (
                ("t", "x", "y", "component"),
                np.asarray(result.ys["reservoir_magnetic_field_change"]),
            )
            for name in SOURCE_INVARIANTS:
                dims = ("t", "component") if name == "total_momentum" else ("t",)
                data_vars[f"reservoir_{name}"] = (dims, np.asarray(result.ys[f"reservoir_{name}"]))
            data_vars["source_accounted_total_energy"] = (
                ("t",),
                np.asarray(invariants["accounted_total_energy"] - result.ys["reservoir_total_energy"]),
            )
            data_vars["source_accounted_total_momentum"] = (
                ("t", "component"),
                np.asarray(invariants["total_momentum"] - result.ys["reservoir_total_momentum"]),
            )
            for name in ("electron_number", "ion_number"):
                data_vars[f"source_accounted_{name}"] = (
                    ("t",),
                    np.asarray(invariants[name] - result.ys[f"reservoir_{name}"]),
                )
        if self._kinetic_ohm is not None and self._maxwell is not None:
            ohm_keys = ["resistive", "hall", "nernst", "scalar_pressure", "tensor_pressure"]
            if self.ion_fluid_active:
                ohm_keys.insert(0, "bulk")
            ohm_history = {key: [] for key in ohm_keys}
            for index, time in enumerate(np.asarray(result.ts)):
                frame_flm = flm_jax[index]
                frame_b = result.ys["b"][index]
                if self.spatial_sharding is not None:
                    frame_flm = self.spatial_sharding.put(frame_flm)
                    frame_b = self.spatial_sharding.put(frame_b)
                hidden_dndz = KineticOhmStep._hidden_dndz(float(time), self.args, frame_b[..., 0])
                if self.ion_fluid_active:
                    frame_args = {
                        **self.args,
                        **self._coupled_step.ion_kinematics(result.ys["ions"][index]),
                    }
                else:
                    frame_args = self.args
                _electric, terms = self._kinetic_step.electric_field(
                    frame_flm,
                    frame_b,
                    frame_args,
                    hidden_dndz=hidden_dndz,
                )
                for key, value in terms.items():
                    ohm_history[key].append(value)
            for key, values in ohm_history.items():
                data_vars[f"ohm_{key}"] = (
                    ("t", "x", "y", "component"),
                    np.asarray(jnp.stack(values)),
                )

        ds = xr.Dataset(
            data_vars,
            coords={**coords, "component_2": ["x", "y", "z"]},
            attrs={
                "solver": "vfp-2d",
                "harmonic_convention": "Tzoufras JCP 230 (2011)",
                "length_unit_um": float(self.plasma_norm.L0.to("um").magnitude),
                "time_unit_ps": float(self.plasma_norm.tau.to("ps").magnitude),
                "density_unit_m3": float(self.plasma_norm.n0.to("1/m^3").magnitude),
                "velocity_unit_m_s": float(self.plasma_norm.v0.to("m/s").magnitude),
                "temperature_unit_ev": float(self.plasma_norm.T0.to("eV").magnitude),
                "magnetic_field_unit_t": float(
                    (self.plasma_norm.m0 / (self.plasma_norm.q0 * self.plasma_norm.tau)).to("tesla").magnitude
                ),
                "electric_field_unit_v_m": float(
                    (self.plasma_norm.m0 * self.plasma_norm.v0 / (self.plasma_norm.q0 * self.plasma_norm.tau))
                    .to("V/m")
                    .magnitude
                ),
                "light_speed_normalized": float(self.plasma_norm.speed_of_light_norm()),
                "temperature_energy_normalized": float(0.5 * self.plasma_norm.vth_norm() ** 2),
                "field_solver_mode": self.field_mode,
                "relative_permittivity": self._maxwell.relative_permittivity,
                "spatial_boundary_model": (
                    "periodic with explicit reservoir sources" if self._reservoir_step is not None else "periodic"
                ),
                "reservoir_budget_convention": (
                    "cumulative measured external injection; source_accounted energy also subtracts current projection "
                    "work; other external heating and numerical dissipation are not subtracted"
                    if self._reservoir_step is not None
                    else "not applicable"
                ),
                "ampere_constraint": "current = c^2 curl(magnetic_field); residual is current minus target",
                "ion_fluid_coupling": "kinetic-ohm magnetic force and moment exchange"
                if self.ion_fluid_active
                else "stationary",
                "coupled_energy_convention": (
                    "electron lab kinetic + ion total + magnetic; algebraic kinetic-Ohm E has no field energy"
                    if self.ion_fluid_active
                    else "not applicable"
                ),
            },
        )
        ds = add_reconnection_diagnostics(ds)
        if td:
            n_panels = self.n_panels
            save_artifacts(ds, td, n_panels=n_panels)
        return {"vfp2d": ds, "metrics": reconnection_metrics(ds)}
