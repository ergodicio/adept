"""Coupled 2D3P spherical-harmonic Vlasov--Maxwell vector field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array
from jax.sharding import Mesh

from adept.vfp2d.collisions import CollisionStep
from adept.vfp2d.harmonics import (
    HarmonicLayout,
    HouLiFilter2D,
    TzoufrasVlasov,
    complex_to_real,
    conservative_f00_positivity,
    current,
    density,
    periodic_central_derivative,
    real_to_complex,
)
from adept.vfp2d.ohm import KineticOhm2D, project_current_moment

if TYPE_CHECKING:
    from adept.vfp2d.moving_frame import IonFrameVlasov


def _ib_gate(t: float, args: dict | None) -> Array:
    """Return the optional smooth temporal envelope for IB heating."""

    if not args:
        return jnp.asarray(1.0)
    width = jnp.asarray(args.get("ib_switch_width", 0.0))
    gate = jnp.asarray(1.0)
    if "ib_t_on" in args:
        t_on = jnp.asarray(args["ib_t_on"])
        sharp_on = jnp.where(t >= t_on, 1.0, 0.0)
        smooth_on = 0.5 * (1.0 + jnp.tanh((t - t_on) / jnp.maximum(width, 1e-30)))
        gate = gate * jnp.where(width > 0.0, smooth_on, sharp_on)
    if "ib_t_off" in args:
        t_off = jnp.asarray(args["ib_t_off"])
        sharp_off = jnp.where(t < t_off, 1.0, 0.0)
        smooth_off = 0.5 * (1.0 - jnp.tanh((t - t_off) / jnp.maximum(width, 1e-30)))
        gate = gate * jnp.where(width > 0.0, smooth_off, sharp_off)
    return gate


class Maxwell2D:
    """Full three-component Maxwell curl operator with ``d/dz = 0``."""

    def __init__(
        self,
        kx: Array,
        ky: Array,
        c: float,
        *,
        dx: float | None = None,
        dy: float | None = None,
        mesh: Mesh | None = None,
    ):
        self.kx = jnp.asarray(kx)
        self.ky = jnp.asarray(ky)
        self.c2 = float(c) ** 2
        self.dx = None if dx is None else float(dx)
        self.dy = None if dy is None else float(dy)
        self.mesh = mesh

    def ddx(self, a: Array) -> Array:
        if self.dx is not None:
            return periodic_central_derivative(a, self.dx, axis=0, mesh=self.mesh)
        return jnp.fft.ifft(1j * self.kx[:, None] * jnp.fft.fft(a, axis=0), axis=0).real

    def ddy(self, a: Array) -> Array:
        if self.dy is not None:
            return periodic_central_derivative(a, self.dy, axis=1)
        return jnp.fft.ifft(1j * self.ky[None, :] * jnp.fft.fft(a, axis=1), axis=1).real

    def curl(self, a: Array) -> Array:
        ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
        return jnp.stack((self.ddy(az), -self.ddx(az), self.ddx(ay) - self.ddy(ax)), axis=-1)

    def __call__(self, electric_field: Array, magnetic_field: Array, plasma_current: Array) -> tuple[Array, Array]:
        dedt = self.c2 * self.curl(magnetic_field) - plasma_current
        dbdt = -self.curl(electric_field)
        return dedt, dbdt


class SpectralPoisson2D:
    """Periodic initial Gauss-law solve for ``div(E)=rho``."""

    def __init__(self, kx: Array, ky: Array):
        self.kx = jnp.asarray(kx)[:, None]
        self.ky = jnp.asarray(ky)[None, :]
        k2 = self.kx**2 + self.ky**2
        self.inv_k2 = jnp.where(k2 > 0, 1.0 / k2, 0.0)

    def __call__(self, charge_density: Array) -> Array:
        rho_k = jnp.fft.fft2(charge_density)
        phi_k = self.inv_k2 * rho_k
        ex = jnp.fft.ifft2(-1j * self.kx * phi_k).real
        ey = jnp.fft.ifft2(-1j * self.ky * phi_k).real
        return jnp.stack((ex, ey, jnp.zeros_like(ex)), axis=-1)


class VlasovMaxwell:
    """Explicit collisionless RHS for packed arbitrary-``f_lm`` state arrays.

    The state is ``{"flm": ..., "e": ..., "b": ...}``.  Optional external
    fields and current drivers may be supplied through ``args`` with keys
    ``external_e``, ``external_b``, and ``driver_current``.  Each value may be
    either an array or a callable of time.
    """

    def __init__(
        self,
        vlasov: TzoufrasVlasov,
        maxwell: Maxwell2D,
        layout: HarmonicLayout,
        v: Array,
        dv: float,
        charge: float = -1.0,
        real_storage: bool = False,
        streaming_speed: Array | None = None,
    ):
        self.vlasov = vlasov
        self.maxwell = maxwell
        self.layout = layout
        self.v = jnp.asarray(v)
        self.dv = float(dv)
        self.charge = float(charge)
        self.real_storage = bool(real_storage)
        self.streaming_speed = self.v if streaming_speed is None else jnp.asarray(streaming_speed)

    @staticmethod
    def _arg(args: dict | None, key: str, t: float, template: Array) -> Array:
        if not args or key not in args:
            return jnp.zeros_like(template)
        value = args[key]
        return value(t) if callable(value) else value

    def __call__(self, t: float, state: dict[str, Array], args: dict | None = None) -> dict[str, Array]:
        e = state["e"]
        b = state["b"]
        total_e = e + self._arg(args, "external_e", t, e)
        total_b = b + self._arg(args, "external_b", t, b)
        flm = real_to_complex(state["flm"]) if self.real_storage else state["flm"]
        dfdt = self.vlasov(flm, total_e, total_b)
        plasma_current = current(
            flm,
            self.layout,
            self.v,
            self.dv,
            charge=self.charge,
            streaming_speed=self.streaming_speed,
        )
        driver_current = self._arg(args, "driver_current", t, e)
        dedt, dbdt = self.maxwell(e, b, plasma_current + driver_current)
        return {"flm": complex_to_real(dfdt) if self.real_storage else dfdt, "e": dedt, "b": dbdt}


class SplitStepVFP2D:
    """One second-order explicit Vlasov--Maxwell step with collision splitting.

    The return value is the advanced state (rather than a derivative), matching
    ADEPT's map-style ``Stepper`` interface. Collisions are applied in two half
    steps around an explicit midpoint Vlasov--Maxwell update.
    """

    def __init__(self, rhs: VlasovMaxwell, dt: float, collisions: CollisionStep | None = None):
        self.rhs = rhs
        self.dt = float(dt)
        self.collisions = collisions

    def _collide(self, t: float, state: dict[str, Array], args: dict | None, dt: float) -> dict[str, Array]:
        if self.collisions is None:
            return state
        z = 1.0 if not args else args.get("Z", 1.0)
        ni = 1.0 if not args else args.get("ni", 1.0)
        heating = {}
        if args:
            for key in ("D0_heating", "ib_vosc2", "ib_Z2ni_w0"):
                if key in args:
                    heating[key] = args[key]
        if "ib_vosc2" in heating:
            heating["ib_vosc2"] = heating["ib_vosc2"] * _ib_gate(t, args)
        flm = real_to_complex(state["flm"]) if self.rhs.real_storage else state["flm"]
        flm = self.collisions(flm, Z=z, ni=ni, dt=dt, **heating)
        if self.rhs.real_storage:
            flm = complex_to_real(flm)
        return {**state, "flm": flm}

    def __call__(self, t: float, state: dict[str, Array], args: dict | None = None) -> dict[str, Array]:
        state = self._collide(t, state, args, 0.5 * self.dt)
        k1 = self.rhs(t, state, args)
        midpoint = jtu.tree_map(lambda value, slope: value + 0.5 * self.dt * slope, state, k1)
        k2 = self.rhs(t + 0.5 * self.dt, midpoint, args)
        result = jtu.tree_map(lambda value, slope: value + self.dt * slope, state, k2)
        return self._collide(t + self.dt, result, args, 0.5 * self.dt)


class KineticOhmStep:
    """Long-timescale RK4 step using the inertia-free kinetic Ohm law.

    The quasistatic Ampere current is enforced by a minimal projection of the
    bulk-current moment of ``f1``. This removes light and plasma oscillations
    while preserving the velocity-dependent ``f1`` structure responsible for
    nonlocal heat flow. It is deliberately separate from the future fully
    implicit kinetic-current-response algorithm.
    """

    def __init__(
        self,
        vlasov: TzoufrasVlasov,
        maxwell: Maxwell2D,
        ohm: KineticOhm2D,
        layout: HarmonicLayout,
        v: Array,
        dv: float,
        dt: float,
        collisions: CollisionStep | None = None,
        real_storage: bool = False,
        enforce_f00_positivity: bool = False,
        spatial_filter: HouLiFilter2D | None = None,
        ion_frame: IonFrameVlasov | None = None,
    ):
        self.vlasov = vlasov
        self.maxwell = maxwell
        self.ohm = ohm
        self.layout = layout
        self.v = jnp.asarray(v)
        self.dv = float(dv)
        self.dt = float(dt)
        self.collisions = collisions
        self.real_storage = bool(real_storage)
        self.enforce_f00_positivity = bool(enforce_f00_positivity)
        self.spatial_filter = spatial_filter
        self.ion_frame = ion_frame

    def _positive_f00(self, flm: Array) -> Array:
        if not self.enforce_f00_positivity:
            return flm
        return conservative_f00_positivity(flm, self.layout, self.v, self.dv)

    def _filter(self, value: Array) -> Array:
        return value if self.spatial_filter is None else self.spatial_filter(value)

    def _collide(self, t: float, flm: Array, args: dict | None, dt: float) -> Array:
        if self.collisions is None:
            return flm
        z = 1.0 if not args else args.get("Z", 1.0)
        ni = 1.0 if not args else args.get("ni", 1.0)
        heating = {}
        if args:
            for key in ("D0_heating", "ib_vosc2", "ib_Z2ni_w0"):
                if key in args:
                    heating[key] = args[key]
        if "ib_vosc2" in heating:
            heating["ib_vosc2"] = heating["ib_vosc2"] * _ib_gate(t, args)
        return self.collisions(flm, Z=z, ni=ni, dt=dt, **heating)

    def _target_current(self, magnetic_field: Array) -> Array:
        return self.maxwell.c2 * self.maxwell.curl(magnetic_field)

    def _project(self, flm: Array, magnetic_field: Array) -> Array:
        return project_current_moment(
            flm,
            self.layout,
            self.v,
            self.dv,
            self._target_current(magnetic_field),
        )

    def electric_field(
        self,
        flm: Array,
        magnetic_field: Array,
        args: dict | None,
        *,
        hidden_dndz: Array,
    ) -> tuple[Array, dict[str, Array]]:
        """Return the laboratory electric field and resolved Ohm terms."""

        electric_field, terms = self.ohm(
            flm,
            magnetic_field,
            plasma_current=self._target_current(magnetic_field),
            hidden_dndz=hidden_dndz,
        )
        if self.ion_frame is None:
            return electric_field, terms
        if not args or "ion_velocity" not in args:
            raise ValueError("ion-frame kinetic stepping requires ion_velocity in args")
        ion_velocity = jnp.broadcast_to(jnp.asarray(args["ion_velocity"]), magnetic_field.shape)
        bulk = -jnp.cross(ion_velocity, magnetic_field)
        return electric_field + bulk, {"bulk": bulk, **terms}

    @staticmethod
    def _hidden_dndz(t: float, args: dict | None, template: Array) -> Array:
        if not args or "hidden_dndz" not in args:
            return jnp.zeros_like(template)
        source = jnp.broadcast_to(jnp.asarray(args["hidden_dndz"]), template.shape)
        if "hidden_gradient_t_off" not in args:
            return source
        t_off = jnp.asarray(args["hidden_gradient_t_off"])
        width = jnp.asarray(args.get("hidden_gradient_switch_width", 0.0))
        sharp_gate = jnp.where(t < t_off, 1.0, 0.0)
        smooth_gate = 0.5 * (1.0 - jnp.tanh((t - t_off) / jnp.maximum(width, 1e-30)))
        return source * jnp.where(width > 0.0, smooth_gate, sharp_gate)

    def _rates(self, t: float, flm: Array, magnetic_field: Array, args: dict | None) -> tuple[Array, Array, Array]:
        flm = self._project(flm, magnetic_field)
        hidden_dndz = self._hidden_dndz(t, args, magnetic_field[..., 0])
        electric_field, _terms = self.electric_field(
            flm,
            magnetic_field,
            args,
            hidden_dndz=hidden_dndz,
        )
        ne = density(flm, self.layout, self.v, self.dv)
        safe_ne = jnp.maximum(ne, jnp.finfo(ne.dtype).tiny)
        dfdz = hidden_dndz[..., None, None] * flm / safe_ne[..., None, None]
        if self.ion_frame is None:
            dfdt = self.vlasov(flm, electric_field, magnetic_field, dfdz=dfdz)
        else:
            ion_velocity = jnp.broadcast_to(jnp.asarray(args["ion_velocity"]), magnetic_field.shape)
            velocity_gradient = args.get("ion_velocity_gradient")
            material_acceleration = args.get("ion_material_acceleration")
            dfdt = self.ion_frame(
                flm,
                electric_field,
                magnetic_field,
                ion_velocity,
                velocity_gradient=velocity_gradient,
                material_acceleration=material_acceleration,
                dfdz=dfdz,
            )
        return dfdt, -self.maxwell.curl(electric_field), electric_field

    def __call__(self, t: float, state: dict[str, Array], args: dict | None = None) -> dict[str, Array]:
        flm = real_to_complex(state["flm"]) if self.real_storage else state["flm"]
        magnetic_field = state["b"]
        flm = self._positive_f00(self._collide(t, flm, args, 0.5 * self.dt))
        flm = self._project(flm, magnetic_field)

        df1, db1, _electric1 = self._rates(t, flm, magnetic_field, args)
        stage2_b = magnetic_field + 0.5 * self.dt * db1
        stage2_f = self._positive_f00(self._project(flm + 0.5 * self.dt * df1, stage2_b))
        df2, db2, _electric2 = self._rates(t + 0.5 * self.dt, stage2_f, stage2_b, args)

        stage3_b = magnetic_field + 0.5 * self.dt * db2
        stage3_f = self._positive_f00(self._project(flm + 0.5 * self.dt * df2, stage3_b))
        df3, db3, _electric3 = self._rates(t + 0.5 * self.dt, stage3_f, stage3_b, args)

        stage4_b = magnetic_field + self.dt * db3
        stage4_f = self._positive_f00(self._project(flm + self.dt * df3, stage4_b))
        df4, db4, _electric4 = self._rates(t + self.dt, stage4_f, stage4_b, args)

        result_b = magnetic_field + (self.dt / 6.0) * (db1 + 2.0 * db2 + 2.0 * db3 + db4)
        result_f = flm + (self.dt / 6.0) * (df1 + 2.0 * df2 + 2.0 * df3 + df4)
        result_f = self._positive_f00(self._project(result_f, result_b))
        result_f = self._positive_f00(self._collide(t + self.dt, result_f, args, 0.5 * self.dt))
        # Pseudospectral field products feed unresolved power into the grid
        # scale during long heated runs. Filter only configuration space, once
        # per full step, then restore the f00 and Ampere-moment invariants.
        result_b = jnp.real(self._filter(result_b))
        result_f = self._positive_f00(self._filter(result_f))
        result_f = self._project(result_f, result_b)
        hidden_dndz = self._hidden_dndz(t + self.dt, args, result_b[..., 0])
        result_e, _terms = self.electric_field(
            result_f,
            result_b,
            args,
            hidden_dndz=hidden_dndz,
        )
        if self.real_storage:
            result_f = complex_to_real(result_f)
        return {"flm": result_f, "e": result_e, "b": result_b}
