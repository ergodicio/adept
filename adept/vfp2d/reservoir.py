"""Driven reservoirs on the periodic VFP2D interaction-region mesh.

These are explicit external sources, not open boundary conditions. The source
mixes particles in a common ion frame, drives magnetic *potential* so that its
field increment is a discrete curl, and measures every injected invariant.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from adept.vfp2d.coupling import CoupledIonKineticStep, coupled_invariants
from adept.vfp2d.exchange import electron_kinetic_energy_density, electron_momentum_density
from adept.vfp2d.harmonics import complex_to_real, current, density, real_to_complex
from adept.vfp2d.ohm import project_current_moment

SOURCE_INVARIANTS = (
    "electron_number",
    "ion_number",
    "total_momentum",
    "electron_energy",
    "ion_energy",
    "magnetic_energy",
    "total_energy",
)


def boundary_buffer(grid, *, x_width: float = 0.0, y_width: float = 0.0) -> Array:
    """Return a C2 buffer mask with exactly zero support in the interior.

    Widths include the smooth transition and must be smaller than a half-box.
    The quintic smoothstep is flat at both edges of each boundary band.
    """

    def band(axis, lower, upper, width):
        if not np.isfinite(width) or width < 0 or width >= 0.5 * (upper - lower):
            raise ValueError("buffer widths must be finite, nonnegative, and smaller than the half-box")
        if width == 0:
            return jnp.zeros_like(axis)
        distance = jnp.minimum(axis - lower, upper - axis)
        s = jnp.clip(1.0 - distance / width, 0.0, 1.0)
        return s**3 * (10.0 - 15.0 * s + 6.0 * s**2)

    x = band(grid.x, grid.xmin, grid.xmax, float(x_width))
    y = band(grid.y, grid.ymin, grid.ymax, float(y_width))
    if x_width == 0 and y_width == 0:
        raise ValueError("at least one reservoir buffer width must be positive")
    return 1.0 - (1.0 - x[:, None]) * (1.0 - y[None, :])


class DrivenReservoirStep:
    """Symmetric reservoir half-step / physical step / reservoir half-step.

    Target distributions live in the target ion frame. Both electron populations
    are remapped into the new ion frame before their convex mixture is formed.
    Remapping and current enforcement can change harmonic positivity; they are
    included in the measured source budget, never called conservative transport.
    The periodic mean magnetic flux is unchanged by the potential source.
    """

    def __init__(
        self,
        step: CoupledIonKineticStep,
        target: dict[str, Array],
        rate: Array,
        *,
        ion_mass: float,
        ion_charge: float,
        magnetic: bool = True,
    ):
        if not step.evolve_ions:
            raise ValueError("driven reservoirs require evolving ions")
        if ion_mass <= 0 or ion_charge <= 0:
            raise ValueError("reservoir ion mass and charge must be positive")
        expected = target["ions"].shape[:-1]
        rate_np = np.asarray(rate)
        if rate_np.shape != expected or not np.all(np.isfinite(rate_np)) or np.any(rate_np < 0):
            raise ValueError("reservoir rate must be a finite nonnegative spatial array")
        self.step = step
        self.electrons = step.electron_step
        if self.electrons.maxwell.dx is not None or self.electrons.maxwell.dy is not None:
            raise ValueError("reservoir magnetic potential requires unsharded spectral derivatives")
        self.target = {key: target[key] for key in ("flm", "b", "ions")}
        self.rate = jnp.asarray(rate)
        self.ion_mass = float(ion_mass)
        self.ion_charge = float(ion_charge)
        self.magnetic = bool(magnetic)
        self.dt = step.dt

    @staticmethod
    def initial_ledger(magnetic_field: Array) -> dict[str, Array]:
        ledger = {
            f"reservoir_{name}": jnp.zeros(3 if name == "total_momentum" else (), dtype=magnetic_field.dtype)
            for name in SOURCE_INVARIANTS
        }
        ledger["reservoir_magnetic_field_change"] = jnp.zeros_like(magnetic_field)
        return ledger

    def _complex(self, value: Array) -> Array:
        return real_to_complex(value) if self.electrons.real_storage else value

    def _invariants(self, state: dict[str, Array]) -> dict[str, Array]:
        return coupled_invariants(
            self._complex(state["flm"]),
            state["ions"],
            state["b"],
            self.electrons.layout,
            self.electrons.v,
            self.electrons.dv,
            dx=self.step.hydro.dx,
            dy=self.step.hydro.dy,
            ion_mass=self.ion_mass,
            ion_charge=self.ion_charge,
            light_speed=self.electrons.maxwell.c2**0.5,
            electron_mass=self.step.frame_remap.electron_mass,
        )

    def _potential(self, magnetic: Array) -> Array:
        """Coulomb-gauge A with curl(A)=B for resolved, zero-mean solenoidal B.

        Even-grid Nyquist derivatives vanish for real spectral fields. Their
        effective wavenumber must also vanish in this inverse, matching curl.
        Unresolved Nyquist and mean components are left unchanged by the drive.
        """

        kx = self.electrons.maxwell.kx
        ky = self.electrons.maxwell.ky
        if kx.size % 2 == 0:
            kx = kx.at[kx.size // 2].set(0.0)
        if ky.size % 2 == 0:
            ky = ky.at[ky.size // 2].set(0.0)
        k = jnp.stack(jnp.broadcast_arrays(kx[:, None], ky[None, :], jnp.zeros(magnetic.shape[:2])), axis=-1)
        k2 = jnp.sum(k**2, axis=-1)
        inverse = jnp.where(k2 > 0.0, 1.0 / jnp.where(k2 > 0.0, k2, 1.0), 0.0)
        b_hat = jnp.fft.fft2(magnetic, axes=(0, 1))
        a_hat = 1j * jnp.cross(k, b_hat) * inverse[..., None]
        return jnp.fft.ifft2(a_hat, axes=(0, 1)).real

    def _local_invariants(self, state: dict[str, Array]) -> dict[str, Array]:
        f = self._complex(state["flm"])
        layout, v, dv = self.electrons.layout, self.electrons.v, self.electrons.dv
        mass = self.step.frame_remap.electron_mass
        ne = density(f, layout, v, dv)
        ions = state["ions"]
        velocity = ions[..., 1:4] / ions[..., :1]
        peculiar_momentum = electron_momentum_density(f, layout, v, dv, mass)
        electron_energy = electron_kinetic_energy_density(f, layout, v, dv, mass)
        electron_energy += jnp.sum(velocity * peculiar_momentum, axis=-1)
        electron_energy += 0.5 * mass * ne * jnp.sum(velocity**2, axis=-1)
        magnetic_energy = 0.5 * self.electrons.maxwell.c2 * jnp.sum(state["b"] ** 2, axis=-1)
        return {
            "electron_number": ne,
            "ion_number": ions[..., 0] / self.ion_mass,
            "total_momentum": peculiar_momentum + mass * ne[..., None] * velocity + ions[..., 1:4],
            "electron_energy": electron_energy,
            "ion_energy": ions[..., 4],
            "magnetic_energy": magnetic_energy,
            "total_energy": electron_energy + ions[..., 4] + magnetic_energy,
        }

    def source(self, t: float, state: dict[str, Array], args: dict | None, duration: float) -> dict[str, Array]:
        before = self._local_invariants(state)
        alpha = -jnp.expm1(-self.rate * duration)
        old_ions = state["ions"]
        target_ions = self.target["ions"]
        ions = old_ions + alpha[..., None] * (target_ions - old_ions)
        old_u = old_ions[..., 1:4] / old_ions[..., :1]
        target_u = target_ions[..., 1:4] / target_ions[..., :1]
        new_u = ions[..., 1:4] / ions[..., :1]
        old_f = self.step.frame_remap(self._complex(state["flm"]), new_u - old_u)
        target_f = self.step.frame_remap(self._complex(self.target["flm"]), new_u - target_u)
        f = old_f + alpha[..., None, None] * (target_f - old_f)
        magnetic = state["b"]
        if self.magnetic:
            potential_change = self._potential(self.target["b"] - magnetic)
            magnetic = magnetic + self.electrons.maxwell.curl(alpha[..., None] * potential_change)
        # Dampen the pre-existing current residual only inside the source bands.
        # Otherwise a nominally zero drive would repair unrelated errors over
        # the entire mesh and misattribute their work to the external source.
        previous_residual = current(
            self._complex(state["flm"]), self.electrons.layout, self.electrons.v, self.electrons.dv
        ) - self.electrons._target_current(state["b"])
        target_current = self.electrons._target_current(magnetic) + (1.0 - alpha[..., None]) * previous_residual
        f = project_current_moment(f, self.electrons.layout, self.electrons.v, self.electrons.dv, target_current)
        source_args = {**({} if args is None else args), **self.step.ion_kinematics(ions)}
        hidden = self.electrons._hidden_dndz(t, source_args, ions[..., 0])
        electric, _terms = self.electrons.electric_field(f, magnetic, source_args, hidden_dndz=hidden)
        result = {
            **state,
            "ions": ions,
            "flm": complex_to_real(f) if self.electrons.real_storage else f,
            "b": magnetic,
            "e": electric,
            "reservoir_magnetic_field_change": state["reservoir_magnetic_field_change"] + (magnetic - state["b"]),
        }
        after = self._local_invariants(result)
        for name in SOURCE_INVARIANTS:
            key = f"reservoir_{name}"
            # Difference locally before summing: weak injection should not be
            # lost by subtracting two much larger global thermal energies.
            change = self.step.hydro.dx * self.step.hydro.dy * jnp.sum(after[name] - before[name], axis=(0, 1))
            result[key] = state[key] + change
        return result

    def __call__(self, t: float, state: dict[str, Array], args: dict | None = None) -> dict[str, Array]:
        first = self.source(t, state, args, 0.5 * self.dt)
        advanced = {**first, **self.step(t, first, args)}
        return self.source(t + self.dt, advanced, args, 0.5 * self.dt)
