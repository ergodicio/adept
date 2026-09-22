"""JAX program adapters for continuous and discrete ADEPT systems."""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .contracts import RawResult
from .observations import ObservationPlan, ObservationReduction, ScheduleKind
from .observations_jax import reduce_observation, validate_observation_spec


class _ObservationAdapter(eqx.Module):
    function: Any
    reduction: ObservationReduction = eqx.field(static=True)

    def __call__(self, t: Any, state: Any, inputs: Any) -> Any:
        return reduce_observation(self.function(t, state, inputs), self.reduction)


class DiffraxProgram(eqx.Module):
    """Execute a true continuous system through Diffrax."""

    system: Any
    solver: Any
    observations: tuple[_ObservationAdapter, ...]
    observation_times: tuple[jax.Array, ...]
    observation_names: tuple[str, ...] = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    @classmethod
    def from_observation_plan(
        cls,
        *,
        system: Any,
        solver: Any,
        plan: ObservationPlan,
        state: Any,
        inputs: Any,
        t0: float,
        t1: float,
        dt0: float,
        max_steps: int,
    ) -> DiffraxProgram:
        """Validate and compile a host plan into Diffrax ``SaveAt`` inputs."""

        adapters = []
        schedules = []
        names = []
        for spec in plan.observations:
            if spec.schedule.kind is not ScheduleKind.TIME:
                raise ValueError(f"continuous observation {spec.name!r} requires a time schedule")
            validate_observation_spec(spec, t=t0, state=state, inputs=inputs)
            points = tuple(float(point) for point in spec.schedule.retained_points(spec.retention))
            if points[0] < t0 or points[-1] > t1:
                raise ValueError(f"observation {spec.name!r} is scheduled outside [{t0}, {t1}]")
            adapters.append(_ObservationAdapter(spec.function, spec.reduction))
            schedules.append(jnp.asarray(points))
            names.append(spec.name)
        return cls(
            system=system,
            solver=solver,
            observations=tuple(adapters),
            observation_times=tuple(schedules),
            observation_names=tuple(names),
            t0=t0,
            t1=t1,
            dt0=dt0,
            max_steps=max_steps,
        )

    def __call__(self, params: Any, state: Any, inputs: Any, key: jax.Array) -> RawResult:
        del key

        def rhs(t, current_state, runtime_inputs):
            return self.system.rhs(t, current_state, params, runtime_inputs)

        save_subs = {
            name: diffrax.SubSaveAt(ts=times, fn=observation)
            for name, observation, times in zip(
                self.observation_names,
                self.observations,
                self.observation_times,
                strict=True,
            )
        }
        save_subs["__final_state__"] = diffrax.SubSaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(rhs),
            solver=self.solver,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            y0=state,
            args=inputs,
            max_steps=self.max_steps,
            saveat=diffrax.SaveAt(subs=save_subs),
        )
        final_state = jax.tree.map(lambda value: value[0], solution.ys["__final_state__"])
        return RawResult(
            final_state=final_state,
            observations={name: solution.ys[name] for name in self.observation_names},
            times={name: solution.ts[name] for name in self.observation_names},
            status=solution.result,
            stats=solution.stats,
        )


class ScanProgram(eqx.Module):
    """Execute a complete next-state map for a fixed number of steps."""

    system: Any
    t0: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    observations: tuple[_ObservationAdapter, ...] = ()
    observation_steps: tuple[tuple[int, ...], ...] = eqx.field(static=True, default=())
    observation_names: tuple[str, ...] = eqx.field(static=True, default=())
    observation_times: tuple[jax.Array, ...] = ()
    observation_fractions: tuple[jax.Array | None, ...] = ()
    samples_per_step: tuple[int, ...] = eqx.field(static=True, default=())

    @classmethod
    def from_observation_plan(
        cls,
        *,
        system: Any,
        plan: ObservationPlan,
        state: Any,
        inputs: Any,
        t0: float,
        dt: float,
        num_steps: int,
        interpolate: bool = False,
    ) -> ScanProgram:
        """Compile bounded buffers, optionally interpolating time-scheduled states.

        ``interpolate=True`` accepts physical times between steps using linear
        interpolation of the complete state, as in the legacy ``Stepper``. This
        requires floating/complex state leaves and does not change the step map.
        """

        adapters = []
        schedules = []
        names = []
        times = []
        fractions = []
        counts = []
        for spec in plan.observations:
            if spec.schedule.kind is not ScheduleKind.STEP and not interpolate:
                raise ValueError(f"discrete observation {spec.name!r} requires a step schedule")
            validate_observation_spec(spec, t=t0, state=state, inputs=inputs)
            retained = spec.schedule.retained_points(spec.retention)
            if spec.schedule.kind is ScheduleKind.TIME:
                if dt <= 0:
                    raise ValueError("interpolated observations require dt > 0")
                if any(not jnp.issubdtype(leaf.dtype, jnp.inexact) for leaf in jax.tree.leaves(state)):
                    raise TypeError("interpolated observations require floating or complex state leaves")
                sample_times = np.asarray(retained, dtype=float)
                if sample_times[0] < t0 or sample_times[-1] > t0 + dt * num_steps:
                    raise ValueError(f"observation {spec.name!r} is scheduled outside the simulation interval")
                positions = (sample_times - t0) / dt
                # Avoid assigning an on-grid sample to the following step due to
                # division roundoff; retain the requested physical output times.
                nearest = np.rint(positions)
                positions = np.where(np.isclose(positions, nearest, rtol=0, atol=1e-10), nearest, positions)
                points = tuple(int(point) for point in np.ceil(positions))
                weights = np.where(np.asarray(points) == 0, 0.0, positions - np.asarray(points) + 1.0)
            else:
                points = tuple(int(point) for point in retained)
                sample_times = t0 + dt * np.asarray(points)
                weights = None
            if points[-1] > num_steps:
                raise ValueError(f"observation {spec.name!r} step {points[-1]} exceeds final step {num_steps}")
            adapters.append(_ObservationAdapter(spec.function, spec.reduction))
            schedules.append(points)
            names.append(spec.name)
            if interpolate:
                times.append(jnp.asarray(sample_times))
                fractions.append(None if weights is None else jnp.asarray(weights))
                counts.append(int(np.max(np.unique(points, return_counts=True)[1])))
        return cls(
            system=system,
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            observation_times=tuple(times),
            observation_fractions=tuple(fractions),
            samples_per_step=tuple(counts),
            observations=tuple(adapters),
            observation_steps=tuple(schedules),
            observation_names=tuple(names),
        )

    def __call__(self, params: Any, state: Any, inputs: Any, key: jax.Array) -> RawResult:
        examples = tuple(observation(self.t0, state, inputs) for observation in self.observations)
        buffers = tuple(
            jax.tree.map(
                lambda leaf, sample_steps=steps: jnp.empty((len(sample_steps), *leaf.shape), dtype=leaf.dtype),
                example,
            )
            for example, steps in zip(examples, self.observation_steps, strict=True)
        )

        def record(step, previous_state, current_state, current_buffers):
            time = self.t0 + step * self.dt
            updated = []
            for index, (observation, steps, buffer) in enumerate(
                zip(
                    self.observations,
                    self.observation_steps,
                    current_buffers,
                    strict=True,
                )
            ):
                if self.observation_times and self.observation_fractions[index] is not None:
                    sample_times = self.observation_times[index]
                    weights = self.observation_fractions[index]
                    sample_steps = jnp.asarray(steps)
                    first = jnp.searchsorted(sample_steps, step, side="left")
                    for offset in range(self.samples_per_step[index]):
                        slot = first + offset
                        safe_slot = jnp.minimum(slot, len(steps) - 1)

                        def write_interpolated(
                            target,
                            weight=weights[safe_slot],
                            observed=observation,
                            sample_time=sample_times[safe_slot],
                            target_slot=safe_slot,
                        ):
                            interpolated = jax.tree.map(
                                lambda left, right: left + weight * (right - left), previous_state, current_state
                            )
                            value = observed(sample_time, interpolated, inputs)
                            return jax.tree.map(lambda leaf, sample: leaf.at[target_slot].set(sample), target, value)

                        buffer = jax.lax.cond(
                            (slot < len(steps)) & (sample_steps[safe_slot] == step),
                            write_interpolated,
                            lambda target: target,
                            buffer,
                        )
                    updated.append(buffer)
                    continue
                matches = jnp.asarray(steps) == step
                slot = jnp.argmax(matches)

                def write(target, observed=observation, target_slot=slot):
                    value = observed(time, current_state, inputs)
                    return jax.tree.map(
                        lambda leaf, sample: leaf.at[target_slot].set(sample),
                        target,
                        value,
                    )

                updated.append(jax.lax.cond(jnp.any(matches), write, lambda target: target, buffer))
            return tuple(updated)

        buffers = record(jnp.asarray(0), state, state, buffers)

        def advance(step, carry):
            current_state, current_buffers = carry
            step_key = jax.random.fold_in(key, step)
            next_state = self.system.step(step, current_state, params, inputs, step_key)
            next_buffers = record(step + 1, current_state, next_state, current_buffers)
            return next_state, next_buffers

        final_state, buffers = jax.lax.fori_loop(0, self.num_steps, advance, (state, buffers))
        return RawResult(
            final_state=final_state,
            observations=dict(zip(self.observation_names, buffers, strict=True)),
            times={
                name: self.observation_times[index]
                if self.observation_times
                else self.t0 + self.dt * jnp.asarray(steps)
                for index, (name, steps) in enumerate(zip(self.observation_names, self.observation_steps, strict=True))
            },
            status=diffrax.RESULTS.successful,
            stats={"num_steps": jnp.asarray(self.num_steps)},
        )


__all__ = ["DiffraxProgram", "ScanProgram"]
