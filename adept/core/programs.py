"""JAX program adapters for continuous and discrete ADEPT systems."""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from .contracts import RawResult


class DiffraxProgram(eqx.Module):
    """Execute a true continuous system through Diffrax."""

    system: Any
    solver: Any
    observation: Any
    save_times: jax.Array
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __call__(self, params: Any, state: Any, inputs: Any, key: jax.Array) -> RawResult:
        del key

        def rhs(t, current_state, runtime_inputs):
            return self.system.rhs(t, current_state, params, runtime_inputs)

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(rhs),
            solver=self.solver,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            y0=state,
            args=inputs,
            max_steps=self.max_steps,
            saveat=diffrax.SaveAt(
                subs={
                    "observations": diffrax.SubSaveAt(ts=self.save_times, fn=self.observation),
                    "final_state": diffrax.SubSaveAt(t1=True),
                }
            ),
        )
        final_state = jax.tree.map(lambda value: value[0], solution.ys["final_state"])
        return RawResult(
            final_state=final_state,
            observations=solution.ys["observations"],
            times=solution.ts["observations"],
            status=solution.result,
            stats=solution.stats,
        )


class ScanProgram(eqx.Module):
    """Execute a complete next-state map for a fixed number of steps."""

    system: Any
    t0: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)

    def __call__(self, params: Any, state: Any, inputs: Any, key: jax.Array) -> RawResult:
        def advance(step, current_state):
            step_key = jax.random.fold_in(key, step)
            return self.system.step(step, current_state, params, inputs, step_key)

        final_state = jax.lax.fori_loop(0, self.num_steps, advance, state)
        return RawResult(
            final_state=final_state,
            observations=(),
            times=jnp.empty((0,)),
            status=diffrax.RESULTS.successful,
            stats={"num_steps": jnp.asarray(self.num_steps)},
        )


__all__ = ["DiffraxProgram", "ScanProgram"]
