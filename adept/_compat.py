"""Compatibility adapters between ``ergoExo`` and prepared simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from diffrax import Solution

from adept.core import PreparedSimulation, SimulationSpec, run_prepared, solver_registry

_SUPPORTED_SOLVERS = frozenset({"tf-1d", "pic-1d"})


def _legacy_seed(config: dict[str, Any]) -> int:
    """Recover the primary seed used by legacy solver initialization."""

    if config["solver"] != "pic-1d":
        return 0

    for species in config.get("terms", {}).get("species", ()):
        for component_name in species.get("density_components", ()):
            component = config.get("density", {}).get(component_name, {})
            if "noise_seed" in component:
                return int(component["noise_seed"])
    return 0


def _tree_leaf_signature(tree: Any) -> tuple[Any, tuple[int, ...]]:
    return jax.tree.structure(tree), tuple(id(leaf) for leaf in jax.tree.leaves(tree))


def _states_equal(left: Any, right: Any) -> bool:
    if jax.tree.structure(left) != jax.tree.structure(right):
        return False
    return all(
        np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _config_fallback_reason(config: dict[str, Any]) -> str | None:
    solver = config["solver"]
    if solver == "tf-1d":
        save_names = set(config.get("save", {}))
        unsupported = sorted(save_names - {"t", "x", "kx"})
        if unsupported:
            return f"tf-1d saves {unsupported!r} are not supported by the prepared compatibility path"
        if "t" not in save_names or not save_names.intersection({"x", "kx"}):
            return "tf-1d full-state saves are not supported by the prepared compatibility path"
    return None


@dataclass(frozen=True, slots=True)
class LegacyPreparedExecution:
    """Run a prepared simulation while preserving the legacy result contract."""

    prepared: PreparedSimulation[Any, Any, Any, Any, Any]
    key: Any
    solver: str
    t0: Any
    t1: Any
    max_steps: Any
    state_leaf_signature: tuple[Any, tuple[int, ...]]
    args_leaf_signature: tuple[Any, tuple[int, ...]]

    @classmethod
    def try_create(
        cls,
        config: dict[str, Any],
        *,
        legacy_module: Any,
        custom_module: bool,
    ) -> tuple[LegacyPreparedExecution | None, str | None]:
        """Create an adapter when the new path can preserve legacy behavior."""

        if custom_module:
            return None, "custom ADEPTModule injection requires the legacy execution path"

        solver = config["solver"]
        if solver not in _SUPPORTED_SOLVERS:
            return None, f"solver {solver!r} is not yet enabled for the prepared compatibility path"

        if reason := _config_fallback_reason(config):
            return None, reason

        seed = _legacy_seed(config)
        try:
            prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=seed)
        except ValueError as error:
            return None, f"the prepared {solver} builder rejected this legacy configuration: {error}"

        if not _states_equal(prepared.state, legacy_module.state):
            return None, "prepared initialization did not reproduce the legacy initial state"

        # Keep one initial-state tree alive and make subsequent replacement detectable.
        legacy_module.state = prepared.state
        time_quantities = legacy_module.time_quantities
        return (
            cls(
                prepared=prepared,
                key=jax.random.key(seed),
                solver=solver,
                t0=time_quantities["t0"],
                t1=time_quantities["t1"],
                max_steps=time_quantities["max_steps"],
                state_leaf_signature=_tree_leaf_signature(legacy_module.state),
                args_leaf_signature=_tree_leaf_signature(legacy_module.args),
            ),
            None,
        )

    def fallback_reason(self, *, state: Any, legacy_args: Any, modules: Any, args: Any) -> str | None:
        """Explain why one invocation cannot use the prepared execution path."""

        if args is not None:
            return "explicit legacy args require the legacy execution path"
        if modules is not None and (not isinstance(modules, dict) or modules):
            return "trainable legacy modules require the legacy execution path"
        if _tree_leaf_signature(state) != self.state_leaf_signature:
            return "legacy state replacement requires the legacy execution path"
        if _tree_leaf_signature(legacy_args) != self.args_leaf_signature:
            return "legacy args replacement requires the legacy execution path"
        return None

    def execute(self) -> dict[str, Solution]:
        """Execute via the host runtime and restore ``output['solver result']``."""

        completed = run_prepared(self.prepared, key=self.key)
        raw_result = completed.raw_result
        if self.solver == "tf-1d":
            # TF1D historically used one SaveAt schedule shared by all saved trees.
            times = next(iter(raw_result.times.values()))
            stats = raw_result.stats
        else:
            # PIC1D historically used named SubSaveAt schedules.
            times = raw_result.times
            num_steps = raw_result.stats["num_steps"]
            stats = {
                "max_steps": self.max_steps,
                "num_accepted_steps": num_steps,
                "num_rejected_steps": num_steps * 0,
                "num_steps": num_steps,
            }

        solution = Solution(
            t0=self.t0,
            t1=self.t1,
            ts=times,
            ys=raw_result.observations,
            interpolation=None,
            stats=stats,
            result=raw_result.status,
            solver_state=None,
            controller_state=None,
            made_jump=None,
            event_mask=None,
        )
        return {"solver result": solution}


__all__ = ["LegacyPreparedExecution"]
