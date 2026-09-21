"""Pure objective composition and explicit differentiation helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .contracts import ObjectiveResult, RawResult


class ParameterPartition(NamedTuple):
    """Complementary PyTrees containing selected parameters and fixed inputs."""

    trainable: Any
    frozen: Any


class ObjectiveRun(NamedTuple):
    """An objective evaluation paired with the simulation that produced it."""

    objective: ObjectiveResult
    simulation: RawResult


class ValueAndGradResult(NamedTuple):
    """Objective, gradient, and simulation output from one differentiated run."""

    objective: ObjectiveResult
    gradients: Any
    simulation: RawResult


def partition_parameters(values: Any, filter_spec: Any) -> ParameterPartition:
    """Split a PyTree into explicitly trainable and frozen complementary trees.

    ``filter_spec`` follows :func:`equinox.partition`: it may be a prefix boolean
    PyTree or a callable such as :func:`equinox.is_inexact_array`. Selected leaves
    must be floating-point or complex JAX arrays, and at least one leaf must be
    selected.
    """

    trainable, frozen = eqx.partition(values, filter_spec)
    leaves = jax.tree.leaves(trainable)
    if not leaves:
        raise ValueError("Parameter selection did not include any trainable array leaves")
    invalid = [leaf for leaf in leaves if not eqx.is_inexact_array(leaf)]
    if invalid:
        names = sorted({type(leaf).__name__ for leaf in invalid})
        raise TypeError("Selected parameters must be inexact JAX arrays; received " + ", ".join(names))
    return ParameterPartition(trainable=trainable, frozen=frozen)


def _validated_objective(value: Any) -> ObjectiveResult:
    if not isinstance(value, ObjectiveResult):
        raise TypeError("Objectives must return adept.ObjectiveResult(loss, metrics, aux)")
    loss = jnp.asarray(value.loss)
    if loss.shape != ():
        raise ValueError(f"Objective loss must be scalar; received shape {loss.shape}")
    if not isinstance(value.metrics, Mapping):
        raise TypeError("Objective metrics must be a mapping with a stable key structure")
    return ObjectiveResult(loss=loss, metrics=value.metrics, aux=value.aux)


def evaluate_objective(
    program: Any,
    objective: Any,
    params: Any,
    state: Any,
    inputs: Any,
    key: jax.Array,
) -> ObjectiveRun:
    """Run a program and evaluate a pure objective without differentiation."""

    simulation = program(params, state, inputs, key)
    evaluation = _validated_objective(objective(simulation, params, inputs))
    return ObjectiveRun(objective=evaluation, simulation=simulation)


def value_and_grad(
    program: Any,
    objective: Any,
    params: Any,
    state: Any,
    inputs: Any,
    key: jax.Array,
) -> ValueAndGradResult:
    """Differentiate an objective with respect to ``params`` only.

    Floating or complex arrays in ``state`` and ``inputs`` remain explicit runtime
    values but are never accidental differentiation targets. Use
    :func:`partition_parameters` before this call when selecting trainable leaves
    from a larger runtime PyTree.
    """

    def loss_fn(current_params):
        run = evaluate_objective(program, objective, current_params, state, inputs, key)
        evaluation = run.objective
        return evaluation.loss, (evaluation.metrics, evaluation.aux, run.simulation)

    (loss, (metrics, aux, simulation)), gradients = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
    evaluation = ObjectiveResult(loss=loss, metrics=metrics, aux=aux)
    return ValueAndGradResult(objective=evaluation, gradients=gradients, simulation=simulation)


class CallableObjective(eqx.Module):
    """Adapt a scalar pure callable to the structured objective protocol."""

    fn: Callable[[RawResult, Any, Any], Any]
    metric_name: str = eqx.field(static=True)

    def __init__(self, fn: Callable[[RawResult, Any, Any], Any], metric_name: str = "objective") -> None:
        if not metric_name:
            raise ValueError("metric_name must be non-empty")
        self.fn = fn
        self.metric_name = metric_name

    def __call__(self, result: RawResult, params: Any, inputs: Any) -> ObjectiveResult:
        loss = jnp.asarray(self.fn(result, params, inputs))
        return ObjectiveResult(loss=loss, metrics={self.metric_name: loss}, aux=())


class L2Penalty(eqx.Module):
    """Squared-L2 regularization over the explicit parameter PyTree."""

    coefficient: float = eqx.field(static=True)
    metric_name: str = eqx.field(static=True)

    def __init__(self, coefficient: float = 1.0, metric_name: str = "parameter_l2") -> None:
        if coefficient < 0.0:
            raise ValueError("coefficient must be non-negative")
        if not metric_name:
            raise ValueError("metric_name must be non-empty")
        self.coefficient = float(coefficient)
        self.metric_name = metric_name

    def __call__(self, result: RawResult, params: Any, inputs: Any) -> ObjectiveResult:
        del result, inputs
        leaves = [leaf for leaf in jax.tree.leaves(params) if eqx.is_inexact_array(leaf)]
        squared_l2 = sum((jnp.real(jnp.vdot(leaf, leaf)) for leaf in leaves), start=jnp.asarray(0.0))
        loss = self.coefficient * squared_l2
        return ObjectiveResult(loss=loss, metrics={self.metric_name: squared_l2}, aux=())


class WeightedSumObjective(eqx.Module):
    """Compose named objectives into one weighted scalar loss."""

    objectives: tuple[Any, ...]
    weights: tuple[float, ...] = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        objectives: Iterable[Any],
        *,
        weights: Iterable[float] | None = None,
        names: Iterable[str] | None = None,
    ) -> None:
        objective_values = tuple(objectives)
        if not objective_values:
            raise ValueError("WeightedSumObjective requires at least one objective")
        raw_weights = (1.0,) * len(objective_values) if weights is None else weights
        raw_names = (f"term_{index}" for index in range(len(objective_values))) if names is None else names
        weight_values = tuple(float(weight) for weight in raw_weights)
        name_values = tuple(raw_names)
        if len(weight_values) != len(objective_values):
            raise ValueError("weights must have the same length as objectives")
        if len(name_values) != len(objective_values):
            raise ValueError("names must have the same length as objectives")
        if len(set(name_values)) != len(name_values) or any(not name for name in name_values):
            raise ValueError("objective names must be non-empty and unique")
        self.objectives = objective_values
        self.weights = weight_values
        self.names = name_values

    def __call__(self, result: RawResult, params: Any, inputs: Any) -> ObjectiveResult:
        evaluations = tuple(_validated_objective(objective(result, params, inputs)) for objective in self.objectives)
        weighted_losses = tuple(
            weight * evaluation.loss for weight, evaluation in zip(self.weights, evaluations, strict=True)
        )
        loss = sum(weighted_losses, start=jnp.asarray(0.0))
        metrics: dict[str, Any] = {"loss": loss}
        aux: dict[str, Any] = {}
        for name, evaluation, weighted_loss in zip(self.names, evaluations, weighted_losses, strict=True):
            metrics[f"{name}/loss"] = evaluation.loss
            metrics[f"{name}/weighted_loss"] = weighted_loss
            metrics.update({f"{name}/{key}": value for key, value in evaluation.metrics.items()})
            aux[name] = evaluation.aux
        return ObjectiveResult(loss=loss, metrics=metrics, aux=aux)


class LegacyVGAdapter(eqx.Module):
    """Temporary ``trainable_modules``/``vg`` bridge for a pure JAX program."""

    program: Any
    objective: Any
    state: Any
    inputs: Any
    key: jax.Array

    def __init__(self, program: Any, objective: Any, state: Any, inputs: Any, key: jax.Array) -> None:
        warnings.warn(
            "LegacyVGAdapter is a migration bridge. Pass explicit params/state/inputs/key "
            "to adept.value_and_grad in new code.",
            FutureWarning,
            stacklevel=2,
        )
        self.program = program
        self.objective = objective
        self.state = state
        self.inputs = inputs
        self.key = key

    def __call__(self, trainable_modules: Any, args: Any | None = None) -> tuple[Any, dict[str, Any]]:
        runtime_inputs = self.inputs if args is None else args
        run = evaluate_objective(
            self.program,
            self.objective,
            trainable_modules,
            self.state,
            runtime_inputs,
            self.key,
        )
        return run.objective.loss, {"solver result": run.simulation, "objective": run.objective}

    def vg(self, trainable_modules: Any, args: Any | None = None) -> tuple[tuple[Any, dict[str, Any]], Any]:
        runtime_inputs = self.inputs if args is None else args
        run = value_and_grad(
            self.program,
            self.objective,
            trainable_modules,
            self.state,
            runtime_inputs,
            self.key,
        )
        output = {"solver result": run.simulation, "objective": run.objective}
        return (run.objective.loss, output), run.gradients


__all__ = [
    "CallableObjective",
    "L2Penalty",
    "LegacyVGAdapter",
    "ObjectiveRun",
    "ParameterPartition",
    "ValueAndGradResult",
    "WeightedSumObjective",
    "evaluate_objective",
    "partition_parameters",
    "value_and_grad",
]
