"""A one-dimensional barycentric Lagrange treecode for the FARSIGHT field.

Independent implementation of the interpolation approach of Wang, Krasny &
Tlupova, Commun. Comput. Phys. 28 (2020), 1415--1436,
https://arxiv.org/abs/1902.02250, used in Sandberg, Krasny & Thomas,
JCP 523 (2025), 113664. This implementation uses a sorted binary source tree,
not the reference implementation's GPU batching/tree data structures.

Construction and storage have static shapes. Evaluation uses an actual
data-dependent tree walk, so JAX supports JIT and forward-mode derivatives,
but not reverse-mode derivatives of the accelerated path. ``theta=0`` uses
the differentiable direct solver. Sorting, activity, and opening decisions
are discrete: derivatives are local to a fixed tree and interaction list.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .numerics import electric_field, periodic_kernel


def _check_options(degree, theta, leaf_size, chunk_size):
    for name, value in (("degree", degree), ("leaf_size", leaf_size), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(theta, bool) or not math.isfinite(theta) or not 0 <= theta < 1:
        raise ValueError("theta must be finite and satisfy 0 <= theta < 1")


def _chebyshev(degree, dtype):
    index = jnp.arange(degree + 1)
    nodes = -jnp.cos(jnp.pi * index / degree).astype(dtype)
    weights = jnp.where(index % 2, -1.0, 1.0).astype(dtype)
    weights = weights.at[jnp.array([0, degree])].multiply(0.5)
    return nodes, weights


def _basis(coordinates, nodes, weights):
    """Barycentric cardinal polynomials, including their limits at nodes.

    At an interpolation node use the cardinal value plus its linear Taylor
    term. The latter is zero in value but supplies the correct JVP, which a
    plain ``where(hit, one_hot, ratio)`` would otherwise erase.
    """
    difference = coordinates[..., None] - nodes
    hit = difference == 0
    ratios = weights / jnp.where(hit, 1.0, difference)
    normal = ratios / jnp.sum(ratios, axis=-1, keepdims=True)
    index = jnp.argmax(hit, axis=-1)
    node_difference = nodes[index][..., None] - nodes
    diagonal = jnp.arange(nodes.size) == index[..., None]
    derivative = weights / weights[index][..., None] / jnp.where(diagonal, 1.0, node_difference)
    derivative = jnp.where(diagonal, 0.0, derivative)
    derivative = derivative - diagonal * jnp.sum(derivative, axis=-1, keepdims=True)
    exact = diagonal + (coordinates - nodes[index])[..., None] * derivative
    return jnp.where(jnp.any(hit, axis=-1, keepdims=True), exact, normal)


def _build_tree(sources, charges, length, degree, leaf_size):
    """Build interpolation charges bottom-up with O(N p + nodes p**2) work."""
    number_leaves = 1 << max(0, ((sources.size + leaf_size - 1) // leaf_size - 1).bit_length())
    number_nodes = 2 * number_leaves - 1
    first_leaf = number_leaves - 1
    nodes, barycentric_weights = _chebyshev(degree, sources.dtype)
    # Zero charges are inactive slots, including the padded AMR capacity.
    # Their positions are immaterial, even when a slot contains NaN or inf.
    source_active = charges != 0
    wrapped = jnp.mod(jnp.where(source_active, sources, 0.0), length)
    order = jnp.argsort(jnp.where(source_active, wrapped, jnp.inf), stable=True)
    padding = number_leaves * leaf_size - sources.size
    sx = jnp.pad(wrapped[order], (0, padding)).reshape(number_leaves, leaf_size)
    sq = jnp.pad(charges[order], (0, padding)).reshape(number_leaves, leaf_size)
    present = sq != 0
    active = jnp.zeros(number_nodes, dtype=bool).at[first_leaf:].set(jnp.any(present, axis=-1))
    lower = jnp.min(jnp.where(present, sx, jnp.inf), axis=-1)
    upper = jnp.max(jnp.where(present, sx, -jnp.inf), axis=-1)
    lower, upper = jnp.where(active[first_leaf:], lower, 0.0), jnp.where(active[first_leaf:], upper, 0.0)
    center = jnp.zeros(number_nodes, dtype=sources.dtype).at[first_leaf:].set(0.5 * (lower + upper))
    radius = jnp.zeros_like(center).at[first_leaf:].set(0.5 * (upper - lower))
    coordinates = (sx - center[first_leaf:, None]) / jnp.where(
        radius[first_leaf:, None] > 0, radius[first_leaf:, None], 1.0
    )
    # Evaluate inactive positions at the center to avoid extrapolation overflow.
    coordinates = jnp.where(present, coordinates, 0.0)
    moment = jnp.zeros((number_nodes, degree + 1), dtype=charges.dtype)
    moment = moment.at[first_leaf:].set(
        jnp.sum(sq[..., None] * _basis(coordinates, nodes, barycentric_weights), axis=1)
    )

    for level in range(number_leaves.bit_length() - 2, -1, -1):
        parents = jnp.arange((1 << level) - 1, (1 << (level + 1)) - 1)
        children = 2 * parents[:, None] + jnp.array([1, 2])
        child_active = active[children]
        parent_active = jnp.any(child_active, axis=1)
        lo = jnp.min(jnp.where(child_active, center[children] - radius[children], jnp.inf), axis=1)
        hi = jnp.max(jnp.where(child_active, center[children] + radius[children], -jnp.inf), axis=1)
        lo, hi = jnp.where(parent_active, lo, 0.0), jnp.where(parent_active, hi, 0.0)
        pc, pr = 0.5 * (hi + lo), 0.5 * (hi - lo)
        child_nodes = center[children, None] + radius[children, None] * nodes
        coordinates = (child_nodes - pc[:, None, None]) / jnp.where(pr[:, None, None] > 0, pr[:, None, None], 1.0)
        coordinates = jnp.where(child_active[..., None], coordinates, 0.0)
        transfer = _basis(coordinates, nodes, barycentric_weights)
        parent_moment = jnp.sum(moment[children, :, None] * transfer, axis=(1, 2))
        center, radius = center.at[parents].set(pc), radius.at[parents].set(pr)
        active, moment = active.at[parents].set(parent_active), moment.at[parents].set(parent_moment)

    # A threaded tree avoids a stack: finishing a subtree jumps to its next
    # sibling/ancestor sibling, while opening a cluster visits its left child.
    escape = np.full(number_nodes, number_nodes, dtype=np.int32)
    for parent in range(first_leaf):
        escape[2 * parent + 1], escape[2 * parent + 2] = 2 * parent + 2, escape[parent]
    return center, radius, active, moment, sx, sq, nodes, jnp.asarray(escape)


def electric_field_treecode_with_info(
    targets, sources, charges, length, epsilon, *, degree=8, theta=0.5, leaf_size=32, chunk_size=64
):
    """Evaluate the field and return mathematical interaction-work counters.

    Far clusters use degree+1 Chebyshev-Lobatto interpolation charges. The
    opening condition requires radius/distance < theta, a distance from the
    nearest source interval greater than epsilon, and an interval entirely
    inside one smooth branch of the nearest-image periodic kernel. Near
    leaves are summed directly, including exact zero self interaction.

    The counters count visited clusters, accepted clusters, direct source
    interactions, and kernel evaluations; they are not wall-clock or GPU
    instruction measurements. Degenerate/clumped inputs can force quadratic
    direct work. This particle-cluster approximation is not pair-symmetric
    and does not enforce exact momentum or energy conservation.

    All options are static under JIT. Inputs must have finite active source
    positions, finite charges and targets, and positive length/epsilon.
    Zero-charge positions are ignored. First-order forward derivatives are
    supported within fixed topology; reverse mode requires ``theta=0``.
    """
    _check_options(degree, theta, leaf_size, chunk_size)
    degree, leaf_size, chunk_size = int(degree), int(leaf_size), int(chunk_size)
    targets, sources, charges = jnp.asarray(targets), jnp.asarray(sources), jnp.asarray(charges)
    dtype = jnp.result_type(targets, sources, charges, length, epsilon, 0.0)
    targets, sources, charges = (
        targets.astype(dtype),
        sources.astype(dtype).reshape(-1),
        charges.astype(dtype).reshape(-1),
    )
    if sources.shape != charges.shape:
        raise ValueError("sources and charges must have the same number of elements")
    zero = jnp.asarray(0, dtype=jnp.int32)
    counter_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
    counter_zero = jnp.asarray(0, dtype=counter_dtype)
    empty_info = {
        "visited_nodes": counter_zero,
        "accepted_clusters": counter_zero,
        "direct_pairs": counter_zero,
        "kernel_evaluations": counter_zero,
    }
    if not sources.size or not targets.size:
        return jnp.zeros_like(targets), empty_info
    if theta == 0:
        # Preserve finite zero-charge positions: their charge derivatives are
        # still kernel(target - source). Only invalid inactive slots need a
        # replacement coordinate to avoid the otherwise undefined 0 * NaN.
        sources = jnp.where((charges == 0) & ~jnp.isfinite(sources), 0.0, sources)
        values = electric_field(targets, sources, charges, length, epsilon, chunk_size)
        pairs = jnp.asarray(targets.size * sources.size, dtype=counter_dtype)
        return values, {**empty_info, "direct_pairs": pairs, "kernel_evaluations": pairs}

    center, radius, active, moment, sx, sq, nodes, escape = _build_tree(sources, charges, length, degree, leaf_size)
    first_leaf = sx.shape[0] - 1

    def at_target(target):
        def visit(state):
            node, value, visited, accepted_count, pair_count = state
            difference = target - center[node]
            difference -= length * jnp.floor(difference / length + 0.5)
            distance = jnp.abs(difference)
            far = (
                active[node]
                & (radius[node] < theta * distance)
                & (distance - radius[node] > epsilon)
                & (distance + radius[node] < 0.5 * length)
            )
            leaf = node >= first_leaf

            def interpolate(_):
                points = center[node] + radius[node] * nodes
                return jnp.sum(moment[node] * periodic_kernel(target - points, length, epsilon))

            def direct_or_open(_):
                leaf_index = jnp.maximum(node - first_leaf, 0)
                return jax.lax.cond(
                    leaf & active[node],
                    lambda _: jnp.sum(sq[leaf_index] * periodic_kernel(target - sx[leaf_index], length, epsilon)),
                    lambda _: jnp.zeros((), dtype=dtype),
                    operand=None,
                )

            contribution = jax.lax.cond(far, interpolate, direct_or_open, operand=None)
            direct_count = jnp.where(leaf & active[node] & ~far, leaf_size, 0)
            next_node = jnp.where(far | leaf | ~active[node], escape[node], 2 * node + 1)
            return next_node, value + contribution, visited + 1, accepted_count + far, pair_count + direct_count

        initial = (zero, jnp.zeros((), dtype=dtype), counter_zero, counter_zero, counter_zero)
        _, value, visited, accepted_count, pair_count = jax.lax.while_loop(
            lambda state: state[0] < center.size, visit, initial
        )
        return value, jnp.stack((visited, accepted_count, pair_count))

    values, counters = jax.lax.map(at_target, targets.reshape(-1), batch_size=min(chunk_size, targets.size))
    visited, accepted, pairs = jnp.sum(counters, axis=0)
    info = {
        "visited_nodes": visited,
        "accepted_clusters": accepted,
        "direct_pairs": pairs,
        "kernel_evaluations": pairs + (degree + 1) * accepted,
    }
    return values.reshape(targets.shape), info


def electric_field_treecode(
    targets, sources, charges, length, epsilon, *, degree=8, theta=0.5, leaf_size=32, chunk_size=64
):
    """Field-only interface; see :func:`electric_field_treecode_with_info`."""
    return electric_field_treecode_with_info(
        targets,
        sources,
        charges,
        length,
        epsilon,
        degree=degree,
        theta=theta,
        leaf_size=leaf_size,
        chunk_size=chunk_size,
    )[0]


class TreecodeField(eqx.Module):
    """Static tree controls with the direct field solver's calling convention."""

    degree: int = eqx.field(static=True, default=8)
    theta: float = eqx.field(static=True, default=0.5)
    leaf_size: int = eqx.field(static=True, default=32)

    def __call__(self, targets, sources, charges, length, epsilon, chunk_size=64):
        return electric_field_treecode(
            targets,
            sources,
            charges,
            length,
            epsilon,
            degree=self.degree,
            theta=self.theta,
            leaf_size=self.leaf_size,
            chunk_size=chunk_size,
        )
