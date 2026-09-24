"""Experimental reconstruction limiting for independent AMR panels.

Source limiting bounds the actual remesh queries, not the complete deformed
polynomial. Destination limiting certifies a rectangular biquadratic through
its Bernstein coefficients and dissipates its native/exact polynomial C2.
Source material-node C2 also contracts, but resampled source-stage C2 need not.
Neither operation makes the remap conservative. No clipping or renormalization.
"""

import jax.numpy as jnp
import numpy as np


def bernstein_coefficients(f):
    """Convert (..., 9) midpoint/end-point nodal data to (..., 3, 3)."""
    transform = jnp.asarray([[1, 0, 0], [-0.5, 2, -0.5], [0, 0, 1]], dtype=f.dtype)
    values = f.reshape((*f.shape[:-1], 3, 3))
    return jnp.einsum("ai,...ij,bj->...ab", transform, values, transform)


def polynomial_c2(f, areas):
    """Exact rectangular biquadratic-square integral, up to roundoff."""
    nodes, weights = np.polynomial.legendre.leggauss(3)
    nodes, weights = (nodes + 1) / 2, weights / 2
    basis = jnp.asarray(
        np.stack((2 * (nodes - 0.5) * (nodes - 1), 4 * nodes * (1 - nodes), 2 * nodes * (nodes - 0.5)), axis=1),
        dtype=f.dtype,
    )
    values = jnp.einsum("ai,pij,bj->pab", basis, f.reshape(-1, 3, 3), basis)
    quadrature = jnp.asarray(np.outer(weights, weights), dtype=f.dtype)
    return jnp.sum(areas[:, None, None] * quadrature * values**2)


def _material_data(f, weights, active):
    """Mask padding before arithmetic; invalid active data remain invalid."""
    f = jnp.where(active[:, None], f, 0.0)
    weights = jnp.where(active[:, None], weights, 0.0)
    areas = jnp.sum(weights, axis=1)
    denominator = jnp.where(active & (areas > 0), areas, 1.0)
    means = jnp.sum((weights / denominator[:, None]) * f, axis=1)
    ok = jnp.all(jnp.isfinite(f) & jnp.isfinite(weights) & (weights > 0), axis=1)
    ok &= jnp.isfinite(areas) & (areas > 0) & jnp.isfinite(means) & (means >= 0)
    return f, weights, areas, means, ~active | ok


def _scaling(means, minima, eligible):
    """Scale around nonnegative means, retaining a small roundoff margin."""
    needs = eligible & (minima < 0)
    # Normalized division avoids overflowing mean-minimum and works for tails.
    scale = jnp.maximum(jnp.abs(means), jnp.abs(jnp.where(needs, minima, 0.0)))
    scale = jnp.where(needs & (scale > 0), scale, 1.0)
    normalized_mean = means / scale
    normalized_minimum = jnp.where(needs, minima / scale, 0.0)
    denominator = jnp.where(needs, normalized_mean - normalized_minimum, 1.0)
    theta = jnp.where(needs, normalized_mean / denominator, 1.0)
    # This scales the polynomial, not individual negative values. Without the
    # margin, cancellation at the active bound can recreate a negative sample.
    return jnp.where(needs, theta * (1 - 16 * jnp.finfo(means.dtype).eps), 1.0)


def limit_source_samples(values, owners, exterior, f, weights, active):
    """Use one affine polynomial scaling per owner over ALL candidate queries.

    The anchor is the source panel's material-quadrature mean. Its preservation
    does not imply preservation of the deformed physical integral, nor of the
    destination quadrature after remapping. Exterior zero-inflow is unchanged.
    """
    f, weights, _, means, panel_ok = _material_data(f, weights, active)
    owner_in_range = (owners >= 0) & (owners < active.size)
    safe_owners = jnp.clip(owners, 0, active.size - 1)
    query_ok = exterior | (owner_in_range & active[safe_owners] & jnp.isfinite(values))
    minima = jnp.full(active.shape, jnp.inf, dtype=f.dtype).at[safe_owners].min(jnp.where(exterior, jnp.inf, values))
    panel_ok &= ~active | (jnp.isfinite(minima) | jnp.isposinf(minima))
    theta = _scaling(means, minima, active & panel_ok)
    query_theta, query_mean = theta[safe_owners], means[safe_owners]
    scaled = (1 - query_theta) * query_mean + query_theta * values
    limited = jnp.where(exterior, 0.0, jnp.where(query_theta < 1, scaled, values))
    tolerance = 64 * jnp.finfo(f.dtype).eps * jnp.maximum(jnp.abs(query_mean), jnp.abs(limited))
    valid = jnp.any(active) & jnp.all(panel_ok) & jnp.all(query_ok)
    valid &= jnp.all(jnp.isfinite(limited) & (limited >= -tolerance))
    scaled_nodes = jnp.where(theta[:, None] < 1, (1 - theta[:, None]) * means[:, None] + theta[:, None] * f, f)
    return limited, {
        "valid": valid,
        "theta": theta,
        "limited_panels": jnp.sum(active & (theta < 1), dtype=jnp.int32),
        "failed_panels": jnp.sum(active & ~panel_ok, dtype=jnp.int32),
        "min_theta": jnp.min(theta),
        "material_mass_change": jnp.sum(weights * (scaled_nodes - f)),
        "material_c2_change": jnp.sum(weights * (scaled_nodes**2 - f**2)),
    }


def limit_rectangles(f, weights, active):
    """Certify nonnegative rectangle polynomials while preserving Simpson mass.

    Bernstein positivity is sufficient, not necessary; a nonnegative polynomial
    with a negative Bernstein coefficient may also be damped. Independent
    panels can acquire different traces at their shared geometric boundary.
    Negative/nonfinite means or incompatible quadrature are explicit failures.
    """
    f, weights, areas, means, panel_ok = _material_data(f, weights, active)
    axis = jnp.asarray([1, 4, 1], dtype=f.dtype) / 6
    expected_weights = areas[:, None] * jnp.outer(axis, axis).reshape(1, 9)
    eps = jnp.finfo(f.dtype).eps
    quadrature_ok = jnp.all(jnp.abs(weights - expected_weights) <= 64 * eps * expected_weights, axis=1)
    panel_ok &= ~active | quadrature_ok
    coefficients = bernstein_coefficients(f)
    minima = jnp.min(coefficients, axis=(1, 2))
    panel_ok &= ~active | jnp.all(jnp.isfinite(coefficients), axis=(1, 2))
    theta = _scaling(means, minima, active & panel_ok)
    scaled = (1 - theta[:, None]) * means[:, None] + theta[:, None] * f
    limited = jnp.where(theta[:, None] < 1, scaled, f)
    limited_coefficients = bernstein_coefficients(limited)
    tolerance = 64 * eps * jnp.max(jnp.abs(limited_coefficients), axis=(1, 2))
    panel_ok &= ~active | jnp.all(limited_coefficients >= -tolerance[:, None, None], axis=(1, 2))
    minimum = jnp.min(jnp.where(active[:, None, None], limited_coefficients, jnp.inf))
    return limited, {
        "valid": jnp.any(active) & jnp.all(panel_ok),
        "theta": theta,
        "limited_panels": jnp.sum(active & (theta < 1), dtype=jnp.int32),
        "failed_panels": jnp.sum(active & ~panel_ok, dtype=jnp.int32),
        "min_theta": jnp.min(theta),
        "minimum_coefficient": jnp.where(jnp.any(active), minimum, 0.0),
        "mass_change": jnp.sum(weights * (limited - f)),
        "c2_change": jnp.sum(weights * (limited**2 - f**2)),
        "polynomial_c2_change": polynomial_c2(limited, areas) - polynomial_c2(f, areas),
    }


def initialize_positivity(state):
    """Limit the initial AMR representation; never renormalize its mass.

    Initial effects are separate from cumulative evolution/remap budgets. Call
    after the builder's one initial selected-mass normalization, and only for
    the experimental AMR/Simpson path. All additional state entries are scalar.
    """
    f, info = limit_rectangles(state["f"], state["weights"], state["active"])
    zero = jnp.zeros((), dtype=f.dtype)
    count = jnp.asarray(0, dtype=jnp.int32)
    budgets = {
        f"{stage}_{moment}_change": zero
        for stage in ("interpolation", "source_limiter", "destination_limiter")
        for moment in ("mass", "c2")
    }
    return {
        **state,
        "f": f,
        "valid": state["valid"] & info["valid"],
        **{
            f"initial_positivity_{key}": info[key]
            for key in (
                "mass_change",
                "c2_change",
                "polynomial_c2_change",
                "limited_panels",
                "failed_panels",
                "min_theta",
            )
        },
        **budgets,
        "destination_limiter_polynomial_c2_change": zero,
        "source_limiter_panels": count,
        "destination_limiter_panels": count,
        "positivity_failed_panels": count,
        "source_limiter_min_theta": jnp.ones((), dtype=f.dtype),
        "destination_limiter_min_theta": jnp.ones((), dtype=f.dtype),
        "min_bernstein_coefficient": info["minimum_coefficient"],
    }
