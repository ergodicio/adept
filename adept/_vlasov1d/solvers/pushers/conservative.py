"""Positive conservative semi-Lagrangian translations of uniform-grid cell averages.

Every row has one constant displacement. Whole-cell shifts are exact; only the
fractional shift is reconstructed. If a donor cell contains ``f``, its outgoing
fractional mass ``F`` is bounded by ``0 <= F <= f``. The update adds the donor's
nonnegative remainder to its upstream neighbor's nonnegative outgoing mass.
The same flux is used by both recipients, preserving mass to roundoff on a
periodic domain. There is no transport CFL restriction or internal subcycling.

PFC3 uses a mean-preserving positive quadratic reconstruction (a PFC-style
remap). SL-WENO5 combines three quadratic integral reconstructions with WENO-Z
weights and a donor-mass flux limiter. This is a positivity-only limiter, not
the full maximum-principle limiter of Xiong et al. (2014).
"""

from functools import reduce

from jax import numpy as jnp


def _pfc3_flux(fm, f0, fp, fraction):
    """Integrate a positive quadratic over the downwind fractional cell."""
    slope = (fp - fm) / 2
    curvature = (fm - 2 * f0 + fp) / 2
    # p(y) = f0 + slope*y + curvature*(y**2 - 1/12), y in [-1/2, 1/2].
    # Include an interior minimum; limiting only endpoints misses this case.
    safe_curvature = jnp.where(curvature > 0, curvature, 1.0)
    vertex = jnp.clip(-slope / (2 * safe_curvature), -0.5, 0.5)
    endpoint_min = f0 + curvature / 6 - jnp.abs(slope) / 2
    vertex_value = f0 + slope * vertex + curvature * (vertex**2 - 1 / 12)
    minimum = jnp.minimum(endpoint_min, jnp.where(curvature > 0, vertex_value, endpoint_min))
    denominator = jnp.where(minimum < 0, f0 - minimum, 1.0)
    theta = jnp.where(minimum < 0, f0 / denominator, 1.0)
    correction = (1 - fraction) * (slope / 2 + (1 - 2 * fraction) * curvature / 6)
    return fraction * (f0 + theta * correction)


def _weno5_flux(fmm, fm, f0, fp, fpp, fraction):
    """Fifth-order conservative fractional-cell integral with WENO-Z weights."""
    slopes = ((fmm - 4 * fm + 3 * f0) / 2, (fp - fm) / 2, (-3 * f0 + 4 * fp - fpp) / 2)
    curves = ((fmm - 2 * fm + f0) / 2, (fm - 2 * f0 + fp) / 2, (f0 - 2 * fp + fpp) / 2)
    beta = (
        13 / 12 * (fmm - 2 * fm + f0) ** 2 + (fmm - 4 * fm + 3 * f0) ** 2 / 4,
        13 / 12 * (fm - 2 * f0 + fp) ** 2 + (fm - fp) ** 2 / 4,
        13 / 12 * (f0 - 2 * fp + fpp) ** 2 + (3 * f0 - 4 * fp + fpp) ** 2 / 4,
    )
    # Optimal weights depend on the integrated strip width. Fixed interface
    # weights (0.1, 0.6, 0.3) would lose fifth-order accuracy for finite shifts.
    d0 = (fraction + 1) * (fraction + 2) / 20
    d2 = (3 - fraction) * (2 - fraction) / 20
    linear_weights = (d0, 1 - d0 - d2, d2)
    tau = jnp.abs(beta[0] - beta[2])
    weights = tuple(d * (1 + (tau / (b + 1e-6)) ** 2) for d, b in zip(linear_weights, beta, strict=True))
    weight_sum = sum(weights)
    correction = sum(
        w / weight_sum * (s / 2 + (1 - 2 * fraction) * c / 6) for w, s, c in zip(weights, slopes, curves, strict=True)
    )
    return fraction * (f0 + (1 - fraction) * correction)


def conservative_remap(f, shift, spacing, *, method: str, periodic: bool):
    """Translate nonnegative cell-average rows by ``shift`` in physical units.

    ``f`` has shape (number of lines, number of cells); ``shift`` is scalar or
    has one value per line. ``method`` and ``periodic`` must be static under JIT.
    Both displacement signs, multiple domain crossings and zero shifts are
    supported. Nonperiodic lines have zero inflow and discard escaped mass;
    no tail wraps around and no distribution clipping/renormalization is used.

    Input values must be finite nonnegative float32/float64 cell averages, with
    positive uniform spacing. Half precision is not supported.
    Formal spatial accuracy assumes accurate initial cell averages. Existing
    ADEPT initialization supplies midpoint approximations to those averages.
    Gradients are piecewise smooth away from stencil and limiter switches.
    """
    if method not in {"pfc3", "sl-weno5"}:
        raise ValueError(f"Unknown conservative remap: {method}")
    width = 3 if method == "pfc3" else 5
    if f.ndim != 2 or f.shape[1] < width:
        raise ValueError(f"{method} requires 2D input with at least {width} cells per line")
    if f.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("conservative remapping requires float32 or float64 cell averages")

    n = f.shape[1]
    spacing = jnp.asarray(spacing, dtype=f.dtype)
    courant = jnp.broadcast_to(jnp.asarray(shift, dtype=f.dtype) / spacing, (f.shape[0],))[:, None]
    direction = jnp.where(courant < 0, -1, 1)
    distance = jnp.abs(courant)
    # Bound before integer conversion, also for displacements many domains wide.
    distance = jnp.remainder(distance, n) if periodic else jnp.minimum(distance, n)
    whole = jnp.floor(distance).astype(jnp.int32)
    fraction = distance - whole
    donor = jnp.arange(n, dtype=jnp.int32)[None, :] - direction * whole

    def sample(index):
        if periodic:
            return jnp.take_along_axis(f, jnp.mod(index, n), axis=1)
        value = jnp.take_along_axis(f, jnp.clip(index, 0, n - 1), axis=1)
        return jnp.where((index >= 0) & (index < n), value, 0.0)

    def split_mass(index):
        samples = tuple(sample(index + direction * k) for k in range(-(width // 2), width // 2 + 1))
        center = samples[width // 2]
        # Normalize locally so WENO weights and quadratic limiting remain finite
        # for vacuum, tiny tails and large amplitudes, without a global reduction.
        # sqrt(tiny), rather than tiny, also keeps the reciprocal's derivative
        # finite in vacuum during reverse-mode AD (which squares this scale).
        scale = jnp.maximum(reduce(jnp.maximum, samples), jnp.sqrt(jnp.finfo(f.dtype).tiny))
        normalized = tuple(value / scale for value in samples)
        flux = _pfc3_flux(*normalized, fraction) if method == "pfc3" else _weno5_flux(*normalized, fraction)
        outgoing = jnp.clip(flux * scale, 0.0, center)
        return center - outgoing, outgoing

    remaining, _ = split_mass(donor)
    _, incoming = split_mass(donor - direction)
    return remaining + incoming
