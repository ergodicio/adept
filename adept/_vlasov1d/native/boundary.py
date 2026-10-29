"""Tail checks shared by native preprocessing and the standalone MPI worker."""

import warnings

import numpy as np


def tail_weights(edges):
    """Cell overlap with the outer 5% of the velocity extent on each side."""
    width = 0.05 * (edges[-1] - edges[0])
    return np.maximum(0, np.minimum(edges[1:], edges[0] + width) - edges[:-1]) + np.maximum(
        0, edges[1:] - np.maximum(edges[:-1], edges[-1] - width)
    )


def initial_diagnostics(edges, values, components, mass, dx):
    """Check the actual state and truncation hidden by initializer normalization."""
    from scipy.special import gammaincc, gammaln

    widths, tails = np.diff(edges), tail_weights(edges)
    maximum, total = 0.0, 0.0
    # Keep preprocessing bounded even for full production x-v states.
    block = max(1, 1048576 // values.shape[1])
    for start in range(0, values.shape[0], block):
        absolute = np.abs(np.asarray(values[start : start + block]))
        row_mass = absolute @ widths
        fraction = np.divide(absolute @ tails, row_mass, out=np.zeros_like(row_mass), where=row_mass > 0)
        maximum = max(maximum, float(np.max(fraction)))
        total += float(np.sum(row_mass))
    missing = 0.0
    for component in components:
        m = component.supergaussian_order
        scale = np.sqrt(3 * np.exp(gammaln(3 / m) - gammaln(5 / m)) * component.T0 / mass)
        z = (edges[[0, -1]] - component.v0) / scale
        tails = 0.5 * gammaincc(1 / m, np.abs(z) ** m)
        left = tails[0] if z[0] <= 0 else 1 - tails[0]
        right = tails[1] if z[1] >= 0 else 1 - tails[1]
        missing = max(missing, float(left + right))
    return dict(
        initial_tail_fraction=maximum,
        initial_missing_fraction=missing,
        initial_absolute_mass=total * dx,
    )


def boundary_message(name, diagnostics, tolerance):
    keys = [
        "initial_missing_fraction",
        "initial_tail_fraction",
        "max_tail_fraction",
        "max_attempted_escape_fraction",
        "cumulative_attempted_escape_fraction",
    ]
    exceeded = [
        f"{key}={diagnostics[key]:.3e}"
        for key in keys
        if key in diagnostics and (not np.isfinite(diagnostics[key]) or diagnostics[key] > tolerance)
    ]
    if not exceeded:
        return None
    return (
        f"Velocity boundary check for {name}: {', '.join(exceeded)} exceeds {tolerance:.3e}. "
        "Widen vmin/vmax into the tails and retain the required interior resolution. "
        "Zero-flux boundaries conserve mass but can distort a distribution that reaches them."
    )


def check_boundary(name, diagnostics, tolerance, action):
    message = boundary_message(name, diagnostics, tolerance)
    if message:
        if action == "error":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
