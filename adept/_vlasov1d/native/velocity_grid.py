"""Cell edges for the native conservative velocity remap."""

import numpy as np


def velocity_edges(species, config=None):
    """Resolve explicit edges or the two-grid construction in Afeyan et al. §2.1.

    Preserve exactly ``nv`` cells and an integer coarse/fine ratio. Requested
    refinement endpoints snap to coarse edges; reject unresolvable intervals.
    """
    n, lower, upper = species.nv, species.vmin, species.vmax
    if n < 4 or not np.isfinite([lower, upper, upper - lower]).all() or upper <= lower:
        raise ValueError("native velocity grids require nv >= 4 and finite ordered bounds")
    if config is None:
        edges = np.linspace(lower, upper, n + 1)
    elif config.edges is not None:
        edges = np.asarray(config.edges, dtype=float)
    else:
        a, b, ratio = config.refined_min, config.refined_max, config.ratio
        if not lower <= a < b <= upper:
            raise ValueError("refined_min < refined_max must lie inside the species velocity bounds")
        alpha, beta = (a - lower) / (upper - lower), (b - lower) / (upper - lower)
        coarse_estimate = int(np.floor(n / (1 + (beta - alpha) * (ratio - 1))))
        refined_cells = (n - coarse_estimate) // (ratio - 1)
        coarse_cells = n - refined_cells * (ratio - 1)
        if not 1 <= refined_cells <= coarse_cells:
            raise ValueError("refinement is unresolved at this nv/ratio; increase nv or reduce ratio")
        first = min(int(np.floor(alpha * coarse_cells)), coarse_cells - refined_cells)
        coarse_edges = np.linspace(lower, upper, coarse_cells + 1)
        edges = np.concatenate(
            [
                coarse_edges[:first],
                np.linspace(coarse_edges[first], coarse_edges[first + refined_cells], ratio * refined_cells + 1),
                coarse_edges[first + refined_cells + 1 :],
            ]
        )
    if edges.shape != (n + 1,) or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("velocity edges must contain nv+1 finite, strictly increasing values")
    if not np.allclose(edges[[0, -1]], [lower, upper], rtol=0, atol=1e-12):
        raise ValueError("velocity edges must span the species vmin/vmax bounds")
    return edges
