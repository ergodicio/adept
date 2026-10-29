"""Coordinates for plots of saved Vlasov-1D distributions."""

import numpy as np


def phase_space_coordinates(dataset, save_key, dx):
    """Use physical cell edges for full-resolution velocity cell averages.

    Point samples (including interpolated and Fourier saves) retain their
    center coordinates. The full-resolution spatial grid is uniform.
    """
    distribution = dataset[save_key]
    spatial_dim = "kx" if "kx" in distribution.dims else "x"
    velocity_dim = next(dim for dim in distribution.dims if dim.startswith("v_"))
    x = distribution.coords[spatial_dim].values
    v = distribution.coords[velocity_dim].values
    bounds_name = dataset[velocity_dim].attrs.get("bounds")
    if spatial_dim == "x" and distribution.attrs.get("velocity_representation") == "cell_average" and bounds_name:
        bounds = dataset[bounds_name].values
        v = np.concatenate([bounds[:, 0], bounds[-1:, 1]])
        x = np.concatenate([x - dx / 2, x[-1:] + dx / 2])
    return x, v
