"""Physical-unit initial profiles for periodic VFP-2D reconnection models.

Tabulated profiles are an explicit exchange format, not a native GORGON reader.
All spatial derivatives use the same periodic stencil as the field evolution.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from adept.normalization import UREG, normalize
from adept.vfp1d.helpers import load_profile_on_grid
from adept.vfp2d.grid import Grid
from adept.vfp2d.vector_field import Maxwell2D


def _quantity(value, reference) -> float:
    """Physical strings use ``reference``; numbers are already normalized."""
    if isinstance(value, str):
        if reference is None:
            raise ValueError("A physical reference quantity is required for unit-bearing profiles")
        result = float((UREG.Quantity(value) / reference).to("").magnitude)
    elif np.isscalar(value) and not isinstance(value, (bool, np.bool_)):
        result = float(value)
    else:
        raise TypeError("Profile amplitudes must be numbers or physical-unit strings")
    if not np.isfinite(result):
        raise ValueError("Profile amplitudes must be finite")
    return result


def load_xy_profile(profile: Mapping, grid: Grid, norm, reference) -> jnp.ndarray:
    """Linearly interpolate explicit-unit NPZ data without extrapolation.

    ``x``, ``y`` are strictly increasing one-dimensional coordinate arrays and
    ``values`` has shape ``(len(x), len(y))``. Key names are configurable. The
    caller must state ``x_unit``, ``y_unit`` and ``value_unit`` in the config.
    ``allow_pickle=False`` deliberately excludes object arrays and executables.
    """
    for key in ("path", "x_unit", "y_unit", "value_unit"):
        if key not in profile:
            raise ValueError(f"file_xy profiles require {key!r}")
    if reference is None:
        raise ValueError("file_xy profiles require a physical reference quantity")
    with np.load(profile["path"], allow_pickle=False) as data:
        x = np.asarray(data[profile.get("x_key", "x")], dtype=float)
        y = np.asarray(data[profile.get("y_key", "y")], dtype=float)
        values = np.asarray(data[profile.get("value_key", "values")], dtype=float)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("file_xy coordinates must be 1D arrays with at least two entries")
    if values.shape != (x.size, y.size):
        raise ValueError("file_xy values must have shape (len(x), len(y)); axes are never guessed")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(values))):
        raise ValueError("file_xy coordinates and values must be finite")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("file_xy coordinates must be strictly increasing")
    x = x * float((UREG.Quantity(1, profile["x_unit"]) / norm.L0).to("").magnitude)
    y = y * float((UREG.Quantity(1, profile["y_unit"]) / norm.L0).to("").magnitude)
    values = values * float((UREG.Quantity(1, profile["value_unit"]) / reference).to("").magnitude)
    target_x, target_y = np.asarray(grid.x), np.asarray(grid.y)
    for label, source, target in (("x", x, target_x), ("y", y, target_y)):
        tolerance = 32 * np.finfo(float).eps * max(1.0, abs(source[0]), abs(source[-1]))
        if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
            raise ValueError(f"file_xy {label} coordinates do not cover target cell centers; extrapolation is disabled")
    # Tensor-product interpolation preserves all cross terms of bilinear data.
    along_x = np.stack([np.interp(target_x, x, values[:, iy]) for iy in range(y.size)], axis=1)
    result = np.stack([np.interp(target_y, y, row) for row in along_x], axis=0)
    return jnp.asarray(result)


def profile_1d(profile: dict, axis, norm, reference=None) -> jnp.ndarray:
    basis = profile.get("basis", "uniform")
    baseline = _quantity(profile.get("baseline", profile.get("value", 1.0)), reference)
    if basis == "uniform":
        return baseline * jnp.ones_like(axis)
    if basis in ("sine", "cosine"):
        amplitude = float(profile.get("amplitude", 0.0))
        wavelength = normalize(profile["wavelength"], norm, dim="x")
        if wavelength <= 0:
            raise ValueError("Profile wavelength must be positive")
        center = normalize(profile.get("center", 0.0), norm, dim="x")
        trig = jnp.sin if basis == "sine" else jnp.cos
        return baseline * (1.0 + amplitude * trig(2.0 * jnp.pi * (axis - center) / wavelength))
    if basis == "periodic_tanh":
        # Additive amplitude permits signed counterstreams with zero baseline.
        amplitude = _quantity(profile.get("amplitude", 1.0), reference)
        width = normalize(profile["width"], norm, dim="x")
        wavelength = normalize(profile["wavelength"], norm, dim="x")
        center = normalize(profile.get("center", 0.0), norm, dim="x")
        if width <= 0 or wavelength <= 0:
            raise ValueError("periodic_tanh width and wavelength must be positive")
        k = 2 * jnp.pi / wavelength
        return baseline + amplitude * jnp.tanh(jnp.sin(k * (axis - center)) / (k * width))
    if basis == "tanh":
        center = normalize(profile["center"], norm, dim="x")
        width = normalize(profile["width"], norm, dim="x")
        rise = normalize(profile["rise"], norm, dim="x")
        if width <= 0 or rise <= 0:
            raise ValueError("tanh width and rise must be positive")
        left, right = center - 0.5 * width, center + 0.5 * width
        envelope = 0.5 * (jnp.tanh((axis - left) / rise) - jnp.tanh((axis - right) / rise))
        if profile.get("bump_or_trough", "bump") == "trough":
            envelope = 1.0 - envelope
        return baseline + _quantity(profile.get("bump_height", 0.0), reference) * envelope
    if basis == "file":
        loaded = load_profile_on_grid(profile, axis, norm)
        if reference is None:
            raise ValueError("A physical reference quantity is required for file profiles")
        return jnp.asarray((loaded / reference).to("").magnitude)
    raise NotImplementedError(f"Unsupported VFP-2D profile basis: {basis}")


def profile_2d(profile, grid: Grid, norm, reference=None) -> jnp.ndarray:
    """Build a field, preserving the original dimensionless profile syntax.

    Numbers and physical strings are uniform fields. ``scale`` times ``profile``
    separates a dimensional amplitude from a dimensionless shape. A separable
    physical field should use that form, rather than multiplying unit-bearing
    x and y profiles.
    """
    if not isinstance(profile, Mapping):
        return jnp.full((grid.nx, grid.ny), _quantity(profile, reference))
    if "scale" in profile:
        if "profile" not in profile:
            raise ValueError("A scaled profile requires a 'profile' shape")
        return _quantity(profile["scale"], reference) * profile_2d(profile["profile"], grid, norm)
    if profile.get("basis") == "file_xy":
        return load_xy_profile(profile, grid, norm, reference)
    if profile.get("basis") == "gaussian_spots":
        x_center = normalize(profile.get("x_center", 0.0), norm, dim="x")
        x_radius = normalize(profile["x_radius"], norm, dim="x")
        y_radius = normalize(profile.get("y_radius", profile["x_radius"]), norm, dim="x")
        if x_radius <= 0 or y_radius <= 0:
            raise ValueError("Gaussian radii must be positive")
        y_centers = profile.get("y_centers", [profile.get("y_center", 0.0)])
        y_centers = jnp.asarray([normalize(center, norm, dim="x") for center in y_centers])
        x_envelope = jnp.exp(-(((grid.x - x_center) / x_radius) ** 2))
        y_envelope = jnp.sum(jnp.exp(-(((grid.y[:, None] - y_centers[None, :]) / y_radius) ** 2)), axis=1)
        return _quantity(profile.get("amplitude", 1.0), reference) * x_envelope[:, None] * y_envelope[None, :]
    if "x" in profile or "y" in profile:

        def has_physical_amplitude(child):
            amplitude_keys = ("baseline", "value", "amplitude", "bump_height")
            return any(
                isinstance(child.get(key), str) and not UREG.Quantity(child[key]).dimensionless
                for key in amplitude_keys
            )

        if has_physical_amplitude(profile.get("x", {})) and has_physical_amplitude(profile.get("y", {})):
            raise ValueError(
                "Separable x/y profiles cannot both have physical amplitudes; "
                "use one physical 'scale' with a dimensionless 'profile' shape"
            )
        px = profile_1d(profile.get("x", {"basis": "uniform", "baseline": 1.0}), grid.x, norm, reference)
        py = profile_1d(profile.get("y", {"basis": "uniform", "baseline": 1.0}), grid.y, norm, reference)
        return px[:, None] * py[None, :]
    target_axis = profile.get("axis", "x")
    if target_axis not in ("x", "y"):
        raise ValueError("VFP-2D profile axis must be 'x' or 'y'")
    if target_axis == "y":
        return jnp.broadcast_to(profile_1d(profile, grid.y, norm, reference)[None, :], (grid.nx, grid.ny))
    return jnp.broadcast_to(profile_1d(profile, grid.x, norm, reference)[:, None], (grid.nx, grid.ny))


def vector_profile(spec, grid: Grid, norm, reference) -> jnp.ndarray:
    """Read exactly three Cartesian scalar fields in x/y/z order."""
    if isinstance(spec, Mapping):
        if not set(spec) <= {"x", "y", "z"}:
            raise ValueError("Vector profiles accept only x, y and z components")
        spec = [spec.get(axis, 0.0) for axis in ("x", "y", "z")]
    if isinstance(spec, str) or len(spec) != 3:
        raise ValueError("Vector profiles require three components in x/y/z order")
    result = jnp.stack([profile_2d(value, grid, norm, reference) for value in spec], axis=-1)
    if not np.all(np.isfinite(np.asarray(result))):
        raise ValueError("Initial vector profiles must be finite")
    return result


def initial_magnetic_field(
    spec: Mapping, grid: Grid, norm, *, finite_difference=False, mesh: Mesh | None = None
) -> jnp.ndarray:
    """Return ``curl(A) + B_uniform`` with the evolution's discrete derivatives.

    ``periodic_sheet`` provides an antiparallel Bx reversal across y=``center``
    with a second periodic reversal at the domain seam. It is not an open-boundary
    equilibrium. The optional cosine Az perturbation seeds a central X point.
    A mesh selects the same partition-dependent finite-difference stencil as
    evolution, including its second-order fallback for one x cell per shard.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("initial_conditions.magnetic_field must be a mapping")
    unknown = set(spec) - {"uniform", "vector_potential", "periodic_sheet"}
    if unknown:
        raise ValueError(f"Unknown magnetic initialization keys: {sorted(unknown)}")
    if "vector_potential" in spec and "periodic_sheet" in spec:
        raise ValueError("Choose vector_potential or periodic_sheet, not both")
    b0 = norm.m0 / (norm.q0 * norm.tau)
    a0 = b0 * norm.L0
    uniform = spec.get("uniform", [0.0, 0.0, 0.0])
    if isinstance(uniform, str) or len(uniform) != 3 or any(isinstance(value, Mapping) for value in uniform):
        raise ValueError("magnetic_field.uniform requires three scalar constants")
    b_uniform = jnp.asarray([_quantity(value, b0) for value in uniform])
    finite_difference = finite_difference or mesh is not None
    curl = Maxwell2D(
        grid.kx,
        grid.ky,
        norm.speed_of_light_norm(),
        dx=grid.dx if finite_difference else None,
        dy=grid.dy if finite_difference else None,
        mesh=mesh,
    ).curl
    potential = jnp.zeros((grid.nx, grid.ny, 3))
    if "vector_potential" in spec:
        potential = vector_profile(spec["vector_potential"], grid, norm, a0)
    elif "periodic_sheet" in spec:
        sheet = spec["periodic_sheet"]
        unknown = set(sheet) - {"field", "width", "center", "perturbation", "x_center"}
        if unknown:
            raise ValueError(f"Unknown periodic_sheet keys: {sorted(unknown)}")
        center = normalize(sheet.get("center", 0.5 * (grid.ymin + grid.ymax)), norm, dim="x")
        width = normalize(sheet["width"], norm, dim="x")
        if width < 2 * grid.dy:
            raise ValueError("periodic_sheet width must span at least two y cells")
        wave_number = 2 * jnp.pi / (grid.ymax - grid.ymin)
        bx = _quantity(sheet["field"], b0) * jnp.tanh(jnp.sin(wave_number * (grid.y - center)) / (wave_number * width))
        # Integrate Bx spectrally to a real periodic Az; curl then guarantees
        # discrete div(B)=0 even when the evolution uses the finite-difference
        # stencil. On that path its sampled Bx differs by truncation error.
        nonzero_k = jnp.where(grid.ky == 0, 1.0, grid.ky)
        az_hat = jnp.where(grid.ky == 0, 0.0, jnp.fft.fft(bx) / (1j * nonzero_k))
        az = jnp.broadcast_to(jnp.fft.ifft(az_hat).real[None, :], (grid.nx, grid.ny))
        if "perturbation" in sheet:
            amplitude = _quantity(sheet["perturbation"], a0)
            x_center = normalize(sheet.get("x_center", 0.5 * (grid.xmin + grid.xmax)), norm, dim="x")
            # Az_yy of the sheet is positive at the origin; positive cosine
            # perturbation gives Az_xx<0 and thus an X point there.
            az += (
                amplitude
                * jnp.cos(2 * jnp.pi * (grid.x - x_center) / (grid.xmax - grid.xmin))[:, None]
                * jnp.cos(wave_number * (grid.y - center))[None, :]
            )
        potential = potential.at[..., 2].set(az)
    return curl(potential) + b_uniform
