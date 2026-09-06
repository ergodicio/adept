"""Vlasov advection pushers for the one-dimensional phase-space solver."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import numpy as np
from interpax import interp2d
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


class VlasovExternalE(eqx.Module):
    """Split Vlasov pusher for externally supplied electric fields."""

    x: jnp.ndarray
    v: jnp.ndarray
    f_interp: Callable
    dt: int
    dummy_x: jnp.ndarray
    dummy_v: jnp.ndarray
    interp_e: Callable

    def __init__(self, cfg, interp_e):
        """Create interpolation helpers for x-advection and v-advection steps."""
        self.x = cfg["grid"]["x"]
        self.v = cfg["grid"]["v"]
        self.f_interp = partial(interp2d, x=self.x, y=self.v, period=cfg["grid"]["xmax"] - cfg["grid"]["xmin"])
        self.dt = cfg["grid"]["dt"]
        self.dummy_x = jnp.ones_like(self.x)
        self.dummy_v = jnp.ones_like(self.v)[None, :]
        self.interp_e = interp_e

    def step_vdfdx(self, t, f, frac_dt):
        """Advect the distribution along x characteristics for a fractional step."""
        old_x = self.x[:, None] - self.v[None, :] * frac_dt * self.dt
        old_v = self.dummy_x[:, None] * self.v[None, :]
        new_f = self.f_interp(xq=old_x.flatten(), yq=old_v.flatten(), f=f)
        return jnp.reshape(new_f, (self.x.size, self.v.size))

    def step_edfdv(self, t, f, frac_dt):
        """Advect the distribution along v characteristics using the external field."""
        interp_e = self.interp_e(self.dummy_x * t, self.x)
        old_v = self.v[None, :] + interp_e[:, None] * frac_dt * self.dt
        old_x = self.dummy_v * self.x[:, None]
        new_f = self.f_interp(xq=old_x.flatten(), yq=old_v.flatten(), f=f)
        return jnp.reshape(new_f, (self.x.size, self.v.size))

    def __call__(self, t, y, args):
        """Apply Strang splitting for one externally driven Vlasov step."""
        f = y["electron"]

        new_f = self.step_vdfdx(t, f, 0.5)
        new_f = self.step_edfdv(t, new_f, 1.0)
        new_f = self.step_vdfdx(t, new_f, 0.5)

        return {"electron": new_f}


class VelocityExponential:
    """Spectral velocity-space advection under electric and ponderomotive forces."""

    def __init__(self, species_grids, species_params, parallel=False):
        """Store per-species velocity grids and optional sharding metadata."""
        self.species_grids = species_grids
        self.species_params = species_params
        self.parallel = parallel
        if parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f_dict, e, pond, dt):
        """Apply the unsharded spectral velocity push to each species."""
        result = {}
        for species_name, f in f_dict.items():
            kv_real = self.species_grids[species_name]["kvr"]
            q = self.species_params[species_name]["charge"]
            m = self.species_params[species_name]["mass"]
            # force = q*E + (q²/m)*pond where pond = -(1/2)*grad(a²)
            # accel = force / m
            force = q * e + (q**2 / m) * pond
            accel = force / m
            result[species_name] = jnp.real(
                jnp.fft.irfft(
                    jnp.exp(-1j * kv_real[None, :] * dt * accel[:, None]) * jnp.fft.rfft(f, axis=1),
                    axis=1,
                )
            )
        return result

    def __call__(self, f_dict, e, pond, dt):
        """Dispatch the velocity push, optionally through shard_map."""
        if self.parallel:
            return shard_map(
                self.push,
                mesh=self.mesh,
                in_specs=(P("device", None), P("device"), P("device"), P()),
                out_specs=P("device", None),
            )(f_dict, e, pond, dt)
        else:
            return self.push(f_dict, e, pond, dt)


def _uniform_cubic_interp(f: jnp.ndarray, shift: jnp.ndarray, dv: float) -> jnp.ndarray:
    """Shift each row of ``f`` on a uniform grid with local cubic splines.

    This is the uniform-grid specialization of ``interpax.interp1d(method="cubic")``
    used by the velocity pusher. Each row has one constant velocity shift, so its
    cell offset and fractional displacement are shared by every velocity point.
    The direct four-point stencil avoids constructing query grids, binary searches,
    and a full array of spline derivatives.
    """
    _, nv = f.shape
    if nv < 2:
        raise ValueError("cubic interpolation requires at least two velocity cells")

    scaled_shift = shift / jnp.asarray(dv, dtype=f.dtype)
    velocity_index = jnp.arange(nv, dtype=jnp.int32)[None, :]

    # For query index u = j - shift/dv, floor(u) separates into the velocity
    # index j and one row-wise offset. Clipping selects the endpoint segment;
    # t is then bounded to that segment so exterior queries cannot overflow.
    row_offset = jnp.floor(-scaled_shift).astype(jnp.int32)[:, None]
    left = jnp.clip(velocity_index + row_offset, 0, nv - 2)
    query_index = velocity_index - scaled_shift[:, None]
    t = jnp.clip(query_index - left, 0.0, 1.0)

    fm1 = jnp.take_along_axis(f, jnp.clip(left - 1, 0, nv - 1), axis=1)
    f0 = jnp.take_along_axis(f, left, axis=1)
    f1 = jnp.take_along_axis(f, left + 1, axis=1)
    f2 = jnp.take_along_axis(f, jnp.clip(left + 2, 0, nv - 1), axis=1)

    # interpax's local cubic method uses one-sided endpoint slopes and the
    # average of adjacent secant slopes in the interior. Multiplication by dv
    # is folded into m0/m1, leaving differences of f values directly.
    m0 = jnp.where(left == 0, f1 - f0, 0.5 * (f1 - fm1))
    m1 = jnp.where(left == nv - 2, f1 - f0, 0.5 * (f2 - f0))

    t2 = t * t
    t3 = t2 * t
    interpolated = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * f0 + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * f1 + (t3 - t2) * m1
    )

    outside = (query_index < 0.0) | (query_index > nv - 1)
    return jnp.where(outside, jnp.asarray(1.0e-30, dtype=f.dtype), interpolated)


def _uniform_lagrange7_interp(f: jnp.ndarray, shift: jnp.ndarray, dv: float) -> jnp.ndarray:
    """Translate uniform-grid rows with an eight-point, degree-7 Lagrange stencil.

    Weights depend only on the row's fractional displacement. The stencil is
    centered around the departure cell, with exterior samples and exterior
    queries set to 1e-30 (no velocity wraparound). This is an unlimited
    interpolant: it does not enforce positivity or renormalize escaped mass.
    """
    _, nv = f.shape
    if nv < 8:
        raise ValueError("lagrange7 interpolation requires at least eight velocity cells")

    # Once a displacement exceeds the domain plus the stencil width, every
    # query is exterior. Bound it before conversion to avoid integer overflow.
    displacement = jnp.clip(-shift / jnp.asarray(dv, dtype=f.dtype), -nv - 8, nv + 8)
    offset = jnp.floor(displacement).astype(jnp.int32)
    fraction = displacement - offset
    cell = jnp.arange(nv, dtype=jnp.int32)[None, :] + offset[:, None]
    floor = jnp.asarray(1.0e-30, dtype=f.dtype)
    interpolated = jnp.zeros_like(f)

    for node in range(-3, 5):
        weight = jnp.ones_like(fraction)
        for other in range(-3, 5):
            if other != node:
                weight = weight * (fraction - other) / (node - other)
        index = cell + node
        samples = jnp.take_along_axis(f, jnp.clip(index, 0, nv - 1), axis=1)
        samples = jnp.where((index >= 0) & (index < nv), samples, floor)
        interpolated = interpolated + weight[:, None] * samples

    query = cell + fraction[:, None]
    return jnp.where((query >= 0) & (query <= nv - 1), interpolated, floor)


class _VelocityInterpolation:
    """Shared force calculation and sharding for velocity interpolation pushers."""

    def __init__(self, species_grids, species_params, parallel=False):
        """Store per-species velocity grids and optional sharding metadata."""
        self.species_grids = species_grids
        self.species_params = species_params
        self.parallel = parallel
        if self.parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f_dict, e, pond, dt):
        """Apply the unsharded interpolation push to each species."""
        result = {}
        for species_name, f in f_dict.items():
            dv = self.species_grids[species_name]["dv"]
            q = self.species_params[species_name]["charge"]
            m = self.species_params[species_name]["mass"]
            force = q * e + (q**2 / m) * pond
            accel = force / m
            result[species_name] = self.interpolate(f, accel * dt, dv)
        return result

    def __call__(self, f_dict, e, pond, dt):
        """Dispatch the velocity push, optionally through shard_map."""
        if self.parallel:
            return shard_map(
                self.push,
                mesh=self.mesh,
                in_specs=(P("device", None), P("device"), P("device"), P()),
                out_specs=P("device", None),
            )(f_dict, e, pond, dt)
        else:
            return self.push(f_dict, e, pond, dt)


class VelocityCubicSpline(_VelocityInterpolation):
    """Local cubic-spline velocity advection under electric and ponderomotive forces."""

    interpolate = staticmethod(_uniform_cubic_interp)


class VelocityLagrange7(_VelocityInterpolation):
    """Degree-7 Lagrange velocity advection under electric and ponderomotive forces."""

    interpolate = staticmethod(_uniform_lagrange7_interp)


class HouLiFilter:
    """Hou-Li spectral filter.

    Applies the exponential filter from Hou & Li (2007) to damp high-frequency
    numerical oscillations without affecting well-resolved modes.

    Configuration-space (x) ONLY. Velocity-space filtering was removed: the FFT-based
    filter is periodic in v, so it wraps the forward tail onto the -v edge and corrupts
    f(v). It must never be applied in velocity space.

    The filter kernel in Fourier space is:

        sigma(j) = exp(-alpha * (j / N)^(2*order))

    where j is the mode index, N is the maximum mode, and
    alpha = -log(machine_epsilon) ~ 36 for float64.

    References:
        Hou, T.Y. & Li, R. (2007). Computing nearly singular solutions using
        pseudo-spectral methods. J. Comput. Phys., 226(1), 379-397.
    """

    def __init__(self, nx: int, alpha: float, order: int):
        """Precompute the x (configuration-space) Fourier filter kernel."""
        j_x = jnp.arange(nx // 2 + 1)
        eta_x = j_x / (nx // 2)
        self.filter_x = jnp.exp(-alpha * eta_x ** (2 * order))

    def __call__(self, f_dict: dict) -> dict:
        """Apply the x-space Hou-Li filter to each species distribution."""
        result = {}
        for species_name, f in f_dict.items():
            result[species_name] = jnp.real(jnp.fft.irfft(self.filter_x[:, None] * jnp.fft.rfft(f, axis=0), axis=0))
        return result


class SpaceExponential:
    """Spectral configuration-space advection for each species."""

    def __init__(self, x, species_grids, parallel=False):
        """Precompute x-space wavenumbers and optional velocity-axis sharding metadata."""
        self.kx_real = jnp.fft.rfftfreq(len(x), d=x[1] - x[0]) * 2 * jnp.pi
        self.species_grids = species_grids
        self.parallel = parallel
        if parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f, v):
        """Apply the spectral x-advection update for one species distribution."""
        return jnp.real(
            jnp.fft.irfft(jnp.exp(-1j * self.kx_real[:, None] * v[None, :]) * jnp.fft.rfft(f, axis=0), axis=0)
        )

    def __call__(self, f_dict, dt):
        """Advect every species in configuration space for one timestep."""
        result = {}
        for species_name, f in f_dict.items():
            v = self.species_grids[species_name]["v"] * dt
            if self.parallel:
                result[species_name] = shard_map(
                    self.push, mesh=self.mesh, in_specs=(P(None, "device"), P("device")), out_specs=P(None, "device")
                )(f, v)
            else:
                result[species_name] = self.push(f, v)
        return result
