"""Vlasov advection pushers for the one-dimensional phase-space solver."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import numpy as np
from interpax import interp1d, interp2d
from jax import numpy as jnp
from jax import shard_map, vmap
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


class VelocityCubicSpline:
    """Cubic-spline velocity-space advection under electric and ponderomotive forces."""

    def __init__(self, species_grids, species_params, parallel=False):
        """Store per-species velocity grids, interpolation kernel, and sharding metadata."""
        self.species_grids = species_grids
        self.species_params = species_params
        self.interp = vmap(partial(interp1d, extrap=1.0e-30), in_axes=0)
        self.parallel = parallel
        if self.parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f_dict, e, pond, dt):
        """Apply the unsharded cubic-spline velocity push to each species."""
        result = {}
        for species_name, f in f_dict.items():
            v = self.species_grids[species_name]["v"]
            q = self.species_params[species_name]["charge"]
            m = self.species_params[species_name]["mass"]
            nx = f.shape[0]
            v_repeated = jnp.repeat(v[None, :], repeats=nx, axis=0)
            force = q * e + (q**2 / m) * pond
            accel = force / m
            vq = v_repeated - accel[:, None] * dt
            result[species_name] = self.interp(xq=vq, x=v_repeated, f=f)
        return result

    def __call__(self, f_dict, e, pond, dt):
        """Dispatch the cubic-spline velocity push, optionally through shard_map."""
        if self.parallel:
            return shard_map(
                self.push,
                mesh=self.mesh,
                in_specs=(P("device", None), P("device"), P("device"), P()),
                out_specs=P("device", None),
            )(f_dict, e, pond, dt)
        else:
            return self.push(f_dict, e, pond, dt)


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
