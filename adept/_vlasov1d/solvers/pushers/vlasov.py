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
    x: jnp.ndarray
    v: jnp.ndarray
    f_interp: Callable
    dt: int
    dummy_x: jnp.ndarray
    dummy_v: jnp.ndarray
    interp_e: Callable

    def __init__(self, cfg, interp_e):
        self.x = cfg["grid"]["x"]
        self.v = cfg["grid"]["v"]
        self.f_interp = partial(interp2d, x=self.x, y=self.v, period=cfg["grid"]["xmax"])
        self.dt = cfg["grid"]["dt"]
        self.dummy_x = jnp.ones_like(self.x)
        self.dummy_v = jnp.ones_like(self.v)[None, :]
        self.interp_e = interp_e

    def step_vdfdx(self, t, f, frac_dt):
        old_x = self.x[:, None] - self.v[None, :] * frac_dt * self.dt
        old_v = self.dummy_x[:, None] * self.v[None, :]
        new_f = self.f_interp(xq=old_x.flatten(), yq=old_v.flatten(), f=f)
        return jnp.reshape(new_f, (self.x.size, self.v.size))

    def step_edfdv(self, t, f, frac_dt):
        interp_e = self.interp_e(self.dummy_x * t, self.x)
        old_v = self.v[None, :] + interp_e[:, None] * frac_dt * self.dt
        old_x = self.dummy_v * self.x[:, None]
        new_f = self.f_interp(xq=old_x.flatten(), yq=old_v.flatten(), f=f)
        return jnp.reshape(new_f, (self.x.size, self.v.size))

    def __call__(self, t, y, args):
        f = y["electron"]

        new_f = self.step_vdfdx(t, f, 0.5)
        new_f = self.step_edfdv(t, new_f, 1.0)
        new_f = self.step_vdfdx(t, new_f, 0.5)

        return {"electron": new_f}


class VelocityExponential:
    def __init__(self, species_grids, species_params, parallel=False):
        self.species_grids = species_grids
        self.species_params = species_params
        self.parallel = parallel
        if parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f_dict, e, pond, dt):
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
    def __init__(self, species_grids, species_params, parallel=False):
        self.species_grids = species_grids
        self.species_params = species_params
        self.interp = vmap(partial(interp1d, extrap=1.0e-30), in_axes=0)
        self.parallel = parallel
        if self.parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f_dict, e, pond, dt):
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
        if self.parallel:
            return shard_map(
                self.push,
                mesh=self.mesh,
                in_specs=(P("device", None), P("device"), P("device"), P()),
                out_specs=P("device", None),
            )(f_dict, e, pond, dt)
        else:
            return self.push(f_dict, e, pond, dt)


def _minmod(a, b):
    """Minmod slope limiter: returns the smaller magnitude slope when signs agree, else zero."""
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


class VelocityMUSCL:
    def __init__(self, species_grids, species_params, parallel=False):
        self.species_grids = species_grids
        self.species_params = species_params
        self.parallel = parallel
        if parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f_dict, e, pond, dt):
        result = {}
        for species_name, f in f_dict.items():
            v = self.species_grids[species_name]["v"]
            dv = self.species_grids[species_name]["dv"]
            q = self.species_params[species_name]["charge"]
            m = self.species_params[species_name]["mass"]
            # acceleration: a[nx]
            force = q * e + (q**2 / m) * pond
            accel = force / m  # shape (nx,)

            nx, nv = f.shape
            # CFL number per spatial point: shape (nx,)
            cfl = accel * dt / dv

            # Slopes between adjacent cells in velocity: shape (nx, nv-1)
            delta = f[:, 1:] - f[:, :-1]

            # Minmod limited slopes: shape (nx, nv)
            # Interior slopes use minmod of left and right differences
            # Boundary slopes are zero (zero-slope extrapolation)
            slopes = jnp.zeros_like(f)
            slopes = slopes.at[:, 1:-1].set(_minmod(delta[:, :-1], delta[:, 1:]))

            # Reconstruct interface values at i+1/2 for i = 0..nv-2
            # f_L[i+1/2] = f[i] + 0.5 * slope[i]    (left state)
            # f_R[i+1/2] = f[i+1] - 0.5 * slope[i+1] (right state)
            f_L = f[:, :-1] + 0.5 * slopes[:, :-1]  # shape (nx, nv-1)
            f_R = f[:, 1:] - 0.5 * slopes[:, 1:]    # shape (nx, nv-1)

            # Upwind flux at each interface: flux[i+1/2]
            # For linear advection with velocity-independent acceleration,
            # the characteristic speed is accel (same at all interfaces)
            a_pos = jnp.maximum(accel, 0.0)[:, None]  # shape (nx, 1)
            a_neg = jnp.minimum(accel, 0.0)[:, None]
            flux = a_pos * f_L + a_neg * f_R  # shape (nx, nv-1)

            # Zero-flux boundaries: pad with zeros at v_min-1/2 and v_max+1/2
            flux_padded = jnp.pad(flux, ((0, 0), (1, 1)), mode="constant")  # shape (nx, nv+1)

            # Conservative update: f_new[i] = f[i] - (dt/dv) * (flux[i+1/2] - flux[i-1/2])
            result[species_name] = f - (dt / dv) * (flux_padded[:, 1:] - flux_padded[:, :-1])

        return result

    def __call__(self, f_dict, e, pond, dt):
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
    numerical oscillations without affecting well-resolved modes. Can filter
    in x (position), v (velocity), or both dimensions.

    The filter kernel in Fourier space is:

        sigma(j) = exp(-alpha * (j / N)^(2*order))

    where j is the mode index, N is the maximum mode, and
    alpha = -log(machine_epsilon) ~ 36 for float64.

    References:
        Hou, T.Y. & Li, R. (2007). Computing nearly singular solutions using
        pseudo-spectral methods. J. Comput. Phys., 226(1), 379-397.
    """

    def __init__(self, species_grids: dict, nx: int, alpha: float, order: int, dimensions: list[str]):
        if "x" in dimensions:
            j_x = jnp.arange(nx // 2 + 1)
            eta_x = j_x / (nx // 2)
            self.filter_x = jnp.exp(-alpha * eta_x ** (2 * order))
        else:
            self.filter_x = None

        self.filters_v = {}
        if "v" in dimensions:
            for species_name, sg in species_grids.items():
                nv = sg["nv"]
                j = jnp.arange(nv // 2 + 1)
                eta = j / (nv // 2)
                self.filters_v[species_name] = jnp.exp(-alpha * eta ** (2 * order))

    def __call__(self, f_dict: dict) -> dict:
        result = {}
        for species_name, f in f_dict.items():
            filtered = f

            if self.filter_x is not None:
                filtered = jnp.real(jnp.fft.irfft(self.filter_x[:, None] * jnp.fft.rfft(filtered, axis=0), axis=0))

            if species_name in self.filters_v:
                sigma_v = self.filters_v[species_name]
                filtered = jnp.real(jnp.fft.irfft(sigma_v[None, :] * jnp.fft.rfft(filtered, axis=1), axis=1))

            result[species_name] = filtered
        return result


class SpaceExponential:
    def __init__(self, x, species_grids, parallel=False):
        self.kx_real = jnp.fft.rfftfreq(len(x), d=x[1] - x[0]) * 2 * jnp.pi
        self.species_grids = species_grids
        self.parallel = parallel
        if parallel:
            self.mesh = Mesh(np.array(jax.devices()), ("device",))

    def push(self, f, v):
        return jnp.real(
            jnp.fft.irfft(jnp.exp(-1j * self.kx_real[:, None] * v[None, :]) * jnp.fft.rfft(f, axis=0), axis=0)
        )

    def __call__(self, f_dict, dt):
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
