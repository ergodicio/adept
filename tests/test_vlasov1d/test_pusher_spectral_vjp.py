"""Tests for the custom VJP on the spectral velocity push.

``_shift_spectrum`` replaces a plain ``exp(...) * spectrum`` with an identical primal that
rebuilds the phase factor on the backward pass instead of storing it. These tests pin the
two properties that make that substitution safe — the primal and its gradients must be
*unchanged* — and the property that makes it worth doing, namely that the reverse pass
stops carrying the phase factor as a residual.
"""

import subprocess
import sys

import jax
import numpy as np
import pytest
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

from adept._vlasov1d.solvers.pushers.vlasov import VelocityExponential, _shift_spectrum

NX, NV, DT = 32, 256, 0.1


def _grids():
    """Return the wavenumber grid and a reproducible (f, accel) pair."""
    kv = jnp.fft.rfftfreq(NV, d=12.8 / NV) * 2 * jnp.pi
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.normal(size=(NX, NV)))
    accel = jnp.asarray(rng.normal(size=NX)) * 0.1
    return kv, f, accel


def _push_naive(f, kv, accel):
    """The expression `_shift_spectrum` replaced, written out inline."""
    return jnp.real(jnp.fft.irfft(jnp.exp(-1j * kv[None, :] * DT * accel[:, None]) * jnp.fft.rfft(f, axis=1), axis=1))


def _push_custom(f, kv, accel):
    """The same push routed through the custom-VJP helper."""
    return jnp.real(jnp.fft.irfft(_shift_spectrum(jnp.fft.rfft(f, axis=1), kv, DT, accel), axis=1))


def _loss(push):
    """A scalar loss whose gradient exercises both the `f` and `accel` paths."""

    def loss(f, kv, accel):
        return jnp.sum(push(f, kv, accel) ** 2 * jnp.cos(jnp.arange(NV)[None, :] / NV))

    return loss


def _residual_bytes(fn, *args):
    """Bytes autodiff keeps alive between the forward and backward passes of `fn`.

    The residuals of a linearized function are exactly the constants closed over by its
    linear part, so this reads them straight off the jaxpr. That makes the measurement a
    property of JAX's partial evaluation rather than of any particular XLA backend.
    """
    _, linear = jax.linearize(fn, *args)
    jaxpr = jax.make_jaxpr(linear)(*jax.tree.map(jnp.zeros_like, args))
    return sum(jnp.asarray(c).size * jnp.asarray(c).dtype.itemsize for c in jaxpr.consts)


def test_forward_is_bit_identical_to_naive():
    """The custom VJP must not perturb the primal, not even in the last bit."""
    kv, f, accel = _grids()
    assert jnp.array_equal(_push_naive(f, kv, accel), _push_custom(f, kv, accel))


@pytest.mark.parametrize("argnum, name", [(0, "f"), (2, "accel")])
def test_gradients_are_bit_identical_to_naive(argnum, name):
    """Gradients must match the inline expression exactly, through both input paths.

    Bit-identity rather than a tolerance: the backward rule differentiates a recomputation
    of the *same* primal expression, so any discrepancy at all would mean the recomputation
    has drifted from what the forward pass actually did.
    """
    kv, f, accel = _grids()
    expected = jax.grad(_loss(_push_naive), argnums=argnum)(f, kv, accel)
    actual = jax.grad(_loss(_push_custom), argnums=argnum)(f, kv, accel)
    assert jnp.array_equal(expected, actual), f"gradient wrt {name} changed"


def test_phase_factor_is_not_kept_as_a_residual():
    """The reverse pass should carry the spectrum but no longer the phase factor.

    Both are ``(nx, nv // 2 + 1)`` complex128, so dropping one of the two is a ~2x cut. The
    assertion is a band rather than an equality so that unrelated small residuals elsewhere
    in the expression do not make it brittle.
    """
    kv, f, accel = _grids()
    naive = _residual_bytes(lambda a, b: _push_naive(a, kv, b), f, accel)
    custom = _residual_bytes(lambda a, b: _push_custom(a, kv, b), f, accel)

    phase_bytes = NX * (NV // 2 + 1) * jnp.complex128.dtype.itemsize
    assert custom <= naive - phase_bytes, f"phase factor still resident: {naive} -> {custom} bytes"
    assert custom < 0.6 * naive


def test_velocity_exponential_gradients_match_naive_end_to_end():
    """Drive the real pusher class and compare against the inline expression."""
    kv, f, accel = _grids()
    species_grids = {"electron": {"kvr": kv}}
    species_params = {"electron": {"charge": -1.0, "mass": 1.0}}
    pusher = VelocityExponential(species_grids, species_params)

    # `push` takes the field, not the acceleration: with q = -1 and m = 1, accel = -e.
    def via_class(e_field):
        return jnp.sum(pusher.push({"electron": f}, e=e_field, pond=jnp.zeros(NX), dt=DT)["electron"] ** 2)

    def via_naive(e_field):
        return jnp.sum(_push_naive(f, kv, -e_field) ** 2)

    e = -accel
    assert jnp.allclose(jax.grad(via_class)(e), jax.grad(via_naive)(e), rtol=0, atol=0)


def test_custom_vjp_composes_with_shard_map():
    """The sharded push must still trace and differentiate under `shard_map`.

    `parallel: [x]` (used by the large IAW configs) wraps `VelocityExponential.push` in a
    `shard_map`, so the custom VJP has to survive that composition. Runs in a subprocess
    because faking multiple devices requires XLA flags set before JAX initializes.
    """
    script = """
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax import numpy as jnp
from adept._vlasov1d.solvers.pushers.vlasov import VelocityExponential

nx, nv, dt = 32, 256, 0.1
kv = jnp.fft.rfftfreq(nv, d=12.8 / nv) * 2 * jnp.pi
rng = np.random.default_rng(0)
f = jnp.asarray(rng.normal(size=(nx, nv)))
e = jnp.asarray(rng.normal(size=nx)) * 0.1
pond = jnp.zeros(nx)

grids = {"electron": {"kvr": kv}}
params = {"electron": {"charge": -1.0, "mass": 1.0}}
serial = VelocityExponential(grids, params, parallel=False)
sharded = VelocityExponential(grids, params, parallel=True)

def loss(pusher):
    return lambda ef: jnp.sum(pusher({"electron": f}, e=ef, pond=pond, dt=dt)["electron"] ** 2)

assert len(jax.devices()) == 4, jax.devices()
np.testing.assert_allclose(loss(serial)(e), loss(sharded)(e), rtol=1e-12)
np.testing.assert_allclose(
    jax.grad(loss(serial))(e), jax.grad(loss(sharded))(e), rtol=1e-10, atol=1e-12
)
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "ok" in result.stdout
