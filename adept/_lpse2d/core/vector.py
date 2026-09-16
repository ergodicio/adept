"""Component-axis helpers for the light fields ``(nx, ny, nc)``.

The pump ``E0`` and the Raman / combined field ``E1`` carry ``nc = 3`` components
``(x, y, z)`` (LPSE's ``XcComplex3`` on any grid; plan 2 F.1). The grid is 2-D, so
``k_z = 0``: the z component has no longitudinal part -- every k-space projector below
acts on components 0 and 1 and passes the rest through as transverse -- and it never
enters the EPW potential (``phi_k = i k . E_k / k^2``) or the TPD source (``E_h`` is
in-plane). A field with ``nc = 2`` (the pre-F.1 layout, still used by tests) is handled
identically; with ``E_z = 0`` every result is bit-identical to the two-component code.
"""

from jax import Array
from jax import numpy as jnp


def fft2c(field: Array) -> Array:
    """2-D forward FFT over the grid axes of an ``(nx, ny, nc)`` field."""
    return jnp.fft.fft2(field, axes=(0, 1))


def ifft2c(field_k: Array) -> Array:
    return jnp.fft.ifft2(field_k, axes=(0, 1))


def k_dot(field_k: Array, kx: Array, ky: Array) -> Array:
    """``k . F_k`` (nx, ny) -- only the in-plane components contribute since ``k_z = 0``."""
    return kx[:, None] * field_k[..., 0] + ky[None, :] * field_k[..., 1]


def split_k(field_k: Array, kx: Array, ky: Array, one_over_k_sq: Array) -> tuple[Array, Array]:
    """Longitudinal and transverse parts of a k-space field: ``L = k (k . F) / k^2``,
    ``T = F - L``; ``L`` has no z component."""
    kdote = k_dot(field_k, kx, ky) * one_over_k_sq
    longitudinal = jnp.zeros_like(field_k)
    longitudinal = longitudinal.at[..., 0].set(kx[:, None] * kdote).at[..., 1].set(ky[None, :] * kdote)
    return longitudinal, field_k - longitudinal


def transverse_part(field: Array, kx: Array, ky: Array, one_over_k_sq: Array) -> Array:
    """The transverse (divergence-free) part of an x-space vector field, via the k-space
    projector ``F - k (k . F) / k^2`` (LPSE ``LightSolver::getTransversePartOfSourceTerm``)."""
    _, transverse_k = split_k(fft2c(field), kx, ky, one_over_k_sq)
    return ifft2c(transverse_k)


def dot_conj(a: Array, b: Array) -> Array:
    """``a . conj(b)`` summed over the component axis, (nx, ny)."""
    return jnp.sum(a * jnp.conj(b), axis=-1)


def with_components(field: Array, nc: int) -> Array:
    """``field`` with its component axis padded with zeros (or truncated) to ``nc``."""
    have = field.shape[-1]
    if have == nc:
        return field
    if have < nc:
        pad = [(0, 0)] * (field.ndim - 1) + [(0, nc - have)]
        return jnp.pad(field, pad)
    return field[..., :nc]


def in_plane_to_field(fx: Array, fy: Array, like: Array | None = None, nc: int | None = None) -> Array:
    """Stack in-plane components into an ``(nx, ny, nc)`` field, ``nc`` from ``like`` (or
    given); the z component (when present) is zero."""
    nc = like.shape[-1] if like is not None else (nc or 2)
    parts = [fx, fy] + [jnp.zeros_like(fx)] * (nc - 2)
    return jnp.stack(parts, axis=-1)
