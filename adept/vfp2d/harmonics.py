"""Spherical-harmonic layout and Vlasov operators for VFP-2D.

The normalization and complex ``f[l, m]`` convention follow Tzoufras et al.,
J. Comput. Phys. 230 (2011), equations (5)--(24).  Only non-negative ``m``
are stored because a real distribution satisfies ``f[l, -m] = conj(f[l,m])``.

Arrays use the compact layout ``(..., harmonic, speed)``.  This is much more
amenable to JAX transformations than a nested ``dict[l][m]`` PyTree and allows
``m_max`` to be chosen independently of ``l_max``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass(frozen=True)
class HarmonicLayout:
    """Packed indexing for ``0 <= l <= l_max`` and ``0 <= m <= min(l,m_max)``."""

    l_max: int
    m_max: int
    pairs: tuple[tuple[int, int], ...]
    _indices: tuple[tuple[int, ...], ...]

    def __init__(self, l_max: int, m_max: int | None = None):
        if l_max < 0:
            raise ValueError("l_max must be non-negative")
        if m_max is None:
            m_max = l_max
        if not 0 <= m_max <= l_max:
            raise ValueError("m_max must satisfy 0 <= m_max <= l_max")

        pairs = tuple((ell, m) for ell in range(l_max + 1) for m in range(min(ell, m_max) + 1))
        lookup = {pair: i for i, pair in enumerate(pairs)}
        rows = tuple(tuple(lookup.get((ell, m), -1) for m in range(m_max + 1)) for ell in range(l_max + 1))
        object.__setattr__(self, "l_max", l_max)
        object.__setattr__(self, "m_max", m_max)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "_indices", rows)

    @property
    def size(self) -> int:
        return len(self.pairs)

    @property
    def ell(self) -> np.ndarray:
        return np.asarray([ell for ell, _ in self.pairs], dtype=np.int32)

    @property
    def m(self) -> np.ndarray:
        return np.asarray([m for _, m in self.pairs], dtype=np.int32)

    def index(self, ell: int, m: int) -> int:
        """Return the packed index, or ``-1`` when the mode is truncated."""

        if ell < 0 or ell > self.l_max or m < 0 or m > self.m_max:
            return -1
        return self._indices[ell][m]


def _spectral_derivative(a: Array, k: Array, axis: int) -> Array:
    """Periodic spectral derivative, preserving complex-valued harmonics."""

    shape = [1] * a.ndim
    shape[axis] = k.size
    multiplier = 1j * k.reshape(shape)
    return jnp.fft.ifft(multiplier * jnp.fft.fft(a, axis=axis), axis=axis)


class TzoufrasVlasov:
    """Arbitrary-``f_lm`` 2D3P Vlasov operator from Tzoufras (2011).

    Configuration space is periodic in ``x`` and ``y``.  Momentum space is
    represented by a positive, cell-centred speed grid and a truncated complex
    spherical-harmonic expansion.  The third configuration-space derivative is
    zero, while all three components of ``E`` and ``B`` are retained.
    """

    def __init__(
        self,
        layout: HarmonicLayout,
        v: Array,
        dv: float,
        kx: Array,
        ky: Array,
        streaming_speed: Array | None = None,
    ):
        self.layout = layout
        self.v = jnp.asarray(v)
        self.dv = float(dv)
        self.kx = jnp.asarray(kx)
        self.ky = jnp.asarray(ky)
        self.streaming_speed = self.v if streaming_speed is None else jnp.asarray(streaming_speed)

    def ddv(self, f: Array, ell: int) -> Array:
        """Centred radial derivative with the regularity parity ``f_l(-v)=(-1)^l f_l(v)``."""

        left = ((-1) ** ell) * f[..., :1]
        right = jnp.zeros_like(f[..., :1])
        padded = jnp.concatenate((left, f, right), axis=-1)
        return (padded[..., 2:] - padded[..., :-2]) / (2.0 * self.dv)

    def gh(self, f: Array) -> tuple[Array, Array]:
        """Return the radial operators ``G_l`` and ``H_l`` from Eqs. (20)--(22)."""

        g = jnp.zeros_like(f)
        h = jnp.zeros_like(f)
        inv_v = 1.0 / self.v
        for i, (ell, _m) in enumerate(self.layout.pairs):
            derivative = self.ddv(f[..., i, :], ell)
            g = g.at[..., i, :].set(derivative - ell * inv_v * f[..., i, :])
            h = h.at[..., i, :].set(derivative + (ell + 1) * inv_v * f[..., i, :])
        return g, h

    def streaming(self, f: Array, dfdz: Array | None = None) -> Array:
        """Spatial-advection contribution from Tzoufras Eqs. (17)--(19).

        Configuration space is evolved in x and y. ``dfdz`` optionally supplies
        a prescribed derivative in the unresolved direction, which is useful
        for an integrable 2.5D density gradient without allocating a z grid.
        """

        dfdx = _spectral_derivative(f, self.kx, axis=0)
        dfdy = _spectral_derivative(f, self.ky, axis=1)
        if dfdz is None:
            dfdz = jnp.zeros_like(f)
        transverse_minus = dfdy - 1j * dfdz
        transverse_plus = dfdy + 1j * dfdz
        out = jnp.zeros_like(f)
        v = self.streaming_speed

        for target, (ell, m) in enumerate(self.layout.pairs):
            value = jnp.zeros_like(f[..., target, :])
            lower = self.layout.index(ell - 1, m)
            upper = self.layout.index(ell + 1, m)
            if lower >= 0:
                value -= v * (ell - m) / (2 * ell - 1) * dfdx[..., lower, :]
            if upper >= 0:
                value -= v * (ell + m + 1) / (2 * ell + 3) * dfdx[..., upper, :]

            if m > 0:
                lm = self.layout.index(ell - 1, m - 1)
                lp = self.layout.index(ell - 1, m + 1)
                um = self.layout.index(ell + 1, m - 1)
                up = self.layout.index(ell + 1, m + 1)
                if lm >= 0:
                    value -= 0.5 * v / (2 * ell - 1) * transverse_minus[..., lm, :]
                if lp >= 0:
                    value += 0.5 * v * (ell - m) * (ell - m - 1) / (2 * ell - 1) * transverse_plus[..., lp, :]
                if um >= 0:
                    value += 0.5 * v / (2 * ell + 3) * transverse_minus[..., um, :]
                if up >= 0:
                    value -= 0.5 * v * (ell + m + 1) * (ell + m + 2) / (2 * ell + 3) * transverse_plus[..., up, :]
            else:
                lower1 = self.layout.index(ell - 1, 1)
                upper1 = self.layout.index(ell + 1, 1)
                transverse = jnp.zeros_like(value)
                if lower1 >= 0:
                    transverse -= ell * (ell - 1) / (2 * ell - 1) * transverse_plus[..., lower1, :]
                if upper1 >= 0:
                    transverse += (ell + 1) * (ell + 2) / (2 * ell + 3) * transverse_plus[..., upper1, :]
                value -= v * jnp.real(transverse)

            out = out.at[..., target, :].set(value)
        return out

    def electric(self, f: Array, electric_field: Array) -> Array:
        """Electric-force contribution, Eqs. (20)--(22)."""

        g, h = self.gh(f)
        ex = electric_field[..., 0, None]
        ey_minus_iez = (electric_field[..., 1] - 1j * electric_field[..., 2])[..., None]
        ey_plus_iez = (electric_field[..., 1] + 1j * electric_field[..., 2])[..., None]
        out = jnp.zeros_like(f)

        for target, (ell, m) in enumerate(self.layout.pairs):
            value = jnp.zeros_like(f[..., target, :])
            lower = self.layout.index(ell - 1, m)
            upper = self.layout.index(ell + 1, m)
            if lower >= 0:
                value += ex * (ell - m) / (2 * ell - 1) * g[..., lower, :]
            if upper >= 0:
                value += ex * (ell + m + 1) / (2 * ell + 3) * h[..., upper, :]

            if m > 0:
                lm = self.layout.index(ell - 1, m - 1)
                lp = self.layout.index(ell - 1, m + 1)
                um = self.layout.index(ell + 1, m - 1)
                up = self.layout.index(ell + 1, m + 1)
                if lm >= 0:
                    value += 0.5 * ey_minus_iez / (2 * ell - 1) * g[..., lm, :]
                if lp >= 0:
                    value -= 0.5 * ey_plus_iez * (ell - m) * (ell - m - 1) / (2 * ell - 1) * g[..., lp, :]
                if um >= 0:
                    value -= 0.5 * ey_minus_iez / (2 * ell + 3) * h[..., um, :]
                if up >= 0:
                    value += 0.5 * ey_plus_iez * (ell + m + 1) * (ell + m + 2) / (2 * ell + 3) * h[..., up, :]
            else:
                lower1 = self.layout.index(ell - 1, 1)
                upper1 = self.layout.index(ell + 1, 1)
                transverse = jnp.zeros_like(value)
                if lower1 >= 0:
                    transverse -= ell * (ell - 1) / (2 * ell - 1) * g[..., lower1, :]
                if upper1 >= 0:
                    transverse += (ell + 1) * (ell + 2) / (2 * ell + 3) * h[..., upper1, :]
                value += jnp.real(ey_plus_iez * transverse)

            out = out.at[..., target, :].set(value)
        return out

    def magnetic(self, f: Array, magnetic_field: Array) -> Array:
        """Magnetic-rotation contribution, Eqs. (23)--(24)."""

        bx = magnetic_field[..., 0, None]
        bz_minus_iby = (magnetic_field[..., 2] - 1j * magnetic_field[..., 1])[..., None]
        bz_plus_iby = (magnetic_field[..., 2] + 1j * magnetic_field[..., 1])[..., None]
        out = jnp.zeros_like(f)

        for target, (ell, m) in enumerate(self.layout.pairs):
            value = jnp.zeros_like(f[..., target, :])
            if m > 0:
                value -= 1j * bx * m * f[..., target, :]
                plus = self.layout.index(ell, m + 1)
                minus = self.layout.index(ell, m - 1)
                if plus >= 0:
                    value += 0.5 * (ell - m) * (ell + m + 1) * bz_minus_iby * f[..., plus, :]
                if minus >= 0:
                    value -= 0.5 * bz_plus_iby * f[..., minus, :]
            elif ell > 0:
                one = self.layout.index(ell, 1)
                if one >= 0:
                    value += ell * (ell + 1) * jnp.real(bz_minus_iby * f[..., one, :])
            out = out.at[..., target, :].set(value)
        return out

    def __call__(
        self,
        f: Array,
        electric_field: Array,
        magnetic_field: Array,
        dfdz: Array | None = None,
    ) -> Array:
        result = self.streaming(f, dfdz=dfdz) + self.electric(f, electric_field) + self.magnetic(f, magnetic_field)
        # m=0 coefficients represent real surface harmonics. Project away
        # roundoff-level imaginary parts so the invariant is explicit.
        for i, (_ell, m) in enumerate(self.layout.pairs):
            if m == 0:
                result = result.at[..., i, :].set(jnp.real(result[..., i, :]))
        return result


def density(f: Array, layout: HarmonicLayout, v: Array, dv: float) -> Array:
    """Electron number density, Eq. (10) with ``g=1``."""

    f00 = f[..., layout.index(0, 0), :]
    return 4.0 * jnp.pi * jnp.sum(jnp.real(f00) * v**2, axis=-1) * dv


def conservative_f00_positivity(f: Array, layout: HarmonicLayout, v: Array, dv: float) -> Array:
    """Clip negative ``f00`` cells while preserving each spatial density.

    Explicit harmonic transport is not positivity preserving. Small high-speed
    undershoots can therefore grow until the Coulomb collision coefficients are
    undefined. The angular average must be non-negative physically; after
    clipping, rescale its positive part so the ``4 pi integral(f00 v^2 dv)``
    density is unchanged wherever the pre-projection density is positive.
    """

    i00 = layout.index(0, 0)
    f00 = jnp.real(f[..., i00, :])
    weights = 4.0 * jnp.pi * jnp.asarray(v) ** 2 * float(dv)
    target_density = jnp.sum(f00 * weights, axis=-1, keepdims=True)
    positive = jnp.maximum(f00, 0.0)
    positive_density = jnp.sum(positive * weights, axis=-1, keepdims=True)
    tiny = jnp.finfo(f00.dtype).tiny
    scale = jnp.where(
        (target_density > 0.0) & (positive_density > tiny),
        target_density / jnp.maximum(positive_density, tiny),
        0.0,
    )
    return f.at[..., i00, :].set((positive * scale).astype(f.dtype))


def current(
    f: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    charge: float = -1.0,
    streaming_speed: Array | None = None,
) -> Array:
    """Current vector from the ``l=1`` modes, using Eq. (11)."""

    shape = f.shape[:-2]
    i10 = layout.index(1, 0)
    i11 = layout.index(1, 1)
    if i10 < 0:
        return jnp.zeros((*shape, 3), dtype=jnp.real(f).dtype)
    speed = v if streaming_speed is None else streaming_speed
    weight = v**2 * speed
    m10 = jnp.sum(jnp.real(f[..., i10, :]) * weight, axis=-1) * dv
    if i11 >= 0:
        m11 = jnp.sum(f[..., i11, :] * weight, axis=-1) * dv
    else:
        m11 = jnp.zeros_like(m10, dtype=f.dtype)
    velocity_moment = (4.0 * jnp.pi / 3.0) * jnp.stack((m10, 2.0 * jnp.real(m11), -2.0 * jnp.imag(m11)), axis=-1)
    return charge * velocity_moment


def scalar_velocity_moment(f: Array, layout: HarmonicLayout, v: Array, dv: float, power: int) -> Array:
    """Return ``<v^power>`` using the Joglekar et al. (2014) convention.

    The moment is normalized by the local electron density, so ``power=0``
    returns one (up to velocity-grid truncation).  Keeping these definitions
    next to the harmonic convention avoids duplicating delicate angular
    normalization factors in diagnostics and Ohm-law closures.
    """

    ne = density(f, layout, v, dv)
    f00 = jnp.real(f[..., layout.index(0, 0), :])
    numerator = 4.0 * jnp.pi * jnp.sum(f00 * v ** (power + 2), axis=-1) * dv
    return numerator / jnp.maximum(ne, jnp.finfo(ne.dtype).tiny)


def vector_velocity_moment(f: Array, layout: HarmonicLayout, v: Array, dv: float, power: int) -> Array:
    """Return ``<v_vec v^power>`` from the packed ``l=1`` harmonics."""

    ne = density(f, layout, v, dv)
    i10 = layout.index(1, 0)
    i11 = layout.index(1, 1)
    if i10 < 0:
        return jnp.zeros((*f.shape[:-2], 3), dtype=jnp.real(f).dtype)
    weight = v ** (power + 3)
    m10 = jnp.sum(jnp.real(f[..., i10, :]) * weight, axis=-1) * dv
    m11 = jnp.sum(f[..., i11, :] * weight, axis=-1) * dv if i11 >= 0 else jnp.zeros_like(m10, dtype=f.dtype)
    numerator = (4.0 * jnp.pi / 3.0) * jnp.stack((m10, 2.0 * jnp.real(m11), -2.0 * jnp.imag(m11)), axis=-1)
    return numerator / jnp.maximum(ne[..., None], jnp.finfo(ne.dtype).tiny)


def cartesian_l2(f: Array, layout: HarmonicLayout) -> Array:
    """Convert packed ``l=2`` coefficients to a symmetric traceless tensor.

    Tzoufras uses the x axis as the polar axis.  With that convention the
    tensor coefficients are ``Fxx=f20``, ``Fxy=3 Re(f21)``,
    ``Fxz=-3 Im(f21)``, ``Fyy=-f20/2+6 Re(f22)``, and
    ``Fyz=-6 Im(f22)``; ``Fzz`` follows from tracelessness.
    """

    shape = (*f.shape[:-2], 3, 3, f.shape[-1])
    result = jnp.zeros(shape, dtype=jnp.real(f).dtype)
    i20, i21, i22 = (layout.index(2, m) for m in range(3))
    if i20 < 0:
        return result
    f20 = jnp.real(f[..., i20, :])
    f21 = f[..., i21, :] if i21 >= 0 else jnp.zeros_like(f20, dtype=f.dtype)
    f22 = f[..., i22, :] if i22 >= 0 else jnp.zeros_like(f20, dtype=f.dtype)
    fxx = f20
    fxy = 3.0 * jnp.real(f21)
    fxz = -3.0 * jnp.imag(f21)
    fyy = -0.5 * f20 + 6.0 * jnp.real(f22)
    fyz = -6.0 * jnp.imag(f22)
    fzz = -0.5 * f20 - 6.0 * jnp.real(f22)
    result = result.at[..., 0, 0, :].set(fxx)
    result = result.at[..., 0, 1, :].set(fxy)
    result = result.at[..., 1, 0, :].set(fxy)
    result = result.at[..., 0, 2, :].set(fxz)
    result = result.at[..., 2, 0, :].set(fxz)
    result = result.at[..., 1, 1, :].set(fyy)
    result = result.at[..., 1, 2, :].set(fyz)
    result = result.at[..., 2, 1, :].set(fyz)
    return result.at[..., 2, 2, :].set(fzz)


def tensor_velocity_moment(f: Array, layout: HarmonicLayout, v: Array, dv: float, power: int) -> Array:
    """Return the traceless ``<vv v^power>`` moment used in kinetic Ohm's law."""

    ne = density(f, layout, v, dv)
    tensor = cartesian_l2(f, layout)
    numerator = (8.0 * jnp.pi / 15.0) * jnp.sum(tensor * v ** (power + 4), axis=-1) * dv
    return numerator / jnp.maximum(ne[..., None, None], jnp.finfo(ne.dtype).tiny)


def nernst_velocity(
    f: Array,
    layout: HarmonicLayout,
    v: Array,
    dv: float,
    plasma_current: Array | None = None,
) -> Array:
    """Return the distribution-function Nernst velocity from PRL Eq. (2)."""

    ne = density(f, layout, v, dv)
    if plasma_current is None:
        plasma_current = current(f, layout, v, dv)
    v3 = scalar_velocity_moment(f, layout, v, dv, power=3)
    vv3 = vector_velocity_moment(f, layout, v, dv, power=3)
    safe_v3 = jnp.maximum(v3, jnp.finfo(v3.dtype).tiny)
    return vv3 / (2.0 * safe_v3[..., None]) + plasma_current / jnp.maximum(ne[..., None], jnp.finfo(ne.dtype).tiny)


def complex_to_real(f: Array) -> Array:
    """Store a complex harmonic array as a final real/imaginary axis."""

    return jnp.stack((jnp.real(f), jnp.imag(f)), axis=-1)


def real_to_complex(f: Array) -> Array:
    """Restore a complex harmonic array from a final real/imaginary axis."""

    if f.shape[-1] != 2:
        raise ValueError("real-embedded harmonic arrays must have a final axis of length 2")
    return f[..., 0] + 1j * f[..., 1]
