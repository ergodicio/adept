"""Resonance absorption for the FD light solver (LPSE ``laser.evolution.resonanceAbsorption``;
``LightSolver::computeResonanceAbsorption``, ``LightSolver.cpp:4614-4810``) -- plan 2 L.3.

LPSE adds two split-step terms to the light field every light sub-step so that p-polarised
light obliquely incident on the critical surface converts into a longitudinal (plasma-wave)
component there and is damped:

1. the warm-plasma term, in x-space,

       dE/dt = C0 [ (div E) grad(n)/n - 3 grad(div E) ],   C0 = v_te^2 / (2 i w),

   the electron-pressure correction that gives the longitudinal part of E the Bohm-Gross
   dispersion, masked off inside the absorbing layers with a quadratic ramp
   (``Beach::halfBeachX``);
2. Landau damping of the longitudinal part, in k-space,

       E_k -= dt min(gamma_L(k), 1/dt) k (k . E_k) / k^2,

   at the EPW Landau rate (the same table as the EPW step; LPSE builds it whenever RA is on,
   whatever ``lw.landauDamping.enable``), masked with an s-curve over the absorbers
   (``Beach::fullBeachX``); then, optionally, a radial low-pass filter on the field
   (``Beach::fullRadialK``: 1 below ``filter_width * K_nyq - W``, an s-curve to 0 at
   ``filter_width * K_nyq``, ``W = K_nyq (1 - 1/sqrt(2))`` in 2-D).

LPSE applies the k-space part every ``LdUpdate`` sub-steps and re-uses the increment in
between; here it is applied every ``landau_update`` sub-steps without the stale increment
(``landau_update: 1`` is exact). LPSE refuses RA with its spectral solver, as does this.
"""

import numpy as np
from jax import Array
from jax import numpy as jnp


def s_curve(x, start, width):
    """LPSE ``Beach::sCurve``: 0 for x <= start, (3 - 2y) y^2 across ``width``, 1 beyond."""
    y = np.clip((np.asarray(x, dtype=np.float64) - start) / width, 0.0, 1.0) if width > 0 else np.ones_like(x)
    return (3.0 - 2.0 * y) * y**2


def quadratic_ramp(x, start, width):
    """LPSE ``Beach::leftRamp``: 0 for x <= start, 1 - (1 - y)^2 across ``width``, 1 beyond."""
    y = np.clip((np.asarray(x, dtype=np.float64) - start) / width, 0.0, 1.0) if width > 0 else np.ones_like(x)
    return 1.0 - (1.0 - y) ** 2


class ResonanceAbsorption:
    def __init__(self, cfg: dict, light, dt_l: float, landau_rate: Array):
        from astropy.units import Quantity as _Q

        ra = cfg["terms"].get("light", {}).get("resonance_absorption", None)
        if ra is True:
            ra = {}
        self.enabled = isinstance(ra, dict)
        if not self.enabled:
            return
        derived = cfg["units"]["derived"]
        grid = cfg["grid"]
        self.light = light
        self.dt_l = float(dt_l)
        self.w = light.w0
        self.c0 = derived["vte_sq"] / (2.0j * self.w)
        self.t_start = _Q(ra.get("t_start", "0ps")).to("ps").value
        self.t_stop = _Q(ra["t_stop"]).to("ps").value if ra.get("t_stop") is not None else np.inf
        self.landau_update = int(ra.get("landau_update", 1))
        if self.landau_update < 0:
            raise ValueError(
                "terms.light.resonance_absorption.landau_update must be >= 0 (0 disables the k-space term)"
            )
        filter_on = bool(ra.get("filter", True))
        filter_width = float(ra.get("filter_width", 1.0))
        if not 0.0 < filter_width <= 1.0:
            raise ValueError("terms.light.resonance_absorption.filter_width must be in (0, 1]")

        x = np.asarray(grid["x"], dtype=np.float64)
        y = np.asarray(grid["y"], dtype=np.float64)
        nx, ny = x.size, y.size
        dx, dy = float(grid["dx"]), float(grid["dy"])
        density = np.asarray(grid["background_density"], dtype=np.float64)
        # grad(n)/n on the light grid (LPSE gradNoOverNo: central differences of the density)
        gx = np.gradient(density, dx, axis=0) if nx > 1 else np.zeros_like(density)
        gy = np.gradient(density, dy, axis=1) if ny > 1 else np.zeros_like(density)
        safe = np.where(density > 0.0, density, 1.0)
        self.grad_n_over_n = (jnp.asarray(gx / safe), jnp.asarray(gy / safe))

        # masks (LPSE beachSource.halfBeachX / beachRA.fullBeachX): the absorbing layers are
        # boundary_width wide from each x wall; the warm term ramps up quadratically over
        # `edge_width` past the layer, the Landau increment follows an s-curve over the layer
        # the pump's layer (laser.evolution.Labc)
        boundary_width = float(grid.get("light_boundary_width_um", _Q(grid["boundary_width"]).to("um").value))
        edge_width = _Q(ra["edge_width"]).to("um").value if ra.get("edge_width") is not None else 2.0 * dx
        edge_width = max(edge_width, dx)
        xmin, xmax = float(grid["xmin"]), float(grid["xmax"])
        warm = quadratic_ramp(x, xmin + boundary_width, edge_width) * quadratic_ramp(
            -x, -(xmax - boundary_width), edge_width
        )
        landau = s_curve(x, xmin, boundary_width) * s_curve(-x, -xmax, boundary_width)
        if ny > 1 and str(cfg["terms"]["epw"]["boundary"].get("y", "periodic")) != "periodic":
            ymin, ymax = float(grid["ymin"]), float(grid["ymax"])
            warm = (
                warm[:, None]
                * (
                    quadratic_ramp(y, ymin + boundary_width, edge_width)
                    * quadratic_ramp(-y, -(ymax - boundary_width), edge_width)
                )[None, :]
            )
            landau = landau[:, None] * (s_curve(y, ymin, boundary_width) * s_curve(-y, -ymax, boundary_width))[None, :]
        else:
            warm = np.repeat(warm[:, None], ny, axis=1)
            landau = np.repeat(landau[:, None], ny, axis=1)
        self.warm_mask = jnp.asarray(warm)[..., None]
        self.landau_mask = jnp.asarray(landau)[..., None]

        # k-space: -dt min(gamma_L, 1/dt) / k^2 on the longitudinal projector k k
        kx = np.asarray(grid["kx"], dtype=np.float64)
        ky = np.asarray(grid["ky"], dtype=np.float64)
        k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
        gamma = np.minimum(np.asarray(landau_rate, dtype=np.float64), 1.0 / self.dt_l)
        self.kx, self.ky = jnp.asarray(kx), jnp.asarray(ky)
        self.damping_over_k_sq = jnp.asarray(
            np.where(k_sq > 0.0, self.dt_l * self.landau_update * gamma / np.where(k_sq > 0.0, k_sq, 1.0), 0.0)
        )
        if filter_on:
            k_nyq = np.sqrt(kx.max() ** 2 + (ky.max() ** 2 if ny > 1 else 0.0))
            width = k_nyq * (1.0 - 1.0 / np.sqrt(2.0 if ny > 1 else 1.0))
            radius = filter_width * k_nyq
            self.k_filter = jnp.asarray(1.0 - s_curve(np.sqrt(k_sq), radius - width, width))[..., None]
        else:
            self.k_filter = None

    def active(self, t):
        return jnp.logical_and(t >= self.t_start, t <= self.t_stop)

    def warm_term(self, E: Array) -> Array:
        """``C0 [(div E) grad n / n - 3 grad(div E)]`` with the light solver's stencils, masked."""
        light = self.light
        ex, ey = E[..., 0], E[..., 1]
        div_e = light._dx(ex) + light._dy(ey)
        grad_div = [light._d2x(ex) + light._dxdy(ey), light._dxdy(ex) + light._d2y(ey)]
        gx, gy = self.grad_n_over_n
        terms = [self.c0 * (div_e * gx - 3.0 * grad_div[0]), self.c0 * (div_e * gy - 3.0 * grad_div[1])]
        if E.shape[-1] == 3:
            terms.append(jnp.zeros_like(E[..., 2]))  # k_z = 0: E_z is transverse, no divergence
        return jnp.stack(terms, axis=-1) * self.warm_mask

    def landau_step(self, E: Array) -> Array:
        """Damp the longitudinal part at the EPW Landau rate (masked), then filter."""
        e_k = jnp.fft.fft2(E, axes=(0, 1))
        k_dot_e = self.kx[:, None] * e_k[..., 0] + self.ky[None, :] * e_k[..., 1]
        delta = -self.damping_over_k_sq * k_dot_e
        delta_x = jnp.fft.ifft2(delta * self.kx[:, None], axes=(0, 1))
        delta_y = jnp.fft.ifft2(delta * self.ky[None, :], axes=(0, 1))
        increment = jnp.stack([delta_x, delta_y] + ([jnp.zeros_like(delta_x)] if E.shape[-1] == 3 else []), axis=-1)
        E = E + self.landau_mask * increment
        if self.k_filter is not None:
            E = jnp.fft.ifft2(jnp.fft.fft2(E, axes=(0, 1)) * self.k_filter, axes=(0, 1))
        return E

    def __call__(self, t, i_sub, E: Array) -> Array:
        """Both split-step terms on the light field over one sub-step (LPSE's order: the warm
        term as an explicit x-space increment, then the k-space damping every
        ``landau_update`` sub-steps)."""
        on = self.active(t)
        E_warm = E + self.dt_l * self.warm_term(E)
        if self.landau_update > 0:
            do_landau = jnp.logical_and(on, (i_sub % self.landau_update) == 0)
            E_new = jnp.where(do_landau, self.landau_step(E_warm), E_warm)
        else:
            E_new = E_warm
        return jnp.where(on, E_new, E)
