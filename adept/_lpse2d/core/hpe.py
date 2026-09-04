"""Hybrid particle evolution (HPE) for the envelope-2d solver.

Tail electrons are pushed relativistically in the real, de-enveloped
electrostatic field. Particle positions are used only to gather that local
field; the distribution that feeds Landau damping is a *single box-averaged
ensemble*, following Follett et al., Phys. Plasmas 24, 102134 (2017).

The 2-D tracker is 2D2V: particles carry ``(x, y, p_x, p_y)`` and gather both
components of the EPW field. Rather than storing particles or histograms at
every mesh point, it accumulates a small bank of global projected distributions
``f(v . k_hat)``. The projection angle and resonant velocity are interpolated
for every ``(kx, ky)`` mode when evaluating the damping. The existing ny == 1
path remains available for inexpensive SRS calculations and for its validation
suite.

The feedback is imaginary-part-only: trapping can reduce Landau damping and
generate hot electrons, but it does not alter the real EPW dispersion.
"""

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from scipy import special

from adept._lpse2d.core.epw import landau_damping_rate

PARTICLE_KEYS = ("x_e", "y_e", "u_e", "epw_hist", "gamma_L")


def _hpe_cfg(cfg: dict) -> dict:
    return cfg["terms"]["hpe"]


def _analytic_rate(cfg: dict) -> np.ndarray:
    derived = cfg["units"]["derived"]
    kx = np.asarray(cfg["grid"]["kx"])
    ky = np.asarray(cfg["grid"]["ky"])
    k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
    zero_mask = np.where(k_sq > 0.0, 1.0, 0.0)
    return np.asarray(
        landau_damping_rate(jnp.asarray(k_sq), derived["wp0"], derived["vte_sq"], jnp.asarray(zero_mask)),
        dtype=np.float64,
    )


def resonance_arrays(cfg: dict) -> dict:
    """Build the velocity grid, resonance map, blend mask, and calibration input.

    For ``ny == 1``, ``v_phi`` and ``mask_res`` retain their historical ``(nx,)``
    shape. In 2-D they have shape ``(nx, ny)`` and ``angles`` contains the
    oriented projection axes spanning ``[0, 2*pi)``. Opposite directions are
    deliberately distinct: an asymmetric distribution can damp ``+k`` and
    ``-k`` independently.
    """
    hpe = _hpe_cfg(cfg)
    derived = cfg["units"]["derived"]
    wp0, vte, c = derived["wp0"], np.sqrt(derived["vte_sq"]), derived["c"]
    nx, ny = int(cfg["grid"]["nx"]), int(cfg["grid"]["ny"])
    is_2d = ny > 1

    nv = int(hpe["nv"])
    v_max = float(hpe["v_max"]) * c
    edges = np.linspace(-v_max, v_max, nv + 1)
    v_centers = 0.5 * (edges[1:] + edges[:-1])
    dv = float(edges[1] - edges[0])

    kx = np.asarray(cfg["grid"]["kx"])
    ky = np.asarray(cfg["grid"]["ky"])
    k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
    k_mag = np.sqrt(k_sq)
    gamma_analytic = _analytic_rate(cfg)

    if is_2d:
        theta_k = np.mod(np.arctan2(ky[None, :], kx[:, None]), 2.0 * np.pi)
        n_angles = int(hpe["n_angles"])
        angles = 2.0 * np.pi * np.arange(n_angles) / n_angles
        k_safe = np.where(k_mag > 0.0, k_mag, 1.0)
        if hpe["omega_res"] == "wp0":
            omega_res = wp0 * np.ones_like(k_mag)
        else:
            omega_res = np.sqrt(wp0**2 + 3.0 * k_sq * derived["vte_sq"])
        v_phi = np.where(k_mag > 0.0, omega_res / k_safe, 0.0)
    else:
        angles = np.asarray([0.0])
        theta_k = np.zeros((nx, 1))
        kx_safe = np.where(np.abs(kx) > 0.0, kx, 1.0)
        if hpe["omega_res"] == "wp0":
            omega_res = wp0 * np.ones_like(kx)
        else:
            omega_res = np.sqrt(wp0**2 + 3.0 * kx**2 * derived["vte_sq"])
        v_phi = np.where(np.abs(kx) > 0.0, omega_res / kx_safe, 0.0)

    v_min = float(hpe["v_min"]) * vte
    buffer = float(hpe["v_blend_buffer"]) * vte
    if is_2d:
        mask_res = (k_mag > 0.0) & (v_phi > v_min + buffer) & (v_phi < 0.9 * v_max)
        # Fraction of an isotropic 2-D Maxwellian outside the loaded speed circle.
        f_tail_frac = float(np.exp(-0.5 * (v_min / vte) ** 2))

        # Conditional marginal of a radially truncated 2-D Maxwellian. Above the
        # cutoff (the only region used by HPE damping), this is the full Gaussian
        # divided by the radial-tail probability. The inner expression keeps the
        # complete expected histogram normalized for diagnostics and calibration.
        abs_v = np.abs(v_centers)
        perpendicular_cut = np.sqrt(np.maximum(v_min**2 - abs_v**2, 0.0))
        perpendicular_tail = np.where(
            abs_v >= v_min,
            1.0,
            special.erfc(perpendicular_cut / (np.sqrt(2.0) * vte)),
        )
        f0_expected = (
            np.exp(-0.5 * (v_centers / vte) ** 2) / (np.sqrt(2.0 * np.pi) * vte) * perpendicular_tail / f_tail_frac
        )
        f0_expected /= np.sum(f0_expected) * dv
    else:
        mask_res = (np.abs(kx) > 0.0) & (np.abs(v_phi) > v_min + buffer) & (np.abs(v_phi) < 0.9 * v_max)
        f_tail_frac = float(special.erfc(v_min / (np.sqrt(2.0) * vte)))

        def cdf(v):
            return 0.5 * (1.0 + special.erf(v / (np.sqrt(2.0) * vte)))

        lo, hi = edges[:-1], edges[1:]
        p_bin = np.clip(cdf(hi) - cdf(np.maximum(lo, v_min)), 0.0, None) * (hi > v_min) + np.clip(
            cdf(np.minimum(hi, -v_min)) - cdf(lo), 0.0, None
        ) * (lo < -v_min)
        f0_expected = p_bin / (np.sum(p_bin) * dv)

    return {
        "v_centers": v_centers,
        "dv": dv,
        "v_phi": v_phi,
        "mask_res": mask_res,
        "gamma_analytic": gamma_analytic,
        "f_tail_frac": f_tail_frac,
        "f0_expected": f0_expected,
        "angles": angles,
        "theta_k": theta_k,
    }


def load_particles(cfg: dict) -> dict:
    """Sample one box-wide tail ensemble and return its initial solver state."""
    from scipy import stats

    hpe = _hpe_cfg(cfg)
    derived = cfg["units"]["derived"]
    vte, c = np.sqrt(derived["vte_sq"]), derived["c"]
    n_p = int(hpe["n_particles"])
    is_2d = int(cfg["grid"]["ny"]) > 1
    rng = np.random.default_rng(int(hpe["seed"]))

    if is_2d:
        # Conditional Rayleigh draw for an isotropic 2-D Maxwellian with
        # |v| > v_min*vte. The relativistic cap only removes a negligible tail.
        uni = rng.uniform(np.finfo(float).tiny, 1.0, size=n_p)
        speed = np.sqrt((float(hpe["v_min"]) * vte) ** 2 - 2.0 * vte**2 * np.log(uni))
        speed = np.minimum(speed, 0.99 * c)
        velocity_angle = rng.uniform(0.0, 2.0 * np.pi, size=n_p)
        velocity = speed[:, None] * np.stack((np.cos(velocity_angle), np.sin(velocity_angle)), axis=-1)
        gamma_rel = 1.0 / np.sqrt(1.0 - (speed / c) ** 2)
        u = gamma_rel[:, None] * velocity
    else:
        a, b = float(hpe["v_min"]), 0.99 * c / vte
        speed = stats.truncnorm.rvs(a, b, loc=0.0, scale=1.0, size=n_p, random_state=rng) * vte
        velocity = rng.choice(np.asarray([-1.0, 1.0]), size=n_p) * speed
        u = velocity / np.sqrt(1.0 - (velocity / c) ** 2)

    # Draw cells from the complete 2-D density map. This remains one ensemble:
    # no particle list or velocity distribution is attached to an individual cell.
    x_grid = np.asarray(cfg["grid"]["x"])
    y_grid = np.asarray(cfg["grid"]["y"])
    density = np.asarray(cfg["grid"]["background_density"])
    p_cell = density.reshape(-1) / np.sum(density)
    cells = rng.choice(density.size, size=n_p, p=p_cell)
    ix, iy = np.unravel_index(cells, density.shape)
    x = x_grid[ix] + (rng.uniform(size=n_p) - 0.5) * cfg["grid"]["dx"]
    y = y_grid[iy] + (rng.uniform(size=n_p) - 0.5) * cfg["grid"]["dy"]

    arrays = resonance_arrays(cfg)
    nv = int(hpe["nv"])
    v_max = float(hpe["v_max"]) * c
    if is_2d:
        directions = np.stack((np.cos(arrays["angles"]), np.sin(arrays["angles"])), axis=-1)
        projected = velocity @ directions.T
        hist = np.stack(
            [np.histogram(projected[:, i], bins=nv, range=(-v_max, v_max))[0] for i in range(directions.shape[0])]
        ).astype(np.float64)
        hist /= n_p * arrays["dv"]
    else:
        counts, _ = np.histogram(velocity, bins=nv, range=(-v_max, v_max))
        hist = counts.astype(np.float64) / (n_p * arrays["dv"])

    state = {
        "x_e": np.asarray(x, dtype=np.float64),
        "u_e": np.asarray(u, dtype=np.float64),
        "epw_hist": hist,
        "gamma_L": arrays["gamma_analytic"].copy(),
    }
    if is_2d:
        state["y_e"] = np.asarray(y, dtype=np.float64)
    return state


class HybridParticleEvolution:
    """Subcycled 1-D or 2-D particle push plus global damping feedback."""

    def __init__(self, cfg: dict):
        hpe = _hpe_cfg(cfg)
        derived = cfg["units"]["derived"]
        grid = cfg["grid"]

        self.dt = grid["dt"]
        self.wp0 = derived["wp0"]
        self.c = derived["c"]
        self.vte = float(np.sqrt(derived["vte_sq"]))
        self.q_over_m = derived["e"] / derived["me"]
        self.nx, self.ny = int(grid["nx"]), int(grid["ny"])
        self.is_2d = self.ny > 1
        self.dx, self.dy = grid["dx"], grid["dy"]
        self.xmin, self.xmax = grid["xmin"], grid["xmax"]
        self.ymin, self.ymax = grid["ymin"], grid["ymax"]
        self.Lx, self.Ly = self.xmax - self.xmin, self.ymax - self.ymin
        self.kx, self.ky = jnp.asarray(grid["kx"]), jnp.asarray(grid["ky"])
        self.periodic_x = cfg["terms"]["epw"]["boundary"]["x"] == "periodic"
        self.periodic_y = cfg["terms"]["epw"]["boundary"]["y"] == "periodic"
        # Historical public attribute retained for quasi-1D validation tests.
        self.periodic = self.periodic_x

        self.refine = int(hpe["gather_refine"])
        self.nx_f, self.ny_f = self.refine * self.nx, self.refine * self.ny
        self.dx_f, self.dy_f = self.dx / self.refine, self.dy / self.refine
        self.x0, self.y0 = float(np.asarray(grid["x"])[0]), float(np.asarray(grid["y"])[0])
        self.pad_x_lo = (self.nx_f - self.nx) // 2
        self.pad_x_hi = self.nx_f - self.nx - self.pad_x_lo
        self.pad_y_lo = (self.ny_f - self.ny) // 2
        self.pad_y_hi = self.ny_f - self.ny - self.pad_y_lo
        self.pad_lo, self.pad_hi = self.pad_x_lo, self.pad_x_hi

        self.n_sub = int(hpe["substeps"])
        self.dtp = self.dt / self.n_sub
        self.alpha = self.dt / hpe["tau_damping_ps"]
        self.t_start = hpe["t_start_ps"]
        self.feedback = bool(hpe["feedback"])
        self.n_p = int(hpe["n_particles"])
        self.nv = int(hpe["nv"])
        self.v_min = float(hpe["v_min"]) * self.vte
        self.wall_key = jax.random.fold_in(jax.random.PRNGKey(int(hpe["seed"])), 314159)

        arrays = resonance_arrays(cfg)
        self.v_max = float(hpe["v_max"]) * self.c
        self.dv = arrays["dv"]
        self.v_centers = jnp.asarray(arrays["v_centers"])
        self.v_edges = jnp.linspace(-self.v_max, self.v_max, self.nv + 1)
        self.v_phi = jnp.asarray(arrays["v_phi"])
        self.mask_res = jnp.asarray(arrays["mask_res"])
        self.gamma_analytic = jnp.asarray(arrays["gamma_analytic"])
        self.f_tail_frac = arrays["f_tail_frac"]
        self.angles = jnp.asarray(arrays["angles"])
        self.directions = jnp.stack((jnp.cos(self.angles), jnp.sin(self.angles)), axis=-1)
        self.theta_k = jnp.asarray(arrays["theta_k"])
        self.n_angles = int(self.angles.size)
        self.dtheta = 2.0 * np.pi / self.n_angles

        if self.is_2d:
            expected = jnp.broadcast_to(jnp.asarray(arrays["f0_expected"]), (self.n_angles, self.nv))
            gamma_raw0 = np.asarray(self._gamma_raw_2d(expected))
            gamma_an = np.asarray(arrays["gamma_analytic"])
            mask = np.asarray(arrays["mask_res"])
        else:
            gamma_raw0 = np.asarray(self._gamma_raw_1d(jnp.asarray(arrays["f0_expected"])))
            gamma_an = np.asarray(arrays["gamma_analytic"][:, 0])
            mask = np.asarray(arrays["mask_res"])
        with np.errstate(divide="ignore", invalid="ignore"):
            calib = np.where(mask & (gamma_raw0 > 0.0), gamma_an / np.where(gamma_raw0 > 0.0, gamma_raw0, 1.0), 1.0)
        self.calibration = jnp.asarray(calib)

        c_band = calib[mask & (gamma_an > 1.0e-6 * self.wp0)]
        if c_band.size:
            dimensionality = f"2D2V/{self.n_angles} angles" if self.is_2d else "1D1V"
            print(
                f"HPE: {self.n_p} particles ({dimensionality}), {self.n_sub} substeps/step, "
                f"calibration C(k) in [{c_band.min():.3f}, {c_band.max():.3f}] over the resonant band"
            )

    # ---------------------------------------------------------------- fields --

    def refine_ex(self, phi_k: Array) -> Array:
        """Quasi-1D compatibility helper returning the refined Ex envelope."""
        ex_k = -1j * self.kx * phi_k[:, 0]
        ex_k_fine = jnp.fft.ifftshift(jnp.pad(jnp.fft.fftshift(ex_k), (self.pad_x_lo, self.pad_x_hi)))
        return jnp.fft.ifft(ex_k_fine) * self.refine / self.ny

    def _pad_field_2d(self, field_k: Array) -> Array:
        shifted = jnp.fft.fftshift(field_k)
        padded = jnp.pad(shifted, ((self.pad_x_lo, self.pad_x_hi), (self.pad_y_lo, self.pad_y_hi)))
        return jnp.fft.ifft2(jnp.fft.ifftshift(padded)) * self.refine**2

    def refine_e(self, phi_k: Array) -> Array:
        """Return refined ``(nx_f, ny_f, 2)`` Ex/Ey envelopes."""
        ex = self._pad_field_2d(-1j * self.kx[:, None] * phi_k)
        ey = self._pad_field_2d(-1j * self.ky[None, :] * phi_k)
        return jnp.stack((ex, ey), axis=-1)

    def _linear_indices(self, position, origin, spacing, count, periodic):
        index = (position - origin) / spacing
        raw = jnp.floor(index).astype(jnp.int32)
        weight = index - raw
        if periodic:
            i0 = jnp.mod(raw, count)
            i1 = jnp.mod(raw + 1, count)
        else:
            i0 = jnp.clip(raw, 0, count - 2)
            i1 = i0 + 1
            weight = jnp.clip(index - i0, 0.0, 1.0)
        return i0, i1, weight

    def _accel(self, x: Array, ex_env: Array, t: float) -> Array:
        """Quasi-1D field gather retained in its historical numerical convention."""
        i0, i1, weight = self._linear_indices(x, self.x0, self.dx_f, self.nx_f, False)
        ex_p = ex_env[i0] * (1.0 - weight) + ex_env[i1] * weight
        return -self.q_over_m * jnp.real(ex_p * jnp.exp(-1j * self.wp0 * t))

    def _accel_2d(self, x: Array, y: Array, e_env: Array, t: float) -> Array:
        ix0, ix1, wx = self._linear_indices(x, self.x0, self.dx_f, self.nx_f, self.periodic_x)
        iy0, iy1, wy = self._linear_indices(y, self.y0, self.dy_f, self.ny_f, self.periodic_y)
        e00, e10 = e_env[ix0, iy0], e_env[ix1, iy0]
        e01, e11 = e_env[ix0, iy1], e_env[ix1, iy1]
        e_particle = (
            (1.0 - wx)[:, None] * (1.0 - wy)[:, None] * e00
            + wx[:, None] * (1.0 - wy)[:, None] * e10
            + (1.0 - wx)[:, None] * wy[:, None] * e01
            + wx[:, None] * wy[:, None] * e11
        )
        return -self.q_over_m * jnp.real(e_particle * jnp.exp(-1j * self.wp0 * t))

    # ------------------------------------------------------------------ push --

    def _substep(self, i, carry, ex_env, t0):
        x, u, acceleration = carry
        t_i = t0 + i * self.dtp
        u_half = u + 0.5 * self.dtp * acceleration
        gamma_rel = jnp.sqrt(1.0 + (u_half / self.c) ** 2)
        x = x + self.dtp * u_half / gamma_rel
        if self.periodic_x:
            x = jnp.mod(x - self.xmin, self.Lx) + self.xmin
        acceleration = self._accel(x, ex_env, t_i + self.dtp)
        u = u_half + 0.5 * self.dtp * acceleration
        return x, u, acceleration

    def _substep_2d(self, i, carry, e_env, t0):
        x, y, u, acceleration = carry
        t_i = t0 + i * self.dtp
        u_half = u + 0.5 * self.dtp * acceleration
        gamma_rel = jnp.sqrt(1.0 + jnp.sum((u_half / self.c) ** 2, axis=-1))
        x = x + self.dtp * u_half[:, 0] / gamma_rel
        y = y + self.dtp * u_half[:, 1] / gamma_rel
        if self.periodic_x:
            x = jnp.mod(x - self.xmin, self.Lx) + self.xmin
        if self.periodic_y:
            y = jnp.mod(y - self.ymin, self.Ly) + self.ymin
        acceleration = self._accel_2d(x, y, e_env, t_i + self.dtp)
        u = u_half + 0.5 * self.dtp * acceleration
        return x, y, u, acceleration

    def _apply_boundaries(self, x: Array, u: Array, t: float) -> tuple[Array, Array]:
        if self.periodic_x:
            return x, u
        out_left, out_right = x < self.xmin, x > self.xmax
        key = jax.random.fold_in(self.wall_key, jnp.asarray(t / self.dt).astype(jnp.int32))
        uni = jax.random.uniform(key, (self.n_p,), minval=1.0e-12, maxval=1.0)
        speed = jnp.minimum(jnp.sqrt(self.v_min**2 - 2.0 * self.vte**2 * jnp.log(uni)), 0.99 * self.c)
        u_new = speed / jnp.sqrt(1.0 - (speed / self.c) ** 2)
        x = jnp.where(out_left, self.xmin, jnp.where(out_right, self.xmax, x))
        u = jnp.where(out_left, u_new, jnp.where(out_right, -u_new, u))
        return x, u

    def _apply_boundaries_2d(self, x, y, u, t):
        out_left, out_right = x < self.xmin, x > self.xmax
        out_bottom, out_top = y < self.ymin, y > self.ymax
        any_out = (out_left | out_right) if self.periodic_y else (out_left | out_right | out_bottom | out_top)
        if self.periodic_x:
            any_out = (out_bottom | out_top) if not self.periodic_y else jnp.zeros_like(out_left)

        step_key = jax.random.fold_in(self.wall_key, jnp.asarray(t / self.dt).astype(jnp.int32))
        radial_key, angle_key = jax.random.split(step_key)
        uni = jax.random.uniform(radial_key, (self.n_p,), minval=1.0e-12, maxval=1.0)
        speed = jnp.minimum(jnp.sqrt(self.v_min**2 - 2.0 * self.vte**2 * jnp.log(uni)), 0.99 * self.c)
        angle = jax.random.uniform(angle_key, (self.n_p,), minval=0.0, maxval=2.0 * np.pi)
        velocity = speed[:, None] * jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
        velocity = velocity.at[:, 0].set(jnp.where(out_left, jnp.abs(velocity[:, 0]), velocity[:, 0]))
        velocity = velocity.at[:, 0].set(jnp.where(out_right, -jnp.abs(velocity[:, 0]), velocity[:, 0]))
        velocity = velocity.at[:, 1].set(jnp.where(out_bottom, jnp.abs(velocity[:, 1]), velocity[:, 1]))
        velocity = velocity.at[:, 1].set(jnp.where(out_top, -jnp.abs(velocity[:, 1]), velocity[:, 1]))
        gamma_rel = 1.0 / jnp.sqrt(1.0 - (speed / self.c) ** 2)
        u_new = gamma_rel[:, None] * velocity

        if not self.periodic_x:
            x = jnp.where(out_left, self.xmin, jnp.where(out_right, self.xmax, x))
        if not self.periodic_y:
            y = jnp.where(out_bottom, self.ymin, jnp.where(out_top, self.ymax, y))
        u = jnp.where(any_out[:, None], u_new, u)
        return x, y, u

    def push(self, x: Array, u: Array, ex_env: Array, t: float) -> tuple[Array, Array]:
        """Historical quasi-1D KDK entry point."""
        acceleration = self._accel(x, ex_env, t)
        x, u, _ = jax.lax.fori_loop(
            0, self.n_sub, lambda i, carry: self._substep(i, carry, ex_env, t), (x, u, acceleration)
        )
        return self._apply_boundaries(x, u, t)

    def push_2d(self, x: Array, y: Array, u: Array, e_env: Array, t: float):
        """Relativistic KDK push for one ``(x,y,p_x,p_y)`` ensemble."""
        acceleration = self._accel_2d(x, y, e_env, t)
        x, y, u, _ = jax.lax.fori_loop(
            0,
            self.n_sub,
            lambda i, carry: self._substep_2d(i, carry, e_env, t),
            (x, y, u, acceleration),
        )
        return self._apply_boundaries_2d(x, y, u, t)

    # ------------------------------------------------- histogram and damping --

    def histogram(self, u: Array) -> Array:
        if not self.is_2d:
            velocity = u / jnp.sqrt(1.0 + (u / self.c) ** 2)
            counts, _ = jnp.histogram(velocity, bins=self.v_edges)
            return counts / (self.n_p * self.dv)

        gamma_rel = jnp.sqrt(1.0 + jnp.sum((u / self.c) ** 2, axis=-1))
        velocity = u / gamma_rel[:, None]

        def projected_histogram(direction):
            counts, _ = jnp.histogram(velocity @ direction, bins=self.v_edges)
            return counts / (self.n_p * self.dv)

        return jax.lax.map(projected_histogram, self.directions)

    def _gamma_raw_1d(self, hist: Array) -> Array:
        dfdv = jnp.gradient(hist, self.dv) * self.f_tail_frac
        dfdv_at_vphi = jnp.interp(self.v_phi, self.v_centers, dfdv)
        kx_safe = jnp.where(jnp.abs(self.kx) > 0.0, self.kx, 1.0)
        return jnp.where(
            jnp.abs(self.kx) > 0.0,
            -0.5 * np.pi * self.wp0**3 * jnp.sign(kx_safe) / kx_safe**2 * dfdv_at_vphi,
            0.0,
        )

    def _gamma_raw_2d(self, hist: Array) -> Array:
        """Evaluate the projected Follett operator at every 2-D spectral mode."""
        dfdv = jnp.gradient(hist, self.dv, axis=-1) * self.f_tail_frac

        angle_coordinate = self.theta_k / self.dtheta
        ia0_raw = jnp.floor(angle_coordinate).astype(jnp.int32)
        angle_weight = angle_coordinate - ia0_raw
        ia0 = jnp.mod(ia0_raw, self.n_angles)
        ia1 = jnp.mod(ia0_raw + 1, self.n_angles)

        velocity_coordinate = (self.v_phi - self.v_centers[0]) / self.dv
        iv0 = jnp.clip(jnp.floor(velocity_coordinate).astype(jnp.int32), 0, self.nv - 2)
        velocity_weight = jnp.clip(velocity_coordinate - iv0, 0.0, 1.0)

        def gather_slope(angle_index):
            lower = dfdv[angle_index, iv0]
            upper = dfdv[angle_index, iv0 + 1]
            return (1.0 - velocity_weight) * lower + velocity_weight * upper

        slope = (1.0 - angle_weight) * gather_slope(ia0) + angle_weight * gather_slope(ia1)
        k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2
        k_sq_safe = jnp.where(k_sq > 0.0, k_sq, 1.0)
        return jnp.where(k_sq > 0.0, -0.5 * np.pi * self.wp0**3 / k_sq_safe * slope, 0.0)

    def _gamma_raw(self, hist: Array) -> Array:
        return self._gamma_raw_2d(hist) if self.is_2d else self._gamma_raw_1d(hist)

    def damping(self, hist: Array) -> Array:
        gamma_hpe = jnp.maximum(self.calibration * self._gamma_raw(hist), 0.0)
        if self.is_2d:
            return jnp.where(self.mask_res, gamma_hpe, self.gamma_analytic)
        gamma_1d = jnp.where(self.mask_res, gamma_hpe, self.gamma_analytic[:, 0])
        return jnp.broadcast_to(gamma_1d[:, None], (self.nx, self.ny))

    # ---------------------------------------------------------------- driver --

    def __call__(self, t: float, y: dict[str, Array]) -> dict[str, Array]:
        if self.is_2d:

            def active(operand):
                x, y_position, u, hist, gamma_l = operand
                x, y_position, u = self.push_2d(x, y_position, u, self.refine_e(y["epw"]), t)
                hist = (1.0 - self.alpha) * hist + self.alpha * self.histogram(u)
                if self.feedback:
                    gamma_l = self.damping(hist)
                return x, y_position, u, hist, gamma_l

            operand = (y["x_e"], y["y_e"], y["u_e"], y["epw_hist"], y["gamma_L"])
            x, y_position, u, hist, gamma_l = jax.lax.cond(t >= self.t_start, active, lambda value: value, operand)
            return {**y, "x_e": x, "y_e": y_position, "u_e": u, "epw_hist": hist, "gamma_L": gamma_l}

        def active_1d(operand):
            x, u, hist, gamma_l = operand
            x, u = self.push(x, u, self.refine_ex(y["epw"]), t)
            hist = (1.0 - self.alpha) * hist + self.alpha * self.histogram(u)
            if self.feedback:
                gamma_l = self.damping(hist)
            return x, u, hist, gamma_l

        operand = (y["x_e"], y["u_e"], y["epw_hist"], y["gamma_L"])
        x, u, hist, gamma_l = jax.lax.cond(t >= self.t_start, active_1d, lambda value: value, operand)
        return {**y, "x_e": x, "u_e": u, "epw_hist": hist, "gamma_L": gamma_l}
