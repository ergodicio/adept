"""Hybrid particle evolution (HPE) for the envelope-2d solver.

Test-particle Landau-damping feedback following Follett et al., Phys. Plasmas 24,
102134 (2017): tail electrons are pushed relativistically in the de-enveloped
electrostatic field, a spatially-averaged velocity distribution is accumulated,
and the Landau damping rate applied by ``SpectralEPWSolver`` is recomputed from
that evolving distribution (their Eq. 4). The feedback is Im-only -- the real part
of the EPW dispersion is untouched -- so this captures trapping-induced damping
reduction (kinetic inflation) and hot-electron generation, but not the nonlinear
frequency shift.

**1-D and 2-D.** Follett's Eq. 4 integrates ``d<F>/dv`` over the resonance surface
``omega = k.v``. In 1-D that surface is a point and the rate is one interpolation of
``df/dv`` at ``v_phi = omega/k``. In 2-D it is a *line* in ``(vx, vy)``, and the
integral over it is exactly the 1-D formula applied to the **projection of f onto the
mode's own direction**::

    P_khat(v) = int d^2v' f(v') delta(v'.khat - v),    int P dv = 1
    gamma_L(k) = -(pi/2) (wp0^3 / |k|^2) dP_khat/dv |_{v = omega_res/|k|}

So the whole 2-D generalization is "bin the particles along each of ``n_angles``
directions instead of along x". Because ``P`` for direction ``khat`` and for ``-khat``
are mirror images, only angles in ``[0, pi)`` are accumulated and the opposite
half-plane is reached through the sign ``s`` below -- which is precisely the
``sgn(kx)`` the 1-D implementation already carried. Setting ``ny == 1`` gives
``n_angles == 1``, ``khat = +/- xhat``, and every expression here collapses to the
original quasi-1-D code.

Particle state is ``(Np, ndim)`` with ``ndim = 1`` or ``2``; the push uses ``Ex``
alone in 1-D and ``(Ex, Ey)`` in 2-D.

Departures from the paper, chosen for JAX friendliness (see
docs/dev/lpse2d-hpe-plan.md for the full rationale):

- **Tail-only loading**: particles are drawn from the Maxwellian tail
  ``|v| > v_min * vte``; modes whose phase velocity falls below the cutoff keep the
  analytic damping rate (blended per k-mode). The tail cut is on the *speed*, so in
  2-D the projected distribution equals the true 1-D Maxwellian exactly for
  ``|v_par| > v_min * vte`` -- i.e. everywhere the extraction actually reads it.
- **EMA histogram** instead of interval damping updates: the distribution is a state
  variable updated every field step by an exponential moving average with time
  constant ``tau_damping``.
- **Per-k calibration**: the discrete histogram -> gamma_L operator is normalized per
  k-mode so that the *expected* initial (Maxwellian-tail) histogram reproduces the
  analytic Landau rate exactly. The evolving rate is therefore
  ``gamma_analytic(k) * Op[f](k) / Op[f_Maxwell](k)``, and all binning /
  finite-difference biases cancel.
- ``gamma_HPE`` is clamped >= 0 (a noisy negative rate would pump the field).
"""

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from scipy import special

from adept._lpse2d.core.epw import landau_damping_rate

PARTICLE_KEYS = ("x_e", "u_e", "epw_hist", "gamma_L")


def _hpe_cfg(cfg: dict) -> dict:
    return cfg["terms"]["hpe"]


def _ndim(cfg: dict) -> int:
    return 1 if cfg["grid"]["ny"] == 1 else 2


def tail_fraction(v_min: float, ndim: int) -> float:
    """Fraction of a full Maxwellian carried by the loaded tail.

    1-D cuts on ``|vx| > v_min vte`` (two-sided) -> ``erfc(v_min/sqrt2)``; 2-D cuts on
    the *speed* ``|v| > v_min vte`` -> ``exp(-v_min^2/2)``.
    """
    if ndim == 1:
        return float(special.erfc(v_min / np.sqrt(2.0)))
    return float(np.exp(-0.5 * v_min**2))


def _f0_expected(edges: np.ndarray, vte: float, v_min: float, ndim: int) -> np.ndarray:
    """Expected initial *projected* histogram of the loaded tail, normalized like the
    sampled one (``sum(f) * dv == 1``).

    1-D: exact per-bin integrals of the truncated Maxwellian, so bins straddling the
    cutoff carry their correct partial weight.

    2-D: the projection of the speed-truncated 2-D Maxwellian onto any direction is
    ``f_1D(v) * erfc(sqrt(max(vc^2 - v^2, 0)) / (sqrt2 vte))`` -- unity beyond the
    cutoff, a partial transverse integral inside it. Integrated per bin by
    Gauss-Legendre quadrature; the integrand is smooth, and the only bins that are not
    exactly 1-D Maxwellian sit below the cutoff, where the resonance never reads it.
    """
    vc = v_min * vte
    lo, hi = edges[:-1], edges[1:]
    dv = edges[1] - edges[0]

    if ndim == 1:

        def cdf(v):
            return 0.5 * (1.0 + special.erf(v / (np.sqrt(2.0) * vte)))

        p_bin = np.clip(cdf(hi) - cdf(np.maximum(lo, vc)), 0.0, None) * (hi > vc) + np.clip(
            cdf(np.minimum(hi, -vc)) - cdf(lo), 0.0, None
        ) * (lo < -vc)
    else:
        nq = 16
        xq, wq = np.polynomial.legendre.leggauss(nq)
        vq = 0.5 * (hi[:, None] + lo[:, None]) + 0.5 * dv * xq[None, :]  # (nbins, nq)
        f1 = np.exp(-0.5 * (vq / vte) ** 2) / (np.sqrt(2.0 * np.pi) * vte)
        transverse = special.erfc(np.sqrt(np.clip(vc**2 - vq**2, 0.0, None)) / (np.sqrt(2.0) * vte))
        p_bin = 0.5 * dv * np.sum(wq[None, :] * f1 * transverse, axis=1)

    norm = np.sum(p_bin) * dv
    return p_bin / norm


def resonance_arrays(cfg: dict) -> dict:
    """Static arrays shared by the pusher, the damping extraction, and diagnostics.

    Returns a dict with

    - ``v_centers`` (nv,): histogram bin centers, um/ps
    - ``dv``: bin width
    - ``angles`` (na,): projection directions in ``[0, pi)``
    - ``n_hat`` (na, ndim): unit vectors of those directions
    - ``v_phi`` (nx, ny): *signed* phase velocity of each mode along its projection
      axis (0 where masked)
    - ``mask_res`` (nx, ny): True where the particle-based rate is valid (blend mask)
    - ``gamma_analytic`` (nx, ny): the static analytic Landau rate
    - ``f_tail_frac``: fraction of the full Maxwellian carried by the loaded tail
    - ``f0_expected`` (nv,): expected initial projected histogram
    - ``sign`` (nx, ny): +1 if khat is the projection direction, -1 if its opposite
    - ``a_idx0`` / ``a_w`` (nx, ny): angle-table index and interpolation weight
    - ``v_idx0`` / ``v_w`` (nx, ny): velocity-bin index and weight at ``v_phi``
    """
    hpe = _hpe_cfg(cfg)
    derived = cfg["units"]["derived"]
    wp0, vte, c = derived["wp0"], np.sqrt(derived["vte_sq"]), derived["c"]
    ndim = _ndim(cfg)

    nv = int(hpe["nv"])
    v_max = float(hpe["v_max"]) * c
    edges = np.linspace(-v_max, v_max, nv + 1)
    v_centers = 0.5 * (edges[1:] + edges[:-1])
    dv = edges[1] - edges[0]

    kx = np.array(cfg["grid"]["kx"])
    ky = np.array(cfg["grid"]["ky"])
    KX = kx[:, None] + 0.0 * ky[None, :]
    KY = 0.0 * kx[:, None] + ky[None, :]
    k_sq = KX**2 + KY**2
    k_mag = np.sqrt(k_sq)
    zero_mask = np.where(k_sq > 0, 1.0, 0.0)
    gamma_analytic = np.array(
        landau_damping_rate(jnp.array(k_sq), wp0, derived["vte_sq"], jnp.array(zero_mask)), dtype=np.float64
    )

    n_angles = 1 if ndim == 1 else int(hpe["n_angles"])
    angles = np.arange(n_angles) * (np.pi / n_angles)
    n_hat = np.ones((1, 1)) if ndim == 1 else np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # resonance frequency: Bohm-Gross by default (what the analytic rate the k-blend
    # hands off to is derived with); "wp0" reproduces Follett's bare-carrier choice.
    k_safe = np.where(k_mag > 0, k_mag, 1.0)
    if hpe["omega_res"] == "wp0":
        omega_res = wp0 * np.ones_like(k_sq)
    else:  # bohm_gross
        omega_res = np.sqrt(wp0**2 + 3.0 * k_sq * derived["vte_sq"])

    # each mode's direction, folded into [0, pi): phi is the angle of k, theta the angle
    # of the projection axis, and s = +1 when khat == n_hat(theta), -1 when
    # khat == -n_hat(theta). In 1-D this is exactly sgn(kx).
    phi = np.arctan2(KY, KX)
    theta = np.mod(phi, np.pi)
    sign = np.where(np.mod(phi, 2.0 * np.pi) < np.pi, 1.0, -1.0)
    v_phi = np.where(k_mag > 0, sign * omega_res / k_safe, 0.0)

    v_min = float(hpe["v_min"]) * vte
    buffer = float(hpe["v_blend_buffer"]) * vte
    mask_res = (k_mag > 0) & (np.abs(v_phi) > v_min + buffer) & (np.abs(v_phi) < 0.9 * v_max)

    # static bilinear-gather indices into the (n_angles + 1, nv) df/dv table. The extra
    # angle row is the mirror of row 0 (theta = pi), which closes the wrap. v_phi is
    # static, so the velocity index/weight are static too -- only the table changes.
    a_f = theta / (np.pi / n_angles)
    a_idx0 = np.clip(np.floor(a_f).astype(np.int32), 0, n_angles - 1)
    a_w = np.clip(a_f - a_idx0, 0.0, 1.0)
    v_f = (v_phi - v_centers[0]) / dv
    v_idx0 = np.clip(np.floor(v_f).astype(np.int32), 0, nv - 2)
    v_w = np.clip(v_f - v_idx0, 0.0, 1.0)

    return {
        "v_centers": v_centers,
        "dv": dv,
        "angles": angles,
        "n_hat": n_hat,
        "v_phi": v_phi,
        "mask_res": mask_res,
        "gamma_analytic": gamma_analytic,
        "f_tail_frac": tail_fraction(float(hpe["v_min"]), ndim),
        "f0_expected": _f0_expected(edges, vte, float(hpe["v_min"]), ndim),
        "sign": sign,
        "k_sq": k_sq,
        "a_idx0": a_idx0,
        "a_w": a_w,
        "v_idx0": v_idx0,
        "v_w": v_w,
        "n_angles": n_angles,
        "ndim": ndim,
    }


def _sample_tail_velocities(rng, n, vte, v_min, c, ndim):
    """Draw ``n`` velocity vectors from the Maxwellian tail (1-D: ``|vx| > v_min vte``
    with random sign; 2-D: speed ``> v_min vte``, isotropic direction)."""
    if ndim == 1:
        from scipy import stats

        a, b = v_min, 0.99 * c / vte
        speed = stats.truncnorm.rvs(a, b, loc=0.0, scale=1.0, size=n, random_state=rng) * vte
        return (rng.choice(np.array([-1.0, 1.0]), size=n) * speed)[:, None]

    # 2-D Maxwellian speed beyond the cut: p(s) ~ s exp(-s^2/2vte^2), exact inverse CDF
    uni = rng.uniform(low=1.0e-12, high=1.0, size=n)
    speed = np.minimum(np.sqrt((v_min * vte) ** 2 - 2.0 * vte**2 * np.log(uni)), 0.99 * c)
    alpha = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack([speed * np.cos(alpha), speed * np.sin(alpha)], axis=1)


def load_particles(cfg: dict) -> dict:
    """Sample the initial particle state (numpy, once, at init).

    Positions are drawn proportional to the background density; velocities from the
    Maxwellian tail. Returns the state-dict entries: ``x_e`` and ``u_e``
    (``(Np, ndim)``, ``u = gamma*v``), ``epw_hist`` (``(n_angles, nv)``, the sampled
    projected histograms, which seed the EMA), and ``gamma_L`` (the analytic rate
    array, so the first steps are identical to a fluid run).
    """
    hpe = _hpe_cfg(cfg)
    derived = cfg["units"]["derived"]
    vte, c = np.sqrt(derived["vte_sq"]), derived["c"]
    ndim = _ndim(cfg)
    n_p = int(hpe["n_particles"])
    rng = np.random.default_rng(int(hpe["seed"]))

    v = _sample_tail_velocities(rng, n_p, vte, float(hpe["v_min"]), c, ndim)
    u = v / np.sqrt(1.0 - np.sum(v**2, axis=1, keepdims=True) / c**2)

    # positions: cell weighted by density (x), uniform in y
    x_grid = np.array(cfg["grid"]["x"])
    dx = cfg["grid"]["dx"]
    density = np.array(cfg["grid"]["background_density"])[:, 0]
    p_cell = density / np.sum(density)
    cells = rng.choice(density.size, size=n_p, p=p_cell)
    x_pos = [x_grid[cells] + (rng.uniform(size=n_p) - 0.5) * dx]
    if ndim == 2:
        x_pos.append(rng.uniform(cfg["grid"]["ymin"], cfg["grid"]["ymax"], size=n_p))
    x = np.stack(x_pos, axis=1)

    arrays = resonance_arrays(cfg)
    nv, n_angles = int(hpe["nv"]), arrays["n_angles"]
    v_max = float(hpe["v_max"]) * c
    v_par = v @ arrays["n_hat"].T  # (Np, na)
    counts = np.stack([np.histogram(v_par[:, a], bins=nv, range=(-v_max, v_max))[0] for a in range(n_angles)])
    hist = counts.astype(np.float64) / (n_p * arrays["dv"])

    return {
        "x_e": np.asarray(x, dtype=np.float64),
        "u_e": np.asarray(u, dtype=np.float64),
        "epw_hist": hist,
        "gamma_L": arrays["gamma_analytic"].copy(),
    }


class HybridParticleEvolution:
    """One HPE step: subcycled relativistic push + histogram EMA + damping update.

    Called by ``SplitStep.__call__`` after the EPW update; consumes and returns the
    full state dict (complex-viewed). The damping array it writes into ``y["gamma_L"]``
    is consumed by ``SpectralEPWSolver`` on the *next* field step (one-step lag, far
    tighter than the paper's 100 fs update interval).
    """

    def __init__(self, cfg: dict):
        hpe = _hpe_cfg(cfg)
        derived = cfg["units"]["derived"]
        grid = cfg["grid"]

        self.dt = grid["dt"]
        self.wp0 = derived["wp0"]
        self.c = derived["c"]
        self.vte = float(np.sqrt(derived["vte_sq"]))
        self.q_over_m = derived["e"] / derived["me"]  # electron charge magnitude / mass

        self.nx, self.ny = grid["nx"], grid["ny"]
        self.ndim = _ndim(cfg)
        self.dx, self.dy = grid["dx"], grid["dy"]
        self.xmin, self.xmax = grid["xmin"], grid["xmax"]
        self.ymin, self.ymax = grid["ymin"], grid["ymax"]
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.kx = jnp.array(grid["kx"])
        self.ky = jnp.array(grid["ky"])
        self.periodic = cfg["terms"]["epw"]["boundary"]["x"] == "periodic"

        # gather from a spectrally upsampled field: linear interpolation of a wave
        # sampled at k*dx ~ 1-2 rad/cell attenuates the gathered field by
        # sinc^2(k dx/2) (~15-30%!); zero-padding phi_k by gather_refine before the
        # ifft makes that error ~1% for one cheap (refine*nx, refine*ny) FFT per step
        self.refine = int(hpe["gather_refine"])
        self.refine_y = self.refine if self.ndim == 2 else 1
        self.nx_f = self.refine * self.nx
        self.ny_f = self.refine_y * self.ny
        self.dx_f = self.dx / self.refine
        self.dy_f = self.dy / self.refine_y
        # cell-centered origins, shared by the coarse and refined grids
        self.x0 = float(np.array(grid["x"])[0])
        self.y0 = float(np.array(grid["y"])[0]) if self.ndim == 2 else 0.0
        self.pad_lo = (self.nx_f - self.nx) // 2
        self.pad_hi = self.nx_f - self.nx - self.pad_lo
        self.pad_lo_y = (self.ny_f - self.ny) // 2
        self.pad_hi_y = self.ny_f - self.ny - self.pad_lo_y

        self.n_sub = int(hpe["substeps"])
        self.dtp = self.dt / self.n_sub
        self.alpha = self.dt / hpe["tau_damping_ps"]  # EMA weight
        self.t_start = hpe["t_start_ps"]
        self.feedback = bool(hpe["feedback"])
        self.n_p = int(hpe["n_particles"])
        self.nv = int(hpe["nv"])
        self.v_min = float(hpe["v_min"]) * self.vte
        self.y_thermal_frac = float(hpe.get("y_thermal_frac", 0.0))
        self.hist_smooth = int(hpe.get("hist_smooth", 0))
        # wall re-injection stream: a fold_in stream tag decorrelates it from every
        # other consumer of the HPE seed (the numpy loader) and from the EPW noise
        # stream regardless of how the two seeds relate -- the old additive +7919
        # offset overlapped the EPW stream after 7919 steps
        self.wall_key = jax.random.fold_in(jax.random.PRNGKey(int(hpe["seed"])), 314159)

        arrays = resonance_arrays(cfg)
        self.v_max = float(hpe["v_max"]) * self.c
        self.dv = arrays["dv"]
        self.n_angles = arrays["n_angles"]
        self.v_centers = jnp.array(arrays["v_centers"])
        self.v_edge0 = float(arrays["v_centers"][0] - 0.5 * arrays["dv"])
        self.n_hat = jnp.array(arrays["n_hat"])
        self.v_phi = jnp.array(arrays["v_phi"])
        self.mask_res = jnp.array(arrays["mask_res"])
        self.gamma_analytic = jnp.array(arrays["gamma_analytic"])
        self.f_tail_frac = arrays["f_tail_frac"]
        self.k_sq = jnp.array(np.where(arrays["k_sq"] > 0, arrays["k_sq"], 1.0))
        self.sign = jnp.array(arrays["sign"])
        # flat bilinear-gather indices into the (n_angles + 1, nv) df/dv table
        a0, a1 = arrays["a_idx0"], arrays["a_idx0"] + 1
        i0, i1 = arrays["v_idx0"], arrays["v_idx0"] + 1
        self.g_idx = tuple(jnp.array((a * self.nv + i).ravel()) for a in (a0, a1) for i in (i0, i1))
        self.a_w = jnp.array(arrays["a_w"])
        self.v_w = jnp.array(arrays["v_w"])

        # 2-D thermalizing wall: the flux-weighted emission speed beyond the tail cut
        # goes as s^2 exp(-s^2/2vte^2) (the 1-D law is s exp(...)), which has no
        # closed-form inverse CDF -- tabulate it once instead.
        self.wall_table = jnp.array(self._wall_speed_table())

        # per-k calibration: the discrete operator applied to the expected initial
        # histogram must return the analytic rate exactly (see module docstring)
        f0 = np.broadcast_to(arrays["f0_expected"], (self.n_angles, self.nv))
        gamma_raw0 = np.array(self._gamma_raw(jnp.array(f0)))
        gamma_an = np.array(arrays["gamma_analytic"])
        mask = np.array(arrays["mask_res"])
        with np.errstate(divide="ignore", invalid="ignore"):
            calib = np.where(mask & (gamma_raw0 > 0), gamma_an / np.where(gamma_raw0 > 0, gamma_raw0, 1.0), 1.0)
        self.calibration = jnp.array(calib)
        if np.any(mask):
            c_band = calib[mask & (gamma_an > 1.0e-6 * self.wp0)]
            if c_band.size:
                print(
                    f"HPE: {self.n_p} particles, {self.n_sub} substeps/step, {self.ndim}-D "
                    f"({self.n_angles} projection angle{'s' if self.n_angles > 1 else ''}), "
                    f"calibration C(k) in [{c_band.min():.3f}, {c_band.max():.3f}] over the resonant band"
                )

    def _wall_speed_table(self, n_table: int = 4096) -> np.ndarray:
        """Inverse CDF of the flux-weighted emission speed on ``[v_min, 0.99c]``.

        ``p(s) ~ s^ndim exp(-s^2/2vte^2)``: the extra power of ``s`` in 2-D comes from
        the polar area element. Returned as the speeds at ``n_table`` uniformly spaced
        CDF values, ready for ``jnp.interp(U, linspace(0,1), table)``.
        """
        s = np.linspace(self.v_min, 0.99 * self.c, 8192)
        pdf = s**self.ndim * np.exp(-0.5 * (s / self.vte) ** 2)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(s))])
        cdf /= cdf[-1]
        # strictly increasing for np.interp
        cdf, keep = np.unique(cdf, return_index=True)
        return np.interp(np.linspace(0.0, 1.0, n_table), cdf, s[keep])

    # ------------------------------------------------------------------ push --

    def refine_e(self, phi_k: Array) -> tuple[Array, ...]:
        """Zero-padded ``(nx_f, ny_f)`` complex E-field envelopes gathered from phi_k.

        Returns ``(Ex,)`` in 1-D and ``(Ex, Ey)`` in 2-D. The ``refine * refine_y``
        factor restores the ``1/(nx*ny)`` normalization of the unpadded ``ifft2``.
        """
        e_k = [-1j * self.kx[:, None] * phi_k]
        if self.ndim == 2:
            e_k.append(-1j * self.ky[None, :] * phi_k)
        out = []
        for ek in e_k:
            padded = jnp.pad(jnp.fft.fftshift(ek), ((self.pad_lo, self.pad_hi), (self.pad_lo_y, self.pad_hi_y)))
            out.append(jnp.fft.ifft2(jnp.fft.ifftshift(padded)) * self.refine * self.refine_y)
        return tuple(out)

    def _accel(self, x: Array, e_env: tuple[Array, ...], t: float) -> Array:
        """``-(e/m) Re[E_envelope(x) exp(-i wp0 t)]`` gathered at particle positions
        ((bi)linear interpolation on the upsampled grid, clamped at the walls)."""
        idx = (x[:, 0] - self.x0) / self.dx_f
        i0 = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, self.nx_f - 2)
        wx = jnp.clip(idx - i0, 0.0, 1.0)
        if self.ndim == 1:

            def gather(f):
                return f[i0, 0] * (1.0 - wx) + f[i0 + 1, 0] * wx

        else:
            idy = (x[:, 1] - self.y0) / self.dy_f
            j0 = jnp.clip(jnp.floor(idy).astype(jnp.int32), 0, self.ny_f - 2)
            wy = jnp.clip(idy - j0, 0.0, 1.0)

            def gather(f):
                return (
                    f[i0, j0] * (1.0 - wx) * (1.0 - wy)
                    + f[i0 + 1, j0] * wx * (1.0 - wy)
                    + f[i0, j0 + 1] * (1.0 - wx) * wy
                    + f[i0 + 1, j0 + 1] * wx * wy
                )

        carrier = jnp.exp(-1j * self.wp0 * t)
        return -self.q_over_m * jnp.stack([jnp.real(gather(f) * carrier) for f in e_env], axis=1)

    def _drift_kick(self, i, x, u, a, e_env, t0):
        # the leading kick reuses the acceleration carried from the previous substep's
        # trailing kick (same x, same t -- bitwise identical), so _accel runs once per
        # substep instead of twice; XLA does not CSE across fori_loop iterations, and
        # the gather in _accel dominates the GPU cost of this loop
        t_i = t0 + i * self.dtp
        u_half = u + 0.5 * self.dtp * a
        gamma = jnp.sqrt(1.0 + jnp.sum((u_half / self.c) ** 2, axis=1, keepdims=True))
        x = x + self.dtp * u_half / gamma
        if self.periodic:
            x = x.at[:, 0].set(jnp.mod(x[:, 0] - self.xmin, self.Lx) + self.xmin)
        crossed = None
        if self.ndim == 2:
            # transverse boundary is periodic in the field solver; wrap to match, and
            # record the crossing so _apply_boundaries can thermalize a fraction of them
            y = x[:, 1]
            crossed = (y < self.ymin) | (y > self.ymax)
            x = x.at[:, 1].set(jnp.mod(y - self.ymin, self.Ly) + self.ymin)
        a = self._accel(x, e_env, t_i + self.dtp)
        return x, u_half + 0.5 * self.dtp * a, a, crossed

    def _substep(self, i, carry, e_env, t0):
        x, u, a = carry
        x, u, a, _ = self._drift_kick(i, x, u, a, e_env, t0)
        return x, u, a

    def _substep_counting(self, i, carry, e_env, t0):
        """2-D variant: also accumulates each particle's transverse crossing count."""
        x, u, a, n_cross = carry
        x, u, a, crossed = self._drift_kick(i, x, u, a, e_env, t0)
        return x, u, a, n_cross + crossed.astype(n_cross.dtype)

    def _emission_speed(self, key, n: int) -> Array:
        """Flux-weighted inward emission speed from the thermalizing wall.

        1-D has the exact inverse CDF of ``p(v) ~ v exp(-v^2/2vte^2)``; 2-D picks up an
        extra power of ``s`` from the polar area element and is tabulated instead.
        """
        if self.ndim == 1:
            uni = jax.random.uniform(key, (n,), minval=1.0e-12, maxval=1.0)
            return jnp.sqrt(self.v_min**2 - 2.0 * self.vte**2 * jnp.log(uni))
        uni = jax.random.uniform(key, (n,), minval=0.0, maxval=1.0)
        return jnp.interp(uni, jnp.linspace(0.0, 1.0, self.wall_table.size), self.wall_table)

    def _apply_boundaries(self, x: Array, u: Array, t: float, n_cross: Array | None = None) -> tuple[Array, Array]:
        """Thermalizing walls in x: exiting particles are re-injected at the wall with
        an inward, flux-weighted tail speed, which keeps the loaded tail distribution
        stationary. In 2-D the emission direction is cosine-weighted about the inward
        normal (``p(alpha) ~ cos alpha``, the standard thermal-wall law), and a
        configurable fraction of transverse crossings is re-thermalized isotropically
        to mimic a finite plasma (Follett used 20%)."""
        thermalize_y = self.ndim == 2 and self.y_thermal_frac > 0.0 and n_cross is not None
        if self.periodic and not thermalize_y:
            return x, u

        step_key = jax.random.fold_in(self.wall_key, jnp.asarray(t / self.dt).astype(jnp.int32))
        k_speed, k_ang, k_y, k_ythermal = jax.random.split(step_key, 4)
        speed = jnp.minimum(self._emission_speed(k_speed, self.n_p), 0.99 * self.c)
        gamma = 1.0 / jnp.sqrt(1.0 - (speed / self.c) ** 2)

        # a periodic x box has no walls to re-inject from, but may still thermalize in y
        out_x = jnp.zeros(self.n_p, dtype=bool)
        if not self.periodic:
            out_left = x[:, 0] < self.xmin
            out_right = x[:, 0] > self.xmax
            if self.ndim == 1:
                u_in = (speed * gamma)[:, None]
            else:
                # p(alpha) ~ cos(alpha) on (-pi/2, pi/2) about the inward normal
                uni = jax.random.uniform(k_ang, (self.n_p,), minval=-1.0, maxval=1.0)
                alpha = jnp.arcsin(uni)
                u_in = (speed * gamma)[:, None] * jnp.stack([jnp.cos(alpha), jnp.sin(alpha)], axis=1)

            out_x = out_left | out_right
            # left wall emits +x, right wall emits -x; the tangential component is unbiased
            flip = jnp.where(out_right, -1.0, 1.0)[:, None] * jnp.ones((1, self.ndim))
            u = jnp.where(out_x[:, None], u_in * flip, u)
            x = x.at[:, 0].set(jnp.where(out_left, self.xmin, jnp.where(out_right, self.xmax, x[:, 0])))

        if thermalize_y:
            # Follett thermalizes a fraction of transverse *crossings*, not a fraction
            # of the population: a particle that wrapped n times is thermalized with
            # probability 1 - (1 - frac)^n. Positions already wrapped inside the push.
            p_thermal = 1.0 - (1.0 - self.y_thermal_frac) ** n_cross
            hit = jax.random.uniform(k_ythermal, (self.n_p,)) < p_thermal
            ang = jax.random.uniform(k_y, (self.n_p,), minval=0.0, maxval=2.0 * jnp.pi)
            u_iso = (speed * gamma)[:, None] * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
            u = jnp.where((hit & ~out_x)[:, None], u_iso, u)
        return x, u

    def push(self, x: Array, u: Array, e_env: tuple[Array, ...], t: float) -> tuple[Array, Array]:
        """Subcycled relativistic KDK leapfrog across one field step ``[t, t + dt]``."""
        a0 = self._accel(x, e_env, t)
        if self.ndim == 2 and self.y_thermal_frac > 0.0:
            n0 = jnp.zeros(x.shape[0], dtype=jnp.int32)
            x, u, _, n_cross = jax.lax.fori_loop(
                0, self.n_sub, lambda i, carry: self._substep_counting(i, carry, e_env, t), (x, u, a0, n0)
            )
        else:
            n_cross = None
            x, u, _ = jax.lax.fori_loop(0, self.n_sub, lambda i, carry: self._substep(i, carry, e_env, t), (x, u, a0))
        return self._apply_boundaries(x, u, t, n_cross)

    # ------------------------------------------------- histogram and damping --

    def histogram(self, u: Array) -> Array:
        """``(n_angles, nv)`` projected velocity distributions.

        One fused scatter over ``Np * n_angles`` samples rather than ``n_angles``
        separate histograms -- the projection ``v . n_hat`` is a single matmul and the
        bin index is offset by ``a * nv`` so every angle lands in one flat array.
        """
        v = u / jnp.sqrt(1.0 + jnp.sum((u / self.c) ** 2, axis=1, keepdims=True))
        v_par = v @ self.n_hat.T  # (Np, na)
        ibin = jnp.floor((v_par - self.v_edge0) / self.dv).astype(jnp.int32)
        inside = (ibin >= 0) & (ibin < self.nv)
        offset = jnp.arange(self.n_angles, dtype=jnp.int32) * self.nv
        flat = jnp.where(inside, ibin + offset[None, :], self.n_angles * self.nv)
        counts = jnp.zeros(self.n_angles * self.nv + 1, dtype=v.dtype).at[flat.ravel()].add(1.0)
        return counts[:-1].reshape(self.n_angles, self.nv) / (self.n_p * self.dv)

    def _smooth(self, hist: Array) -> Array:
        """Optional binomial smoothing of the projected histograms in v before the slope
        is taken (``hist_smooth`` passes of a [1, 2, 1]/4 kernel; 0 = off, the default).

        Motivation: the rate is read from ``df/dv`` at one velocity bin per mode, and
        with a finite particle count some modes draw a slope noisy enough that the
        ``gamma >= 0`` clamp sends them to exactly zero. A mode pinned at zero is
        *undamped*, so it then grows relative to the rest of the band -- the error does
        not average out, it selects for itself. That is a mild nuisance in quasi-1D (a
        few hundred band modes) and a real one in 2-D, where the same box carries ~10x
        as many band modes and the chance that at least one is clamped is correspondingly
        higher.

        Smoothing is bias-free here by construction: the per-k calibration ``C(k)`` is
        derived by applying *this same operator* to the expected Maxwellian histogram, so
        any linear filter is divided back out and only the variance reduction survives.
        """
        if self.hist_smooth <= 0:
            return hist
        for _ in range(self.hist_smooth):
            padded = jnp.concatenate([hist[:, :1], hist, hist[:, -1:]], axis=1)
            hist = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
        return hist

    def _gamma_raw(self, hist: Array) -> Array:
        """Uncalibrated Follett Eq. 4 for every mode:

        ``gamma(k) = -(pi/2) wp0^3 / |k|^2 * s * dP_theta/dv |_{v_phi}``

        with ``P_theta`` the projection of f onto the mode's axis and ``s = +/-1`` the
        direction of ``khat`` along it. The sign makes the damping even in ``k`` for a
        symmetric distribution: opposite modes resonate on opposite tails, so each
        propagation direction damps on its own. In 1-D ``s`` is ``sgn(kx)`` and this is
        the original expression.
        """
        # close the angular wrap: theta = pi is the mirror of theta = 0
        hist_ext = jnp.concatenate([hist, hist[0:1, ::-1]], axis=0)
        hist_ext = self._smooth(hist_ext)
        dfdv = jnp.gradient(hist_ext, self.dv, axis=-1).ravel() * self.f_tail_frac
        g00, g01, g10, g11 = (dfdv[i].reshape(self.sign.shape) for i in self.g_idx)
        aw, vw = self.a_w, self.v_w
        dfdv_at = (1.0 - aw) * ((1.0 - vw) * g00 + vw * g01) + aw * ((1.0 - vw) * g10 + vw * g11)
        return -0.5 * np.pi * self.wp0**3 * self.sign / self.k_sq * dfdv_at

    def damping(self, hist: Array) -> Array:
        """Blended ``(nx, ny)`` damping array: calibrated HPE rate on resonant modes
        (clamped >= 0), analytic rate elsewhere."""
        gamma_hpe = jnp.maximum(self.calibration * self._gamma_raw(hist), 0.0)
        return jnp.where(self.mask_res, gamma_hpe, self.gamma_analytic)

    # ---------------------------------------------------------------- driver --

    def __call__(self, t: float, y: dict[str, Array]) -> dict[str, Array]:
        phi_k = y["epw"]

        def active(operand):
            x, u, hist, gamma_L = operand
            e_env = self.refine_e(phi_k)
            x, u = self.push(x, u, e_env, t)
            hist = (1.0 - self.alpha) * hist + self.alpha * self.histogram(u)
            if self.feedback:
                gamma_L = self.damping(hist)
            return x, u, hist, gamma_L

        def inactive(operand):
            return operand

        x, u, hist, gamma_L = jax.lax.cond(
            t >= self.t_start, active, inactive, (y["x_e"], y["u_e"], y["epw_hist"], y["gamma_L"])
        )
        return {**y, "x_e": x, "u_e": u, "epw_hist": hist, "gamma_L": gamma_L}
