"""Hybrid particle evolution (HPE) for the envelope-2d solver.

Test-particle Landau-damping feedback following Follett et al., Phys. Plasmas 24,
102134 (2017): tail electrons are pushed relativistically in the de-enveloped
electrostatic field, a spatially-averaged velocity distribution is accumulated,
and the Landau damping rate applied by ``SpectralEPWSolver`` is recomputed from
that evolving distribution (their Eq. 4). The feedback is Im-only -- the real part
of the EPW dispersion is untouched -- so this captures trapping-induced damping
reduction (kinetic inflation) and hot-electron generation, but not the nonlinear
frequency shift.

Quasi-1D only for now (``ny == 1``, enforced at config time): the push uses only
``Ex`` and particles carry a single momentum component.

Departures from the paper, chosen for JAX friendliness (see
docs/dev/lpse2d-hpe-plan.md for the full rationale):

- **Tail-only loading**: particles are drawn from the Maxwellian tail
  ``|v| > v_min * vte``; modes whose phase velocity falls below the cutoff keep
  the analytic damping rate (blended per k-mode).
- **EMA histogram** instead of interval damping updates: the distribution is a
  state variable updated every field step by an exponential moving average with
  time constant ``tau_damping``.
- **Per-k calibration**: the discrete histogram -> gamma_L operator is normalized
  per k-mode so that the *expected* initial (Maxwellian-tail) histogram
  reproduces the analytic Landau rate exactly. The evolving rate is therefore
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


def resonance_arrays(cfg: dict) -> dict:
    """Static arrays shared by the pusher, the damping extraction, and diagnostics.

    Returns a dict with

    - ``v_centers`` (nv,): histogram bin centers, um/ps
    - ``dv``: bin width
    - ``v_phi`` (nx,): signed phase velocity of each kx mode (0 where masked)
    - ``mask_res`` (nx,): True where the particle-based rate is valid (blend mask)
    - ``gamma_analytic`` (nx, ny): the static analytic Landau rate
    - ``f_tail_frac``: fraction of the full Maxwellian carried by the loaded tail
    - ``f0_expected`` (nv,): expected initial histogram (exact per-bin integrals of
      the truncated Maxwellian, normalized like the sampled histogram)
    """
    hpe = _hpe_cfg(cfg)
    derived = cfg["units"]["derived"]
    wp0, vte, c = derived["wp0"], np.sqrt(derived["vte_sq"]), derived["c"]

    nv = int(hpe["nv"])
    v_max = float(hpe["v_max"]) * c
    edges = np.linspace(-v_max, v_max, nv + 1)
    v_centers = 0.5 * (edges[1:] + edges[:-1])
    dv = edges[1] - edges[0]

    kx = np.array(cfg["grid"]["kx"])
    ky = np.array(cfg["grid"]["ky"])
    k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
    zero_mask = np.where(k_sq > 0, 1.0, 0.0)
    gamma_analytic = np.array(
        landau_damping_rate(jnp.array(k_sq), wp0, derived["vte_sq"], jnp.array(zero_mask)), dtype=np.float64
    )

    # signed phase velocity of each kx mode. The resonance frequency is the
    # Bohm-Gross frequency by default (this is what the analytic rate the k-blend
    # hands off to is derived with); "wp0" reproduces Follett's bare-carrier choice.
    kx_safe = np.where(np.abs(kx) > 0, kx, 1.0)
    if hpe["omega_res"] == "wp0":
        omega_res = wp0 * np.ones_like(kx)
    else:  # bohm_gross
        omega_res = np.sqrt(wp0**2 + 3.0 * kx**2 * derived["vte_sq"])
    v_phi = np.where(np.abs(kx) > 0, omega_res / kx_safe, 0.0)

    v_min = float(hpe["v_min"]) * vte
    buffer = float(hpe["v_blend_buffer"]) * vte
    mask_res = (np.abs(kx) > 0) & (np.abs(v_phi) > v_min + buffer) & (np.abs(v_phi) < 0.9 * v_max)

    # two-sided Maxwellian tail fraction beyond v_min
    f_tail_frac = float(special.erfc(v_min / (np.sqrt(2.0) * vte)))

    # expected histogram of the truncated Maxwellian: exact per-bin integrals so
    # bins straddling the cutoff carry their correct partial weight
    def cdf(v):
        return 0.5 * (1.0 + special.erf(v / (np.sqrt(2.0) * vte)))

    lo, hi = edges[:-1], edges[1:]
    p_bin = np.clip(cdf(hi) - cdf(np.maximum(lo, v_min)), 0.0, None) * (hi > v_min) + np.clip(
        cdf(np.minimum(hi, -v_min)) - cdf(lo), 0.0, None
    ) * (lo < -v_min)
    norm = np.sum(p_bin) * dv
    f0_expected = p_bin / norm

    return {
        "v_centers": v_centers,
        "dv": dv,
        "v_phi": v_phi,
        "mask_res": mask_res,
        "gamma_analytic": gamma_analytic,
        "f_tail_frac": f_tail_frac,
        "f0_expected": f0_expected,
    }


def load_particles(cfg: dict) -> dict:
    """Sample the initial particle state (numpy, once, at init).

    Positions are drawn proportional to the background density; velocities from
    the two-sided truncated Maxwellian tail ``|v| > v_min * vte``. Returns the
    state-dict entries: ``x_e``, ``u_e`` (u = gamma*v), ``epw_hist`` (the sampled
    histogram, which seeds the EMA), and ``gamma_L`` (the analytic rate array, so
    the first steps are identical to a fluid run).
    """
    from scipy import stats

    hpe = _hpe_cfg(cfg)
    derived = cfg["units"]["derived"]
    vte, c = np.sqrt(derived["vte_sq"]), derived["c"]
    n_p = int(hpe["n_particles"])
    rng = np.random.default_rng(int(hpe["seed"]))

    # velocities: |v|/vte ~ truncnorm on [v_min, 0.99c/vte], random sign
    a, b = float(hpe["v_min"]), 0.99 * c / vte
    speed = stats.truncnorm.rvs(a, b, loc=0.0, scale=1.0, size=n_p, random_state=rng) * vte
    sign = rng.choice(np.array([-1.0, 1.0]), size=n_p)
    v = sign * speed
    u = v / np.sqrt(1.0 - (v / c) ** 2)

    # positions: cell weighted by density, uniform within the cell
    x_grid = np.array(cfg["grid"]["x"])
    dx = cfg["grid"]["dx"]
    density = np.array(cfg["grid"]["background_density"])[:, 0]
    p_cell = density / np.sum(density)
    cells = rng.choice(density.size, size=n_p, p=p_cell)
    x = x_grid[cells] + (rng.uniform(size=n_p) - 0.5) * dx

    arrays = resonance_arrays(cfg)
    nv = int(hpe["nv"])
    v_max = float(hpe["v_max"]) * c
    counts, _ = np.histogram(v, bins=nv, range=(-v_max, v_max))
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
    full state dict (complex-viewed). The damping array it writes into
    ``y["gamma_L"]`` is consumed by ``SpectralEPWSolver`` on the *next* field step
    (one-step lag, far tighter than the paper's 100 fs update interval).
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
        self.dx = grid["dx"]
        self.xmin, self.xmax = grid["xmin"], grid["xmax"]
        self.Lx = self.xmax - self.xmin
        self.kx = jnp.array(grid["kx"])
        self.periodic = cfg["terms"]["epw"]["boundary"]["x"] == "periodic"

        # gather from a spectrally upsampled field: linear interpolation of a wave
        # sampled at k*dx ~ 1-2 rad/cell attenuates the gathered field by
        # sinc^2(k dx/2) (~15-30%!); zero-padding ex_k by gather_refine before the
        # ifft makes that error ~1% for one cheap length-(refine*nx) FFT per step
        self.refine = int(hpe["gather_refine"])
        self.nx_f = self.refine * self.nx
        self.dx_f = self.dx / self.refine
        self.x0 = float(np.array(grid["x"])[0])  # cell-centered origin, shared by both grids
        self.pad_lo = (self.nx_f - self.nx) // 2
        self.pad_hi = self.nx_f - self.nx - self.pad_lo

        self.n_sub = int(hpe["substeps"])
        self.dtp = self.dt / self.n_sub
        self.alpha = self.dt / hpe["tau_damping_ps"]  # EMA weight
        self.t_start = hpe["t_start_ps"]
        self.feedback = bool(hpe["feedback"])
        self.n_p = int(hpe["n_particles"])
        self.nv = int(hpe["nv"])
        self.v_min = float(hpe["v_min"]) * self.vte
        self.seed = int(hpe["seed"]) + 7919  # decorrelate from the EPW noise seed
        # jnp.histogram needs concrete bin edges; keep them (and centers) on-device
        arrays = resonance_arrays(cfg)
        self.v_max = float(hpe["v_max"]) * self.c
        self.dv = arrays["dv"]
        self.v_centers = jnp.array(arrays["v_centers"])
        self.v_edges = jnp.linspace(-self.v_max, self.v_max, self.nv + 1)
        self.v_phi = jnp.array(arrays["v_phi"])
        self.mask_res = jnp.array(arrays["mask_res"])
        self.gamma_analytic = jnp.array(arrays["gamma_analytic"])
        self.f_tail_frac = arrays["f_tail_frac"]

        # per-k calibration: the discrete operator applied to the expected initial
        # histogram must return the analytic rate exactly (see module docstring)
        gamma_raw0 = np.array(self._gamma_raw(jnp.array(arrays["f0_expected"])))
        gamma_an_1d = np.array(arrays["gamma_analytic"][:, 0])
        mask = np.array(arrays["mask_res"])
        with np.errstate(divide="ignore", invalid="ignore"):
            calib = np.where(mask & (gamma_raw0 > 0), gamma_an_1d / np.where(gamma_raw0 > 0, gamma_raw0, 1.0), 1.0)
        self.calibration = jnp.array(calib)
        if np.any(mask):
            c_band = calib[mask & (gamma_an_1d > 1.0e-6 * self.wp0)]
            if c_band.size:
                print(
                    f"HPE: {self.n_p} particles, {self.n_sub} substeps/step, "
                    f"calibration C(k) in [{c_band.min():.3f}, {c_band.max():.3f}] over the resonant band"
                )

    # ------------------------------------------------------------------ push --

    def refine_ex(self, phi_k: Array) -> Array:
        """(refine * nx,) complex Ex envelope on the upsampled grid from phi_k."""
        ex_k = -1j * self.kx * phi_k[:, 0]
        ex_k_fine = jnp.fft.ifftshift(jnp.pad(jnp.fft.fftshift(ex_k), (self.pad_lo, self.pad_hi)))
        return jnp.fft.ifft(ex_k_fine) * self.refine / self.ny  # ifft2(phi_k) would carry a 1/ny

    def _accel(self, x: Array, ex_env: Array, t: float) -> Array:
        """-(e/m) * Re[Ex_envelope(x) * exp(-i wp0 t)] gathered at particle positions
        (linear interpolation on the upsampled grid, clamped at the walls)."""
        idx = (x - self.x0) / self.dx_f
        i0 = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, self.nx_f - 2)
        w = jnp.clip(idx - i0, 0.0, 1.0)
        ex_p = ex_env[i0] * (1.0 - w) + ex_env[i0 + 1] * w
        carrier = jnp.exp(-1j * self.wp0 * t)
        return -self.q_over_m * jnp.real(ex_p * carrier)

    def _substep(self, i, carry, ex_env, t0):
        x, u = carry
        t_i = t0 + i * self.dtp
        u_half = u + 0.5 * self.dtp * self._accel(x, ex_env, t_i)
        gamma = jnp.sqrt(1.0 + (u_half / self.c) ** 2)
        x = x + self.dtp * u_half / gamma
        if self.periodic:
            x = jnp.mod(x - self.xmin, self.Lx) + self.xmin
        u = u_half + 0.5 * self.dtp * self._accel(x, ex_env, t_i + self.dtp)
        return x, u

    def _apply_boundaries(self, x: Array, u: Array, t: float) -> tuple[Array, Array]:
        """Thermalizing walls: exiting particles are re-injected at the wall with an
        inward, flux-weighted tail speed (p(|v|) ~ |v| exp(-v^2/2vte^2), |v| > v_min,
        analytic inverse CDF), which keeps the loaded tail distribution stationary."""
        if self.periodic:
            return x, u
        out_left = x < self.xmin
        out_right = x > self.xmax
        key = jax.random.PRNGKey(self.seed + jnp.asarray(t / self.dt).astype(jnp.int32))
        uni = jax.random.uniform(key, (self.n_p,), minval=1.0e-12, maxval=1.0)
        speed = jnp.sqrt(self.v_min**2 - 2.0 * self.vte**2 * jnp.log(uni))
        speed = jnp.minimum(speed, 0.99 * self.c)
        u_new = speed / jnp.sqrt(1.0 - (speed / self.c) ** 2)
        x = jnp.where(out_left, self.xmin, jnp.where(out_right, self.xmax, x))
        u = jnp.where(out_left, u_new, jnp.where(out_right, -u_new, u))
        return x, u

    def push(self, x: Array, u: Array, ex_env: Array, t: float) -> tuple[Array, Array]:
        """Subcycled relativistic KDK leapfrog across one field step [t, t + dt]."""
        x, u = jax.lax.fori_loop(0, self.n_sub, lambda i, carry: self._substep(i, carry, ex_env, t), (x, u))
        return self._apply_boundaries(x, u, t)

    # ------------------------------------------------- histogram and damping --

    def histogram(self, u: Array) -> Array:
        v = u / jnp.sqrt(1.0 + (u / self.c) ** 2)
        counts, _ = jnp.histogram(v, bins=self.v_edges)
        # int counts / float promotes to the active float dtype (envelope-2d may run f32)
        return counts / (self.n_p * self.dv)

    def _gamma_raw(self, hist: Array) -> Array:
        """Uncalibrated Follett Eq. 4 in 1D: gamma(kx) = -(pi/2) wp0^3/k^2 sgn(k) f'(v_phi).

        The sgn(k) makes the damping even in k for a symmetric distribution: -k modes
        resonate at v_phi < 0 where the Maxwellian slope is positive -- each
        propagation direction damps on its own tail."""
        dfdv = jnp.gradient(hist, self.dv) * self.f_tail_frac
        dfdv_at_vphi = jnp.interp(self.v_phi, self.v_centers, dfdv)
        kx_safe = jnp.where(jnp.abs(self.kx) > 0, self.kx, 1.0)
        return jnp.where(
            jnp.abs(self.kx) > 0, -0.5 * np.pi * self.wp0**3 * jnp.sign(kx_safe) / kx_safe**2 * dfdv_at_vphi, 0.0
        )

    def damping(self, hist: Array) -> Array:
        """Blended (nx, ny) damping array: calibrated HPE rate on resonant modes
        (clamped >= 0), analytic rate elsewhere."""
        gamma_hpe = jnp.maximum(self.calibration * self._gamma_raw(hist), 0.0)
        gamma_1d = jnp.where(self.mask_res, gamma_hpe, self.gamma_analytic[:, 0])
        return jnp.broadcast_to(gamma_1d[:, None], (self.nx, self.ny))

    # ---------------------------------------------------------------- driver --

    def __call__(self, t: float, y: dict[str, Array]) -> dict[str, Array]:
        phi_k = y["epw"]

        def active(operand):
            x, u, hist, gamma_L = operand
            ex_env = self.refine_ex(phi_k)
            x, u = self.push(x, u, ex_env, t)
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
