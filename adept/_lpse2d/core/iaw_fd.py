"""The finite-difference ion-acoustic solver with plasma-flow profiles (LPSE ``iaw.solver = fd``,
``IawSolver.cpp``) -- plan 2 I.1 / I.2.

LPSE's FD path advances the density perturbation ``n`` and the velocity divergence ``w`` on a
grid refined ``super_samples`` times (``iaw.fd.superSamples``, default 2) in sub-steps of
``dt_fraction h / (sqrt(nDim) cs + |U|_max)``:

1. ``n`` is advected by the flow ``U(x, t)`` with a dimensionally split PPM sweep per axis
   (``ppmAdvection``: Colella-Woodward parabolas with the van Leer limited slopes, conservative
   fluxes with the face velocities ``(U_i + U_{i+1})/2``; both upwind states are formed here,
   LPSE forms only the one of each cell's own flow sign), then ``n -= dt w`` with the
   second-order temporal correction ``+ dt^2/(4h) U . grad w``, the absorbing layer, the amplitude
   clamp, zero edge cells and LPSE's zero average (``IawSolver::oneTimeStep``);
2. ``w`` is advected likewise, then ``w += dt S`` with ``S = -cs^2 lap n + E2`` and the correction
   ``- dt^2/(4h) U . grad S`` (``addSourceToWo``), the absorbing layer, zero edges and zero average.
   ``E2`` is the ponderomotive term ``-lap(PP) = k^2 PP``, formed spectrally on the EPW grid and
   interpolated up (``ZakharovSolver::getPonderomotivePotential`` / ``advanceNelfAndDivV_fd``), plus
   the thermal-filamentation source;
3. every ``landau_update`` sub-steps, counted over the whole run from the first one
   (``IawSolver::evolve``: ``timeStepIndex % (numStepsPerStep * numStepsPerLandauDampingUpdate)``),
   ``w`` is damped in k-space at ``2 gamma_L(k) + 2 nu_coll`` over the ``landau_update`` sub-steps
   and the noise is added for the same interval, on the EPW grid's band
   (``applyFFTDampingAndNoise``, ``addNoise``), then the edges and the average are zeroed again.
   As LPSE, a ``simplified`` Landau form with a zero rate switches the whole k-space step off
   (collisions and noise included, ``IawSolver.cpp:2428``).

The zero average is LPSE's ``forceAverageToZero``: the positive (or negative) values are
rescaled so that the sum vanishes, not shifted; it is off with thermal filamentation, as in LPSE.

The coarse-grid ``iaw_density`` / ``iaw_velocity_divergence`` the light and EPW steps read are
the band-limited restriction of the fine fields (FFT truncation); the fine fields are the state
keys ``iaw_density_fine`` / ``iaw_velocity_divergence_fine`` when ``super_samples > 1``.

Flow profiles (``terms.iaw.flow`` as a mapping; LPSE ``iaw.velocityProfile.*``): ``shape``
``linear`` | ``gaussian`` | ``log``, ``from_location`` / ``to_location`` (um from the box centre),
``from_mach`` / ``to_mach``, ``sg_order``, ``geometry`` ``cartesian`` (flow along from -> to, speed
by the projected distance) | ``spherical`` (radial from ``from_location``), ``temporal_slope``
(1/ps: ``U(t) = U (1 + slope t)``), or ``file`` (a ``.npy`` / ``.npz`` with ``ux``, ``uy`` in Mach on
the EPW grid). A two-element list keeps the uniform flow of the spectral solver.
"""

import jax
import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._lpse2d.helpers import _Q

FD_STATE_KEYS = ("iaw_density_fine", "iaw_velocity_divergence_fine")


# ----------------------------------------------------------------------- flow --


def flow_profile(cfg: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``U(x, y)`` in um/ps, shape ``(nx, ny, 2)``, from ``terms.iaw.flow`` (zeros without it)."""
    flow = cfg["terms"]["iaw"].get("flow")
    cs = float(cfg["units"]["derived"]["cs"])
    u = np.zeros((x.size, y.size, 2))
    if flow is None:
        return u
    if not isinstance(flow, dict):
        mach = np.asarray(flow, dtype=np.float64)
        u[..., 0] = mach[0] * cs
        u[..., 1] = (mach[1] if mach.size > 1 else 0.0) * cs
        return u
    if flow.get("file"):
        data = np.load(flow["file"])
        ux = np.asarray(data["ux"] if "ux" in data else data, dtype=np.float64)
        uy = np.asarray(data["uy"], dtype=np.float64) if "uy" in data else np.zeros_like(ux)
        if ux.shape != (x.size, y.size):
            raise ValueError(f"terms.iaw.flow.file arrays must be (nx, ny) = {(x.size, y.size)}, got {ux.shape}")
        u[..., 0], u[..., 1] = ux * cs, uy * cs
        return u
    shape = str(flow.get("shape", "linear")).lower()
    xc = 0.5 * (x[0] + x[-1])
    yc = 0.5 * (y[0] + y[-1])
    loc = np.stack(np.meshgrid(x - xc, y - yc, indexing="ij"), axis=-1)  # (nx, ny, 2) from the centre
    from_loc = np.asarray([_Q(v).to("um").value for v in list(flow.get("from_location", ["0um", "0um"]))[:2]])
    to_loc = np.asarray([_Q(v).to("um").value for v in list(flow.get("to_location", ["1um", "0um"]))[:2]])
    from_speed = float(flow.get("from_mach", 0.0)) * cs
    to_speed = float(flow.get("to_mach", 0.0)) * cs
    from_to = to_loc - from_loc
    separation = float(np.linalg.norm(from_to))
    if separation <= 0.0:
        raise ValueError("terms.iaw.flow from_location and to_location must differ")
    cartesian = str(flow.get("geometry", "cartesian")).lower() == "cartesian"
    if cartesian:
        direction = np.broadcast_to(from_to / separation, loc.shape)
        dist = np.einsum("...i,i->...", loc - from_loc, from_to) / separation
    else:
        rel = loc - from_loc
        dist = np.linalg.norm(rel, axis=-1)
        direction = rel / np.where(dist > 0.0, dist, 1.0)[..., None]
    lo, hi = min(from_speed, to_speed), max(from_speed, to_speed)
    if shape == "linear":
        speed = np.clip((to_speed - from_speed) / separation * dist + from_speed, lo, hi)
    elif shape == "gaussian":
        if from_speed <= 0.0 or to_speed <= 0.0:
            raise ValueError("a gaussian flow profile needs positive from_mach and to_mach")
        order = float(flow.get("sg_order", 2.0))
        sd = separation / np.sqrt(-np.log(lo / hi))
        if to_speed < from_speed:
            speed = from_speed * np.exp(-(np.abs(dist / sd) ** order))
        else:
            speed = (1.0 - np.exp(-(np.abs(dist / sd) ** order))) * to_speed + from_speed
    elif shape == "log":
        if cartesian:
            raise ValueError("the log flow profile is spherical (geometry: spherical)")
        x0, x1 = to_loc[0], to_loc[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dist - x0 > 0.0, from_speed + to_speed * np.log(np.maximum(dist - x0, 1e-300) / x1), 0.0)
        speed = np.maximum(speed, 0.0)
    else:
        raise ValueError(f"terms.iaw.flow.shape must be linear, gaussian, log or file, got {shape!r}")
    return speed[..., None] * direction


# ------------------------------------------------------------------ resampling --


def upsample(field: Array, s: int) -> Array:
    """Band-limited (FFT zero-padded) interpolation of a periodic ``(nx, ny)`` field to ``(s nx, s ny)``
    (``(s nx, 1)`` for a 1-D box, whose y axis is not refined)."""
    if s == 1:
        return field
    nx, ny = field.shape
    sy = s if ny > 1 else 1
    f_k = jnp.fft.fftshift(jnp.fft.fft2(field))
    padded = jnp.zeros((s * nx, sy * ny), dtype=f_k.dtype)
    ox, oy = (s * nx - nx) // 2, (sy * ny - ny) // 2
    padded = padded.at[ox : ox + nx, oy : oy + ny].set(f_k)
    return jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(padded))) * (s * sy)


def downsample(field: Array, s: int) -> Array:
    """The band-limited restriction of a ``(s nx, s ny)`` (1-D: ``(s nx, 1)``) field to ``(nx, ny)``
    (FFT truncation)."""
    if s == 1:
        return field
    snx, sny = field.shape
    sy = s if sny > 1 else 1
    nx, ny = snx // s, sny // sy
    f_k = jnp.fft.fftshift(jnp.fft.fft2(field))
    ox, oy = (snx - nx) // 2, (sny - ny) // 2
    kept = f_k[ox : ox + nx, oy : oy + ny]
    return jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(kept))) / (s * sy)


# ------------------------------------------------------------------------ PPM --


def ppm_sweep(f: Array, lam: Array, axis: int) -> Array:
    """One conservative PPM advection sweep of ``f`` along ``axis`` with the cell Courant numbers
    ``lam = U dt / h`` (LPSE ``ppmAdvection``; periodic indexing -- the edge cells are zeroed
    by the caller when the axis is absorbing)."""

    def roll(a, n):
        return jnp.roll(a, n, axis=axis)

    fm, fp, fm2, fp2 = roll(f, 1), roll(f, -1), roll(f, 2), roll(f, -2)

    def limited(fm_, fc_, fp_):
        mono = (fp_ - fc_) * (fc_ - fm_) >= 0.0
        d = jnp.minimum(jnp.abs(fp_ - fm_) / 2.0, 2.0 * jnp.abs(fp_ - fc_))
        d = jnp.sign(fp_ - fc_) * jnp.minimum(d, 2.0 * jnp.abs(fc_ - fm_))
        return jnp.where(mono, d, 0.0)

    d_i = limited(fm, f, fp)
    d_p = limited(f, fp, fp2)
    d_m = limited(fm2, fm, f)
    f_r = f + 0.5 * (fp - f) + (d_i - d_p) / 6.0
    f_l = fm + 0.5 * (f - fm) + (d_m - d_i) / 6.0
    flat = (f_r - f) * (f - f_l) <= 0.0
    f_r = jnp.where(flat, f, f_r)
    f_l = jnp.where(flat, f, f_l)
    delta = f_r - f_l
    f6 = 6.0 * (f - 0.5 * (f_l + f_r))
    f_l = jnp.where(delta * f6 > delta**2, 3.0 * f - 2.0 * f_r, f_l)
    f_r = jnp.where(-(delta**2) > delta * f6, 3.0 * f - 2.0 * f_l, f_r)
    delta = f_r - f_l
    f6 = 6.0 * (f - 0.5 * (f_l + f_r))
    fbar_plus = f_r - 0.5 * lam * (delta - (1.0 - 2.0 / 3.0 * lam) * f6)
    fbar_minus = f_l - 0.5 * lam * (delta + (1.0 + 2.0 / 3.0 * lam) * f6)
    lam_minus = 0.5 * (roll(lam, 1) + lam)
    lam_plus = 0.5 * (lam + roll(lam, -1))
    rightward = lam_plus + lam_minus > 0.0
    flux_in = jnp.where(rightward, lam_minus * roll(fbar_plus, 1), -lam_plus * roll(fbar_minus, -1))
    flux_out = jnp.where(rightward, lam_plus * fbar_plus, -lam_minus * fbar_minus)
    return f + flux_in - flux_out


# --------------------------------------------------------------------- solver --


class FDIonAcoustic:
    """The FD IAW step, driven by ``IAW`` (which owns the units, damping table and drive)."""

    def __init__(self, iaw, cfg: dict):
        grid = cfg["grid"]
        opts = cfg["terms"]["iaw"]
        self.iaw = iaw
        self.s = int(opts.get("super_samples", 2))
        if self.s < 1:
            raise ValueError("terms.iaw.super_samples must be >= 1")
        self.dt_fraction = float(opts.get("dt_fraction", 0.95))
        self.landau_update = int(opts.get("landau_update", 1))
        self.temporal_correction = bool(opts.get("temporal_correction", True))
        self.nx, self.ny = int(grid["nx"]), int(grid["ny"])
        self.is_2d = self.ny > 1
        self.fnx, self.fny = self.s * self.nx, (self.s * self.ny if self.is_2d else 1)
        self.h = float(grid["dx"]) / self.s
        if self.is_2d and abs(grid["dy"] - grid["dx"]) > 1e-12 * grid["dx"]:
            raise ValueError("terms.iaw.solver: fd needs dx == dy (LPSE's h_xyz)")
        self.dt_iaw = float(iaw.dt)  # the IAW step (stride EPW steps)
        cs = float(iaw.cs)
        x = np.asarray(grid["x"], dtype=np.float64)
        y = np.asarray(grid["y"], dtype=np.float64)
        # the flow on the fine grid (the coarse profile interpolated linearly)
        u_coarse = flow_profile(cfg, x, y)
        self.flow_enabled = bool(np.any(u_coarse != 0.0))
        self.u = jnp.asarray(self._refine_profile(u_coarse))
        flow = opts.get("flow")
        self.temporal_slope = float(flow.get("temporal_slope", 0.0)) if isinstance(flow, dict) else 0.0
        u_max = float(np.max(np.abs(u_coarse))) if self.flow_enabled else 0.0
        t_end = float(_Q(grid["tmax"]).to("ps").value) if isinstance(grid["tmax"], str) else float(grid["tmax"])
        u_max *= max(1.0, abs(1.0 + self.temporal_slope * t_end))
        dt_crit = self.h / (np.sqrt(2.0 if self.is_2d else 1.0) * cs + u_max)
        self.n_sub = max(1, int(np.ceil(self.dt_iaw / (self.dt_fraction * dt_crit))))
        self.dt_sub = self.dt_iaw / self.n_sub
        self.cs_sq = cs**2
        # boundary profile, edge mask and the k-space damping on the fine grid
        boundary = np.asarray(grid["iaw_absorbing_boundaries"], dtype=np.float64)
        self.boundary = jnp.asarray(self._refine_profile(boundary[..., None])[..., 0])
        edge = np.ones((self.fnx, self.fny))
        epw_boundary = cfg["terms"]["epw"]["boundary"]
        # LPSE zeroOutEdgeCells: only along an axis with an IAW layer (iaw.Labc > 0)
        has_layer = float(grid.get("iaw_boundary_width_um", 1.0)) > 0.0
        if has_layer and str(opts.get("boundary", epw_boundary).get("x", "periodic")) != "periodic":
            edge[0, :] = edge[-1, :] = 0.0
        if has_layer and self.is_2d and str(opts.get("boundary", epw_boundary).get("y", "periodic")) != "periodic":
            edge[:, 0] = edge[:, -1] = 0.0
        self.edge = jnp.asarray(edge)
        fkx = 2.0 * np.pi * np.fft.fftfreq(self.fnx, d=self.h)
        fky = 2.0 * np.pi * np.fft.fftfreq(self.fny, d=self.h) if self.is_2d else np.zeros(1)
        fk_sq = fkx[:, None] ** 2 + fky[None, :] ** 2
        from adept._lpse2d.core.iaw import ion_landau_rate

        gamma = ion_landau_rate(cfg, fk_sq)
        # the same physical band as the coarse grid (LPSE rescales the anti-aliasing range)
        k_max_x = float(np.max(np.abs(np.asarray(grid["kx"]))))
        k_max_y = float(np.max(np.abs(np.asarray(grid["ky"])))) if self.is_2d else np.inf
        band = (np.abs(fkx)[:, None] <= k_max_x) & (np.abs(fky)[None, :] <= k_max_y)
        coarse_filter = np.asarray(grid["low_pass_filter_grid"] * grid["zero_mask"])
        coarse_k_sq = np.asarray(grid["kx"])[:, None] ** 2 + np.asarray(grid["ky"])[None, :] ** 2
        k_cut = np.sqrt(np.max(np.where(coarse_filter > 0.0, coarse_k_sq, 0.0)))
        band = band & (np.sqrt(fk_sq) <= k_cut) & (fk_sq > 0.0)
        # k-space step every landau_update sub-steps over their combined length (LPSE
        # applyFFTDampingAndNoise(stepSize * numStepsPerLandauDampingUpdate)): w_k *= exp(-W_i dt)
        # with W_i = 2 gamma_L + 2 nu_coll; modes outside the band are zeroed
        self.dt_damp = self.dt_sub * self.landau_update
        nu_coll = float(iaw.nu_coll)
        self.damping_factor = jnp.asarray(np.where(band, np.exp(-(2.0 * gamma + 2.0 * nu_coll) * self.dt_damp), 0.0))
        self.band = jnp.asarray(band.astype(np.float64))
        damping = opts["damping"]
        # LPSE IawSolver.cpp:2428: a simplified Landau form with a zero rate skips the whole k-space
        # step (collisional damping and noise included)
        self.kspace_step = not (
            str(damping.get("landau_form", "simplified")) == "simplified" and float(damping["landau"]) <= 0.0
        )
        # LPSE forceAverageToZero is off with thermal filamentation (IawSolver.cpp:196-199)
        self.force_average = not bool(iaw.thermal_waves)
        # noise (LPSE IawSolver::addNoise) on the fine grid: A N_fine sqrt(exp(2 dt (gamma_L + nu)) - 1)
        # with a random phase on |k| < (1 - antiAliasing.range) pi / h_coarse, k != 0; the field
        # outside that disc is zeroed when the noise is on
        self.noise = bool(iaw.noise_enabled) and self.kspace_step
        if self.noise:
            k_noise = float(grid.get("low_pass_filter", 1.0)) * k_max_x
            noise_band = (np.sqrt(fk_sq) < k_noise) & (fk_sq > 0.0)
            amplitude = float(opts.get("noise_amplitude", 1.0))
            kick = amplitude * float(self.fnx * self.fny) * np.sqrt(np.expm1(2.0 * self.dt_damp * (gamma + nu_coll)))
            self.noise_kick = jnp.asarray(np.where(noise_band, kick, 0.0))
            self.noise_band = jnp.asarray(noise_band.astype(np.float64))
            self.noise_key = jax.random.fold_in(iaw.noise_key, 1)

    def _refine_profile(self, a: np.ndarray) -> np.ndarray:
        """Linear interpolation of a coarse ``(nx, ny, c)`` profile to the fine grid."""
        if self.s == 1:
            return a
        from scipy.ndimage import zoom

        return zoom(a, (self.s, self.s if self.is_2d else 1, 1), order=1, mode="nearest")

    def laplacian(self, f: Array) -> Array:
        lap = (jnp.roll(f, -1, axis=0) - 2.0 * f + jnp.roll(f, 1, axis=0)) / self.h**2
        if self.is_2d:
            lap = lap + (jnp.roll(f, -1, axis=1) - 2.0 * f + jnp.roll(f, 1, axis=1)) / self.h**2
        return lap

    def _u_dot_grad(self, u: Array, f: Array) -> Array:
        """``U . grad f`` with central differences (the temporal-correction terms)."""
        out = u[..., 0] * (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * self.h)
        if self.is_2d:
            out = out + u[..., 1] * (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * self.h)
        return out

    def _advect(self, f: Array, lam: Array) -> Array:
        if not self.flow_enabled:
            return f
        f = ppm_sweep(f, lam[..., 0], axis=0) * self.edge
        if self.is_2d:
            f = ppm_sweep(f, lam[..., 1], axis=1) * self.edge
        return f

    def _average_to_zero(self, f: Array) -> Array:
        """LPSE ``IawSolver::forceAverageToZero``: rescale the positive values (when their sum
        outweighs the negative one) or the negative values so that the total vanishes."""
        if not self.force_average:
            return f
        s_plus = jnp.sum(jnp.where(f > 0.0, f, 0.0))
        s_minus = jnp.sum(jnp.where(f < 0.0, f, 0.0))
        both = (s_plus != 0.0) & (s_minus != 0.0)
        scale_positive = jnp.where(f > 0.0, f * (-s_minus / jnp.where(s_plus != 0.0, s_plus, 1.0)), f)
        scale_negative = jnp.where(f < 0.0, f * (-s_plus / jnp.where(s_minus != 0.0, s_minus, 1.0)), f)
        return jnp.where(both, jnp.where(s_plus > jnp.abs(s_minus), scale_positive, scale_negative), f)

    def _edges_and_average(self, f: Array) -> Array:
        return self._average_to_zero(f * self.edge)

    def step(self, t: float, n: Array, w: Array, drive_lap: Array) -> tuple[Array, Array]:
        """Advance the fine-grid ``(n, w)`` over one IAW step (``n_sub`` sub-steps); ``t`` is the
        step's start, which numbers the sub-steps over the run for the k-space cadence."""
        u_t = self.u * (1.0 + self.temporal_slope * t)
        lam = u_t * self.dt_sub / self.h
        dt = self.dt_sub
        clamp = self.iaw.max_density_perturbation
        first = jnp.round(t / self.dt_iaw).astype(jnp.int32) * self.n_sub

        def substep(i, carry):
            n, w = carry
            # 1. density (IawSolver::oneTimeStep): advect, -dt w (+ the second-order correction),
            # absorbers, clamp, edges, average
            n = self._advect(n, lam)
            n = n - dt * w
            if self.temporal_correction:
                n = n + 0.5 * dt**2 * self._u_dot_grad(u_t, w)
            n = n * self.boundary
            if clamp is not None:
                n = jnp.clip(n, -clamp, clamp)
            n = self._edges_and_average(n)
            # 2. velocity divergence: advect, + dt S (+ correction), absorbers, edges, average
            w = self._advect(w, lam)
            source = -self.cs_sq * self.laplacian(n) + drive_lap
            w = w + dt * source
            if self.temporal_correction:
                w = w - 0.5 * dt**2 * self._u_dot_grad(u_t, source)
            w = self._edges_and_average(w * self.boundary)
            if not self.kspace_step:
                return n.astype(carry[0].dtype), w.astype(carry[1].dtype)

            # 3. damping and noise in k-space on the sub-steps first, first + landau_update, ...
            g = first + i

            def damp(w):
                w_k = jnp.fft.fft2(w) * self.damping_factor
                if self.noise:
                    key = jax.random.fold_in(self.noise_key, g)
                    phases = 2.0 * jnp.pi * jax.random.uniform(key, w_k.shape)
                    w_k = (w_k + self.noise_kick * jnp.exp(1j * phases)) * self.noise_band
                return self._edges_and_average(jnp.real(jnp.fft.ifft2(w_k)))

            w = lax.cond(g % self.landau_update == 0, damp, lambda w: w, w)
            return n.astype(carry[0].dtype), w.astype(carry[1].dtype)

        return lax.fori_loop(0, self.n_sub, substep, (n, w))
