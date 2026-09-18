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
   clamp, zero edge cells and a zero mean;
2. ``w`` is advected likewise, then ``w += dt S`` with ``S = -cs^2 lap n + lap(drive)`` (the
   ponderomotive drive interpolated up from the EPW grid) and the correction ``- dt^2/(4h) U . grad S``,
   the absorbing layer, zero edges and zero mean;
3. every ``landau_update`` sub-steps the ion Landau damping (and the noise) act on ``w`` in
   k-space, on the same physical band as the EPW grid.

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
    """Band-limited (FFT zero-padded) interpolation of a periodic ``(nx, ny)`` field to ``(s nx, s ny)``."""
    if s == 1:
        return field
    nx, ny = field.shape
    f_k = jnp.fft.fftshift(jnp.fft.fft2(field))
    padded = jnp.zeros((s * nx, s * ny), dtype=f_k.dtype)
    ox, oy = (s * nx - nx) // 2, (s * ny - ny) // 2
    padded = padded.at[ox : ox + nx, oy : oy + ny].set(f_k)
    return jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(padded))) * s**2


def downsample(field: Array, s: int) -> Array:
    """The band-limited restriction of a ``(s nx, s ny)`` field to ``(nx, ny)`` (FFT truncation)."""
    if s == 1:
        return field
    snx, sny = field.shape
    nx, ny = snx // s, sny // s
    f_k = jnp.fft.fftshift(jnp.fft.fft2(field))
    ox, oy = (snx - nx) // 2, (sny - ny) // 2
    kept = f_k[ox : ox + nx, oy : oy + ny]
    return jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(kept))) / s**2


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
        if str(opts.get("boundary", epw_boundary).get("x", "periodic")) != "periodic":
            edge[0, :] = edge[-1, :] = 0.0
        if self.is_2d and str(opts.get("boundary", epw_boundary).get("y", "periodic")) != "periodic":
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
        self.damping_factor = jnp.asarray(np.where(band, np.exp(-2.0 * gamma * self.dt_sub * self.landau_update), 0.0))
        self.band = jnp.asarray(band.astype(np.float64))

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

    def _clean(self, f: Array) -> Array:
        f = f * self.boundary * self.edge
        return f - jnp.mean(f)

    def step(self, t: float, n: Array, w: Array, drive_lap: Array, noise_k) -> tuple[Array, Array]:
        """Advance the fine-grid ``(n, w)`` over one IAW step (``n_sub`` sub-steps)."""
        u_t = self.u * (1.0 + self.temporal_slope * t)
        lam = u_t * self.dt_sub / self.h
        dt = self.dt_sub
        clamp = self.iaw.max_density_perturbation

        def substep(i, carry):
            n, w = carry
            # 1. density: advect, -dt w (+ the second-order correction), absorbers, clamp
            n = self._advect(n, lam)
            n = n - dt * w
            if self.temporal_correction:
                n = n + 0.5 * dt**2 * self._u_dot_grad(u_t, w)
            if clamp is not None:
                n = jnp.clip(n, -clamp, clamp)
            n = self._clean(n)
            # 2. velocity divergence: advect, + dt S (+ correction), absorbers
            w = self._advect(w, lam)
            source = -self.cs_sq * self.laplacian(n) + drive_lap
            w = w + dt * source
            if self.temporal_correction:
                w = w - 0.5 * dt**2 * self._u_dot_grad(u_t, source)
            w = self._clean(w)

            # 3. Landau damping (and noise) on w in k-space every landau_update sub-steps
            def damp(w):
                w_k = jnp.fft.fft2(w) * self.damping_factor
                if noise_k is not None:
                    w_k = w_k + noise_k
                return self._clean(jnp.real(jnp.fft.ifft2(w_k)))

            w = lax.cond((i + 1) % self.landau_update == 0, damp, lambda w: w, w)
            return n.astype(carry[0].dtype), w.astype(carry[1].dtype)

        return lax.fori_loop(0, self.n_sub, substep, (n, w))
