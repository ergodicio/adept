"""Quasilinear evolution of the box-averaged electron distribution (LPSE ``qle.*``,
``QuasilinearEvolution.cpp``) -- plan 2 K.1.

The electron velocity distribution ``f(v)`` (box-averaged, on a ``nv``-point grid per
dimension over ``[-v_max, v_max]``, initially the Maxwellian of ``Te``) diffuses in the EPW
spectrum,

    df/dt = d/dv_i ( C_ij(v) df/dv_j ),

    C_ij(v) = (pi/2) (e/m)^2 (Lx Ly / (2 pi)^2) (1/v) sum_perp dk_perp  k_i k_j |phi_k / N|^2

with the sum over the resonant plane ``k . v = w(k)`` (``w = wpe`` cold, or the Bohm-Gross
``sqrt(wpe^2 + 3 vte^2 k^2)`` with ``thermal_correction``), parametrised as LPSE does:
``k_par = sqrt(wpe^2 + 3 vte^2 k_perp^2) / sqrt(v^2 - 3 vte^2)`` along ``v``, ``k_perp`` on
``nx`` points across ``[-k_max sqrt(nDim), k_max sqrt(nDim)]``, ``|phi_k|^2`` interpolated on
the periodic k grid (``getDiffusionCoefficient``). The ``1/v`` above is the resonance
Jacobian ``1/|v - v_g|`` with the Bohm-Gross group velocity when ``thermal_correction`` is on
(LPSE keeps the cold ``1/v`` there, under-counting the exchange by ``3 (k lambda_D)^2 wpe^2 / w_k^2``);
the constant is derived in this code's scaled-Gaussian units from the envelope field
``E = Re(E_env e^{-i wpe t})`` -- the +-k pair of an envelope mode shares one phase velocity,
hence pi/2 with ``|E_env,k|^2`` -- and verified by the wave-energy / electron-heating balance
(tests/test_lpse2d/test_qle.py). The
explicit update (``updateVelocityDistribution_explicit``) is sub-cycled to the diffusion
limit ``dv^2 / (2 nDim max C)`` -- in conservative flux form here rather than LPSE's
``C d^2 f + (d_i C_ij) d_j f`` (see ``_flux_increment``; ``derivative_in_tensor`` is
accepted from the deck and implied) -- relaxes ``f`` towards the Maxwellian at
``p_x |v_x| / Lx + p_y |v_y| / Ly`` (particles leaving the box return thermalised with
``thermalization_probability``), clips ``f >= 0`` and renormalises the density. With
``landau_evolution`` the Landau rate of every EPW mode is recomputed from
the evolved ``f`` (``updateLandauDamping``),

    gamma_L(k) = -(pi/2) wpe^3 / k^3  sum_perp dv_perp  k . grad_v f |_{v = v_phase k^ + v_perp}

(the reduced distribution along ``k``; the Maxwellian gives the textbook rate), and written
to the state's ``gamma_L`` that the EPW step applies; modes with ``v_phase > v_max`` are
undamped, as in LPSE. Every ``update_every`` EPW steps from ``t_start``.
"""

import numpy as np
from jax import Array, lax
from jax import numpy as jnp
from jax.scipy.ndimage import map_coordinates

QLE_KEYS = ("vdf", "gamma_L")


def _qle_cfg(cfg: dict) -> dict:
    from adept._lpse2d.datamodel import QLEModel

    qle = cfg["terms"].get("qle", {}) or {}
    return {**QLEModel(**qle).model_dump(), **qle}


def initial_vdf(cfg: dict) -> np.ndarray:
    """The Maxwellian on the velocity grid, ``(nv,)`` for a quasi-1-D box, ``(nv, nv)`` in 2-D."""
    qle = _qle_cfg(cfg)
    derived = cfg["units"]["derived"]
    vte = np.sqrt(derived["vte_sq"])
    v_max = float(qle["v_max"]) * derived["c"]
    v = np.linspace(-v_max, v_max, int(qle["nv"]))
    if int(cfg["grid"]["ny"]) > 1:
        v_sq = v[:, None] ** 2 + v[None, :] ** 2
        return np.exp(-v_sq / (2.0 * vte**2)) / (2.0 * np.pi * vte**2)
    return np.exp(-(v**2) / (2.0 * vte**2)) / np.sqrt(2.0 * np.pi * vte**2)


def load_qle_state(cfg: dict) -> dict:
    from adept._lpse2d.core.epw import analytic_landau_rate

    return {"vdf": initial_vdf(cfg), "gamma_L": np.asarray(analytic_landau_rate(cfg), dtype=np.float64)}


class QuasilinearEvolution:
    def __init__(self, cfg: dict):
        qle = _qle_cfg(cfg)
        derived = cfg["units"]["derived"]
        grid = cfg["grid"]
        self.wpe = float(derived["wp0"])
        self.vte_sq = float(derived["vte_sq"])
        self.q_over_m = float(derived["e"] / derived["me"])
        self.c = float(derived["c"])
        self.nx, self.ny = int(grid["nx"]), int(grid["ny"])
        self.is_2d = self.ny > 1
        self.ndim = 2 if self.is_2d else 1
        self.dx, self.dy = float(grid["dx"]), float(grid["dy"])
        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        self.dt = float(grid["dt"])
        tmin = grid["tmin"]
        if isinstance(tmin, str):
            from astropy.units import Quantity as _Q

            tmin = _Q(tmin).to("ps").value
        self.tmin = float(tmin)
        self.update_every = int(qle["update_every"])
        self.t_start = float(qle["t_start"])
        self.thermal_correction = bool(qle["thermal_correction"])
        self.derivative_in_tensor = bool(qle["derivative_in_tensor"])
        self.landau_evolution = bool(qle["landau_evolution"])
        self.subcycling = int(qle["subcycling"])
        self.max_subcycles = int(qle["max_subcycles"])
        self.multiplier = float(qle["multiplier"])
        p = list(qle["thermalization_probability"]) + [0.0, 0.0]
        self.p_therm = (float(p[0]), float(p[1]))

        # velocity grid
        self.nv = int(qle["nv"])
        self.v_max = float(qle["v_max"]) * self.c
        v = np.linspace(-self.v_max, self.v_max, self.nv)
        self.v = jnp.asarray(v)
        self.dv = float(v[1] - v[0])
        if self.is_2d:
            vx, vy = np.meshgrid(v, v, indexing="ij")
            self.v_nodes = jnp.asarray(np.stack([vx.ravel(), vy.ravel()], axis=-1))  # (nv^2, 2)
        else:
            self.v_nodes = jnp.asarray(v[:, None])  # (nv, 1)
        f0 = initial_vdf(cfg)
        self.f0 = jnp.asarray(f0)
        self.volume0 = float(np.sum(f0) * self.dv**self.ndim)
        # thermalisation rate on the grid
        rate = self.p_therm[0] * np.abs(v) / self.Lx
        if self.is_2d:
            rate = rate[:, None] + self.p_therm[1] * np.abs(v)[None, :] / self.Ly
        self.therm_rate = jnp.asarray(rate)

        # k grid and the resonant-plane sampling (LPSE: kMax per axis from the anti-aliasing
        # range, Nx points across [-kMax sqrt(nDim), +kMax sqrt(nDim)])
        kx = np.asarray(grid["kx"], dtype=np.float64)
        ky = np.asarray(grid["ky"], dtype=np.float64)
        self.kx, self.ky = jnp.asarray(kx), jnp.asarray(ky)
        self.dkx = 2.0 * np.pi / self.Lx
        self.dky = 2.0 * np.pi / self.Ly if self.is_2d else 0.0
        band = float(grid.get("low_pass_filter", 1.0))
        self.k_max = np.pi / self.dx * band
        self.k_max_plane = self.k_max * np.sqrt(self.ndim)
        if self.is_2d:
            self.nk_plane = self.nx
            self.k_perp = jnp.asarray(np.linspace(-self.k_max_plane, self.k_max_plane, self.nk_plane))
            self.dk_plane = float(2.0 * self.k_max_plane / (self.nk_plane - 1))
        else:
            self.nk_plane = 1
            self.k_perp = jnp.zeros(1)
            self.dk_plane = 1.0
        # the plane of the Landau integral: nv points across [-v_max, v_max]
        if self.is_2d:
            self.v_perp = self.v
            self.dv_plane = self.dv
        else:
            self.v_perp = jnp.zeros(1)
            self.dv_plane = 1.0
        # D_ij constant: (pi/2)(e/m)^2 (Lx Ly/(2 pi)^2) dk_perp  [1-D: (Lx/2 pi)]
        box = self.Lx * self.Ly / (2.0 * np.pi) ** 2 if self.is_2d else self.Lx / (2.0 * np.pi)
        self.diffusion_constant = self.multiplier * 0.5 * np.pi * self.q_over_m**2 * box * self.dk_plane
        self.n_cells = self.nx * self.ny
        # k validity for the Landau update: non-zero, inside the band, phase velocity on the grid
        k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
        k_mag = np.sqrt(k_sq)
        omega = np.sqrt(self.wpe**2 + 3.0 * self.vte_sq * k_sq) if self.thermal_correction else self.wpe
        with np.errstate(divide="ignore", invalid="ignore"):
            v_phase = np.where(k_mag > 0, omega / np.where(k_mag > 0, k_mag, 1.0), np.inf)
        inside = (np.abs(kx)[:, None] <= self.k_max) & (np.abs(ky)[None, :] <= self.k_max)
        self.k_valid = jnp.asarray((k_mag > 0) & inside & (v_phase <= self.v_max))
        self.v_phase = jnp.asarray(np.where(np.isfinite(v_phase), v_phase, 0.0))
        self.k_mag = jnp.asarray(k_mag)
        self.landau_constant = -0.5 * np.pi * self.wpe**3 * self.dv_plane

    # ----------------------------------------------------------------- tensor --

    def _phi_sq_coef(self, phi_k: Array) -> Array:
        """``|phi_k / N|^2`` on the (nx, ny) k grid: the squared Fourier coefficient."""
        return jnp.abs(phi_k) ** 2 / self.n_cells**2

    def diffusion_tensor(self, phi_k: Array) -> Array:
        """``C_ij(v)`` on the velocity grid: ``(nv, nv, 3)`` [C_xx, C_yy, C_xy] in 2-D, ``(nv, 1)``
        [C_xx] in 1-D."""
        phi_sq = self._phi_sq_coef(phi_k)
        vte_sq, wpe = self.vte_sq, self.wpe
        k_perp = self.k_perp

        def at_velocity(v_vec):
            v_sq = jnp.sum(v_vec**2)
            v_mag = jnp.sqrt(v_sq)
            if self.thermal_correction:
                denom_sq = v_sq - 3.0 * vte_sq
                valid = (v_mag > 0.0) & (denom_sq > (wpe / self.k_max_plane) ** 2)
                denom = jnp.sqrt(jnp.where(valid, denom_sq, 1.0))
                k_par = jnp.sqrt(wpe**2 + 3.0 * vte_sq * k_perp**2) / denom
                # the resonance Jacobian |d(w_k - k . v)/dk_par| = v - v_g with the Bohm-Gross
                # group velocity v_g = 3 vte^2 k_par / w_k (LPSE keeps the cold 1/v, which
                # under-counts the exchange by 3 (k lambda_D)^2 wpe^2 / w_k^2 -- 16 % at 0.25)
                omega_k = jnp.sqrt(wpe**2 + 3.0 * vte_sq * (k_par**2 + k_perp**2))
                jacobian = v_mag - 3.0 * vte_sq * k_par / omega_k
            else:
                valid = (v_mag > 0.0) & (v_sq > (wpe / self.k_max_plane) ** 2)
                k_par = wpe / jnp.where(valid, v_mag, 1.0) * jnp.ones_like(k_perp)
                jacobian = v_mag * jnp.ones_like(k_perp)
            jacobian = jnp.where(jacobian > 0.0, jacobian, 1.0)
            v_hat = v_vec / jnp.where(v_mag > 0.0, v_mag, 1.0)
            if self.is_2d:
                perp_hat = jnp.array([-v_hat[1], v_hat[0]])
                kxs = k_par * v_hat[0] + k_perp * perp_hat[0]
                kys = k_par * v_hat[1] + k_perp * perp_hat[1]
                on_grid = (jnp.abs(kxs) <= self.k_max) & (jnp.abs(kys) <= self.k_max)
                coords = [jnp.mod(kxs / self.dkx, self.nx), jnp.mod(kys / self.dky, self.ny)]
                p = map_coordinates(phi_sq, coords, order=1, mode="wrap")
                weight = jnp.where(on_grid & valid, p / jacobian, 0.0)
                c_xx = jnp.sum(weight * kxs * kxs)
                c_yy = jnp.sum(weight * kys * kys)
                c_xy = jnp.sum(weight * kxs * kys)
                return self.diffusion_constant * jnp.array([c_xx, c_yy, c_xy])
            kxs = k_par * v_hat[0]
            on_grid = jnp.abs(kxs) <= self.k_max
            coords = [jnp.mod(kxs / self.dkx, self.nx), jnp.zeros_like(kxs)]
            p = map_coordinates(phi_sq, coords, order=1, mode="wrap")
            weight = jnp.where(on_grid & valid, p / jacobian, 0.0)
            return self.diffusion_constant * jnp.array([jnp.sum(weight * kxs * kxs)])

        tensor = lax.map(at_velocity, self.v_nodes)
        if self.is_2d:
            return tensor.reshape(self.nv, self.nv, 3)
        return tensor.reshape(self.nv, 1)

    # ------------------------------------------------------------- FP update --

    def _flux_increment(self, f: Array, tensor: Array) -> Array:
        """``d/dv_i (C_ij d f/dv_j)`` in conservative flux form: ``C`` averaged to the cell faces,
        the cross derivative on a face as the mean of the two adjacent central differences,
        zero flux through the velocity walls. Density is conserved exactly and the update is
        stable at the diffusion limit ``dv^2 / (2 nDim max C)``. (LPSE differentiates the
        tensor into a drift ``D_j = d_i C_ij`` and steps ``C d^2 f + D . d f``; on a resonance
        narrower than a velocity cell that drift's own CFL ``dv Delta v / C`` is far below the
        diffusion limit its sub-cycling honours, and the update runs away.)"""
        dv = self.dv
        if not self.is_2d:
            c = tensor[:, 0]
            flux = 0.5 * (c[1:] + c[:-1]) * (f[1:] - f[:-1]) / dv
            flux = jnp.concatenate([jnp.zeros(1), flux, jnp.zeros(1)])
            return (flux[1:] - flux[:-1]) / dv
        cxx, cyy, cxy = tensor[..., 0], tensor[..., 1], tensor[..., 2]
        # x faces (i + 1/2, j)
        dfdy = (
            jnp.pad(f, ((0, 0), (1, 1)), mode="edge")[:, 2:] - jnp.pad(f, ((0, 0), (1, 1)), mode="edge")[:, :-2]
        ) / (2.0 * dv)
        fx = 0.5 * (cxx[1:, :] + cxx[:-1, :]) * (f[1:, :] - f[:-1, :]) / dv + 0.5 * (cxy[1:, :] + cxy[:-1, :]) * 0.5 * (
            dfdy[1:, :] + dfdy[:-1, :]
        )
        fx = jnp.pad(fx, ((1, 1), (0, 0)))
        # y faces (i, j + 1/2)
        dfdx = (
            jnp.pad(f, ((1, 1), (0, 0)), mode="edge")[2:, :] - jnp.pad(f, ((1, 1), (0, 0)), mode="edge")[:-2, :]
        ) / (2.0 * dv)
        fy = 0.5 * (cyy[:, 1:] + cyy[:, :-1]) * (f[:, 1:] - f[:, :-1]) / dv + 0.5 * (cxy[:, 1:] + cxy[:, :-1]) * 0.5 * (
            dfdx[:, 1:] + dfdx[:, :-1]
        )
        fy = jnp.pad(fy, ((0, 0), (1, 1)))
        return (fx[1:, :] - fx[:-1, :]) / dv + (fy[:, 1:] - fy[:, :-1]) / dv

    def evolve_vdf(self, f: Array, phi_k: Array, dt: float) -> Array:
        """One quasilinear update of ``f`` over ``dt`` in the spectrum ``phi_k``."""
        tensor = self.diffusion_tensor(phi_k)
        c_max = jnp.max(jnp.abs(tensor))
        dt_max = self.dv**2 / (2.0 * self.ndim * jnp.where(c_max > 0.0, c_max, 1.0))
        n_needed = jnp.where(c_max > 0.0, jnp.ceil(dt / dt_max), 1.0) * self.subcycling
        # LPSE sub-cycles to the diffusion limit however many steps that takes ([qle:N]); the
        # trip count is data-dependent (a while loop under jit), capped at max_subcycles
        n_sub = jnp.clip(n_needed, 1, self.max_subcycles).astype(jnp.int32)
        dt_sub = dt / n_sub

        def substep(i, f):
            f_new = f + dt_sub * self._flux_increment(f, tensor)
            f_new = jnp.maximum(0.0, f_new - dt_sub * self.therm_rate * (f_new - self.f0))
            return f_new * (self.volume0 / (jnp.sum(f_new) * self.dv**self.ndim))

        return lax.fori_loop(0, n_sub, substep, f)

    # ------------------------------------------------------------ Landau rate --

    def landau_rate(self, f: Array) -> Array:
        """``gamma_L(k)`` on the EPW k grid from the evolved ``f`` (0 where the mode's phase
        velocity is off the grid, as LPSE)."""
        # grad f = f grad(ln f): exact for a Maxwellian on any grid (a plain central difference
        # of the exponential over-estimates |f'| by 20 % at dv = 0.13 vte, LPSE's default)
        log_f = jnp.log(jnp.maximum(f, 1e-300))
        if self.is_2d:
            grad = f[..., None] * jnp.stack(
                [jnp.gradient(log_f, self.dv, axis=0), jnp.gradient(log_f, self.dv, axis=1)], axis=-1
            )
        else:
            grad = (f * jnp.gradient(log_f, self.dv))[:, None]
        v_perp = self.v_perp

        def at_k(args):
            kx_, ky_, k_mag, v_ph = args
            k_hat = jnp.array([kx_, ky_]) / jnp.where(k_mag > 0.0, k_mag, 1.0)
            if self.is_2d:
                perp_hat = jnp.array([-k_hat[1], k_hat[0]])
                vxs = v_ph * k_hat[0] + v_perp * perp_hat[0]
                vys = v_ph * k_hat[1] + v_perp * perp_hat[1]
                inside = (jnp.abs(vxs) <= self.v_max) & (jnp.abs(vys) <= self.v_max)
                coords = [(vxs + self.v_max) / self.dv, (vys + self.v_max) / self.dv]
                gx = map_coordinates(grad[..., 0], coords, order=1, mode="constant", cval=0.0)
                gy = map_coordinates(grad[..., 1], coords, order=1, mode="constant", cval=0.0)
                k_dot = jnp.sum(jnp.where(inside, kx_ * gx + ky_ * gy, 0.0))
            else:
                vxs = v_ph * k_hat[0] * jnp.ones(1)
                inside = jnp.abs(vxs) <= self.v_max
                gx = map_coordinates(grad[:, 0], [(vxs + self.v_max) / self.dv], order=1, mode="constant", cval=0.0)
                k_dot = jnp.sum(jnp.where(inside, kx_ * gx, 0.0))
            return self.landau_constant * k_dot / jnp.where(k_mag > 0.0, k_mag, 1.0) ** 3

        kx_grid = jnp.broadcast_to(self.kx[:, None], (self.nx, self.ny)).ravel()
        ky_grid = jnp.broadcast_to(self.ky[None, :], (self.nx, self.ny)).ravel()
        gamma = lax.map(at_k, (kx_grid, ky_grid, self.k_mag.ravel(), self.v_phase.ravel())).reshape(self.nx, self.ny)
        return jnp.where(self.k_valid, gamma, 0.0)

    # -------------------------------------------------------------------- step --

    def __call__(self, t: float, y: dict) -> dict:
        step = jnp.round((t - self.tmin) / self.dt).astype(jnp.int32)
        due = jnp.logical_and(t + 0.5 * self.dt >= self.t_start, (step % self.update_every) == 0)
        f = y["vdf"]

        def update(f):
            f_new = self.evolve_vdf(f, y["epw"], self.update_every * self.dt)
            gamma = self.landau_rate(f_new) if self.landau_evolution else y["gamma_L"]
            return f_new, gamma

        f_new, gamma_new = lax.cond(due, update, lambda f: (f, y["gamma_L"]), f)
        return {**y, "vdf": f_new, "gamma_L": gamma_new}
