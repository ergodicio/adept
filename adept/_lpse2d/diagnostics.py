"""Scalar diagnostics for lpse2d SRS runs, mirroring the OSIRIS scan2 metrics.

The point of this module is one-to-one MLflow comparability with the OSIRIS PIC
post-processing in the ``osiris-lpi`` repo: metric *names and definitions* here
match ``osiris_lpi/laser_budget.py`` (R/T/absorbed window means, per-segment
scalars) and ``osiris_lpi/epw_growth.py`` (growth-rate fit). adept must not
import osiris-lpi, so the pure-numpy pieces are copied verbatim; if you change
the fit or windowing on either side, change it on both.

Unit conventions:

- Series arrive on a ``t (ps)`` axis. OSIRIS logs times and rates in ``1/w0``
  code units, so the growth fit is fed ``t * w0`` and ``epw_growth_rate`` is
  per ``w0`` (with ``epw_growth_rate_per_ps`` logged alongside for humans).
- ``epw_energy`` and ``epw_dissipation`` are already converted to OSIRIS code
  units by the save function (fields in ``me*c*w0/e``, lengths in ``c/w0``).
- Flux series (``incident_flux``, ``reflected_flux``, ``transmitted_flux``,
  ``backrefl_flux``) are normalized to the nominal incident flux ``I0_code``.
"""

from __future__ import annotations

import numpy as np

# --- EPW growth fit: copied verbatim from osiris_lpi/epw_growth.py (fit_epw_growth) ---

ONSET_FACTOR = 2.0  # onset = first sustained W > ONSET_FACTOR*floor
START_FACTOR = 5.0  # fit window starts at first sustained W > START_FACTOR*floor
END_FRACTION = 0.1  # fit window ends when W first reaches END_FRACTION*max(W)
MEASURABLE_FACTOR = 10.0  # need max(W) >= MEASURABLE_FACTOR*floor to fit at all
MIN_FIT_POINTS = 6
SUSTAIN = 5  # "sustained" crossing = 4 of the next 5 samples also above


def _sustained_crossing(W: np.ndarray, level: float) -> int | None:
    """Index of the first crossing of ``level`` where >=4 of the next 5 samples stay above."""
    above = W > level
    idx = np.where(above)[0]
    for i in idx:
        if above[i : i + SUSTAIN].sum() >= min(SUSTAIN - 1, len(W) - i):
            return int(i)
    return None


def fit_epw_growth(t: np.ndarray, W: np.ndarray, floor: float | None = None) -> dict[str, float]:
    """Automated exponential-rise fit of the EPW energy history; returns metric dict.

    Copied verbatim from ``osiris_lpi/epw_growth.py:fit_epw_growth`` so the window
    selection is bit-identical to the scan2 backfill. See that module's docstring
    for the algorithm; do not modify one copy without the other.

    adept extension (the one deliberate departure): ``floor`` optionally supplies an
    externally computed noise floor -- adept's EPW starts at exactly zero and its
    noise level equilibrates over ``1/(2(gamma + nu))``, which can exceed the run
    length, so the OSIRIS early-sample median badly underestimates the floor and can
    certify spurious growth on pure-noise runs. When ``floor`` is given it drives
    the window selection and is logged as ``epw_energy_floor``; the OSIRIS-style
    estimate is still computed and logged as ``epw_energy_floor_measured``.
    """
    t = np.asarray(t, dtype=float)
    W = np.asarray(W, dtype=float)
    good = (W > 0) & np.isfinite(W)
    t, W = t[good], W[good]
    out: dict[str, float] = {}
    if W.size == 0:
        out["epw_growth_measurable"] = 0.0
        return out

    # iterative noise floor: seed from the earliest ~50 dumps (pre-onset even for
    # the fastest-growing runs), then widen the window to the detected onset so
    # slow-onset runs get the robust full pre-onset median
    i_seed = min(52, len(W))
    measured = float(np.median(W[2:i_seed])) if i_seed > 4 else float(np.median(W))
    for _ in range(5):
        onset_i = _sustained_crossing(W, ONSET_FACTOR * measured)
        if onset_i is None or onset_i <= 60:
            break
        new_floor = float(np.median(W[2:onset_i]))
        if abs(new_floor - measured) <= 0.02 * measured:
            measured = new_floor
            break
        measured = new_floor

    out["epw_energy_floor_measured"] = measured
    floor = measured if floor is None else float(floor)

    Wmax = float(W.max())
    out["epw_energy_floor"] = floor
    out["epw_energy_max"] = Wmax

    measurable = Wmax >= MEASURABLE_FACTOR * floor
    i0 = _sustained_crossing(W, START_FACTOR * floor) if measurable else None
    i1 = -1
    if i0 is not None:
        after = np.where(W[i0:] >= END_FRACTION * Wmax)[0]
        i1 = i0 + int(after[0]) if after.size else len(W) - 1
        if i1 - i0 + 1 < MIN_FIT_POINTS:  # near-vertical rise: widen the window
            lo = _sustained_crossing(W, ONSET_FACTOR * floor)
            if lo is not None:
                after = np.where(W[lo:] >= 0.5 * Wmax)[0]
                i0 = lo
                i1 = lo + int(after[0]) if after.size else len(W) - 1
    if i0 is None or i1 - i0 + 1 < MIN_FIT_POINTS:
        out["epw_growth_measurable"] = 0.0
        return out

    def _lin_fit(j0: int, j1: int) -> dict | None:
        if j1 - j0 + 1 < MIN_FIT_POINTS:
            return None
        ts, lnW = t[j0 : j1 + 1], np.log(W[j0 : j1 + 1])
        p = np.polyfit(ts, lnW, 1)
        resid = lnW - np.polyval(p, ts)
        ss_tot = float(np.sum((lnW - lnW.mean()) ** 2))
        return dict(
            epw_growth_rate=float(p[0] / 2.0),
            epw_growth_rate_r2=1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0,
            epw_growth_fit_tstart=float(ts[0]),
            epw_growth_fit_tend=float(ts[-1]),
            epw_growth_efolds=float((lnW[-1] - lnW[0]) / 2.0),
        )

    best = _lin_fit(i0, i1)
    if best is not None and best["epw_growth_efolds"] < 1.0:
        # narrow window (marginal growth): also try extending to 0.5*max(W) and
        # keep whichever window the data fit better -- near-threshold bursty runs
        # stay on the standard window with their honest low R^2
        after = np.where(W[i0:] >= 0.5 * Wmax)[0]
        if after.size:
            wide = _lin_fit(i0, i0 + int(after[0]))
            if wide is not None and wide["epw_growth_rate_r2"] > best["epw_growth_rate_r2"]:
                best = wide
    if best is None:
        out["epw_growth_measurable"] = 0.0
        return out

    out["epw_growth_measurable"] = 1.0
    out.update(best)
    return out


def expected_noise_energy(cfg: dict, t_ps: float) -> float:
    """Expected EPW energy of the pure noise-driven field at time ``t_ps`` (OSIRIS units).

    Each retained k-mode is a damped random walk: the solver adds ``dt * A * e^{i phi}``
    every step and damps by ``e^{-(gamma_k + nu) dt}``, so

        <|phi_k|^2>(t) = (dt A)^2 * (1 - e^{-2 g t}) / (1 - e^{-2 g dt}),   g = gamma_k + nu

    with the ``g -> 0`` limit ``(dt A)^2 * t/dt``. Summed to the same OSIRIS-normalized
    energy as the ``epw_energy`` save quantity (Parseval). Absorbing boundaries and
    detuning-induced mode mixing are neglected, so on absorbing/ramped boxes this is a
    mild overestimate -- conservative for use as a growth-fit noise floor.
    """
    from adept._lpse2d.core.epw import landau_damping_rate

    grid = cfg["grid"]
    derived = cfg["units"]["derived"]
    kx = np.array(grid["kx"])
    ky = np.array(grid["ky"])
    k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
    zero_mask = np.where(k_sq > 0, 1.0, 0.0)
    if cfg["terms"]["epw"]["damping"].get("landau", True):
        gamma = np.array(landau_damping_rate(k_sq, derived["wp0"], derived["vte_sq"], zero_mask))
    else:
        gamma = np.zeros_like(k_sq)
    g = gamma + derived.get("nu_coll", 0.0) * zero_mask

    dt = grid["dt"]
    amp = float(cfg["terms"]["epw"]["source"].get("noise_amplitude", 1e-10))
    mode_amp_sq = (dt * amp) ** 2 * np.array(grid["low_pass_filter_grid"]) ** 2 * zero_mask

    small = 2.0 * g * dt < 1.0e-12  # undamped modes: plain random walk, variance ~ t
    denom = np.where(small, 1.0, 1.0 - np.exp(-2.0 * g * dt))
    growth = np.where(small, t_ps / dt, (1.0 - np.exp(-2.0 * g * t_ps)) / denom)

    prefactor = 0.25 * grid["dx"] * derived["x_norm"] * derived["e_norm"] ** 2
    return float(prefactor / (grid["nx"] * grid["ny"] ** 2) * np.sum(k_sq * mode_amp_sq * growth))


# --- Time windows: same semantics as osiris_lpi/laser_budget.py:_segment_windows ---


def segment_windows(t: np.ndarray, last_frac: float = 0.25, n_segments: int = 4) -> list[dict]:
    """Equal-time portions of the run whose final portion is the last-``last_frac`` window.

    Same semantics as ``osiris_lpi/laser_budget.py:_segment_windows``: the final
    segment coincides with the headline (last-25%) scalars; the leading span is
    split into ``n_segments - 1`` equal portions. With the defaults this is four
    equal quarters.
    """
    t = np.asarray(t, dtype=float)
    n = max(int(n_segments), 1)
    span = float(t[-1] - t[0]) if t.size else 0.0
    t_lo = t[0] + (1.0 - last_frac) * span if span > 0 else float(t[0])
    edges = list(np.linspace(float(t[0]), t_lo, n)) + [float(t[-1])]
    out: list[dict] = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        is_last = i == n - 1
        mask = (t >= lo) if is_last else ((t >= lo) & (t < hi))
        out.append({"index": i + 1, "n": n, "t_lo": float(lo), "t_hi": float(hi), "is_last": is_last, "mask": mask})
    return out


def _wmean(v: np.ndarray, mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    return float(np.mean(np.asarray(v)[mask]))


# --- Metric assembly from the default-save series ---


def series_metrics(series, cfg: dict) -> dict[str, float]:
    """Build the OSIRIS-comparable scalar metrics from the default-save time series.

    ``series`` is the xarray Dataset built by ``make_series_xarrays`` (keys are the
    save-function outputs, coord ``t (ps)``). Returns a flat metrics dict for MLflow.
    """
    metrics: dict[str, float] = {}
    if "epw_energy" not in series:
        return metrics

    derived = cfg["units"]["derived"]
    w0 = derived["w0"]  # 1/ps
    t_ps = np.asarray(series["t (ps)"].values, dtype=float)
    t_w0 = t_ps * w0

    # EPW growth fit on the OSIRIS-normalized energy vs t in 1/w0, so
    # epw_growth_rate / epw_energy_floor / epw_energy_max are directly comparable
    # to the scan2 rows.
    W = np.asarray(series["epw_energy"].values, dtype=float)
    # noise-seeded runs get the analytic expected-noise floor (see expected_noise_energy);
    # seeded/driven runs keep the OSIRIS measured floor
    floor = None
    if cfg["terms"]["epw"]["source"].get("noise", False) and t_ps.size:
        floor = expected_noise_energy(cfg, float(t_ps[-1]))
    fit = fit_epw_growth(t_w0, W, floor=floor)
    metrics.update(fit)
    if "epw_growth_rate" in fit:
        metrics["epw_growth_rate_per_ps"] = fit["epw_growth_rate"] * w0

    # Electron energy: cumulative EPW dissipation (Landau + collisional), i.e. the
    # energy handed to electrons. epw_dissipation is OSIRIS energy per ps, so the
    # trapezoid over t (ps) gives OSIRIS energy; the incident fluence is
    # I0_osiris * t_w0 with I0_osiris = a0^2/2 (flux in OSIRIS units).
    a0 = derived["E0_source"] * derived["e_norm"]
    I0_osiris = 0.5 * a0**2
    if "epw_dissipation" in series:
        dissip_ps = np.asarray(series["epw_dissipation"].values, dtype=float)
        electron_energy = np.concatenate([[0.0], np.cumsum(0.5 * (dissip_ps[1:] + dissip_ps[:-1]) * np.diff(t_ps))])
        metrics["electron_energy_final"] = float(electron_energy[-1])
        if t_w0[-1] > 0:
            metrics["electron_energy_frac_final"] = float(electron_energy[-1] / (I0_osiris * t_w0[-1]))

    # Laser budget. Flux series are already normalized to the nominal incident flux
    # I0_code. With the pump prescribed (no depletion) only the one-way "naive"
    # estimators are meaningful; with pump depletion on, the net-Poynting R/T/absorbed
    # match osiris_lpi/laser_budget.py exactly (R = 1 - S_left/I0, T = S_right/I0,
    # absorbed = (S_left - S_right)/I0, so R + T + absorbed == 1 identically).
    pump_evolved = bool(cfg["terms"].get("light", {}).get("pump_depletion", False))
    have_flux = all(k in series for k in ("incident_flux", "reflected_flux", "transmitted_flux", "backrefl_flux"))
    if have_flux:
        inc = np.asarray(series["incident_flux"].values, dtype=float)
        refl = np.asarray(series["reflected_flux"].values, dtype=float)
        trans = np.asarray(series["transmitted_flux"].values, dtype=float)
        backrefl = np.asarray(series["backrefl_flux"].values, dtype=float)
        s_left = inc - refl  # F0_left + F1_left, in units of I0_code
        s_right = trans + backrefl

        # With an evolved pump the injector's launched amplitude carries a small grid-
        # dispersion deficit (sin(k0 dx)/sin(k_grid dx), ~2% at 8 cells/wavelength), so
        # the budget is normalized to the *measured* incident flux -- i.e. fractions of
        # the pump that actually entered the box, exactly like OSIRIS normalizes to its
        # own launched flux. With a prescribed pump the physical incident flux is
        # nominal by construction, so normalize by 1.
        inc_ref = 1.0
        if pump_evolved and len(inc) > 1:
            inc_ref = float(np.mean(inc[len(inc) // 2 :]))
            if not np.isfinite(inc_ref) or inc_ref <= 0:
                inc_ref = 1.0

        windows = segment_windows(t_ps, last_frac=0.25, n_segments=4)
        for win in windows:
            suffix = "" if win["is_last"] else f"_seg{win['index']}of{win['n']}"
            r_naive = _wmean(refl, win["mask"])
            t_naive = _wmean(trans, win["mask"])
            if r_naive is None:
                continue
            metrics[f"laser_reflectivity_naive{suffix}"] = r_naive / inc_ref
            metrics[f"laser_transmissivity_naive{suffix}"] = t_naive / inc_ref
            if pump_evolved:
                metrics[f"laser_reflectivity{suffix}"] = 1.0 - _wmean(s_left, win["mask"]) / inc_ref
                metrics[f"laser_transmissivity{suffix}"] = _wmean(s_right, win["mask"]) / inc_ref
                metrics[f"laser_absorbed_frac{suffix}"] = _wmean(s_left - s_right, win["mask"]) / inc_ref
            else:
                # pump is prescribed: the one-way Raman flux is the only measurable
                # reflectivity, and there is no transmission/absorption budget
                metrics[f"laser_reflectivity{suffix}"] = r_naive / inc_ref
            if "epw_dissipation" in series:
                d = _wmean(np.asarray(series["epw_dissipation"].values, dtype=float), win["mask"])
                if d is not None and I0_osiris > 0:
                    metrics[f"laser_absorbed_frac_epw{suffix}"] = d / w0 / I0_osiris

        # injector/prescription health: the measured incident flux should be ~1
        # (skip the first 5% of the run to avoid the fill-in transient)
        metrics["laser_incident_flux_ratio"] = float(np.mean(inc[max(1, len(inc) // 20) :]))

    # --- HPE hot-electron metrics, named to match the OSIRIS scan2 set ---
    if "fhot_50keV" in series:
        hpe = cfg["terms"]["hpe"]
        # one-particle resolution of the hot fraction; "first hot electrons" means a
        # sustained signal of at least a few particles above threshold
        try:
            from adept._lpse2d.core.hpe import tail_fraction

            # the tail cut is one-sided in |vx| (1D) but on the speed (2D), so the
            # fraction of the full Maxwellian it carries differs between them
            f_tail = tail_fraction(float(hpe["v_min"]), 1 if cfg["grid"]["ny"] == 1 else 2)
        except Exception:
            f_tail = 1.0
        min_frac = 3.0 * f_tail / float(hpe["n_particles"])

        for key in ("fhot_50keV", "fhot_100keV"):
            if key not in series:
                continue
            fhot = np.asarray(series[key].values, dtype=float)
            windows = segment_windows(t_ps, last_frac=0.25, n_segments=4)
            metrics[key] = _wmean(fhot, windows[-1]["mask"])
            # onset threshold: a few particles above the *initial* population -- at
            # high Te the loaded Maxwellian tail itself crosses the energy threshold,
            # so onset means a sustained excess over t=0, not any nonzero count
            onset = _sustained_crossing(fhot, max(min_frac, 2.0 * fhot[0]))
            thresh_kev = key.split("_")[1].replace("keV", "")
            if onset is not None:
                # OSIRIS logs onset times in 1/w0 code units
                metrics[f"t_first_hot_e_{thresh_kev}keV"] = float(t_w0[onset])
                metrics[f"t_first_hot_e_{thresh_kev}keV_ps"] = float(t_ps[onset])

        if "hpe_gamma_ratio_kpeak" in series:
            # headline damping-reduction factor: at the dominant band mode (robust);
            # the band-min is also logged but is shot-noise-limited at low n_particles
            # the series is NaN wherever the EPW had no in-band energy yet (pre-onset),
            # so all reductions here must be NaN-aware
            ratio = np.asarray(series["hpe_gamma_ratio_kpeak"].values, dtype=float)
            windows = segment_windows(t_ps, last_frac=0.25, n_segments=4)
            ratio_last = ratio[np.asarray(windows[-1]["mask"], dtype=bool)]
            if np.isfinite(ratio_last).any():
                metrics["hpe_damping_reduction_final"] = float(np.nanmean(ratio_last))
            if np.isfinite(ratio).any():
                metrics["hpe_damping_reduction_min"] = float(np.nanmin(ratio))
        if "hpe_gamma_ratio_min" in series:
            metrics["hpe_damping_reduction_band_min"] = float(
                np.nanmin(np.asarray(series["hpe_gamma_ratio_min"].values, dtype=float))
            )
        if "hpe_mean_energy_keV" in series:
            metrics["hpe_mean_energy_keV_final"] = float(
                np.asarray(series["hpe_mean_energy_keV"].values, dtype=float)[-1]
            )

    return metrics
