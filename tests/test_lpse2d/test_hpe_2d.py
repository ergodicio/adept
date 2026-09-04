"""Tests for the 2-D path of the hybrid particle evolution (HPE) module.

The quasi-1D behaviour is pinned by ``test_hpe.py``; everything here exercises the
parts that only exist when ``ny > 1``:

- the resonance integral becomes a line in ``(vx, vy)``, evaluated as the projection
  of f onto each mode's own direction (``n_angles`` projections spanning ``[0, pi)``)
- particles carry two momentum components and feel ``Ey`` as well as ``Ex``
- loading fills the *speed* tail isotropically rather than the ``|vx|`` tail

The two load-bearing checks are ``test_2d_damping_is_isotropic_and_matches_analytic``
(an isotropic tail must give a rate that depends on ``|k|`` alone and equals the
analytic Landau rate) and ``test_2d_reduces_to_1d_for_axial_modes`` (the 2-D
extraction is a strict generalization of the 1-D one, not a reformulation).
"""

import os
import tempfile

# harmless defaults for standalone invocation; CI/user env vars win
os.environ.setdefault("MPLBACKEND", "Agg")
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{tempfile.mkdtemp(prefix='mlflow-hpe2d-test')}"

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from tests.test_lpse2d.test_hpe import HPE_DEFAULTS, _make_cfg


def _make_cfg_2d(hpe_overrides=None, ny_um="1.0um", cfg_overrides=None):
    """Derived config with a genuinely 2-D box (ny > 1), matching the 1-D fixture's
    x grid so the two can be compared mode by mode."""
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"]["ymax"] = ny_um
    cfg["grid"]["ymin"] = "-" + ny_um
    cfg["terms"]["epw"]["damping"]["landau"] = True
    cfg["terms"]["hpe"] = dict(HPE_DEFAULTS)
    cfg["terms"]["hpe"]["n_angles"] = 32
    if hpe_overrides:
        cfg["terms"]["hpe"].update(hpe_overrides)
    if cfg_overrides:
        for path, val in cfg_overrides.items():
            d = cfg
            keys = path.split(".")
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = val
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    assert cfg["grid"]["ny"] > 1, "2-D fixture collapsed to ny == 1"
    return cfg


def test_2d_config_is_accepted_and_angles_collapse_in_1d():
    """ny > 1 no longer raises, and a quasi-1D box forces a single projection axis."""
    assert _make_cfg_2d()["terms"]["hpe"]["n_angles"] == 32
    assert _make_cfg({"n_angles": 32})["terms"]["hpe"]["n_angles"] == 1


def test_2d_loading_is_isotropic_and_on_the_tail():
    """2-D loading fills the speed tail with an isotropic direction distribution, and
    every projected histogram is separately normalized."""
    from adept._lpse2d.core.hpe import load_particles, resonance_arrays

    n_p = 200000
    cfg = _make_cfg_2d({"n_particles": n_p})
    state = load_particles(cfg)
    arrays = resonance_arrays(cfg)

    assert state["u_e"].shape == (n_p, 2)
    assert state["x_e"].shape == (n_p, 2)
    assert state["epw_hist"].shape == (arrays["n_angles"], cfg["terms"]["hpe"]["nv"])

    # each angle's projection is a normalized 1-D density
    np.testing.assert_allclose(np.sum(state["epw_hist"], axis=1) * arrays["dv"], 1.0, rtol=1e-12)

    c = cfg["units"]["derived"]["c"]
    vte = np.sqrt(cfg["units"]["derived"]["vte_sq"])
    u = state["u_e"]
    v = u / np.sqrt(1.0 + np.sum((u / c) ** 2, axis=1, keepdims=True))
    speed = np.linalg.norm(v, axis=1)
    assert np.all(speed > 2.5 * vte - 1e-9)
    assert np.all(speed < c)

    # isotropy: the direction histogram is flat to within Poisson noise
    ang = np.arctan2(v[:, 1], v[:, 0])
    counts, _ = np.histogram(ang, bins=16, range=(-np.pi, np.pi))
    expected = len(ang) / 16
    assert np.all(np.abs(counts - expected) < 6.0 * np.sqrt(expected)), counts

    # particles fill the box in y
    ly = cfg["grid"]["ymax"] - cfg["grid"]["ymin"]
    assert state["x_e"][:, 1].min() < cfg["grid"]["ymin"] + 0.1 * ly
    assert state["x_e"][:, 1].max() > cfg["grid"]["ymax"] - 0.1 * ly


def test_2d_damping_is_isotropic_and_matches_analytic():
    """The headline 2-D check: with an isotropic loaded tail the extracted rate must
    (a) reproduce the analytic Landau rate over the resonant band and (b) depend only
    on ``|k|``, not on the direction of k -- i.e. the projection/angle machinery adds
    no spurious anisotropy."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    cfg = _make_cfg_2d({"n_particles": 2000000})
    hpe = HybridParticleEvolution(cfg)
    state = load_particles(cfg)

    gamma = np.array(hpe.damping(jnp.array(state["epw_hist"])))
    gamma_analytic = np.array(hpe.gamma_analytic)
    mask = np.array(hpe.mask_res)

    kx = np.array(cfg["grid"]["kx"])
    ky = np.array(cfg["grid"]["ky"])
    KX = kx[:, None] + 0.0 * ky[None, :]
    KY = 0.0 * kx[:, None] + ky[None, :]
    derived = cfg["units"]["derived"]
    kld = np.sqrt(KX**2 + KY**2) * np.sqrt(derived["vte_sq"]) / derived["wp0"]
    band = mask & (kld > 0.25) & (kld < 0.40)
    assert band.sum() > 20, "2-D test box does not resolve the calibration band"

    # (a) unbiased against the analytic rate across the whole band
    ratio = gamma[band] / gamma_analytic[band]
    np.testing.assert_allclose(np.mean(ratio), 1.0, atol=0.05)

    # (b) no angular bias: bin the band by the direction of k and compare the mean
    # ratio per direction. An error in the projection axis or in the sign convention
    # shows up here as a systematic drift with angle.
    theta = np.mod(np.arctan2(KY, KX), np.pi)
    edges = np.linspace(0.0, np.pi, 7)
    tb = np.digitize(theta[band], edges[1:-1])
    per_angle = np.array([ratio[tb == i].mean() for i in range(6) if np.any(tb == i)])
    assert len(per_angle) >= 4, "not enough populated angle bins"
    np.testing.assert_allclose(per_angle, 1.0, atol=0.12)


def test_2d_reduces_to_1d_for_axial_modes():
    """A mode along x in a 2-D box must get the same damping as the same mode in the
    quasi-1D box: the 2-D extraction is a strict generalization, not a reformulation."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    cfg1 = _make_cfg({"n_particles": 2000000})
    cfg2 = _make_cfg_2d({"n_particles": 2000000})
    # the two fixtures share dx/nx, so the kx grids coincide
    np.testing.assert_allclose(np.array(cfg1["grid"]["kx"]), np.array(cfg2["grid"]["kx"]))

    g1 = np.array(HybridParticleEvolution(cfg1).damping(jnp.array(load_particles(cfg1)["epw_hist"])))[:, 0]
    hpe2 = HybridParticleEvolution(cfg2)
    g2 = np.array(hpe2.damping(jnp.array(load_particles(cfg2)["epw_hist"])))[:, 0]  # the ky = 0 column

    mask = np.array(hpe2.mask_res)[:, 0]
    np.testing.assert_array_equal(mask, np.array(HybridParticleEvolution(cfg1).mask_res)[:, 0])

    # restrict to the band where Landau damping is actually resolvable: below
    # k*lambda_D ~ 0.25 the analytic rate underflows (1e-31 at the low-k edge here),
    # the slope at v_phi ~ 13 vte is pure shot noise, and both estimates clamp to
    # exactly 0 -- a 0/0 ratio that says nothing about the extraction
    derived = cfg2["units"]["derived"]
    kld = np.abs(np.array(cfg2["grid"]["kx"])) * np.sqrt(derived["vte_sq"]) / derived["wp0"]
    band = mask & (kld > 0.25) & (kld < 0.40)
    assert band.sum() > 10, "test box does not resolve the comparison band"

    # both are shot-noise-limited estimates of the same analytic curve
    np.testing.assert_allclose(np.mean(g2[band] / g1[band]), 1.0, atol=0.05)


def test_2d_free_streaming_and_transverse_wrap():
    """Zero field in 2-D: both momentum components are untouched and both coordinates
    advance ballistically, with y wrapping periodically."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    cfg = _make_cfg_2d({"n_particles": 1000})
    hpe = HybridParticleEvolution(cfg)

    rng = np.random.default_rng(0)
    x = np.stack(
        [
            rng.uniform(cfg["grid"]["xmin"], cfg["grid"]["xmax"], 1000),
            rng.uniform(cfg["grid"]["ymin"], cfg["grid"]["ymax"], 1000),
        ],
        axis=1,
    )
    u = rng.normal(0.0, 100.0, (1000, 2))
    zero = jnp.zeros((hpe.nx_f, hpe.ny_f), dtype=jnp.complex128)

    x_new, u_new = hpe.push(jnp.array(x), jnp.array(u), (zero, zero), 0.0)

    gamma = np.sqrt(1.0 + np.sum((u / hpe.c) ** 2, axis=1, keepdims=True))
    x_exp = x + hpe.dt * u / gamma
    lx = cfg["grid"]["xmax"] - cfg["grid"]["xmin"]
    ly = cfg["grid"]["ymax"] - cfg["grid"]["ymin"]
    x_exp[:, 0] = np.mod(x_exp[:, 0] - cfg["grid"]["xmin"], lx) + cfg["grid"]["xmin"]
    x_exp[:, 1] = np.mod(x_exp[:, 1] - cfg["grid"]["ymin"], ly) + cfg["grid"]["ymin"]

    np.testing.assert_allclose(np.array(u_new), u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(x_new), x_exp, rtol=0, atol=1e-9)


def test_2d_transverse_force_moves_particles():
    """A pure ky mode must accelerate particles in y and not in x: this pins the Ey
    branch of the gather, which has no quasi-1D counterpart."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    n_test = 256
    cfg = _make_cfg_2d({"n_particles": n_test})
    hpe = HybridParticleEvolution(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    ky = np.array(cfg["grid"]["ky"])

    m = int(np.argmin(np.abs(ky - 2.0 * np.pi / (cfg["grid"]["ymax"] - cfg["grid"]["ymin"]))))
    assert ky[m] != 0.0
    phi_k = np.zeros((nx, ny), dtype=np.complex128)
    phi_k[0, m] = 1.0e3 * nx * ny
    e_env = hpe.refine_e(jnp.array(phi_k))

    # Ex_k = -i kx phi_k is identically zero for a kx = 0 mode
    np.testing.assert_allclose(np.abs(np.array(e_env[0])), 0.0, atol=1e-20)
    assert np.max(np.abs(np.array(e_env[1]))) > 0.0

    rng = np.random.default_rng(1)
    x = np.stack(
        [
            rng.uniform(cfg["grid"]["xmin"], cfg["grid"]["xmax"], n_test),
            rng.uniform(cfg["grid"]["ymin"], cfg["grid"]["ymax"], n_test),
        ],
        axis=1,
    )
    u = np.zeros((n_test, 2))
    _, u_new = hpe.push(jnp.array(x), jnp.array(u), e_env, 0.0)
    u_new = np.array(u_new)
    assert np.max(np.abs(u_new[:, 1])) > 0.0, "no transverse acceleration from a ky mode"
    np.testing.assert_allclose(u_new[:, 0], 0.0, atol=1e-12)


@pytest.mark.slow
def test_2d_srs_smoke_with_hpe():
    """End-to-end in a real 2-D box: the run completes, stays finite, and emits the
    hot-electron series with the per-angle histogram (nt, n_angles, nv) that the save
    and plotting paths only see when ny > 1."""
    from adept import ergoExo

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"]["ymax"] = "0.25um"
    cfg["grid"]["ymin"] = "-0.25um"
    cfg["grid"]["tmax"] = "0.1ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.1ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.05ps"
    # 200k rather than the 1-D smoke test's 20k: this box has ~6600 modes in the
    # resonant band, and at 20k particles ~6% of them read a shot-noise slope that
    # clamps to zero (measured: frac_zero 0.061 -> 0.008 -> 0.000 at 20k -> 200k -> 2M,
    # with the band-mean ratio converging 1.058 -> 1.050 -> 1.002). hpe_gamma_ratio_kpeak
    # reads a single mode, so it inherits that per-mode noise.
    cfg["terms"]["hpe"] = {"active": True, "n_particles": 200000, "nv": 256, "n_angles": 8}
    cfg["terms"]["epw"]["source"]["noise_seed"] = 12345  # see the 1-D smoke test
    cfg["mlflow"]["run"] = "srs-hpe-2d-smoke"

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, _ = exo(modules)

    assert exo.adept_module.cfg["grid"]["ny"] > 1, "smoke config collapsed to quasi-1D"
    series = ppo["series"]
    for key in ("fhot_50keV", "hpe_gamma_ratio_kpeak", "hpe_hist"):
        assert key in series, key
    # one projected distribution per angle, per save time
    assert series["hpe_hist"].values.ndim == 3
    assert series["hpe_hist"].values.shape[1] == 8
    for k in series:
        vals = np.asarray(series[k].values, dtype=float)
        if k == "hpe_gamma_ratio_kpeak":
            # documented sentinel: before the EPW carries any band energy the peak
            # mode is undefined, so the save func emits NaN there on purpose (the
            # metrics layer skips it with nanmin/nanmean). Infinities are still bugs,
            # and once the EPW has grown every sample must be finite.
            assert not np.any(np.isinf(vals)), "infinite hpe_gamma_ratio_kpeak"
            assert np.all(np.isfinite(vals[len(vals) // 2 :])), "NaN after the EPW has grown"
            continue
        assert np.all(np.isfinite(vals)), f"non-finite values in {k}"
    # Damping is applied, finite and positive. Deliberately NOT a tight threshold:
    # kpeak reads the single band mode currently holding the most EPW energy, and energy
    # preferentially accumulates where damping is lowest, so the statistic sits below 1
    # by construction -- more so in 2D, where the band offers ~10x as many modes to
    # select from. What a smoke test can assert is that the rate is finite, positive and
    # not absurd; the extraction's accuracy is pinned by the t = 0 tests above. The
    # `> 0` is the load-bearing part: it catches the shot-noise clamp-to-zero that
    # terms.hpe.hist_smooth exists to prevent.
    ratio = np.asarray(series["hpe_gamma_ratio_kpeak"].values, dtype=float)
    tail = ratio[-10:]
    assert np.all(np.isfinite(tail)), tail
    assert np.all(tail > 0.0), f"damping collapsed to zero at the peak mode: {tail}"
    assert np.median(tail) < 3.0, tail


def test_2d_projection_matches_direct_line_integral():
    """The angle-interpolated ``dP/dv`` the extraction uses must agree with a directly
    binned projection along the mode's own k direction -- the one place where the
    ``n_angles`` discretization could silently bias the rate."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    # hist_smooth off: this test compares against an unsmoothed reference slope, so
    # it isolates the projection geometry rather than the filter
    cfg = _make_cfg_2d({"n_particles": 1000000, "n_angles": 64, "hist_smooth": 0})
    hpe = HybridParticleEvolution(cfg)
    state = load_particles(cfg)

    c = cfg["units"]["derived"]["c"]
    u = state["u_e"]
    v = u / np.sqrt(1.0 + np.sum((u / c) ** 2, axis=1, keepdims=True))

    gamma = np.array(hpe.damping(jnp.array(state["epw_hist"])))
    kx = np.array(cfg["grid"]["kx"])
    ky = np.array(cfg["grid"]["ky"])
    mask = np.array(hpe.mask_res)
    v_centers, dv = np.array(hpe.v_centers), hpe.dv
    edges = np.concatenate([v_centers - 0.5 * dv, [v_centers[-1] + 0.5 * dv]])
    f_tail = hpe.f_tail_frac

    # a handful of genuinely off-axis resonant modes
    idx = [(i, j) for i in range(len(kx)) for j in range(len(ky)) if mask[i, j] and kx[i] != 0 and ky[j] != 0]
    assert len(idx) > 4
    rng = np.random.default_rng(0)
    for i, j in [idx[t] for t in rng.choice(len(idx), size=5, replace=False)]:
        kvec = np.array([kx[i], ky[j]])
        k_mag = np.linalg.norm(kvec)
        khat = kvec / k_mag
        # direct: bin v.khat, differentiate, read at the mode's own v_phi
        hist, _ = np.histogram(v @ khat, bins=edges)
        hist = hist / (len(v) * dv)
        dfdv = np.gradient(hist, dv) * f_tail
        wp0, vte_sq = cfg["units"]["derived"]["wp0"], cfg["units"]["derived"]["vte_sq"]
        v_phi = np.sqrt(wp0**2 + 3.0 * k_mag**2 * vte_sq) / k_mag
        direct = -0.5 * np.pi * wp0**3 / k_mag**2 * np.interp(v_phi, v_centers, dfdv)
        direct *= float(hpe.calibration[i, j])
        np.testing.assert_allclose(gamma[i, j], direct, rtol=0.15)


def test_2d_transverse_thermalization_tracks_crossings():
    """``y_thermal_frac`` re-thermalizes a fraction of transverse *crossings* (Follett's
    finite-plasma proxy), not a fraction of the population per step.

    Two things are easy to get wrong here and both are pinned: the crossing count has to
    survive the substep loop (the wrap happens inside it), and the thermalization must
    still run on a box with periodic x, where there are no x walls to re-inject from.
    """
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    n_test = 4000
    cfg = _make_cfg_2d({"n_particles": n_test, "y_thermal_frac": 1.0})
    hpe = HybridParticleEvolution(cfg)
    assert hpe.periodic, "fixture is meant to have periodic x, which used to skip this path"

    rng = np.random.default_rng(0)
    x = np.stack(
        [
            rng.uniform(cfg["grid"]["xmin"], cfg["grid"]["xmax"], n_test),
            np.full(n_test, 0.5 * (cfg["grid"]["ymin"] + cfg["grid"]["ymax"])),
        ],
        axis=1,
    )
    zero = jnp.zeros((hpe.nx_f, hpe.ny_f), dtype=jnp.complex128)
    ly = cfg["grid"]["ymax"] - cfg["grid"]["ymin"]

    # slow in y: nobody reaches the transverse boundary within one field step
    u_slow = np.zeros((n_test, 2))
    u_slow[:, 1] = 0.01 * ly / hpe.dt
    _, u_out = hpe.push(jnp.array(x), jnp.array(u_slow), (zero, zero), 0.0)
    np.testing.assert_allclose(np.array(u_out), u_slow, rtol=0, atol=1e-12)

    # fast in y: everyone crosses, and with frac = 1 everyone is re-thermalized onto
    # the tail with an isotropic direction
    u_fast = np.zeros((n_test, 2))
    u_fast[:, 1] = 5.0 * ly / hpe.dt
    _, u_out = hpe.push(jnp.array(x), jnp.array(u_fast), (zero, zero), 0.0)
    u_out = np.array(u_out)
    assert not np.allclose(u_out, u_fast), "crossings were not thermalized"

    c, vte = hpe.c, hpe.vte
    v = u_out / np.sqrt(1.0 + np.sum((u_out / c) ** 2, axis=1, keepdims=True))
    speed = np.linalg.norm(v, axis=1)
    assert np.all(speed > 2.5 * vte - 1e-9), "thermalized particles fell below the tail cut"
    assert np.all(speed < c)
    # isotropic: both signs of vy are represented, so this is not just a reflection
    frac_up = np.mean(v[:, 1] > 0)
    assert 0.4 < frac_up < 0.6, frac_up

    # frac = 0 leaves identical crossings untouched
    hpe0 = HybridParticleEvolution(_make_cfg_2d({"n_particles": n_test, "y_thermal_frac": 0.0}))
    _, u_out0 = hpe0.push(jnp.array(x), jnp.array(u_fast), (zero, zero), 0.0)
    np.testing.assert_allclose(np.array(u_out0), u_fast, rtol=0, atol=1e-12)


@pytest.mark.parametrize("ndim", [1, 2])
def test_thermalizing_x_wall_reinjects_on_the_tail(ndim):
    """Absorbing-x boxes re-inject escaping particles at the wall with an inward,
    flux-weighted tail speed. Covers the shared boundary code for both geometries --
    the other fast fixtures use periodic x, which returns before any of this runs.
    """
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    n_test = 4000
    over = {"n_particles": n_test}
    bc = {"terms.epw.boundary.x": "absorbing"}
    cfg = _make_cfg(over, bc) if ndim == 1 else _make_cfg_2d(over, cfg_overrides=bc)
    hpe = HybridParticleEvolution(cfg)
    assert not hpe.periodic, "fixture did not pick up the absorbing x boundary"
    assert hpe.ndim == ndim

    # half the particles parked beyond each wall, all with zero momentum so that only
    # the boundary can put anything back
    x = np.zeros((n_test, ndim))
    x[: n_test // 2, 0] = cfg["grid"]["xmin"] - 1.0
    x[n_test // 2 :, 0] = cfg["grid"]["xmax"] + 1.0
    if ndim == 2:
        x[:, 1] = 0.5 * (cfg["grid"]["ymin"] + cfg["grid"]["ymax"])
    u = np.zeros((n_test, ndim))

    x_new, u_new = hpe._apply_boundaries(jnp.array(x), jnp.array(u), 0.0)
    x_new, u_new = np.array(x_new), np.array(u_new)

    # everyone is put back exactly on the wall they left through
    np.testing.assert_allclose(x_new[: n_test // 2, 0], cfg["grid"]["xmin"])
    np.testing.assert_allclose(x_new[n_test // 2 :, 0], cfg["grid"]["xmax"])

    # ... moving inward: +x off the left wall, -x off the right wall
    assert np.all(u_new[: n_test // 2, 0] > 0.0)
    assert np.all(u_new[n_test // 2 :, 0] < 0.0)

    # ... and on the tail, so the loaded distribution stays stationary
    c, vte = hpe.c, hpe.vte
    v = u_new / np.sqrt(1.0 + np.sum((u_new / c) ** 2, axis=1, keepdims=True))
    speed = np.linalg.norm(v, axis=1)
    assert np.all(speed > 2.5 * vte - 1e-9), speed.min() / vte
    assert np.all(speed < c)

    if ndim == 2:
        # cosine-weighted emission about the inward normal: the transverse component is
        # unbiased and genuinely spread, not a pure normal-incidence beam
        assert 0.4 < np.mean(v[:, 1] > 0) < 0.6
        assert np.max(np.abs(v[:, 1])) > 0.2 * np.max(np.abs(v[:, 0]))


def test_hist_smooth_removes_clamped_modes_without_bias():
    """`hist_smooth` exists because the rate is read from df/dv at a single velocity bin
    per mode: with finite particles some modes draw a slope steep enough that the
    gamma >= 0 clamp sends them to exactly zero, and a mode pinned at zero is *undamped*,
    so it grows relative to the band rather than the error averaging out. A 2-D box
    carries ~10x the band modes of a quasi-1D one, so it meets this far more often.

    Two things must hold: smoothing removes the clamped modes, and it does not bias the
    rate -- the per-k calibration is derived through the same filter, so any linear
    filter divides back out.
    """
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    n_p = 100000
    frac_zero, means = [], []
    for smooth in (0, 2):
        cfg = _make_cfg_2d({"n_particles": n_p, "n_angles": 16, "hist_smooth": smooth})
        hpe = HybridParticleEvolution(cfg)
        gamma = np.array(hpe.damping(jnp.array(load_particles(cfg)["epw_hist"])))
        ga = np.array(hpe.gamma_analytic)
        band = np.array(hpe.mask_res) & (ga > 1.0e-4 * cfg["units"]["derived"]["wp0"])
        assert band.sum() > 100
        r = gamma[band] / ga[band]
        frac_zero.append(float((r == 0).mean()))
        means.append(float(r.mean()))

    assert frac_zero[0] > 0.0, "fixture does not exhibit the clamping this guards against"
    assert frac_zero[1] < 0.6 * frac_zero[0], f"smoothing did not reduce clamping: {frac_zero}"
    # Unbiased: smoothing must not move the band mean away from the analytic rate. At
    # this particle count the *unsmoothed* mean is itself several % high from shot noise
    # (clamped zeros and noisy outliers), so the meaningful statement is that filtering
    # does not make it worse -- in practice it improves it, which is the variance
    # reduction doing its job.
    np.testing.assert_allclose(means, 1.0, atol=0.15)
    assert abs(means[1] - 1.0) <= abs(means[0] - 1.0) + 0.02, f"smoothing biased the rate: {means}"


def test_hist_smooth_defaults_on_in_2d_and_off_in_1d():
    """Quasi-1D keeps the original (unsmoothed) estimator so existing results are
    reproducible; 2-D turns it on by default, and an explicit setting always wins."""
    assert _make_cfg({})["terms"]["hpe"]["hist_smooth"] == 0
    assert _make_cfg_2d({})["terms"]["hpe"]["hist_smooth"] == 2
    assert _make_cfg_2d({"hist_smooth": 0})["terms"]["hpe"]["hist_smooth"] == 0
    assert _make_cfg_2d({"hist_smooth": 3})["terms"]["hpe"]["hist_smooth"] == 3
