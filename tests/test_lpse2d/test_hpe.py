"""Tests for the hybrid particle evolution (HPE) module (Follett et al. 2017).

Milestone map (docs/dev/lpse2d-hpe-plan.md):

- M0: free streaming, trapped-particle bounce frequency + carrier-sign resonance
- M1: histogram -> gamma_L calibration against the analytic Landau rate
- M3a: linear closure -- a small-amplitude EPW damps at the analytic rate with the
  feedback loop active (slow)
- M3b (qualitative): O'Neil flattening -- a large frozen wave digs a plateau and the
  extracted damping collapses at the resonant mode (slow)
- end-to-end: short SRS run with HPE on exercises the full save/metrics path (slow)
"""

import os
import tempfile

# harmless defaults for standalone invocation; CI/user env vars win
os.environ.setdefault("MPLBACKEND", "Agg")
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{tempfile.mkdtemp(prefix='mlflow-hpe-test')}"

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

HPE_DEFAULTS = {
    "active": True,
    "n_particles": 100000,
    "seed": 42,
}


def _make_cfg(hpe_overrides=None, cfg_overrides=None):
    """Derived config for unit tests (no mlflow / ergoExo): quasi-1D uniform periodic box."""
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)

    # quasi-1D: single transverse cell
    cfg["grid"]["ymax"] = "0.02um"
    cfg["grid"]["ymin"] = "-0.02um"
    cfg["terms"]["epw"]["damping"]["landau"] = True
    cfg["terms"]["hpe"] = dict(HPE_DEFAULTS)
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
    return cfg


def _make_cfg_2d(hpe_overrides=None, cfg_overrides=None):
    """Small periodic 2-D box using the same plasma parameters as the 1-D tests."""
    overrides = {"n_particles": 20000, "n_angles": 16, "gather_refine": 2}
    if hpe_overrides:
        overrides.update(hpe_overrides)
    config = {
        "grid.ymin": "-0.4um",
        "grid.ymax": "0.4um",
    }
    if cfg_overrides:
        config.update(cfg_overrides)
    return _make_cfg(
        overrides,
        config,
    )


def _single_mode_phi_k(cfg, k_target, e_amp):
    """phi_k with one +k mode such that ex_envelope(x) = e_amp * exp(i k x).

    Returns (phi_k (nx, ny) complex, k actual grid wavenumber)."""
    kx = np.array(cfg["grid"]["kx"])
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    m = int(np.argmin(np.abs(kx - k_target)))
    k = kx[m]
    phi_k = np.zeros((nx, ny), dtype=np.complex128)
    # ex_env = ifft2(-i kx phi_k) => a single mode m gives ex = -i k phi_k[m] e^{ikx}/(nx*ny)
    phi_k[m, 0] = 1j * e_amp * nx * ny / k
    return phi_k, k


# ------------------------------------------------------------------- M0 tests --


def test_free_streaming():
    """Zero field: u unchanged, x advances by exactly dt * u / gamma (periodic wrap)."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    cfg = _make_cfg({"n_particles": 1000})
    hpe = HybridParticleEvolution(cfg)

    rng = np.random.default_rng(0)
    x = jnp.array(rng.uniform(cfg["grid"]["xmin"], cfg["grid"]["xmax"], 1000))
    u = jnp.array(rng.normal(0.0, 100.0, 1000))
    ex_env = jnp.zeros(hpe.nx_f, dtype=jnp.complex128)

    x_new, u_new, *_ = hpe.push(x, u, ex_env, 0.0)

    gamma = np.sqrt(1.0 + (np.array(u) / hpe.c) ** 2)
    x_expected = np.array(x) + hpe.dt * np.array(u) / gamma
    L = cfg["grid"]["xmax"] - cfg["grid"]["xmin"]
    x_expected = np.mod(x_expected - cfg["grid"]["xmin"], L) + cfg["grid"]["xmin"]

    np.testing.assert_allclose(np.array(u_new), np.array(u), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(x_new), x_expected, rtol=0, atol=1e-9)


def test_2d_free_streaming_and_both_field_components():
    """The 2D2V pusher advances both coordinates and gathers the full grad(phi)."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    cfg = _make_cfg_2d({"n_particles": 256})
    hpe = HybridParticleEvolution(cfg)
    rng = np.random.default_rng(7402)
    x = jnp.asarray(rng.uniform(cfg["grid"]["xmin"], cfg["grid"]["xmax"], hpe.n_p))
    y = jnp.asarray(rng.uniform(cfg["grid"]["ymin"], cfg["grid"]["ymax"], hpe.n_p))
    u = jnp.asarray(rng.normal(0.0, 100.0, (hpe.n_p, 2)))
    zero_field = jnp.zeros((hpe.nx_f, hpe.ny_f, 2), dtype=jnp.complex128)

    x_new, y_new, u_new, *_ = hpe.push_2d(x, y, u, zero_field, 0.0)
    gamma = np.sqrt(1.0 + np.sum((np.asarray(u) / hpe.c) ** 2, axis=-1))
    x_expected = np.mod(np.asarray(x) + hpe.dt * np.asarray(u)[:, 0] / gamma - hpe.xmin, hpe.Lx) + hpe.xmin
    y_expected = np.mod(np.asarray(y) + hpe.dt * np.asarray(u)[:, 1] / gamma - hpe.ymin, hpe.Ly) + hpe.ymin
    np.testing.assert_allclose(u_new, u, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(x_new, x_expected, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(y_new, y_expected, rtol=0.0, atol=1.0e-9)

    # A single oblique potential mode has Ey/Ex = ky/kx everywhere. This pins the
    # transverse spectral reconstruction and the bilinear gather convention.
    phi_k = np.zeros((hpe.nx, hpe.ny), dtype=np.complex128)
    mx, my = 3, 2
    phi_k[mx, my] = 1.0e-4 * hpe.nx * hpe.ny
    e_env = hpe.refine_e(jnp.asarray(phi_k))
    acceleration = np.asarray(hpe._accel_2d(x, y, e_env, 0.0))
    k_ratio = float(np.asarray(hpe.ky)[my] / np.asarray(hpe.kx)[mx])
    active = np.abs(acceleration[:, 0]) > 1.0e-12 * np.max(np.abs(acceleration[:, 0]))
    np.testing.assert_allclose(acceleration[active, 1], k_ratio * acceleration[active, 0], rtol=2.0e-6, atol=1.0e-8)


def test_2d_wall_reinjection_is_flux_weighted():
    """Incoming wall particles follow the truncated Maxwell-flux radial/angular law."""
    import jax
    from scipy import stats

    from adept._lpse2d.core.hpe import HybridParticleEvolution

    n_particles = 100000
    cfg = _make_cfg_2d(
        {"n_particles": n_particles},
        {
            "terms.epw.boundary.x": "absorbing",
            "terms.epw.boundary.y": "absorbing",
        },
    )
    hpe = HybridParticleEvolution(cfg)
    normal, tangent = hpe._sample_flux_weighted_tail(jax.random.PRNGKey(7403))
    normal, tangent = np.asarray(normal), np.asarray(tangent)
    speed = np.sqrt(normal**2 + tangent**2)

    assert np.all(normal >= 0.0)
    assert np.all(speed >= hpe.v_min)
    assert np.all(speed <= 0.99 * hpe.c)

    # Radial probability integral transform for the Maxwell speed law truncated
    # to [v_min, 0.99c] must be uniform.
    cdf_min = stats.maxwell.cdf(hpe.v_min, scale=hpe.vte)
    cdf_max = stats.maxwell.cdf(0.99 * hpe.c, scale=hpe.vte)
    radial_quantile = (stats.maxwell.cdf(speed, scale=hpe.vte) - cdf_min) / (cdf_max - cdf_min)
    np.testing.assert_allclose(np.mean(radial_quantile), 0.5, atol=4.0e-3)
    np.testing.assert_allclose(np.var(radial_quantile), 1.0 / 12.0, atol=2.0e-3)

    # p(theta)=cos(theta)/2 is equivalent to sin(theta) uniform on [-1,1].
    sin_theta = tangent / speed
    np.testing.assert_allclose(np.mean(sin_theta), 0.0, atol=5.0e-3)
    np.testing.assert_allclose(np.var(sin_theta), 1.0 / 3.0, atol=4.0e-3)

    # The boundary mapping rotates the sampled wall frame correctly for all four
    # faces. Each quarter of the ensemble exits through a different wall.
    quarter = n_particles // 4
    x = np.full(n_particles, 0.5 * (hpe.xmin + hpe.xmax))
    y = np.full(n_particles, 0.5 * (hpe.ymin + hpe.ymax))
    x[:quarter] = hpe.xmin - hpe.dx
    x[quarter : 2 * quarter] = hpe.xmax + hpe.dx
    y[2 * quarter : 3 * quarter] = hpe.ymin - hpe.dy
    y[3 * quarter :] = hpe.ymax + hpe.dy
    u = jnp.zeros((n_particles, 2))
    x_new, y_new, u_new, *_ = hpe._apply_boundaries_2d(jnp.asarray(x), jnp.asarray(y), u, 0.0)
    gamma_rel = np.sqrt(1.0 + np.sum((np.asarray(u_new) / hpe.c) ** 2, axis=-1))
    velocity = np.asarray(u_new) / gamma_rel[:, None]
    np.testing.assert_allclose(x_new[:quarter], hpe.xmin, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(x_new[quarter : 2 * quarter], hpe.xmax, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(y_new[2 * quarter : 3 * quarter], hpe.ymin, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(y_new[3 * quarter :], hpe.ymax, rtol=0.0, atol=0.0)
    assert np.all(velocity[:quarter, 0] > 0.0)
    assert np.all(velocity[quarter : 2 * quarter, 0] < 0.0)
    assert np.all(velocity[2 * quarter : 3 * quarter, 1] > 0.0)
    assert np.all(velocity[3 * quarter :, 1] < 0.0)


def test_bounce_frequency_and_carrier_sign():
    """A monochromatic +k wave traps particles at v = +wp0/k with the relativistic
    bounce frequency w_b = sqrt(e k E / (me gamma^3)); particles at v = -wp0/k are
    non-resonant. This pins the carrier sign convention AND the push normalization."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    n_test = 64
    cfg = _make_cfg({"n_particles": n_test})
    hpe = HybridParticleEvolution(cfg)
    derived = cfg["units"]["derived"]
    wp0 = derived["wp0"]

    # kld = 0.35 mode; field amplitude chosen for a ~0.5 ps bounce period
    vte = np.sqrt(derived["vte_sq"])
    k_target = 0.35 * wp0 / vte
    v_phi_guess = wp0 / k_target
    gamma_phi = 1.0 / np.sqrt(1.0 - (v_phi_guess / derived["c"]) ** 2)
    w_b_target = 12.5  # 1/ps
    e_over_m = derived["e"] / derived["me"]

    phi_k, k = _single_mode_phi_k(cfg, k_target, 1.0)
    v_phi = wp0 / k
    gamma_phi = 1.0 / np.sqrt(1.0 - (v_phi / derived["c"]) ** 2)
    e_amp = w_b_target**2 * gamma_phi**3 / (e_over_m * k)
    phi_k, k = _single_mode_phi_k(cfg, k_target, e_amp)
    ex_env = hpe.refine_ex(jnp.array(phi_k))

    dt = cfg["grid"]["dt"]
    n_steps = 500  # 3 ps at 6 fs

    def run(v0):
        u0 = v0 / np.sqrt(1.0 - (v0 / derived["c"]) ** 2)
        x = jnp.array(np.linspace(cfg["grid"]["xmin"], cfg["grid"]["xmin"] + 2.0 * np.pi / k, n_test, endpoint=False))
        u = jnp.full((n_test,), u0)
        us = np.zeros((n_steps, n_test))
        import jax

        push = jax.jit(hpe.push)
        for i in range(n_steps):
            x, u, *_ = push(x, u, ex_env, i * dt)
            us[i] = np.array(u)
        return us

    u_phi = v_phi / np.sqrt(1.0 - (v_phi / derived["c"]) ** 2)
    us_res = run(v_phi)
    us_ctr = run(-v_phi)

    # resonant particles oscillate around u_phi with amplitudes up to the trapping
    # width; counter-propagating particles barely feel the wave
    du_res = us_res - u_phi
    du_ctr = us_ctr + u_phi
    rms_res = np.sqrt(np.mean(du_res**2))
    rms_ctr = np.sqrt(np.mean(du_ctr**2))
    assert rms_res > 5.0 * rms_ctr, f"no resonance at +v_phi: rms {rms_res:.3g} vs counter {rms_ctr:.3g}"

    # bounce frequency: dominant FFT peak of du(t) per particle. Use moderately
    # trapped particles (small pendulum amplitude -> w ~ w_b within ~2%); a Hann
    # window plus 8x zero padding gives ~2% frequency resolution over the record.
    amps = np.max(np.abs(du_res), axis=0)
    sel = (amps > 0.1 * amps.max()) & (amps < 0.35 * amps.max())
    assert sel.sum() >= 5, "not enough moderately trapped particles for the fit"
    window = np.hanning(n_steps)
    n_pad = 8 * n_steps
    freqs = 2.0 * np.pi * np.fft.rfftfreq(n_pad, d=dt)
    w_measured = []
    for j in np.where(sel)[0]:
        sig = (du_res[:, j] - np.mean(du_res[:, j])) * window
        spec = np.abs(np.fft.rfft(sig, n=n_pad))
        w_measured.append(freqs[np.argmax(spec)])
    w_measured = np.median(w_measured)

    w_b = np.sqrt(e_over_m * k * e_amp / gamma_phi**3)
    np.testing.assert_allclose(w_measured, w_b, rtol=0.08)


# ------------------------------------------------------------------- M1 tests --


def test_loading_and_histogram_normalization():
    from adept._lpse2d.core.hpe import load_particles, resonance_arrays

    cfg = _make_cfg({"n_particles": 200000})
    state = load_particles(cfg)
    arrays = resonance_arrays(cfg)

    # histogram is a normalized density on the tail
    np.testing.assert_allclose(np.sum(state["epw_hist"]) * arrays["dv"], 1.0, rtol=1e-12)

    # all loaded particles are beyond the cutoff and inside the box
    vte = np.sqrt(cfg["units"]["derived"]["vte_sq"])
    c = cfg["units"]["derived"]["c"]
    v = state["u_e"] / np.sqrt(1.0 + (state["u_e"] / c) ** 2)
    assert np.all(np.abs(v) > 2.5 * vte - 1e-9)
    assert np.all(np.abs(v) < c)
    assert np.all(state["x_e"] >= cfg["grid"]["xmin"] - cfg["grid"]["dx"])
    assert np.all(state["x_e"] <= cfg["grid"]["xmax"] + cfg["grid"]["dx"])

    # initial gamma_L is exactly the analytic array
    np.testing.assert_allclose(state["gamma_L"], np.array(arrays["gamma_analytic"]), rtol=1e-12)


def test_2d_loading_is_one_box_averaged_ensemble():
    from adept._lpse2d.core.hpe import load_particles, resonance_arrays

    cfg = _make_cfg_2d({"n_particles": 50000})
    state = load_particles(cfg)
    arrays = resonance_arrays(cfg)

    assert state["x_e"].shape == (50000,)
    assert state["y_e"].shape == (50000,)
    assert state["u_e"].shape == (50000, 2)
    assert state["epw_hist"].shape == (16, cfg["terms"]["hpe"]["nv"])
    np.testing.assert_allclose(np.sum(state["epw_hist"], axis=1) * arrays["dv"], 1.0, rtol=2.0e-4)

    c = cfg["units"]["derived"]["c"]
    vte = np.sqrt(cfg["units"]["derived"]["vte_sq"])
    gamma = np.sqrt(1.0 + np.sum((state["u_e"] / c) ** 2, axis=-1))
    speed = np.linalg.norm(state["u_e"] / gamma[:, None], axis=-1)
    assert np.all(speed > cfg["terms"]["hpe"]["v_min"] * vte - 1.0e-9)
    assert np.all(speed < c)
    assert arrays["mask_res"].shape == (cfg["grid"]["nx"], cfg["grid"]["ny"])
    assert np.any(arrays["mask_res"] & (np.abs(np.asarray(cfg["grid"]["ky"]))[None, :] > 0.0))


def test_2d_expected_tail_calibrates_every_oblique_mode():
    """The global directional marginals reproduce analytic damping over the 2-D band."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, resonance_arrays

    cfg = _make_cfg_2d({"n_particles": 64})
    hpe = HybridParticleEvolution(cfg)
    arrays = resonance_arrays(cfg)
    expected = jnp.broadcast_to(jnp.asarray(arrays["f0_expected"]), (hpe.n_angles, hpe.nv))
    gamma = np.asarray(hpe.damping(expected))
    analytic = np.asarray(hpe.gamma_analytic)
    mask = np.asarray(hpe.mask_res)
    assert np.any(mask)
    np.testing.assert_allclose(gamma[mask], analytic[mask], rtol=2.0e-6, atol=1.0e-10)


def test_2d_damping_is_directional_not_broadcast_over_ky():
    """Flattening one global projection only undamps modes in that direction."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, resonance_arrays

    cfg = _make_cfg_2d({"n_particles": 64})
    hpe = HybridParticleEvolution(cfg)
    arrays = resonance_arrays(cfg)
    expected = jnp.broadcast_to(jnp.asarray(arrays["f0_expected"]), (hpe.n_angles, hpe.nv))
    flat = jnp.full((hpe.nv,), 1.0 / (hpe.nv * hpe.dv))
    gamma = np.asarray(hpe.damping(expected.at[0].set(flat)))
    analytic = np.asarray(hpe.gamma_analytic)
    mask = np.asarray(hpe.mask_res)
    kx, ky = np.asarray(cfg["grid"]["kx"]), np.asarray(cfg["grid"]["ky"])

    plus_x = mask[:, 0] & (kx > 0.0)
    plus_y = mask[0, :] & (ky > 0.0)
    assert np.any(plus_x) and np.any(plus_y)
    assert np.all(gamma[plus_x, 0] < 1.0e-10 * analytic[plus_x, 0])
    np.testing.assert_allclose(gamma[0, plus_y], analytic[0, plus_y], rtol=2.0e-6, atol=1.0e-10)


def test_damping_calibration():
    """M1: the sampled Maxwellian-tail histogram must reproduce the analytic Landau
    rate over the resonant band (the per-k calibration removes discretization bias;
    what is being tested is the sampling + the end-to-end extraction)."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    cfg = _make_cfg({"n_particles": 4000000})
    hpe = HybridParticleEvolution(cfg)
    state = load_particles(cfg)

    gamma = np.array(hpe.damping(jnp.array(state["epw_hist"])))[:, 0]
    gamma_analytic = np.array(hpe.gamma_analytic)[:, 0]

    derived = cfg["units"]["derived"]
    kld = np.abs(np.array(cfg["grid"]["kx"])) * np.sqrt(derived["vte_sq"]) / derived["wp0"]
    band = np.array(hpe.mask_res) & (kld > 0.25) & (kld < 0.40)
    assert band.sum() > 10, "test box does not resolve the calibration band"

    # pointwise: limited by shot noise on the histogram derivative (~10% at the
    # low-k edge even at 4M particles); band mean: tests the normalization itself
    np.testing.assert_allclose(gamma[band], gamma_analytic[band], rtol=0.2)
    np.testing.assert_allclose(np.mean(gamma[band] / gamma_analytic[band]), 1.0, atol=0.03)

    # the per-k calibration itself should be a small correction
    calib_band = np.array(hpe.calibration)[band]
    assert np.all((calib_band > 0.7) & (calib_band < 1.4)), f"calibration out of range: {calib_band}"


def test_blend_and_clamp():
    """Modes with v_phi below the tail cutoff keep the analytic rate; a rigged
    histogram with a positive slope at v_phi must clamp to zero, not go negative."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    cfg = _make_cfg({"n_particles": 1000})
    hpe = HybridParticleEvolution(cfg)

    # inverted-slope histogram: f rising with |v| -> raw gamma < 0 everywhere resonant
    v = np.array(hpe.v_centers)
    fake = np.abs(v) / np.sum(np.abs(v) * hpe.dv)
    gamma = np.array(hpe.damping(jnp.array(fake)))[:, 0]

    mask = np.array(hpe.mask_res)
    gamma_analytic = np.array(hpe.gamma_analytic)[:, 0]
    np.testing.assert_allclose(gamma[~mask], gamma_analytic[~mask], rtol=1e-12)
    assert np.all(gamma[mask] >= 0.0)
    assert np.all(gamma[mask] < 1e-12)  # clamped, not analytic


# ------------------------------------------------------- slow / full-loop tests --


def _run_exo(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)
    return exo, sol, ppo


@pytest.mark.slow
def test_srs_smoke_with_hpe():
    """End-to-end: the SRS growth box with HPE active runs, stays finite, and emits
    the hot-electron series/metrics."""
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"]["ymax"] = "0.02um"
    cfg["grid"]["ymin"] = "-0.02um"
    cfg["grid"]["tmax"] = "0.1ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.1ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.05ps"
    cfg["terms"]["hpe"] = {"active": True, "n_particles": 20000, "nv": 256}
    cfg["mlflow"]["run"] = "srs-hpe-smoke"

    exo, sol, ppo = _run_exo(cfg)

    series = ppo["series"]
    assert "fhot_50keV" in series
    assert "hpe_gamma_ratio_kpeak" in series
    assert "hpe_hist" in series
    for k in series:
        assert np.all(np.isfinite(series[k].values)), f"non-finite values in {k}"
    # nothing exciting happens in 0.1 ps: damping at the dominant band mode should
    # still be ~analytic (the band-min is shot-noise-limited at 20k particles)
    ratio = np.asarray(series["hpe_gamma_ratio_kpeak"].values, dtype=float)
    assert ratio[-1] > 0.5


@pytest.mark.slow
def test_linear_closure():
    """M3a: with HPE feedback ON, a small-amplitude EPW still damps at the analytic
    rate (the loop reproduces linear theory when there is no trapping).

    The amplitude must be genuinely linear (bounce frequency << gamma_L: a0 ~ 1e-12
    in code units -- 1e-6 traps the whole tail!) and the driver shutoff (tr) must be
    faster than the Landau decay or the tanh tail masquerades as the measured rate.
    The residual tolerance is set by histogram shot noise at the resonant mode
    (~7% at 1M particles), which enters the applied rate as a near-constant bias."""
    cfg = _make_cfg({})
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        run_cfg = yaml.safe_load(fi)
    run_cfg["grid"]["ymax"] = "0.02um"
    run_cfg["grid"]["ymin"] = "-0.02um"
    run_cfg["grid"]["dt"] = "6fs"
    run_cfg["grid"]["tmax"] = "0.45ps"
    run_cfg["save"]["fields"]["t"]["tmax"] = "0.45ps"
    run_cfg["save"]["fields"]["t"]["dt"] = "0.15ps"
    run_cfg["terms"]["epw"]["damping"]["landau"] = True
    run_cfg["terms"]["hpe"] = {"active": True, "n_particles": 1000000, "substep_courant": 0.1}
    run_cfg["mlflow"]["run"] = "hpe-linear-closure"

    derived = cfg["units"]["derived"]
    vte = np.sqrt(derived["vte_sq"])
    kld = 0.28
    k0 = kld * derived["wp0"] / vte
    # snap to the k grid: an off-grid driver leaks into many weakly damped sideband
    # modes whose slow decay contaminates the fit
    kx_grid = np.array(cfg["grid"]["kx"])
    k0 = float(kx_grid[np.argmin(np.abs(kx_grid - k0))])
    run_cfg["drivers"]["E2"]["k0"] = float(k0)
    # the driver populates the -k0 envelope mode, which free-evolves as
    # exp(-i delta_omega t); the driver envelope rotates as exp(+i w0 t), so
    # resonant driving needs w0 = -delta_omega (an off-resonant drive leaves the
    # fit window contaminated by the forced response during the shutoff tail)
    run_cfg["drivers"]["E2"]["w0"] = float(-1.5 * k0**2 * derived["vte_sq"] / derived["wp0"])
    run_cfg["drivers"]["E2"]["a0"] = 1.0e-13  # resonant response ~ k^2 a0 / gamma: keeps w_bounce << gamma
    run_cfg["drivers"]["E2"]["envelope"]["tc"] = "150fs"
    run_cfg["drivers"]["E2"]["envelope"]["tw"] = "300fs"
    run_cfg["drivers"]["E2"]["envelope"]["tr"] = "10fs"

    exo, sol, ppo = _run_exo(run_cfg)

    result = sol["solver result"]
    t = np.array(result.ts["default"])
    e_sq = np.array(result.ys["default"]["e_sq"])

    from adept._lpse2d.core.epw import landau_damping_rate

    d = exo.adept_module.cfg["units"]["derived"]
    kx = np.array(exo.adept_module.cfg["grid"]["kx"])
    m = int(np.argmin(np.abs(kx - k0)))
    gamma_k = float(np.array(landau_damping_rate(jnp.array(kx[m] ** 2), d["wp0"], d["vte_sq"], jnp.array(1.0))))

    # driver is fully off past 0.31 ps; fit ~5 e-folds of clean decay
    fit = (t > 0.315) & (t < 0.40)
    slope = np.polyfit(t[fit], np.log(e_sq[fit]), 1)[0]
    np.testing.assert_allclose(-slope / 2.0, gamma_k, rtol=0.15)


@pytest.mark.slow
def test_oneil_flattening():
    """M3b (qualitative): a large-amplitude wave digs a plateau at v_phi and the
    extracted damping at that mode collapses (O'Neil-style inflation).

    The wave amplitude is held fixed but its envelope phase must rotate at
    exp(-i delta_omega(k) t) like a free solver mode: the physical wave then moves
    at the Bohm-Gross phase velocity, which is where the damping extraction reads
    the slope. (A phase-frozen envelope oscillates at bare wp0 and would dig its
    plateau ~0.5 vte below the extraction point.) The trapping width 2 w_b / k
    must also span several histogram bins for the plateau to be resolvable."""
    import jax

    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    n_p = 100000
    cfg = _make_cfg({"n_particles": n_p, "tau_damping": "50fs"})
    hpe = HybridParticleEvolution(cfg)
    state = load_particles(cfg)

    derived = cfg["units"]["derived"]
    vte = np.sqrt(derived["vte_sq"])
    wp0 = derived["wp0"]
    k_target = 0.32 * wp0 / vte

    # bounce period ~0.04 ps: ~25 bounce periods in 1 ps, and a trapping width
    # 2 w_b / k ~ 0.4 vte ~ 6 histogram bins
    e_over_m = derived["e"] / derived["me"]
    phi_k, k = _single_mode_phi_k(cfg, k_target, 1.0)
    w_b = 2.0 * np.pi / 0.04
    e_amp = w_b**2 / (e_over_m * k)
    phi_k, k = _single_mode_phi_k(cfg, k_target, e_amp)

    y = {
        "epw": jnp.array(phi_k),
        "x_e": jnp.array(state["x_e"]),
        "u_e": jnp.array(state["u_e"]),
        "epw_hist": jnp.array(state["epw_hist"]),
        "gamma_L": jnp.array(state["gamma_L"]),
    }

    step = jax.jit(hpe.__call__)
    dt = cfg["grid"]["dt"]
    delta_omega = 1.5 * derived["vte_sq"] / wp0 * k**2
    n_steps = int(1.0 / dt)
    for i in range(n_steps):
        y = step(i * dt, y)
        # fixed amplitude, free-mode phase rotation (see docstring)
        y["epw"] = jnp.array(phi_k) * jnp.exp(-1j * delta_omega * (i + 1) * dt)

    m = int(np.argmin(np.abs(np.array(cfg["grid"]["kx"]) - k)))
    gamma_final = float(y["gamma_L"][m, 0])
    gamma_analytic = float(np.array(hpe.gamma_analytic)[m, 0])
    # qualitative O'Neil check: the finite-bin plateau and passing particles leave a
    # residual slope, so "collapses" means well under the analytic rate, not zero
    assert gamma_final < 0.6 * gamma_analytic, (
        f"no damping reduction: gamma {gamma_final:.3g} vs analytic {gamma_analytic:.3g}"
    )


# ----------------------------------------------------------------------------------
# LPSE HPE controls and instruments (plan item D.1)


def test_gamma_limits_and_allow_growth():
    """LPSE hpe.gammaLimit: inside the resonant band the applied rate is clipped at
    [-growth (if allowed) or 0, damping]; outside it the analytic rate is kept."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    cfg = _make_cfg({"gamma_limit_damping": 0.5})
    hpe = HybridParticleEvolution(cfg)
    band = np.asarray(hpe.mask_res)
    # an inverted (rising) tail on the +v side gives negative raw rates for +k modes
    v = np.asarray(hpe.v_centers)
    rising = np.exp(np.clip(v / hpe.vte, -50, 50) * 0.5) * (np.abs(v) > hpe.v_min)
    rising = rising / (rising.sum() * hpe.dv)
    gamma = np.asarray(hpe.damping(jnp.asarray(rising)))[:, 0]
    assert gamma[band].max() <= 0.5 + 1e-12
    assert gamma[band].min() >= 0.0  # growth not allowed by default
    cfg2 = _make_cfg({"gamma_limit_damping": 0.5, "allow_growth": True, "gamma_limit_growth": 0.25})
    hpe2 = HybridParticleEvolution(cfg2)
    gamma2 = np.asarray(hpe2.damping(jnp.asarray(rising)))[:, 0]
    assert gamma2[band].min() < 0.0 and gamma2[band].min() >= -0.25 - 1e-12


def test_thermalization_probability_zero_passes_through_periodically():
    """LPSE particle walls: a crossing is counted, then the particle is thermalized with the
    per-direction probability or else wraps around periodically."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    for p_therm, expect_wrap in ((0.0, True), (1.0, False)):
        cfg = _make_cfg_2d(
            {"n_particles": 4000, "thermalization_probability": [p_therm, p_therm]},
            {"terms.epw.boundary.x": "absorbing", "terms.epw.boundary.y": "absorbing"},
        )
        hpe = HybridParticleEvolution(cfg)
        n = hpe.n_p
        x = jnp.full((n,), hpe.xmax + 0.1 * hpe.dx)  # all just past the right wall
        y = jnp.full((n,), 0.5 * (hpe.ymin + hpe.ymax))
        u = jnp.stack((jnp.full((n,), 3.0 * hpe.vte), jnp.full((n,), 1.0 * hpe.vte)), axis=-1)
        x2, y2, u2, flux, cone = hpe._apply_boundaries_2d(x, y, u, 0.0)
        if expect_wrap:
            np.testing.assert_allclose(np.asarray(x2), hpe.xmin + 0.1 * hpe.dx, rtol=1e-9)
            np.testing.assert_allclose(np.asarray(u2), np.asarray(u))
        else:
            assert float(jnp.max(x2)) <= hpe.xmax
            assert np.all(np.asarray(u2[:, 0]) < 0.0)  # re-injected inward with resampled speeds
            assert np.std(np.asarray(u2[:, 1])) > 0.1 * hpe.vte
        # the wall-flux instrument saw every particle leave through the right wall either way
        flux = np.asarray(flux)
        assert flux[1].sum() > 0.0 and flux[0].sum() == 0.0 and flux[2].sum() == 0.0 and flux[3].sum() == 0.0
    # periodic field boundaries default to pass-through, absorbing ones to thermalization
    hpe_p = HybridParticleEvolution(_make_cfg_2d({"n_particles": 100}))
    assert hpe_p.p_therm_x == 0.0 and hpe_p.p_therm_y == 0.0
    hpe_a = HybridParticleEvolution(_make_cfg_2d({"n_particles": 100}, {"terms.epw.boundary.x": "absorbing"}))
    assert hpe_a.p_therm_x == 1.0 and hpe_a.p_therm_y == 0.0


def test_wall_flux_bins_and_cone_energy():
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    cfg = _make_cfg_2d(
        {"n_particles": 1000, "flux_bins": [0.0, 20.0, 1.0e9], "cone_angle": 30.0, "cone_direction": [1.0, 0.0]},
        {"terms.epw.boundary.x": "absorbing", "terms.epw.boundary.y": "absorbing"},
    )
    hpe = HybridParticleEvolution(cfg)
    n = hpe.n_p
    c = hpe.c
    # half the particles at 10 keV leaving right along +x (inside the cone), half at 40 keV leaving
    # through the top wall along +y (outside the cone)
    gamma_10, gamma_40 = 1.0 + 10.0 / 510.999, 1.0 + 40.0 / 510.999
    u10, u40 = c * np.sqrt(gamma_10**2 - 1.0), c * np.sqrt(gamma_40**2 - 1.0)
    half = n // 2
    x_mid, y_mid = 0.5 * (hpe.xmin + hpe.xmax), 0.5 * (hpe.ymin + hpe.ymax)
    x = jnp.concatenate((jnp.full((half,), hpe.xmax + 0.1 * hpe.dx), jnp.full((n - half,), x_mid)))
    y = jnp.concatenate((jnp.full((half,), y_mid), jnp.full((n - half,), hpe.ymax + 0.1 * hpe.dy)))
    ux = jnp.concatenate((jnp.full((half,), u10), jnp.zeros((n - half,))))
    uy = jnp.concatenate((jnp.zeros((half,)), jnp.full((n - half,), u40)))
    _, _, _, flux, cone = hpe._apply_boundaries_2d(x, y, jnp.stack((ux, uy), axis=-1), 0.0)
    flux = np.asarray(flux)
    np.testing.assert_allclose(flux[1, 0], 10.0 * half, rtol=1e-6)  # right wall, bin [0, 20) keV
    np.testing.assert_allclose(flux[3, 1], 40.0 * (n - half), rtol=1e-6)  # top wall, bin [20, inf) keV
    assert flux[1, 1] == 0.0 and flux[3, 0] == 0.0
    np.testing.assert_allclose(float(cone[0]), 10.0 * half, rtol=1e-6)


def test_magnetic_field_rotates_momentum_without_changing_energy():
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    b_tesla = 2000.0
    cfg = _make_cfg_2d({"n_particles": 100, "magnetic_field": b_tesla})
    hpe = HybridParticleEvolution(cfg)
    n = hpe.n_p
    x = jnp.full((n,), 0.5 * (hpe.xmin + hpe.xmax))
    y = jnp.full((n,), 0.5 * (hpe.ymin + hpe.ymax))
    u = jnp.stack((jnp.full((n,), 2.0 * hpe.vte), jnp.zeros((n,))), axis=-1)
    zero_field = jnp.zeros((hpe.nx_f, hpe.ny_f, 2), dtype=jnp.complex128)
    _, _, u2, *_ = hpe.push_2d(x, y, u, zero_field, 0.0)
    u2 = np.asarray(u2)
    np.testing.assert_allclose(np.hypot(u2[:, 0], u2[:, 1]), 2.0 * hpe.vte, rtol=1e-9)
    gamma_rel = np.sqrt(1.0 + (2.0 * hpe.vte / hpe.c) ** 2)
    theta = 0.175882 * b_tesla * hpe.dt / gamma_rel  # cyclotron angle over one EPW step
    np.testing.assert_allclose(np.arctan2(u2[:, 1], u2[:, 0]), theta, rtol=1e-6)


def test_energy_conservation_multiplier_tracks_the_particle_gain():
    """Plan 2 N.2: the multiplier solves LPSE's exponential expected-loss form,
    ``W (1 - exp(-2 dt M gamma)) = gain`` on a single mode, i.e. ``M = -ln(1 - gain/W) / (2 dt gamma)``,
    to LPSE's 1 % tolerance; the linearised ``gain / (2 dt gamma W)`` is only its small-argument
    limit and under-estimates ``M`` on strongly damped modes."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution

    cfg = _make_cfg({"energy_conservation": True, "energy_conservation_steps": 1.0})
    hpe = HybridParticleEvolution(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128).at[3, 0].set(1.0e-3)
    wave = float(hpe.wave_energy_scale * jnp.sum(hpe.k_sq * jnp.abs(phi_k) ** 2))
    scale = hpe.particle_energy_scale

    def exact(gamma, gain):
        return -np.log(1.0 - gain / wave) / (2.0 * hpe.dt * gamma)

    # weakly damped: the exponential and linear forms agree to 0.1 %
    gamma_l = jnp.full((nx, ny), 0.1)
    gain = 2.0 * (2.0 * 0.1 * hpe.dt * wave)
    m = hpe.ld_multiplier(jnp.ones((1,)), gamma_l, phi_k, jnp.asarray(gain / scale))
    np.testing.assert_allclose(float(m[0]), exact(0.1, gain), rtol=1e-2)
    assert abs(float(m[0]) - 2.0) < 5e-3
    assert float(hpe.expected_wave_loss(m[0], gamma_l, phi_k)) == pytest.approx(gain, rel=1e-2)
    # strongly damped (2 gamma dt = 1): gain = W/2 needs M = ln 2 = 0.693, the linear form says 0.5
    gamma_s = jnp.full((nx, ny), 0.5 / hpe.dt)
    m_s = hpe.ld_multiplier(jnp.ones((1,)), gamma_s, phi_k, jnp.asarray(0.5 * wave / scale))
    np.testing.assert_allclose(float(m_s[0]), np.log(2.0), rtol=1e-2)
    assert float(m_s[0]) > 0.6
    # clamped at LPSE's MAX_MULTIPLIER = 10 when the particles gained far more than any loss,
    # and at 1/10 when they lost energy
    m_clamped = hpe.ld_multiplier(jnp.ones((1,)), gamma_l, phi_k, jnp.asarray(100.0 * gain / scale))
    np.testing.assert_allclose(float(m_clamped[0]), 10.0, rtol=1e-6)
    m_floor = hpe.ld_multiplier(jnp.ones((1,)), gamma_l, phi_k, jnp.asarray(-gain / scale))
    np.testing.assert_allclose(float(m_floor[0]), 0.1, rtol=1e-6)
    # with a running average over 4 steps the multiplier moves a quarter of the way
    cfg4 = _make_cfg({"energy_conservation": True, "energy_conservation_steps": 4.0})
    m4 = HybridParticleEvolution(cfg4).ld_multiplier(jnp.ones((1,)), gamma_l, phi_k, jnp.asarray(gain / scale))
    np.testing.assert_allclose(float(m4[0]), 1.0 + (exact(0.1, gain) - 1.0) / 4.0, rtol=1e-2)


def test_energy_conservation_running_average_and_warm_up_gate():
    """LPSE averages the particle energy change over ``numStepsToAverageEnergyChange`` steps
    and re-solves the multiplier only after that many feedback steps; before that the
    applied rate is the HPE rate times the previous (initial, 1) multiplier."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    cfg = _make_cfg({"energy_conservation": True, "energy_conservation_steps": 3.0, "n_particles": 2000})
    hpe = HybridParticleEvolution(cfg)
    derived = cfg["units"]["derived"]
    state = {k: jnp.asarray(v) for k, v in load_particles(cfg).items()}
    phi_k, _ = _single_mode_phi_k(cfg, 0.28 * derived["wp0"] / np.sqrt(derived["vte_sq"]), 1e-9)
    state["epw"] = jnp.asarray(phi_k)
    mults, powers = [], []
    for i in range(6):
        state = hpe(i * cfg["grid"]["dt"], state)
        mults.append(float(state["hpe_ld_multiplier"][0]))
        powers.append(float(state["hpe_particle_power"][0]))
    # gate: counts 1, 2, 3 keep the initial multiplier; from the 4th call it is re-solved
    assert mults[:3] == [1.0, 1.0, 1.0]
    assert any(m != 1.0 for m in mults[3:])
    # the running average carries state between calls
    assert any(powers[i] != powers[i - 1] for i in range(1, 6))
    assert state["hpe_particle_power"].shape == (1,)


def test_translator_maps_the_hpe_controls():
    from adept._lpse2d.lpse_deck import translate_parms

    parms = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "lw.enable": "true",
        "lw.spectral.dt": "0.005",
        "simulation.time.end": "1",
        "laser.1.intensity": "1e15",
        "hpe.enable": "true",
        "hpe.nElectrons": "12345",
        "hpe.velocityGrid": "128",
        "hpe.startEvolutionAt": "0.5",
        "hpe.thermalizationProbability": "1 0.1 0.1",
        "hpe.magneticField": "0 0 50",
        "hpe.gammaLimit.damping": "300",
        "hpe.gammaLimit.growth": "20",
        "hpe.allowGrowth": "true",
        "hpe.enforceEnergyConservation": "true",
        "hpe.numStepsToAverageEnergyChange": "5",
        "hpe.metrics.nFluxMetrics": "2",
        "hpe.metrics.fluxMetric.1.energy.min": "0",
        "hpe.metrics.fluxMetric.1.energy.max": "50",
        "hpe.metrics.fluxMetric.2.energy.min": "50",
        "hpe.metrics.fluxMetric.2.energy.max": "1e9",
        "hpe.metrics.nPowerMetrics": "1",
        "hpe.metrics.powerMetric.1.angle": "20",
        "hpe.metrics.powerMetric.1.direction": "0 1 0",
    }
    cfg, report = translate_parms(parms, experiment="x", run="y")
    h = cfg["terms"]["hpe"]
    assert h["active"] and h["n_particles"] == 12345 and h["nv"] == 128 and h["t_start"] == "0.5ps"
    assert h["thermalization_probability"] == [1.0, 0.1] and h["magnetic_field"] == 50.0
    assert h["gamma_limit_damping"] == 300.0 and h["gamma_limit_growth"] == 20.0 and h["allow_growth"]
    assert h["energy_conservation"] and h["energy_conservation_steps"] == 5.0
    assert h["flux_bins"] == [0.0, 50.0, 1e9] and h["cone_angle"] == 20.0 and h["cone_direction"] == [0.0, 1.0]


# ---------------------------------------------- in-plane magnetic field, p_z (plan 2 F.4) --


def test_in_plane_magnetic_field_rotates_momentum_into_pz():
    """``magnetic_field: [B_x, 0, 0]``: the 2-D tracker carries p_z and a free-streaming
    particle's momentum turns about x by omega_c dt / gamma -- after a quarter cyclotron
    period p_y has become p_z, with |p| and the kinetic energy unchanged."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    cfg = _make_cfg_2d({"n_particles": 64, "magnetic_field": [1.0, 0.0, 0.0]})
    hpe = HybridParticleEvolution(cfg)
    assert hpe.u_dim == 3 and hpe.in_plane_b
    state = load_particles(cfg)
    assert state["u_e"].shape == (64, 3) and np.all(state["u_e"][:, 2] == 0.0)
    # quarter period over one HPE step: omega_c dt / gamma = pi / 2 -> choose B accordingly
    u_y = 30.0  # um/ps, non-relativistic (gamma ~ 1 + 5e-9)
    gamma = np.sqrt(1.0 + (u_y / hpe.c) ** 2)
    omega_needed = 0.5 * np.pi * gamma / hpe.dt
    cfg = _make_cfg_2d({"n_particles": 64, "magnetic_field": [omega_needed / 0.175882, 0.0, 0.0]})
    hpe = HybridParticleEvolution(cfg)
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.uniform(cfg["grid"]["xmin"] + 1.0, cfg["grid"]["xmax"] - 1.0, 64))
    y = jnp.asarray(rng.uniform(0.5 * cfg["grid"]["ymin"], 0.5 * cfg["grid"]["ymax"], 64))  # clear of the walls
    u = jnp.asarray(np.tile([0.0, u_y, 0.0], (64, 1)))
    zero_field = jnp.zeros((hpe.nx_f, hpe.ny_f, 2), dtype=jnp.complex128)
    x_new, y_new, u_new, flux, cone, d_kinetic = hpe.push_2d(x, y, u, zero_field, 0.0)
    u_new = np.asarray(u_new)
    np.testing.assert_allclose(u_new[:, 2], u_y, rtol=1e-9)
    np.testing.assert_allclose(u_new[:, 1], 0.0, atol=1e-9 * u_y)
    np.testing.assert_allclose(u_new[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(u_new, axis=-1), u_y, rtol=1e-12)
    assert abs(float(d_kinetic)) < 1e-12
    # the in-plane displacement follows the in-plane momentum: each sub-step rotates u_half
    # by d_theta and advances y with the rotated p_y, so dy = sum_i dtp u_y cos(i d_theta) / gamma
    # (the discrete form of the quarter-turn integral (2/pi) u_y dt)
    d_theta = 0.5 * np.pi / hpe.n_sub
    expected_dy = hpe.dtp * u_y / gamma * np.sum(np.cos(d_theta * np.arange(1, hpe.n_sub + 1)))
    dy = np.asarray(y_new) - np.asarray(y)
    np.testing.assert_allclose(dy, expected_dy, rtol=1e-9)
    assert abs(expected_dy - 2.0 / np.pi * u_y * hpe.dt / gamma) < 0.1 * expected_dy
    assert np.all(np.abs(np.asarray(x_new) - np.asarray(x)) < 1e-12)


def test_out_of_plane_field_keeps_the_two_momentum_tracker():
    """``[0, 0, B_z]`` is the float form ``B_z`` bit for bit, with (p_x, p_y) particles."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    cfg_f = _make_cfg_2d({"n_particles": 256, "magnetic_field": 2.0})
    cfg_l = _make_cfg_2d({"n_particles": 256, "magnetic_field": [0.0, 0.0, 2.0]})
    hpe_f, hpe_l = HybridParticleEvolution(cfg_f), HybridParticleEvolution(cfg_l)
    assert hpe_f.u_dim == hpe_l.u_dim == 2 and not hpe_l.in_plane_b
    s_f, s_l = load_particles(cfg_f), load_particles(cfg_l)
    assert s_f["u_e"].shape == (256, 2)
    np.testing.assert_array_equal(s_f["u_e"], s_l["u_e"])
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.uniform(cfg_f["grid"]["xmin"], cfg_f["grid"]["xmax"], 256))
    y = jnp.asarray(rng.uniform(cfg_f["grid"]["ymin"], cfg_f["grid"]["ymax"], 256))
    u = jnp.asarray(rng.normal(0.0, 30.0, (256, 2)))
    zero_field = jnp.zeros((hpe_f.nx_f, hpe_f.ny_f, 2), dtype=jnp.complex128)
    out_f = hpe_f.push_2d(x, y, u, zero_field, 0.0)
    out_l = hpe_l.push_2d(x, y, u, zero_field, 0.0)
    for a, b in zip(out_f, out_l, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # and it does rotate in the plane
    assert np.abs(np.asarray(out_f[2]) - np.asarray(u)).max() > 0.0


def test_translator_maps_the_full_magnetic_field_vector():
    from adept._lpse2d.lpse_deck import translate_parms

    base = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "laser.wavelength": "0.351",
        "lw.envelopeDensity": "0.25",
        "hpe.enable": "true",
        "hpe.nElectrons": "1000",
        "simulation.time.end": "1",
    }
    cfg, report = translate_parms({**base, "hpe.magneticField": "0 0 3"}, run="bz")
    assert cfg["terms"]["hpe"]["magnetic_field"] == 3.0
    cfg, report = translate_parms({**base, "hpe.magneticField": "1.5 0 3"}, run="bx")
    assert cfg["terms"]["hpe"]["magnetic_field"] == [1.5, 0.0, 3.0]
    assert not any("magneticField" in u for u in report["unsupported"])
