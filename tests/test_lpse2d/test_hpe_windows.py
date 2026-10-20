"""HPE with a spatially windowed electron distribution (terms.hpe.n_windows > 1).

The criteria and tolerances were fixed before the tests were run (vault investigation
2026-10-08-182046-claude-lpse2d-hpe-spatial-fv-b47179bd, T2-T10; T6 and T7 as corrected
there): windows form a cos^2 partition of unity, per-window histograms sum to the global
one, the windowed damping operator is the uniform multiply for one shared k-independent
rate, the local multiplier sum_j w_j exp(-gamma_j dt) for k-independent rates, and never
raises the EPW energy; a plateau imposed on half the particles shows up in the window
rates in proportion to the windows' weight shares.
"""

import copy
import os
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{tempfile.mkdtemp(prefix='mlflow-hpe-windows-test')}"

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _make_cfg(hpe_overrides=None, cfg_overrides=None):
    """Quasi-1D uniform periodic box (epw.yaml, Te = 2 keV) with HPE on."""
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"]["ymax"] = "0.02um"
    cfg["grid"]["ymin"] = "-0.02um"
    cfg["terms"]["epw"]["damping"]["landau"] = True
    cfg["terms"]["hpe"] = {"active": True, "n_particles": 20000, "seed": 42, **(hpe_overrides or {})}
    for path, val in (cfg_overrides or {}).items():
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


def _random_phi(cfg, seed):
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * np.asarray(
        cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"]
    )
    return jnp.asarray(phi)


def _energy(solver, phi_k):
    return float(jnp.sum(solver.k_sq * jnp.abs(phi_k) ** 2))


def _psi_x(solver, phi_k):
    return np.asarray(jnp.fft.ifft2(solver.k_mag * phi_k))


# ---- T2: windows ----------------------------------------------------------------------------------


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("n_windows", [2, 3, 7])
def test_windows_are_a_cos2_partition_of_unity(periodic, n_windows):
    from adept._lpse2d.core.hpe import window_matrix, window_weights

    xmin, length = -1.5, 7.0
    rng = np.random.default_rng(0)
    x = rng.uniform(xmin, xmin + length, 10_000)
    j0, j1, w0, w1 = (np.asarray(a) for a in window_weights(jnp.asarray(x), xmin, length, n_windows, periodic))
    np.testing.assert_allclose(w0 + w1, 1.0, rtol=0, atol=1e-15)
    assert np.all((w0 >= 0) & (w1 >= 0)) and np.all((j0 >= 0) & (j0 < n_windows) & (j1 >= 0) & (j1 < n_windows))
    w = window_matrix(x, xmin, length, n_windows, periodic)
    np.testing.assert_allclose(w.sum(axis=0), 1.0, rtol=0, atol=1e-15)
    assert np.all(np.count_nonzero(w > 0, axis=0) <= 2)
    # each window is 1 at its centre
    centres = xmin + (np.arange(n_windows) + 0.5) * length / n_windows
    np.testing.assert_allclose(window_matrix(centres, xmin, length, n_windows, periodic), np.eye(n_windows), atol=1e-15)
    edge = window_matrix(np.array([xmin, xmin + length - 1e-12]), xmin, length, n_windows, periodic)
    if periodic:  # the windows wrap: the two walls share the first and last window
        np.testing.assert_allclose(edge[:, 0], edge[:, 1], atol=1e-9)
        np.testing.assert_allclose(edge[[0, -1], 0], [0.5, 0.5], atol=1e-9)
    else:  # edge windows are flat out to the walls
        np.testing.assert_allclose(edge[0, 0], 1.0, atol=1e-15)
        np.testing.assert_allclose(edge[-1, 1], 1.0, atol=1e-9)
    np.testing.assert_array_equal(window_matrix(x, xmin, length, 1, periodic), np.ones((1, x.size)))


# ---- T3: per-window histograms --------------------------------------------------------------------


@pytest.mark.parametrize("two_d", [False, True])
def test_window_histograms_sum_to_the_global_histogram(two_d):
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles

    cfg_over = {"grid.ymin": "-0.4um", "grid.ymax": "0.4um"} if two_d else None
    hpe_over = {"n_windows": 3, "n_particles": 20000, **({"n_angles": 8} if two_d else {})}
    cfg = _make_cfg(hpe_over, cfg_over)
    hpe = HybridParticleEvolution(cfg)
    state = load_particles(cfg)
    u, x = jnp.asarray(state["u_e"]), jnp.asarray(state["x_e"])
    hist_w = np.asarray(hpe.window_histogram(u, x))
    from adept._lpse2d.core.hpe import window_matrix

    norm = window_matrix(np.asarray(x), hpe.xmin, hpe.Lx, 3, hpe.periodic_x).sum(axis=1)
    weighted = np.tensordot(norm, hist_w, axes=1) * hpe.dv  # sum_j N_j f_j dv: the global counts
    hpe.windowed = False
    global_counts = np.asarray(hpe.histogram(u)) * hpe.n_p * hpe.dv
    np.testing.assert_allclose(weighted, global_counts, rtol=1e-12, atol=1e-9)
    # the windowed state built at load time is the same computation (numpy)
    np.testing.assert_allclose(np.asarray(state["epw_hist"]), hist_w, rtol=1e-12, atol=1e-12)
    # binning identical to jnp.histogram: the raw counts are equal exactly, the normalised
    # histograms to 1e-15 (the normalisation differs by an ulp in 2-D: constant vs summed weights)
    ones, zeros = jnp.ones_like(x), jnp.zeros_like(x)
    idx0 = jnp.zeros(x.shape, dtype=jnp.int32)
    if two_d:
        gamma_rel = jnp.sqrt(1.0 + jnp.sum((u / hpe.c) ** 2, axis=-1))
        samples = [((u / gamma_rel[:, None])[:, :2]) @ d for d in hpe.directions]
    else:
        samples = [u / jnp.sqrt(1.0 + (u / hpe.c) ** 2)]
    for sample in samples:
        counts = np.asarray(hpe._window_counts(sample, idx0, idx0, ones, zeros, 1))[0]
        np.testing.assert_array_equal(counts, np.asarray(jnp.histogram(sample, bins=hpe.v_edges)[0]))
    np.testing.assert_allclose(
        np.asarray(hpe.window_histogram(u, x, n_windows=1))[0], np.asarray(hpe.histogram(u)), rtol=1e-15, atol=0
    )


# ---- T4 / T5 / T6': the windowed damping operator ----------------------------------------------


def _solver(n_windows=2):
    from adept._lpse2d.core.epw import SpectralEPWSolver

    cfg = _make_cfg({"n_windows": n_windows})
    return cfg, SpectralEPWSolver(cfg)


def test_one_shared_k_independent_rate_is_the_uniform_multiply():
    cfg, solver = _solver(3)
    phi = _random_phi(cfg, 1)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    gamma = jnp.full((3, nx, ny), 7.0)
    out = np.asarray(solver.windowed_landau_damping(phi, gamma))
    expected = np.asarray(phi) * np.exp(-7.0 * solver.dt)
    k_pos = np.asarray(solver.k_mag) > 0
    np.testing.assert_allclose(out[k_pos], expected[k_pos], rtol=0, atol=1e-12 * np.abs(expected).max())
    # k = 0 carries no field energy and no Landau rate in any code path: left untouched
    np.testing.assert_array_equal(out[~k_pos], np.asarray(phi)[~k_pos])


def test_windowed_damping_never_raises_the_epw_energy():
    cfg, solver = _solver(4)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(2)
    for draw in range(20):
        phi = _random_phi(cfg, 10 + draw)
        gamma = jnp.asarray(rng.uniform(0.0, 0.5 / solver.dt, size=(4, nx, ny)))  # up to exp(-0.5) per step
        before, after = _energy(solver, phi), _energy(solver, solver.windowed_landau_damping(phi, gamma))
        assert after <= before * (1.0 + 1e-12), (draw, after / before)


def test_k_independent_window_rates_are_a_local_multiplier():
    """T6': for k-independent per-window rates the operator multiplies psi = |k| phi by
    m(x) = sum_j w_j(x) exp(-gamma_j dt) (k = 0 projected out); a packet at either window
    centre keeps sum |psi|^2 m^(2N) / sum |psi|^2 of its energy after N steps."""
    from adept._lpse2d.core.hpe import window_matrix

    cfg, solver = _solver(2)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    gamma0 = 4.0
    n_steps = round(1.0 / (gamma0 * solver.dt))
    gamma = jnp.stack([jnp.zeros((nx, ny)), jnp.full((nx, ny), gamma0)])
    x = np.asarray(cfg["grid"]["x"])
    w = window_matrix(x, cfg["grid"]["xmin"], cfg["grid"]["xmax"] - cfg["grid"]["xmin"], 2, True)
    m = (w[0] + w[1] * np.exp(-gamma0 * solver.dt))[:, None]
    # one application on a random band-limited field: P(m psi), P removing k = 0
    phi = _random_phi(cfg, 3)
    psi = _psi_x(solver, phi)
    expected = m * psi
    expected -= expected.mean()
    got = _psi_x(solver, solver.windowed_landau_damping(phi, gamma))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12 * np.abs(expected).max())
    # N steps on packets at the two window centres
    length = cfg["grid"]["xmax"] - cfg["grid"]["xmin"]
    k0 = np.asarray(cfg["grid"]["kx"])[40]
    for centre in (cfg["grid"]["xmin"] + 0.25 * length, cfg["grid"]["xmin"] + 0.75 * length):
        packet = (np.exp(1j * k0 * x) * np.exp(-0.5 * ((x - centre) / (length / 30.0)) ** 2))[:, None]
        k_mag = np.asarray(solver.k_mag)
        phi_p = jnp.asarray(np.where(k_mag > 0, np.fft.fft2(packet) / np.where(k_mag > 0, k_mag, 1.0), 0.0))
        e0 = _energy(solver, phi_p)
        for _ in range(n_steps):
            phi_p = solver.windowed_landau_damping(phi_p, gamma)
        predicted = np.sum(np.abs(packet) ** 2 * m ** (2 * n_steps)) / np.sum(np.abs(packet) ** 2)
        assert _energy(solver, phi_p) / e0 == pytest.approx(predicted, rel=1e-12)


# ---- T7': HPE end to end --------------------------------------------------------------------------


def test_a_plateau_on_half_the_particles_reaches_the_window_rates():
    """Two periodic windows; the left-half particles are flattened around the resonance of a
    mode at v_phi ~ 3.5 vte. Each window's rate ratio at that mode is the share of its weight
    from the (Maxwellian) right half, within +-0.08 (fixed in advance)."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles, window_matrix

    cfg = _make_cfg({"n_windows": 2, "n_particles": 2_000_000, "hist_smooth": 2})
    hpe = HybridParticleEvolution(cfg)
    state = load_particles(cfg)
    x, u = np.asarray(state["x_e"]), np.asarray(state["u_e"])
    v = u / np.sqrt(1.0 + (u / hpe.c) ** 2)
    v_phi = np.asarray(hpe.v_phi)
    mask = np.asarray(hpe.mask_res) & (v_phi > 0)
    m = int(np.argmin(np.where(mask, np.abs(v_phi - 3.5 * hpe.vte), np.inf)))
    half = cfg["grid"]["xmin"] + 0.5 * hpe.Lx
    left = x < half
    band = left & (np.abs(v - v_phi[m]) < 0.4 * hpe.vte)
    rng = np.random.default_rng(7)
    v_new = v_phi[m] + rng.uniform(-0.4, 0.4, band.sum()) * hpe.vte  # plateau: flat in v
    u = u.copy()
    u[band] = v_new / np.sqrt(1.0 - (v_new / hpe.c) ** 2)
    gamma = np.asarray(hpe.damping(hpe.window_histogram(jnp.asarray(u), jnp.asarray(x))))  # (2, nx, ny)
    ratio = gamma[:, m, 0] / np.asarray(hpe.gamma_analytic)[m, 0]
    w = window_matrix(x, cfg["grid"]["xmin"], hpe.Lx, 2, hpe.periodic_x)
    # window 0 is centred in the left half, window 1 in the right
    expected = [w[j][~left].sum() / w[j].sum() for j in (0, 1)]
    assert ratio[0] == pytest.approx(expected[0], abs=0.08), (ratio, expected)
    assert ratio[1] == pytest.approx(expected[1], abs=0.08), (ratio, expected)


# ---- T8: smoothing --------------------------------------------------------------------------------


@pytest.mark.parametrize("hist_smooth", [0, 2])
def test_smoothing_keeps_the_maxwellian_calibration(hist_smooth):
    from adept._lpse2d.core.hpe import HybridParticleEvolution, resonance_arrays

    cfg = _make_cfg({"hist_smooth": hist_smooth})
    hpe = HybridParticleEvolution(cfg)
    arrays = resonance_arrays(cfg)
    gamma = np.asarray(hpe.damping(jnp.asarray(arrays["f0_expected"])))[:, 0]
    raw0 = np.asarray(hpe._gamma_raw(hpe._smooth(jnp.asarray(arrays["f0_expected"]))))
    band = np.asarray(hpe.mask_res) & (raw0 > 0)
    assert band.sum() > 10
    analytic = np.asarray(hpe.gamma_analytic)[:, 0]
    np.testing.assert_allclose(gamma[band], analytic[band], rtol=1e-12)


# ---- T10: validation ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hpe_over, cfg_over, match",
    [
        ({"n_windows": 0}, {}, "n_windows"),
        ({"hist_smooth": -1}, {}, "hist_smooth"),
        ({"n_windows": 2, "energy_conservation": True}, {}, "energy_conservation"),
    ],
)
def test_unsupported_window_settings_are_refused(hpe_over, cfg_over, match):
    with pytest.raises(ValueError, match=match):
        _make_cfg(hpe_over, cfg_over)


def test_windows_with_the_combined_solver_are_refused():
    from adept._lpse2d.helpers import get_derived_quantities, write_units

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["epw"]["source"].update({"tpd": True, "srs": True})
    cfg["terms"]["light"] = {"solver": "spectral"}
    cfg["terms"]["epw"]["damping"]["landau"] = True
    cfg["terms"]["hpe"] = {"active": True, "n_windows": 2}
    write_units(cfg)
    with pytest.raises(ValueError, match=r"n_windows > 1 is not implemented for terms\.epw\.solver: combined"):
        get_derived_quantities(cfg)


# ---- T9: end to end -------------------------------------------------------------------------------


@pytest.mark.slow
def test_srs_smoke_with_hpe_windows():
    """Quasi-1D SRS with HPE in 4 windows runs 0.2 ps, stays finite, and saves the windows."""
    from adept import ergoExo

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"]["ymax"] = "0.02um"
    cfg["grid"]["ymin"] = "-0.02um"
    cfg["grid"]["tmax"] = "0.2ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.2ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.05ps"
    cfg["terms"]["hpe"] = {"active": True, "n_particles": 20000, "nv": 256, "n_windows": 4, "hist_smooth": 2}
    cfg["mlflow"]["run"] = "srs-hpe-windows-smoke"
    exo = ergoExo()
    modules = exo.setup(cfg)
    _, ppo, _ = exo(modules)
    series = ppo["series"]
    assert series["hpe_hist_windows"].shape[1:] == (4, 256)
    assert series["hpe_hist"].shape[1:] == (256,)
    for k in ("hpe_hist_windows", "hpe_hist", "epw_energy", "epw_dissipation", "fhot_50keV"):
        assert np.all(np.isfinite(np.asarray(series[k].values, dtype=float))), k


@pytest.mark.slow
def test_hist_save_interval_thins_only_the_histograms():
    """terms.hpe.hist_save_interval saves the histograms on every n-th step plus the last, on their
    own t_hist axis, equal to the every-step histograms there; every scalar series is unchanged.
    Tolerance (fixed in advance): rtol 1e-10 with atol 1e-10 x max|reference| -- the two runs
    compile different XLA programs, so bit identity is not required."""
    from adept import ergoExo

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        base = yaml.safe_load(fi)
    base["grid"]["ymax"] = "0.02um"
    base["grid"]["ymin"] = "-0.02um"
    base["grid"]["tmax"] = "0.2ps"
    base["save"]["fields"]["t"]["tmax"] = "0.2ps"
    base["save"]["fields"]["t"]["dt"] = "0.05ps"
    base["terms"]["epw"]["source"]["noise_seed"] = 7
    hpe = {"active": True, "n_particles": 20000, "nv": 256, "n_windows": 4, "hist_smooth": 2}
    series = {}
    for label, extra in (("every", {}), ("thinned", {"hist_save_interval": "0.03ps"})):
        cfg = copy.deepcopy(base)
        cfg["terms"]["hpe"] = {**hpe, **extra}
        cfg["mlflow"]["run"] = f"srs-hpe-hist-interval-{label}"
        exo = ergoExo()
        _, ppo, _ = exo(exo.setup(cfg))
        series[label] = ppo["series"]
    every, thin = series["every"], series["thinned"]

    def close(actual, reference, what):
        reference = np.asarray(reference, dtype=float)
        atol = 1e-10 * float(np.max(np.abs(reference))) if reference.size else 0.0
        np.testing.assert_allclose(np.asarray(actual, dtype=float), reference, rtol=1e-10, atol=atol, err_msg=what)

    t = np.asarray(every["t (ps)"].values)
    idx = list(range(0, t.size, 30))  # 0.03 ps / 1 fs
    if idx[-1] != t.size - 1:
        idx.append(t.size - 1)
    assert len(idx) < t.size
    np.testing.assert_array_equal(np.asarray(thin["t_hist (ps)"].values), t[idx])
    for k in ("hpe_hist", "hpe_hist_windows"):
        assert every[k].dims[0] == "t (ps)"
        assert thin[k].dims[0] == "t_hist (ps)"
        close(thin[k].values, every[k].values[idx], k)
    scalars = [k for k, v in every.data_vars.items() if v.dims == ("t (ps)",)]
    assert scalars
    for k in scalars:
        assert thin[k].dims == ("t (ps)",)
        close(thin[k].values, every[k].values, k)


def test_windowed_dissipation_is_signed_with_allow_growth():
    """terms.hpe.allow_growth with n_windows > 1: epw_dissipation from the windowed rates is the signed
    sum prefactor * factor * sum_j sum_k L_jk |chunk_jk|^2 / (nx ny^2), chunk_j = fft2(sqrt(w_j) psi),
    psi = ifft2(|k| phi_k), L = (1 - exp(-2 rate dt)) / dt (Parseval; uniform box, constant
    total-to-electric factor). Before the fix sqrt(L < 0) made it NaN (PR #387 review). With
    non-negative rates the series is bitwise the same with and without allow_growth. rtol 1e-12,
    fixed in advance."""
    from adept._lpse2d.core.hpe import load_particles, window_matrix
    from adept._lpse2d.helpers import get_default_save_func, get_save_quantities

    def dissipation(allow_growth, signed_rates):
        cfg = get_save_quantities(_make_cfg({"n_windows": 3, "allow_growth": allow_growth}))
        nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
        phi_k = 1e-3 * _random_phi(cfg, 5)
        state = {k: jnp.asarray(v) for k, v in load_particles(cfg).items()}
        rate = np.random.default_rng(7).normal(size=state["gamma_L_windows"].shape) * 50.0
        gamma = rate if signed_rates else np.abs(rate)
        zeros = jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64)
        state.update({"gamma_L_windows": jnp.asarray(gamma), "epw": phi_k.view(jnp.float64), "E0": zeros, "E1": zeros})
        out = get_default_save_func(cfg)["func"](0.0, state, None)
        return cfg, np.asarray(phi_k), gamma, float(out["epw_dissipation"])

    cfg, phi_k, gamma, got = dissipation(True, True)
    assert np.isfinite(got)
    d, grid = cfg["units"]["derived"], cfg["grid"]
    nx, ny, dt = grid["nx"], grid["ny"], grid["dt"]
    k_sq = np.asarray(grid["kx"])[:, None] ** 2 + np.asarray(grid["ky"])[None, :] ** 2
    w = window_matrix(
        np.asarray(grid["x"]),
        grid["xmin"],
        grid["xmax"] - grid["xmin"],
        3,
        cfg["terms"]["epw"]["boundary"]["x"] == "periodic",
    )
    chunks = np.fft.fft2(np.sqrt(w)[:, :, None] * np.fft.ifft2(np.sqrt(k_sq) * phi_k)[None], axes=(1, 2))
    loss = (1.0 - np.exp(-2.0 * (gamma + d.get("nu_coll", 0.0) * (k_sq > 0)) * dt)) / dt
    weighted = loss * np.abs(chunks) ** 2
    assert np.sum(weighted * (loss < 0)) < 0.0  # the gain modes contribute
    factor = 2.0 * (2.0 - np.asarray(grid["background_density"]) / cfg["units"]["envelope density"])
    assert np.ptp(factor) == 0.0
    prefactor = 0.25 * grid["dx"] * d["x_norm"] * d["e_norm"] ** 2
    assert got == pytest.approx(prefactor * float(factor.flat[0]) * np.sum(weighted) / (nx * ny**2), rel=1e-12)
    assert dissipation(True, False)[3] == dissipation(False, False)[3]


@pytest.mark.parametrize("two_d", [False, True])
def test_sparse_loading_leaves_empty_windows_finite(two_d):
    """3 particles in 7 windows: each particle weighs at most two adjacent windows, so at least one
    window is empty. Its initial histogram is zero (the runtime window_histogram policy), not 0/0 NaN
    (PR #387 review); the loaded epw_hist equals the runtime histogram of the same particles (1e-12)."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles, window_matrix

    cfg_over = {"grid.ymin": "-0.4um", "grid.ymax": "0.4um"} if two_d else None
    hpe_over = {"n_windows": 7, "n_particles": 3, **({"n_angles": 8} if two_d else {})}
    cfg = _make_cfg(hpe_over, cfg_over)
    state = load_particles(cfg)
    hpe = HybridParticleEvolution(cfg)
    norm = window_matrix(np.asarray(state["x_e"]), hpe.xmin, hpe.Lx, 7, hpe.periodic_x).sum(axis=1)
    empty = norm == 0.0
    assert empty.any()
    hist = np.asarray(state["epw_hist"])
    assert np.all(np.isfinite(hist))
    assert np.all(hist[empty] == 0.0) and np.all(hist[~empty].sum(axis=-1) > 0.0)
    runtime = np.asarray(hpe.window_histogram(jnp.asarray(state["u_e"]), jnp.asarray(state["x_e"])))
    np.testing.assert_allclose(hist, runtime, rtol=1e-12, atol=1e-12)
