"""Tests for the interpolated field save (``save.fields.x.dx`` / ``save.fields.y.dy``).

An interpolated save puts every saved field on a coarser, cell-centred grid of
``n_save = round(L / dx_save)`` points by linear interpolation in x-space. The full-grid
save (no ``save.fields.x`` / ``y``) is the reference: sampling a full-grid field at the
save coordinates with the same linear interpolation must reproduce the interpolated save.
"""

import numpy as np
import pytest
import yaml
from scipy.interpolate import RegularGridInterpolator

GRID_DX_UM = 0.05
X_LENGTH_UM = 6.4
Y_HALF_WIDTH_UM = {"2d": 0.8, "1d": 0.02}  # "1d" collapses the box to ny = 1
EXPECTED_FIELDS = {
    "phi",
    "ex",
    "ey",
    "e0_x",
    "e0_y",
    "e1_x",
    "e1_y",
    "background_density",
    "iaw_density",
    "iaw_velocity_divergence",
}
KX = r"kx ($kc\omega_0^{-1}$)"
KY = r"ky ($kc\omega_0^{-1}$)"

# Round-off budget, in units of the saved dtype's machine epsilon, for two evaluations of the
# same linear interpolation of the same data (one inside the jitted save, one here in numpy):
#   * query-coordinate rounding: |df/dx| * |x| * eps / 2 <= (pi / dx) * L * eps / 2 * max|f|
#     = pi * nx / 2 * eps ~ 200 eps for nx = 128 (band-limited field, |k| <= pi / dx),
#   * the x-space FFT of phi (XLA vs numpy FFT): ~ log2(nx * ny) eps ~ 12 eps,
#   * the interpolation weights: a few eps.
# ~ 220 eps in total; 2000 eps leaves a 9x margin and is still > 1e10 below the O(1) defect.
ROUNDOFF_EPS = 2000.0


def _box_cfg(geometry: str, save_spacing: str | None, run_name: str) -> dict:
    """A small TPD+SRS+IAW box: noise-seeded EPW, static uniform pump, 6 fs."""
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    half = Y_HALF_WIDTH_UM[geometry]
    cfg["density"] = {"basis": "uniform", "val": 0.24}
    cfg["grid"].update(
        {
            "boundary_width": "1um",
            "dt": "1fs",
            "dx": f"{GRID_DX_UM}um",
            "xmax": f"{X_LENGTH_UM}um",
            "tmax": "6fs",
            "ymax": f"{half}um",
            "ymin": f"{-half}um",
        }
    )
    cfg["save"]["fields"] = {"t": {"tmin": "0fs", "tmax": "6fs", "dt": "3fs"}}
    if save_spacing is not None:
        cfg["save"]["fields"]["x"] = {"dx": save_spacing}
        cfg["save"]["fields"]["y"] = {"dy": save_spacing}
    cfg["terms"]["epw"]["source"].update({"noise": True, "noise_seed": 5, "tpd": True, "srs": True})
    cfg["terms"]["iaw"] = {
        "active": True,
        "boundary": {"x": "absorbing", "y": "periodic"},
        "damping": {"collisions": 1.0e-5, "landau": 0.1},
        "max_density_perturbation": 0.1,
    }
    cfg["drivers"]["E0"]["params"] = {"phases": {"seed": 42}}
    cfg["mlflow"] = {"experiment": "test-lpse2d-field-save", "run": run_name}
    return cfg


def _prepared_cfg(cfg: dict) -> dict:
    """Run the cfg through the setup steps up to the save functions, without MLflow."""
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_save_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return get_save_quantities(cfg)


def _run(cfg: dict) -> dict:
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    _, ppo, _ = exo(modules)
    return ppo


def _sample_linear(values: np.ndarray, x_src, y_src, x_q, y_q) -> np.ndarray:
    """Linear interpolation of (nt, nx, ny) data onto the (x_q, y_q) tensor grid."""
    if len(y_src) == 1:
        sampled = [np.interp(x_q, x_src, v[:, 0].real) + 1j * np.interp(x_q, x_src, v[:, 0].imag) for v in values]
        return np.asarray(sampled)[..., None]
    xg, yg = np.meshgrid(x_q, y_q, indexing="ij")
    points = np.stack([xg.ravel(), yg.ravel()], axis=-1)
    return np.asarray([RegularGridInterpolator((x_src, y_src), v)(points).reshape(xg.shape) for v in values])


def _expected_save_axis(n_save: int, lo: float, hi: float) -> np.ndarray:
    return lo + (np.arange(n_save) + 0.5) * (hi - lo) / n_save


@pytest.mark.parametrize("geometry", ["2d", "1d"])
def test_interpolated_save_matches_full_grid_save(geometry, monkeypatch):
    """Same run, saved on the full grid and with save dx = dy = 1.6 x the grid spacing.

    Every x-space field of the interpolated save (phi, ex, ey, e0_x/y, e1_x/y,
    background_density, iaw_density, iaw_velocity_divergence) must equal the full-grid
    field sampled at the save coordinates by linear interpolation. Both sides are the same
    linear interpolation of the same data, so the tolerance is round-off:
    max|interp - full(at save points)| <= ROUNDOFF_EPS * eps(saved dtype) * max|full(at save points)|
    (2000 eps; derivation at ROUNDOFF_EPS). A field that is zero in the full-grid save must
    be exactly zero.

    The save grid has n_save = round(L / 0.08 um) cell-centred points: 80 in x; 20 in y in
    2D, 1 in the quasi-1D box (L_y = one 0.05 um cell). Its coordinates and the k axes of
    k-fields.xr are those of that grid, to 1e-13 (a few float64 operations each side, ~1e-15);
    every saved value (x- and k-space) is finite.

    The full-grid reference run skips the plots (90% of its run time; full-grid plotting is
    exercised elsewhere in the suite); the interpolated run plots, so post-processing is run
    on the save grid.
    """
    from adept._lpse2d import helpers

    with monkeypatch.context() as patch:
        for plot in ("plot_series", "plot_srs_diagnostics", "plot_fields", "plot_kt"):
            patch.setattr(helpers, plot, lambda *args, **kwargs: None)
        full = _run(_box_cfg(geometry, None, f"field-save-full-{geometry}"))
    interp = _run(_box_cfg(geometry, "0.08um", f"field-save-interp-{geometry}"))

    xs_full = full["x"].coords["x (um)"].values
    ys_full = full["x"].coords["y (um)"].values
    nx, ny = xs_full.size, ys_full.size
    assert nx == 128 and ny == {"2d": 32, "1d": 1}[geometry]

    # the save grid: n_save cell-centred points spanning the simulation box
    n_save_x, n_save_y = 80, {"2d": 20, "1d": 1}[geometry]
    xs = interp["x"].coords["x (um)"].values
    ys = interp["x"].coords["y (um)"].values
    ly = ny * GRID_DX_UM
    np.testing.assert_allclose(xs, _expected_save_axis(n_save_x, 0.0, X_LENGTH_UM), rtol=0, atol=1e-13 * X_LENGTH_UM)
    np.testing.assert_allclose(ys, _expected_save_axis(n_save_y, -ly / 2, ly / 2), rtol=0, atol=1e-13 * ly)

    assert set(interp["x"].data_vars) == EXPECTED_FIELDS
    assert set(full["x"].data_vars) == EXPECTED_FIELDS
    nt = full["x"].coords["t (ps)"].size
    for name in EXPECTED_FIELDS:
        got = np.asarray(interp["x"][name].values)
        assert got.shape == (nt, n_save_x, n_save_y), name
        assert np.all(np.isfinite(got)), f"{name} has non-finite values"

        ref = _sample_linear(np.asarray(full["x"][name].values), xs_full, ys_full, xs, ys)
        scale = np.max(np.abs(ref))
        eps = np.finfo(got.dtype).eps
        err = np.max(np.abs(got - ref))
        assert err <= ROUNDOFF_EPS * eps * scale, f"{name}: max err {err:.3e} vs max |full| {scale:.3e}"

    # every derived k-space field lives on the save grid's k axes. The k coords are stored in
    # units of w0/c; read that scale off the full-grid save, whose kx axis is the grid's.
    kx_full_axis = np.fft.fftshift(np.fft.fftfreq(nx, d=GRID_DX_UM / 2.0 / np.pi))
    ratio = full["k"].coords[KX].values.max() / kx_full_axis.max()
    np.testing.assert_allclose(full["k"].coords[KX].values, kx_full_axis * ratio, rtol=1e-13, atol=0)
    dx_save = X_LENGTH_UM / n_save_x
    dy_save = ly / n_save_y
    kx_expected = np.fft.fftshift(np.fft.fftfreq(n_save_x, d=dx_save / 2.0 / np.pi)) * ratio
    ky_expected = np.fft.fftshift(np.fft.fftfreq(n_save_y, d=dy_save / 2.0 / np.pi)) * ratio
    np.testing.assert_allclose(interp["k"].coords[KX].values, kx_expected, rtol=1e-13, atol=0)
    np.testing.assert_allclose(interp["k"].coords[KY].values, ky_expected, rtol=1e-13, atol=0)
    for name in interp["k"].data_vars:
        got = np.asarray(interp["k"][name].values)
        assert got.shape == (nt, n_save_x, n_save_y), name
        assert np.all(np.isfinite(got)), f"k-space {name} has non-finite values"


def _plane_wave(cfg, amplitude, mx, my):
    """amplitude * exp(i (kx (x - x0) + ky (y - y0))) on the simulation grid, for grid mode (mx, my)."""
    x, y = np.asarray(cfg["grid"]["x"]), np.asarray(cfg["grid"]["y"])
    kx, ky = cfg["grid"]["kx"][mx], cfg["grid"]["ky"][my]
    return amplitude * np.exp(1j * (kx * (x[:, None] - x[0]) + ky * (y[None, :] - y[0]))), kx, ky


@pytest.mark.parametrize("geometry", ["2d", "1d"])
def test_interpolated_save_of_band_limited_state(geometry, tmp_path):
    """Analytic check of the save function and make_field_xarrays on a single-mode state,
    save dx = dy = 1.6 x the grid spacing.

    The EPW state is phi_k with one grid mode (mx, my) = (3, 1) in 2D, (3, 0) in quasi-1D,
    phi_k[mx, my] = A nx ny, i.e. phi(x) = A exp(i(kx (x - x0) + ky (y - y0))) with the
    solver's convention phi(x) = ifft2(phi_k), E = ifft2(-i k phi_k). E0, E1 and the IAW
    fields are other plane waves of the grid.

    Expected, on the save grid (x_s, y_s):
      * x-space: phi(x_s), ex = -i kx phi(x_s), ey = -i ky phi(x_s) -- the solver's field, as in
        the full-grid save -- and E0, E1, IAW at x_s.
      * k-space at the mode's bin of the save grid: phi_k = A nx ny exp(i(kx (x_s0 - x0) + ky (y_s0 - y0)))
        (the full-grid coefficient; the phase is the DFT origin moving to the first save point),
        ex_k = -i kx phi_k, ey_k = -i ky phi_k.
    Tolerance: bilinear interpolation of exp(i q.x) errs by at most (qx^2 hx^2 + qy^2 hy^2) / 8 per
    real and imaginary part (tensor-product bound, h = grid spacing), so
    |err| <= sqrt(2) (qx^2 + qy^2) h^2 / 8 * max|f|; the DFT of the interpolation error is bounded by
    the same relative amount. Round-off allowance: ROUNDOFF_EPS * eps on top.
    """
    from adept._lpse2d.helpers import make_field_xarrays

    save_spacing = "0.08um"
    cfg = _prepared_cfg(_box_cfg(geometry, save_spacing, "unused"))
    grid = cfg["grid"]
    nx, ny = grid["nx"], grid["ny"]
    my = 1 if geometry == "2d" else 0
    h = grid["dx"]

    amp = 2.0e-3
    phi_x, kx, ky = _plane_wave(cfg, amp, 3, my)
    phi_k = np.fft.fft2(phi_x)
    e0y, q0x, q0y = _plane_wave(cfg, 1.0e-2, 2, my)
    e1x, q1x, q1y = _plane_wave(cfg, 3.0e-4, nx - 4, 0)
    iaw, qix, qiy = _plane_wave(cfg, 1.0e-3, 1, my)
    zeros = np.zeros_like(e0y)
    state = {
        "epw": phi_k.view(np.float64),
        "E0": np.stack([zeros, e0y], axis=-1).view(np.float64),
        "E1": np.stack([e1x, zeros], axis=-1).view(np.float64),
        "iaw_density": iaw.real.copy(),
        "iaw_velocity_divergence": iaw.imag.copy(),
    }
    import jax.numpy as jnp

    saved = cfg["save"]["fields"]["func"](0.0, {k: jnp.asarray(v) for k, v in state.items()}, None)
    saved = {k: np.asarray(v)[None] for k, v in saved.items()}  # one save time
    (tmp_path / "binary").mkdir()
    kfields, fields = make_field_xarrays(cfg, np.array([0.0]), saved, str(tmp_path))

    n_save_x = round(X_LENGTH_UM / float(save_spacing.removesuffix("um")))
    n_save_y = 1 if geometry == "1d" else round(ny * h / float(save_spacing.removesuffix("um")))
    xs = fields.coords["x (um)"].values
    ys = fields.coords["y (um)"].values
    assert xs.shape == (n_save_x,) and ys.shape == (n_save_y,)
    x0, y0 = grid["x"][0], grid["y"][0]

    def at_save(amplitude, qx, qy):
        return amplitude * np.exp(1j * (qx * (xs[:, None] - x0) + qy * (ys[None, :] - y0)))

    def tol(qx, qy, scale, dtype):
        interp_bound = np.sqrt(2.0) * (qx**2 + qy**2) * h**2 / 8.0
        return (interp_bound + ROUNDOFF_EPS * np.finfo(dtype).eps) * scale

    phi_s = at_save(amp, kx, ky)
    expected_x = {
        "phi": (phi_s, kx, ky),
        "ex": (-1j * kx * phi_s, kx, ky),
        "ey": (-1j * ky * phi_s, kx, ky),
        "e0_x": (np.zeros_like(phi_s), 0.0, 0.0),
        "e0_y": (at_save(1.0e-2, q0x, q0y), q0x, q0y),
        "e1_x": (at_save(3.0e-4, q1x, q1y), q1x, q1y),
        "e1_y": (np.zeros_like(phi_s), 0.0, 0.0),
        "iaw_density": (at_save(1.0e-3, qix, qiy).real, qix, qiy),
        "iaw_velocity_divergence": (at_save(1.0e-3, qix, qiy).imag, qix, qiy),
    }
    for name, (expected, qx, qy) in expected_x.items():
        got = np.asarray(fields[name].values)[0]
        assert got.shape == (n_save_x, n_save_y), name
        assert np.all(np.isfinite(got)), name
        scale = np.max(np.abs(expected))
        err = np.max(np.abs(got - expected))
        assert err <= tol(qx, qy, scale, got.dtype), f"{name}: max err {err:.3e}, max |f| {scale:.3e}"

    # k-space: the mode sits in the same bin of the save grid (both grids span the same box)
    phase = np.exp(1j * (kx * (xs[0] - x0) + ky * (ys[0] - y0)))
    phi_k_expected = amp * nx * ny * phase
    for name, factor in (("phi", 1.0), ("ex", -1j * kx), ("ey", -1j * ky)):
        got_k = np.fft.ifftshift(np.asarray(kfields[name].values)[0])
        assert got_k.shape == (n_save_x, n_save_y), name
        assert np.all(np.isfinite(got_k)), name
        expected = factor * phi_k_expected
        scale = np.abs(amp * nx * ny * factor) if factor != 0 else 0.0
        err = np.abs(got_k[3, my] - expected)
        assert err <= tol(kx, ky, scale, got_k.dtype), f"k-space {name}: err {err:.3e} of {scale:.3e}"


@pytest.mark.parametrize("axis", ["x", "y"])
def test_interpolated_save_spacing_bounds(axis):
    """A save spacing below the simulation grid spacing (here half of it) is refused at setup;
    a save spacing equal to it is accepted and reproduces the simulation grid.

    An interpolated save finer than the grid carries no information the grid does not have,
    and its first/last points fall outside the outermost cell centres, where linear
    interpolation is undefined (interpax returned NaN rows there).

    At equal spacing the save axis must be the simulation axis exactly (bitwise), so the
    outermost save points sit on the outermost cell centres rather than a rounding error
    outside them (the other axis keeps a 0.08 um save spacing). The save of a random E0 and
    IAW density is then finite everywhere and equals the same linear interpolation done here
    (the identity along the equal-spacing axis) to ROUNDOFF_EPS * eps * max|f|.

    Giving only one of save.fields.x / save.fields.y is refused too (y alone used to be
    silently ignored).
    """
    import jax.numpy as jnp

    key = {"x": "dx", "y": "dy"}[axis]
    cfg = _box_cfg("2d", "0.08um", "unused")
    cfg["save"]["fields"][axis][key] = f"{GRID_DX_UM / 2}um"
    with pytest.raises(ValueError, match=rf"save\.fields\.{axis}\.{key}"):
        _prepared_cfg(cfg)

    cfg = _box_cfg("2d", "0.08um", "unused")
    del cfg["save"]["fields"][{"x": "y", "y": "x"}[axis]]
    with pytest.raises(NotImplementedError, match=r"save\.fields\.x and save\.fields\.y"):
        _prepared_cfg(cfg)

    cfg = _box_cfg("2d", "0.08um", "unused")
    cfg["save"]["fields"][axis][key] = f"{GRID_DX_UM}um"
    cfg = _prepared_cfg(cfg)
    grid, fld = cfg["grid"], cfg["save"]["fields"]
    np.testing.assert_array_equal(fld[axis]["ax"], grid[axis])

    rng = np.random.default_rng(0)
    nx, ny = grid["nx"], grid["ny"]
    e0 = rng.standard_normal((nx, ny, 2)) + 1j * rng.standard_normal((nx, ny, 2))
    iaw = rng.standard_normal((nx, ny))
    state = {"E0": e0.view(np.float64), "iaw_density": iaw}
    saved = fld["func"](0.0, {k: jnp.asarray(v) for k, v in state.items()}, None)
    other = {"x": "y", "y": "x"}[axis]
    for name, value in (("E0", e0), ("iaw_density", iaw)):
        got = np.asarray(saved[name])
        if name == "E0":
            got = got.view(np.complex128 if got.dtype == np.float64 else np.complex64)
        assert np.all(np.isfinite(got)), name
        # along the equal-spacing axis the save is the identity; along the other (0.08 um) axis
        # it is a linear interpolation, so compare against that
        ref = _sample_linear(
            np.moveaxis(value, -1, 0) if name == "E0" else value[None],
            grid["x"],
            grid["y"],
            np.asarray(fld["x"]["ax"]),
            np.asarray(fld["y"]["ax"]),
        )
        ref = np.moveaxis(ref, 0, -1) if name == "E0" else ref[0]
        assert got.shape == ref.shape, name
        err = np.max(np.abs(got - ref))
        assert err <= ROUNDOFF_EPS * np.finfo(got.dtype).eps * np.max(np.abs(ref)), f"{name}: {err:.3e}"
    assert fld[other]["ax"].size == round({"x": X_LENGTH_UM, "y": ny * GRID_DX_UM}[other] / 0.08)
