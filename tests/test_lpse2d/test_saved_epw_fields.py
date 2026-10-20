"""Normalization of the EPW fields in the full-grid field save (``fields.xr`` / ``k-fields.xr``).

The solver state is the k-space potential phi_k (numpy FFT convention, DC at [0, 0]). The
solver's own field is E = ifft2(-i k phi_k) (``SpectralEPWSolver.phi_k_to_e_fields``), and the
``series.xr`` quantities ``e_sq`` / ``epw_energy`` integrate that E. ``fields.xr`` stores
phi = ifft2(phi_k), so its ``ex`` / ``ey`` must be E = -grad(phi) of the saved phi with no further
normalization, and ``k-fields.xr`` stores phi_k and -i k phi_k.

Tolerances (fixed before the tests were first run). Every comparison is between two evaluations of
the same quantity in the saved dtype (complex128 here) that differ only by round-off:

* An FFT of size N has a relative L2 error of order log2(N) eps (pocketfft and XLA alike); here
  N = nx ny = 4096 (2D) or 128 (quasi-1D), log2(N) <= 12. A forward + inverse pair (the spectral
  derivative of the saved phi) doubles that: ~24 eps.
* The analytic single mode exp(2 pi i (mx i / nx + my j / ny)) costs ~2 pi mx eps <= 20 eps per
  element for the phase rounding, plus the inverse FFT.
* The integrated e_sq: |E|^2 doubles the relative error of E, the sums add ~log2(N) eps.

All of these are below ~60 eps; ROUNDOFF_EPS = 200 eps of the reference's scale (max |.| for the
analytic mode, L2 norm for the random field, the value itself for e_sq) is the bound. The defect this
guards against is a factor 4 / (nx ny) on the saved ex / ey: a relative error of 1 - 4 / (nx ny)
(0.999 in 2D, 0.97 in quasi-1D), about 1e13 times the bound.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
import yaml

GRID_DX_UM = 0.05
X_LENGTH_UM = 6.4  # nx = 128
Y_HALF_WIDTH_UM = {"2d": 0.8, "1d": 0.02}  # "1d" collapses the box to ny = 1
EXPECTED_SHAPE = {"2d": (128, 32), "1d": (128, 1)}
MODE = {"2d": (3, 1), "1d": (3, 0)}  # (mx, my) of the analytic single-mode slice
AMPLITUDE = 2.0e-3  # |phi| of the single mode, in the solver's units
RANDOM_SEED = 20261020  # any seed: the assertions are identities that hold for every phi_k
ROUNDOFF_EPS = 200.0
KX = r"kx ($kc\omega_0^{-1}$)"
KY = r"ky ($kc\omega_0^{-1}$)"


def _prepared_cfg(geometry: str) -> dict:
    """A small TPD box taken through the setup steps up to the save functions (no MLflow, no run)."""
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_save_quantities,
        get_solver_quantities,
        write_units,
    )

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
    # full-grid save: no save.fields.x / save.fields.y
    cfg["save"]["fields"] = {"t": {"tmin": "0fs", "tmax": "6fs", "dt": "3fs"}}
    cfg["drivers"]["E0"]["params"] = {"phases": {"seed": 42}}

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return get_save_quantities(cfg)


def _phi_k_slices(cfg: dict, geometry: str) -> np.ndarray:
    """(2, nx, ny) phi_k: slice 0 a single grid mode, slice 1 a random band-limited field."""
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    mx, my = MODE[geometry]
    phi_k = np.zeros((2, nx, ny), dtype=np.complex128)
    # phi_k[mx, my] = A nx ny  <=>  phi(x_i, y_j) = A exp(2 pi i (mx i / nx + my j / ny))
    phi_k[0, mx, my] = AMPLITUDE * nx * ny

    rng = np.random.default_rng(RANDOM_SEED)
    mode_x = np.abs(np.fft.fftfreq(nx, d=1.0 / nx))
    mode_y = np.abs(np.fft.fftfreq(ny, d=1.0 / ny))
    band = (mode_x[:, None] <= nx // 4) & (mode_y[None, :] <= max(ny // 4, 0))
    band[0, 0] = False  # the solver zeroes k = 0
    coeffs = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    phi_k[1] = np.where(band, coeffs, 0.0) * AMPLITUDE * np.sqrt(nx * ny)
    return phi_k


def _saved_fields(geometry: str, td) -> tuple[dict, np.ndarray, xr.Dataset, xr.Dataset]:
    """Run the full-grid save path (save function + make_field_xarrays) on a two-slice state and
    read fields.xr / k-fields.xr back from disk."""
    from adept._lpse2d.helpers import make_field_xarrays

    cfg = _prepared_cfg(geometry)
    assert "x" not in cfg["save"]["fields"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    assert (nx, ny) == EXPECTED_SHAPE[geometry]

    phi_k = _phi_k_slices(cfg, geometry)
    zeros = np.zeros((nx, ny, 2), dtype=np.complex128)
    saved = []
    for phi_k_t in phi_k:
        state = {"epw": jnp.asarray(phi_k_t).view(jnp.float64), "E0": jnp.asarray(zeros).view(jnp.float64)}
        state["E1"] = state["E0"]
        saved.append(cfg["save"]["fields"]["func"](0.0, state, None))
    stacked = {k: np.stack([np.asarray(s[k]) for s in saved]) for k in saved[0]}

    (td / "binary").mkdir()
    make_field_xarrays(cfg, np.array([0.0, 0.003]), stacked, str(td))
    fields = xr.load_dataset(td / "binary" / "fields.xr", engine="h5netcdf")
    kfields = xr.load_dataset(td / "binary" / "k-fields.xr", engine="h5netcdf")
    return cfg, phi_k, fields, kfields


def _tol(dtype) -> float:
    return ROUNDOFF_EPS * np.finfo(dtype).eps


def _rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


@pytest.mark.parametrize("geometry", ["2d", "1d"])
def test_saved_epw_fields_single_mode(geometry, tmp_path):
    """Single grid mode phi_k[mx, my] = A nx ny, i.e. phi = A exp(i (kx (x - x0) + ky (y - y0))).

    fields.xr: phi = phi(x), ex = -i kx phi(x), ey = -i ky phi(x) (ey = 0 in quasi-1D);
    k-fields.xr (fftshifted): phi_k, -i kx phi_k, -i ky phi_k.
    Bound: max |got - expected| <= ROUNDOFF_EPS eps |k| A in x-space (|k| = sqrt(kx^2 + ky^2)),
    <= ROUNDOFF_EPS eps |k| A nx ny in k-space.
    """
    cfg, _, fields, kfields = _saved_fields(geometry, tmp_path)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    mx, my = MODE[geometry]
    kx, ky = cfg["grid"]["kx"][mx], cfg["grid"]["ky"][my]
    k_abs = np.hypot(kx, ky)
    np.testing.assert_allclose(kx, 2.0 * np.pi * mx / (cfg["grid"]["xmax"] - cfg["grid"]["xmin"]), rtol=1e-14)

    i, j = np.arange(nx)[:, None], np.arange(ny)[None, :]
    phi_x = AMPLITUDE * np.exp(2j * np.pi * (mx * i / nx + my * j / ny))
    expected_x = {"phi": phi_x, "ex": -1j * kx * phi_x, "ey": -1j * ky * phi_x}
    for name, expected in expected_x.items():
        got = fields[name].values[0]
        assert got.dtype == np.complex128, name
        scale = AMPLITUDE * (1.0 if name == "phi" else k_abs)
        err = np.max(np.abs(got - expected))
        assert err <= _tol(got.dtype) * scale, f"fields.xr {name}: max err {err:.3e}, scale {scale:.3e}"

    expected_k = {"phi": 1.0, "ex": -1j * kx, "ey": -1j * ky}
    for name, factor in expected_k.items():
        got_k = np.fft.ifftshift(kfields[name].values[0])
        expected = np.zeros((nx, ny), dtype=np.complex128)
        expected[mx, my] = factor * AMPLITUDE * nx * ny
        scale = AMPLITUDE * nx * ny * (1.0 if name == "phi" else k_abs)
        err = np.max(np.abs(got_k - expected))
        assert err <= _tol(got_k.dtype) * scale, f"k-fields.xr {name}: max err {err:.3e}, scale {scale:.3e}"


@pytest.mark.parametrize("geometry", ["2d", "1d"])
def test_saved_epw_fields_are_the_solver_field(geometry, tmp_path):
    """Both time slices (single mode and random band-limited phi_k):

    * saved ex, ey = -d(phi)/dx, -d(phi)/dy of the saved phi, differentiated spectrally on the grid;
    * saved ex, ey = SpectralEPWSolver.phi_k_to_e_fields(phi_k), the field the solver evolves;
    * saved ex, ey = ifft2 of the k-fields.xr ex, ey (the two datasets agree);
    * dx dy sum(|ex|^2 + |ey|^2) of the saved fields = the e_sq series of the default save.
    Bound: relative L2 error (relative error for e_sq) <= ROUNDOFF_EPS eps.
    """
    from adept._lpse2d.core.epw import SpectralEPWSolver

    cfg, phi_k, fields, kfields = _saved_fields(geometry, tmp_path)
    kx = cfg["grid"]["kx"][:, None]
    ky = cfg["grid"]["ky"][None, :]
    solver = SpectralEPWSolver(cfg)
    series_func = cfg["save"]["default"]["func"]

    for it in range(phi_k.shape[0]):
        ex = fields["ex"].values[it]
        ey = fields["ey"].values[it]
        tol = _tol(ex.dtype)

        phi_saved_k = np.fft.fft2(fields["phi"].values[it])
        grad_ex = np.fft.ifft2(-1j * kx * phi_saved_k)
        grad_ey = np.fft.ifft2(-1j * ky * phi_saved_k)
        solver_ex, solver_ey = (np.asarray(e) for e in solver.phi_k_to_e_fields(jnp.asarray(phi_k[it])))
        k_ex = np.fft.ifft2(np.fft.ifftshift(kfields["ex"].values[it]))
        k_ey = np.fft.ifft2(np.fft.ifftshift(kfields["ey"].values[it]))

        references = {"-grad(saved phi)": (grad_ex, grad_ey), "solver E": (solver_ex, solver_ey)}
        references["ifft2(k-fields)"] = (k_ex, k_ey)
        for label, (ref_x, ref_y) in references.items():
            err_x = _rel_l2(ex, ref_x)
            assert err_x <= tol, f"slice {it}: ex vs {label}: relative L2 err {err_x:.3e}"
            if np.any(ref_y != 0):
                err_y = _rel_l2(ey, ref_y)
                assert err_y <= tol, f"slice {it}: ey vs {label}: relative L2 err {err_y:.3e}"
            else:  # quasi-1D: ky = 0 everywhere
                assert np.all(ey == 0), f"slice {it}: ey vs {label}: expected exactly 0"

        state = {
            "epw": jnp.asarray(phi_k[it]).view(jnp.float64),
            "E0": jnp.zeros((*phi_k.shape[1:], 2), dtype=jnp.complex128).view(jnp.float64),
        }
        state["E1"] = state["E0"]
        e_sq_series = float(series_func(0.0, state, None)["e_sq"])
        e_sq_saved = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2) * cfg["grid"]["dx"] * cfg["grid"]["dy"])
        err = abs(e_sq_saved - e_sq_series) / e_sq_series
        assert err <= tol, f"slice {it}: e_sq from fields.xr {e_sq_saved:.6e} vs series {e_sq_series:.6e}"
