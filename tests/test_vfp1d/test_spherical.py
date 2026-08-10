"""Tests for the spherical (radial) geometry of the VFP-1D solver, the file-based
profile initialization, and the Spitzer-Härm / SNB heat-flux diagnostics."""

import os

import numpy as np
import pytest
import yaml

from adept import ergoExo
from adept.normalization import UREG, laser_normalization

TEST_DIR = os.path.join(os.getcwd(), "tests", "test_vfp1d")


def _load_cfg(config_name: str) -> dict:
    with open(os.path.join(TEST_DIR, f"{config_name}.yaml")) as fi:
        return yaml.safe_load(fi)


def test_spherical_uniform_preservation():
    """A uniform plasma in spherical geometry must stay uniform: the geometric
    2*f10/r term in the f0 equation only acts on a nonzero radial flux."""
    cfg = _load_cfg("spherical-uniform")

    exo = ergoExo()
    exo.setup(cfg)
    _, datasets, _ = exo(None)

    dataT = datasets["fields"]["fields-T keV"].data
    datan = datasets["fields"]["fields-n n_c"].data

    # spatially uniform at the end and unchanged from the start
    np.testing.assert_allclose(dataT[-1], np.mean(dataT[-1]), rtol=1e-6)
    np.testing.assert_allclose(np.mean(dataT[-1]), np.mean(dataT[0]), rtol=1e-6)
    np.testing.assert_allclose(datan[-1], np.mean(datan[-1]), rtol=1e-6)


def test_spherical_hotspot():
    """A central hotspot in spherical geometry: heat must flow outward (q >= 0),
    volume-weighted energy must be conserved, and the heat-flux comparison
    diagnostic must produce all three fluxes."""
    cfg = _load_cfg("spherical-hotspot")

    exo = ergoExo()
    exo.setup(cfg)
    _, datasets, _ = exo(None)

    fields = datasets["fields"]
    r = fields.coords["x (um)"].data
    U = fields["fields-U a.u."].data

    # volume-weighted energy conservation (reflective walls, no drivers); use the
    # exact finite-volume cell volumes so the divergence telescopes
    dr = r[1] - r[0]
    r_edge = np.concatenate([[0.0], 0.5 * (r[1:] + r[:-1]), [r[-1] + dr / 2]])
    vols = np.diff(r_edge**3) / 3.0
    energy = np.sum(U * vols[None, :], axis=1)
    np.testing.assert_allclose(energy[-1], energy[0], rtol=1e-3)

    # outward heat flow in the gradient region
    q = fields["fields-q a.u."].data[-1]
    assert q[np.argmax(np.abs(q))] > 0.0

    hf = datasets["heat_flux"]
    for k in ["q_kinetic", "q_spitzer_harm", "q_snb"]:
        assert np.all(np.isfinite(hf[k].data))

    # both model fluxes should peak outward as well and be within an order of
    # magnitude of the kinetic flux at the kinetic peak
    ipk = np.argmax(np.abs(hf["q_kinetic"].data[-1]))
    q_kin = hf["q_kinetic"].data[-1][ipk]
    for k in ["q_spitzer_harm", "q_snb"]:
        ratio = hf[k].data[-1][ipk] / q_kin
        assert 0.1 < ratio < 10.0, f"{k}/q_kinetic = {ratio}"


def test_snb_local_limit():
    """In the highly collisional (local) limit the SNB flux must reduce to the
    Spitzer-Härm flux; in the long-mean-free-path limit it must be inhibited."""
    from adept.vfp1d.heat_flux import snb_heat_flux

    nx = 200
    dx = 1.0
    x = np.linspace(dx / 2, nx * dx - dx / 2, nx)

    vth_norm = 0.05
    T0 = vth_norm**2 / 2.0
    n = np.ones(nx)
    Z = np.ones(nx)
    ni = np.ones(nx)
    T = T0 * (1.0 + 0.01 * np.cos(2 * np.pi * x / (nx * dx)))

    cfg = {
        "units": {
            "Z": 1,
            "derived": {
                "ne": UREG.Quantity(1.5e21, "1/cc"),
                "n0": UREG.Quantity(1.5e21, "1/cc"),
                "vth_norm": vth_norm,
                "nuei_epphaines_norm": UREG.Quantity(1e-3, ""),
                "logLam_ratio": 1.0,
                "nuee_coeff": None,  # set per case below
            },
        }
    }

    # collisional: group mfps << both the cell size and the gradient length
    cfg["units"]["derived"]["nuee_coeff"] = 1e3
    res = snb_heat_flux(x, dx, n, T, Z, ni, cfg)
    np.testing.assert_allclose(res["q_snb"], res["q_sh"], rtol=0, atol=2e-2 * np.max(np.abs(res["q_sh"])))

    # nonlocal: mfps comparable to the gradient length -> flux inhibition
    cfg["units"]["derived"]["nuee_coeff"] = 1e-4
    res_nl = snb_heat_flux(x, dx, n, T, Z, ni, cfg)
    ipk = np.argmax(np.abs(res_nl["q_sh"]))
    assert abs(res_nl["q_snb"][ipk]) < abs(res_nl["q_sh"][ipk])


def test_file_basis_profiles(tmp_path):
    """Profiles loaded from CSV files must end up in the initial state with the
    right normalizations, including the ion mass-density -> ni, Z conversion."""
    from adept.vfp1d.base import BaseVFP1D

    r_um = np.linspace(0.0, 125.0, 100)
    ne_cc = 1.5e21 * (1.0 + 0.5 * np.exp(-((r_um / 50.0) ** 2)))
    Te_eV = 300.0 * (1.0 - 0.3 * r_um / 125.0)
    A_ion = 9.0122  # beryllium
    Z_ion = 4.0
    m_p_g = float(UREG.Quantity(1.0, "amu").to("g").magnitude)
    rho_gcc = ne_cc / Z_ion * A_ion * m_p_g

    files = {}
    for name, vals in [("ne", ne_cc), ("Te", Te_eV), ("rho", rho_gcc)]:
        path = tmp_path / f"{name}.csv"
        np.savetxt(path, np.stack([r_um, vals], axis=1), delimiter=",", header="r_um,value")
        files[name] = str(path)

    cfg = _load_cfg("spherical-uniform")
    cfg["units"]["Z"] = Z_ion
    cfg["density"]["species-background"]["n"] = {
        "basis": "file",
        "path": files["ne"],
        "units": "1/cm^3",
        "coordinate_units": "um",
    }
    cfg["density"]["species-background"]["T"] = {
        "basis": "file",
        "path": files["Te"],
        "units": "eV",
        "coordinate_units": "um",
    }
    cfg["density"]["ion"] = {
        "A": A_ion,
        "mass_density": {
            "basis": "file",
            "path": files["rho"],
            "units": "g/cm^3",
            "coordinate_units": "um",
        },
    }

    module = BaseVFP1D(cfg)
    module.write_units()
    module.init_state_and_args()

    grid = module.grid
    state = module.state

    # density moment of f0 should follow ne(r) (in units of n0)
    n0_cc = float(module.plasma_norm.n0.to("1/cc").magnitude)
    n_from_f0 = 4 * np.pi * np.sum(np.asarray(state["f0"]) * np.asarray(grid.v) ** 2, axis=1) * grid.dv
    r_grid_um = np.asarray(grid.x) * float(module.plasma_norm.L0.to("um").magnitude)
    ne_expected = np.interp(r_grid_um, r_um, ne_cc) / n0_cc
    np.testing.assert_allclose(n_from_f0, ne_expected, rtol=2e-2)

    # rho was constructed for uniform Z_ion, so the relative Z must be ~1
    np.testing.assert_allclose(np.asarray(state["Z"]), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(state["ni"]), ne_expected / Z_ion, rtol=2e-2)


def test_ion_mixture(tmp_path):
    """A fully ionized CD mixture must be represented by an effective ion that
    preserves quasineutrality (Z*ni = ne) and e-i collisionality (Z^2*ni = Zeff*ne)."""
    from adept.vfp1d.base import BaseVFP1D

    r_um = np.linspace(0.0, 125.0, 100)
    ne_cc = 1.0e21 * np.ones_like(r_um)
    zbar, zeff, abar = 3.0, 5.0, 6.304
    m_u_g = float(UREG.Quantity(1.0, "amu").to("g").magnitude)
    rho_gcc = ne_cc / zbar * abar * m_u_g

    files = {}
    for name, vals in [("ne", ne_cc), ("rho", rho_gcc)]:
        path = tmp_path / f"{name}.csv"
        np.savetxt(path, np.stack([r_um, vals], axis=1), delimiter=",", header="r_um,value")
        files[name] = str(path)

    cfg = _load_cfg("spherical-uniform")
    cfg["units"]["Z"] = zeff
    cfg["units"]["reference electron density"] = "1.0e21/cm^3"
    cfg["units"]["logLambda"] = "lee-more"
    cfg["density"]["species-background"]["n"] = {
        "basis": "file",
        "path": files["ne"],
        "units": "1/cm^3",
        "coordinate_units": "um",
    }
    cfg["density"]["ion"] = {
        "A": abar,
        "mixture": [
            {"Z": 6, "A": 12.011, "fraction": 0.4},
            {"Z": 1, "A": 2.0141, "fraction": 0.6},
        ],
        "mass_density": {
            "basis": "file",
            "path": files["rho"],
            "units": "g/cm^3",
            "coordinate_units": "um",
        },
    }

    module = BaseVFP1D(cfg)
    module.write_units()
    module.init_state_and_args()
    state = module.state

    Z_phys = np.asarray(state["Z"]) * cfg["units"]["Z"]
    np.testing.assert_allclose(Z_phys, zeff, rtol=1e-12)

    # quasineutrality: Z * ni = ne
    n0_cc = float(module.plasma_norm.n0.to("1/cc").magnitude)
    ne_expected = ne_cc[0] / n0_cc
    np.testing.assert_allclose(Z_phys * np.asarray(state["ni"]), ne_expected, rtol=1e-6)

    # e-i collisionality: Z^2 * ni = Zeff * ne = sum_i n_i Z_i^2
    np.testing.assert_allclose(Z_phys**2 * np.asarray(state["ni"]), zeff * ne_expected, rtol=1e-6)


def test_lee_more_loglambda():
    """Lee-More Coulomb logarithm: sensible values at conduction-zone conditions,
    close to NRL there, and floored at 2 for cold dense plasma."""
    from adept.vfp1d.helpers import calc_logLambda

    def _cfg(loglam, Ti="1000eV"):
        return {"units": {"logLambda": loglam, "reference ion temperature": Ti}}

    ne = UREG.Quantity(1.0e21, "1/cc")
    Te = UREG.Quantity(1000.0, "eV")
    lm_ei, lm_ee = calc_logLambda(_cfg("lee-more"), ne, Te, Z=5, ion_species="CD")
    assert lm_ee == lm_ei
    assert 5.0 < lm_ei < 9.0

    nrl_ei, _ = calc_logLambda(_cfg("nrl"), ne, Te, Z=5, ion_species="CD")
    assert abs(lm_ei - nrl_ei) / nrl_ei < 0.25

    # cold, dense -> hits the floor of 2
    lm_cold, _ = calc_logLambda(
        _cfg("lee-more", Ti="1eV"), UREG.Quantity(1.0e24, "1/cc"), UREG.Quantity(1.0, "eV"), Z=5, ion_species="CD"
    )
    assert lm_cold == 2.0


def test_spherical_config_validation():
    """Spherical geometry requires reflective boundaries; xmin > 0 is a valid
    annular domain with dx = (xmax - xmin)/nx."""
    from adept.vfp1d.grid import Grid

    cfg = _load_cfg("spherical-uniform")
    norm = laser_normalization(cfg["units"]["laser_wavelength"], cfg["units"]["reference electron temperature"])

    cfg["grid"]["boundary"] = "periodic"
    with pytest.raises(ValueError):
        Grid.from_config(cfg["grid"], norm)

    # annular domain
    cfg["grid"]["boundary"] = "reflective"
    cfg["grid"]["xmin"] = "25.0um"
    grid = Grid.from_config(cfg["grid"], norm)
    np.testing.assert_allclose(grid.dx, (grid.xmax - grid.xmin) / grid.nx)
    np.testing.assert_allclose(np.asarray(grid.x_edge)[0], grid.xmin)
