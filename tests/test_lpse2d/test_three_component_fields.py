"""Three-component light fields (plan 2 F.1): E0 and E1 carry (x, y, z).

The grid stays 2-D (k_z = 0), so E_z is purely transverse, never enters the EPW
potential and never drives TPD; it is what an s-polarised beam or seed lives in. With
E_z = 0 every solver path must reproduce the two-component code bit for bit.
"""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._lpse2d.core.vector import fft2c, split_k, transverse_part, with_components


def _finish(cfg):
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _cfg(*, epw_solver="separate", light_solver="fd", pump_depletion=True, tpd=False, srs=True, iaw=False, density=0.2):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": density}
    cfg["units"]["envelope density"] = 0.25 if tpd else density
    cfg["drivers"]["E0"]["envelope"]["tc"] = "19.95ps"
    cfg["grid"].update(
        {
            "boundary_width": "0.4um",
            "dt": "1fs",
            "dx": "0.04um",
            "xmax": "2.56um",
            "tmax": "0.1ps",
            "ymax": "0.32um",
            "ymin": "-0.32um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["solver"] = epw_solver
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": tpd, "srs": srs})
    cfg["terms"]["light"] = {"solver": light_solver, "pump_depletion": pump_depletion}
    if iaw:
        cfg["terms"]["iaw"] = {"active": True, "solver": "spectral"}
    return _finish(cfg)


def _seed(cfg, nc, seed=11, relative_amplitude=1.0e-3):
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k *= np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    field = np.stack([np.fft.ifft2(-1j * kx * phi_k), np.fft.ifft2(-1j * ky * phi_k)], -1)
    scale = relative_amplitude * cfg["units"]["derived"]["E0_source"] / np.abs(field).max()
    phi_k, field = phi_k * scale, field * scale
    state = {
        "epw": jnp.asarray(phi_k),
        "E0": jnp.zeros((nx, ny, nc), dtype=jnp.complex128),
        "E1": jnp.zeros((nx, ny, nc), dtype=jnp.complex128),
    }
    if cfg["terms"]["epw"].get("solver") == "combined":
        state["E1"] = with_components(jnp.asarray(field), nc)
    if cfg["terms"].get("iaw", {}).get("active", False):
        state["iaw_density"] = jnp.zeros((nx, ny))
        state["iaw_velocity_divergence"] = jnp.zeros((nx, ny))
    return {k: v.view(jnp.float64) if jnp.iscomplexobj(v) else v for k, v in state.items()}


def _run(cfg, nc, n_steps=12):
    from adept._lpse2d.core.vector_field import SplitStep

    step = SplitStep(cfg)
    state = _seed(cfg, nc)
    ny = cfg["grid"]["ny"]
    pump = {**cfg["drivers"]["E0"]["derived"], "delta_omega": jnp.zeros(1), "phases": jnp.zeros((1, ny))}
    args = {"drivers": {"E0": {**pump, "intensities": jnp.ones((1, ny))}}}
    for i in range(n_steps):
        state = step(i * cfg["grid"]["dt"], dict(state), args)
    return {k: np.asarray(v.view(jnp.complex128) if k in ("E0", "E1", "epw") else v) for k, v in state.items()}


PATHS = {
    "separate/fd": dict(light_solver="fd"),
    "separate/spectral": dict(light_solver="spectral"),
    "separate/static": dict(light_solver="fd", pump_depletion=False),
    "combined/spectral": dict(epw_solver="combined", light_solver="spectral", tpd=True, srs=True, density=0.22),
    "combined/spectral+iaw": dict(
        epw_solver="combined", light_solver="spectral", tpd=True, srs=True, density=0.22, iaw=True
    ),
}


@pytest.mark.parametrize("path", list(PATHS))
def test_three_components_with_zero_ez_are_bit_identical_to_two(path):
    cfg = _cfg(**PATHS[path])
    two = _run(cfg, 2)
    three = _run(cfg, 3)
    for k in two:
        if k in ("E0", "E1"):
            np.testing.assert_array_equal(three[k][..., :2], two[k], err_msg=k)
            assert np.all(three[k][..., 2] == 0.0), k
        else:
            np.testing.assert_array_equal(three[k], two[k], err_msg=k)
    assert np.any(three["E0"] != 0.0) and np.any(three["epw"] != 0.0)


def test_projectors_treat_ez_as_transverse():
    cfg = _cfg()
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx, ky = jnp.asarray(cfg["grid"]["kx"]), jnp.asarray(cfg["grid"]["ky"])
    k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
    one_over = jnp.where(k_sq > 0, 1.0 / jnp.where(k_sq > 0, k_sq, 1.0), 0.0)
    rng = np.random.default_rng(3)
    field = jnp.asarray(rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3)))
    longitudinal_k, transverse_k = split_k(fft2c(field), kx, ky, one_over)
    assert np.all(np.asarray(longitudinal_k)[..., 2] == 0.0)
    np.testing.assert_array_equal(np.asarray(transverse_k)[..., 2], np.asarray(fft2c(field))[..., 2])
    projected = transverse_part(field, kx, ky, one_over)
    np.testing.assert_allclose(np.asarray(projected)[..., 2], np.asarray(field)[..., 2], rtol=1e-12, atol=1e-12)
    # the in-plane projection is the two-component one
    two = transverse_part(field[..., :2], kx, ky, one_over)
    np.testing.assert_array_equal(np.asarray(projected)[..., :2], np.asarray(two))


def test_s_polarised_pump_drives_srs_but_not_tpd():
    """A pump along z (s-polarised, out of the 2-D plane) has no in-plane part to pair with
    the in-plane EPW field, so the TPD source is identically zero; the SRS source, which is
    E0 . E1*, is non-zero when the Raman light is also along z."""
    from adept._lpse2d.core.epw import SpectralEPWSolver

    # the separate solver takes one instability at a time (plan 2 N.4): one solver per source
    cfg = _cfg(tpd=True, srs=False, density=0.22)
    epw = SpectralEPWSolver(cfg)
    epw_srs = SpectralEPWSolver(_cfg(tpd=False, srs=True, density=0.22))
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(5)
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k = jnp.asarray(phi_k * np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"]))
    ex, ey = epw.phi_k_to_e_fields(phi_k)
    amp = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    E0_s = jnp.asarray(np.stack([np.zeros_like(amp), np.zeros_like(amp), amp], -1))
    E0_p = jnp.asarray(np.stack([np.zeros_like(amp), amp, np.zeros_like(amp)], -1))
    E1_s = jnp.asarray(np.stack([np.zeros_like(amp), np.zeros_like(amp), 0.3 * amp], -1))
    E1_p = jnp.asarray(np.stack([np.zeros_like(amp), 0.3 * amp, np.zeros_like(amp)], -1))

    tpd_s = np.asarray(epw.calc_tpd_source(0.1, phi_k, ex, ey, E0_s))
    tpd_p = np.asarray(epw.calc_tpd_source(0.1, phi_k, ex, ey, E0_p))
    assert np.all(tpd_s == 0.0)
    assert np.abs(tpd_p).max() > 0.0

    srs_s = np.asarray(epw_srs.calc_srs_source(E0_s, E1_s))
    srs_p = np.asarray(epw_srs.calc_srs_source(E0_p, E1_p))
    assert np.abs(srs_s).max() > 0.0
    # E0 . E1* is the same scalar whichever transverse direction carries the pair
    np.testing.assert_allclose(srs_s, srs_p, rtol=1e-12, atol=1e-12 * np.abs(srs_p).max())
    # crossed polarisations do not couple
    assert np.abs(np.asarray(epw_srs.calc_srs_source(E0_s, E1_p))).max() == 0.0


@pytest.mark.parametrize("solver", ["fd", "spectral"])
def test_ez_propagates_as_a_plane_wave(solver):
    """With no EPW the z component obeys the same paraxial equation as the in-plane wave:
    a rightward E0z plane wave advances at the group velocity without acquiring an in-plane
    part (k_z = 0: the curl-curl and the L-T projector leave it alone) and keeps its
    amplitude where it has not yet run into the absorber."""
    from adept._lpse2d.core.light import CoupledLight
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight

    cfg = _cfg(light_solver=solver)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    light = (CoupledLight if solver == "fd" else SpectralCoupledLight)(cfg)
    x = np.asarray(cfg["grid"]["x"])
    derived = cfg["units"]["derived"]
    dk = 2.0 * np.pi / (nx * cfg["grid"]["dx"])
    k0 = np.round(derived["w0"] / derived["c"] * np.sqrt(1.0 - 0.2) / dk) * dk  # a grid mode: periodic
    E0 = np.zeros((nx, ny, 3), dtype=np.complex128)
    E0[..., 2] = (1.0e-6 * np.exp(1j * k0 * x))[:, None]
    E1 = jnp.zeros((nx, ny, 3), dtype=jnp.complex128)
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128)
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.zeros((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    out, _ = light(0.0, jnp.asarray(E0), E1, phi_k, pump_args, None, None)
    out = np.asarray(out)
    assert np.all(out[..., :2] == 0.0)
    # one EPW step: the wave moved v_g dt = 0.24 um; the band 1.0 < x < 1.6 um was fed from
    # the interior and is still clear of the absorbers
    travelled = derived["c"] ** 2 * k0 / derived["w0"] * cfg["grid"]["dt"]
    assert 0.2 < travelled < 0.3
    band = (x > 1.0) & (x < 1.6)
    np.testing.assert_allclose(np.abs(out[band, :, 2]), 1.0e-6, rtol=2e-2)
    # and the same in-plane wave (E0y) behaves identically under either solver
    E0p = np.zeros((nx, ny, 3), dtype=np.complex128)
    E0p[..., 1] = E0[..., 2]
    outp, _ = light(0.0, jnp.asarray(E0p), E1, phi_k, pump_args, None, None)
    np.testing.assert_allclose(np.asarray(outp)[..., 1], out[..., 2], rtol=1e-12, atol=1e-18)


# ----------------------------------------------------------- beam polarization (plan 2 F.2) --


def _cfg_pol(polarization, **kw):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["drivers"]["E0"]["polarization"] = polarization
    cfg["density"] = {"basis": "uniform", "val": kw.get("density", 0.2)}
    cfg["units"]["envelope density"] = 0.25 if kw.get("tpd") else kw.get("density", 0.2)
    cfg["drivers"]["E0"]["envelope"]["tc"] = "19.95ps"
    cfg["grid"].update(
        {
            "boundary_width": "0.4um",
            "dt": "1fs",
            "dx": "0.04um",
            "xmax": "2.56um",
            "tmax": "0.1ps",
            "ymax": "0.32um",
            "ymin": "-0.32um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["solver"] = kw.get("epw_solver", "separate")
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": kw.get("tpd", False), "srs": kw.get("srs", True)})
    cfg["terms"]["light"] = {"solver": kw.get("light_solver", "fd"), "pump_depletion": kw.get("pump_depletion", True)}
    if kw.get("angle"):
        cfg["drivers"]["E0"]["angle"] = kw["angle"]
    return _finish(cfg)


def _pump_after(cfg, n_steps=8, nc=3, relative_amplitude=0.0):
    """The pump (and Raman) fields after ``n_steps``; by default with no EPW seed, so only
    the injector and the propagators act."""
    from adept._lpse2d.core.vector_field import SplitStep

    step = SplitStep(cfg)
    state = _seed(cfg, nc, relative_amplitude=relative_amplitude)
    ny = cfg["grid"]["ny"]
    pump = {**cfg["drivers"]["E0"]["derived"], "delta_omega": jnp.zeros(1), "phases": jnp.zeros((1, ny))}
    args = {"drivers": {"E0": {**pump, "intensities": jnp.ones((1, ny))}}}
    for i in range(n_steps):
        state = step(i * cfg["grid"]["dt"], dict(state), args)
    return np.asarray(state["E0"].view(jnp.complex128)), np.asarray(state["E1"].view(jnp.complex128))


POL_PATHS = {
    "separate/fd": dict(light_solver="fd"),
    "separate/spectral": dict(light_solver="spectral"),
    "separate/static": dict(light_solver="fd", pump_depletion=False),
    "separate/static+oblique": dict(light_solver="fd", pump_depletion=False, angle=10.0),
    "combined/spectral": dict(epw_solver="combined", light_solver="spectral", tpd=True, srs=True, density=0.22),
}


@pytest.mark.parametrize("path", list(POL_PATHS))
def test_s_polarised_pump_lives_in_ez_on_every_injector(path):
    """``drivers.E0.polarization: 90`` (or ``s``) puts the launched pump entirely in E0z with
    the same amplitude the p-polarised launch puts in the in-plane transverse component."""
    kw = POL_PATHS[path]

    def cfg(pol):
        c = _cfg_pol(pol, **kw)
        # the same light step for both launches: the FD stability bound gives E_z (the 2-D Laplacian)
        # about twice the sub-steps of in-plane light (6a73f94), which alone moves the launch by ~0.3 %
        c["grid"]["light_substeps"] = 40
        return c

    e0_p, _ = _pump_after(cfg("p"))
    e0_s, _ = _pump_after(cfg("s"))
    e0_90, _ = _pump_after(cfg(90.0))
    assert np.abs(e0_p[..., 2]).max() == 0.0
    assert np.abs(e0_s[..., 2]).max() > 0.0
    np.testing.assert_array_equal(e0_s, e0_90)
    # the s launch carries what the p launch carries in the plane, rotated out of it
    in_plane_p = np.sqrt(np.abs(e0_p[..., 0]) ** 2 + np.abs(e0_p[..., 1]) ** 2)
    np.testing.assert_allclose(np.abs(e0_s[..., 2]), in_plane_p, rtol=1e-9, atol=1e-9 * in_plane_p.max())
    assert np.abs(e0_s[..., :2]).max() <= 1e-12 * in_plane_p.max()


def test_intermediate_polarization_splits_cos_sin():
    e0_p, _ = _pump_after(_cfg_pol("p", light_solver="spectral"))
    e0_30, _ = _pump_after(_cfg_pol(30.0, light_solver="spectral"))
    scale = np.abs(e0_p[..., 1]).max()
    np.testing.assert_allclose(e0_30[..., 1], np.cos(np.deg2rad(30.0)) * e0_p[..., 1], rtol=1e-12, atol=1e-13 * scale)
    np.testing.assert_allclose(e0_30[..., 2], np.sin(np.deg2rad(30.0)) * e0_p[..., 1], rtol=1e-12, atol=1e-13 * scale)


def test_s_polarised_pump_drives_no_tpd_in_a_run():
    """TPD on, SRS off, a pump along z: the EPW is never driven -- its energy after 40 steps
    equals the free-decay value bit for bit -- while the p-polarised pump drives it."""
    from adept._lpse2d.core.vector_field import SplitStep

    def final_energy(pol, intensity):
        cfg = _cfg_pol(pol, light_solver="spectral", tpd=True, srs=False, density=0.25)
        step = SplitStep(cfg)
        state = _seed(cfg, 3, relative_amplitude=1e-2)
        ny = cfg["grid"]["ny"]
        pump = {**cfg["drivers"]["E0"]["derived"], "delta_omega": jnp.zeros(1), "phases": jnp.zeros((1, ny))}
        args = {"drivers": {"E0": {**pump, "intensities": intensity * jnp.ones((1, ny))}}}
        for i in range(40):
            state = step(i * cfg["grid"]["dt"], dict(state), args)
        return float(step.epw.energy(state["epw"].view(jnp.complex128)))

    w_off = final_energy("p", 0.0)
    w_p = final_energy("p", 30.0)
    w_s = final_energy("s", 30.0)
    assert w_off > 0.0
    # to round-off: the source is identically zero, but XLA's fusion of the s-pol path can
    # round the free decay differently by an ulp or two (seen on the laptop, jax 0.9.0.1)
    assert np.isclose(w_s, w_off, rtol=1e-12, atol=0.0)
    assert not np.isclose(w_p, w_off, rtol=1e-6, atol=0.0)


def test_two_component_fields_refuse_an_s_polarised_pump():
    cfg = _cfg_pol("s", light_solver="fd")
    with pytest.raises(ValueError, match="three-component"):
        _pump_after(cfg, n_steps=1, nc=2)


def test_translator_maps_polarization_angles():
    from adept._lpse2d.lpse_deck import translate_parms
    from adept._lpse2d.parity import deck_path

    parms = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "laser.nBeams": "2",
        "laser.1.intensity": "1e15",
        "laser.1.polarization": "90",
        "laser.2.intensity": "1e15",
        "laser.2.polarization": "45",
        "laser.wavelength": "0.351",
        "raman.enable": "true",
        "raman.solver": "spectral",
        "raman.nBeams": "1",
        "raman.1.intensity": "1e12",
        "raman.1.polarization": "90",
        "lw.SRS.enable": "true",
        "lw.envelopeDensity": "0.2",
        "densityProfile.NminOverNc": "0.2",
        "densityProfile.NmaxOverNc": "0.2",
        "simulation.time.end": "1",
    }
    cfg, report = translate_parms(parms, run="pol")
    assert cfg["drivers"]["E0"]["polarization"] == 90.0
    assert [b["polarization"] for b in cfg["drivers"]["E0"]["beams"]] == [90.0, 45.0]
    assert cfg["drivers"]["E1"]["polarization"] == 90.0
    assert not any("polarization" in u for u in report["unsupported"])
    deck = deck_path("test_017")
    if deck is not None:
        from adept._lpse2d.lpse_deck import parse_parms

        cfg17, report17 = translate_parms(parse_parms(deck), run="test_017")
        assert cfg17["drivers"]["E0"]["polarization"] == 90.0
        assert not any("polarization" in u for u in report17["unsupported"])
