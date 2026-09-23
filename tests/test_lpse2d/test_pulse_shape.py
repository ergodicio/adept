"""The pump pulse shape (drivers.E0.pulse_*) against LPSE's ``laser.pulseShape``.

LPSE's shape is a power factor: every injector multiplies its field by ``sqrt(shape)``
(``SchrodingerSolver3::addInjectorSources``) and the static pump's field scales as
``sqrt(max(shape, 1e-12))`` (``LightSolver::applyPulseShapeStatic``). The reference below is a
line-by-line transcription of ``LightSolver::pulseShape`` (LightSolver.cpp:238-308).

A ramp table 0 -> 1 over 0.2 ps gives power 0.25 at 0.05 ps, where the field factor is 0.5; the
earlier amplitude reading of the table gave 0.25, so every path check below separates the two.
Tolerances are fixed in advance: 1e-12 relative on the shape itself, 1e-9 on source ratios.
"""

import math
from copy import deepcopy

import jax
import numpy as np
import pytest
import yaml
from jax import numpy as jnp

RAMP = [[0.0, 0.0], [0.2, 1.0], [2.0, 1.0]]
T_QUARTER = 0.05  # ps: RAMP's power is 0.25 here


def _lpse_pulse_shape(t, shape, table=None, period=0.1, duty=0.5):
    """LightSolver::pulseShape, transcribed (time in ps; the square wave's static state machine
    written as the periodic function it produces)."""
    if shape == "file":
        times, scales = [r[0] for r in table], [r[1] for r in table]
        if len(times) == 1 or t <= times[0]:
            return scales[0]
        i = 0
        while i < len(times) and not t < times[i]:
            i += 1
        if i >= len(times):
            return scales[-1]
        a1 = min(max((t - times[i - 1]) / (times[i] - times[i - 1]), 0.0), 1.0)
        return (1.0 - a1) * scales[i - 1] + a1 * scales[i]
    if shape == "square":
        return 1.0 / duty if math.fmod(t, period) <= duty * period else 0.0
    if shape == "sin":
        return 2.0 * math.sin(math.pi * t / period) ** 2
    raise ValueError(shape)


def _finish(cfg):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _raw(light_solver, *, template="srs.yaml", pump_depletion=True, **e0):
    with open(f"tests/test_lpse2d/configs/{template}") as fi:
        raw = deepcopy(yaml.safe_load(fi))
    raw["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "6.4um", "ymin": "-6.4um", "dx": "0.1um"})
    raw["terms"]["light"] = {"solver": light_solver, "pump_depletion": pump_depletion}
    raw["terms"]["epw"]["source"]["noise"] = False
    raw["terms"]["epw"]["boundary"] = {"x": "absorbing" if pump_depletion else "periodic", "y": "periodic"}
    raw["drivers"]["E0"].update(e0)
    return raw


def _pump_args(cfg):
    from adept._lpse2d.modules.driver import UniformDriver

    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    return args["drivers"]["E0"]


def _ramp_file(tmp_path):
    path = tmp_path / "pulseShape.dat"
    path.write_text("# t_ps scale\n// LPSE also skips C++ comments\n" + "".join(f"{t} {s}\n" for t, s in RAMP))
    return str(path)


@pytest.mark.parametrize(
    "shape, extra",
    [("file", {}), ("square", {"pulse_period": "0.1ps", "pulse_duty_cycle": 0.3}), ("sin", {"pulse_period": "0.4ps"})],
)
def test_power_factor_is_lpse_pulse_shape(tmp_path, shape, extra):
    from adept._lpse2d.core.pulse import PulseShape

    e0 = {"pulse_file": _ramp_file(tmp_path)} if shape == "file" else {"pulse_shape": shape, **extra}
    cfg = _finish(_raw("spectral", **e0))
    pulse = PulseShape(cfg["drivers"]["E0"]["derived"])
    period = {"square": 0.1, "sin": 0.4}.get(shape, 0.1)
    duty = extra.get("pulse_duty_cycle", 0.5)
    # avoid the square wave's switching instants, where LPSE's state machine lags by a step
    times = [t for t in np.linspace(0.0, 2.5, 1001) if shape != "square" or abs(math.fmod(t, 0.1) - 0.03) > 1e-9]
    got = np.array([float(pulse.power(t)) for t in times])
    want = np.array([_lpse_pulse_shape(t, shape, RAMP, period, duty) for t in times])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def _source_ratio(make, t):
    """|source with the RAMP pulse| / |source without| on the source's support."""
    with_pulse, without = make(True), make(False)
    num, den = np.abs(np.asarray(with_pulse(t))), np.abs(np.asarray(without(t)))
    support = den > 1e-6 * den.max()
    return num[support] / den[support]


def test_spectral_injector_carries_sqrt_power(tmp_path):
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight

    pulse_file = _ramp_file(tmp_path)

    def make(pulsed):
        cfg = _finish(_raw("spectral", **({"pulse_file": pulse_file} if pulsed else {})))
        light, pa = SpectralCoupledLight(cfg), _pump_args(cfg)
        return lambda t: light.calc_pump_source(t, pa)

    np.testing.assert_allclose(_source_ratio(make, T_QUARTER), 0.5, rtol=1e-9)


def test_fd_row_injector_carries_sqrt_power(tmp_path):
    from adept._lpse2d.core.light import CoupledLight

    pulse_file = _ramp_file(tmp_path)

    def make(pulsed):
        cfg = _finish(_raw("fd", **({"pulse_file": pulse_file} if pulsed else {})))
        light, pa = CoupledLight(cfg), _pump_args(cfg)
        assert not light.fd_general_injector
        return lambda t: jnp.stack([v for _, v in light.calc_pump_source(t, pa)])

    np.testing.assert_allclose(_source_ratio(make, T_QUARTER), 0.5, rtol=1e-9)


def test_fd_general_injector_carries_sqrt_power(tmp_path):
    from adept._lpse2d.core.light import CoupledLight

    pulse_file = _ramp_file(tmp_path)

    def make(pulsed):
        cfg = _finish(_raw("fd", angle=20.0, **({"pulse_file": pulse_file} if pulsed else {})))
        light, pa = CoupledLight(cfg), _pump_args(cfg)
        assert light.fd_general_injector
        patterns = light.pump_pattern(pa)
        return lambda t: jnp.stack([v for _, v in light.general_pump_rows(t, pa, patterns)])

    np.testing.assert_allclose(_source_ratio(make, T_QUARTER), 0.5, rtol=1e-9)


def test_fd_file_injector_shares_the_time_factor(tmp_path):
    """The file injector's rows are the stored pattern times ``pump_time_factor`` (LPSE applies
    temporalSourceAmplitudeMultiplier to loaded injectors as well), and a pulse shape is
    accepted together with drivers.E0.injector_file."""
    from adept._lpse2d.core.light import CoupledLight

    pulse_file = _ramp_file(tmp_path)
    with_pulse = CoupledLight(_finish(_raw("fd", pulse_file=pulse_file)))
    without = CoupledLight(_finish(_raw("fd")))
    pa = _pump_args(_finish(_raw("fd")))
    ratio = float(with_pulse.pump_time_factor(T_QUARTER, pa)) / float(without.pump_time_factor(T_QUARTER, pa))
    assert ratio == pytest.approx(0.5, rel=1e-9)
    assert float(with_pulse.pump_time_factor(0.0, pa)) == 0.0


def test_combined_injector_carries_sqrt_power(tmp_path):
    from adept._lpse2d.core.combined import CombinedSolver

    pulse_file = _ramp_file(tmp_path)

    def make(pulsed):
        raw = _raw("spectral", template="tpd.yaml", **({"pulse_file": pulse_file} if pulsed else {}))
        raw["density"] = {"basis": "uniform", "val": 0.22}
        raw["terms"]["epw"]["solver"] = "combined"
        raw["terms"]["epw"]["source"].update({"tpd": True, "srs": True})
        cfg = _finish(raw)
        solver = CombinedSolver(cfg)
        ny = cfg["grid"]["ny"]
        pa = {
            **cfg["drivers"]["E0"]["derived"],
            "delta_omega": jnp.zeros(1),
            "intensities": jnp.ones((1, ny)),
            "phases": jnp.zeros((1, ny)),
        }
        return lambda t: solver.calc_pump_source(t, pa)

    np.testing.assert_allclose(_source_ratio(make, T_QUARTER), 0.5, rtol=1e-9)


def test_static_pump_scales_as_sqrt_power_with_lpse_floor(tmp_path):
    from adept._lpse2d.core.laser import Light

    pulse_file = _ramp_file(tmp_path)
    cfgs = {
        p: _finish(_raw("spectral", pump_depletion=False, **({"pulse_file": pulse_file} if p else {}))) for p in (1, 0)
    }
    ny = cfgs[1]["grid"]["ny"]
    wave = {"delta_omega": jnp.array([0.0]), "intensities": jnp.ones((1, ny)), "phases": jnp.zeros((1, ny))}
    e_pulse = np.abs(np.asarray(Light(cfgs[1]).laser_update(T_QUARTER, None, wave)))
    e_flat = np.abs(np.asarray(Light(cfgs[0]).laser_update(T_QUARTER, None, wave)))
    support = e_flat > 1e-6 * e_flat.max()
    np.testing.assert_allclose(e_pulse[support] / e_flat[support], 0.5, rtol=1e-9)
    # at zero power LPSE keeps the static field at sqrt(1e-12) of its amplitude
    e_zero = np.abs(np.asarray(Light(cfgs[1]).laser_update(0.0, None, wave)))
    np.testing.assert_allclose(e_zero[support] / e_flat[support], 1e-6, rtol=1e-9)


def test_field_factor_has_a_finite_gradient_at_zero_power(tmp_path):
    from adept._lpse2d.core.pulse import PulseShape

    cfg = _finish(_raw("spectral", pulse_file=_ramp_file(tmp_path)))
    pulse = PulseShape(cfg["drivers"]["E0"]["derived"])
    assert float(pulse.field_factor(0.0)) == 0.0
    assert np.isfinite(float(jax.grad(pulse.field_factor)(0.0)))


@pytest.mark.parametrize(
    "rows, message",
    [("0.1 1\n0.0 1\n", "strictly increasing"), ("0 11\n", r"\[0, 10\]"), ("0 1 2\n", "two columns")],
)
def test_pulse_file_is_validated_like_lpse(tmp_path, rows, message):
    path = tmp_path / "bad.dat"
    path.write_text(rows)
    with pytest.raises(ValueError, match=message):
        _finish(_raw("spectral", pulse_file=str(path)))


def _translate(tmp_path, pulse_lines):
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    deck = tmp_path / "lpse.parms"
    deck.write_text(
        "grid.sizes = 20 10;\ngrid.nodes = 200 100;\nsimulation.time.end = 1;\nlaser.enable = true;\n"
        "laser.nBeams = 1;\nlaser.1.intensity = 1e15;\n" + pulse_lines
    )
    return translate_parms(parse_parms(deck), run="x")


def test_translator_maps_every_lpse_pulse_shape(tmp_path):
    cfg, _ = _translate(
        tmp_path, "laser.pulseShape.enable = true;\nlaser.pulseShape.shape = sin;\nlaser.pulseShape.period = 0.4;\n"
    )
    assert cfg["drivers"]["E0"]["pulse_shape"] == "sin" and cfg["drivers"]["E0"]["pulse_period"] == "0.4ps"
    cfg, _ = _translate(
        tmp_path,
        "laser.pulseShape.enable = true;\nlaser.pulseShape.shape = square;\nlaser.pulseShape.dutyCycle = 0.25;\n",
    )
    assert cfg["drivers"]["E0"]["pulse_shape"] == "square" and cfg["drivers"]["E0"]["pulse_duty_cycle"] == 0.25
    with pytest.raises(ValueError, match=r"needs laser\.pulseShape\.file"):
        _translate(tmp_path, "laser.pulseShape.enable = true;\n")
    cfg, report = _translate(tmp_path, "laser.pulseShape.enable = false;\nlaser.pulseShape.file = x.dat;\n")
    assert "pulse_file" not in cfg["drivers"]["E0"] and "pulse_shape" not in cfg["drivers"]["E0"]
    _, report = _translate(tmp_path, "raman.pulseShape.enable = true;\n")
    assert any("raman.pulseShape" in u for u in report["unsupported"])
