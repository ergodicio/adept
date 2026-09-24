"""Adaptive panel verification against an independently integrated linear IVP."""

from pathlib import Path

import jax
import numpy as np
import yaml

from adept import SimulationSpec, run_prepared, solver_registry
from tests.test_farsight1d.test_physics import _linear_reference


def test_adaptive_two_stream_example_matches_linear_fourier_history():
    """The shipped AMR example uses fewer active leaves than uniform finest level.

    Compare the full complex field history, retaining the initial transient.
    Maximum-level saturation is reported; this is not a tolerance-convergence
    or late-time turbulence test.
    """
    path = Path(__file__).parents[2] / "configs" / "farsight-1d" / "two-stream-amr.yaml"
    config = yaml.safe_load(path.read_text())
    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=7)
    completed = run_prepared(prepared, key=jax.random.key(7))
    fields = completed.report.result["fields"]
    scalars = completed.report.result["scalars"]
    grid, initial = config["grid"], config["initial"]
    mode = np.fft.rfft(fields.electric_field, axis=1)[:, initial["mode"]] / grid["nx"]
    reference, frequency, multiplier = _linear_reference(
        np.asarray(fields.t),
        wave_number=2 * np.pi * initial["mode"] / (grid["xmax"] - grid["xmin"]),
        epsilon=config["numerical"]["epsilon"],
        drift=initial["drift"],
        thermal_speed=initial["thermal_speed"],
        amplitude=initial["amplitude"],
        vmax=grid["vmax"],
    )
    relative_error = np.linalg.norm(mode - reference) / np.linalg.norm(reference)
    relative_mass_change = float((scalars.mass[-1] - scalars.mass[0]) / scalars.mass[0])
    relative_c2_change = float((scalars.c2[-1] - scalars.c2[0]) / scalars.c2[0])
    uniform_finest_panels = grid["nx"] * grid["nv"] // 4 * 4 ** config["amr"]["max_level"]
    print(
        f"AMR two-stream: kernel multiplier={multiplier:.9g}, asymptotic omega={frequency:.9g}, "
        f"field-history relative L2={relative_error:.9g}, mass change={relative_mass_change:.9g}, "
        f"C2 change={relative_c2_change:.9g}, active panels={int(scalars.active_panels.max())}, "
        f"max-level saturation={int(scalars.refinement_limited_panels.max())}, "
        f"max gap fraction={float(scalars.max_gap_fraction.max()):.9g}"
    )
    assert bool(scalars.valid.all())
    assert not bool(scalars.capacity_exceeded.any())
    assert int(scalars.invalid_panels.max()) == 0
    assert int(scalars.active_panels.max()) < uniform_finest_panels
    assert relative_error < 0.02
    assert abs(mode[-1]) > 3 * abs(mode[0])
    assert abs(relative_mass_change) < 1e-4
    assert abs(relative_c2_change) < 1e-3
    np.testing.assert_allclose(scalars.remap_c2_change[-1], scalars.c2[-1] - scalars.c2[0], rtol=1e-9, atol=2e-13)
