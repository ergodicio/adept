"""Combined adaptive/treecode verification against direct sums and linear theory."""

from copy import deepcopy
from pathlib import Path

import jax
import numpy as np
import yaml

from adept import SimulationSpec, run_prepared, solver_registry
from tests.test_farsight1d.test_physics import _linear_reference


def test_treecode_amr_example_matches_direct_and_linear_fourier_histories():
    """Retain the complete complex mode transient and both invariant budgets.

    This small regularized case tests coupled evolution, not treecode speedup,
    pairwise momentum conservation, or late-time turbulence convergence.
    """
    path = Path(__file__).parents[2] / "configs" / "farsight-1d" / "two-stream-treecode.yaml"
    config = yaml.safe_load(path.read_text())
    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=7)
    tree = run_prepared(prepared, key=jax.random.key(7))
    direct_config = deepcopy(config)
    direct_config["numerical"]["field_solver"] = "direct"
    direct_prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(direct_config), key=7)
    direct = run_prepared(direct_prepared, key=jax.random.key(7))
    grid, initial = config["grid"], config["initial"]
    fields = tree.report.result["fields"]
    tree_mode = np.fft.rfft(fields.electric_field, axis=1)[:, initial["mode"]] / grid["nx"]
    direct_mode = np.fft.rfft(direct.report.result["fields"].electric_field, axis=1)[:, initial["mode"]] / grid["nx"]
    reference, _, _ = _linear_reference(
        np.asarray(fields.t),
        wave_number=2 * np.pi * initial["mode"] / (grid["xmax"] - grid["xmin"]),
        epsilon=config["numerical"]["epsilon"],
        drift=initial["drift"],
        thermal_speed=initial["thermal_speed"],
        amplitude=initial["amplitude"],
        vmax=grid["vmax"],
    )
    linear_error = np.linalg.norm(tree_mode - reference) / np.linalg.norm(reference)
    direct_error = np.linalg.norm(tree_mode - direct_mode) / np.linalg.norm(direct_mode)
    scalars, direct_scalars = tree.report.result["scalars"], direct.report.result["scalars"]
    print(
        f"AMR/treecode: linear field-history relative L2={linear_error:.9g}, "
        f"direct field-history relative L2={direct_error:.9g}, "
        f"final momentum={float(scalars.momentum[-1]):.9g}, "
        f"tree/direct run seconds={tree.run_time_seconds:.3f}/{direct.run_time_seconds:.3f}"
    )
    assert linear_error < 0.02
    assert direct_error < 5e-4
    assert abs(tree_mode[-1]) > 3 * abs(tree_mode[0])
    for observations in (scalars, direct_scalars):
        assert bool(observations.valid.all())
        assert not bool(observations.capacity_exceeded.any())
        assert int(observations.invalid_panels.max()) == 0
        assert abs(float((observations.mass[-1] - observations.mass[0]) / observations.mass[0])) < 1e-4
        assert abs(float((observations.c2[-1] - observations.c2[0]) / observations.c2[0])) < 1e-3
        np.testing.assert_allclose(
            observations.remap_c2_change[-1], observations.c2[-1] - observations.c2[0], rtol=1e-9, atol=2e-13
        )
    np.testing.assert_allclose(scalars.mass, direct_scalars.mass, rtol=2e-8, atol=2e-13)
    np.testing.assert_allclose(scalars.c2, direct_scalars.c2, rtol=2e-8, atol=2e-13)
