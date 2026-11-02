"""The C++ port is opt-in; configuring it must not alter the verified envelope path."""

import ast
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from adept import ergoExo, lpse2d, lpse2d_cpp


def _cfg(solver, tpd, srs):
    with open("tests/test_lpse2d/configs/srs.yaml") as stream:
        cfg = yaml.safe_load(stream)
    cfg["solver"] = solver
    cfg["grid"].update(
        dx="0.05um",
        xmax="3.2um",
        ymin="-0.4um",
        ymax="0.4um",
        dt="1fs",
        tmax="6fs",
        boundary_width="0.4um",
    )
    cfg["save"]["fields"] = {"t": {"tmin": "0fs", "tmax": "6fs", "dt": "3fs"}}
    cfg["terms"]["epw"]["source"].update(noise_seed=123, tpd=tpd, srs=srs)
    return cfg


def _prepare(cfg, tmp_path):
    exo = ergoExo()
    modules = exo._setup_(deepcopy(cfg), str(tmp_path), log=False)
    return exo, modules


def test_public_solver_keys_select_independent_classes():
    exo = ergoExo()
    with jax.enable_x64():
        old = exo._get_adept_module_({"solver": "envelope-2d"})
        new = exo._get_adept_module_({"solver": "lpse-2d"})
    assert type(old) is lpse2d.BaseLPSE2D
    assert type(new) is lpse2d_cpp.BaseLPSE2D
    assert type(old) is not type(new)
    with jax.enable_x64(False):
        assert type(exo._get_adept_module_({"solver": "envelope-2d"})) is lpse2d.BaseLPSE2D
        assert type(exo._get_adept_module_({"solver": "lpse-2d"})) is lpse2d_cpp.BaseLPSE2D


@pytest.mark.parametrize("tpd,srs", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("x64", [False, True])
def test_legacy_run_is_unchanged_after_using_port(tmp_path, tpd, srs, x64):
    """Exercise the real dispatcher and saved fields, including the old simultaneous-source path.

    Results must remain bit-identical even after the port has been configured and stepped
    in the same process. The two paths deliberately retain different field layouts/defaults.
    """
    with jax.enable_x64(x64):
        legacy_cfg = _cfg("envelope-2d", tpd, srs)
        before, before_modules = _prepare(legacy_cfg, tmp_path)
        expected = before._execute_simulation(before_modules, None)["solver result"].ys

        port_cfg = _cfg("lpse-2d", tpd, srs)
        formulation = "combined" if tpd and srs else "separate"
        port_cfg["terms"]["epw"]["solver"] = formulation
        port_cfg["terms"]["light"] = {"solver": "spectral"}
        port, port_modules = _prepare(port_cfg, tmp_path)
        assert port.adept_module.cfg["grid"]["ny"] > 1
        assert port.adept_module.state["E0"].shape[-1] == 6  # float view of three complex components
        step = port.adept_module.diffeqsolve_quants["terms"].vector_field
        assert step.epw_solver == formulation
        if formulation == "separate":
            assert step.epw.noise_model == "thermal"
            assert step.epw.tpd_form == "lpse"
        result = port._execute_simulation(port_modules, None)["solver result"].ys
        assert np.max(np.asarray(result["default"]["e_sq"])) > 0
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(result))

        after, after_modules = _prepare(legacy_cfg, tmp_path)
        assert after.adept_module.state["E0"].shape[-1] == 4  # legacy two-component fields
        assert after.adept_module.cfg["terms"]["epw"]["source"].get("noise_model", "flat") == "flat"
        actual = after._execute_simulation(after_modules, None)["solver result"].ys
        assert np.max(np.asarray(actual["default"]["e_sq"])) > 0
        for old_leaf, new_leaf in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            np.testing.assert_array_equal(old_leaf, new_leaf)


def test_solver_packages_do_not_import_each_other():
    """Guard the isolation of helpers, drivers, diagnostics and optional solver modules too."""
    root = Path(__file__).resolve().parents[2] / "adept"
    for package, forbidden in (("_lpse2d", "_lpse2d_cpp"), ("_lpse2d_cpp", "_lpse2d")):
        for path in (root / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                    if node.module == "adept":
                        modules += [f"adept.{alias.name}" for alias in node.names]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                forbidden_names = {f"adept.{forbidden}", f"adept.{forbidden[1:]}"}
                for module in modules:
                    assert not any(module == name or module.startswith(name + ".") for name in forbidden_names), path


def test_translator_always_selects_new_solver():
    from adept._lpse2d_cpp.lpse_deck import translate_parms

    cfg, _ = translate_parms(
        {"grid.sizes": "6.4 1.6", "grid.nodes": "65 17", "lw.spectral.dt": "0.001", "lw.enable": "true"}
    )
    assert cfg["solver"] == "lpse-2d"


def test_port_examples_select_new_solver():
    for path in Path("configs/lpse-2d").glob("*.yaml"):
        assert yaml.safe_load(path.read_text())["solver"] == "lpse-2d", path
