"""Tests for asymmetric velocity grids (vmin != -vmax) in Vlasov-1D."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from jax import numpy as jnp

from adept._vlasov1d.modules import BaseVlasov1D


def _load(config_name: str) -> dict:
    config_path = Path(__file__).parent / "configs" / config_name
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build(config_dict: dict) -> BaseVlasov1D:
    module = BaseVlasov1D(config_dict)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    return module


def _assert_grid(species_grid: dict, vmin: float, vmax: float, nv: int) -> None:
    """Check a species velocity grid is the expected cell-centered, asymmetric grid."""
    dv = (vmax - vmin) / nv
    v = np.asarray(species_grid["v"])

    assert species_grid["nv"] == nv
    assert np.isclose(species_grid["vmin"], vmin)
    assert np.isclose(species_grid["vmax"], vmax)
    assert np.isclose(species_grid["dv"], dv)
    assert len(v) == nv
    # Cell-centered grid: first/last cell centers are half a cell inside the bounds
    assert np.isclose(v[0], vmin + dv / 2.0)
    assert np.isclose(v[-1], vmax - dv / 2.0)
    # Uniform spacing
    assert np.allclose(np.diff(v), dv)


def test_single_species_asymmetric_grid():
    """Grid-level vmin/vmax produces an asymmetric electron velocity grid."""
    config_dict = _load("resonance.yaml")
    config_dict["grid"]["vmin"] = -4.0
    config_dict["grid"]["vmax"] = 8.0
    nv = config_dict["grid"]["nv"]

    module = _build(config_dict)

    electron_grid = module.cfg["grid"]["species_grids"]["electron"]
    _assert_grid(electron_grid, vmin=-4.0, vmax=8.0, nv=nv)

    # The single-species top-level convenience grid matches the species grid
    np.testing.assert_allclose(np.asarray(module.cfg["grid"]["v"]), np.asarray(electron_grid["v"]))


def test_single_species_defaults_to_symmetric():
    """Omitting vmin reproduces the historical symmetric grid (vmin == -vmax)."""
    config_dict = _load("resonance.yaml")
    config_dict["grid"].pop("vmin", None)
    vmax = config_dict["grid"]["vmax"]
    nv = config_dict["grid"]["nv"]

    module = _build(config_dict)

    electron_grid = module.cfg["grid"]["species_grids"]["electron"]
    _assert_grid(electron_grid, vmin=-vmax, vmax=vmax, nv=nv)


def test_multispecies_asymmetric_grid():
    """Per-species vmin overrides the default symmetric grid for that species only."""
    config_dict = _load("multispecies_ion_acoustic.yaml")
    species = config_dict["terms"]["species"]
    electron_cfg = next(s for s in species if s["name"] == "electron")
    ion_cfg = next(s for s in species if s["name"] == "ion")

    # Asymmetric electron grid; ion left symmetric (no vmin key)
    electron_cfg["vmin"] = -3.0
    electron_cfg["vmax"] = 9.0

    module = _build(config_dict)

    species_grids = module.cfg["grid"]["species_grids"]
    _assert_grid(species_grids["electron"], vmin=-3.0, vmax=9.0, nv=electron_cfg["nv"])
    # Ion falls back to the symmetric default
    _assert_grid(species_grids["ion"], vmin=-ion_cfg["vmax"], vmax=ion_cfg["vmax"], nv=ion_cfg["nv"])


def test_asymmetric_grid_distribution_is_normalized():
    """An asymmetric grid wide enough to contain the bulk still integrates to ~density."""
    config_dict = _load("resonance.yaml")
    config_dict["grid"]["vmin"] = -5.0
    config_dict["grid"]["vmax"] = 10.0

    module = _build(config_dict)

    electron_grid = module.cfg["grid"]["species_grids"]["electron"]
    f = np.asarray(module.state["electron"])
    dv = electron_grid["dv"]
    density = f.sum(axis=1) * dv
    # Background density is uniform and normalized to 1
    np.testing.assert_allclose(density, 1.0, rtol=1e-3)


def test_collisions_conserve_density_on_asymmetric_grid():
    """Fokker-Planck + Krook collisions run and conserve density on an asymmetric grid.

    The zero-flux boundary conditions of the drift-diffusion scheme conserve density
    independent of where vmin/vmax sit, provided the grid brackets the bulk so that
    f ~ 0 at both edges. resonance.yaml has both fokker_planck and krook enabled.
    """
    from adept._vlasov1d.solvers.pushers.fokker_planck import Collisions

    config_dict = _load("resonance.yaml")
    # Asymmetric but still wide enough to contain the v0=0, T0=1 bulk at both edges
    config_dict["grid"]["vmin"] = -5.0
    config_dict["grid"]["vmax"] = 8.0

    module = _build(config_dict)
    cfg = module.cfg
    dv = cfg["grid"]["species_grids"]["electron"]["dv"]
    f0 = np.asarray(module.state["electron"])

    collisions = Collisions(cfg)
    nx = f0.shape[0]
    nu = jnp.ones(nx)
    dt = cfg["grid"]["dt"]

    f1 = np.asarray(collisions(nu, nu, jnp.asarray(f0), dt))

    assert np.all(np.isfinite(f1))
    # Density (zeroth moment) is conserved by the collision operators
    np.testing.assert_allclose(f1.sum(axis=1) * dv, f0.sum(axis=1) * dv, rtol=1e-6)


def test_cubic_spline_accepts_asymmetric_grid():
    """The cubic-spline edfdv pusher runs on an asymmetric velocity grid."""
    config_dict = deepcopy(_load("resonance.yaml"))
    config_dict["grid"]["vmin"] = -4.0
    config_dict["grid"]["vmax"] = 8.0
    config_dict["terms"]["edfdv"] = "cubic-spline"

    module = _build(config_dict)

    # State builds and the velocity grid is the requested asymmetric one
    electron_grid = module.cfg["grid"]["species_grids"]["electron"]
    _assert_grid(electron_grid, vmin=-4.0, vmax=8.0, nv=config_dict["grid"]["nv"])
    assert module.state["electron"].shape == (config_dict["grid"]["nx"], config_dict["grid"]["nv"])


if __name__ == "__main__":
    test_single_species_asymmetric_grid()
    test_single_species_defaults_to_symmetric()
    test_multispecies_asymmetric_grid()
    test_asymmetric_grid_distribution_is_normalized()
    test_collisions_conserve_density_on_asymmetric_grid()
    test_cubic_spline_accepts_asymmetric_grid()
    print("All asymmetric velocity grid tests passed!")
