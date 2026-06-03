"""Tests for distributed Vlasov-2D distribution initialization."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from jax.sharding import NamedSharding

from adept._vlasov2d.datamodel import Vlasov2DConfig
from adept._vlasov2d.distributed import create_distribution_sharding, reshard_for_global_axis_fft
from adept._vlasov2d.helpers import _initialize_total_distribution_
from adept._vlasov2d.modules import sim_from_config


def _base_cfg() -> dict:
    return {
        "solver": "vlasov-2d",
        "units": {
            "normalizing_temperature": "2000eV",
            "normalizing_density": "1.5e21/cc",
        },
        "density": {
            "quasineutrality": True,
            "species-background": {
                "noise_seed": 1,
                "noise_type": "none",
                "noise_val": 0.0,
                "v0x": 0.1,
                "v0y": -0.2,
                "T0": 1.0,
                "m": 2.0,
                "basis": "sine",
                "baseline": 1.0,
                "amplitude": 1.0e-3,
                "wavenumber-x": 0.3,
                "wavenumber-y": 0.2,
            },
        },
        "grid": {
            "dt": 0.1,
            "nx": 8,
            "ny": 6,
            "nvx": 10,
            "nvy": 12,
            "tmin": 0.0,
            "tmax": 1.0,
            "vmax": 6.0,
            "xmin": 0.0,
            "xmax": 20.94,
            "ymin": -10.0,
            "ymax": 10.0,
        },
        "terms": {
            "edfdv": "exponential",
            "vdfdx": "exponential",
            "fokker_planck": {"is_on": False},
            "krook": {"is_on": False},
            "hou_li_filter": {"is_on": False},
        },
        "drivers": {"ex": {}, "ey": {}},
        "save": {"fields": {"t": {"nt": 2}}},
        "mlflow": {"experiment": "test-vlasov2d", "run": "distributed-init"},
    }


def test_sharded_distribution_initializer_matches_host_initializer():
    cfg = _base_cfg()
    sim = sim_from_config(Vlasov2DConfig.model_validate(cfg))
    full = _initialize_total_distribution_(cfg, sim)

    sharded_cfg = deepcopy(cfg)
    sharded_cfg["grid"]["distribution-sharding"] = {
        "enabled": True,
        "mesh_axes": ["x"],
        "mesh_shape": [1],
        "partition": ["x", None, None, None],
    }
    sharded = _initialize_total_distribution_(sharded_cfg, sim)

    n_full, f_full, vx_full, vy_full = full["electron"]
    n_sharded, f_sharded, vx_sharded, vy_sharded = sharded["electron"]

    assert isinstance(f_sharded.sharding, NamedSharding)
    assert f_sharded.shape == f_full.shape
    np.testing.assert_allclose(n_sharded, n_full)
    np.testing.assert_allclose(vx_sharded, vx_full)
    np.testing.assert_allclose(vy_sharded, vy_full)
    np.testing.assert_allclose(np.asarray(f_sharded), f_full, rtol=1e-6, atol=1e-7)


def test_distribution_sharding_can_derive_fft_resharding():
    dist = create_distribution_sharding(
        {
            "enabled": True,
            "mesh_axes": ["x"],
            "mesh_shape": [1],
            "partition": ["x", None, None, None],
        }
    )

    assert dist is not None
    fft_sharding = dist.with_unsharded_axis("x")
    assert isinstance(fft_sharding, NamedSharding)
    assert fft_sharding.spec[0] is None


def test_distribution_sharding_rejects_unknown_partition_axis():
    with pytest.raises(ValueError, match="mesh axes"):
        create_distribution_sharding(
            {
                "enabled": True,
                "mesh_axes": ["x"],
                "mesh_shape": [1],
                "partition": ["not-a-mesh-axis", None, None, None],
            }
        )


def test_reshard_for_global_axis_fft_preserves_values():
    cfg = _base_cfg()
    cfg["grid"]["distribution-sharding"] = {
        "enabled": True,
        "mesh_axes": ["x"],
        "mesh_shape": [1],
        "partition": ["x", None, None, None],
    }
    sim = sim_from_config(Vlasov2DConfig.model_validate(cfg))
    dist_result = _initialize_total_distribution_(cfg, sim)
    _, f_sharded, _, _ = dist_result["electron"]
    dist = create_distribution_sharding(cfg["grid"]["distribution-sharding"])

    local_for_x_fft = reshard_for_global_axis_fft(f_sharded, dist, "x")

    np.testing.assert_allclose(np.asarray(local_for_x_fft), np.asarray(f_sharded))
    assert local_for_x_fft.sharding.spec[0] is None
