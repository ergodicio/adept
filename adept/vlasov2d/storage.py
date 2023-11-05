from typing import Dict
import os

import numpy as np
import xarray as xr
from jax import numpy as jnp
import mlflow


def store_fields(cfg: Dict, td: str, fields: Dict, this_t: np.ndarray) -> xr.Dataset:
    fields_dir = os.path.join(td, "binary", "fields")
    os.makedirs(fields_dir, exist_ok=True)

    das = {
        k: xr.DataArray(v, coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])))
        for k, v in fields.items()
    }
    fields = xr.Dataset(das)
    fields.to_netcdf(os.path.join(fields_dir, f"fields-t={round(this_t[-1],4)}.nc"))


def store_f(cfg, this_t, td, this_f):
    f_store = xr.DataArray(
        this_f[None, :, :, :, :],
        coords=(
            ("t", this_t),
            ("x", cfg["grid"]["x"]),
            ("y", cfg["grid"]["y"]),
            ("v_x", cfg["grid"]["vx"]),
            ("v_y", cfg["grid"]["vy"]),
        ),
    )
    f_store.to_netcdf(os.path.join(td, "binary", "f", f"f-{round(this_t[0],4)}.nc"))


def clean_td(td):
    _ = [os.remove(os.path.join(td, "binary", "fields", fl)) for fl in os.listdir(os.path.join(td, "binary", "fields"))]
    _ = [os.remove(os.path.join(td, "binary", "f", fl)) for fl in os.listdir(os.path.join(td, "binary", "f"))]


def first_store(td, cfg):
    os.makedirs(os.path.join(td, "binary", "f"))
    os.makedirs(os.path.join(td, "binary", "fields"))

    start_f = jnp.array(cfg["grid"]["f"])
    store_f(cfg, np.array([0.0]), td, start_f)
    fields = {
        "ex": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
        "ey": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
        "total_ex": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
        "total_ey": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
        "dex": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
    }
    store_fields(cfg, td, fields, np.array([0.0]))
    mlflow.log_artifacts(td)
    clean_td(td)

    return start_f


def store_everything(td, cfg, this_t, fields, this_driver, i, running_f):
    fields["dex"] = this_driver[:, 0]
    store_fields(cfg, td, fields, this_t)

    if i % (cfg["grid"]["num_checkpoints"] // cfg["grid"]["num_fs"]) == 0:
        store_f(cfg, this_t[-1:], td, running_f)
        mlflow.log_artifacts(td)
        clean_td(td)
