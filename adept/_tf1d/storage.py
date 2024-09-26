from typing import Dict

import os, xarray as xr
from flatdict import FlatDict
from diffrax import Solution
from matplotlib import pyplot as plt


def save_arrays(result: Solution, td: str, cfg: Dict, label: str) -> xr.Dataset:
    """
    This function saves the arrays to an xarray netcdf file

    """
    if label is None:
        label = "x"
        flattened_dict = dict(FlatDict(result.ys, delimiter="-"))
        save_ax = cfg["grid"]["x"]
    else:
        flattened_dict = dict(FlatDict(result.ys[label], delimiter="-"))
        save_ax = cfg["save"][label]["ax"]
    data_vars = {
        k: xr.DataArray(v, coords=(("t", cfg["save"]["t"]["ax"]), (label, save_ax))) for k, v in flattened_dict.items()
    }

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"state_vs_{label}.nc"))
    return saved_arrays_xr


def plot_xrs(which: str, td: str, xrs: Dict):
    """
    This function plots the xarray datasets


    """
    os.makedirs(os.path.join(td, "plots", which))
    os.makedirs(os.path.join(td, "plots", which, "ion"))
    os.makedirs(os.path.join(td, "plots", which, "electron"))

    for k, v in xrs.items():
        fname = f"{'-'.join(k.split('-')[1:])}.png"
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
        v.plot(ax=ax, cmap="gist_ncar")
        ax.grid()
        fig.savefig(os.path.join(td, "plots", which, k.split("-")[0], fname), bbox_inches="tight")
        plt.close(fig)

        if which == "kx":
            os.makedirs(os.path.join(td, "plots", which, "ion", "hue"), exist_ok=True)
            os.makedirs(os.path.join(td, "plots", which, "electron", "hue"), exist_ok=True)
            # only plot
            if v.coords["kx"].size > 8:
                hue_skip = v.coords["kx"].size // 8
            else:
                hue_skip = 1

            for log in [True, False]:
                fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
                v[:, ::hue_skip].plot(ax=ax, hue="kx")
                ax.set_yscale("log" if log else "linear")
                ax.grid()
                fig.savefig(
                    os.path.join(
                        td, "plots", which, k.split("-")[0], f"hue", f"{'-'.join(k.split('-')[1:])}-log-{log}.png"
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig)
