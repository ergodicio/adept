import os

import equinox
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

from jax import numpy as jnp
from es1d import pushers


def post_process(result, cfg, td):
    os.makedirs(os.path.join(td, "binary"))

    data_vars = {
        **{
            f"ion-{k}": xr.DataArray(v, coords=(("t", cfg["save"]["t_save"]), ("x", cfg["grid"]["x"])))
            for k, v in result.ys["ion"].items()
        },
        **{
            f"electron-{k}": xr.DataArray(v, coords=(("t", cfg["save"]["t_save"]), ("x", cfg["grid"]["x"])))
            for k, v in result.ys["electron"].items()
        },
    }

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", "stored_state.nc"))

    os.makedirs(os.path.join(td, "plots"))
    for k, v in saved_arrays_xr.items():
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
        v.plot(ax=ax, cmap="gist_ncar")
        fig.savefig(os.path.join(td, "plots", f"{k}.png"), bbox_inches="tight")
        plt.close(fig)


def get_derived_quantities(cfg_grid):
    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dt"] = 0.5 * cfg_grid["dx"] / 10

    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    return cfg_grid


def get_solver_quantities(cfg_grid):
    cfg_grid = {
        **cfg_grid,
        **{
            "x": jnp.linspace(cfg_grid["dx"] / 2, cfg_grid["xmax"] - cfg_grid["dx"] / 2, cfg_grid["nx"]),
            "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
        },
    }

    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

    return cfg_grid


def get_save_quantities(cfg):
    cfg["save"] = {
        **cfg["save"],
        **{"t_save": jnp.linspace(0, cfg["grid"]["tmax"], cfg["grid"]["nt"] // cfg["save"]["skip"])},
    }

    return cfg


def init_state(cfg):
    state = {}
    for species in ["ion", "electron"]:
        state[species] = dict(
            n=jnp.ones(cfg["grid"]["nx"]),
            p=cfg["physics"]["T0"] * jnp.ones(cfg["grid"]["nx"]),
            u=jnp.zeros(cfg["grid"]["nx"]),
        )

    return state


def get_vector_field(cfg):
    pusher_dict = {"ion": {}, "electron": {}}
    for species_name in ["ion", "electron"]:
        pusher_dict[species_name]["push_n"] = pushers.DensityStepper(cfg["grid"]["kx"])
        pusher_dict[species_name]["push_u"] = pushers.VelocityStepper(
            cfg["grid"]["kx"], cfg["grid"]["kxr"], cfg["physics"]
        )
        pusher_dict[species_name]["push_e"] = pushers.EnergyStepper(cfg["grid"]["kx"], cfg["physics"]["gamma"])

    push_driver = pushers.Driver(cfg["grid"]["x"])
    poisson_solver = pushers.PoissonSolver(cfg["grid"]["one_over_kx"])

    def push_everything(t, y, args):
        e = poisson_solver(
            cfg["physics"]["charge"]["ion"] * y["ion"]["n"] + cfg["physics"]["charge"]["electron"] * y["electron"]["n"]
        )
        ed = push_driver(cfg["drivers"]["ex"]["0"], t)
        total_e = e + ed

        dstate_dt = {"ion": {}, "electron": {}}
        for species_name in ["ion", "electron"]:
            if cfg["physics"][species_name]:
                n = y[species_name]["n"]
                u = y[species_name]["u"]
                p = y[species_name]["p"]
                q_over_m = cfg["physics"]["charge"][species_name] / cfg["physics"]["mass"][species_name]
                p_over_m = p / cfg["physics"]["mass"][species_name]

                dstate_dt[species_name]["n"] = pusher_dict[species_name]["push_n"](n, u)
                dstate_dt[species_name]["u"] = pusher_dict[species_name]["push_u"](n, u, p_over_m, q_over_m * total_e)
                dstate_dt[species_name]["p"] = pusher_dict[species_name]["push_e"](n, u, p_over_m, q_over_m * total_e)
            else:
                dstate_dt[species_name]["n"] = 0.0
                dstate_dt[species_name]["u"] = 0.0
                dstate_dt[species_name]["p"] = 0.0

        return dstate_dt

    return push_everything
