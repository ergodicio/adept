import os
from typing import Dict, Callable, Tuple, List
from collections import defaultdict


import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp
import numpy as np
from scipy import constants
import equinox as eqx
from diffrax import ODETerm
import xarray as xr

from adept.lpse2d.core import integrator, driver


def get_derived_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dy"] = cfg_grid["ymax"] / cfg_grid["ny"]

    # cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    return cfg_grid


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    # cfg["save"]["func"] = {**cfg["save"]["func"], **{"callable": get_save_func(cfg)}}
    cfg["save"]["t"]["ax"] = jnp.linspace(cfg["save"]["t"]["tmin"], cfg["save"]["t"]["tmax"], cfg["save"]["t"]["nt"])

    return cfg


def get_solver_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are arrays

    This is run after the log params step

    :param cfg_grid:
    :return:
    """

    cfg_grid = cfg["grid"]

    cfg_grid = {
        **cfg_grid,
        **{
            "x": jnp.linspace(
                cfg_grid["xmin"] + cfg_grid["dx"] / 2, cfg_grid["xmax"] - cfg_grid["dx"] / 2, cfg_grid["nx"]
            ),
            "y": jnp.linspace(
                cfg_grid["ymin"] + cfg_grid["dy"] / 2, cfg_grid["ymax"] - cfg_grid["dy"] / 2, cfg_grid["ny"]
            ),
            "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "ky": jnp.fft.fftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
            "kyr": jnp.fft.rfftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
        },
    }

    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

    one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

    one_over_ky = np.zeros_like(cfg_grid["ky"])
    one_over_ky[1:] = 1.0 / cfg_grid["ky"][1:]
    cfg_grid["one_over_ky"] = jnp.array(one_over_ky)

    one_over_kyr = np.zeros_like(cfg_grid["kyr"])
    one_over_kyr[1:] = 1.0 / cfg_grid["kyr"][1:]
    cfg_grid["one_over_kyr"] = jnp.array(one_over_kyr)

    one_over_ksq = np.array(1.0 / (cfg_grid["kx"][:, None] ** 2.0 + cfg_grid["ky"][None, :] ** 2.0))
    one_over_ksq[0, 0] = 0.0
    cfg_grid["one_over_ksq"] = jnp.array(one_over_ksq)

    if cfg["terms"]["epw"]["boundary"]["x"] == "absorbing":
        envelope_x = driver.get_envelope(50.0, 50.0, 300.0, cfg["grid"]["xmax"] - 300.0, cfg_grid["x"])[:, None]
    else:
        envelope_x = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

    if cfg["terms"]["epw"]["boundary"]["y"] == "absorbing":
        envelope_y = driver.get_envelope(50.0, 50.0, 300.0, cfg["grid"]["ymax"] - 300.0, cfg_grid["y"])[None, :]
    else:
        envelope_y = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

    cfg_grid["absorbing_boundaries"] = np.exp(-cfg_grid["dt"] * (1.0 - envelope_x * envelope_y))

    return cfg_grid


def init_state(cfg: Dict, td=None) -> Dict:
    """
    This function initializes the state for the PDE solve

    The state is initialized using
    random seeds
    drivers


    :param cfg:
    :return: state: Dict
    """

    e0 = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"], 2), dtype=jnp.complex128)
    phi = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]), dtype=jnp.complex128)
    # phi += (
    #     1e-3
    #     * jnp.exp(-(((cfg["grid"]["x"][:, None] - 2000) / 400.0) ** 2.0))
    #     * jnp.exp(-1j * 0.2 * cfg["grid"]["x"][:, None])
    # )
    # e0 = jnp.concatenate(
    #     [jnp.exp(1j * cfg["drivers"]["E0"]["k0"] * cfg["grid"]["x"])[:, None] for _ in range(cfg["grid"]["ny"])], axis=-1
    # )
    # e0 = jnp.concatenate([e0[:, :, None], jnp.zeros_like(e0)[:, :, None]], axis=-1)
    # e0 *= cfg["drivers"]["E0"]["e0"]

    random_amps_x = 1.0e-12 * np.random.uniform(0.1, 1, cfg["grid"]["nx"])
    random_amps_y = 1.0e-12 * np.random.uniform(0.1, 1, cfg["grid"]["ny"])

    # phi = jnp.sum(random_amps_x * jnp.exp(1j * cfg["grid"]["kx"][None, :] * cfg["grid"]["x"][:, None]), axis=-1)[
    #     :, None
    # ]
    # phi += jnp.sum(random_amps_y * jnp.exp(1j * cfg["grid"]["ky"][None, :] * cfg["grid"]["y"][:, None]), axis=-1)[
    #     None, :
    # ]
    # phi = jnp.fft.fft2(phi)

    state = {
        "e0": e0,
        "nb": (0.8 + 0.4 * cfg["grid"]["x"] / cfg["grid"]["xmax"])[:, None] * np.ones_like(phi, dtype=np.float64),
        "temperature": jnp.ones_like(e0[..., 0], dtype=jnp.float64),
        "dn": jnp.zeros_like(e0[..., 0], dtype=jnp.float64),
        "phi": phi,
        "delta": (
            0
            + 0.0
            * jnp.sin(2 * jnp.pi * cfg["grid"]["x"][:, None] / cfg["grid"]["xmax"])
            * jnp.ones_like(e0[..., 0], dtype=jnp.float64)
        ),
    }

    if td is not None:
        plot_dir = os.path.join(td, "plots", "init_state")
        os.makedirs(plot_dir)

        for comp, label in zip(np.arange(2), ["x", "y"]):
            for func, nm in zip([np.real, np.abs], ["real", "abs"]):
                fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
                cb = ax.contourf(cfg["grid"]["x"], cfg["grid"]["y"], func(state["e0"][..., comp].T), 32)
                ax.grid()
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"{nm}(e0_{label}(x,y))")
                fig.colorbar(cb)
                fig.savefig(os.path.join(plot_dir, f"{nm}-e0-{label}.png"), bbox_inches="tight")
                plt.close()

        for k in ["nb", "dn", "temperature", "delta"]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
            cb = ax.contourf(cfg["grid"]["x"], cfg["grid"]["y"], state[k].T, 32)
            ax.grid()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(k)
            fig.colorbar(cb)
            fig.savefig(os.path.join(plot_dir, f"{k}.png"), bbox_inches="tight")
            plt.close()

        for func, nm in zip([np.real, np.abs], ["real", "abs"]):
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
            cb = ax.contourf(cfg["grid"]["x"], cfg["grid"]["y"], func(state["phi"].T), 32)
            ax.grid()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{nm}(phi(x,y))")
            fig.colorbar(cb)
            fig.savefig(os.path.join(plot_dir, f"{nm}-phi.png"), bbox_inches="tight")
            plt.close()

    return {k: v.view(dtype=np.float64) for k, v in state.items()}


def calc_e0(cfg):
    e_laser = np.sqrt(2.0 * (float(cfg["drivers"]["E0"]["intensity"]) * 100) / constants.c / constants.epsilon_0)
    e_norm = constants.m_e * cfg["norms"]["velocity"] * cfg["norms"]["frequency"] / constants.e

    cfg["norms"]["electric field"] = e_norm
    cfg["norms"]["laser field"] = e_laser

    cfg["drivers"]["E0"]["e0"] = e_laser / e_norm
    cfg["drivers"]["E0"]["k0"] = np.sqrt(
        (cfg["drivers"]["E0"]["w0"] ** 2.0 - cfg["plasma"]["wp0"] ** 2.0) / cfg["norms"]["c"] ** 2.0
    )

    print("laser parameters: ")
    print(f'w0 = {round(cfg["drivers"]["E0"]["w0"],4)}')
    print(f'k0 = {round(cfg["drivers"]["E0"]["k0"],4)}')
    print(f"a0 = {round(e_laser / e_norm, 4)}")
    print()

    return cfg


def calc_norms(cfg: Dict):
    """

    :type cfg: object
    """
    cfg["norms"] = {}
    cfg["norms"]["n0"] = float(cfg["plasma"]["density"])
    cfg["norms"]["T0"] = cfg["plasma"]["temperature"]
    cfg["norms"]["frequency"] = np.sqrt(cfg["norms"]["n0"] * constants.e**2.0 / constants.m_e / constants.epsilon_0)
    cfg["norms"]["velocity"] = (
        2.0
        * np.sqrt(
            np.average(cfg["plasma"]["temperature"])
            / (1000.0 * constants.physical_constants["electron mass energy equivalent in MeV"][0])
        )
        * constants.c
    )
    cfg["norms"]["c"] = constants.c / cfg["norms"]["velocity"]

    cfg["norms"]["space"] = cfg["norms"]["velocity"] / cfg["norms"]["frequency"]
    cfg["norms"]["time"] = 1.0 / cfg["norms"]["frequency"]

    cfg = calc_e0(cfg)

    return cfg


def plot_fields(fields, td):
    t_skip = int(fields.coords["t"].data.size // 8)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    for k, v in fields.items():
        fld_dir = os.path.join(td, "plots", k)
        os.makedirs(fld_dir)

        np.abs(v[tslice]).T.plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_x.png"), bbox_inches="tight")
        plt.close()

        np.real(v[tslice]).T.plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_x_r.png"), bbox_inches="tight")
        plt.close()

        # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        # np.abs(v[:, 1, 0]).plot(ax=ax)
        # fig.savefig(os.path.join(td, "plots", f"{k}_k1.png"))
        # plt.close()
        ymidpt = int(fields.coords["y"].data.size // 2)
        slice_dir = os.path.join(fld_dir, "slice-along-x")
        os.makedirs(slice_dir)
        np.log10(np.abs(v[tslice, :, ymidpt])).plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(slice_dir, f"log-{k}.png"))
        plt.close()

        np.abs(v[tslice, :, ymidpt]).plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(slice_dir, f"{k}.png"))
        plt.close()

        np.real(v[tslice, :, ymidpt]).plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(slice_dir, f"real-{k}.png"))
        plt.close()

        np.log10(np.abs(v[:, :, ymidpt])).plot()
        plt.savefig(os.path.join(slice_dir, f"spacetime-log-{k}.png"))
        plt.close()

        np.abs(v[:, :, ymidpt]).plot()
        plt.savefig(os.path.join(slice_dir, f"spacetime-{k}.png"))
        plt.close()

        np.real(v[:, :, ymidpt]).plot()
        plt.savefig(os.path.join(slice_dir, f"spacetime-real-{k}.png"))
        plt.close()


def plot_kt(kfields, td):
    t_skip = int(kfields.coords["t"].data.size // 8)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    for k, v in kfields.items():
        fld_dir = os.path.join(td, "plots", k)
        os.makedirs(fld_dir, exist_ok=True)

        np.log10(np.abs(v[tslice, :, 0])).T.plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_kx.png"), bbox_inches="tight")
        plt.close()

        kx = kfields.coords["kx"].data


def post_process(result, cfg: Dict, td: str) -> Tuple[xr.Dataset, xr.Dataset]:
    os.makedirs(os.path.join(td, "binary"))
    kfields, fields = make_xarrays(cfg, result.ts, result.ys, td)

    plot_fields(fields, td)
    plot_kt(kfields, td)

    return kfields, fields


def make_xarrays(cfg, this_t, state, td):
    phi_vs_t = state["phi"].view(np.complex128)
    phi_k = xr.DataArray(
        phi_vs_t,
        coords=(("t", this_t), ("kx", cfg["grid"]["kx"]), ("ky", cfg["grid"]["ky"])),
    )

    ex_k = xr.DataArray(
        -1j * cfg["grid"]["kx"][:, None] * phi_vs_t,
        coords=(("t", this_t), ("kx", cfg["grid"]["kx"]), ("ky", cfg["grid"]["ky"])),
    )

    ey_k = xr.DataArray(
        -1j * cfg["grid"]["ky"][None, :] * phi_vs_t,
        coords=(("t", this_t), ("kx", cfg["grid"]["kx"]), ("ky", cfg["grid"]["ky"])),
    )

    phi_x = xr.DataArray(
        np.fft.ifft2(phi_vs_t) / cfg["grid"]["nx"] / cfg["grid"]["ny"] * 4,
        coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])),
    )

    ex = xr.DataArray(
        -np.fft.ifft2(1j * cfg["grid"]["kx"][:, None] * phi_vs_t) / cfg["grid"]["nx"] / cfg["grid"]["ny"] * 4,
        coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])),
    )
    ey = xr.DataArray(
        -np.fft.ifft2(1j * cfg["grid"]["ky"][None, :] * phi_vs_t) / cfg["grid"]["nx"] / cfg["grid"]["ny"] * 4,
        coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])),
    )
    delta = xr.DataArray(state["delta"], coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])))

    kfields = xr.Dataset({"phi": phi_k, "ex": ex_k, "ey": ey_k})
    fields = xr.Dataset({"phi": phi_x, "ex": ex, "ey": ey, "delta": delta})
    kfields.to_netcdf(os.path.join(td, "binary", "k-fields.xr"), engine="h5netcdf", invalid_netcdf=True)
    fields.to_netcdf(os.path.join(td, "binary", "fields.xr"), engine="h5netcdf", invalid_netcdf=True)

    return kfields, fields


def get_models(model_config: Dict) -> defaultdict[eqx.Module]:
    if model_config:
        model_keys = jax.random.split(jax.random.PRNGKey(420), len(model_config.keys()))
        model_dict = defaultdict(eqx.Module)
        for (term, config), this_key in zip(model_config.items(), model_keys):
            if term == "file":
                pass
            else:
                for act in ["activation", "final_activation"]:
                    if config[act] == "tanh":
                        config[act] = jnp.tanh

                model_dict[term] = eqx.nn.MLP(**{**config, "key": this_key})
        if model_config["file"]:
            model_dict = eqx.tree_deserialise_leaves(model_config["file"], model_dict)

        return model_dict
    else:
        return False


def mva(actual_ek1, mod_defaults, results, td, coords):
    loss_t = np.linspace(200, 400, 64)
    ek1 = -1j * mod_defaults["grid"]["kx"][None, :, None] * results.ys["phi"].view(np.complex128)
    ek1 = jnp.mean(jnp.abs(ek1[:, -1, :]), axis=-1)
    rescaled_ek1 = ek1 / jnp.amax(ek1) * np.amax(actual_ek1.data)
    # interp_ek1 = jnp.interp(loss_t, mod_defaults["save"]["t"]["ax"], rescaled_ek1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    ax[0].plot(coords["t"].data, actual_ek1, label="Vlasov")
    ax[0].plot(mod_defaults["save"]["t"]["ax"], rescaled_ek1, label="NN + Fluid")
    ax[0].axvspan(loss_t[0], loss_t[-1], alpha=0.1)
    ax[1].semilogy(coords["t"].data, actual_ek1, label="Vlasov")
    ax[1].semilogy(mod_defaults["save"]["t"]["ax"], rescaled_ek1, label="NN + Fluid")
    ax[1].axvspan(loss_t[0], loss_t[-1], alpha=0.1)
    ax[0].set_xlabel(r"t ($\omega_p^{-1}$)", fontsize=12)
    ax[1].set_xlabel(r"t ($\omega_p^{-1}$)", fontsize=12)
    ax[0].set_ylabel(r"$|\hat{n}|^{1}$", fontsize=12)
    ax[0].grid()
    ax[1].grid()
    ax[0].legend(fontsize=14)
    fig.savefig(os.path.join(td, "plots", "vlasov_v_fluid.png"), bbox_inches="tight")
    plt.close(fig)


def get_diffeqsolve_quants(cfg):
    return dict(
        terms=ODETerm(integrator.VectorField(cfg)),
        solver=integrator.Stepper(),
        saveat=dict(ts=cfg["save"]["t"]["ax"]),  # , fn=cfg["save"]["func"]["callable"]),
    )
