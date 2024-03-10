import os
from typing import Dict, Callable, Tuple, List
from collections import defaultdict


import matplotlib.pyplot as plt
import jax, pint, yaml
from jax import numpy as jnp
import numpy as np
from scipy import constants
import equinox as eqx
from diffrax import ODETerm
import xarray as xr
from plasmapy.formulary.collisions.frequencies import fundamental_electron_collision_freq

from adept.lpse2d.core import integrator, driver
from adept.tf1d.pushers import get_envelope


def write_units(cfg, td):
    ureg = pint.UnitRegistry()
    _Q = ureg.Quantity

    n0 = _Q(cfg["units"]["normalizing density"]).to("1/cc")
    T0 = _Q(cfg["units"]["normalizing temperature"]).to("eV")

    wp0 = np.sqrt(n0 * ureg.e**2.0 / (ureg.m_e * ureg.epsilon_0)).to("rad/s")
    tp0 = (1 / wp0).to("fs")

    v0 = np.sqrt(2.0 * T0 / ureg.m_e).to("m/s")
    x0 = (v0 / wp0).to("nm")
    c_light = _Q(1.0 * ureg.c).to("m/s") / v0
    beta = (v0 / ureg.c).to("dimensionless")

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * x0).to("microns")
    if "ymax" in cfg["grid"].keys():
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * x0).to("microns")
    else:
        box_width = "inf"
    sim_duration = (cfg["grid"]["tmax"] * tp0).to("ps")

    # collisions
    logLambda_ee = 23.5 - np.log(n0.magnitude**0.5 / T0.magnitude**-1.25)
    logLambda_ee -= (1e-5 + (np.log(T0.magnitude) - 2) ** 2.0 / 16) ** 0.5
    nuee = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0.magnitude**1.5, "Hz")
    nuee_norm = nuee / wp0

    all_quantities = {
        "wp0": wp0,
        "tp0": tp0,
        "n0": n0,
        "v0": v0,
        "T0": T0,
        "c_light": c_light,
        "beta": beta,
        "x0": x0,
        "nuee": nuee,
        "logLambda_ee": logLambda_ee,
        "box_length": box_length,
        "box_width": box_width,
        "sim_duration": sim_duration,
    }

    cfg["units"]["derived"] = all_quantities

    cfg["grid"]["beta"] = beta.magnitude

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump({k: str(v) for k, v in all_quantities.items()}, fi)

    return cfg


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

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

    # cfg = get_more_units(cfg)

    cfg["grid"] = cfg_grid

    return cfg


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

    # e0 = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"], 2), dtype=jnp.complex128)
    # phi = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]), dtype=jnp.complex128)
    # phi += (
    #     1e-3
    #     * jnp.exp(-(((cfg["grid"]["x"][:, None] - 2000) / 400.0) ** 2.0))
    #     * jnp.exp(-1j * 0.2 * cfg["grid"]["x"][:, None])
    # )
    e0 = jnp.concatenate(
        [jnp.exp(1j * cfg["drivers"]["E0"]["k0"] * cfg["grid"]["x"])[:, None] for _ in range(cfg["grid"]["ny"])],
        axis=-1,
    )
    e0 = jnp.concatenate([e0[:, :, None], jnp.zeros_like(e0)[:, :, None]], axis=-1)
    e0 *= cfg["drivers"]["E0"]["a0"]

    if cfg["density"]["noise"]["type"] == "uniform":
        random_amps = np.random.uniform(
            cfg["density"]["noise"]["min"], cfg["density"]["noise"]["max"], (cfg["grid"]["nx"], cfg["grid"]["ny"])
        )

    elif cfg["density"]["noise"]["type"] == "normal":
        loc = 0.5 * (cfg["density"]["noise"]["min"] + cfg["density"]["noise"]["max"])
        scale = 1.0
        random_amps = np.random.normal(loc, scale, (cfg["grid"]["nx"], cfg["grid"]["ny"]))

    else:
        raise NotImplementedError

    random_phases = np.random.uniform(0, 2 * np.pi, (cfg["grid"]["nx"], cfg["grid"]["ny"]))

    phi = random_amps * np.exp(1j * random_phases)
    phi = jnp.fft.fft2(phi)

    if cfg["density"]["basis"] == "uniform":
        nprof = np.ones((cfg["grid"]["nx"], cfg["grid"]["ny"]))

    elif cfg["density"]["basis"] == "linear":
        left = cfg["density"]["center"] - cfg["density"]["width"] * 0.5
        right = cfg["density"]["center"] + cfg["density"]["width"] * 0.5
        rise = cfg["density"]["rise"]
        mask = get_envelope(rise, rise, left, right, cfg["grid"]["x"])

        ureg = pint.UnitRegistry()
        _Q = ureg.Quantity

        L = (
            _Q(cfg["density"]["gradient scale length"]).to("nm").magnitude
            / cfg["units"]["derived"]["x0"].to("nm").magnitude
        )
        nprof = cfg["density"]["val at center"] + (cfg["grid"]["x"] - cfg["density"]["center"]) / L
        nprof = mask * nprof

    elif cfg["density"]["basis"] == "exponential":
        left = cfg["density"]["center"] - cfg["density"]["width"] * 0.5
        right = cfg["density"]["center"] + cfg["density"]["width"] * 0.5
        rise = cfg["density"]["rise"]
        mask = get_envelope(rise, rise, left, right, cfg["grid"]["x"])

        ureg = pint.UnitRegistry()
        _Q = ureg.Quantity

        L = (
            _Q(cfg["density"]["gradient scale length"]).to("nm").magnitude
            / cfg["units"]["derived"]["x0"].to("nm").magnitude
        )
        nprof = cfg["density"]["val at center"] * np.exp((cfg["grid"]["x"] - cfg["density"]["center"]) / L)
        nprof = mask * nprof

    elif cfg["density"]["basis"] == "tanh":
        left = cfg["density"]["center"] - cfg["density"]["width"] * 0.5
        right = cfg["density"]["center"] + cfg["density"]["width"] * 0.5
        rise = cfg["density"]["rise"]
        nprof = get_envelope(rise, rise, left, right, cfg["grid"]["x"])

        if cfg["density"]["bump_or_trough"] == "trough":
            nprof = 1 - nprof
        nprof = cfg["density"]["baseline"] + cfg["density"]["bump_height"] * nprof

    elif cfg["density"]["basis"] == "sine":
        baseline = cfg["density"]["baseline"]
        amp = cfg["density"]["amplitude"]
        kk = cfg["density"]["wavenumber"]
        nprof = baseline * (1.0 + amp * jnp.sin(kk * cfg["grid"]["x"]))
    else:
        raise NotImplementedError

    state = {
        "e0": e0,
        "nb": nprof,
        "temperature": jnp.ones_like(e0[..., 0], dtype=jnp.float64),
        "dn": jnp.zeros_like(e0[..., 0], dtype=jnp.float64),
        "phi": phi,
        "delta": jnp.zeros_like(e0[..., 0], dtype=jnp.float64),
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


def get_more_units(cfg: Dict):
    """

    :type cfg: object
    """

    # ureg = pint.UnitRegistry()
    # _Q = ureg.Quantity
    import astropy.units as u

    n0 = _Q(cfg["units"]["normalizing density"]).to("1/cc")
    wp0 = np.sqrt(n0 * ureg.e**2.0 / (ureg.m_e * ureg.epsilon_0)).to("rad/s")
    T0 = _Q(cfg["units"]["normalizing temperature"]).to("eV")
    v0 = np.sqrt(2.0 * T0 / ureg.m_e).to("m/s")
    c_light = _Q(1.0 * ureg.c).to("m/s") / v0

    _nuei_ = 0.0
    # fundamental_electron_collision_freq(
    #     T_e=(Te := _Q(cfg["units"]["electron temperature"]).to("eV")).magnitude * u.eV,
    #     n_e=n0.to("1/m^3").magnitude / u.m**3,
    #     ion=f'{cfg["units"]["gas fill"]} {cfg["units"]["ionization state"]}+',
    # ).value
    # cfg["units"]["derived"]["nuei"] = _Q(f"{_nuei_} Hz")
    # cfg["units"]["derived"]["nuei_norm"] = (cfg["units"]["derived"]["nuei"].to("rad/s") / wp0).magnitude

    lambda_0 = _Q(cfg["units"]["laser wavelength"])
    laser_frequency = (2 * np.pi / lambda_0 * ureg.c).to("rad/s")
    laser_period = (1 / laser_frequency).to("fs")

    e_laser = np.sqrt(2.0 * _Q(cfg["drivers"]["E0"]["intensity"]) / ureg.c / ureg.epsilon_0).to("V/m")
    e_norm = (ureg.m_e * (np.sqrt(2.0 * Te / ureg.m_e).to("m/s")) * laser_frequency / ureg.e).to("V/m")

    cfg["units"]["derived"]["electric field"] = e_norm
    cfg["units"]["derived"]["laser field"] = e_laser

    cfg["drivers"]["E0"]["w0"] = (laser_frequency / wp0).magnitude
    cfg["drivers"]["E0"]["a0"] = (e_laser / e_norm).magnitude
    cfg["drivers"]["E0"]["k0"] = np.sqrt(
        (cfg["drivers"]["E0"]["w0"] ** 2.0 - cfg["plasma"]["wp0"] ** 2.0) / c_light.magnitude**2.0
    )

    print("laser parameters: ")
    print(f'w0 = {round(cfg["drivers"]["E0"]["w0"], 4)}')
    print(f'k0 = {round(cfg["drivers"]["E0"]["k0"], 4)}')
    print(f'a0 = {round(cfg["drivers"]["E0"]["a0"], 4)}')
    print()

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

        np.abs(v[tslice, :, :]).T.plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_kx_ky.png"), bbox_inches="tight")
        plt.close()

        # np.log10(np.abs(v[tslice, :, :])).T.plot(col="t", col_wrap=4)
        # plt.savefig(os.path.join(fld_dir, f"{k}_kx_ky.png"), bbox_inches="tight")
        # plt.close()
        #
        # kx = kfields.coords["kx"].data


def post_process(result, cfg: Dict, td: str) -> Tuple[xr.Dataset, xr.Dataset]:
    os.makedirs(os.path.join(td, "binary"))
    kfields, fields = make_xarrays(cfg, result.ts, result.ys, td)

    plot_fields(fields, td)
    # plot_kt(kfields, td)

    return kfields, fields


def make_xarrays(cfg, this_t, state, td):
    phi_vs_t = state["phi"].view(np.complex128)
    phi_k = xr.DataArray(phi_vs_t, coords=(("t", this_t), ("kx", cfg["grid"]["kx"]), ("ky", cfg["grid"]["ky"])))

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

    e0x = xr.DataArray(
        state["e0"].view(np.complex128)[..., 0],
        coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])),
    )

    e0y = xr.DataArray(
        state["e0"].view(np.complex128)[..., 1],
        coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])),
    )

    delta = xr.DataArray(state["delta"], coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])))

    kfields = xr.Dataset({"phi": phi_k, "ex": ex_k, "ey": ey_k})
    fields = xr.Dataset({"phi": phi_x, "ex": ex, "ey": ey, "delta": delta, "e0_x": e0x, "e0_y": e0y})
    # kfields.to_netcdf(os.path.join(td, "binary", "k-fields.xr"), engine="h5netcdf", invalid_netcdf=True)
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
