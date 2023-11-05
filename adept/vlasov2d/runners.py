import shutil
from typing import Dict, Tuple

import mlflow, tempfile, yaml, time, os
import xarray as xr
from jax.lax import scan
import numpy as np
from jax import jit, Array
import equinox as eqx
from tqdm import tqdm
from matplotlib import pyplot as plt
import pint
from diffrax import diffeqsolve, ODETerm, SubSaveAt, SaveAt
from adept.vlasov2d import helpers, storage
from adept.vlasov2d.integrator import Stepper, VectorField


def write_units(cfg, td):
    ureg = pint.UnitRegistry()
    _Q = ureg.Quantity

    lambda0 = _Q(cfg["units"]["laser wavelength"])
    w0 = (2 * np.pi / lambda0 * ureg.c).to("rad/s")
    t0 = (1 / w0).to("fs")
    n0 = (w0**2 * ureg.m_e * ureg.epsilon_0 / ureg.e**2.0).to("1/cc")
    T0 = _Q(cfg["units"]["electron temperature"]).to("eV")
    v0 = np.sqrt(2.0 * T0 / (ureg.m_e)).to("m/s")
    debye_length = (v0 / w0).to("nm")

    logLambda_ee = 23.5 - np.log(n0.magnitude**0.5 / T0.magnitude**-1.25)
    logLambda_ee -= (1e-5 + (np.log(T0.magnitude) - 2) ** 2.0 / 16) ** 0.5
    nuee = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0.magnitude**1.5, "Hz")
    nuee_norm = nuee / w0

    # if (Ti * me / mi) < Te:
    #     if Te > 10 * Z ^ 2:
    #         logLambda_ei = 24 - np.log(ne.magnitude**0.5 / Te.magnitude)
    #     else:
    #         logLambda_ei = 23 - np.log(ne.magnitude**0.5 * Z * Te.magnitude**-1.5)
    # else:
    #     logLambda_ei = 30 - np.log(ni.magnitude**0.5 * Z**2 / mu * Ti.magnitude**-1.5)

    # nuei = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0**1.5, "Hz")
    # nuee_norm = nuee / w0

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * debye_length).to("microns")
    box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * debye_length).to("microns")
    sim_duration = (cfg["grid"]["tmax"] * t0).to("ps")

    all_quantities = {
        "w0": w0,
        "t0": t0,
        "n0": n0,
        "v0": v0,
        "T0": T0,
        "lambda_D": debye_length,
        "logLambda_ee": logLambda_ee,
        "nuee": nuee,
        "nuee_norm": nuee_norm,
        "box_length": box_length,
        "box_width": box_width,
        "sim_duration": sim_duration,
    }

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump(all_quantities, fi)


def run_sim(cfg: Dict):
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as this_run:
        with tempfile.TemporaryDirectory() as td:
            helpers.log_params(cfg)
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            cfg = helpers.add_derived_quantities(cfg)

            cfg = helpers.get_save_quantities(cfg)

            write_units(cfg, td)
            mlflow.log_artifacts(td)

            # run sim - td is empty afterwards because it gets cleared at each time step
            do_time_loop(cfg, td)

            # post process
            fields = postprocess(cfg, this_run.info.run_id, td)
            mlflow.log_artifacts(td)

    return this_run.info.run_id, fields


# def do_time_loop(cfg: Dict, td: str):
#     def _run_(this_f: Array, t_arr: Array) -> Tuple[Array, Dict]:
#         vp_step = LeapfrogIntegrator(cfg)
#         last_state, full_storage = scan(f=vp_step, init=this_f, xs=t_arr)
#         return last_state, full_storage
#
#     init_func, run_func = hk.without_apply_rng(hk.transform(_run_))
#
#     t0 = time.time()
#     assert cfg["grid"]["nt"] % cfg["grid"]["num_checkpoints"] == 0
#     assert cfg["grid"]["num_checkpoints"] % cfg["grid"]["num_fs"] == 0
#
#     num_substeps = cfg["grid"]["nt"] // cfg["grid"]["num_checkpoints"]
#     running_f = storage.first_store(td, cfg)
#
#     for i in tqdm(range(cfg["grid"]["num_checkpoints"]), total=cfg["grid"]["num_checkpoints"]):
#         this_t = cfg["derived"]["t"][i * num_substeps + 1 : (i + 1) * num_substeps + 1]
#         this_driver = np.array(
#             cfg["derived"]["driver_function"](this_t[:, None] + np.array([0.0, 1.0])[None, :], cfg["pulses"])
#         )
#         running_f, fields = jit(run_func)(None, running_f, this_driver)
#         storage.store_everything(td, cfg, this_t, fields, this_driver, i, running_f)
#
#     mlflow.log_metrics({"run_time_min": round((time.time() - t0) / 60, 3)})
#     storage.store_everything(td, cfg, this_t, fields, this_driver, i, running_f)
#
#     mlflow.log_artifacts(td)


def do_time_loop(cfg: Dict, td: str):
    @eqx.filter_jit
    def _run_():
        args = {"driver": cfg["drivers"], "b_ext": 0.0}
        return diffeqsolve(
            terms=ODETerm(VectorField(cfg)),
            solver=Stepper(),
            t0=cfg["grid"]["tmin"],
            t1=cfg["grid"]["tmax"],
            max_steps=cfg["grid"]["max_steps"],
            dt0=cfg["grid"]["dt"],
            y0=state,
            args=args,
            # adjoint=DirectAdjoint(),
            saveat=SaveAt(
                subs=[SubSaveAt(ts=cfg["save"][k]["t"]["ax"], fn=cfg["save"][k]["func"]) for k in cfg["save"].keys()]
            ),
        )

    print("starting run")
    result = _run_()
    print("finished run")

    t0 = time.time()
    mlflow.log_metrics({"run_time_min": round((time.time() - t0) / 60, 3)})
    # storage.store_everything(td, cfg, this_t, fields, this_driver, i, running_f)

    mlflow.log_artifacts(td)


def postprocess(cfg: Dict, run_id, td: str):
    t0 = time.time()
    flds_path = os.path.join(td, "binary", "fields")
    os.makedirs(flds_path, exist_ok=True)
    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=td)
    flds_list = os.listdir(flds_path)

    # merge
    flds_paths = [os.path.join(flds_path, tf) for tf in flds_list]
    arr = xr.open_mfdataset(flds_paths, combine="by_coords", parallel=True)
    arr.to_netcdf(os.path.join(flds_path, "fields.nc"))
    _ = [os.remove(fl) for fl in flds_paths]
    del arr

    os.makedirs(os.path.join(td, "plots"), exist_ok=True)

    arr = xr.open_dataset(os.path.join(flds_path, "fields.nc"))
    t_skip = int(arr.coords["t"].data.size // 10)
    t_skip = t_skip if t_skip > 1 else 1

    for quant in ["ex", "ey", "dex"]:
        _plot_2d_(arr, quant, t_skip, td)

    mlflow.log_metrics({"postprocess_time_min": round((time.time() - t0) / 60, 3)})

    return arr


def _plot_2d_(arr, quant, t_skip, td):
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    for plot_num in range(9):
        this_ax_row = plot_num // 3
        this_ax_col = plot_num - 3 * this_ax_row
        arr[quant][plot_num * t_skip + t_skip // 2].T.plot(ax=ax[this_ax_row, this_ax_col])

    fig.savefig(os.path.join(td, "plots", f"{quant}.png"), bbox_inches="tight")
    plt.close()
