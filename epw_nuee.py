#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml, os, argparse, tempfile
from itertools import product

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import mlflow

import parsl
from parsl.config import Config
from parsl.providers import SlurmProvider, PBSProProvider, LocalProvider
from parsl.launchers import SrunLauncher, MpiExecLauncher
from parsl.executors import HighThroughputExecutor
from parsl.app.app import python_app

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def _modify_defaults_(defaults, Te, log10_ne, log10_a0, kld):
    defaults["units"]["reference electron temperature"] = f"{Te}eV"
    defaults["units"]["reference electron density"] = f"{10**log10_ne}/cm^3"

    defaults["drivers"]["ex"]["0"]["k0"] = f"{kld}kl_D"
    defaults["drivers"]["ex"]["0"]["a0"] = float(10**log10_a0)

    xmax = float(2.0 * np.pi / kld)
    defaults["grid"]["xmax"] = f"{xmax}l_D"

    return defaults


@python_app
def remote_run(run_id):
    from jax import config

    config.update("jax_enable_x64", True)
    from utils import misc, runner

    runner.run_job(run_id, nested=False)
    misc.export_run(run_id)

    return True


def generate_data():
    Te = np.linspace(400, 4000, 2)
    ne = np.linspace(18, 21, 2)
    a0s = np.linspace(-8, -4, 2)
    klds = np.linspace(0.26, 0.4, 2)

    klds = [float(np.round(kld, 2)) for kld in klds]

    tf = []
    for Te, log10_ne, log10_a0, kld in product(Te, ne, a0s, klds):
        # load cfg
        with open(f"configs/vlasov-1d2v/epw.yaml", "r") as file:
            cfg = yaml.safe_load(file)

        # modify cfg
        mod_defaults = _modify_defaults_(cfg, Te, log10_ne, log10_a0, kld)

        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        # modify config
        with mlflow.start_run(run_name=f"{kld=}, {log10_a0=}, {log10_ne=}, {Te=}") as run:
            tags = {"sim_status": "queued"}
            with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
                with open(os.path.join(temp_path, "config.yaml"), "w") as fp:
                    yaml.dump(mod_defaults, fp)
                mlflow.log_artifacts(temp_path)
            mlflow.set_tags(tags)

        tf.append(remote_run(run.info.run_id))

    print(sum([tt.result() for tt in tf]))


if __name__ == "__main__":
    shared_args = dict(walltime="0:12:00", cmd_timeout=120, nodes_per_block=1)
    # elif args.machine == "nersc":
    # nersc
    this_provider = SlurmProvider
    sched_args = ["#SBATCH -C gpu", "#SBATCH --qos=debug"]
    provider_args = dict(
        partition=None,
        account="m4490_g",
        scheduler_options="\n".join(sched_args),
        worker_init="module load cudnn/8.9.3_cuda12.lua; \
                module load cudatoolkit/12.0.lua; \
                source /pscratch/sd/a/archis/venvs/adept-gpu/bin/activate; \
                export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'; \
                export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';\
                export MLFLOW_EXPORT=True; which python3",
        launcher=SrunLauncher(overrides="--gpus-per-node 4 -c 64"),
        # cores_per_node=128,
    )
    provider_args = {**provider_args, **shared_args}
    # elif args.machine == "local":
    # nersc
    # this_provider = LocalProvider
    # provider_args = dict(
    #     # scheduler_options="\n".join(sched_args),
    #     worker_init="module load python; source /global/homes/a/archis/adept/venv/bin/activate; \
    #             export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'; \
    #             export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
    #             export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow'; \
    #             export MLFLOW_EXPORT=True",
    #     init_blocks=1,
    #     max_blocks=1,
    # )
    # else:
    # raise NotImplementedError(f"{args.machine} not implemented as a provider")

    batch_size = 4
    htex = HighThroughputExecutor(
        available_accelerators=4,
        label="generate-2v-epw-data",
        provider=this_provider(**provider_args),
        # cores_per_worker=int(128 // batch_size),
        # max_workers=batch_size,
        cpu_affinity="block",
    )
    print(f"{htex.workers_per_node=}")
    config = Config(executors=[htex], retries=1)

    # load the Parsl config
    parsl.load(config)

    generate_data()
    # elif args.mode == "run":
    #     remote_run(run_id=args.run_id)
    # else:
    #     raise NotImplementedError
