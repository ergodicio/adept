#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml, os, argparse, tempfile
from itertools import product

import numpy as np
from jax.config import config

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


def _modify_defaults_(defaults, k0, nuee, a0):
    defaults["driver"]["0"]["k0"] = float(k0)
    defaults["driver"]["0"]["a0"] = float(a0)

    xmax = float(2.0 * np.pi / k0)
    defaults["grid"]["xmax"] = xmax

    defaults["nuee"] = nuee

    return defaults


@python_app
def remote_run(run_id):
    from jax.config import config

    config.update("jax_enable_x64", True)
    from utils import misc, runner

    runner.run_job(run_id, nested=False)
    if "MLFLOW_EXPORT" in os.environ:
        misc.export_run(run_id)

    return True


def generate_data():
    nuees = np.logspace(-7, -3, 5)
    klds = np.linspace(0.26, 0.4, 8)
    a0s = np.logspace(-6, -2, 5)

    for nuee, kld, a0 in product(nuees, klds, a0s):
        # load cfg
        with open(f"configs/vlasov-1d/epw.yaml", "r") as file:
            cfg = yaml.safe_load(file)

        # modify cfg
        mod_defaults = _modify_defaults_(cfg, kld, nuee, a0)

        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        # modify config
        with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as run:
            tags = {"sim_status": "queued"}
            with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
                with open(os.path.join(temp_path, "config.yaml"), "w") as fp:
                    yaml.dump(mod_defaults, fp)
                mlflow.log_artifacts(temp_path)
            mlflow.set_tags(tags)

        tf = remote_run(run.info.run_id)

    print(sum(tf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EPW data")
    parser.add_argument("--mode", help="enter mode")
    parser.add_argument("--run_id", help="enter run id")
    parser.add_argument("--machine", help="enter machine")
    args = parser.parse_args()

    if args.mode == "generate":
        shared_args = dict(walltime="8:00:00", cmd_timeout=120, nodes_per_block=1)
        if args.machine == "storm":
            # typhoon
            this_provider = PBSProProvider
            sched_args = [
                # "#PBS -S /bin/bash",
                # "#PBS -l nodes=1:ppn=36",
                "#PBS -N train-nlepw",
                "#PBS -j oe",
            ]
            provider_args = dict(
                queue="slpi",
                # account="m4434",
                scheduler_options="\n".join(sched_args),
                worker_init="export PYTHONPATH='$PYTHONPATH:/home/ajog/adept/'; \
                            export BASE_TEMPDIR='/b1/ajog/tmp/'; \
                            export MLFLOW_TRACKING_URI='/b1/ajog/mlflow'; \
                            module load anaconda/5.0.1; conda activate adept-cpu",
                cpus_per_node=12,
                launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=12 --ppn 1"),
            )
            provider_args = {**provider_args, **shared_args}
        elif args.machine == "nersc":
            # nersc
            this_provider = SlurmProvider
            sched_args = ["#SBATCH -C cpu", "#SBATCH --qos=regular"]
            provider_args = dict(
                partition=None,
                account="m4434",
                scheduler_options="\n".join(sched_args),
                worker_init="module load python; source /global/homes/a/archis/adept/venv/bin/activate; \
                        export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'; \
                        export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                        export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';",
                launcher=SrunLauncher(overrides="-c 128"),
                cores_per_node=128,
            )
            provider_args = {**provider_args, **shared_args}
        elif args.machine == "local":
            # nersc
            this_provider = LocalProvider
            provider_args = dict(
                # scheduler_options="\n".join(sched_args),
                worker_init="module load python; source /global/homes/a/archis/adept/venv/bin/activate; \
                        export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'; \
                        export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                        export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';",
                init_blocks=1,
                max_blocks=1,
            )
        else:
            raise NotImplementedError(f"{args.machine} not implemented as a provider")

        batch_size = 32
        htex = HighThroughputExecutor(
            label="generate-epw-data",
            provider=this_provider(**provider_args),
            cores_per_worker=int(128 // batch_size),
            max_workers=batch_size,
            cpu_affinity="block",
        )
        print(f"{htex.workers_per_node=}")
        config = Config(executors=[htex], retries=4)

        # load the Parsl config
        parsl.load(config)

        generate_data()
    elif args.mode == "run":
        remote_run(run_id=args.run_id)
    else:
        raise NotImplementedError
