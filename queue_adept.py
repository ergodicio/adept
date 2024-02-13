import argparse
import os
import tempfile
import time

import mlflow
import yaml

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def _queue_run_(machine, run_id, mode, run_name):
    if "cpu" in machine:
        base_job_file = os.environ["CPU_BASE_JOB_FILE"]
    elif "gpu" in machine:
        base_job_file = os.environ["GPU_BASE_JOB_FILE"]
    else:
        raise NotImplementedError

    with open(base_job_file, "r") as fh:
        base_job = fh.read()

    with open(os.path.join(os.getcwd(), f"queue-{mode}-{run_name}.sh"), "w") as job_file:
        job_file.write(base_job)
        job_file.writelines(f"\nsrun python run.py --run_id {run_id}")
    
    os.system(f"sbatch queue-{mode}-{run_name}.sh")
    time.sleep(0.1)
    os.system("sqs")


def load_and_make_folders(cfg_path):
    with open(f"{cfg_path}.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as run:
        tags = {"sim_status": "queued"}
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
            with open(os.path.join(temp_path, "config.yaml"), "w") as fp:
                yaml.dump(cfg, fp)
            mlflow.log_artifacts(temp_path)
        mlflow.set_tags(tags)

    return cfg, run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Differentiation Enabled Plasma Transport")
    parser.add_argument("--cfg", help="enter path to cfg")
    args = parser.parse_args()

    cfg, run_id = load_and_make_folders(args.cfg)
    _queue_run_(cfg["machine"], run_id, cfg["mode"], cfg["mlflow"]["run"])
