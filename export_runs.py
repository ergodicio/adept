import os
from tqdm import tqdm

os.environ["MLFLOW_TRACKING_URI"] = "/pscratch/sd/a/archis/mlflow"
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "False"

from utils.misc import export_run

if __name__ == "__main__":
    with open("/global/homes/a/archis/adept/completed_run_ids.txt", "r") as f:
        run_ids = f.read().split("\n")

    with open("/global/homes/a/archis/adept/uploaded_run_ids.txt", "r") as f:
        uploaded_run_ids = f.read().split("\n")
        left_run_ids = list(set(run_ids) - set(uploaded_run_ids))

        print(f"found {len(run_ids)} completed runs")
        print(f"found {len(uploaded_run_ids)} uploaded runs")
        print(f"uploading {len(left_run_ids)} runs")

    for run_id in tqdm(left_run_ids):
        export_run(run_id)
        with open("/global/homes/a/archis/adept/uploaded_run_ids.txt", "a") as f:
            f.write(run_id + "\n")
