import flatdict, mlflow, requests, os, boto3, botocore, shutil, pickle, yaml
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
import jax


def log_params(cfg):
    flattened_dict = dict(flatdict.FlatDict(cfg, delimiter="."))
    num_entries = len(flattened_dict.keys())

    if num_entries > 100:
        num_batches = num_entries % 100
        fl_list = list(flattened_dict.items())
        for i in range(num_batches):
            end_ind = min((i + 1) * 100, num_entries)
            trunc_dict = {k: v for k, v in fl_list[i * 100 : end_ind]}
            mlflow.log_params(trunc_dict)
    else:
        mlflow.log_params(flattened_dict)


def get_cfg(artifact_uri, temp_path):
    dest_file_path = download_file("config.yaml", artifact_uri, temp_path)
    with open(dest_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    return cfg


def get_weights(artifact_uri, temp_path):
    dest_file_path = download_file("weights.pkl", artifact_uri, temp_path)
    if dest_file_path is not None:
        with open(dest_file_path, "rb") as file:
            weights = pickle.load(file)
        return weights
    else:
        return None


def download_file(fname, artifact_uri, destination_path):
    file_uri = mlflow.get_artifact_uri(fname)
    dest_file_path = os.path.join(destination_path, fname)

    if "s3" in artifact_uri:
        s3 = boto3.client("s3")
        out = urlparse(file_uri, allow_fragments=False)
        bucket_name = out.netloc
        rest_of_path = out.path
        try:
            s3.download_file(bucket_name, rest_of_path[1:], dest_file_path)
        except botocore.exceptions.ClientError as e:
            return None
    elif "file" in artifact_uri:
        file_uri = file_uri[7:]
        if os.path.exists(file_uri):
            shutil.copyfile(file_uri, dest_file_path)
        else:
            return None
    else:
        raise NotImplementedError

    return dest_file_path


def is_job_done(run_id):
    return MlflowClient().get_run(run_id).data.tags["status"] == "completed"


def get_this_metric_of_this_run(metric_name, run_id):
    run = MlflowClient().get_run(run_id)
    return run.data.metrics[metric_name]


def download_and_open_file_from_this_run(fname, run_id, destination_path):
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=fname, dst_path=destination_path)
    with open(os.path.join(destination_path, fname), "rb") as f:
        this_file = pickle.load(f)

    return this_file


def all_reduce_gradients(gradients, num):
    if num > 1:
        summed_gradients = jax.tree_map(operator.add, gradients[0], gradients[1])
        for i in range(2, num):
            summed_gradients = jax.tree_map(operator.add, summed_gradients, gradients[i])
        average_gradient = jax.tree_map(lambda x: x / num, summed_gradients)
    else:
        average_gradient = gradients[0]

    return average_gradient