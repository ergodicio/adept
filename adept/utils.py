import flatdict, mlflow, os, boto3, botocore, shutil, pickle, yaml, time, tempfile
from urllib.parse import urlparse

from pint import Quantity
from mlflow.tracking import MlflowClient
import jax
import equinox as eqx


def log_params(cfg):
    flattened_dict = dict(flatdict.FlatDict(cfg, delimiter="."))
    num_entries = len(flattened_dict.keys())

    flattened_dict = {k: str(v) if isinstance(v, Quantity) else v for k, v in flattened_dict.items()}

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


def get_weights(artifact_uri, temp_path, models):
    dest_file_path = download_file("weights.eqx", artifact_uri, temp_path)
    if dest_file_path is not None:
        # with open(dest_file_path, "rb") as file:
        #     weights = pickle.load(file)
        # return weights
        return eqx.tree_deserialise_leaves(dest_file_path, like=models)

    else:
        return None


def download_from_s3(filename, target_path):
    import boto3

    bucket = filename.split("/")[2]
    key = "/".join(filename.split("/")[3:])

    s3 = boto3.client("s3")
    with open(target_path, "wb") as f:
        s3.download_fileobj(bucket, key, f)

    return target_path


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
    else:
        if "file" in artifact_uri:
            file_uri = file_uri[7:]
        if os.path.exists(file_uri):
            shutil.copyfile(file_uri, dest_file_path)
        else:
            return None

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

        def _safe_add(a1, a2):
            if a1 is None:
                return a2
            else:
                return a1 + a2

        def _is_none(x):
            return x is None

        def _safe_divide(a1):
            if a1 is None:
                return a1
            else:
                return a1 / num

        summed_gradients = jax.tree_map(_safe_add, gradients[0], gradients[1], is_leaf=_is_none)
        for i in range(2, num):
            summed_gradients = jax.tree_map(_safe_add, summed_gradients, gradients[i], is_leaf=_is_none)

        average_gradient = jax.tree_map(_safe_divide, summed_gradients, is_leaf=_is_none)
    else:
        average_gradient = gradients[0]

    return average_gradient


def upload_dir_to_s3(local_directory: str, bucket: str, destination: str, run_id: str, prefix="individual", step=0):
    """
    Uploads directory to s3 bucket for ingestion into mlflow on remote / cloud side

    This requires you to have permission to access the s3 bucket

    :param local_directory:
    :param bucket:
    :param destination:
    :param run_id:
    :return:
    """
    client = boto3.client("s3")

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)
            client.upload_file(local_path, bucket, s3_path)

    with open(os.path.join(local_directory, f"ingest-{run_id}.txt"), "w") as fi:
        fi.write("ready")

    if prefix == "individual":
        fname = f"ingest-{run_id}.txt"
    else:
        fname = f"{prefix}-{run_id}-{step}.txt"

    client.upload_file(os.path.join(local_directory, f"ingest-{run_id}.txt"), bucket, fname)


def robust_log_artifacts(directory, retries=5, delay=5):
    for attempt in range(retries):
        try:
            mlflow.log_artifacts(directory)
            print(f"Successfully removed {directory}")
            break
        except Exception as e:
            # if e.errno == 5:  # Input/output error
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)  # Wait before retrying

    else:
        print(f"Failed to log artifacts after {retries} attempts.")
