import flatdict, mlflow, os, boto3, botocore, shutil, pickle, yaml, operator
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
import jax
import equinox as eqx


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


def get_weights(artifact_uri, temp_path, models):
    dest_file_path = download_file("weights.eqx", artifact_uri, temp_path)
    if dest_file_path is not None:
        # with open(dest_file_path, "rb") as file:
        #     weights = pickle.load(file)
        # return weights
        return eqx.tree_deserialise_leaves(dest_file_path, like=models)

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


def get_jq(client: boto3.client, desired_machine: str):
    queues = client.describe_job_queues()
    for queue in queues["jobQueues"]:
        if desired_machine == queue["jobQueueName"]:
            return queue["jobQueueArn"]


def get_jd(client: boto3.client, sim_type: str, desired_machine: str):
    jobdefs = client.describe_job_definitions()
    for jobdef in jobdefs["jobDefinitions"]:
        if (
            desired_machine in jobdef["jobDefinitionName"]
            and jobdef["status"] == "ACTIVE"
            and sim_type in jobdef["jobDefinitionName"]
        ):
            return jobdef["jobDefinitionArn"]


def queue_sim(sim_request):
    client = boto3.client("batch", region_name="us-east-1")

    job_template = {
        "jobQueue": get_jq(client, sim_request["machine"]),
        "jobDefinition": get_jd(client, sim_request["sim_type"], sim_request["machine"]),
        "jobName": sim_request["job_name"],
        "parameters": {"run_id": sim_request["run_id"], "run_type": sim_request["run_type"]},
        "retryStrategy": {"attempts": 10, "evaluateOnExit": [{"action": "RETRY", "onStatusReason": "Host EC2*"}]},
    }

    submissionResult = client.submit_job(**job_template)

    return submissionResult
