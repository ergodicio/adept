import os
import pickle
import shutil
import time

import equinox as eqx
import jax
import yaml
from mlflow.tracking import MlflowClient
from pint import Quantity

from . import patched_mlflow as mlflow
from .functions import (
    EnvelopeFunction,
    ExponentialFunction,
    LinearFunction,
    NoiseProfile,
    SineFunction,
    SpaceTimeEnvelopeFunction,
    UniformFunction,
)


def flatten_dict(d, delimiter=".", _prefix=""):
    items = {}
    for k, v in d.items():
        key = f"{_prefix}{delimiter}{k}" if _prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, delimiter, key))
        else:
            items[key] = v
    return items


def log_params(cfg):
    flattened_dict = flatten_dict(cfg, delimiter=".")
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
    with open(dest_file_path) as file:
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


def all_reduce_gradients(gradients: list, num: int):
    """
    Averages gradients across multiple devices and returns a single gradient pytree.

    The gradients object is a list of a pytree, one for each device. Each of those pytrees contains a gradient value
    at the right attribute or location. The algorithm should calculate the average of each of those gradient values
    across devices and return a single pytree with the same structure as the input pytrees, but with the averaged
    gradient values

    Need to make NaN proof and introduce gradient clipping

    :param gradients: List of gradient dictionaries from each device.
    :param num: Number of devices.
    """

    # This is the simple version without nan and clipping
    # if num > 1:
    #     def _safe_add(a1, a2):
    #         if a1 is None:
    #             return a2
    #         else:
    #             return a1 + a2

    #     def _is_none(x):
    #         return x is None

    #     def _safe_divide(a1):
    #         if a1 is None:
    #             return a1
    #         else:
    #             return a1 / num

    #     summed_gradients = jax.tree.map(_safe_add, gradients[0], gradients[1], is_leaf=_is_none)
    #     for i in range(2, num):
    #         summed_gradients = jax.tree_map(_safe_add, summed_gradients, gradients[i], is_leaf=_is_none)

    #     average_gradient = jax.tree_map(_safe_divide, summed_gradients, is_leaf=_is_none)
    # else:
    #     average_gradient = gradients[0]

    # this is the better version with nan
    # if num > 1:
    #     def _safe_add(a1, a2):
    #         if a1 is None:
    #             return a2
    #         elif a2 is None:
    #             return a1
    #         else:
    #             return jax.numpy.where(jax.numpy.isnan(a1), a2, jax.numpy.where(jax.numpy.isnan(a2), a1, a1 + a2))

    #     def _is_none(x):
    #         return x is None

    #     def _safe_divide(a1):
    #         if a1 is None:
    #             return a1
    #         else:
    #             return jax.numpy.where(jax.numpy.isnan(a1), a1, a1 / num)

    #     summed_gradients = jax.tree_map(_safe_add, gradients[0], gradients[1], is_leaf=_is_none)
    #     for i in range(2, num):
    #         summed_gradients = jax.tree_map(_safe_add, summed_gradients, gradients[i], is_leaf=_is_none)

    #     average_gradient = jax.tree_map(_safe_divide, summed_gradients, is_leaf=_is_none)
    # else:
    #     average_gradient = gradients[0]

    # this is the best version with nan and clipping
    if num > 1:

        def _safe_add(a1, a2):
            if a1 is None:
                return a2
            elif a2 is None:
                return a1
            else:
                return jax.numpy.where(jax.numpy.isnan(a1), a2, jax.numpy.where(jax.numpy.isnan(a2), a1, a1 + a2))

        def _is_none(x):
            return x is None

        def _safe_divide(a1):
            if a1 is None:
                return a1
            else:
                return jax.numpy.where(jax.numpy.isnan(a1), a1, a1 / num)

        def _clip_gradient(g):
            if g is None:
                return g
            else:
                return jax.numpy.where(jax.numpy.isnan(g), g, jax.numpy.clip(g, -1e3, 1e3))

        summed_gradients = jax.tree.map(_safe_add, gradients[0], gradients[1], is_leaf=_is_none)
        for i in range(2, num):
            summed_gradients = jax.tree.map(_safe_add, summed_gradients, gradients[i], is_leaf=_is_none)

        average_gradient = jax.tree.map(_safe_divide, summed_gradients, is_leaf=_is_none)
        average_gradient = jax.tree.map(_clip_gradient, average_gradient, is_leaf=_is_none)

    return average_gradient


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


def is_scalar(value) -> bool:
    """Check if a value is a scalar (not an array)."""
    import jax.numpy as jnp
    import numpy as np

    if isinstance(value, (int, float, bool, str, type(None))):
        return True
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return False
    # Handle numpy scalar types
    if hasattr(value, "ndim") and value.ndim == 0:
        return True
    return False


def filter_scalars(d: dict) -> dict:
    """Filter a dict to only include scalar values (recursively).

    Useful for logging domain object contents before array quantities are needed.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            filtered = filter_scalars(v)
            if filtered:  # Only include non-empty dicts
                result[k] = filtered
        elif is_scalar(v):
            result[k] = v
    return result
