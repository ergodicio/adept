from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np


class MLRunId(eqx.Module):
    """
    An mlflow run id encoded as a JAX array of bytes.
    This class is needed to pass around the run id in JIT-compiled sections.
    """

    byte_array: jax.Array

    def __init__(self, id: str):
        self.byte_array = jnp.frombuffer(id.encode("UTF-8"), dtype=jnp.int8)

    def __str__(self):
        return bytes(np.array(self.byte_array)).decode("UTF-8")

    @staticmethod
    def example():
        "Generate an MLRunId of the correct length for shape,dtype purposes"
        return MLRunId("0" * 32)


class MlflowLoggingModule(eqx.Module):
    with_mlflow: bool

    @staticmethod
    def create_mlflow_run_in_callback(dummy_arg, parent_run_id_bytes=None):
        def create_mlflow_run(dummy_arg, parent_run_id_bytes=None):
            parent_run_id_str = None if parent_run_id_bytes is None else str(parent_run_id_bytes)
            with mlflow.start_run(parent_run_id=parent_run_id_str, nested=(parent_run_id_str is not None)) as run:
                return MLRunId(run.info.run_id)

        return jax.experimental.io_callback(
            create_mlflow_run, MLRunId.example(), dummy_arg, parent_run_id_bytes=parent_run_id_bytes
        )

    def __call__(self, *args, mlflow_batch_num=None, mlflow_parent_run_id_bytes=None, **kwargs):
        """
        Params:
            - `*args`: The list of positional arguments to pass to `call`
            - `mlflow_batch_num`: A dummy argument to ensure that multiple mlflow runs
                                  are opened when we are `vmap`ping over `__call__`. Must
                                  be specified by the caller.
                                  Outside of a vmap context, mlflow_batch_num=0 is an appropriate choice.
                                  When vmapping, you should use `mlflow_batch_num=jnp.arange(batch_size)`.
            - `**kwargs`: The dict of keyword args to pass to `call`
        """
        if self.with_mlflow:
            if mlflow_batch_num is None:
                raise ValueError("mlflow_batch_num must be specified")
            run_id = MlflowLoggingModule.create_mlflow_run_in_callback(
                mlflow_batch_num, parent_run_id_bytes=mlflow_parent_run_id_bytes
            )
        else:
            run_id = None

        if self.with_mlflow:
            jax.debug.callback(self.pre_logging, *args, mlflow_run_id_bytes=run_id, **kwargs)

        result = self.call(*args, mlflow_run_id_bytes=run_id, **kwargs)

        if self.with_mlflow:
            jax.debug.callback(self.post_logging, result, *args, mlflow_run_id_bytes=run_id, **kwargs)

        return result

    @abstractmethod
    def call(self, *args, mlflow_run_id_bytes=None, **kwargs):
        """
        The computation to wrap.
        """
        pass

    @abstractmethod
    def pre_logging(self, *args, mlflow_run_id_bytes=None, **kwargs):
        """
        Any mlflow logging to be performed before the computation, such as logging params.
        """
        pass

    @abstractmethod
    def post_logging(self, result, *args, mlflow_run_id_bytes=None, **kwargs):
        """
        Any mlflow logging to be performed after the computation.
        """
        pass
