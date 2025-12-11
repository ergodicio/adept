from abc import ABC, abstractmethod
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


class AdroitModule(eqx.Module, ABC):
    with_mlflow: bool

    @staticmethod
    def create_mlflow_run_in_callback(dummy_arg):
        def create_mlflow_run(dummy_arg):
            with mlflow.start_run() as run:
                return MLRunId(run.info.run_id)

        return jax.experimental.io_callback(
            create_mlflow_run, MLRunId.example(), dummy_arg, sharding=None, ordered=False
        )

    def call_in_callback(self, callable, *args, mlflow_run_id, **kwargs):
        def call_with_byte_array_id(*args, mlflow_run_id, **kwargs):
            callable(*args, mlflow_run_id=str(mlflow_run_id), **kwargs)

        jax.debug.callback(call_with_byte_array_id, *args, mlflow_run_id=mlflow_run_id, **kwargs)

    def __call__(self, *args, mlflow_batch_num=None, **kwargs):
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
            run_id = AdroitModule.create_mlflow_run_in_callback(mlflow_batch_num)
        else:
            run_id = None

        if self.with_mlflow:
            self.call_in_callback(self.pre_logging, *args, mlflow_run_id=run_id, **kwargs)

        result = self.call(*args, mlflow_run_id=run_id, **kwargs)

        if self.with_mlflow:
            self.call_in_callback(self.post_logging, result, *args, mlflow_run_id=run_id, **kwargs)

        return result

    @abstractmethod
    def call(self, *args, **kwargs):
        """
        The computation to wrap.
        """
        pass

    @abstractmethod
    def pre_logging(self, *args, mlflow_run_id=None, **kwargs):
        """
        Any mlflow logging to be performed before the computation, such as logging params.
        """
        pass

    @abstractmethod
    def post_logging(self, result, *args, mlflow_run_id=None, **kwargs):
        """
        Any mlflow logging to be performed after the computation.
        """
        pass
