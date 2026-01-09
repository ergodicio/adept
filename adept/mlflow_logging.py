from abc import abstractmethod
from functools import wraps

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import patched_mlflow as mlflow


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


def create_mlflow_run(
    dummy_arg,
    experiment_id=None,
    run_name=None,
    parent_run_id=None,
    tags=None,
    description=None,
    log_system_metrics=None,
) -> MLRunId:
    """
    Create an mlflow run from within a JIT-compiled function.

    Args:
        dummy_arg: Ensures multiple runs are opened when vmapping.
                   Use 0 outside vmap, jnp.arange(batch_size) when vmapping.
        parent_run_id: If provided, must be an MLRunId (not a string).
        **kwargs: Passed to mlflow.start_run.

    Returns:
        MLRunId usable in @mlflow_callback decorated functions.
    """
    if parent_run_id is not None and not isinstance(parent_run_id, MLRunId):
        raise TypeError("parent_run_id must be an MLRunId, not a string")

    def _create_run(dummy_arg, parent_run_id=None):
        nested = parent_run_id is not None
        parent_run_id_str = str(parent_run_id) if nested else None

        with mlflow.start_run(
            parent_run_id=parent_run_id_str,
            nested=nested,
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
            description=description,
            log_system_metrics=log_system_metrics,
        ) as run:
            return MLRunId(run.info.run_id)

    return jax.experimental.io_callback(
        _create_run,
        MLRunId.example(),
        dummy_arg,
        parent_run_id=parent_run_id,
    )


def mlflow_callback(func):
    """
    Decorator that wraps a logging function to run via jax.debug.callback,
    automatically converting MLRunId to string.

    The wrapped function should accept `mlflow_run_id` as a keyword argument.
    If mlflow_run_id is None, the callback is skipped entirely.

    Example:
        @mlflow_callback
        def log_metrics(loss, accuracy, mlflow_run_id=None):
            mlflow.log_metrics({
                "loss": float(loss),
                "accuracy": float(accuracy)
            }, run_id=mlflow_run_id)

        # Inside jitted code:
        run_id = create_mlflow_run(0)
        log_metrics(loss, accuracy, mlflow_run_id=run_id)
    """

    @wraps(func)
    def wrapper(*args, mlflow_run_id: MLRunId | None = None, **kwargs):
        if mlflow_run_id is None:
            return

        assert isinstance(mlflow_run_id, MLRunId), (
            f"mlflow_run_id must be an MLRunId, got {type(mlflow_run_id).__name__}"
        )

        def _inner(*args, mlflow_run_id: MLRunId, **kwargs):
            func(*args, mlflow_run_id=str(mlflow_run_id), **kwargs)

        jax.debug.callback(_inner, *args, mlflow_run_id=mlflow_run_id, **kwargs)

    return wrapper


class MlflowLoggingModule(eqx.Module):
    """
    Base class for modules that log to mlflow from within JIT-compiled functions.

    For simpler use cases, consider using the standalone `create_mlflow_run` helper
    and `@mlflow_callback` decorator directly instead of subclassing this module.
    """

    with_mlflow: bool

    def __call__(self, *args, mlflow_batch_num=None, mlflow_kwargs=None, **kwargs):
        """
        Params:
            - `*args`: The list of positional arguments to pass to `call`
            - `mlflow_batch_num`: A dummy argument to ensure that multiple mlflow runs
                                  are opened when we are `vmap`ping over `__call__`. Must
                                  be specified by the caller.
                                  Outside of a vmap context, mlflow_batch_num=0 is an appropriate choice.
                                  When vmapping, you should use `mlflow_batch_num=jnp.arange(batch_size)`.
            - `mlflow_kwargs`: A dict of keyword arguments to pass to `mlflow.start_run`. If "parent_run_id"
                               is included, it must be passed as an MLRunId, not a string.
            - `**kwargs`: The dict of keyword args to pass to `call`
        """
        if self.with_mlflow:
            if mlflow_batch_num is None:
                raise ValueError("mlflow_batch_num must be specified")
            mlflow_kwargs = {} if mlflow_kwargs is None else mlflow_kwargs
            run_id = create_mlflow_run(mlflow_batch_num, **mlflow_kwargs)
        else:
            run_id = None

        setup_result = self.setup(*args, mlflow_run_id=run_id, **kwargs)

        mlflow_callback(self.pre_logging)(*args, setup_result=setup_result, mlflow_run_id=run_id, **kwargs)

        result = self.call(*args, setup_result=setup_result, mlflow_run_id=run_id, **kwargs)

        mlflow_callback(self.post_logging)(result, *args, setup_result=setup_result, mlflow_run_id=run_id, **kwargs)

        return result

    @abstractmethod
    def setup(self, *args, mlflow_run_id=None, **kwargs):
        """
        Perform any preprocessing of `*args`, `**kwargs`, and `self` that should
        be logged as part of `self.pre_logging`. For example, computing derived units
        or grid arrays.

        The return value of this function will be piped to the other lifecycle
        methods as the `setup_result` kwarg.

        params:
            - `mlflow_run_id`: The ID of the mlflow run for the current module call
            - `*args`: Any remaining positional arguments to `self.__call__`
            - `*kwargs`: Any remaining keyword arguments to `self.__call__`
        """
        pass

    @abstractmethod
    def call(self, *args, setup_result=None, mlflow_run_id=None, **kwargs):
        """
        The computation to wrap.

        params:
            - `setup_result`: The return value of `setup()`
            - `mlflow_run_id`: The ID of the mlflow run for the current module call
            - `*args`: Any remaining positional arguments to `self.__call__`
            - `*kwargs`: Any remaining keyword arguments to `self.__call__`
        """
        pass

    @abstractmethod
    def pre_logging(self, *args, setup_result=None, mlflow_run_id=None, **kwargs):
        """
        Any mlflow logging to be performed before the computation, such as logging params.

        params:
            - `setup_result`: The return value of `setup()`
            - `mlflow_run_id`: The ID of the mlflow run for the current module call
            - `*args`: Any remaining positional arguments to `self.__call__`
            - `*kwargs`: Any remaining keyword arguments to `self.__call__`
        """
        pass

    @abstractmethod
    def post_logging(self, result, *args, setup_result=None, mlflow_run_id=None, **kwargs):
        """
        Any mlflow logging to be performed after the computation, such as metrics and plot artifacts.

        params:
            - `result`: The return value of `call()`
            - `setup_result`: The return value of `setup()`
            - `mlflow_run_id`: The ID of the mlflow run for the current module call
            - `*args`: Any remaining positional arguments to `self.__call__`
            - `*kwargs`: Any remaining keyword arguments to `self.__call__`
        """
        pass
