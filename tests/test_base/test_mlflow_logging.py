import re

import jax
import jax.numpy as jnp

import adept.patched_mlflow as mlflow
from adept import MlflowLoggingModule
from adept.mlflow_logging import MLRunId


def parse_unique_mlflow_run_id(capsys):
    captured = capsys.readouterr()
    pattern = r"""PRE_LOGGING \| a=2.0, b=4.5, abs2\(s\)=9.0, mlflow_run_id=(\w{32})
POST_LOGGING \| a=2.0, b=4.5, abs2\(s\)=9.0, result=10.5, mlflow_run_id=(\w{32})
"""
    print(captured.out)
    match = re.search(pattern, captured.out)
    assert match is not None
    assert match.group(1) == match.group(2)
    return match.group(1)


class ScaleAndAddModule(MlflowLoggingModule):
    s: float

    def setup(self, *args, **kwargs):
        return {"abs2(s)": jnp.abs(self.s) ** 2}

    def call(self, a, mlflow_run_id=None, b=0.0, **kwargs):
        return self.s * a + b

    def pre_logging(self, a, setup_result, mlflow_run_id, b):
        print(f"PRE_LOGGING | a={a}, b={b}, abs2(s)={setup_result['abs2(s)']}, mlflow_run_id={mlflow_run_id}")

    def post_logging(self, result, a, setup_result, mlflow_run_id, b):
        print(
            f"POST_LOGGING | a={a}, b={b}, abs2(s)={setup_result['abs2(s)']}, "
            + f"result={result}, mlflow_run_id={mlflow_run_id}"
        )


def test_adroit_module_base_call():
    saam = ScaleAndAddModule(True, 3.0)

    result = saam(2.0, b=4.5, mlflow_batch_num=0)
    assert result == 10.5


def test_adroit_module_grad_of_arg():
    saam = ScaleAndAddModule(True, 3.0)

    result = jax.grad(saam)(2.0, b=4.5, mlflow_batch_num=0)
    assert result == 3.0


def test_adroit_module_grad_of_kwarg():
    saam = ScaleAndAddModule(True, 3.0)

    def f(b):
        return saam(2.0, b=b, mlflow_batch_num=0)

    result = jax.grad(f)(4.5)
    assert result == 1.0


def test_adroit_module_vmap():
    saam = ScaleAndAddModule(True, 3.0)

    result = jax.vmap(saam)(jnp.linspace(0.0, 10.0, 11), mlflow_batch_num=jnp.arange(11), b=4.5 * jnp.ones(11))
    assert (result == jnp.linspace(4.5, 34.5, 11)).all()


def test_adroit_module_vmap_of_grad():
    saam = ScaleAndAddModule(True, 3.0)

    result = jax.vmap(jax.grad(saam))(
        jnp.linspace(0.0, 10.0, 11), mlflow_batch_num=jnp.arange(11), b=4.5 * jnp.ones(11)
    )
    assert (result == 3 * jnp.ones(11)).all()


def test_pre_and_post_logging(capsys):
    saam = ScaleAndAddModule(True, 3.0)
    saam(2.0, b=4.5, mlflow_batch_num=0)

    parse_unique_mlflow_run_id(capsys)


def test_pre_and_post_logging_grad(capsys):
    saam = ScaleAndAddModule(True, 3.0)
    jax.grad(saam)(2.0, b=4.5, mlflow_batch_num=0)

    parse_unique_mlflow_run_id(capsys)


def test_adroit_module_mlflow_with_vmap(capsys):
    saam = ScaleAndAddModule(True, 3.0)

    result = jax.vmap(saam)(2.0 * jnp.ones(3), mlflow_batch_num=jnp.arange(3), b=4.5 * jnp.ones(3))

    captured = capsys.readouterr()

    pre_logging_pattern = r"PRE_LOGGING \| a=2.0, b=4.5, abs2\(s\)=9.0, mlflow_run_id=(\w{32})"
    post_logging_pattern = r"POST_LOGGING \| a=2.0, b=4.5, abs2\(s\)=9.0, result=10.5, mlflow_run_id=(\w{32})"

    pre_logging_matches = list(re.finditer(pre_logging_pattern, captured.out))
    post_logging_matches = list(re.finditer(post_logging_pattern, captured.out))

    assert len(pre_logging_matches) == 3
    assert len(set([m.group(1) for m in pre_logging_matches])) == 3
    assert len(post_logging_matches) == 3
    assert len(set([m.group(1) for m in post_logging_matches])) == 3

    assert set([m.group(1) for m in pre_logging_matches]) == set([m.group(1) for m in post_logging_matches])


def test_mlflow_logging_module_in_manually_created_parent_run(capsys):
    with mlflow.start_run() as parent_run:
        saam = ScaleAndAddModule(True, 3.0)
        parent_run_id = MLRunId(parent_run.info.run_id)
        saam(2.0, b=4.5, mlflow_batch_num=0, mlflow_kwargs={"parent_run_id": parent_run_id})

    run_id = parse_unique_mlflow_run_id(capsys)
    child_runs = mlflow.search_runs(filter_string=f"attributes.run_id = '{run_id}'", output_format="list")
    assert len(child_runs) == 1
    assert child_runs[0].data.tags["mlflow.parentRunId"] == parent_run.info.run_id


def test_adroit_module_base_call_mlflow_kwargs(capsys):
    saam = ScaleAndAddModule(True, 3.0)

    result = saam(
        2.0,
        b=4.5,
        mlflow_batch_num=0,
        mlflow_kwargs={"run_name": "ScaleAndAdd", "description": "really great", "tags": {"foo": "bar"}},
    )
    assert result == 10.5

    run_id = parse_unique_mlflow_run_id(capsys)

    runs = mlflow.search_runs(filter_string=f"attributes.run_id = '{run_id}'", output_format="list")
    assert len(runs) == 1
    assert runs[0].data.tags["foo"] == "bar"
    assert runs[0].data.tags["mlflow.note.content"] == "really great"
    assert runs[0].info.run_name == "ScaleAndAdd"


class OuterScalingAndAddingModule(MlflowLoggingModule):
    saam: ScaleAndAddModule
    b: float

    def setup(self, *args, **kwargs):
        pass

    def call(self, a, mlflow_run_id=None, **kwargs):
        return self.saam(a, b=self.b, mlflow_kwargs={"parent_run_id": mlflow_run_id}, mlflow_batch_num=0)

    def pre_logging(self, *args, **kwargs):
        pass

    def post_logging(self, *args, **kwargs):
        pass


def test_nested_mlflow_logging_modules(capsys):
    saam = ScaleAndAddModule(True, 3.0)
    b = 4.5
    outer_module = OuterScalingAndAddingModule(True, saam, b)

    outer_module(2.0, mlflow_batch_num=0)

    child_run_id = parse_unique_mlflow_run_id(capsys)
    child_runs = mlflow.search_runs(filter_string=f"attributes.run_id = '{child_run_id}'", output_format="list")
    assert len(child_runs) == 1
    assert child_runs[0].data.tags["mlflow.parentRunId"] is not None


def test_nested_mlflow_logging_modules_vmap(capsys):
    saam = ScaleAndAddModule(True, 3.0)
    b = 4.5
    outer_module = OuterScalingAndAddingModule(True, saam, b)

    jax.vmap(outer_module)(2 * jnp.ones(3), mlflow_batch_num=jnp.arange(3))

    captured = capsys.readouterr()

    print(captured.out)

    pre_logging_pattern = r"PRE_LOGGING \| a=2.0, b=4.5, abs2\(s\)=9.0, mlflow_run_id=(\w{32})"
    post_logging_pattern = r"POST_LOGGING \| a=2.0, b=4.5, abs2\(s\)=9.0, result=10.5, mlflow_run_id=(\w{32})"

    pre_logging_matches = list(re.finditer(pre_logging_pattern, captured.out))
    post_logging_matches = list(re.finditer(post_logging_pattern, captured.out))

    assert len(pre_logging_matches) == 3
    assert len(set([m.group(1) for m in pre_logging_matches])) == 3
    assert len(post_logging_matches) == 3
    assert len(set([m.group(1) for m in post_logging_matches])) == 3

    assert set([m.group(1) for m in pre_logging_matches]) == set([m.group(1) for m in post_logging_matches])
