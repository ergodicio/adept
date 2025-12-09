import re

import jax
import jax.numpy as jnp

from adept import AdroitModule


class ScaleAndAddModule(AdroitModule):
    s: float

    def call(self, a, mlflow_run_id=None, b=0.0):
        return self.s * a + b

    def pre_logging(self, a, mlflow_run_id=None, b=0.0):
        print(f"PRE_LOGGING | a={a}, b={b}, mlflow_run_id={mlflow_run_id}")

    def post_logging(self, result, a, mlflow_run_id=None, b=0.0):
        print(f"POST_LOGGING | a={a}, b={b}, result={result}, mlflow_run_id={mlflow_run_id}")


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

    captured = capsys.readouterr()
    pattern = r"""PRE_LOGGING | a=2.0, b=4.5, mlflow_run_id=(\w{32})
POST_LOGGING | a=2.0, b=4.5, result=10.5, mlflow_run_id=(\w{32})
"""
    match = re.search(pattern, captured.out)
    assert match is not None
    assert match.group(1) == match.group(2)


def test_pre_and_post_logging_grad(capsys):
    saam = ScaleAndAddModule(True, 3.0)
    jax.grad(saam)(2.0, b=4.5, mlflow_batch_num=0)

    captured = capsys.readouterr()
    pattern = r"""PRE_LOGGING \| a=2.0, b=4.5, mlflow_run_id=(\w{32})
POST_LOGGING \| a=2.0, b=4.5, result=10.5, mlflow_run_id=(\w{32})
"""
    match = re.fullmatch(pattern, captured.out, re.MULTILINE)
    assert match is not None
    assert match.group(1) is not None
    assert match.group(1) == match.group(2)


def test_adroit_module_mlflow_with_vmap(capsys):
    saam = ScaleAndAddModule(True, 3.0)

    result = jax.vmap(saam)(2.0 * jnp.ones(3), mlflow_batch_num=jnp.arange(3), b=4.5 * jnp.ones(3))

    captured = capsys.readouterr()

    pre_logging_pattern = r"PRE_LOGGING \| a=2.0, b=4.5, mlflow_run_id=(\w{32})"
    post_logging_pattern = r"POST_LOGGING \| a=2.0, b=4.5, result=10.5, mlflow_run_id=(\w{32})"

    pre_logging_matches = list(re.finditer(pre_logging_pattern, captured.out))
    post_logging_matches = list(re.finditer(post_logging_pattern, captured.out))

    assert len(pre_logging_matches) == 3
    assert len(set([m.group(1) for m in pre_logging_matches])) == 3
    assert len(post_logging_matches) == 3
    assert len(set([m.group(1) for m in post_logging_matches])) == 3

    assert set([m.group(1) for m in pre_logging_matches]) == set([m.group(1) for m in post_logging_matches])
