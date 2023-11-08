#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import tempfile, os, yaml

import numpy as np
from jax.config import config
import pytest

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import mlflow

from theory import electrostatic
from utils.runner import run

from matplotlib import pyplot as plt


def _modify_defaults_(defaults, rng, field_solver):
    rand_k0 = np.round(rng.uniform(0.25, 0.4), 3)

    # wepw = np.sqrt(1.0 + 3.0 * rand_k0**2.0)
    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)
    # print(rand_k0, wepw, root)

    defaults["solver"]["field"] = field_solver
    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(np.real(root))
    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    # defaults["save"]["fields"]["x"]["xmax"] = xmax
    defaults["mlflow"]["experiment"] = "test-landau-damping"

    return defaults, float(np.imag(root))


@pytest.mark.parametrize("field_solver", ["poisson", "ampere"])
def test_single_resonance(field_solver):
    with open("tests/test_vlasov2d/configs/damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    # modify config
    rng = np.random.default_rng()
    mod_defaults, actual_damping_rate = _modify_defaults_(defaults, rng, field_solver)

    # run
    mlflow.set_experiment(mod_defaults["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"]) as mlflow_run:
        result, datasets = run(mod_defaults)

        xax = datasets["fields"].coords["x"].data
        tax = datasets["fields"].coords["t"].data
        dt = tax[1] - tax[0]

        ex = datasets["fields"]["fields-e"].data[:, :, :, 0]
        ek1 = 2.0 / xax.size * np.abs(np.fft.fft(ex, axis=1)[:, 1, 0])
        frslc = slice(-48, -1)
        measured_damping_rate = np.mean(np.gradient(ek1[frslc], dt) / ek1[frslc])
        print(
            f"Landau Damping rate check \n"
            f"measured: {np.round(measured_damping_rate, 5)}, "
            f"actual: {np.round(actual_damping_rate, 5)}, "
        )
        mlflow.log_metrics(
            {"actual damping rate": float(actual_damping_rate), "measured damping rate": float(measured_damping_rate)}
        )

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        ax[0].plot(tax, ek1, label="observed")
        ax[0].plot(tax[frslc], ek1[frslc], label="calculation")
        ax[0].plot(
            tax[frslc],
            ek1[frslc][0] * np.exp(actual_damping_rate * (tax[frslc] - tax[frslc][0])),
            "--",
            label="desired",
        )
        ax[0].set_xlabel("$t (\omega_p^{-1})$", fontsize=14)
        ax[0].set_ylabel("$|\hat{E}|^1$", fontsize=14)
        ax[0].legend(fontsize=14)
        ax[0].grid()
        ax[1].semilogy(tax, ek1, label="observed")
        ax[1].plot(tax[frslc], ek1[frslc], label="calculation")
        ax[1].plot(
            tax[frslc],
            ek1[frslc][0] * np.exp(actual_damping_rate * (tax[frslc] - tax[frslc][0])),
            "--",
            label="desired",
        )
        ax[1].legend(fontsize=14)
        ax[1].set_xlabel("$t (\omega_p^{-1})$", fontsize=14)
        ax[1].set_ylabel("$|\hat{E}|^1$", fontsize=14)
        ax[1].grid()
        with tempfile.TemporaryDirectory() as td:
            fig.savefig(os.path.join(td, "damping.png"), bbox_inches="tight")
            mlflow.log_artifacts(td)

        np.testing.assert_almost_equal(measured_damping_rate, actual_damping_rate, decimal=2)


if __name__ == "__main__":
    test_single_resonance("poisson")
