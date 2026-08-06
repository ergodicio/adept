"""Tests for the IAW turbulence module (solver: vlasov-1d-iaw).

The module rides on the Boltzmann-electron field solver and adds an ``nk``
save stream (low-|k| complex charge-density spectrum, densely sampled) plus
IAW-specific post-processing artifacts.
"""

import os

import mlflow
import numpy as np
import yaml

from adept import ergoExo


def test_iaw_module_nk_save_and_postprocess():
    """A short driven run must produce the nk stream and IAW artifacts."""
    with open("tests/test_vlasov1d/configs/boltzmann_iaw.yaml") as file:
        config = yaml.safe_load(file)

    config["solver"] = "vlasov-1d-iaw"
    config["grid"]["tmax"] = 200.0
    config["save"]["fields"]["t"]["nt"] = 51
    config["save"]["ion"]["main"]["t"].update(tmax=200.0, nt=5)
    config["density"]["species-ion-background"]["amplitude"] = 0.0
    config["drivers"]["ex_stochastic"] = {"modes": [1, 2], "amplitude": 1.0e-3, "tau": 50.0, "seed": 11}
    config["iaw_diagnostics"] = {"nk_modes": 16, "nk_nt": 101, "spectrum_window": [0.5, 1.0]}
    config["mlflow"]["run"] = "iaw-module-smoke"

    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    # nk stream: (nt, 2, nk_modes + 1), finite, and the mean (m=0) mode is
    # the constant background density in the rfft normalization used
    assert "nk" in solver_result.ys
    nk = np.array(solver_result.ys["nk"])
    assert nk.shape == (101, 2, 17)
    assert np.all(np.isfinite(nk))
    np.testing.assert_allclose(nk[:, 0, 0], 1.0, rtol=1e-2)
    assert np.all(np.abs(nk[:, 1, 0]) < 1e-12), "m=0 mode of a real field must be real"

    # The driven modes must carry power by the end of the run
    late_power = nk[-1, 0, 1:3] ** 2 + nk[-1, 1, 1:3] ** 2
    assert np.all(late_power > 0)

    # post_process returned the nk dataset
    assert "nk" in datasets
    assert "P" in datasets["nk"]
    assert np.all(np.asarray(datasets["nk"]["P"]) >= 0)
    assert datasets["nk"].attrs["lambda_De"] == 1.0


def test_iaw_module_artifacts_written(tmp_path):
    """post_process must write the IAW plots and nk netcdf to the artifact dir."""
    with open("tests/test_vlasov1d/configs/boltzmann_iaw.yaml") as file:
        config = yaml.safe_load(file)

    config["solver"] = "vlasov-1d-iaw"
    config["grid"]["tmax"] = 100.0
    config["save"]["fields"]["t"]["nt"] = 26
    config["save"]["ion"]["main"]["t"].update(tmax=100.0, nt=3)
    config["drivers"]["ex_stochastic"] = {"modes": [1], "amplitude": 1.0e-3, "tau": 50.0, "seed": 3}
    config["iaw_diagnostics"] = {"nk_modes": 8, "nk_nt": 51}
    config["mlflow"]["run"] = "iaw-module-artifacts"

    from adept.vlasov1d import IAWTurbulence1D

    module = IAWTurbulence1D(config)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    run_output = module({})

    td = str(tmp_path)
    # post_process logs a timing metric; scope an explicit run so the implicit
    # one mlflow would otherwise create doesn't stay active and leak into
    # other tests in this process
    with mlflow.start_run(run_name="iaw-module-artifacts-post"):
        module.post_process(run_output, td)

    for rel in [
        "binary/nk.nc",
        "plots/iaw/density_spectrum.png",
        "plots/iaw/nk_spectrogram.png",
        "plots/dists/ion.main/phase_space_dfx.png",
    ]:
        assert os.path.exists(os.path.join(td, rel)), f"missing artifact: {rel}"
