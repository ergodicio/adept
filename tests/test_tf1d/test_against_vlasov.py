#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml

import numpy as np
from jax import config

from adept import ergoExo

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import xarray as xr
from adept import electrostatic


def _modify_defaults_(defaults):
    rand_k0 = 0.358

    wepw = np.sqrt(1.0 + 3.0 * rand_k0**2.0)
    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)

    defaults["physics"]["landau_damping"] = True
    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(wepw)
    xmax = float(2.0 * np.pi / rand_k0)
    # defaults["save"]["field"]["xmax_to_store"] = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax
    defaults["save"]["kx"]["kxmax"] = rand_k0
    defaults["mlflow"]["experiment"] = "test-against-vlasov"

    return defaults, float(np.imag(root))


def test_single_resonance():
    with open("tests/test_tf1d/configs/vlasov_comparison.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    # modify config
    mod_defaults, actual_damping_rate = _modify_defaults_(defaults)

    exo = ergoExo()
    exo.setup(mod_defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]
    vds = xr.open_dataset("tests/test_tf1d/vlasov-reference/all-fields-kx.nc", engine="h5netcdf")

    nk1_fluid = result.ys["kx"]["electron"]["n"]["mag"][:, 1]
    nk1_vlasov = vds["n-(k_x)"][:, 1].data
    t_fluid = result.ts
    t_vlasov = vds.coords["t"].data
    fluid_slc = slice(80, 160)
    vlasov_slc = slice(700, 850)

    vlasov_damping_rate = np.mean(
        np.gradient(nk1_vlasov[vlasov_slc], (t_vlasov[1] - t_vlasov[0])) / nk1_vlasov[vlasov_slc]
    )
    fluid_damping_rate = np.mean(np.gradient(nk1_fluid[fluid_slc], (t_fluid[1] - t_fluid[0])) / nk1_fluid[fluid_slc])

    print(f"{vlasov_damping_rate=}, {fluid_damping_rate=}")
    print(f"{np.amax(nk1_vlasov)=}, {np.amax(nk1_fluid)=}")

    np.testing.assert_almost_equal(vlasov_damping_rate, fluid_damping_rate, decimal=2)
    np.testing.assert_allclose(np.amax(nk1_fluid), np.amax(nk1_vlasov), rtol=0.05)


if __name__ == "__main__":
    test_single_resonance()
