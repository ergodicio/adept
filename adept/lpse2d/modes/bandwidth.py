from typing import Dict

from jax import numpy as jnp
from jax.random import normal, PRNGKey
import numpy as np
from diffrax import diffeqsolve, SaveAt
from equinox import filter_value_and_grad, filter_jit

from adept.lpse2d.run_helpers import get_diffeqsolve_quants


def get_apply_func(cfg):
    """
    This function applies models or weights to the state and args or just returns the config values
    that have already been initialized if neither are present

    In the case of a parameter learning problem, or the case where learned parameters
    are loaded, it goes into the optimization condition, where the learned parameters are
    applied to the state and args

    Otherwise, this function will use an NN to generate the modification to the state and args.
    The NN is typically a function of some other variables in state and args


    """

    def _unnorm_weights_(amps_and_phases: Dict, these_args: Dict):
        amps = amps_and_phases["amps"]
        phases = amps_and_phases["phases"]

        these_args["drivers"]["E0"]["amplitudes"] = jnp.tanh(amps)
        these_args["drivers"]["E0"]["initial_phase"] = jnp.tanh(phases)

        these_args["drivers"]["E0"]["delta_omega"] = jnp.linspace(
            -cfg["drivers"]["E0"]["delta_omega_max"],
            cfg["drivers"]["E0"]["delta_omega_max"],
            num=cfg["drivers"]["E0"]["num_colors"],
        )

        these_args["drivers"]["E0"]["amplitudes"] *= 2.0  # from [-1, 1] to [-2, 2]
        these_args["drivers"]["E0"]["amplitudes"] -= 2.0  # from [-2, 2] to [-4, 0]
        these_args["drivers"]["E0"]["amplitudes"] = jnp.power(
            10.0, these_args["drivers"]["E0"]["amplitudes"]
        )  # from [-4, 0] to [1e-4, 1]
        these_args["drivers"]["E0"]["amplitudes"] /= jnp.sum(these_args["drivers"]["E0"]["amplitudes"])
        these_args["drivers"]["E0"]["initial_phase"] *= jnp.pi  # from [-1, 1] to [-pi, pi]

        return these_args

    if "train" in cfg["mode"]:

        def apply_fn(models, _state_, _args_):
            this_model = models["bandwidth"]
            L = float(cfg["density"]["gradient scale length"].strip("um"))
            I0 = float(cfg["units"]["laser intensity"].strip("W/cm^2"))
            Te = float(cfg["units"]["reference electron temperature"].strip("eV"))

            Te = (Te - 3000) / 2000
            L = (L - 300) / 500
            I0 = (jnp.log10(I0) - 15) / 2

            model_outputs = this_model(jnp.array([Te, L, I0]))
            _args_ = _unnorm_weights_(model_outputs, _args_)

            return model_outputs, _state_, _args_

    elif "optimize" in cfg["mode"]:

        def apply_fn(models, _state_, _args_):
            if "hyperparams" in cfg["models"]["bandwidth"]:
                these_params = models["bandwidth"](
                    normal(
                        PRNGKey(seed=np.random.randint(2**20)),
                        shape=(cfg["models"]["bandwidth"]["hyperparams"]["input_width"],),
                    )
                )
            else:
                these_params = models["bandwidth"]
            _args_ = _unnorm_weights_(these_params, _args_)
            return these_params, _state_, _args_

    else:

        def apply_fn(models, _state_, _args_):
            print("using config settings for bandwidth")
            return None, _state_, _args_

    return apply_fn


def get_run_fn(cfg):
    """
    This function returns a function that will run the simulation and calculate the gradient
    if specified

    """
    if cfg["mode"] == "train-bandwidth":
        diffeqsolve_quants = get_diffeqsolve_quants(cfg)

        kx = cfg["save"]["kx"] if "kx" in cfg["save"] else cfg["grid"]["kx"]
        ky = cfg["save"]["ky"] if "ky" in cfg["save"] else cfg["grid"]["ky"]
        kx *= cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]
        ky *= cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]
        dx = cfg["save"]["x"]["dx"] if "dx" in cfg["save"]["x"] else cfg["grid"]["dx"]
        dy = cfg["save"]["y"]["dy"] if "dy" in cfg["save"]["y"] else cfg["grid"]["dy"]
        dt = cfg["save"]["t"]["dt"] if "dt" in cfg["save"]["t"] else cfg["grid"]["dt"]

        apply_models = get_apply_func(cfg)

        def _run_(_models_, _state_, _args_, time_quantities: Dict):

            model_output, _state_, _args_ = apply_models(_models_, _state_, _args_)
            solver_result = diffeqsolve(
                terms=diffeqsolve_quants["terms"],
                solver=diffeqsolve_quants["solver"],
                t0=time_quantities["t0"],
                t1=time_quantities["t1"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=_state_,
                args=_args_,
                saveat=SaveAt(**diffeqsolve_quants["saveat"]),
            )

            phi_k = jnp.fft.fft2(solver_result.ys["epw"].view(jnp.complex128), axes=(1, 2))
            ex_k = kx[None, :, None] * phi_k
            ey_k = ky[None, None, :] * phi_k
            e_sq = jnp.sum(jnp.abs(ex_k) ** 2.0 + jnp.abs(ey_k) ** 2.0) * dx * dy * dt
            loss = jnp.log10(e_sq)
            loss_dict = {"loss": loss}
            if cfg["models"]["bandwidth"]["type"] == "VAE":
                loss += jnp.sum(model_output["kl_loss"])
                loss_dict["kl_loss"] = jnp.sum(model_output["kl_loss"])

            return loss, {"solver_result": solver_result, "state": _state_, "args": _args_, "loss_dict": loss_dict}

        return filter_jit(filter_value_and_grad(_run_, has_aux=True))

    elif cfg["mode"] == "optimize-bandwidth":
        diffeqsolve_quants = get_diffeqsolve_quants(cfg)

        kx = cfg["save"]["kx"] if "kx" in cfg["save"] else cfg["grid"]["kx"]
        ky = cfg["save"]["ky"] if "ky" in cfg["save"] else cfg["grid"]["ky"]
        kx *= cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]
        ky *= cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]
        dx = cfg["save"]["x"]["dx"] if "dx" in cfg["save"]["x"] else cfg["grid"]["dx"]
        dy = cfg["save"]["y"]["dy"] if "dy" in cfg["save"]["y"] else cfg["grid"]["dy"]
        dt = cfg["save"]["t"]["dt"] if "dt" in cfg["save"]["t"] else cfg["grid"]["dt"]

        apply_parameters = get_apply_func(cfg)

        def _run_(_models_, _state_, _args_, time_quantities: Dict):

            _, _state_, _args_ = apply_parameters(_models_, _state_, _args_)
            solver_result = diffeqsolve(
                terms=diffeqsolve_quants["terms"],
                solver=diffeqsolve_quants["solver"],
                t0=time_quantities["t0"],
                t1=time_quantities["t1"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=_state_,
                args=_args_,
                saveat=SaveAt(**diffeqsolve_quants["saveat"]),
            )

            phi_k = jnp.fft.fft2(solver_result.ys["epw"][-30:].view(jnp.complex128), axes=(1, 2))
            ex_k = kx[None, :, None] * phi_k
            ey_k = ky[None, None, :] * phi_k
            e_sq = jnp.sum(jnp.abs(ex_k) ** 2.0 + jnp.abs(ey_k) ** 2.0) * dx * dy * dt
            loss = jnp.log10(e_sq)
            loss_dict = {"loss": loss}
            return loss, {"solver_result": solver_result, "state": _state_, "args": _args_, "loss_dict": loss_dict}

        return filter_jit(filter_value_and_grad(_run_, has_aux=True))
    else:
        raise NotImplementedError("has to have train or optimize in the mode field of the config file")
