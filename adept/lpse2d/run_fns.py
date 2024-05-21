from typing import Dict
from diffrax import diffeqsolve, SaveAt
import jax.numpy as jnp
from equinox import filter_value_and_grad, filter_jit


def bandwidth(cfg, diffeqsolve_quants, state, indep_quants: Dict, mode="bwd"):

    def _run_(this_model, _args_, time_quantities: Dict):

        if cfg["drivers"]["E0"]["amplitude_shape"] == "ML":
            Te = (indep_quants["Te"] - 3000) / 2000
            L = (indep_quants["L"] - 300) / 200
            I0 = (jnp.log10(indep_quants["I0"]) - 15) / 2

            outputs = this_model(jnp.array([Te, L, I0]))
            amps = outputs["amps"]
            phases = outputs["phases"]

            # _args_ = {"drivers": {"E0": {}}}
            _args_["drivers"]["E0"]["amplitudes"] = jnp.tanh(amps)
            _args_["drivers"]["E0"]["initial_phase"] = jnp.tanh(phases)

            _args_["drivers"]["E0"]["delta_omega"] = jnp.linspace(
                -cfg["drivers"]["E0"]["delta_omega_max"],
                cfg["drivers"]["E0"]["delta_omega_max"],
                num=cfg["drivers"]["E0"]["num_colors"],
            )

            _args_["drivers"]["E0"]["amplitudes"] *= 2.0  # from [-1, 1] to [-2, 2]
            _args_["drivers"]["E0"]["amplitudes"] -= 2.0  # from [-2, 2] to [-4, 0]
            _args_["drivers"]["E0"]["amplitudes"] = jnp.power(
                10.0, _args_["drivers"]["E0"]["amplitudes"]
            )  # from [-4, 0] to [1e-4, 1]
            _args_["drivers"]["E0"]["amplitudes"] /= jnp.sqrt(
                jnp.sum(jnp.square(_args_["drivers"]["E0"]["amplitudes"]))
            )  # normalize

            _args_["drivers"]["E0"]["initial_phase"] *= jnp.pi  # from [-1, 1] to [-pi, pi]
        else:
            # _args_["drivers"] = assemble_bandwidth(cfg)
            # raise ValueError("Where we wanted to go")
            print("using config settings for bandwidth")

        result = diffeqsolve(
            terms=diffeqsolve_quants["terms"],
            solver=diffeqsolve_quants["solver"],
            t0=time_quantities["t0"],
            t1=time_quantities["t1"],
            max_steps=cfg["grid"]["max_steps"],  # time_quantities["max_steps"],
            dt0=cfg["grid"]["dt"],
            y0=state,
            args=_args_,
            saveat=SaveAt(**diffeqsolve_quants["saveat"]),
        )

        e_sq = jnp.sum(jnp.abs(result.ys["epw"].view(jnp.complex128)) ** 2)
        loss = jnp.log10(e_sq)
        aux_out = [result, _args_["drivers"]]
        if cfg["model"]["type"] == "VAE":
            loss += jnp.sum(outputs["kl_loss"])
            aux_out += [outputs["kl_loss"]]

        return loss, aux_out

    if mode == "bwd":
        return filter_jit(filter_value_and_grad(_run_, has_aux=True))
    else:
        return filter_jit(_run_)
