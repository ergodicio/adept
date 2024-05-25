from typing import Dict
from diffrax import diffeqsolve, SaveAt
from equinox import filter_jit


from adept.lpse2d.modes import bandwidth
from adept.lpse2d.run_helpers import get_diffeqsolve_quants


def get_apply_func(cfg):
    apply_fncs = {}
    if "models" in cfg:
        if "bandwidth" in cfg["models"]:
            apply_fncs["bandwidth"] = bandwidth.get_apply_func(cfg)
        else:
            raise NotImplementedError("Only bandwidth mode is implemented")

    def apply_fn(models, _state_, _args_):
        model_out = {}
        for key, fn in apply_fncs.items():
            model_out[key], _state_, _args_ = fn(models, _state_, _args_)
        return model_out, _state_, _args_

    return apply_fn


def get_run_fn(cfg):

    if "mode" in cfg:
        if "bandwidth" in cfg["mode"]:
            _run_ = bandwidth.get_run_fn(cfg)
        else:
            raise NotImplementedError("Only bandwidth mode is implemented")
    else:
        # if no mode is specified, then we are just running the simulation
        diffeqsolve_quants = get_diffeqsolve_quants(cfg)
        apply_models = get_apply_func(cfg)

        @filter_jit
        def _run_(_models_, _state_, _args_, time_quantities: Dict):

            _, _state_, _args_ = apply_models(_models_, _state_, _args_)

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

            return {"solver_result": solver_result, "state": _state_, "args": _args_}

    return _run_
