from typing import Dict
import equinox as eqx
from diffrax import diffeqsolve, SaveAt, Solution

from adept.lpse2d.modules.bandwidth import Bandwidth
from adept.lpse2d.run_helpers import get_diffeqsolve_quants


class Runner(eqx.Module):
    cfg: Dict
    pre_time_loop_modules: Dict
    post_time_loop_modules: Dict
    diffeqsolve_quants: Dict
    trainable_modules: list

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.pre_time_loop_modules = self.init_pre_time_loop_submodules(cfg)
        self.post_time_loop_modules = {}
        self.diffeqsolve_quants = get_diffeqsolve_quants(cfg)
        self.trainable_modules = cfg["models"]["trainable"].keys() if "trainable" in cfg["models"] else []

    def time_loop(self, state: Dict, args: Dict, time_quantities: Dict) -> Solution:

        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=time_quantities["t0"],
            t1=time_quantities["t1"],
            max_steps=self.cfg["grid"]["max_steps"],
            dt0=self.cfg["grid"]["dt"],
            y0=state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return solver_result

    def init_pre_time_loop_submodules(self, cfg: Dict) -> Dict:
        sub_modules = {}
        if "models" in cfg:
            if "static" in cfg["models"]:
                if "pre_time_loop" in cfg["models"]["static"]:
                    for key, model in cfg["models"]["static"].items():
                        if "bandwidth" == key:
                            sub_modules["bandwidth"] = Bandwidth(cfg)
                        else:
                            raise NotImplementedError("Only bandwidth mode is implemented")
        return sub_modules

    def apply_before_timeloop(self, models, _state_, _args_):
        model_out = {}
        for key, sub_module in self.pre_time_loop_modules.items():
            model_out[key], _state_, _args_ = sub_module(_state_, _args_)

        for key in self.trainable_modules:
            model_out[key], _state_, _args_ = models[key](_state_, _args_)
        return model_out, _state_, _args_

    def apply_after_timeloop(self, models, _state_, _args_):
        model_out = {}
        for key, sub_module in self.post_time_loop_modules.items():
            model_out[key], _state_, _args_ = sub_module(models, _state_, _args_)
        return model_out, _state_, _args_

    def __call__(self, models, time_quantities, y, args):
        preloop_model_out, y, args = self.apply_before_timeloop(models, y, args)
        y = self.time_loop(y, args, time_quantities)
        postloop_model_out, y, args = self.apply_after_timeloop(models, y, args)

        return preloop_model_out | postloop_model_out, y, args
