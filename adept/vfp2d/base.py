"""Compatibility entry point for the arbitrary-harmonic VFP-2D solver."""

import jax.tree_util as jtu
from diffrax import ODETerm, SaveAt, diffeqsolve

from adept._base_ import ADEPTModule, Stepper
from adept.vfp2d.postprocess import VFP2DPostProcessor
from adept.vfp2d.preparation import VFP2DSetup


class BaseVFP2D(VFP2DSetup, ADEPTModule):
    """Legacy lifecycle sharing preparation with the explicit VFP2D builder."""

    def __init__(self, cfg):
        ADEPTModule.__init__(self, cfg)
        VFP2DSetup.__init__(self, cfg)

    def init_diffeqsolve(self):
        step = self.prepare_step()
        self.prepare_save_times()
        self.time_quantities = {"t0": self.tmin, "t1": self.tmax, "max_steps": self.max_steps}
        if self.spatial_sharding is not None:

            def save_fn(_t, state, _args):
                return jtu.tree_map(self.spatial_sharding.replicate, state)

            saveat = SaveAt(ts=self.save_times, fn=save_fn)
        else:
            saveat = SaveAt(ts=self.save_times)
        self.diffeqsolve_quants = {
            "terms": ODETerm(step),
            "solver": Stepper(),
            "saveat": saveat,
        }

    def __call__(self, trainable_modules: dict | None, args: dict | None):
        def solve():
            return diffeqsolve(
                terms=self.diffeqsolve_quants["terms"],
                solver=self.diffeqsolve_quants["solver"],
                t0=self.tmin,
                t1=self.tmax,
                dt0=self.grid.dt,
                max_steps=self.max_steps,
                y0=self.state,
                args=self.args if args is None else args,
                saveat=self.diffeqsolve_quants["saveat"],
            )

        if self.spatial_sharding is not None:
            with self.spatial_sharding.mesh:
                result = solve()
        else:
            result = solve()
        return {"solver result": result}

    def post_process(self, run_output: dict, td: str) -> dict:
        return VFP2DPostProcessor.from_setup(self).post_process(run_output, td)
