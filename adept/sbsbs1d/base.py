from typing import Dict
from adept import ADEPTModule


class BaseSteadyStateBackwardStimulatedBrilloiunScattering(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def write_units(self) -> Dict:
        return {}

    def get_derived_quantities(self) -> Dict:
        return super().get_derived_quantities()

    def __call__(self, trainable_modules: Dict, args: Dict):
        state = self.state

        if args is None:
            args = self.args

        for name, module in trainable_modules.items():
            state, args = module(self.state, args)

        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=self.cfg["grid"]["max_steps"],
            dt0=self.cfg["grid"]["dt"],
            y0=state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return {"solver result": solver_result, "args": args}
