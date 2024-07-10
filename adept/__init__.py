from typing import Dict, Tuple, Callable
import jax.flatten_util
import os, time, tempfile, yaml, pickle


from diffrax import Solution, Euler, RESULTS
import mlflow, jax, numpy as np
from jax import numpy as jnp


from utils import misc


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))


class Stepper(Euler):
    """
    This is just a dummy stepper

    :param cfg:
    """

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


class ADEPTModule:
    """
    This class is the base class for all the ADEPT modules. It defines the interface that all the ADEPT modules must implement.

    Args:
        cfg: The configuration dictionary

    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def post_process(self, run_output: Dict, td: str):
        pass

    def write_units(self) -> Dict:
        return {}

    def init_diffeqsolve(self):
        pass

    def get_derived_quantities(self):
        pass

    def get_solver_quantities(self):
        pass

    def get_save_func(self):
        pass

    def init_state_and_args(self) -> Dict:
        return {}

    def init_modules(self) -> Dict:
        return {}

    def __call__(self, trainable_modules: Dict, args: Dict):
        return {}

    def vg(self, trainable_modules: Dict, args: Dict):
        raise NotImplementedError(
            "This is the base class and does not have a gradient implemented. This is "
            + "likely because there is no metric in place. Subclass this class and implement the gradient"
        )
        # return eqx.filter_value_and_grad(self.__call__)(trainable_modules)


class ergoExo:
    """
    This class is the main interface for running a simulation. It is responsible for calling all the ADEPT modules in the right order
    and logging parameters and results to mlflow.

    This helps decouple the numerical solvers from the experiment management

    """

    def __init__(self, mlflow_run_id: str = None, mlflow_nested: bool = None) -> None:

        self.mlflow_run_id = mlflow_run_id
        # if mlflow_run_id is not None:
        #     assert self.mlflow_nested is not None
        if mlflow_nested is None:
            self.mlflow_nested = False
        else:
            self.mlflow_nested = mlflow_nested

        if "BASE_TEMPDIR" in os.environ:
            self.base_tempdir = os.environ["BASE_TEMPDIR"]
        else:
            self.base_tempdir = None

        self.ran_setup = False

    def get_adept_module(self, cfg: Dict) -> ADEPTModule:
        """
        This function returns the helper functions for the given solver

        Args:
            solver: The solver to use


        """
        if cfg["solver"] == "tf-1d":
            from adept.tf1d.base import BaseTwoFluid1D as this_module
        # elif solver == "sh-2d":
        #     from adept.sh2d import helpers
        elif cfg["solver"] == "vlasov-1d":
            from adept.vlasov1d.base import BaseVlasov1D as this_module
        # elif solver == "vlasov-1d2v":
        #     from adept.vlasov1d2v import helpers
        # elif solver == "vlasov-2d":
        #     from adept.vlasov2d import helpers
        elif cfg["solver"] == "envelope-2d":
            from adept.lpse2d.base import BaseLPSE2D as this_module
        elif cfg["solver"] == "vfp-1d":
            from adept.vfp1d.base import BaseVFP1D as this_module
        else:
            raise NotImplementedError("This solver approach has not been implemented yet")

        return this_module(cfg)

    def _setup_(self, cfg: Dict, td: str, adept_module: ADEPTModule = None):
        if adept_module is None:
            self.adept_module = self.get_adept_module(cfg)
        else:
            self.adept_module = adept_module

        # dump raw config
        with open(os.path.join(td, "config.yaml"), "w") as fi:
            yaml.dump(self.adept_module.cfg, fi)

        # dump units
        quants_dict = self.adept_module.write_units()  # writes the units to the temporary directory
        with open(os.path.join(td, "units.yaml"), "w") as fi:
            yaml.dump(quants_dict, fi)

        # dump derived config
        self.adept_module.get_derived_quantities()  # gets the derived quantities
        misc.log_params(self.adept_module.cfg)  # logs the parameters to mlflow
        with open(os.path.join(td, "derived_config.yaml"), "w") as fi:
            yaml.dump(self.adept_module.cfg, fi)

        # dump array config
        self.adept_module.get_solver_quantities()
        with open(os.path.join(td, "array_config.yaml"), "wb") as fi:
            pickle.dump(self.adept_module.cfg, fi)

        self.adept_module.init_state_and_args()
        self.adept_module.init_diffeqsolve()
        modules = self.adept_module.init_modules()

        self.ran_setup = True

        return modules

    def setup(self, cfg: Dict, adept_module: ADEPTModule = None) -> Dict:
        """
        This function sets up the simulation by getting the helper functions for the given solver

        Args:
            cfg: The configuration dictionary

        """

        with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
            if self.mlflow_run_id is None:
                mlflow.set_experiment(cfg["mlflow"]["experiment"])
                with mlflow.start_run(run_name=cfg["mlflow"]["run"], nested=self.mlflow_nested) as mlflow_run:
                    modules = self._setup_(cfg, td, adept_module)
                self.mlflow_run_id = mlflow_run.info.run_id

            else:
                with mlflow.start_run(run_id=self.mlflow_run_id, nested=self.mlflow_nested) as mlflow_run:
                    with tempfile.TemporaryDirectory(dir=self.base_tempdir) as temp_path:
                        cfg = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=temp_path)
                    modules = self._setup_(cfg, td, adept_module)

        return modules

    def __call__(self, modules: Dict = None) -> Tuple[Solution, Dict, str]:
        """
        This function is the main entry point for running a simulation. It takes a configuration dictionary and returns a
        ``diffrax.Solution`` object and a dictionary of datasets.

        Returns:
            A tuple of a Solution object, a dictionary of ``xarray.dataset``s, and the mlflow run id

        """

        assert self.ran_setup, "You must run self.setup() before running the simulation"

        with mlflow.start_run(
            run_id=self.mlflow_run_id, nested=self.mlflow_nested, log_system_metrics=True
        ) as mlflow_run:
            t0 = time.time()
            run_output = self.adept_module(modules, None)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

            t0 = time.time()
            with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
                post_processing_output = self.adept_module.post_process(run_output, td)
                mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

                if "metrics" in post_processing_output:
                    mlflow.log_metrics(post_processing_output["metrics"])
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})

        return run_output, post_processing_output, self.mlflow_run_id

    def val_and_grad(self, modules: Dict = None):
        """
        This function is the value and gradient of the simulation. It assumes that this function has been implemented.

        Args:
            modules: The parameters to run the simulation and take the gradient against. All the other parameters are
            static

        Returns: val - The value of the simulation, grad - The gradient of the simulation with respect to the parameters, and the simulation output
        """
        assert self.ran_setup, "You must run self.setup() before running the simulation"
        with mlflow.start_run(
            run_id=self.mlflow_run_id, nested=self.mlflow_nested, log_system_metrics=True
        ) as mlflow_run:
            t0 = time.time()
            (val, run_output), grad = self.adept_module.vg(modules, None)
            flattened_grad, _ = jax.flatten_util.ravel_pytree(grad)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow
            mlflow.log_metrics({"val": float(val), "l2-grad": float(np.linalg.norm(flattened_grad))})

            t0 = time.time()
            with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
                post_processing_output = self.adept_module.post_process(run_output, td)
                mlflow.log_artifacts(td)  # logs the temporary directory to mlflow
                if "metrics" in post_processing_output:
                    mlflow.log_metrics(post_processing_output["metrics"])
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            return val, grad, (run_output, post_processing_output, self.mlflow_run_id)

    def _log_flops_(_run_: Callable, models: Dict, state: Dict, args: Dict, tqs):
        """
        Logs the number of flops to mlflow

        Args:
            _run_: The function that runs the simulation
            models: The models used in the simulation
            tqs: The time quantities used in the simulation

        """
        wrapped = jax.xla_computation(_run_)
        computation = wrapped(models, state, args, tqs)
        module = computation.as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
        flops_sum = analysis["flops"]
        mlflow.log_metrics({"total GigaFLOP": flops_sum / 1e9})  # logs the flops to mlflow
