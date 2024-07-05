from typing import Dict, Tuple, Callable
import jax.flatten_util
import os, time, tempfile, yaml, pickle


from diffrax import Solution
import mlflow, jax, numpy as np


from utils import misc


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

    def __call__(self, params: Dict, args: Dict):
        return {}

    def vg(self, params: Dict, args: Dict):
        raise NotImplementedError(
            "This is the base class and does not have a gradient implemented. This is "
            + "likely because there is no metric in place. Subclass this class and implement the gradient"
        )
        # return eqx.filter_value_and_grad(self.__call__)(params)


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
        # elif solver == "vlasov-1d":
        #     from adept.vlasov1d import helpers
        # elif solver == "vlasov-1d2v":
        #     from adept.vlasov1d2v import helpers
        # elif solver == "vlasov-2d":
        #     from adept.vlasov2d import helpers
        # elif solver == "envelope-2d":
        #     from adept.lpse2d import helpers
        # elif solver == "vfp-2d":
        #     from adept.vfp1d import helpers
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

        self.ran_setup = True

    def setup(self, cfg: Dict, adept_module: ADEPTModule = None):
        """
        This function sets up the simulation by getting the helper functions for the given solver

        Args:
            cfg: The configuration dictionary

        """

        with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
            if self.mlflow_run_id is None:
                mlflow.set_experiment(cfg["mlflow"]["experiment"])
                with mlflow.start_run(run_name=cfg["mlflow"]["run"], nested=self.mlflow_nested) as mlflow_run:
                    self._setup_(cfg, td, adept_module)
                self.mlflow_run_id = mlflow_run.info.run_id

            else:
                with mlflow.start_run(run_id=self.mlflow_run_id, nested=self.mlflow_nested) as mlflow_run:
                    with tempfile.TemporaryDirectory(dir=self.base_tempdir) as temp_path:
                        cfg = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=temp_path)
                    self._setup_(cfg, td, adept_module)

    def __call__(self, params: Dict = None) -> Tuple[Solution, Dict, str]:
        """
        This function is the main entry point for running a simulation. It takes a configuration dictionary and returns a
        ``diffrax.Solution`` object and a dictionary of datasets.

        Returns:
            A tuple of a Solution object, a dictionary of ``xarray.dataset``s, and the mlflow run id

        """

        assert self.ran_setup, "You must run self.setup() before running the simulation"

        with mlflow.start_run(run_id=self.mlflow_run_id, nested=self.mlflow_nested) as mlflow_run:
            t0 = time.time()
            run_output = self.adept_module(params, None)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

            t0 = time.time()
            with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
                post_processing_output = self.adept_module.post_process(run_output, td)
                mlflow.log_artifacts(td)  # logs the temporary directory to mlflow
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})

        return run_output, post_processing_output, self.mlflow_run_id

    def val_and_grad(self, params: Dict = None):
        """
        This function is the value and gradient of the simulation. It assumes that this function has been implemented.

        Args:
            params: The parameters to run the simulation and take the gradient against. All the other parameters are
            static

        Returns: val - The value of the simulation, grad - The gradient of the simulation with respect to the parameters, and the simulation output
        """
        assert self.ran_setup, "You must run self.setup() before running the simulation"
        with mlflow.start_run(run_id=self.mlflow_run_id, nested=self.mlflow_nested) as mlflow_run:
            t0 = time.time()
            (val, run_output), grad = self.adept_module.vg(params, None)
            flattened_grad, _ = jax.flatten_util.ravel_pytree(grad)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow
            mlflow.log_metrics({"val": float(val), "l2-grad": float(np.linalg.norm(flattened_grad))})

            t0 = time.time()
            with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
                post_processing_output = self.adept_module.post_process(run_output, td)
                mlflow.log_artifacts(td)  # logs the temporary directory to mlflow
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
