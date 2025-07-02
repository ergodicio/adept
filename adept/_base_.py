import os
import pickle
import tempfile
import time
from collections.abc import Callable
from copy import deepcopy

import jax
import jax.flatten_util
import mlflow
import numpy as np
import yaml
from diffrax import RESULTS, Euler, Solution
from equinox import Module, filter_jit
from jax import numpy as jnp

from .utils import robust_log_artifacts


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
    This class is the base class for all the ADEPT modules.
    It defines the interface that all the ADEPT modules must implement so that
    the `ergoExo` class can call them in the right order.

    Args:
        cfg: The configuration dictionary

    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.state = None
        self.args = None
        self.diffeqsolve_quants = None
        self.time_quantities = None

    def post_process(self, run_output: dict, td: str) -> dict:
        """
        This function is responsible for post-processing the results of the simulation.
        It is called after the simulation is run and the results are available.

        Args:
            run_output (Dict): The output of the simulation
            td (str): The temporary directory where the results are stored

        Returns:
            A dictionary of the post-processed results. This can include the metrics, the ``xarray`` datasets,
            and any other information that is relevant to the simulation

        """

        return {}

    def write_units(self) -> dict:
        """
        This function is responsible for writing the units, normalizing constants,
        and other important physical quantities to a dictionary.
        This dictionary is then dumped to a yaml file and logged to mlflow by the ``ergoExo`` class.

        Returns:
            A dictionary of the units

        """
        return {}

    def init_diffeqsolve(self) -> dict:
        """
        This function is responsible for initializing the differential equation solver ``diffrax.diffeqsolve``.
        It sets up the time quantities, the solver quantities, and the save function.

        Returns:
            A dictionary of the differential equation solver quantities

        """
        pass

    def get_derived_quantities(self) -> dict:
        """
        This function is responsible for getting the derived quantities from the configuration dictionary.
        This is needed for running the simulation. These quantities do get logged to mlflow
        by the ``ergoExo`` class.

        Returns:
            An updated configuration dictionary

        """
        pass

    def get_solver_quantities(self):
        """
        This function is responsible for getting the solver quantities from the configuration dictionary.
        This is needed for running the simulation. These quantities do NOT get logged
        to mlflow because they are often arrays

        Returns:
            An updated configuration dictionary

        """
        pass

    def get_save_func(self):
        """
        This function is responsible for getting the save function for the differential equation solver.
        This is needed for running the simulation.
        This function lets you subsample your simulation state so as to not save the entire thing at every timestep.

        This dictionary is set as a class attribute for the ``ADEPTModule`` and are used in the ``__call__`` function

        """
        pass

    def init_state_and_args(self):
        """
        This function initializes the state and the arguments that are required to run the simulation.
        The state is the initial conditions of the simulation and the arguments are often the drivers

        These are set as class attributes for the ``ADEPTModule`` and are used in the ``__call__`` function

        """
        return {}

    def init_modules(self) -> dict[str, Module]:
        """
        This function initializes the necessary (trainable) physics modules that are required to run the simulation.
        These can be modules that change the initial conditions, or the driver (boundary conditions),
        or the metric calculation. These modules are usually `eqx.Module`s so that
        you can take derivatives against the (parameters of the) modules.

        Returns:
            Dict: A dictionary of the (trainable) modules that are required to run the simulation

        """
        return {}

    def __call__(self, trainable_modules: dict, args: dict):
        return {}

    def vg(self, trainable_modules: dict, args: dict):
        raise NotImplementedError(
            "This is the base class and does not have a gradient implemented. This is "
            + "likely because there is no metric in place. Subclass this class and implement the gradient"
        )
        # return eqx.filter_value_and_grad(self.__call__)(trainable_modules)


class ergoExo:
    """
    This class is the main interface for running a simulation. It is responsible for calling
    all the ADEPT modules in the right order and logging parameters and results to mlflow.

    This approach helps decouple the numerical solvers from the
    experiment management and facilitates the addition of new solvers

    Typical usage is as follows

    .. code-block:: python

        exoskeleton = ergoExo()
        modules = exoskeleton.setup(cfg)
        run_output, post_processing_output, mlflow_run_id = exoskeleton(modules, args=None)


    If you are resuming an existing mlflow run, you can do the following

    .. code-block:: python

        exoskeleton = ergoExo(mlflow_run_id=mlflow_run_id)
        modules = exoskeleton.setup(cfg)
        run_output, post_processing_output, mlflow_run_id = exoskeleton(modules, args=None)

    If you are introducing a custom `ADEPTModule`, you can do the following

    .. code-block:: python

        exoskeleton = ergoExo()
        modules = exoskeleton.setup(cfg, exoskeleton_module=custom_module)
        run_output, post_processing_output, mlflow_run_id = adept(modules, args=None)


    """

    def __init__(
        self, mlflow_run_id: str | None = None, mlflow_nested: bool = False, parent_run_id: str | None = None
    ) -> None:
        self.mlflow_run_id = mlflow_run_id
        # if mlflow_run_id is not None:
        #     assert self.mlflow_nested is not None
        self.parent_run_id = None
        self.mlflow_nested = mlflow_nested
        if mlflow_nested:
            self.parent_run_id = parent_run_id

        if "BASE_TEMPDIR" in os.environ:
            self.base_tempdir = os.environ["BASE_TEMPDIR"]
        else:
            self.base_tempdir = None

        self.ran_setup = False
        self.cfg = None

    def setup(self, cfg: dict, adept_module: ADEPTModule = None) -> dict[str, Module]:
        """
        This function sets up the differentiable simulation by getting the chosen solver and setting it up
        At this point in time, the setup includes

        1. initializing the mlflow run and setting the runid or resuming an existing run
        2. getting the right ``ADEPTModule`` or using the one passed in. This gets assigned to ``self.adept_module``.
        3. updating the config, units, derived quantities, and array config as defined by the ``ADEPTModule``.
           It also dumps this information to the temporary directory, which will be logged later,
           and logging the parameters to mlflow
        4. initializing the state and args as defined by the ``ADEPTModule``
        5. initializing the `diffeqsolve` as defined by the ``ADEPTModule``
        6. initializing the necessary (trainable) physics modules as defined by the ``ADEPTModule``

        Args:
            cfg: The configuration dictionary

        Returns:
            A dictionary of trainable modules (``Dict[str, eqx.Module]``)

            This is a dictionary of the (trainable) modules that are required to run the simulation.
            These can be modules that change the initial conditions, or the driver (boundary conditions),
            or the metric calculation. These modules are ``equinox`` modules in order to play nice with ``diffrax``

        """

        cfg = deepcopy(cfg)

        with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
            if self.mlflow_run_id is None:
                mlflow.set_experiment(cfg["mlflow"]["experiment"])
                with mlflow.start_run(
                    run_name=cfg["mlflow"]["run"], nested=self.mlflow_nested, parent_run_id=self.parent_run_id
                ) as mlflow_run:
                    modules = self._setup_(cfg, td, adept_module)
                    robust_log_artifacts(td)  # logs the temporary directory to mlflow
                self.mlflow_run_id = mlflow_run.info.run_id

            else:
                with mlflow.start_run(
                    run_id=self.mlflow_run_id, nested=self.mlflow_nested, parent_run_id=self.parent_run_id
                ) as mlflow_run:
                    # with tempfile.TemporaryDirectory(dir=self.base_tempdir) as temp_path:
                    # cfg = get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=temp_path)
                    modules = self._setup_(cfg, td, adept_module)
                    robust_log_artifacts(td)  # logs the temporary directory to mlflow

        self.cfg = cfg

        return modules

    def _get_adept_module_(self, cfg: dict) -> ADEPTModule:
        """
        This function returns the helper functions for the given solver

        Args:
            solver: The solver to use


        """

        if cfg["solver"] == "tf-1d":
            from adept.tf1d import BaseTwoFluid1D as this_module

            # config = ConfigModel(**cfg)

        elif cfg["solver"] == "vlasov-1d":
            from adept.vlasov1d import BaseVlasov1D as this_module

            # config = ConfigModel(**cfg)

        elif cfg["solver"] == "envelope-2d":
            from adept.lpse2d import BaseLPSE2D as this_module

            # from adept.lpse2d.datamodel import ConfigModel

            # config = ConfigModel(**cfg)

        elif cfg["solver"] == "vfp-1d":
            from adept.vfp1d.base import BaseVFP1D as this_module
        elif cfg["solver"] == "sbsbs-1d":
            from adept.sbsbs1d import BaseSteadyStateBackwardStimulatedBrilloiunScattering as this_module
        else:
            raise NotImplementedError("This solver approach has not been implemented yet")

        return this_module(cfg)

    def _setup_(self, cfg: dict, td: str, adept_module: ADEPTModule = None, log: bool = True) -> dict[str, Module]:
        from adept.utils import log_params

        if adept_module is None:
            self.adept_module = self._get_adept_module_(cfg)
        else:
            self.adept_module = adept_module(cfg)

        # dump raw config
        if log:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(self.adept_module.cfg, fi)

        # dump units
        quants_dict = self.adept_module.write_units()  # writes the units to the temporary directory
        if log:
            with open(os.path.join(td, "units.yaml"), "w") as fi:
                yaml.dump(quants_dict, fi)

        # dump derived config
        self.adept_module.get_derived_quantities()  # gets the derived quantities

        if log:
            log_params(self.adept_module.cfg)  # logs the parameters to mlflow
            with open(os.path.join(td, "derived_config.yaml"), "w") as fi:
                yaml.dump(self.adept_module.cfg, fi)

        # dump array config
        self.adept_module.get_solver_quantities()
        if log:
            with open(os.path.join(td, "array_config.pkl"), "wb") as fi:
                pickle.dump(self.adept_module.cfg, fi)

        self.adept_module.init_state_and_args()
        self.adept_module.init_diffeqsolve()
        modules = self.adept_module.init_modules()

        self.ran_setup = True

        return modules

    def __call__(
        self, modules: dict | None = None, args: dict | None = None, export=True
    ) -> tuple[Solution, dict, str]:
        """
        This function is the main entry point for running a simulation.
        It takes a configuration dictionary and returns a ``diffrax.Solution`` object and a dictionary of datasets.
        It calls the ``self.adept_module``'s ``__call__`` function.

        It is also responsible for logging the artifacts and metrics to mlflow.

        Args:
            modules (Dict(str, eqx.Module)): The trainable modules that are required to run the simulation.
            All the other parameters are static and initialized during the setup call

        Returns:
            a tuple of the run_output (``diffrax.Solution``), post_processing_output (``Dict[str, xarray.dataset]``),
            and the mlflow_run_id (``str``).

            The run_output comes from the ``__call__`` function of the ``self.adept_module``.
            The post_processing_output comes from the ``post_process`` method of the ``self.adept_module``.
            The mlflow_run_id is the id of the mlflow run that was created during the setup call
            or passed in during the initialization of the class

        """

        assert self.ran_setup, "You must run self.setup() before running the simulation"

        with mlflow.start_run(
            run_id=self.mlflow_run_id, nested=self.mlflow_nested, log_system_metrics=True
        ) as mlflow_run:
            t0 = time.time()
            run_output = filter_jit(self.adept_module.__call__)(modules, args)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

            t0 = time.time()
            with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
                post_processing_output = self.adept_module.post_process(run_output, td)
                if export:
                    robust_log_artifacts(td)  # logs the temporary directory to mlflow

                if "metrics" in post_processing_output:
                    mlflow.log_metrics(post_processing_output["metrics"])
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})

        return run_output, post_processing_output, self.mlflow_run_id

    def val_and_grad(
        self, modules: dict | None = None, args: dict | None = None, export=True
    ) -> tuple[float, dict, tuple[Solution, dict, str]]:
        """
        This function is the value and gradient of the simulation.
        This is a very similar looking function to the ``__call__`` function
        but calls the ``self.adept_module.vg`` rather than the ``self.adept_module.__call__``.

        It is also responsible for logging the artifacts and metrics to mlflow.


        Args:
            modules: The (trainable) modules that are required to run the simulation and take the gradient against.
            All the other parameters are static and initialized during the setup call

        Returns:
            a tuple of the value (``float``), gradient (``Dict``), and a tuple of the run_output (``diffrax.Solution``),
            post_processing_output (``Dict[str, xarray.dataset]``), and the mlflow_run_id (``str``).

            The value and gradient, and run_output come from the ``adept_module.vg`` function.
            The run_output is the same as that from ``__call__`` function of the ``self.adept_module``.
            The post_processing_output comes from the ``post_process`` method of the ``self.adept_module``.
            The mlflow_run_id is the id of the mlflow run that was created during the setup call
            or passed in during the initialization
        """
        assert self.ran_setup, "You must run self.setup() before running the simulation"
        with mlflow.start_run(
            run_id=self.mlflow_run_id, nested=self.mlflow_nested, log_system_metrics=True
        ) as mlflow_run:
            t0 = time.time()
            (val, run_output), grad = filter_jit(self.adept_module.vg)(modules, args)
            flattened_grad, _ = jax.flatten_util.ravel_pytree(grad)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow
            mlflow.log_metrics({"val": float(val), "l2-grad": float(np.linalg.norm(flattened_grad))})

            t0 = time.time()
            with tempfile.TemporaryDirectory(dir=self.base_tempdir) as td:
                post_processing_output = self.adept_module.post_process(run_output, td)
                if export:
                    robust_log_artifacts(td)  # logs the temporary directory to mlflow
                if "metrics" in post_processing_output:
                    mlflow.log_metrics(post_processing_output["metrics"])
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            return val, grad, (run_output, post_processing_output, self.mlflow_run_id)

    def _log_flops_(_run_: Callable, models: dict, state: dict, args: dict, tqs):
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
