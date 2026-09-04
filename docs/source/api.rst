API Guide
----------

The explicit logging-free program and objective API is described in
:doc:`usage/explicit_programs`. Its tracker, artifact, report, and host execution
services are described in :doc:`usage/host_runtime`. The new path is currently opt-in
for the ``tf-1d`` and electrostatic ``pic-1d`` pilots.

The established API has two primary high level classes.

1. `ergoExo` houses the solver and handles the mlflow logging and experiment management
2. `ADEPTModule` is base class for the solver

If you wanted to create your own differentiable program that uses the ADEPT solvers, you could do

.. code-block:: python

   from adept import ergoExo

   exo = ergoExo()
   modules = exo.setup(cfg)

and

.. code-block:: python

   sol, ppo, run_id = exo(modules)

or

.. code-block:: python

   sol, ppo, run_id = exo.val_and_grad(modules)

This is analogous to `torch.nn.Module` and `eqx.Module` the `Module` workflows in general.

You can see what each of those calls does in API documentation below.

.. toctree::
   ergoExo
   ADEPTModule
   :maxdepth: 3
   :caption: High level API:
