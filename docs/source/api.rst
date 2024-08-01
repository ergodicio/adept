API Guide
----------

There are two primary high level classes.

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
