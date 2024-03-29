.. adept documentation master file, created by
   sphinx-quickstart on Wed Nov  8 15:35:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ADEPT
=========

.. image:: adept-logo.png
   :alt: ADEPT
   :align: right

**ADEPT** is a set of **A** utomatic **D** ifferentation **E** nabled **P** lasma **T** ransport codes.

In some form or the other, they solve the equations of motion for a plasma.

So far, we have implemented

1. Vlasov-Poisson-Fokker-Planck 1D1V
2. Two fluid - Poisson system in 1D
3. Vlasov-Poisson-Fokker-Planck 1D2V
4. Vlasov-Poisson system in 2D

What is novel about it?
------------------------
- Automatic Differentiation (AD) Enabled (bc of JAX)
- GPU-capable (bc of XLA)
- Experiment manager enabled (bc of mlflow)
- Pythonic

What does AD do for us?
------------------------
AD enables the calculation of derivatives of entire simulations, pieces of it, or anything in between. This can be used for
- sensitivity analyses
- parameter estimation
- parameter optimization
- model training

A couple of implemented examples are

1. Find the resonant frequency given a density and temperature

      This is provided as a test in `tests/test_resonance_search.py`. Also see ref. [1]

2. Fit a parameteric model for unresolved / unsolved microphysics

      The gist is that there is a discrepancy in the observable between a "first-principles" and "approximate" simulation.
      You would like for that discrepancy to decrease. To do so, you add a neural network to your "approximate" solver in a smart fashion.
      Then, you calibrate the results of your "approximate" simulation against the "ground-truth" from the "first-principles" simulation.
      After doing that enough times, over a diverse enough set of physical parameters, the neural network is able to help your "approximate" solver
      recover the "correct" answer.

      See ref. [2] for details and an application

--------------------------------------------------

Documentation
------------------

.. toctree::
   usage
   usage/vlasov1d
   usage/tf1d
   usage/vlasov1d2v
   faq
   dev_guide
   api_documentation

   :maxdepth: 2
   :caption: Contents:



.. note::

   This project is under active development.

      
Contributing guide
------------------------
The contributing guide is in development but for now, just make an issue / pull request and we can go from there :)

References
------------

[1] A. S. Joglekar and A. G. R. Thomas, “Machine learning of hidden variables in multiscale fluid simulation,” Mach. Learn.: Sci. Technol., vol. 4, no. 3, p. 035049, Sep. 2023, doi: 10.1088/2632-2153/acf81a.

[2] A. S. Joglekar and A. G. R. Thomas, “Unsupervised discovery of nonlinear plasma physics using differentiable kinetic simulations,” J. Plasma Phys., vol. 88, no. 6, p. 905880608, Dec. 2022, doi: 10.1017/S0022377822000939.


Citation
------------
[1] A. S. Joglekar and A. G. R. Thomas, “Machine learning of hidden variables in multiscale fluid simulation,” Mach. Learn.: Sci. Technol., vol. 4, no. 3, p. 035049, Sep. 2023, doi: 10.1088/2632-2153/acf81a.


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
