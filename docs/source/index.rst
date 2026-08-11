ADEPT
=========

.. image:: adept-logo.png
   :alt: ADEPT
   :align: right

**ADEPT** is a set of **A** utomatic **D** ifferentiation **E** nabled **P** lasma **T** ransport solvers.


Examples
----------
Examples can be found in the tests folder or in the adept-notebooks repository - http://github.com/ergodicio/adept-notebooks. Example configuration files are also provided in `configs/`

--------------------------------------------------

Documentation
------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage

.. toctree::
   :maxdepth: 2
   :caption: Solvers

   solvers
   solvers/vlasov1d/overview
   solvers/vlasov1d2v/overview
   solvers/vlasov2d/overview
   solvers/vfp1d/overview
   solvers/spectrax1d/overview
   solvers/hermite_legendre_1d/overview
   solvers/pic1d/overview
   solvers/lpse2d/overview
   solvers/tf1d/overview

.. toctree::
   :maxdepth: 2
   :caption: Configuration Reference

   solvers/vlasov1d/config
   solvers/vlasov1d2v/config
   solvers/vlasov2d/config
   solvers/vfp1d/config
   solvers/spectrax1d/config
   solvers/hermite_legendre_1d/config
   solvers/pic1d/config
   solvers/lpse2d/config
   solvers/tf1d/config

.. toctree::
   :maxdepth: 2
   :caption: Reference

   faq
   api
   dev_guide
   tests

.. note::

   This project is under active development.


Contributing Guide
------------------------
The contributing guide is in development but for now, just make an issue / pull request and we can go from there :)

Citation
------------
If you are using this package for your research, please cite

   A. Joglekar and A. Thomas, "ADEPT - automatic differentiation enabled plasma transport,"
   ICML - SynS & ML Workshop (https://syns-ml.github.io/2023/contributions/), 2023

References
------------
[1] A. S. Joglekar & A. G. R. Thomas. "Unsupervised discovery of nonlinear plasma physics using differentiable kinetic simulations." J. Plasma Phys. 88, 905880608 (2022).

[2] A. S. Joglekar and A. G. R. Thomas, "Machine learning of hidden variables in multiscale fluid simulation," Mach. Learn.: Sci. Technol., vol. 4, no. 3, p. 035049, Sep. 2023, doi: 10.1088/2632-2153/acf81a.
