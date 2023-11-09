1D Two Fluid Poisson
=======================
This is a description of the 1D Two Fluid Poisson code

Current (Tested) Implementation
------------------------------------
1D Electrostatic Plasma - 3 Moments -

.. math::
    n(t, x), u(t, x), p(t, x)

- Gradients are calculated spectrally using FFTs
- 4th order explicit time integrator (using Diffrax)

Depending on the flags in the config, it can support

- Bohm-Gross oscillation frequency (Fluid dispersion relation)
- Landau damping (through a momentum damping term and a tabulated damping coefficient)
- Kinetic oscillation frequency (Kinetic dispersion relation by modifying adiabatic index)
