Explicit Cartesian-Cartesian Vlasov 1D1V
=========================================

Equations and Quantities
-------------------------
We solve the following coupled set of partial differential equations

.. math::
    \frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + \frac{q}{m} (E + E_D) \frac{\partial f}{\partial v} &= \nu \partial_v (v f + v_0^2 \partial_v f)

    \partial_x E &= 1 - \int f dv 

where :math:`f` is the distribution function, :math:`E` is the electric field, :math:`C(f)` is the collision operator, :math:`q` is the charge, :math:`m` is the mass, and :math:`v` is the velocity.

The distribution function is :math:`f = f(t, x, v)` and the electric field is :math:`E = E(t, x)`

These simulations can be initialized via perturbing the distribution function or the electric field.
The electric field can be "driven" using :math:`E_D` which is a user defined function of time and space.

Configuration Options
----------------------

As with all other solvers, the configuration is passed in via a ``yaml`` file. The datamodel for the configuration is defined in the documentation but most will first care about
how the equations themselves are solved and what the different options are. We list those first here.

Velocity advection
^^^^^^^^^^^^^^^^^^^^^^^^
1. ``exponential`` - This solver (incorrectly) assumes periodic boundaries in the velocity direction and uses a direct exponential solve such that 

.. math::
    f^{n+1} = f^n \times \exp(A*dt) 

where :math:`A` is the advection operator. This is a much faster solver than the cubic-spline solver, but is less accurate. Use this if you are confident that the distribution function will be well behaved in the tails

2. ``cubic-spline`` - This is a semi-Lagrangian solver that uses a cubic-spline interpolator to advect the distribution function in velocity space. Use this if you have trouble with the exponential solver.


Spatial advection
^^^^^^^^^^^^^^^^^^^^^^^^
1. ``exponential`` - This is the only solver that is available. We only have periodic boundaries implemented in space (for the plasma) so this is perfectly fine. It is also very fast.


Field solver
^^^^^^^^^^^^^^^^^^^^^^^^

1. ``poisson`` - This is the only field solver that is available. It uses a spectral solver to solve the Poisson equation. This is the fastest and most accurate solver available.
2. ``hampere`` - This solver uses a Hamiltonian formulation of the Vlasov-Ampere system that conserves energy exactly. This is the 2nd most reliable solver.
3. ``ampere`` - This solver uses the Ampere's law to solve for the electric field.

Collisions
^^^^^^^^^^^^^^^^^^^^^^^^
1. ``none`` - No collisions are included in the simulation
2. ``lenard-bernstein`` - This solver uses the Lenard-Bernstein collision operator to include collisions in the simulation. 
3. ``daugherty`` - This solver uses the Daugherty collision operator to include collisions in the simulation.


Remaining Options
^^^^^^^^^^^^^^^^^^^^^^^^
The following pages document the configuration options for the Vlasov1D1V solver

.. toctree::
    datamodels/vlasov1d
    :maxdepth: 3
    :caption: Configuration Options: