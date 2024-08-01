Vlasov 1D1V
-----------------------------------------

Equations and Quantities
========================
We solve the following coupled set of partial differential equations

.. math::
    \frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + \frac{q}{m} (E + E_D) \frac{\partial f}{\partial v} &= \nu \partial_v (v f + v_0^2 \partial_v f)

    \partial_x E &= 1 - \int f dv 

where :math:`f` is the distribution function, :math:`E` is the electric field, :math:`C(f)` is the collision operator, :math:`q` is the charge, :math:`m` is the mass, and :math:`v` is the velocity.

The distribution function is :math:`f = f(t, x, v)` and the electric field is :math:`E = E(t, x)`

These simulations can be initialized via perturbing the distribution function or the electric field.
The electric field can be "driven" using :math:`E_D` which is a user defined function of time and space.


Solver Options
================
This is where we should have a list of the different solvers that can be chosen including the collision operator. #TODO

Configuration parameters
========================
This is where there should be a line by line explanation of everything in a config file... #TODO