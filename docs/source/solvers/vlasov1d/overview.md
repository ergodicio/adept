# Vlasov 1D1V Solver

## Equations and Quantities

We solve the following coupled set of partial differential equations:

$$
\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + \frac{q}{m} (E + E_D) \frac{\partial f}{\partial v} = \nu \partial_v (v f + v_0^2 \partial_v f)
$$

$$
\partial_x E = 1 - \int f \, dv
$$

where $f$ is the distribution function, $E$ is the electric field, $C(f)$ is the collision operator, $q$ is the charge, $m$ is the mass, and $v$ is the velocity.

The distribution function is $f = f(t, x, v)$ and the electric field is $E = E(t, x)$.

These simulations can be initialized via perturbing the distribution function or the electric field.
The electric field can be "driven" using $E_D$ which is a user defined function of time and space.

## Solver Options

As with all other solvers, the configuration is passed in via a `yaml` file. Below we describe the key solver options, and then link to the full configuration reference.

### Velocity Advection

1. **`exponential`** - This solver (incorrectly) assumes periodic boundaries in the velocity direction and uses a direct exponential solve such that

$$
f^{n+1} = f^n \times \exp(A \cdot dt)
$$

where $A$ is the advection operator. This is a much faster solver than the cubic-spline solver, but is less accurate. Use this if you are confident that the distribution function will be well behaved in the tails.

2. **`cubic-spline`** - This is a semi-Lagrangian solver that uses a cubic-spline interpolator to advect the distribution function in velocity space. Use this if you have trouble with the exponential solver.

### Spatial Advection

1. **`exponential`** - This is the only solver that is available. We only have periodic boundaries implemented in space (for the plasma) so this is perfectly fine. It is also very fast.

### Field Solver

1. **`poisson`** - This is the standard spectral Poisson solver. This is the fastest and most accurate solver available.
2. **`hampere`** - This solver uses a Hamiltonian formulation of the Vlasov-Ampere system that conserves energy exactly. This is the 2nd most reliable solver.
3. **`ampere`** - This solver uses Ampere's law to solve for the electric field.

### Collisions

1. **`none`** - No collisions are included in the simulation.
2. **`lenard-bernstein`** - This solver uses the Lenard-Bernstein collision operator to include collisions in the simulation.
3. **`dougherty`** - This solver uses the Dougherty collision operator to include collisions in the simulation.

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
