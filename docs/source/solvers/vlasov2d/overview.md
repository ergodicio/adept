# Vlasov 2D2V Solver

## Overview

The Vlasov 2D2V solver evolves the electron distribution function $f(t, x, y, v_x, v_y)$ in a 2D spatial domain with 2D velocity space. This solver supports electromagnetic simulations with self-consistent field evolution.

## Equations

We solve the Vlasov-Maxwell system:

$$
\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla_x f + \frac{q}{m} (\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f = 0
$$

coupled with Maxwell's equations for the electromagnetic fields.

## Key Features

- 2D2V phase space (2 spatial dimensions, 2 velocity dimensions)
- Hamiltonian charge-conserving Maxwell solver
- Exponential integrators for spatial and velocity advection
- Support for external electromagnetic drivers

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
