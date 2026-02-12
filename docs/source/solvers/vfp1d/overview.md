# Vlasov-Fokker-Planck 1D Solver

This solver models electron transport over hydrodynamic / collisional time-scales using a spherical harmonic expansion of the distribution function. It is based on the OSHUN algorithm described in Tzoufras et al. (2011, 2013).

## Equations and Quantities

The electron distribution function is expanded in spherical harmonics:

$$
f(t, x, v, \theta) = \sum_{l=0}^{l_{\max}} f_{l0}(t, x, v) P_l(\cos\theta)
$$

where $P_l$ are Legendre polynomials, $v = |\mathbf{v}|$ is the speed (positive-only grid), and $\theta$ is the angle between $\mathbf{v}$ and $\hat{x}$.

The isotropic component $f_0$ and the first anisotropic component $f_{10}$ satisfy:

$$
\frac{\partial f_0}{\partial t} = -\frac{v}{3} \frac{\partial f_{10}}{\partial x} + \frac{E}{3}\left(\frac{2}{v} f_{10} + \frac{\partial f_{10}}{\partial v}\right) + C_{00}[f_0]
$$

$$
\frac{\partial f_{10}}{\partial t} = -v \frac{\partial f_0}{\partial x} + E \frac{\partial f_0}{\partial v} + C_{10}[f_{10}, f_0]
$$

where $E$ is the electric field, $C_{00}$ is the isotropic collision operator, and $C_{10}$ is the anisotropic (FLM) collision operator.

## Staggered Spatial Grid

The solver uses a staggered (Yee-like) spatial grid:

- **Cell centers**: $f_0$ (scalar, isotropic distribution) and ion quantities ($Z$, $n_i$)
- **Cell edges**: $f_{10}$ (vector, anisotropic distribution), $E$, $B$, $j$

```
edges:   x_0       x_1       x_2              x_{nx}
          |         |         |                  |
centers:     x_1/2     x_3/2     ...   x_{nx-1/2}

f0:        [0]       [1]       ...     [nx-1]         (nx cells)
f10:     [0]      [1]      [2]    ... [nx-1]   [nx]   (nx+1 edges)
E:       [0]      [1]      [2]    ... [nx-1]   [nx]   (nx+1 edges)
```

Spatial derivatives naturally map between grids: $\partial f_0 / \partial x$ (centers $\to$ edges) and $\partial f_{10} / \partial x$ (edges $\to$ centers).

## Boundary Conditions

Configurable via `grid.boundary`:

- **`periodic`** (default): Wrapping ghost cells. $f_{10}[0] = f_{10}[n_x]$.
- **`reflective`**: $f_{10} = 0$ and $E = 0$ at boundary edges. $f_0$ ghost cells use mirror reflection.

## Collision Operators

### Isotropic ($f_0$): Lenard-Bernstein / Dougherty

The $f_0$ collision operator is a drift-diffusion operator in velocity space, solved implicitly via a tridiagonal system. Configurable model and differencing scheme.

### Anisotropic ($f_{10}$): FLM

The FLM collision operator (Tzoufras 2013) includes:

- **Electron-ion**: Pitch-angle scattering diagonal term $\propto Z^2 n_i / v^3$
- **Electron-electron** (optional): Rosenbluth potential integrals for diagonal and off-diagonal contributions. When disabled, the electron-ion collision frequency is boosted by the "collision fix": $(Z+4.2)/(Z+0.24)$.

## Electric Field Solver

- **`oshun`**: Implicit E-field via a Taylor expansion of $J(E)$ (Tzoufras 2013). Computes $\Delta J / \Delta E$ by perturbation and solves $E = -J_0 \cdot \Delta E / (J(E + \Delta E) - J_0)$.
- **`ampere`**: Explicit Ampere update $E^{n+1} = E^n + \Delta t \cdot J$.

## Solver Algorithm

Each timestep:

1. Explicit spatial streaming: $v \partial f / \partial x$ (Tsit5 integrator)
2. Implicit $f_0$ collision solve (Lenard-Bernstein)
3. Implicit E-field solve (OSHUN method)
4. Explicit $E \partial f / \partial v$ push (Tsit5 integrator)
5. Implicit $f_{10}$ collision solve (FLM)

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
