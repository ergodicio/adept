# Vlasov 1D1V Solver

Example decks live in `configs/vlasov-1d/`. To run one:

```bash
uv run run.py --cfg configs/vlasov-1d/epw
```

The top-level `solver:` key selects this module — `vlasov-1d`, or `vlasov-1d-iaw` for the
ion-acoustic turbulence variant.

## Things You Might Care About

1. Infinite length (single mode) plasma waves — Landau damping, trapping
2. Finite length plasma waves — everything in 1. plus wavepackets
3. Wave dynamics on density gradients — 2. plus density gradients
4. Stimulated Raman Scattering — 3. plus light waves
5. Ion-acoustic turbulence driven by stochastic box-scale forcing (`vlasov-1d-iaw`)

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

## Multispecies Support

The solver supports multiple particle species (e.g., electrons and ions) evolving self-consistently under a shared electric field. Each species can have:

- Independent charge and mass (determining the species-specific $q/m$ ratio)
- Independent velocity grid parameters (`vmax`, `nv`)
- Multiple density components

For multispecies simulations, the Poisson equation sums over all species:

$$
\partial_x E = \sum_s q_s \int f_s \, dv_s
$$

See the [Configuration Reference](config.md#species-multispecies-configuration) for details on configuring multispecies simulations.

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

## Boundary Conditions

| Axis / quantity | Condition | Notes |
|---|---|---|
| $x$ (distribution and electrostatic field) | **Periodic** | Spatial advection and the Poisson/Ampère solves are spectral, so periodicity is structural rather than a choice. |
| $v$ (`edfdv: exponential`) | **Periodic** | An artifact of doing the velocity push spectrally. It wraps the forward tail onto the $-v$ edge, so it is only safe when $f \approx 0$ at both velocity edges. |
| $v$ (`edfdv: cubic-spline`) | Semi-Lagrangian interpolation | Does not assume periodicity; use it when the tails are populated. |
| $v$ (collision operator) | **Zero-flux** | Applied at both velocity edges, which is what makes the Fokker-Planck operators conserve density exactly. |
| $x$ (transverse vector potential $a$) | **Absorbing**, 2nd order | The transverse wave equation is solved by finite differences on a grid with two boundary cells, independently of the periodic plasma domain. |

The velocity grid is uniform and cell-centered, and may be asymmetric (`vmin != -vmax`). Choose
bounds wide enough that $f \approx 0$ at *both* edges, or the periodic velocity push and the
zero-flux collision stencil stop being accurate.

## Forcing and Drivers

Four mechanisms can drive the system, all optional:

| Block | What it does |
|---|---|
| `drivers.ex` | A prescribed longitudinal field $E_D(x, t)$ added to the force term. Each pulse is a travelling wave $a_0 \sin(k_0 x - (\omega_0 + \delta\omega_0) t)$ with independent tanh envelopes in space and time. It never enters the Poisson solve, so the self-consistent field-energy diagnostic excludes it. |
| `drivers.ex_stochastic` | Time-correlated forcing: a set of Fourier modes whose complex amplitudes evolve as independent Ornstein-Uhlenbeck processes with correlation time `tau` and a prescribed stationary RMS. This is the box-scale stirring used for ion-acoustic turbulence. |
| `drivers.ey` | A transverse EM driver. It sources the wave equation for the vector potential $a$, and the plasma feels the resulting ponderomotive force $-\tfrac{1}{2}\partial_x(a^2)$. Extended sources use $S = -\omega^2 a_0\,\text{env}(x,t)\sin(kx - \omega t)$; point sources use a single-cell delta with amplitude scaled by the vacuum Green's function, and radiate both ways — put them next to the absorbing boundary for a unidirectional wave. |
| `terms.krook` | Not a driver but a forcing term: relaxation toward a Maxwellian at rate $\nu_K$, enveloped in space and time. Mostly useful as a hard thermalization layer at a boundary. |

## What Gets Saved

Every run writes to a temporary directory that `ergoExo` logs to MLflow as artifacts. Two trees:

**`binary/`** — netCDF, one file per save stream:

| File | Contents |
|---|---|
| `fields-shared-t=<t>.nc` | The EM fields on the requested $(t, x)$ grid |
| `fields-<species>-t=<t>.nc` | Per-species real-space moments (density, velocity, temperature, …) |
| `scalars-t=<t>.nc` | Scalar time series — field and kinetic energies, and the conservation diagnostics |
| `dist-<save_key>.nc` | The distribution function for each species save block you configured, on its requested axes |

**`plots/`** — PNGs generated from those datasets:

| Directory | Contents |
|---|---|
| `plots/fields/` | Space-time plots of the shared EM fields, with `logplots/` and `lineouts/` variants |
| `plots/fields/<species>/` | The same for each species' moments, again with `logplots/` and `lineouts/` |
| `plots/scalars/` and `plots/scalars/<species>/` | Scalar time-series plots |
| `plots/dists/<save_key>/` | Distribution-function snapshots, one subdirectory per distribution save block |

With `solver: vlasov-1d-iaw` you additionally get `plots/iaw/density_spectrum.png`,
`nk_spectrogram.png`, and `phase_space_dfx.png`, plus `binary/nk.nc`.

Timing metrics (`run_time`, `postprocess_time_min`, `total_time`) go to MLflow as metrics rather than
files. Which streams exist, and at what cadence, is entirely determined by the `save` block — see the
[Configuration Reference](config.md#save).

## Practical Notes

**Density profile.** Uniform is easy. For a non-uniform profile you have to specify the parameters of
the profile, which can be a sinusoidal perturbation or a tanh flat top — see
[initialization](../../usage/initialization.md) for the tanh flat-top parameters.

**Collision frequency.** `nu_ee` will modify the dynamics substantially depending on how far the
distribution is driven from Maxwellian. Its envelope is specified the same way as a driver's.

**Krook frequency.** In terms of physical correspondence this mostly resembles sideloss. Use it as a
hard thermalization operator, for instance at the boundaries as in the SRS example.

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
