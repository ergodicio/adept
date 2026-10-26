# PIC 1D Solver

Example decks live in `configs/pic-1d/`. To run one:

```bash
uv run run.py --cfg configs/pic-1d/epw
```

`solver: pic-1d` selects this module.

A 1D1V electrostatic particle-in-cell solver. It is deliberately built as a particle-based twin of
the [Vlasov-1D](../vlasov1d/overview.md) solver: it accepts the same units, density, driver, and
save blocks, and reproduces the same initial moments up to sampling noise, so the same input deck
can be run both ways and compared.

## Equations and Quantities

Rather than discretizing $f(x, v)$ on a mesh, the distribution is represented by a finite set of
computational particles, each carrying a position, velocity, and weight $(x_p, v_p, w_p)$. These
obey the characteristics of the Vlasov equation:

$$
\frac{d x_p}{d t} = v_p, \qquad \frac{d v_p}{d t} = \frac{q}{m} \left( E(x_p) + E_D(x_p) \right)
$$

The field is obtained by depositing charge onto the grid and solving Poisson spectrally:

$$
\partial_x E = \rho_{\text{ion}} - \sum_p q \, w_p \, S(x_g - x_p)
$$

where $S$ is the particle shape function. In Fourier space this is $E_k = -i \rho_k / k$, with the
$k=0$ mode set to zero — the same solve the Vlasov-1D module uses, so the two agree field-for-field
given the same charge density.

Each step is deposit → Poisson solve → add the external driver → gather $E$ at the particle
positions → push.

## Solver Options

### Particle Shape (`grid.particle_shape`)

The B-spline used for both charge deposition and field gather. Using the same shape for both is
what keeps the scheme momentum-conserving.

1. **`linear`** — 2-point stencil. Cheapest, noisiest.
2. **`tsc`** — triangular-shaped cloud, 3-point stencil. The default.
3. **`cubic`** — 4-point stencil. Smoothest, most expensive per particle.

### Time Integration (`terms.time`)

1. **`leapfrog`** — 2nd-order kick-drift-kick. Symplectic.
2. **`yoshida4`** — 4th-order symplectic composition of three leapfrog steps (Yoshida 1990), with
   the standard $dt_1 = dt / (2 - 2^{1/3})$, $dt_2 = -2^{1/3} dt_1$ weights. Roughly 3x the cost per
   step, but permits a much larger `dt` for the same energy error.

### Particle Loading (`terms.species[].loading`)

1. **`quiet`** — particles are placed on uniform position slots and velocities are drawn by inverse
   CDF, with the density profile carried in the particle weights. The $x$ and $v$ orderings are
   de-correlated so the quiet start does not imprint a spurious phase-space structure. Much lower
   initial noise; use this for linear-response measurements such as Landau damping.
2. **`random`** — positions sampled from the density profile by inverse CDF and velocities sampled
   randomly, at uniform weight.

### Field Solver (`terms.field`)

1. **`poisson`** — spectral Poisson. Currently the only option.

## Transverse Drivers

Configuring `drivers.ey` additionally advances a transverse vector potential $a(x)$ with a
2nd-order leapfrog wave solver — the same one Vlasov-1D uses — and adds a ponderomotive force
$-\tfrac{1}{2} \partial_x (a^2)$ to the longitudinal kick. The particles themselves stay 1D1V.

## Limitations

Relative to Vlasov-1D, this module has no Fokker-Planck or Krook collision operator and no
self-consistent transverse light-wave physics. Ions are a static neutralizing background unless
configured as an explicit species.

## Boundary Conditions

| Axis / quantity | Condition | Notes |
|---|---|---|
| $x$ (particles) | **Periodic** | Drift wraps positions modulo $L = x_{\max} - x_{\min}$, leaving weight and velocity untouched. |
| $x$ (deposition and gather) | **Periodic** | B-spline stencil indices are wrapped modulo `nx`. The grid is cell-centered: $x_g = x_{\min} + (g + \tfrac{1}{2})\Delta x$. |
| $x$ (field solve) | **Periodic** | Spectral Poisson with the $k=0$ mode set to zero, consistent with a neutralizing static background. |
| $v$ | **None** | Velocity space is unbounded — particles carry whatever velocity they have. `vmax_load` only truncates the *initial* sampling. |
| $x$ (transverse vector potential) | **Absorbing** | Only when `drivers.ey` is configured, using the same wave solver as Vlasov-1D. |

The absence of a velocity boundary is the structural advantage over the grid-based solvers: there is
no tail to truncate and no zero-flux stencil to get right.

## Forcing and Drivers

| Block | What it does |
|---|---|
| `drivers.ex` | A prescribed longitudinal field evaluated at the particle positions during the kick, added to the self-consistent $E$. Uses the nested `params`/`envelope` structure shared with Vlasov-1D. |
| `drivers.ey` | Advances a transverse vector potential $a(x)$ with a 2nd-order leapfrog wave solver and adds a ponderomotive force $-\tfrac{1}{2}\partial_x(a^2)$ to the longitudinal kick. The particles themselves stay 1D1V. |

There is no collisional forcing — this module has no Fokker-Planck or Krook operator.

## What Gets Saved

**`binary/`**:

| File | Contents |
|---|---|
| `scalars-t=<t>.nc` | Scalar time series — field energy, kinetic energy, and the conservation diagnostics |
| `<key>-shared-t=<t>.nc` | Shared field quantities on the requested $(t, x)$ grid |
| `<key>-<species>-t=<t>.nc` | Per-species gridded moments, deposited from the particles |

Moments are deposited onto the grid rather than written per particle, so output size is set by `nx`
and the save cadence, not by `ppc`.

**`plots/`** — PNGs generated from those datasets, in the same layout as Vlasov-1D.

`postprocess_time_min` is logged to MLflow as a metric.

## Practical Notes

**Particle count.** `ppc` is the main accuracy/cost dial. Field noise falls off roughly as
$1/\sqrt{N}$, so a linear-response measurement such as Landau damping needs far more particles than a
nonlinear saturation study — the example decks use `ppc: 32768` for damping measurements. Pair a large
`ppc` with `loading: quiet`.

**Density profile.** Shared with Vlasov-1D, including super-Gaussian shapes; see
[initialization](../../usage/initialization.md).

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
