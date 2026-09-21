# VFP-2D Configuration

Set `solver: vfp-2d`.

## Grid

```yaml
grid:
  xmin: 0um
  xmax: 20um
  nx: 32
  ymin: 0um
  ymax: 20um
  ny: 32
  tmin: 0fs
  tmax: 100fs
  dt: 0.05fs
  nv: 96
  vmax: 8.0
  lmax: 5
  mmax: 3
  relativistic: false
  sharding: {enabled: false, axis: x}
```

`lmax` is the highest retained $\ell$. `mmax` defaults to `lmax`; lowering it provides a controlled transverse-angular truncation. For compatibility, `nl` is accepted as an alias for `lmax`.

By default, `vmax` is expressed in the same number-of-thermal-speeds convention as VFP-1D. Set `vmax_is_normalized: true` to provide the radial coordinate directly in code units. In relativistic mode this direct coordinate is $p/(m_ec)$.

Set `sharding.enabled: true` to partition the state, spatial drivers, and collision batches
along $x$ over every visible JAX device. `nx` must be divisible by the device count. The
sharded path uses fourth-order periodic finite differences (with two-cell halo exchange)
for spatial derivatives, because a global Fourier transform along a partitioned axis would
replicate the dominant distribution array. Saved snapshots are replicated only when they
are written. On this path, an requested $x$ Hou--Li filter is implemented as a shard-local
eighth-difference Nyquist filter with four-cell halo exchange; $y$ retains the spectral
Hou--Li filter.

## Initial distribution

VFP-1D `species-*` components are accepted. A profile without an axis is applied along $x$. Separable 2D profiles use `x` and `y` children:

```yaml
density:
  quasineutrality: false
  species-electron:
    m: 2.0
    n:
      x: {basis: cosine, baseline: 1.0, amplitude: 1.0e-4, wavelength: 20um}
      y: {basis: cosine, baseline: 1.0, amplitude: 1.0e-4, wavelength: 20um}
    T: {basis: uniform, baseline: 1.0}
```

Supported analytic bases are `uniform`, `sine`, `cosine`, and `tanh`; file profiles retain the VFP-1D loader behavior. With `quasineutrality: true`, the stationary ion charge follows the initial electron density. With `false`, it is spatially uniform at the mean density and the initial Poisson solve produces the field associated with electron-density perturbations.

See [periodic MAGPIE geometry and physical-unit imports](magpie_geometry.md) for
spatial ion velocity/temperature, vector-potential magnetic initialization, and
explicit-unit two-dimensional NPZ electron profiles.

## Laser heating

VFP2D shares the conservative inverse-bremsstrahlung and Maxwellian heating operators with
VFP1D. Heating amplitudes may be spatially uniform or multiplied by a two-dimensional
profile. The two-spot profile used by the Joglekar benchmark is:

```yaml
drivers:
  ib:
    intensity_1e15_Wcm2: 0.25
    polarisation: linear
    profile:
      basis: gaussian_spots
      x_center: 0um
      x_radius: 17um
      y_centers: [-8.5um, 8.5um]
      y_radius: 17um
```

`gaussian_spots` evaluates
$A\exp[-((x-x_0)/r_x)^2]\sum_i\exp[-((y-y_i)/r_y)^2]$.
`maxwellian_heating` accepts the same optional `profile` child with a scalar `D0`.

## Collisions

```yaml
terms:
  fokker_planck:
    active: true
    flm:
      ee: true
    f00:
      model: CoulombianKernel
      scheme: chang_cooper
```

`flm.ee: true` uses the full linearized anisotropic electron-electron terms. `false` uses the Epperlein-Haines $Z_*$ approximation. The `f00` model and differencing choices are shared with VFP-1D.
`chang_cooper` is recommended when heating produces a strongly non-Maxwellian distribution
because it is positivity preserving. `log_mean` has a zero semidiscrete spherical-energy
derivative for the kernel model at the frozen distribution, but a finite implicit update is
not exactly energy conserving away from a Maxwellian. Collision-step convergence is required;
keep the collision half-step well below the shortest relevant collision time.

## Long-timescale kinetic Ohm mode

`maxwell` is the default field solver. For collisional transport times, `kinetic-ohm` suppresses
displacement current and electron plasma oscillations, evaluates the full Joglekar Eq. (2),
and projects the current moment onto quasistatic Ampere's law:

```yaml
terms:
  field_solver:
    mode: kinetic-ohm
    hidden_density_gradient:
      active: true
      scale_length: 17um
      switch_off: 17.78ps
      profile:
        basis: gaussian_spots
        x_radius: 17um
        y_centers: [-8.5um, 8.5um]
        y_radius: 17um
```

The optional hidden gradient is the unresolved $\partial_z n$ used by the 2.5D PRL geometry.
It enters the pressure-gradient Ohm residual and can be switched sharply (`switch_width`
omitted) or with a differentiable tanh gate (`switch_width` set). Output variables prefixed
with `ohm_` contain the resistive, Hall, Nernst, scalar-pressure, and $f_2$ tensor-pressure
contributions.

## Moving ion-fluid coupling

Moving ions are opt-in for non-relativistic, unsharded `kinetic-ohm` runs on
periodic grids:

```yaml
terms:
  field_solver: {mode: kinetic-ohm}
  ion_fluid:
    active: true
    mass_ratio: 21874.66
    gamma: 1.6666666666666667
    cfl: 0.4
    boundaries: [periodic, periodic]
    initial_velocity: [0.0, 0.0, 0.0]  # legacy constants normalized to c
    frozen: false
    electron_pressure_feedback: true
    temperature_relaxation_rate: 0.0   # prescribed rate in normalized inverse time
    momentum_relaxation_rate: 0.0
```

`mass_ratio` is the mass of **one ion** divided by the electron mass; charge is
set separately by `units.Z`. A carbon-12 example uses approximately `12*u/m_e = 21874.66`.
Charge state is fixed. Rates are prescribed moment-relaxation rates, not an
atomic-kinetics or full finite-mass Landau model. Zero disables that exchange;
this must not be interpreted as predicting physical electron-ion equilibration.

The symmetric coupled map applies hydro and pressure/exchange/magnetic half-kicks
around the full kinetic step. Ion momentum receives `J cross B - div(Pe)`;
magnetic force also contributes its mechanical work. Changes in ion velocity
remap the kinetic electron distribution between frames. Collision densities use
the midpoint ion state. The laboratory electric field contains the ideal bulk
term `-ui cross B`. `frozen: true` holds ions fixed through hydro and all coupled
source updates.

Initial ion density follows the discretely integrated electron density, so
`ne = Z ni` at initialization. Quasineutrality is diagnosed during evolution.
The initial hydro half-step is checked against the acoustic/advection `cfl`;
this is not a complete magnetic, Hall, electron-streaming or gyrofrequency
stability bound, and later states may be more restrictive. Timestep and radial
resolution convergence remain required.

The moving-ion path rejects spatial sharding, nonperiodic boundaries, and
relativistic coordinates. The alternative implicit-current and slowed-Ampere
field modes currently support stationary ions only. See the
[ion-frame derivation](moving_frame.md) for the implemented operators and their
validation boundaries.

## Driven periodic reservoirs

A prescribed initial state can be maintained in smooth boundary bands:

```yaml
drivers:
  reservoir:
    active: true
    target: initial_state
    relaxation_time: 1ns
    x_width: 2mm
    y_width: 1mm
    magnetic: true
```

Each width must be smaller than the corresponding half-box; zero disables that
pair of bands. The target is the fully initialized state. Particle mixing is
performed in a common ion frame, magnetic increments are curls, and measured
source additions are saved for particle number, momentum and energy. This is a
periodic forced interaction-region model, not an open boundary condition. See
[reservoir equations, budgets and buffer tests](reservoirs.md).

The reconnection diagnostics report a normalized rate and flux only when the upstream
$B_x$ fields are antiparallel and balanced and the origin contains both an in-plane null/
$A_z$ saddle and a central current sheet. Invalid samples are stored as NaN rather than
turning Biermann-ring motion into a false reconnection rate. The normalized rate is also
suppressed when the signed inward Nernst speed is below 10% of its maximum over the saved
history, where division by a vanishing inflow would otherwise create a spurious spike.
`bz_quadrupole_purity` is the local L1 projection of $B_z$ onto the expected four-lobe
`sign(x*y)` pattern. Full-domain facets use a free display aspect ratio so wide boxes remain
readable. The `plots/reconnection_region` artifact directory also contains a curated set of
moments and Ohm-law terms cropped to the central three y-half-widths in x, excluding the
distant x boundaries and the outermost 25% of the y domain. This isolates the X-point and
current-sheet neighborhood from the outflow extent and laser-source boundary cells.

## Saving

```yaml
save:
  t:
    tmin: 0fs
    tmax: 100fs
    nt: 101
```

Post-processing returns an xarray dataset with `flm_real`, `flm_imag`, `e`, `b`, density,
temperature, current, Nernst velocity, and the traceless pressure-anisotropy moment. Harmonics
are labeled by the `ell` and `m` coordinates. Coupled runs additionally save the ion conserved
state and primitives, particle counts, quasineutrality residual, lab-frame total momentum,
separate electron/ion/magnetic energies, total energy, magnetic-divergence residual, negative
isotropic mass, harmonic free energy, and the `ohm_bulk` field.
Because `kinetic-ohm` has no displacement-current evolution, its algebraic electric field
does not carry a separately evolved field-energy term; `total_energy` is electron lab-frame
kinetic plus ion total plus magnetic energy.

See the [Joglekar 2014 reconstruction and hydro-coupling design](joglekar2014.md) for the
distinction between the runnable reduced benchmark and the planned long-time implicit solve.
