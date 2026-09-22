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
are written. On this path, a requested $x$ Hou--Li filter is implemented as a shard-local
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

The [MAGPIE analysis reference](magpie_analysis.md) defines the independent bulk
and Alfvén rate normalizations, physical output units, source-aware flux budgets,
and the carbon upstream scale-report CLI.

Nonrelativistic runs use the [discrete electric-work correction](electric_work.md)
by default. Relativistic runs retain the original electric operator. The
correction preserves the explicit velocity-tail terms and does not remove
current-projection work or guarantee distribution positivity.

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

## Field-solver modes

Choose `terms.field_solver.mode` from the following hierarchy. Stationary ions are
the default (`terms.ion_fluid.active: false`).

| Mode | Field evolution | Ion coupling and sharding |
| --- | --- | --- |
| `maxwell` (default) | Physical explicit Vlasov–Maxwell | Stationary ions; x sharding supported |
| `ampere` | Explicit Ampere residual divided by relative permittivity; explicit Faraday | Stationary ions; x sharding supported |
| `oshun-implicit` | Local discrete kinetic-current response solve; explicit Faraday | Stationary ions; sharding rejected |
| `kinetic-ohm` | Algebraic kinetic Ohm law with an Ampere current-moment projection | Stationary ions with x sharding, or unsharded coupled ions |

For slowed explicit Ampere, `relative_permittivity` is required and must be at least one:

```yaml
terms:
  field_solver:
    mode: ampere
    relative_permittivity: 1.0e6
```

The complete Ampere residual is divided by this factor. Light and plasma frequencies
are reduced by its square root, while the steady target remains $J=c^2\nabla\times B$.
The initial Poisson field satisfies the same permittivity-weighted Gauss constraint,
and electric-field energy is weighted by that permittivity. This is still an explicit
mode, so its timestep must resolve the slowed field dynamics and kinetic transport.

The OSHUN-style alternative uses:

```yaml
terms:
  field_solver:
    mode: oshun-implicit
    response_regularization: 0.0
```

It computes a local $3\times3$ response $\partial J_i/\partial E_j$ and solves for the
electric field that brings the kinetically updated distribution to $J=c^2\nabla\times B$.
The electric force updates the harmonic distribution without projecting `f1`. Faraday
and non-electric transport remain explicit; this is not a fully implicit Maxwell solve.
Both `grid.lmax >= 1` and `grid.mmax >= 1` are required. Optional
`response_regularization` adds a nonnegative diagonal shift to the response matrix;
its default of zero retains the unregularized Ampere constraint.

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

The optional hidden gradient is available only with stationary ions. It is the unresolved
$\partial_z n$ used by the 2.5D PRL geometry.
It enters the pressure-gradient Ohm residual and can be switched sharply (`switch_width`
omitted) or with a differentiable tanh gate (`switch_width` set). Output variables prefixed
with `ohm_` contain the resistive, Hall, Nernst, scalar-pressure, and $f_2$ tensor-pressure
contributions.

## Moving ion-fluid coupling

Opt-in coupling is available for non-relativistic, unsharded `kinetic-ohm` runs on
periodic grids with `density.quasineutrality: true`:

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
    temperature_relaxation_rate: 0.0   # prescribed rate in normalized inverse time
    momentum_relaxation_rate: 0.0
    electron_pressure_feedback: true
```

`mass_ratio` is the mass of **one ion** divided by the electron mass; charge is
set separately by `units.Z`. A carbon-12 example uses approximately `12*u/m_e = 21874.66`.
Charge state is fixed. Rates are prescribed moment-relaxation rates, not an
atomic-kinetics or full finite-mass Landau model. Zero disables that exchange;
this must not be interpreted as predicting physical electron-ion equilibration.

`CoupledIonKineticStep` uses a symmetric composition:

1. Ion Euler half-step.
2. Coupled pressure, magnetic-force, and local exchange half-step.
3. Full kinetic-Ohm step at midpoint ion velocity, gradient, and acceleration.
4. Second coupled-source half-step.
5. Second ion Euler half-step.

The source steps include the full `f0 + f2` electron pressure force and $J\times B$
with midpoint ion mechanical work. Electron pressure work enters once through the
moving-frame deformation operator. Finite-mass frame remapping preserves electron
lab-frame moments as ion velocity changes. Both nonzero momentum and temperature
relaxation rates are supported; they default to zero and use a weak-drift moment
exchange model, not a full finite-mass Landau operator. Electron-pressure feedback
is enabled by default. `frozen: true` suppresses hydro and all coupled ion sources.

The laboratory electric field includes $-\mathbf u_i\times\mathbf B$. Initial ion
density is taken from the discretely integrated electron density so $n_e=Z n_i$
is exact initially; collision and IB density inputs are refreshed from midpoint ions.

Coupled runs require `grid.lmax >= 1` and `grid.mmax >= 1` for all three frame-momentum
components. Spatial sharding, nonperiodic boundaries, relativistic coordinates, and
active `field_solver.hidden_density_gradient` are rejected. These restrictions also
apply when `frozen: true`. The initial ion half-step is checked against the configured
acoustic/advection `cfl`; the timestep must remain conservative as the ion state evolves.
Local transport and nonlinear conservation/refinement tests are implemented, but
production-scale validation remains outstanding. This CFL check is not a complete
magnetic, Hall, electron-streaming or gyrofrequency stability bound; timestep and
radial resolution convergence remain required. See the
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

## Reconnection diagnostics

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
are labeled by the `ell` and `m` coordinates. All field modes save `ampere_target_current`,
`ampere_residual`, `ampere_residual_linf`, and `magnetic_field_energy`, plus solver-mode
and relative-permittivity attributes. Explicit `maxwell`/`ampere` runs also save electric
and total electromagnetic field energy. Relativistic current diagnostics use the same
$p^2(p/\sqrt{1+p^2})$ weight as evolution. Coupled runs additionally save the ion conserved
state and primitives, particle counts, quasineutrality residual, lab-frame total momentum,
separate electron/ion/magnetic energies, total energy, magnetic-divergence residual, negative
isotropic mass, harmonic free energy, and the `ohm_bulk` field.
Because `kinetic-ohm` has no displacement-current evolution, its algebraic electric field
does not carry a separately evolved field-energy term; `total_energy` is electron lab-frame
kinetic plus ion total plus magnetic energy. `current_projection_energy` records the
lab-frame electron work introduced by the Ampere projection, and
`accounted_total_energy = total_energy - current_projection_energy` exposes the remaining
coupled energy defect.

See the [Joglekar 2014 reconstruction and hydro-coupling design](joglekar2014.md) for the
distinction between the implemented field modes and the remaining full-benchmark validation.
