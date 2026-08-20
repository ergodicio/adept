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
are written. A spectral Hou--Li filter may be retained along local $y$, but its `dimensions`
must omit `x`.

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

The reconnection diagnostics report a normalized rate and flux only when the upstream
$B_x$ fields are antiparallel and balanced and the origin contains both an in-plane null/
$A_z$ saddle and a central current sheet. Invalid samples are stored as NaN rather than
turning Biermann-ring motion into a false reconnection rate. `bz_quadrupole_purity` is the
local L1 projection of $B_z$ onto the expected four-lobe `sign(x*y)` pattern.

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
are labeled by the `ell` and `m` coordinates.

See the [Joglekar 2014 reconstruction and hydro-coupling design](joglekar2014.md) for the
distinction between the runnable reduced benchmark and the planned long-time implicit solve.
