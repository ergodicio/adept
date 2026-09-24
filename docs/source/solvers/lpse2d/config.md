# LPSE-2D (Envelope-2D) Configuration Reference

This document describes how to construct a configuration file for the `envelope-2d` solver. This is a 2D laser-plasma simulation using envelope equations for electron plasma waves (EPW). It supports two-plasmon decay (TPD), stimulated Raman scattering (SRS), and other laser-plasma instabilities.

## Top-Level Structure

```yaml
solver: envelope-2d

units:
  # Physical unit normalizations

density:
  # Density profile configuration

grid:
  # Simulation grid parameters

save:
  # Output configuration

mlflow:
  # Experiment tracking

drivers:
  # Laser and EPW drivers

terms:
  # Physics terms configuration
```

## units

Physical unit normalizations. Note: This module uses different unit keys than the Vlasov modules.

| Field | Type | Description |
|-------|------|-------------|
| `atomic number` | int | Atomic number of the ion species |
| `envelope density` | float | Reference density as fraction of critical density |
| `ionization state` | float | Ionization state Z (LPSE `physical.Z` is a float; the deck translator passes it unrounded) |
| `laser intensity` | string | Laser intensity with unit, e.g., `"3.5e+14W/cm^2"` |
| `laser_wavelength` | string | Laser wavelength with unit, e.g., `"351nm"` |
| `reference electron temperature` | string | Electron temperature with unit, e.g., `"2000.0eV"` |
| `reference ion temperature` | string | Ion temperature with unit, e.g., `"1000eV"` |

Example:
```yaml
units:
  atomic number: 40
  envelope density: 0.25
  ionization state: 6
  laser intensity: 1.5e+14W/cm^2
  laser_wavelength: 351nm
  reference electron temperature: 2000.0eV
  reference ion temperature: 1000eV
```

## density

Density profile configuration.

| Field | Type | Description |
|-------|------|-------------|
| `basis` | string | Profile type: `"uniform"` or `"linear"` |
| `val` | float | Density fraction of critical (for `uniform` basis). **Defaults to 1.0 — at critical density — if omitted**, which is almost never what you want; set it explicitly |
| `gradient scale length` | string | Scale length with unit (for `linear` basis) |
| `max` | float | Maximum density fraction (for `linear` basis) |
| `min` | float | Minimum density fraction (for `linear` basis) |
| `lpse-<shape>` bases | | The original LPSE profiles (`ZakharovSolver::backgroundDensityShape`), shape in `linear`, `exp`, `gaussian`, `inverse-power`, `quadratic`, `qd`, `gd`, `file`: `min`/`max` at `min_location`/`max_location` (x; `min_location_y`/`max_location_y` optional, `geometry: cartesian|spherical`), `sg_order` (LPSE `sgOrder`, default 2) for `gaussian`/`inverse-power`/`gd`, `central_density` for `quadratic`, `dip_depth`/`dip_width`/`dip_offset` for the `qd` (parabolic) and `gd` (super-Gaussian) dips on a linear ramp, `origin` (x of LPSE's box centre for the quadratic and the dips, default the box centre), `max_density` clip (default 1.25), `file` (`.npy`, text table, or LPSE grid file) for `lpse-file`. See `helpers._lpse_density_profile` for the formulas |
| `noise` | object | Ignored (legacy). The initial EPW is identically zero; noise-seeded runs use the per-step `terms.epw.source.noise` source instead |

### Example: Uniform Density

```yaml
density:
  basis: uniform
  val: 0.2
  noise:
    max: 1.0e-09
    min: 1.0e-10
    type: uniform
```

### Example: Linear Density Gradient

```yaml
density:
  basis: linear
  gradient scale length: 50um
  max: 0.28
  min: 0.18
  noise:
    max: 1.0e-09
    min: 1.0e-10
    type: uniform
```

Note: When using `linear` basis, the grid size is automatically computed from the gradient scale length and density range.

## grid

Simulation grid parameters. Note: Grid values use physical units as strings.

| Field | Type | Description |
|-------|------|-------------|
| `boundary_abs_coeff` | float | Absorbing boundary coefficient (`tanh` profile: the amplitude damping rate, 1/ps, on the layer plateau) |
| `boundary_width` | string | Width of the EPW absorbing layer with unit (LPSE `lw.Labc`; `0um` = none with the `exp` profile). The light and IAW layers have their own widths (`terms.light.boundary_width`, `raman_boundary_width`, `terms.iaw.boundary_width`), defaulting to this one. With `exp`, every layer is LPSE's index-based profile (`absorbingBoundaries.cpp`, `helpers.lpse_exp_rate`) |
| `boundary_profile` | string | (optional, default `tanh`) Shape of the absorbing layers. `tanh` is the MATLAB envelope (rise `boundary_width / 5`). `exp` is the original LPSE profile (`absorbingBoundaries.cpp`): rate `boundary_max_rate * (e^{lambda s/L} - 1) / (e^lambda - 1)` with `s` the distance into the layer measured from the edge cell (so the edge cell carries the full rate and the layer spans `boundary_width / dx` cells); with both axes absorbing the two rates combine by `max`, as in LPSE |
| `boundary_max_rate` | float | (`exp` profile) Peak amplitude damping rate in 1/ps (default `200`, LPSE `lw.abc.maxDampingRate`). The IAW absorber uses half of it unless `terms.iaw.boundary_max_rate` is set (LPSE `IawSolver` default 100) |
| `boundary_lambda` | float | (`exp` profile) Exponential steepness (default `7`, LPSE `abc.lambda`) |
| `low_pass_filter` | float | Low-pass filter cutoff as fraction of kmax (0-1) |
| `dealias` | string | Shape of the anti-aliasing mask: `isotropic` (default), `shifted-band` or `rectangular` (LPSE `grid.antiAliasing.range`: the outer `1 - low_pass_filter` of each k axis is zeroed; the deck translator uses it with LPSE's range, `0.3334` when the deck omits the key, ParameterManager.cpp:241) |
| `smooth_fft_size` | bool | (default `true`) grow `nx` / `ny` to 5-smooth FFT sizes and rescale `dx`; `false` keeps the node counts `(xmax - xmin) / dx`, `(ymax - ymin) / dx` exactly (LPSE runs on `grid.nodes`; the deck translator sets `false`) |
| `dt` | string | Timestep with unit (the EPW step). The deck translator uses LPSE's own steps (`Lpse::computeMicroTimestep` / `setupSolverTimeSteps` and each solver's `Tstep`, ported in `lpse_deck.lpse_time_steps` and checked against the "Time step sizes" every reference run prints): `dt` is the EPW step (the IAW step without an EPW solver, the light step with neither), `light_substeps` its ratio to the finer light step and `terms.iaw.stride` the IAW step in EPW steps; LPSE makes each a multiple of the finer one, so the ratios are exact. Pump and Raman light stepped differently by LPSE are both run at the finer step (reported) |
| `dx` | string | Spatial resolution with unit |
| `tmax` | string | End time with unit |
| `tmin` | string | Start time with unit |
| `ymax` | string | Domain maximum y with unit |
| `ymin` | string | Domain minimum y with unit |
| `light_substeps` | int | (dynamic light only, optional) Light-wave sub-steps per EPW step. `fd`: computed from the stability limit of the staggered (leapfrog) scheme for every evolved carrier `w` (`w1` when SRS is on, `w0` when pump depletion is on), `dt_light <= 4 w / max(|D_max|, |c^2 K_max - D_min|)` with `D = w^2 - w0^2 n` over the box densities and `K_max` the largest eigenvalue of the discrete curl curl (`stencils.curl_curl_max_eigenvalue`): the in-plane operator's `4/dx^2` times 1, 1.405, 1.686 at `fd_order` 2, 4, 6 when `E_z` is never excited (every beam and the seed p-polarised, no `E_z` injector file), the 2-D Laplacian's (twice that) otherwise -- LPSE's `Tcritical` with its empirical stencil factors replaced by the exact eigenvalue. The automatic choice keeps a 10 % margin; a `ValueError` is raised if a user-supplied value exceeds the tightest limit. `spectral`: no stability limit; a warning is printed when light crosses more than one cell per sub-step |
| `probe_offset` | string | (SRS only, optional) Distance of the laser-budget flux probes from each box edge, with unit. Default `2 * boundary_width`, which is clear of the absorber's tanh skirt (the legacy `reflectivity` probe at `1.6 * boundary_width` sits inside it and reads ~10% low). The deck translator puts the probes 4 cells inside the deepest injector plane or absorbing layer |

Note: `nx` and `ny` are computed automatically from the grid parameters. The grid is optimized for FFT performance (sizes with small prime factors).

Note: setting `ymax`/`ymin` smaller than `dx` collapses the box to `ny = 1`, which runs the solver in a true 1D mode (useful for cheap 1D SRS simulations; TPD requires 2D).

### Anti-aliasing: `dealias`

The TPD and SRS source terms are products of the pump with a plasma-wave field, formed pointwise in
real space. Such a product aliases if it puts content past the Nyquist wavenumber, so part of the
band has to be left empty.

Because the pump is built as a plane wave along x (`laser.py`), the product *translates* the
plasma-wave spectrum by `k0` rather than convolving it against a broad kernel. The band that has to
stay empty is therefore a rectangle, not a disc, and the usual 2/3-style isotropic cutoff is the
wrong shape for the job — it discards high-`ky` modes that can never alias.

| Value | Mask |
|-------|------|
| `isotropic` (default) | `|k| < low_pass_filter * kmax` only. Alias-free only if `low_pass_filter * kmax + k0 <= kmax`, which is not checked. |
| `rectangular` | `|kx| < low_pass_filter * kmax_x` and `|ky| < low_pass_filter * kmax_y` (the original LPSE mask). |
| `shifted-band` | Additionally requires `|kx| <= kmax_x - k0` and `|ky| <= kmax_y - k0 * NA`, which is exactly alias-free for the source products. `NA` is the numerical aperture of the speckle profile, and is zero without one. |

`shifted-band` computes its limits from `k0` and the grid, so it stays correct as `dx`, the laser
wavelength, or the density change — unlike a hand-tuned `low_pass_filter`.

The two knobs are independent, and `low_pass_filter` is still applied on top:

- `dealias` handles aliasing.
- `low_pass_filter` is a *physics* cap. The Landau damping rate in `epw.py` is the asymptotic
  small-`k*lambda_D` expression, which peaks near `k*lambda_D ~ 0.7` and then decreases, so it
  under-damps beyond that. Keep the band edge below roughly `k*lambda_D = 0.5`.

Both limits are printed at setup, along with the fraction of the k-grid retained and the
`k*lambda_D` the band edge reaches, so the interaction between the two is visible.

To take advantage of `shifted-band`, raise `low_pass_filter` until the printed `k*lambda_D` is as
large as you are willing to trust:

```yaml
grid:
  dealias: shifted-band
  low_pass_filter: 1.0
```

Example:
```yaml
grid:
  boundary_abs_coeff: 1.0e4
  boundary_width: 1.5um
  low_pass_filter: 0.66
  dt: 0.010fs
  dx: 40nm
  tmax: 2ps
  tmin: 0.0ns
  ymax: 0.08um
  ymin: -0.08um
```

## save

Configures what data to save and at what times.

### Structure

```yaml
save:
  fields:
    t:
      dt: 100fs
      tmax: 4ps
      tmin: 0ps
    x:
      dx: 50nm
    y:
      dy: 50nm
```

### fields

| Field | Type | Description |
|-------|------|-------------|
| `t` | object | Temporal save configuration |
| `x` | object | Optional spatial subsampling in x |
| `y` | object | Optional spatial subsampling in y |

#### t (temporal)

| Field | Type | Description |
|-------|------|-------------|
| `dt` | string | Time interval between saves, with unit |
| `tmax` | string | End time for saving, with unit |
| `tmin` | string | Start time for saving, with unit |

#### x (optional)

| Field | Type | Description |
|-------|------|-------------|
| `dx` | string | Spatial resolution for saved data, with unit |

#### y (optional)

| Field | Type | Description |
|-------|------|-------------|
| `dy` | string | Spatial resolution for saved data, with unit |

## mlflow

Experiment tracking configuration.

| Field | Type | Description |
|-------|------|-------------|
| `experiment` | string | MLflow experiment name |
| `run` | string | MLflow run name |

Example:
```yaml
mlflow:
  experiment: tpd
  run: srs-test
```

## drivers

Laser and EPW drivers.

### E0 - Pump Laser Driver

The main laser pump for TPD/SRS simulations.

| Field | Type | Description |
|-------|------|-------------|
| `envelope` | object | Spatiotemporal envelope |
| `delta_omega_max` | float | Maximum frequency spread (optional) |
| `num_colors` | int | Number of laser colors (optional) |
| `angle` | float | (default `0`) in-plane angle of incidence from +x in degrees (LPSE `laser.N.direction`). The static pump is the single grid mode nearest to `k0 (cos a, sin a)` (both components snapped, as LPSE's `makeStaticField`), polarized perpendicular to the snapped k; the spectral pump injector launches at the y-snapped transverse wavenumber with the x group velocity setting the flux. Not with `speckle` or the FD injector `|angle| > 90` is a leftward beam (LPSE `laser.N.direction` with a negative x): the evolved pump is launched from the x-max face at `xmax - drivers.E0.offset` (with that plane's density) and the flux probes keep the right probe upstream of its rows -- with the spectral solver any mix of beams from both faces (the CBET / SBS-backscatter geometry), with the FD injector a single beam at exactly 180. `+-90` (a y face) is not supported |
| `polarization` | number or string | (default `p`) polarization angle in degrees about the beam axis, LPSE `laser.N.polarization` (`rotateBeam`: the field starts along y, is rotated about the beam axis by the angle, then carried onto the beam direction): `0` / `p` is in-plane, `90` / `s` is along z, the out-of-plane component of the three-component light fields. Every injector (FD two-point rows, spectral Gaussian source, combined-solver source) and the static pump launch `cos(psi)` along the in-plane transverse direction and `sin(psi)` along z; per-beam values go in `beams[].polarization`. An s-polarised pump drives SRS (into an s-polarised Raman wave) and no TPD |
| `beams` | list | LPSE `laser.N.*` beamlets for the static pump and the spectral injector: `[{intensity, angle, phase, delta_omega}]` (intensities are normalised to fractions; every beam carries every color; `angle` in degrees, `phase` in radians, `delta_omega` as a fraction of w0) |
| `beam_width` | string | transverse (y) standard deviation `sigma` of the injected beams, LPSE `laser.N.evolution.width` `W = sqrt(2) sigma` (the translator converts); with `beam_sg_order` (LPSE `laser.N.evolution.sgOrder`, default 4 as LPSE, 2 a Gaussian, 0 a flat beam) and `beam_offset` (y of the beam centre) it sets LPSE's super-Gaussian `exp(-(|y - y0| / W)^p)` (`SchrodingerSolver3::superGaussian`) of the spectral and general FD injectors |
| `kap_bandwidth` | float | (default `0`) Kubo-Anderson phase bandwidth `dW / w0` of the pump beams (LPSE `laser.N.bandwidth.KAP.frequency`): each beam holds a uniform random phase for a dwell time `2 X / dW`, `X ~ Exp(1)` (mean `2 / dW`), then jumps to a new one, independently of the other beams (LPSE `computeKapTransitionTime`; `core/kap.py`), on the static pump and on every injector (spectral, FD row and general). `kap_seed` seeds the drawn transitions. The deck translator reads beam 1's key and reports differing per-beam bandwidths or shared `laser.N.group`s |
| `pulse_shape` | string | (optional) LPSE `laser.pulseShape.shape`, a *power* factor on the pump: every injector (analytic beams, the FD row and general injectors, the combined solver and injector files) multiplies its field by `sqrt(shape)` and the static pump's field scales as `sqrt(max(shape, 1e-12))` (`SchrodingerSolver3::addInjectorSources`, `LightSolver::applyPulseShapeStatic`; `adept/_lpse2d/core/pulse.py`). `file` (the default when `pulse_file` is set), `square` (`1 / pulse_duty_cycle` for the first `pulse_duty_cycle` of every `pulse_period`, else 0) or `sin` (`2 sin^2(pi t / pulse_period)`) |
| `pulse_file` | string | LPSE `laser.pulseShape.file`: a two-column text table `(t_ps, power scale)`, `#` and `//` comments, times non-negative and strictly increasing, scales in `[0, 10]` (as LPSE's loader), interpolated linearly and held at its end values |
| `pulse_period` | string | (`square`, `sin`) LPSE `laser.pulseShape.period`, default `0.1ps` |
| `pulse_duty_cycle` | float | (`square`) LPSE `laser.pulseShape.dutyCycle`, default `0.5` |
| `shape` | string | Amplitude shape: `"uniform"` (optional) |
| `offset` | string | (pump depletion only, optional) Distance of the pump boundary injector from `xmin` (`xmax` for a leftward beam), with unit: the injector's last scattered-field row (from x-min) or first injected row (from x-max). Default `2 * boundary_width`; the deck translator puts it on LPSE's plane, `int(Labc / h) + int(Loff / h)` nodes in from the face (`SchrodingerSolver3::completeBoundaryConditions`) |
| `turn_on_time` | string | (pump depletion only, optional) Gaussian turn-on time of the injector. Default `10fs` |
| `injector_width` | string | (pump depletion with `terms.light.solver: spectral`, optional) Gaussian width of the smooth pump injector, with unit. Default one local wavelength `2 pi / k0(n_inject)` |
| `injector_file` | object | (pump depletion with the second-order `fd` light solver, optional) the pump launched from LPSE injector files instead of analytic beams, LPSE `laser.E_<c>.loadInjector.<side>.x.filename` (files made by `matlab/m201902_createLpseInjector_v02.m`): `{side: min.x \| max.x, files: {x \| y \| z: path}}`. Each file holds, per time, the time in ps and the complex field (in `e E / (m_e w0 c)`, single precision) of that component on the injector plane and on the plane one cell further in, `ny` points each. The first plane sits on the first injected row -- the row after the one nearest `xmin + offset` from x-min, the row nearest `xmax - offset` from x-max; the translator puts it on LPSE's `int(Labc / h)`. The source is LPSE's `getInjectorSource`: minus the propagation operator applied to the two planes, on the plane's row and the row outside it. Several times are interpolated linearly and repeat with the last time as the period, as LPSE; `turn_on_time` ramps the source. Not with `beams`, `angle`, `beam_width`, more than one color, `kap_bandwidth` or `speckle` (LPSE refuses beams with injector files); a pulse shape applies to the file's source as to every injector |

#### envelope

All values are strings with physical units.

| Field | Type | Description |
|-------|------|-------------|
| `tc` | string | Temporal center |
| `tr` | string | Temporal rise time |
| `tw` | string | Temporal width |
| `xc` | string | Spatial center (x) |
| `xr` | string | Spatial rise (x) |
| `xw` | string | Spatial width (x) |
| `yc` | string | Spatial center (y) |
| `yr` | string | Spatial rise (y) |
| `yw` | string | Spatial width (y) |

Example:
```yaml
drivers:
  E0:
    delta_omega_max: 0.015
    envelope:
      tc: 200.25ps
      tr: 0.1ps
      tw: 400ps
      xc: 50um
      xr: 0.2um
      xw: 1000um
      yc: 50um
      yr: 0.2um
      yw: 1000um
    num_colors: 1
    shape: uniform
```

### E2 - EPW Driver (Optional)

Direct EPW driver for seeding or testing.

| Field | Type | Description |
|-------|------|-------------|
| `envelope` | object | Same structure as E0 envelope |
| `a0` | float | Amplitude |
| `k0` | float | Wavenumber |
| `w0` | float | Frequency |

Example:
```yaml
drivers:
  E2:
    envelope:
      tw: 200fs
      tr: 25fs
      tc: 150fs
      xw: 500um
      xc: 10um
      xr: 0.2um
      yr: 0.2um
      yc: 0um
      yw: 50um
    a0: 1000
    k0: -10.0
    w0: 20.0
```

### E1 - Raman Seed Driver (Optional)

Injects a counter-propagating (-x) scattered-light wave for seeded SRS. Only used when
`terms.epw.source.srs` is on. The injector sits at `x = xmax - offset` and drives the
`E1` field with a two-point antisymmetric source (the MATLAB LPSE injector), so the seed
propagates toward the low-density side while backscatter growth amplifies it against the pump.

| Field | Type | Description |
|-------|------|-------------|
| `intensity` | string | Seed vacuum intensity with unit, e.g. `"1.0e+12W/cm^2"` |
| `delta_omega` | float | Seed frequency shift relative to `w1 = w0 - wp0`, as a fraction of `w1` (default 0) |
| `turn_on_time` | string | Ramp-up time of the injector, `1 - exp(-(t / turn_on_time)^2)` (default `10fs`; the deck translator uses LPSE's `raman.evolution.riseTime`, default 30 fs) |
| `offset` | string | Distance of the injector from the right boundary (its first injected row). Defaults to `1.6 * boundary_width`, just inside the absorbing boundary's tanh skirt; with the tanh profile a warning is printed for smaller values because the seed would be damped at the source. The deck translator puts it on LPSE's plane, `int(Labc / h) + int(Loff / h)` nodes in from x-max (raman.evolution) |
| `yw` | string | Super-Gaussian (4th order) width of the seed in y; omit for uniform in y |
| `injector_width` | string | (`terms.light.solver: spectral`, optional) Gaussian width of the smooth seed injector, with unit. Default one local wavelength `2 pi / k1(n_inject)` |
| `polarization` | number or string | (default `p`) seed polarization angle in degrees about the beam axis (LPSE `raman.N.polarization`): `0` / `p` in-plane (y), `90` / `s` along z; the injector writes `cos(psi)` to y and `sin(psi)` to z |

The density at the injector must be below the `w1` critical density (`n < 0.25 n_c` for
envelope density 0.25), otherwise the seed is evanescent and setup raises an error. Without
`drivers.E1`, SRS grows from the EPW noise source instead (the noise-seeded configuration).

Example:
```yaml
drivers:
  E1:
    intensity: 1.0e+12W/cm^2
    delta_omega: 0.0
    turn_on_time: 10fs
```

## terms

Physics terms configuration.

Each EPW step (`grid.dt`) follows LPSE's order (`ZakharovSolver::evolve`): the ion-acoustic waves (every `terms.iaw.stride` steps) driven by the fields at `t`; the EPW, whose TPD / SRS sources see the light at `t` and whose detuning sees the ion density at the middle of the step; then the light (`grid.light_substeps` sub-steps), each sub-step reading the EPW potential and the ion density at its middle, linearly between their values at the ends of the EPW and IAW steps (LPSE `interpolateSourcesInTime`, `LightSolver::computeDynamicE0` at `time + dt/2`; `terms.light.interpolate_sources`, `terms.epw.interpolate_sources`). Particles (`hpe`) and the quasilinear VDF (`qle`) follow. The combined solver advances the IAW first and then the pump and the combined field in the same way.

| Field | Type | Description |
|-------|------|-------------|
| `epw` | object | Electron plasma wave configuration |
| `light` | object | (optional) Light-wave evolution configuration |
| `iaw` | object | (optional) Ion-acoustic density evolution and ponderomotive feedback |
| `hpe` | object | (optional) Hybrid particle evolution: test-particle Landau damping feedback (Follett et al. 2017) |
| `qle` | object | (optional) Quasilinear evolution of the box-averaged electron distribution (LPSE `qle.*`, `QuasilinearEvolution.cpp`; plan 2 K.1). The VDF `f(v)` on `nv` points per dimension over `[-v_max c, v_max c]` (initially the Maxwellian of `Te`) diffuses in the EPW spectrum, `df/dt = d/dv_i (C_ij df/dv_j)` with `C_ij` the resonant-plane integral of `k_i k_j |phi_k|^2` (Bohm-Gross resonance with `thermal_correction`, whose group-velocity Jacobian is kept -- LPSE's cold `1/v` under-counts the exchange by `3 (k lambda_D)^2 wpe^2/w_k^2`), in conservative flux form sub-cycled to the diffusion limit (up to `max_subcycles`), relaxed towards the Maxwellian at `p_x \|v_x\|/Lx + p_y \|v_y\|/Ly` (`thermalization_probability`), every `update_every` EPW steps from `t_start` (ps). With `landau_evolution: true` the Landau rate of every mode is recomputed from `f` (`gamma_L(k) = -(pi/2) wpe^3 k . grad_v f / k^3` on the resonant plane; the Maxwellian reproduces the analytic `lpse` rate to 5 % at LPSE's default `nv` 100, 0.5 % at 400) and written to the state's `gamma_L` that the EPW step applies; modes whose phase velocity exceeds `v_max c` are undamped, as in LPSE. The diffusion constant is verified by the wave-energy / electron-heating balance (`test_qle.py`). Keys: `active`, `nv` (100), `v_max` (0.5), `update_every` (1), `t_start` (0), `thermal_correction` (true), `derivative_in_tensor` (accepted; implied by the flux form), `landau_evolution` (false), `thermalization_probability` ([1, 0]), `subcycling` (1), `max_subcycles` (20000), `multiplier` (1). Output: `binary/vdf.xr` (`vdf(t, vx[, vy])` at the field save times, `plots/vdf.png`) and `gamma_L` in the k-fields. Not with `terms.hpe`. The translator maps `qle.*` (test_029, test_033) |
| `zero_mask` | bool | Whether to zero out k=0 mode |

### light (optional)

| Field | Type | Description |
|-------|------|-------------|
| `pump_depletion` | bool | (default `false`) Evolve the pump `E0` with the staggered FD envelope solver instead of prescribing it analytically. The pump is launched at `x = xmin + drivers.E0.offset`; active SRS and TPD terms each add their reciprocal pump coupling, and both act when both instabilities are enabled. Requires at least one of `terms.epw.source.srs`/`tpd` and `terms.epw.boundary.x: absorbing`; incompatible with `drivers.E0.speckle`. Enables true net-flux `laser_reflectivity` / `laser_transmissivity` / `laser_absorbed_frac` metrics and above-threshold saturation |
| `coupling` | str | (default `explicit`; `pump_depletion` with `terms.epw.source.srs` only) How the SRS exchange between `E0` and `E1` is integrated inside each light sub-step. `explicit` keeps the MATLAB staggered real/imaginary update, which treats the part of the exchange proportional to `Im(laplacian phi)` with an explicit Euler step: the light pair grows by `1 + sin^2(arg laplacian phi) (Omega dt_l)^2 / 2` per sub-step, `Omega = e \|laplacian phi\| / (4 me sqrt(w0 w1))`, for any `dt_l`. That is invisible at small EPW amplitude but manufactures light (and, through the SRS/TPD sources, EPW) energy once the pump is depleted and `Omega dt_l` reaches ~0.005, ending in a one-step NaN. `rotation` Strang-splits each sub-step as [exact exchange rotation over `dt_l/2`] [staggered propagation with the exchange off] [rotation over `dt_l/2`]; the rotation `exp(tau M) = cos(Omega tau) I + sin(Omega tau)/Omega M` conserves the light action `w1 \|E0\|^2 + w0 \|E1\|^2` pointwise and is stable for any `dt_l`. The TPD pump term and the IAW detuning stay in the staggered update under both settings. See `tests/test_lpse2d/test_light_coupling.py` |
| `filter` | float | (default off; `pump_depletion` only) Isotropic low-pass filter applied to both light fields once per EPW step, keeping `\|k\| <= filter * pi/dx`. The physical light content lies below ~1.2 k0; grid-scale light modes have FD group velocity `c^2 sin(k dx)/(w dx) -> 0`. Diagnostic/numerical-hygiene option |
| `tpd_projection` | bool | (default `true`; `pump_depletion` with `terms.epw.source.tpd`) Take the transverse (divergence-free) part of `E_h div(E_h)` in k-space before it acts on the pump, `F_T = F - k (k . F)/k^2` mode by mode (LPSE `LwSolver::makeExyzDivE`), so the TPD pump term never injects a longitudinal component into the light field. Both pump components receive the term Not applied by the combined solver: LPSE takes the transverse part of source terms only outside combined mode (`LightSolver.cpp:4325`) |
| `tpd_k_filter` | bool | (default `false`) LPSE `lw.kFilter`: restrict the TPD pump term to `\|k\| < 1.2 k0 sqrt(1 - n_min)` |
| `transverse_source` | bool | (default `true`) project the SRS light sources onto their transverse (divergence-free) part every light sub-step (LPSE `raman.takeTransversePartOfSourceTerms`; with the coupled spectral solver the fields themselves are kept transverse). No light propagator moves the longitudinal part of E1, so without it that part accumulates the source and pairs with the EPW in a spurious two-wave instability. The spectral solvers also drop every component outside the retained light band, as LPSE does |
| `transverse_fields` | bool | (default `false`; `fd` solver) Project the evolved light fields -- `E1`, and `E0` with `pump_depletion` -- onto their transverse (divergence-free) part once per EPW step, after the light sub-steps. The FD curl-curl propagator (compact 3-point second differences with a centred cross difference) has a non-zero discrete divergence, so a transverse field with structure in both x and y acquires a longitudinal part every step: on a 0.1 um grid (`k0 dx = 1.8`) a random transverse band-limited pump reaches a 4 % longitudinal fraction (norm) in one EPW step, an oblique plane wave at 5 deg 5 %, at 20 deg 11-14 %; a y-uniform plane wave stays exactly transverse. No propagator moves that part, the EPW sources see the whole field, and the projected TPD depletion term (`tpd_projection`) can only take energy back from the transverse part, so the pair energy `\|E0\|^2 + (2 wp0/w0)\|E_h\|^2` is conserved only for a transverse pump. The spectral solver keeps its fields transverse by construction. Set `false` to reproduce the pre-plan-2 FD behaviour **Default off**: on the srs-2d-testbed P1 case (1600×400, 20 ps) the projected FD scheme goes non-finite at 1.31 ps with the EPW still at its noise floor (unprojected: 3.93 ps), so this is a study option rather than a fix for the separate-solver TPD + SRS blow-up (plan-2 run session) |
| `fd_order` | int | (default `2`; `fd` solver) order of the central stencils of the FD light propagator (LPSE `laser/raman.evolution.solverOrder` 2, 4 or 6; `SchrodingerSolver3::step_2d`): second differences `[1,-2,1]`, `[-1,16,-30,16,-1]/12`, `[2,-27,270,-490,270,-27,2]/180`, the cross derivative the product of the matching first differences. The pump and seed injectors are the stencil's total-field / scattered-field source (`RamanLight.injector_rows`): the two-point source at order 2, `order` rows about the plane in general, so the launched amplitude's grid-dispersion deficit falls from ~2 % at 8 cells per wavelength (order 2) to 0.1 % (4) and 0.01 % (6), and the flux probes use the stencil's own dispersion. The explicit scheme's light dt limit tightens by 4/3 (4) and 68/45 (6); the sub-step count follows automatically |
| `absorber` | string | (default `exp`; `fd` solver) the light fields' absorbing layers. `exp`: the multiplicative layer of `grid.boundary_profile` / `boundary_max_rate` applied every sub-step. `pml` (LPSE `{laser|raman}.evolution.abc.type = pml`, `SchrodingerSolver3::abc_compute`): a complex coordinate stretch of the Laplacian in the `boundary_width`-wide layers, `v = 1/(1 + e^{i pi / pml_denominator} delta^4)` with `delta` the depth into the layer, `sum_nb v (v + v_nb)/2 (E_nb - E)/h^2` on the compact stencil (the grad-div part keeps the plain stencil, as LPSE), no multiplicative damping, and the edge cells held at zero (LPSE's un-updated edge nodes -- the wall the attenuated wave reflects from on its way back). Normally incident pump, 1.5 um layer: reflected power 3e-6 (`pml`) vs 5e-5 (`exp`); 3 um: 1e-9. At `fd_order` 4 / 6 the compact PML branch under the higher-order interior reflects ~3e-4 at the stencil interface (LPSE has the same construction; every shipped PML deck is order 2), so `exp` (3e-5) is the better layer there |
| `pml_denominator` | float | (default `5`; LPSE `abc.SabcDenom`, "PI/5 for 0.351 um light, PI/2.5 for 1 um") the PML phase `pi / pml_denominator` |
| `resonance_absorption` | bool or object | (default off; `fd` solver with `pump_depletion`) LPSE `laser.evolution.resonanceAbsorption`: two split-step terms on the pump every light sub-step so that p-polarised light reaching the critical surface converts into a longitudinal component there and is damped -- the warm-plasma term `C0 [(div E) grad(n)/n - 3 grad(div E)]`, `C0 = v_te^2/(2 i w0)`, in x-space (masked off inside the absorbing layers with LPSE's quadratic ramp over `edge_width`, default 2 cells), and Landau damping of the longitudinal part at the EPW Landau rate in k-space, `E_k -= dt min(gamma_L, 1/dt) k (k . E_k)/k^2` (masked with an s-curve over the absorbers; the rate table is built whatever `terms.epw.damping.landau` says, as LPSE does), then an optional radial low-pass on the field (`filter: true`, an s-curve from `0.71 filter_width K_nyq` to zero at `filter_width K_nyq`). Keys: `t_start`, `t_stop`, `filter`, `filter_width` (default 1), `landau_update` (LPSE `LdUpdate`, the k-space term every N sub-steps; default 1), `edge_width`. Without it the cold resonance at `n_c` is driven without limit by p-polarised light. On a linear ramp with `(k0 L_n)^(1/3) sin(theta) = 0.8` the model absorbs 42 % of a 4th-order FD beam (tests/test_lpse2d/test_resonance_absorption.py); refused with the spectral solver as in LPSE. `raman.evolution.resonanceAbsorption` is not translated |
| `snap_beam_ky` | bool | (default `true`; `spectral` solver) snap an oblique beam's transverse wavenumber `k0 sin(angle)` to the periodic y grid so the injected pump is one exact grid mode (uniform in y). `false` launches the exact wavenumber as LPSE's spectral injector does: on a box whose width does not hold an integer number of transverse wavelengths the source has a phase kink at the periodic seam and its sidebands add to an intensity hot spot there (test_010: +45 %); useful only to reproduce such a reference run |
| `suppress_sources_in_absorbers` | bool | (default `false`) zero the EPW and IAW sources inside the absorbing layers (`boundary_width` from each non-periodic wall; LPSE `suppressSourcesInAbsorbingRegions`) |
| `suppress_sources_at_injectors` | bool | (default `false`; LPSE's default is `true` and the deck translator sets it) zero the EPW and IAW sources across the pump injector rows (with `pump_depletion`) and the seed injector rows: the two FD source rows, or the Gaussian source's width in cells for the spectral solver (LPSE `suppressSourcesAtInjectors`, `ssWidth`) |
| `boundary_max_rate` | float | peak amplitude damping rate (1/ps) of the light fields' absorbing layers with `grid.boundary_profile: exp` (LPSE `{laser|raman}.evolution.abc.maxDampingRate`, default `5e3`; the EPW layer uses `grid.boundary_max_rate`, 200). Light crosses a 3 um layer in 0.01 ps, so at the EPW rate the exp layer reflects ~75 % of the amplitude and an injected pump builds a standing wave between the walls. With the `tanh` profile both use `boundary_abs_coeff` |
| `boundary_width` | string | (`exp` profile) width of the pump's absorbing layer (LPSE `laser.evolution.Labc`); default `grid.boundary_width` |
| `raman_boundary_width` | string | (`exp` profile) width of the Raman light's layer (LPSE `raman.evolution.Labc`); default `boundary_width`. The combined solver's field sees LPSE's double exponential: the EPW layer (`grid.boundary_width`, `grid.boundary_max_rate`) inside this one (`raman_boundary_max_rate`), joined where the outer profile reaches the EPW maximum (`setupAbsorbingBoundaries_doubleExponential`, `helpers.lpse_double_exp_rate`) |
| `raman_boundary_max_rate` | float | (`exp` profile) peak rate of the Raman light's layer, 1/ps (LPSE `raman.evolution.abc.maxDampingRate`); default `boundary_max_rate` |
| `solver` | string | (default `fd`) Light propagator for the evolved fields (`E1`, and `E0` with `pump_depletion`). `fd`: the MATLAB staggered real/imaginary finite-difference scheme, sub-cycled to its CFL limit. `spectral`: the original LPSE `{laser\|raman}.solver = spectral` path -- per sub-step the x-space scattering potential `exp(i dt Vo)` (detuning + absorption), the coupling sources with forward Euler, and the exact k-space propagator `exp(-i dt c^2 k^2/(2 w))` on the transverse part of the field (the longitudinal part is left unpropagated, as in LPSE), modes outside the retained band zeroed. No CFL limit (`grid.light_substeps` defaults to 1), no grid dispersion, and with `pump_depletion` the SRS exchange is the exact local rotation of `coupling: rotation` Strang-split around the propagation. The injectors become smooth Gaussian sources (`drivers.E0/E1.injector_width`, one local wavelength by default) that launch exactly the requested amplitude with negligible leakage in the wrong direction |
| `max_wavenumber` | float | (optional; `spectral` only) Cap the retained light band at `max_wavenumber * k0` (LPSE `{laser\|raman}.maxWavenumber`) |
| `absorption` | bool or float | (default `false`) Collisional (inverse-bremsstrahlung) absorption of the evolved pump (LPSE `laser.evolution.absorption`; `raman_absorption` is the Raman light's): the amplitude decays at `nu (n/nc_w)^2` per wave, `nc_w` its own critical density and `n` including the IAW perturbation (LPSE `calculateScatteringPotential`). `true` takes `nu` from the NRL formula as coded in LPSE, `5.11e10 Z logLambda / (lambda_um^2 Te_keV^1.5) * 1e-12` 1/ps with each wave's own wavelength (`logLambda = 6.68 + ln(lambda_um Te)` for `Te > 0.01 Z^2`, else `9.13 + ln(lambda_um Te^1.5/Z)`); a number is the pump rate at `nc` in 1/ps, the Raman rate scaled by `(w1/w0)^2`. Works with both light solvers (applied per sub-step). LPSE's `resonanceAbsorption` is not implemented |
| `raman_absorption` | bool or float | (default: follows `absorption`) The Raman light's own absorption at its own critical density (LPSE `raman.evolution.absorption`, `wsAbsorptionAtNc` of the Raman class), `false`, `true` (NRL at the Raman wavelength) or a rate in 1/ps; unset, the NRL formula at the Raman wavelength or `absorption` scaled by `(w1/w0)^2`. The combined solver's field (carrier `wp0`, critical density `n_env`) is damped at `raman_absorption (n/n_env)^2` and its thermal noise uses the same rate, both over the light sub-step; `terms.epw.damping.collisions` does not act on it, as in LPSE. The deck translator maps the two keys separately |
| `interpolate_sources` | bool | (default `true`, LPSE `{laser\|raman}.interpolateSourcesInTime`) every light sub-step reads the EPW potential (between its values before and after the EPW step, which is advanced first) and the ion density at the sub-step's middle; `false`: the new values. The Laplacian of the potential is interpolated in x-space. The deck translator uses the evolved Raman light's flag (the pump's without it) and notes a difference |

### epw

| Field | Type | Description |
|-------|------|-------------|
| `boundary` | object | Boundary conditions |
| `damping` | object | Damping mechanisms |
| `density_gradient` | bool | Include density gradient effects |
| `linear` | bool | Linear mode (disables nonlinear coupling) |
| `source` | object | Source terms |
| `hyperviscosity` | object | Optional hyperviscosity for numerical stability |
| `kinetic real part` | bool | Include kinetic correction to real frequency |
| `max_wavenumber` | float | (optional) LPSE `lw.maxWavenumber`: hard cap `\|k\| < max_wavenumber * k0` (vacuum laser wavenumber) on the retained EPW band, applied on top of `grid.low_pass_filter` / `grid.dealias` |
| `solver` | string | (default `separate`) `separate`: the four envelope equations of the MATLAB prototype / LPSE's default spectral path (EPW potential, pump, Raman light each with its own carrier). `combined`: the original LPSE `lw.solver = combined` formulation, which LPSE *requires* whenever TPD and SRS are both on (its users guide: the separate equations "are not valid for simultaneous TPD and SRS", nor "for SRS coupled to IAWs near n_c/4"). One field `E1` enveloped at `wp0` carries the Raman light as its transverse part and the EPW as its longitudinal part; `epw` (the potential `phi_k = i k . E1_k/k^2`) is derived from it every step for the diagnostics, IAWs and HPE. Per light sub-step: the density detuning and collisional damping on the whole field, LPSE's unified source `-i e/(4 me w0) e^{-i(w0 - 2wp0)t} [grad(E0 . E1*) + (1 - w0/wp0) E0 (div E1)*]` (whose longitudinal part is the TPD + SRS source and whose transverse part is the Raman-light generation), the 2x2 longitudinal/transverse propagator (Bohm-Gross dispersion + Landau damping on `k k/k^2 E1`, `exp(-i dt c^2 k^2/(2 wp0))` on the rest), the EPW noise on the longitudinal part, the absorbers; with `terms.light.pump_depletion` the pump advances in the same sub-step with the unified depletion `i e/(2 me w0) e^{+i(w0 - 2wp0)t} E1 div E1` (SRS + TPD depletion in one term; `terms.light.tpd_projection` applies the transverse projection LPSE's combined path omits). Requires `terms.light.solver: spectral`, `tpd` and `srs` both on or both off, and excludes `drivers.E2`, `energy_ledger`, `terms.light.coupling/filter`. With `separate` and both instabilities on, setup raises (LPSE's refusal: 'Must use lw.solver=combined when both SRS and TPD are enabled'; the separate-equation runs went non-finite in the srs-2d-testbed). The Raman-light series (`e1_sq`, reflectivity, fluxes) are evaluated on the transverse part; the IAW ponderomotive drive and the thermal-filamentation heating see the whole combined field at its carrier `wp0` (Raman light and EPW with their cross term; LPSE `IawSolver.cpp:211-217`: `terms.iaw.drive.epw` and `raman` must agree and act together, `thermal_filamentation.lw` is ignored) |
| `source_window` | object | (optional) LPSE `lw.restrictSourceRange`: multiply the TPD / SRS sources (and the combined solver's unified source) in x-space by a window with a flat top of `width: [wx, wy]` about `center: [cx, cy]` (with units, measured from the box centre) and linear ramps of `edge_width` to zero outside; an axis with width `0um` or omitted is unrestricted (`ZakharovSolver::restrictRange`). Combined with `terms.light.suppress_sources_*` into `grid.epw_source_mask` |
| `energy_ledger` | bool | (default `false`) Accumulate, in the state, the change of the EPW energy attributed to every operation of the split step -- `dispersion`, `damping`, `dealias`, `noise`, `detuning`, `boundary`, `reprojection`, `tpd`, `srs`, `driver` -- and report the cumulative values in the default series as `epw_ledger_<channel>` (in `epw_energy` units) together with `epw_ledger_closure = epw_energy - sum(channels)`, which stays at `epw_energy(0)` to round-off. This is the per-step energy-budget check LPSE asserts to 0.1 % (`ZakharovSolver.cpp:1700-1712`); here it is exact by construction and the channels are the diagnostic |
| `interpolate_sources` | bool | (default `true`, LPSE `lw.interpolateSourcesInTime`) the EPW detuning reads the ion density at the middle of the EPW step, linearly between the IAW step's start and end values; `false`: the new (end) value. The deck translator passes the deck's value |

#### boundary

| Field | Type | Description |
|-------|------|-------------|
| `x` | string | `"periodic"` or `"absorbing"` |
| `y` | string | `"periodic"` or `"absorbing"` |

#### damping

| Field | Type | Description |
|-------|------|-------------|
| `collisions` | bool or float | Collisional damping. `true` computes from plasma parameters, or specify rate directly |
| `landau` | bool | Include Landau damping |
| `landau_form` | string | (default `matlab`) Static Landau rate. `matlab`: the prototype's `sqrt(pi/8) (1 + 1.5 x^2) wp^4/(k^3 vte^3) exp(-3/2 - 1/(2x^2))`, `x = k lambda_D`. `lpse`: the C++ `landauDamping_nonRel`, `sqrt(pi/8) (kde/k)^3 w_k exp(-w_k^2/(2 k^2 vte^2))` with `w_k = wp sqrt(1 + 3x^2)` -- identical exponent, prefactor `sqrt(1 + 3x^2)` vs `(1 + 1.5x^2)` (0.7% at x = 0.3, 3.8% at x = 0.5). `relativistic` (= `relativistic_2d`) / `relativistic_3d`: LPSE's Maxwell-Juettner Bessel-function rates (`landauDamping_rel_2D/3D`); modes with phase velocity above `c` are undamped |
| `landau_lower_threshold` | float | (default `0`, 1/ps) LPSE `lw.landauDamping.lowerThreshold`: modes whose Landau rate is below it are treated as undamped (also in the thermal-noise balance) |
| `landau_multiplier` | float | (default `1`) Static multiplier on the Landau rate (LPSE `LD_multiplier`) |

The same rate array (form, threshold, multiplier) is used by the EPW step, the HPE calibration, the dissipation diagnostic and the thermal noise source.

#### source

| Field | Type | Description |
|-------|------|-------------|
| `noise` | bool | Add random noise source |
| `noise_amplitude` | float | (optional) Amplitude of the per-step EPW noise source. Default `1.0e-10` (the MATLAB `noiseAmp`) |
| `noise_seed` | int | (optional) Seed for the EPW noise source. Default `null`, which draws a random seed once and pins it into the config before parameters are logged, so every run is exactly reproducible from its logged `noise_seed` |
| `noise_model` | string | (default `flat`) `flat`: the MATLAB source, a kick `dt * noise_amplitude` with a random phase on every retained mode each step. `thermal`: the original LPSE `lw.noise` source (`ZakharovSolver::addNoiseToPotential_fft`), a fluctuation-dissipation kick `D_k = N A / sqrt(1 + k^2 lambda_D^2) * sqrt(1 - exp(-2 gamma_k dt)) / \|k\|` balanced against the frozen analytic Landau + collisional rate `gamma_k` and added after the damping sub-step, so every mode relaxes to the Cerenkov spectrum `<\|E_k\|^2> = A^2/(1 + k^2 lambda_D^2)` (x-space envelope amplitude squared) independently of `dt`; `A = noise_amplitude`. Undamped modes receive no noise; a band with no damping at all is refused, as in LPSE. Switching an existing deck to `thermal` changes its seed level -- the srs-2d-testbed calibration was done with `flat` |
| `noise_calibrate` | bool or string | (default `false`; `thermal` only) Set `A` from the plasma instead of taking `noise_amplitude` literally; `noise_amplitude` then multiplies it, so `noise_amplitude: 1` is the calibrated level (LPSE `lw.noise.isCalculated` convention). `equipartition` (or `true`): electric energy `kT/2` per mode over the box volume `V` (`Lz = Ly` for the 2-D box, `V = Lx^3` when `ny = 1`, following LPSE's `deltaK3`): `A = noise_amplitude * sqrt(8 pi kT / V)`. `lpse`: LPSE's own constant `lw.noise.calcNoiseAmp_K0` (`ParameterManager.cpp:1440`, ported in `lpse_deck.ZakUnits.noise_amp_k0` and checked against the value LPSE prints for test_010) converted from its ZAK k-space potential to this code's, so an `isCalculated` deck seeds at LPSE's absolute level; the deck translator selects it. The two calibrations differ by a plasma- and grid-dependent factor -- LPSE's source notes its own derivation "doesn't all match" |
| `noise_max_wavenumber` | float | (optional, units of `k0`) LPSE `lw.noise.maxWavenumber`: no noise above `noise_max_wavenumber * k0` |
| `tpd_form` | string | (default `lpse`) `lpse`: the C++ TPD source `i e/(4 me w0) e^{-i(w0 - 2wp0)t} [F(E0 . E*) + (w0/wp0 - 1) i k . F(E0 rho*)/k^2]` with every pump component (`ZakharovSolver::updatePotentialWithTpdSource_fft`). `matlab`: the prototype's `w0 -> 2 wp0` form, `i e/(8 me wp0)` and factor 1. The two are identical at envelope density 0.25 (`w0 = 2 wp0`); at 0.23 the LPSE coefficient is 4% smaller and the charge-density factor is 1.085 |
| `srs_k_filter` | bool | (default `true`) apply a high-k filter to the light fields entering the SRS source, cutting at `srs_k_filter_scale` times the Raman wavenumber at the minimum box density (LPSE `lw.kFilter.enable`, which is *off* by default there; MATLAB `isSuppressHighKSource`). The cutoff assumes the scattered light sits exactly at the envelope frequency, so in a box detuned from the envelope density it can remove the resonant Raman mode entirely (no SRS growth at all at 0.2 nc with envelope density 0.25); translated LPSE decks turn it off |
| `srs_k_filter_scale` | float | (default `1.2`) multiplier on the SRS source k-filter cutoff (LPSE `lw.kFilter.scale`) |
| `tpd` | bool | Include the two-plasmon-decay source. With `terms.light.pump_depletion`, also include LPSE's pump-depletion term `i e/(2 me w0) e^{+i(w0 - 2wp0)t} [E_h div E_h]_T` on both pump components (twice the EPW-side coefficient, so the pair conserves the total wave energy at envelope density n_c/4; see `terms.light.tpd_projection`) |
| `srs` | bool | Include stimulated Raman scattering (optional, default false). Turning this on also evolves the Raman scattered-light field `E1` with a finite-difference paraxial solver, sub-cycled `grid.light_substeps` times per EPW step, and adds the SRS source `i e wp0/(4 me w0 w1) (n/n_env) E0 . conj(E1)` to the EPW potential. The default time series then also records `e1_sq` and `reflectivity` (Poynting-corrected `|E1_y|^2/E0^2` at a probe on the low-density side, `x = 1.6 * boundary_width`) |

#### hyperviscosity (optional)

| Field | Type | Description |
|-------|------|-------------|
| `coeff` | float | Hyperviscosity coefficient |
| `order` | int | Order of hyperviscosity (must be even) |

### iaw (optional)

The IAW state follows the MATLAB LPSE split update for fractional ion-density perturbation `Nelf` and ion-velocity divergence `W`. Its pressure combines the acoustic restoring term with ponderomotive drive from the EPW, pump, and Raman fields. `Nelf` feeds back into the EPW, pump, and Raman detuning terms on the next outer step.

| Field | Type | Description |
|-------|------|-------------|
| `active` | bool | Enable ion-acoustic evolution (default `false`) |
| `solver` | string | (default `explicit`) `explicit`: the MATLAB kick/drift split step (FD Laplacian, stable for `omega_iaw,max * grid.dt < 2`, validated at setup). `spectral`: the original LPSE `iaw.solver = spectral` path -- per k-mode the exact solution of the damped oscillator `dn/dt = -w, dw/dt = cs^2 k^2 n - 2 gamma_k w` (`e^{-gamma dt}` times a `cos/sin(beta dt)` rotation, `beta = sqrt(cs^2 k^2 - gamma^2)`), then collisional damping `e^{-2 nu dt}` on `w`, the ponderomotive kick `dt k^2 PP_k`, and the noise. Unconditionally stable and exact for the acoustic part at any `dt` `fd`: LPSE's finite-difference solver (`iaw.solver = fd`, `IawSolver.cpp`; plan 2 I.1): `n` and `div v` on a grid refined `super_samples` times are advected by the flow profile with a dimensionally split PPM sweep per axis (Colella-Woodward parabolas, van Leer limited slopes, conservative fluxes with the face velocities), then `n -= dt div v`, `div v += dt (-cs^2 lap n + lap drive)` with the second-order temporal corrections, the absorbing layer, the amplitude clamp, zero edge cells and a zero mean, sub-cycled at `dt_fraction h/(sqrt(nDim) cs + |U|)`; the Landau damping and the noise act on `div v` in k-space on the EPW grid's band. Both upwind states are formed at every cell (LPSE forms the one of each cell's own flow sign, which drops a flux at stagnation points). A travelling acoustic wave keeps its amplitude to 0.5 % and runs at `cs k + k . U` to 0.3 % (its grid dispersion) against the exact spectral solver, with and without flow, in 1-D and 2-D (`test_iaw_fd.py`); two counter-propagating equal-frequency beams on a Mach -2 -> +2 ramp drive the IAW at the Mach -+1 layers only (CBET resonance localisation). Needs `dx == dy` |
| `boundary` | object or null | Per-axis `x`/`y` boundary modes. Defaults to `terms.epw.boundary` |
| `boundary_max_rate` | float or null | (`grid.boundary_profile: exp` only) Peak absorber rate for the IAW layer, 1/ps; default half of `grid.boundary_max_rate` (LPSE `IawSolver` default 100 vs 200) |
| `boundary_width` | string or null | (`exp` profile) width of the IAW layer (LPSE `iaw.Labc`, default 0 there: no layer; the deck translator passes it); default `grid.boundary_width`. With `0um` the fd solver also keeps its edge cells (LPSE `zeroOutEdgeCells` acts only with a layer) |
| `damping.collisions` | float | Collisional damping rate in `1/ps` (default `1.0e-5`): on `n` as `(1 - nu dt)` in the explicit solver, on `div v` as `exp(-2 nu dt)` in the spectral one (LPSE) |
| `damping.landau` | float | Dimensionless coefficient in `gamma_iaw(k) = landau * cs * |k|` (default `0.1`); the velocity-divergence equation is damped at `2 gamma_iaw` |
| `damping.landau_form` | string | (default `simplified`) `simplified`: the rate above (LPSE `isSimplified`). `full`: the Z-generalized Krall-Trivelpiece expression used by LPSE's IAW solver, `W_i = W_r sqrt(pi/8) L^{-3/2} [(3/(eta-1))^{3/2} exp(-(3/(eta-1))/(2L)) + sqrt(1/(eta M))]`, `W_r = cs |k|/sqrt(L)`, `L = 1 + k^2 lambda_D^2`, `eta = 1 + 3 Ti/(Z Te)`, `M = (mi/me)/(eta Z)`, with `gamma_iaw = W_i/2`; `damping.landau` is then ignored. The real frequency in the spectral propagator stays `cs |k|`, as in LPSE's spectral branch |
| `feedback` | string | (default `local`) how the IAW density enters the EPW and light detunings and the collisional-absorption density. `iaw_density` is the local fraction `delta n / n_b` (the drive is density-independent), so the waves see `n_b (1 + iaw_density)`, i.e. `iaw_density * n_b / n_env` in units of the envelope density (`local`, LPSE's `Nelf * backgroundDensity`). `envelope` adds `iaw_density` to `n_b / n_env` directly, as the MATLAB prototype does -- exact only where `n_b = n_env`, and `n_env / n_b` too strong elsewhere (2.5x on test_001's 0.1 n_c CBET box; before 2026-09-22 this was the only form) |
| `perturbs` | object | (default all `true`) which waves see the IAW density, `{epw, pump, raman}` (LPSE `{lw|laser|raman}.ionAcousticPerturbations.enable`, all `false` by default there; the deck translator writes the deck's values): a wave switched off ignores it in its detuning and collisional absorption. With the combined solver `epw` and `raman` must agree, as LPSE requires |
| `drive` | object | (default all `true`) which fields drive the IAW, `{epw, pump, raman}` (LPSE `iaw.sourceTerm.{lw|laser|raman}.enable`, all `false` by default there): each switches its term of the ponderomotive drive; with the combined solver `epw` and `raman` must agree |
| `max_density_perturbation` | float or null | Optional symmetric limiter on `|delta n_i/n_0|`; unlimited by default |
| `flow` | list, object or null | A uniform background flow `[Mach_x, Mach_y]` in units of `cs` (`spectral`: the Doppler phase `e^{-i k . V0 dt}`, LPSE `fluid.velocity`; `fd`: advection), or -- `fd` solver only -- a flow **profile** (LPSE `iaw.velocityProfile.*`, plan 2 I.2): `{shape: linear | gaussian | log | file, from_location: [x, y], to_location: [x, y] (um from the box centre), from_mach, to_mach, sg_order, geometry: cartesian | spherical, temporal_slope (1/ps, U(t) = U (1 + slope t)), file}`. `linear`: the speed varies linearly with the projected distance along from -> to between the two Mach numbers and is clipped outside; `gaussian`: super-Gaussian of order `sg_order` between them; `log`: `from_mach + to_mach ln((r - x0)/x1)` radially (spherical); `file`: an `.npz` with `ux`, `uy` in Mach on the EPW grid. Cartesian flow is along from -> to, spherical flow radial from `from_location` |
| `super_samples` | int | (default `2`; `fd` solver) LPSE `iaw.fd.superSamples`: the fd fields live on the EPW grid refined this many times per axis (state keys `iaw_density_fine` / `iaw_velocity_divergence_fine`); the coarse `iaw_density` the light and EPW see is their band-limited restriction |
| `dt_fraction` | float | (default `0.95`; `fd`) LPSE `iaw.fd.dtFraction`: the fd sub-step is this fraction of `h / (sqrt(nDim) cs + |U|_max)`, with `h` the fine cell |
| `landau_update` | int | (default `1`; `fd`) LPSE `iaw.fd.numStepsPerLandauDampingUpdate`: the k-space Landau damping (and noise) of `div v` every this many sub-steps |
| `temporal_correction` | bool | (default `true`; `fd`) LPSE's second-order temporal correction of the split sources, `+ dt^2/2 U . grad(div v)` on `n` and `- dt^2/2 U . grad S` on `div v` |
| `stride` | int | (default `1`; `spectral` or `fd`, not `explicit`) Advance the IAW every `stride` EPW steps with `dt_iaw = stride * grid.dt` (LPSE steps its IAW solver less often than the EPW; the deck translator writes LPSE's ratio). The IAW absorber acts over the IAW step (`exp(-rate dt_iaw)`; the fd solver per sub-step, `exp(-rate dt_sub)`), as LPSE; between IAW steps the waves read the density interpolated across it (`iaw_density_old` in the state) |
| `source_window` | object | (optional) LPSE `iaw.restrictSourceRange`, the same window as `terms.epw.source_window`; it multiplies the ponderomotive drive **squared** (LPSE `getPonderomotivePotential`) |
| `t_start`, `t_stop` | float | (optional, ps) the IAW step acts only for `t_start <= t < t_stop` (LPSE `iaw.startEvolvingTime` / `stopEvolvingTime`) |
| `noise` | bool | (default `false`) LPSE `iaw.noise`: a random-phase source on `div v` every IAW step with the fluctuation-dissipation amplitude `noise_amplitude * nx * ny * sqrt(exp(2 dt (gamma_k + nu)) - 1)` on the retained band; requires IAW damping |
| `noise_amplitude` | float | (default `1.0`) Amplitude of the IAW noise (x-space `div v` units per mode) |
| `noise_seed` | int or null | Seed of the IAW noise stream |

The setup validates the explicit acoustic stability condition `omega_iaw,max * grid.dt < 2` for `solver: explicit`.

```yaml
terms:
  iaw:
    active: true
    boundary: null
    damping:
      collisions: 1.0e-5
      landau: 0.1
    max_density_perturbation: 0.1
```

#### thermal_filamentation (optional, LPSE `thermalFil.*`)

`terms.iaw.thermal_filamentation: {laser: bool, raman: bool, lw: bool, nonlocal: bool, conductivity_multiplier: 1.0}` adds the thermal-filamentation source to the ion velocity-divergence equation: inverse-bremsstrahlung heating by the spatially varying part of each enabled wave's intensity (`|E_w|^2 - <|E_w|^2>`), balanced by Spitzer heat conduction, drives the flow through the electron pressure, `d(div v)/dt += Z nu_w(n) (|E_w|^2 - <|E_w|^2>) / (8 pi m_i kappa')` with `kappa'` the Spitzer conductivity over k_B (times `conductivity_multiplier`) and `nu_w` the wave's energy damping rate (light: `2 nu_abs(n_c) (n/n_c)^2`, requires `terms.light.absorption`; EPW: `2 nu_coll n/n_env`, requires `terms.epw.damping.collisions`). `nonlocal: true` adds LPSE's `k^(4/3)` correction (`1 + (k lambda_nl)^(4/3)`, `lambda_nl = 30 (k_B T_e)^2 / (4 pi e^4 sqrt(Z+1) ln Lambda n_e)`). The source form follows `ZakharovSolver::getThermalFilamentationSource`; the normalization is adept's own.

### hpe (optional)

Hybrid particle evolution, following Follett et al., *Phys. Plasmas* **24**, 102134 (2017): test electrons drawn from the Maxwellian tail are pushed relativistically in the de-enveloped electrostatic field, their spatially averaged velocity distribution is accumulated by exponential moving average, and the Landau damping rate applied by the EPW solver is recomputed from that evolving distribution every step (kinetic inflation + hot-electron generation; Im-only feedback, no nonlinear frequency shift). For `ny == 1` the tracker uses `(x, p_x)`; in a 2-D box it uses `(x, y, p_x, p_y)` and gathers both $E_x$ and $E_y$. In both cases there is one box-wide ensemble, not a particle population at each grid point. HPE requires `terms.epw.damping.landau: true`. The particle push dominates runtime, so HPE runs want a GPU.

In 2-D the box-wide distribution is represented by `n_angles` oriented projections $f(\mathbf{v}\cdot\hat{\mathbf{k}})$ over $[0,2\pi)$. Opposite directions remain distinct, and the two neighboring angular histograms are interpolated for every `(kx, ky)` mode. This is the Radon-projection form of the resonance integral: its memory is `n_angles * nv`, independent of the spatial mesh. The damping extraction is calibrated per k-mode so that a freshly loaded isotropic Maxwellian tail reproduces the analytic Landau rate exactly; modes whose phase velocity lies below the tail cutoff keep the analytic rate. The default time series gains `fhot_50keV`, `fhot_100keV`, `hpe_mean_energy_keV`, `hpe_gamma_ratio_kpeak` (applied-to-analytic damping ratio at the resonant-band mode carrying the most EPW energy), `hpe_gamma_ratio_min` (band minimum; shot-noise-limited at low `n_particles`), and `hpe_hist` (one velocity histogram in 1-D or an angle-by-velocity array in 2-D); MLflow metrics gain `fhot_50keV`, `t_first_hot_e_50keV`, and `hpe_damping_reduction_final`.

At an absorbing particle wall, outgoing particles are thermalized and reinjected from the flux-weighted retained-tail law. In 2-D wall coordinates this is $p(r,\theta) \propto r^2\exp[-r^2/(2v_{te}^2)]\cos\theta$ for $r > v_{\min}$ and inward $-\pi/2 < \theta < \pi/2$. This distinction is required to keep repeated wall crossings from biasing the global directional distribution.

| Field | Type | Description |
|-------|------|-------------|
| `active` | bool | Enable HPE (default `false`) |
| `n_particles` | int | Number of tail test particles (default `500000`) |
| `v_min` | float | Tail cutoff in units of `vte` (default `2.5`); the retained fraction is `erfc(v_min/sqrt(2))` in 1D1V and `exp(-v_min^2/2)` for the radial 2D2V tail. The deck translator sets LPSE's `hpe.VminOverVminPhase * wpe / (2 pi / h)` in `vte`, `0` (the whole Maxwellian) by default |
| `v_max` | float | Histogram half-span in units of `c` (default `1.0`) |
| `nv` | int | Velocity bins spanning `(-v_max, v_max)` (default `512`) |
| `n_angles` | int | Number of oriented global velocity projections spanning $2\pi$ in 2-D (default `32`; ignored for `ny == 1`) |
| `v_blend_buffer` | float | Buffer above `v_min` (units of `vte`) below which modes keep the analytic rate (default `0.5`) |
| `gather_refine` | int | Spectral upsampling factor for `Ex` and `Ey` before the particle gather (default `4`). Linear interpolation of a wave with `k dx ~ 1-2` rad/cell attenuates the gathered field by `sinc^2(k dx / 2)` (15-30%); upsampling makes this ~1% |
| `substep_courant` | float | `wp0 * dt_particle` for the sub-cycled push (default `0.05`; Follett used 0.035) |
| `tau_damping` | string | EMA time constant for the velocity histogram (default `"100fs"`, Follett's update interval) |
| `t_start` | string | Push/feedback disabled before this time (default `"0ps"`); use to let the fluid run reach steady state first |
| `feedback` | bool | (default `true`) `false` = control run: particles evolve but the damping stays analytic (Follett's control experiment). The deck translator sets it from LPSE `hpe.landauDampingEvolution.enable`, default `false` there (the particles are then diagnostics) |
| `seed` | int | RNG seed for particle loading and wall re-injection (default `42`) |
| `omega_res` | string | Resonance convention for `v_phi(k)`: `"bohm_gross"` (default, matches the analytic rate) or `"wp0"` (bare carrier, as in the paper) |
| `gamma_limit_damping` | float | (default `1500`, 1/ps) upper clip on the applied kinetic rate (LPSE `hpe.gammaLimit.damping`, a ZAK rate with default 1e6 -- no limit; the deck translator converts it to 1/ps) |
| `gamma_limit_growth` | float | (default `1500`, 1/ps) lower clip `-gamma_limit_growth` on the applied rate when `allow_growth` is on (LPSE `hpe.gammaLimit.growth`) |
| `allow_growth` | bool | (default `false`) allow negative (inverse-Landau) rates from an inverted tail; otherwise the rate is clipped at 0 (LPSE `hpe.allowGrowth`) |
| `thermalization_probability` | list | probability that a particle crossing an x / y side is thermalized and re-injected from the tail; otherwise it passes through periodically. Every crossing is counted by the wall-flux instrument, on all four sides, independently of the field boundaries (LPSE particle walls, `hpe.thermalizationProbability`). Default: 1 at absorbing field boundaries, 0 at periodic ones. The deck translator writes LPSE's default `(1, 0)` when the deck omits `hpe.thermalizationProbability` |
| `magnetic_field` | float or list | (default `0`, tesla) uniform B (LPSE `hpe.magneticField`), 2-D push only. A number is the out-of-plane `B_z`: the push rotates the in-plane momentum by the cyclotron angle each sub-step. `[B_x, B_y, B_z]` adds in-plane components: the tracker then carries a third momentum component `p_z` (starting at 0, re-injected at 0 by the thermalising walls) and rotates the 3-momentum about `B` by `|omega_c| dt / gamma` (the exact Boris rotation), so an in-plane B turns in-plane momentum into `p_z` (Larmor rotation of the wall fluxes). The EPW force stays in-plane; the projected histograms use the in-plane velocity |
| `energy_conservation` | bool | (default `false`) scale the applied Landau rate by a global multiplier `M` in [0.1, 10] so that the expected EPW energy loss over the step equals the particles' kinetic-energy gain (LPSE `hpe.enforceEnergyConservation`). The port follows `ZakharovSolver::getLD_multiplierForEnergyConservation` / `getExpectedLwEnergyChange`: the expected loss is `sum_k k^2 \|phi_k\|^2 (1 - exp(-2 dt M gamma_k))` over the retained modes with the HPE-derived `gamma_k` (the exact per-mode factor, not its linearisation `2 dt M gamma_k`, which over-counts the loss of strongly damped modes), `M` is solved by secant iteration from the proportional guess to 1 % of the gain, the gain is the running average of the per-step particle energy change over `energy_conservation_steps` steps (LPSE `particleEnergyChange_TW`), and the multiplier is only re-solved once more than `energy_conservation_steps` feedback steps have elapsed since `t_start` (LPSE `countParticleUpdateSteps`). Reported as `hpe_ld_multiplier`; the averaged gain as `hpe_particle_power` (keV per real electron per step) |
| `energy_conservation_steps` | float | (default `1`; the deck translator uses LPSE's default `10` when `hpe.numStepsToAverageEnergyChange` is absent) running-average length of both the particle gain and the multiplier (LPSE `hpe.numStepsToAverageEnergyChange`) |
| `flux_bins` | list | keV edges of the wall-flux instrument (default `[0, 50, 100, 1e9]`): the series `hpe_wall_energy_<left|right|bottom|top>_bin<i>` accumulate the energy (keV per real electron) leaving through each wall in each bin (LPSE `hpe.metrics.flux`) |
| `cone_angle` | float | acceptance half-angle in degrees of the cone-power instrument (`hpe_cone_energy`: cumulative energy leaving inside the cone about `cone_direction`; LPSE `hpe.metrics.power`). `null` = off |
| `cone_direction` | list | (default `[1, 0]`) axis of the acceptance cone |

```yaml
terms:
  hpe:
    active: true
    n_particles: 500000
    v_min: 2.5
    substep_courant: 0.05
    tau_damping: 100fs
    t_start: 2ps
```

## Light-field components

The pump `E0` and the Raman (or combined) field `E1` carry three components `(x, y, z)` on the 2-D grid, as LPSE's `XcComplex3` fields do on any grid. With `k_z = 0` the z component is purely transverse: every longitudinal/transverse projector acts on the in-plane pair and passes `E_z` through, the FD propagator advances it with the plain Laplacian (`-(curl curl E)_z = laplacian E_z`), it never enters the EPW potential `phi_k = i k . E_k / k^2`, and it never drives TPD (the EPW field is in-plane). It does drive SRS through `E0 . E1*`, so an s-polarised pump (`drivers.E0.polarization: 90`) scatters into an s-polarised Raman wave and drives no TPD. The fields output gains `e0_z` and `e1_z`; `reflectivity` and the flux series sum over every transverse component. A two-component checkpoint from before this change is padded with `E_z = 0` on restart. With `E_z = 0` throughout, every solver path reproduces the two-component code bit for bit (tests/test_lpse2d/test_three_component_fields.py).

## Diagnostics: Poynting flux, Thomson probes and light spectrum probes

`save.fields.poynting: true` adds the light energy-flux density maps `s0_x`, `s0_y` (pump) and `s1_x`, `s1_y` (Raman) to the fields output, `S_j = (c^2/omega) Im(E* . d_j E)` in field-squared times um/ps (`v_g |E|^2` for a plane wave; LPSE `laser.save.S0` / `raman.save.S0`). `save.thomson: [{k: [kx, ky], bandwidth: 0.1, field: epw}]` adds synthetic Thomson-scattering probes to the default series (LPSE `thomsonScattering.N.wavevector.lw/.iaw` and `bandwidth`): for probe `i`, `thomson_i_re` / `thomson_i_im` are the summed complex amplitude of the EPW potential (or, with `field: iaw`, of the IAW density) over the k-window `|k - k_probe| < bandwidth k0`, and `thomson_i_power` the summed spectral power there; `k` and `bandwidth` are in units of the vacuum laser wavenumber.

`save.light_spectrum: [{field: E0, interval: 0.05ps, tmin: 0ps, x: [-46.9um, -46.7um], y: [-10um, 10um], poynting: false}]` adds light spectrum probes (LPSE `spectrum.N.laser` / `spectrum.N.raman`: `startTime`, `interval`, `location.min/.max`, `file.E0.*`, `file.S0.*`): probe `i` records the field `E0` (pump, carrier `w0`) or `E1` (Raman light, carrier `w1`) on the sub-box (coordinates from the box centre, nearest grid nodes; an omitted axis is the whole box) every `interval`, and post-processing writes `binary/light_spectrum_i.xr` with the components `e_x, e_y, e_z` against time, their temporal spectra `spectrum_x, ...` against `delta omega (w_carrier)` (the envelope frequency offset, positive above the carrier), the box-summed `power` (plotted in `plots/light_spectrum_i.png`) and, with `poynting: true`, the Poynting components `s_x, s_y` (one guard cell is kept for the differences and cropped afterwards). The translator maps every enabled LPSE probe that names an output file.

## Absolute-threshold search

`adept._lpse2d.threshold.find_threshold` bisects the pump intensity between a stable and an unstable bracket on the fitted EPW growth rate. `find_threshold_lpse` is LPSE's own `absoluteThreshold` search (`AbsoluteThreshold.cpp`): from the configuration's intensity it steps by `dI_fract` of it away from the first run's side until `ln(A_end / A_noise) > gain` flips (`A` the peak EPW amplitude `max_phi`, `A_noise` its mean before `noise_time_range[1]`), then `n_iter` halving steps towards the flip. The deck translator writes `absoluteThreshold.{gain, dI_fract, numIterations, noiseTimeRange}` into a `threshold` block that `find_threshold_lpse` reads (test_016); the defaults are LPSE's (`gain` 23.026, `dI_fract` 0.3333, 5 iterations, 0.01-0.1 ps).

## Initial perturbation (`initial_perturbation`)

LPSE `initialPerturbation` (`InitialPerturbation.cpp`): a plane wave `A env(r) exp(i K . r)` written into one field at `t = 0`, for growth-rate tests that do not depend on a noise seed.

| Field | Type | Description |
|-------|------|-------------|
| `field` | string | (default `epw`) `epw` seeds the EPW potential (k-space state; with `terms.epw.solver: combined` also the longitudinal part of `E1`, `-grad phi`); `E0` / `E1` seed one light-field `component` |
| `component` | string | (default `y`) `x`, `y` or `z` for a light field |
| `amplitude` | float | (default `1`) in LPSE's normalized output units: `e phi / (m_e c^2)` for the potential, `e E / (m_e w0 c)` for a light field (the translator passes `initialPerturbation.amplitude` through) |
| `wavelength` | string | perturbation wavelength with unit; `K = 2 pi / wavelength` along `direction` |
| `direction` | list | (default `[1, 0]`) in-plane direction of `K` (normalised) |
| `envelope_size` | list | (optional) full width at 1/e per axis, with units; omit or `0um` for no envelope along that axis |
| `envelope_offset` | list | (optional) envelope centre per axis relative to the box centre, with units |
| `envelope_sg_order` | float | (default `4`) super-Gaussian order of `exp(-\|(r - offset) / (size/2)\|^order)` |

Coordinates are measured from the box centre, as in LPSE. A seeded EPW mode evolves freely at exactly the solver's analytic Landau rate for that `k` (`tests/test_lpse2d/test_initial_perturbation.py`).

## Checkpoint and restart

`save.checkpoint: true` (or a path) writes the complete solver state at `grid.tmax` as an `.npz` (`binary/checkpoint.npz` in the run's artifacts when `true`); `restart: {file: <that .npz>}` in a later configuration resumes from it: the state replaces the fresh initial condition, integration starts at the checkpoint time, and the series/fields save axes start there too unless their `tmin` is later. The per-step noise and wall keys are folded in from the time index, so a run split into two resumes reproduces the unbroken run to round-off as long as the split time is a multiple of `grid.dt` (the fixed-step solver would otherwise clip one step differently). The configuration must otherwise be the same (grid, terms, drivers); mismatched state shapes are refused, except that a two-component `E0`/`E1` checkpoint is padded with `E_z = 0`.

## Absolute-threshold bisection (`adept._lpse2d.threshold`)

`find_threshold(cfg, intensity_lo, intensity_hi, n_iter=6)` bisects the pump intensity (geometric midpoints) between a stable and an unstable bracket, running the configuration once per point with `ergoExo` and deciding "unstable" from the run's metrics (default: a measurable EPW energy growth fit with a positive rate, `growth_min` adjustable; any callable of the metrics dict can be passed as `criterion`). The bracket endpoints are run first and must straddle the criterion, as in LPSE's `AbsoluteThreshold`. The result holds the threshold (midpoint of the final bracket), the bracket and the per-run history; the bisection is logged as an MLflow run (`<run>-bisection`) next to the individual runs.

### Example: TPD Simulation

```yaml
terms:
  epw:
    boundary:
      x: absorbing
      y: periodic
    damping:
      collisions: 1.0
      landau: true
    density_gradient: true
    linear: true
    source:
      noise: true
      tpd: false
      srs: true
  zero_mask: true
```

### Example: Simple EPW Test

```yaml
terms:
  epw:
    boundary:
      x: periodic
      y: periodic
    damping:
      collisions: false
      landau: false
    density_gradient: false
    linear: True
    source:
      noise: false
      tpd: false
  zero_mask: false
```

## Complete Example

```yaml
solver: envelope-2d

units:
  atomic number: 40
  envelope density: 0.25
  ionization state: 6
  laser intensity: 1.5e+14W/cm^2
  laser_wavelength: 351nm
  reference electron temperature: 2000.0eV
  reference ion temperature: 1000eV

density:
  basis: linear
  gradient scale length: 50um
  max: 0.28
  min: 0.18
  noise:
    max: 1.0e-09
    min: 1.0e-10
    type: uniform

grid:
  boundary_abs_coeff: 1.0e4
  boundary_width: 1.5um
  low_pass_filter: 0.66
  dt: 0.010fs
  dx: 40nm
  tmax: 2ps
  tmin: 0.0ns
  ymax: 0.08um
  ymin: -0.08um

mlflow:
  experiment: tpd
  run: my-simulation

save:
  fields:
    t:
      dt: 0.2ps
      tmax: 2ps
      tmin: 0ps
    x:
      dx: 50nm
    y:
      dy: 50nm

drivers:
  E0:
    delta_omega_max: 0.015
    envelope:
      tc: 200.25ps
      tr: 0.1ps
      tw: 400ps
      xc: 50um
      xr: 0.2um
      xw: 1000um
      yc: 50um
      yr: 0.2um
      yw: 1000um
    num_colors: 1
    shape: uniform

terms:
  epw:
    boundary:
      x: absorbing
      y: periodic
    damping:
      collisions: 1.0
      landau: true
    density_gradient: true
    linear: true
    source:
      noise: true
      tpd: false
      srs: true
  zero_mask: true
```

## Example Configurations

### EPW Linear Propagation

See `configs/envelope-2d/epw.yaml` - Simple EPW test without instabilities.

### Landau Damping

See `configs/envelope-2d/damping.yaml` - EPW with Landau damping and trapping model.

### Two-Plasmon Decay

See `configs/envelope-2d/tpd.yaml` - TPD simulation with linear density gradient.

### SRS / Reflection

See `configs/envelope-2d/srs.yaml` - Noise-seeded backward SRS on a linear density ramp
(the lpse-matlab `srs_1D` case), with a reflectivity time series recorded at a probe on
the low-density side. Also see `configs/envelope-2d/reflection.yaml` - SRS simulation
with kinetic corrections.

### Coupled TPD + SRS + IAW

See `configs/envelope-2d/tpd-srs-iaw.yaml` for the full 2-D model: both parametric
instabilities, reciprocal pump depletion, ion-acoustic evolution, 2D2V particle
feedback, and shifted-band de-aliasing. See `configs/envelope-2d/srs-hpe.yaml` for
the less expensive quasi-1D particle-feedback model.
