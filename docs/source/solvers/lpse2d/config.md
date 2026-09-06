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
| `ionization state` | int | Ionization state Z |
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
| `boundary_abs_coeff` | float | Absorbing boundary coefficient |
| `boundary_width` | string | Width of absorbing boundary layer with unit |
| `low_pass_filter` | float | Low-pass filter cutoff as fraction of kmax (0-1) |
| `dealias` | string | Shape of the anti-aliasing mask: `isotropic` (default) or `shifted-band` |
| `dt` | string | Timestep with unit |
| `dx` | string | Spatial resolution with unit |
| `tmax` | string | End time with unit |
| `tmin` | string | Start time with unit |
| `ymax` | string | Domain maximum y with unit |
| `ymin` | string | Domain minimum y with unit |
| `light_substeps` | int | (SRS only, optional) Raman-light sub-steps per EPW step. Computed from the explicit-scheme stability limit `dt_light < 1 / (2c^2/(dx^2 w1) + \|w1^2 - wpe_max^2\|/(4 w1))` if omitted (with `terms.light.pump_depletion` the limit is the tighter of the `w0` and `w1` carriers); a `ValueError` is raised if a user-supplied value violates it |
| `probe_offset` | string | (SRS only, optional) Distance of the laser-budget flux probes from each box edge, with unit. Default `2 * boundary_width`, which is clear of the absorber's tanh skirt (the legacy `reflectivity` probe at `1.6 * boundary_width` sits inside it and reads ~10% low) |

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
| `shape` | string | Amplitude shape: `"uniform"` (optional) |
| `offset` | string | (pump depletion only, optional) Distance of the pump boundary injector from `xmin`, with unit. Default `2 * boundary_width` |
| `turn_on_time` | string | (pump depletion only, optional) Gaussian turn-on time of the injector. Default `10fs` |

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
| `turn_on_time` | string | Ramp-up time of the injector (default `10fs`) |
| `offset` | string | Distance of the injector from the right boundary. Defaults to `1.6 * boundary_width`, just inside the absorbing boundary's tanh skirt; a warning is printed for smaller values because the seed would be damped at the source |
| `yw` | string | Super-Gaussian (4th order) width of the seed in y; omit for uniform in y |

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

| Field | Type | Description |
|-------|------|-------------|
| `epw` | object | Electron plasma wave configuration |
| `light` | object | (optional) Light-wave evolution configuration |
| `hpe` | object | (optional) Hybrid particle evolution: test-particle Landau damping feedback (Follett et al. 2017) |
| `zero_mask` | bool | Whether to zero out k=0 mode |

### light (optional)

| Field | Type | Description |
|-------|------|-------------|
| `pump_depletion` | bool | (default `false`) Evolve the pump `E0` with the same staggered FD envelope solver as the Raman light instead of prescribing it analytically. The pump is launched by a two-point boundary injector at `x = xmin + drivers.E0.offset` and is depleted by the EPW through `-i e/(4 w1 me) (laplacian phi) E1`. Requires `terms.epw.source.srs: true` and `terms.epw.boundary.x: absorbing`; incompatible with `drivers.E0.speckle`. Enables the true net-flux `laser_reflectivity` / `laser_transmissivity` / `laser_absorbed_frac` metrics and lets above-threshold runs saturate |
| `coupling` | str | (default `explicit`; `pump_depletion` only) How the EPW coupling between `E0` and `E1` is integrated inside each light sub-step. `explicit` keeps the MATLAB staggered real/imaginary update, which treats the part of the coupling proportional to `Im(laplacian phi)` with an explicit Euler step: the light pair grows by `1 + sin^2(arg laplacian phi) (Omega dt_l)^2 / 2` per sub-step, `Omega = e \|laplacian phi\| / (4 me sqrt(w0 w1))`, for any `dt_l`. That is invisible at small EPW amplitude but manufactures light (and, through the SRS/TPD sources, EPW) energy once the pump is depleted and `Omega dt_l` reaches ~0.005, ending in a one-step NaN. `rotation` Strang-splits each sub-step as [exact coupling rotation over `dt_l/2`] [staggered propagation with the coupling off] [rotation over `dt_l/2`]; the rotation `exp(tau M) = cos(Omega tau) I + sin(Omega tau)/Omega M` conserves the light action `w1 \|E0\|^2 + w0 \|E1\|^2` pointwise and is stable for any `dt_l`. See `tests/test_lpse2d/test_light_coupling.py` |
| `tpd_depletion` | bool | (default `false`; `pump_depletion` only, requires `terms.epw.source.tpd`) Also deplete the evolved pump by the TPD source. The MATLAB original (and the default) depletes the pump only through the SRS term, so a TPD-driven EPW draws on an undepleted pump and its energy is not bounded by the laser energy. This adds `i e/(4 wp0 me) exp(i (w0 - 2 wp0) t) (div E) E_y` to `dE0_y/dt`, the exact energy partner of the EPW TPD source (the pair conserves `Int(\|E0\|^2 + \|grad phi\|^2)`; see `tests/test_lpse2d/test_light_coupling.py`) |
| `filter` | float | (default off; `pump_depletion` only) Isotropic low-pass filter applied to both light fields once per EPW step, keeping `\|k\| <= filter * pi/dx`. The physical light content lies below ~1.2 k0; grid-scale light modes have FD group velocity `c^2 sin(k dx)/(w dx) -> 0`. Diagnostic/numerical-hygiene option |

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

#### source

| Field | Type | Description |
|-------|------|-------------|
| `noise` | bool | Add random noise source |
| `noise_amplitude` | float | (optional) Amplitude of the per-step EPW noise source. Default `1.0e-10` (the MATLAB `noiseAmp`) |
| `noise_seed` | int | (optional) Seed for the EPW noise source. Default `null`, which draws a random seed once and pins it into the config before parameters are logged, so every run is exactly reproducible from its logged `noise_seed` |
| `tpd` | bool | Include two-plasmon decay source |
| `srs` | bool | Include stimulated Raman scattering (optional, default false). Turning this on also evolves the Raman scattered-light field `E1` with a finite-difference paraxial solver, sub-cycled `grid.light_substeps` times per EPW step, and adds the SRS source `i e wp0/(4 me w0 w1) (n/n_env) E0 . conj(E1)` to the EPW potential. The default time series then also records `e1_sq` and `reflectivity` (Poynting-corrected `|E1_y|^2/E0^2` at a probe on the low-density side, `x = 1.6 * boundary_width`) |

#### hyperviscosity (optional)

| Field | Type | Description |
|-------|------|-------------|
| `coeff` | float | Hyperviscosity coefficient |
| `order` | int | Order of hyperviscosity (must be even) |

### hpe (optional)

Hybrid particle evolution, following Follett et al., *Phys. Plasmas* **24**, 102134 (2017): test electrons drawn from the Maxwellian tail are pushed relativistically in the de-enveloped electrostatic field $\tilde{\mathbf{E}} = \mathrm{Re}[\mathbf{E} e^{-i\omega_{p0}t}]$, their spatially-averaged velocity distribution is accumulated by exponential moving average, and the Landau damping rate applied by the EPW solver is recomputed from that evolving distribution every step (kinetic inflation + hot-electron generation; Im-only feedback, no nonlinear frequency shift). Requires `terms.epw.damping.landau: true`. The particle push dominates the runtime, so HPE runs want a GPU.

**Geometry.** Runs in quasi-1D (`ny == 1`, push uses $E_x$ only, particles carry $u_x$) and in 2D (`ny > 1`, push uses $(E_x, E_y)$, particles carry $(u_x, u_y)$). Follett's Eq. 4 integrates $\partial \langle F\rangle/\partial v$ over the resonance surface $\omega = \mathbf{k}\cdot\mathbf{v}$, which is a point in 1D but a *line* in 2D. The line integral is exactly the 1D formula applied to the projection of $f$ onto the mode's own direction,

$$P_{\hat k}(v) = \int d^2v'\, f(\mathbf{v}')\,\delta(\mathbf{v}'\cdot\hat k - v), \qquad \gamma_L(\mathbf{k}) = -\frac{\pi}{2}\frac{\omega_{p0}^3}{|k|^2} \left.\frac{\partial P_{\hat k}}{\partial v}\right|_{v = \omega_\mathrm{res}/|k|}$$

so the 2D extraction bins the particles along `n_angles` directions spanning $[0,\pi)$ and interpolates between the two bracketing directions for each mode. Opposite half-plane directions are mirror images of each other, which is what the 1D $\mathrm{sgn}(k_x)$ already encoded; setting `ny == 1` forces `n_angles = 1` and every expression collapses to the original quasi-1D code. Because each projection consumes *every* particle, per-angle statistics match the 1D case at the same `n_particles` — 2D needs no particle-count increase, only more arithmetic.

The damping extraction is calibrated per k-mode so that the freshly loaded Maxwellian tail reproduces the analytic Landau rate exactly; modes whose phase velocity lies below the tail cutoff keep the analytic rate. The default time series gains `fhot_50keV`, `fhot_100keV`, `hpe_mean_energy_keV`, `hpe_gamma_ratio_kpeak` (applied-to-analytic damping ratio at the resonant-band mode carrying the most EPW energy), `hpe_gamma_ratio_min` (band minimum; shot-noise-limited at low `n_particles`), and the tail histogram `hpe_hist` — shape `(nt, nv)` in quasi-1D and `(nt, n_angles, nv)` in 2D. MLflow metrics gain `fhot_50keV`, `t_first_hot_e_50keV`, and `hpe_damping_reduction_final`, named for one-to-one comparison with the OSIRIS scan2 runs.

**Shot-noise clamping (why `hist_smooth` defaults on in 2D).** The rate is read from `df/dv` at a *single* velocity bin per mode. With a finite particle count some modes draw a slope steep enough that the `gamma >= 0` clamp sends them to exactly zero — and a mode pinned at zero is **undamped**, so it grows relative to the rest of the band instead of the error averaging away. The clamp selects for its own errors. A 2D box carries ~10x the resonant-band modes of a quasi-1D one (6644 vs 668 in the shipped smoke configs), so it meets this far more often; in a 0.1 ps test run an unsmoothed 2D case had the peak-energy band mode sitting at exactly zero damping.

Smoothing the histogram before differentiating fixes it, and is bias-free by construction: the per-k calibration `C(k)` is derived by applying the *same* operator to the expected Maxwellian, so any linear filter divides back out and only the variance reduction survives. Measured on the 2D smoke grid, fraction of band modes clamped to zero:

| `n_particles` | `hist_smooth` 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| 20,000 | 0.061 | 0.035 | 0.019 | 0.004 |
| 200,000 | 0.0075 | 0.0021 | 0.0000 | 0.0000 |

with the band-mean ratio to the analytic rate staying at 1.02–1.06 throughout. Raising `n_particles` works too (it is the same variance); smoothing is the cheaper lever. Quasi-1D keeps `hist_smooth: 0` so existing 1D results stay reproducible.

**Cost.** The push and the extraction are both $O(N_p)$ per step and independent of grid size. Measured on CPU (400x120 box, `n_particles = 5e5`, 54 substeps): the 1D HPE add-on is 49 ms/step and the 2D add-on is 357-402 ms/step, i.e. **~8x the 1D add-on** and ~20x the 2D fluid step. Scaled by the measured GPU 1D add-on this lands at roughly **2-4x** the cost of the same run without HPE, the fluid step there being launch-latency-bound.

The add-on is nearly *flat* in `n_angles` (386 / 357 / 402 ms at 16 / 32 / 64 — the spread is run-to-run noise): all angles are binned in one fused scatter, so the extraction is dominated by the single `(Np, n_angles)` projection and stays far below the push. Raise `n_angles` for angular resolution rather than treating it as the cost knob; `n_particles` and `substep_courant` are what actually move the runtime.

| Field | Type | Description |
|-------|------|-------------|
| `active` | bool | Enable HPE (default `false`) |
| `n_particles` | int | Number of tail test particles (default `500000`) |
| `v_min` | float | Tail cutoff in units of `vte` (default `2.5`); only `~erfc(v_min/sqrt(2))` of a Maxwellian is simulated |
| `v_max` | float | Histogram half-span in units of `c` (default `1.0`) |
| `nv` | int | Velocity bins spanning `(-v_max, v_max)` (default `512`) |
| `v_blend_buffer` | float | Buffer above `v_min` (units of `vte`) below which modes keep the analytic rate (default `0.5`) |
| `n_angles` | int | **2D only** (forced to `1` when `ny == 1`). Projection directions spanning `[0, pi)` onto which `f` is binned to build each mode's resonance integral (default `64`). Sets the angular resolution of the extraction *and* its cost, which is linear in `n_angles` |
| `y_thermal_frac` | float | **2D only.** Fraction of transverse wall crossings re-thermalized isotropically to mimic a finite plasma (default `0.0` = particles stay periodic in `y`, matching `terms.epw.boundary.y: periodic`; Follett used `0.2`) |
| `hist_smooth` | int | Passes of a `[1,2,1]/4` binomial filter over the projected histograms before `df/dv` is taken. **Defaults to 2 in 2D, 0 in quasi-1D**; an explicit setting always wins. See the clamping note below |
| `gather_refine` | int | Spectral upsampling factor for the field before the particle gather (default `4`). Linear interpolation of a wave with `k dx ~ 1-2` rad/cell attenuates the gathered field by `sinc^2(k dx / 2)` (15-30%); upsampling makes this ~1%. Applied in both directions in 2D, so the refined field is `(refine*nx, refine*ny)` complex per component — check the memory at large grids |
| `substep_courant` | float | `wp0 * dt_particle` for the sub-cycled push (default `0.05`; Follett used 0.035) |
| `tau_damping` | string | EMA time constant for the velocity histogram (default `"100fs"`, Follett's update interval) |
| `t_start` | string | Push/feedback disabled before this time (default `"0ps"`); use to let the fluid run reach steady state first |
| `feedback` | bool | (default `true`) `false` = control run: particles evolve but the damping stays analytic (Follett's control experiment) |
| `seed` | int | RNG seed for particle loading and wall re-injection (default `42`) |
| `omega_res` | string | Resonance convention for `v_phi(k)`: `"bohm_gross"` (default, matches the analytic rate) or `"wp0"` (bare carrier, as in the paper) |

```yaml
terms:
  hpe:
    active: true
    n_particles: 500000
    v_min: 2.5
    substep_courant: 0.05
    tau_damping: 100fs
    t_start: 2ps
    # 2D only (ignored when ny == 1)
    n_angles: 64
    y_thermal_frac: 0.0
```

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
