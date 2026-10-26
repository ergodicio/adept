# PIC-1D Configuration Reference

This document describes how to construct a configuration file for the `pic-1d` solver, a 1D1V
electrostatic particle-in-cell solver.

PIC-1D shares its `units`, `density`, `save`, `drivers`, and `mlflow` blocks with
[Vlasov-1D](../vlasov1d/config.md) — the same normalization, the same density-profile
parameterization, and the same driver envelopes. Only `grid` and `terms` differ, because the
solver resolves velocity space with particles rather than a mesh. This page documents the blocks
that differ and links out for the rest.

## Top-Level Structure

```yaml
solver: pic-1d
units: ...        # see Vlasov-1D
density: ...      # see Vlasov-1D
grid: ...         # PIC-specific, below
terms: ...        # PIC-specific, below
drivers: ...      # see Vlasov-1D
save: ...         # see Vlasov-1D
mlflow: ...
diagnostics: {}
```

---

## grid

Spatial/time grid plus the particle-resolution knobs.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dt` | float or string | — | Timestep |
| `nx` | int | — | Number of spatial grid cells (for the field solve and deposition) |
| `tmin` | float or string | `0.0` | Start time |
| `tmax` | float or string | — | End time |
| `xmin` | float or string | — | Domain minimum x |
| `xmax` | float or string | — | Domain maximum x |
| `ppc` | int | `256` | Particles per cell. Total particle count is `nx * ppc` per species. |
| `particle_shape` | `linear`, `tsc`, `cubic` | `tsc` | B-spline used for both charge deposition and field gather |

As with Vlasov-1D, dimensional inputs may be given as strings with units (e.g. `xmax: 100um`) and
are converted using the `units` block; plain numbers are taken to be in code units already.

`ppc` is the main accuracy/cost dial. PIC noise in the field falls off as roughly
$1/\sqrt{N_{\text{particles}}}$, so a linear-response measurement such as Landau damping needs a
much larger `ppc` than a nonlinear saturation study — the example decks use `ppc: 32768` for
damping measurements.

Example:
```yaml
grid:
  dt: 0.1
  nx: 32
  tmin: 0.0
  tmax: 100.0
  xmin: 0.0
  xmax: 20.94
  ppc: 32768
  particle_shape: cubic
```

---

## terms

Selects the field solver, the time integrator, and the particle species.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `field` | `poisson` | `poisson` | Field solver. Spectral Poisson is currently the only option. |
| `time` | `leapfrog`, `yoshida4` | `leapfrog` | Symplectic integrator. `leapfrog` is 2nd-order kick-drift-kick; `yoshida4` is a 4th-order composition of three leapfrog steps at ~3x the per-step cost. |
| `species` | list | — | Particle species (see below) |

### species

Each entry defines one particle species. This mirrors `terms.species` in Vlasov-1D, but carries
PIC-specific loading fields in place of the velocity-grid fields (`vmax`, `nv`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `electron` | Species name. Used as the key for per-species save blocks and output variables. |
| `charge` | float | `-1.0` | Species charge in units of $e$ |
| `mass` | float | `1.0` | Species mass in units of $m_e$ |
| `density_components` | list[string] | `null` | Names of blocks in `density` that make up this species' profile. Defaults to all of them. |
| `loading` | `quiet`, `random` | `quiet` | Particle loading scheme. `quiet` places particles on uniform position slots and carries the density profile in the weights; `random` samples positions from the profile at uniform weight. |
| `vmax_load` | float | `8.0` | Velocity cutoff (in thermal velocities) for the inverse-CDF / rejection sampling used to draw particle velocities |

Species not listed here are treated as a static neutralizing background.

Example:
```yaml
terms:
  field: poisson
  time: leapfrog
  species:
    - name: electron
      charge: -1.0
      mass: 1.0
      loading: quiet
      vmax_load: 8.0
```

---

## Shared Blocks

These are identical to Vlasov-1D; follow the links for the full schema.

| Block | Reference |
|-------|-----------|
| `units` | [Vlasov-1D units](../vlasov1d/config.md#units) — normalizing temperature and density, and the normalization convention derived from them |
| `density` | [Vlasov-1D density](../vlasov1d/config.md#density) — profile shape, super-Gaussian order, noise seeding, quasineutrality |
| `drivers` | [Vlasov-1D drivers](../vlasov1d/config.md#drivers) — `ex` longitudinal drivers and `ey` transverse drivers, with tanh envelopes in space and time |
| `save` | [Vlasov-1D save](../vlasov1d/config.md#save) — per-quantity time grids for fields and per-species output |
| `mlflow` | [Vlasov-1D mlflow](../vlasov1d/config.md#mlflow) — experiment and run names |

```{note}
`density.quasineutrality: true` sets the static ion charge density to match the initial electron
density, so the initial field is zero to within deposition error.
```

---

## Complete Example

A driven electron plasma wave, using the deck in `configs/pic-1d/`:

```yaml
solver: pic-1d

units:
  normalizing_temperature: 2000eV
  normalizing_density: 1.5e21/cc

density:
  quasineutrality: true
  species-background:
    noise_seed: 420
    noise_type: gaussian
    noise_val: 0.0
    v0: 0.0
    T0: 1.0
    m: 2.0
    basis: uniform
    baseline: 1.0
    bump_or_trough: bump
    center: 0.0
    rise: 25.0
    bump_height: 0.0
    width: 100000.0

grid:
  dt: 0.1
  nx: 32
  tmin: 0.0
  tmax: 100.0
  xmin: 0.0
  xmax: 20.94
  ppc: 32768
  particle_shape: cubic

terms:
  field: poisson
  time: leapfrog
  species:
    - name: electron
      charge: -1.0
      mass: 1.0
      loading: quiet
      vmax_load: 8.0

drivers:
  ex:
    '0':
      params:
        a0: 1.e-3
        k0: 0.3
        w0: 1.1598
        dw0: 0.
      envelope:
        time: {center: 40.0, rise: 5.0, width: 30.0}
        space: {center: 0.0, rise: 10.0, width: 4000000.0}
  ey: {}

save:
  fields:
    t: {tmin: 0.0, tmax: 100.0, nt: 601}
  electron:
    main:
      t: {tmin: 0.0, tmax: 100.0, nt: 11}

mlflow:
  experiment: basic-epw-pic
  run: leapfrog-cubic

diagnostics: {}
```

See the [Overview](overview.md) for the equations and a discussion of the solver options.
