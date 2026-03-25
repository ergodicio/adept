# Spectrax-1D Configuration Reference

This document describes how to construct a configuration file for the `spectrax-1d`
(and `hermite-epw-1d`) solvers, which implement a Hermite-Fourier spectral
Vlasov-Maxwell solver for 1D plasma kinetics.

## Top-Level Structure

```yaml
solver: spectrax-1d      # or hermite-epw-1d for EPW analysis
mlflow: ...
units: ...
physics: ...
grid: ...
drivers: ...
save: ...
```

---

## physics

Physical parameters for the simulation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Lx` | float | — | Domain length in x (normalized) |
| `Ly` | float | — | Domain length in y (normalized) |
| `Lz` | float | — | Domain length in z (normalized) |
| `mi_me` | float | — | Ion-to-electron mass ratio |
| `qs` | list[float] | — | Species charges `[q_electron, q_ion]`, e.g. `[-1.0, 1.0]` |
| `alpha_s` | list[float] | — | All-species thermal velocities `[αx_e, αy_e, αz_e, αx_i, αy_i, αz_i]` |
| `u_s` | list[float] | — | All-species drift velocities (same layout as `alpha_s`) |
| `Omega_ce_tau` | float | — | Reference cyclotron frequency (electron mass normalized). Species cyclotron frequency is `(q/m) * Omega_ce_tau`. |
| `nu` | float | — | Hypercollision frequency |
| `nx`, `ny`, `nz` | int | — | Perturbation mode numbers for initialization |
| `dn1` | float | — | Density perturbation amplitude |
| `static_ions` | bool | `false` | Freeze ion distribution (no Lorentz force, no ion current, no free-streaming). Ions retain their initial equilibrium background but do not evolve. Useful for studying pure electron physics (EPW, SRS scattering) at reduced cost. |

### `static_ions`

When `static_ions: true`:
- Ion distribution `Ck_ions` is held fixed at its initial (Maxwellian) state
- Ion current is excluded from Ampère's law
- Ion free-streaming and collision operators are bypassed (exponential integrator)

Ion dynamics are irrelevant at electron plasma wave (EPW) frequencies because
`ω_EPW / (k v_{th,i}) ≫ 1`. The EPW dispersion is unchanged to within numerical
tolerance when this flag is enabled. Static ions is a useful diagnostic and
computational shortcut.

---

## grid

Simulation grid and time-integration parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Nx`, `Ny`, `Nz` | int | — | Number of Fourier modes per spatial dimension |
| `Nn`, `Nm`, `Np` | int | — | Hermite modes (legacy: same for both species) |
| `hermite_modes` | dict | — | Per-species Hermite mode counts (see below) |
| `Ns` | int | `2` | Number of species |
| `tmax` | float | — | Simulation end time |
| `dt` | float | — | Timestep (calculated from `tmax/nt` if omitted) |
| `nt` | int | — | Number of timesteps (calculated from `tmax/dt` if omitted) |
| `solver` | string | `"Dopri8"` | Diffrax solver name (`"Dopri8"`, `"Tsit5"`, etc.) |
| `adaptive_time_step` | bool | `true` | Use PID adaptive step-size controller |
| `integrator` | string | `"explicit"` | Time integrator: `"explicit"` (Runge-Kutta via Diffrax) or `"exponential"` (Lawson-RK4) |
| `use_shard_map` | bool | `false` | Enable multi-device sharding along Nx |

### Per-species Hermite modes

Use `hermite_modes` for independent electron/ion resolution:

```yaml
grid:
  hermite_modes:
    electrons:
      Nn: 512    # velocity modes in x
      Nm: 1
      Np: 1
    ions:
      Nn: 32
      Nm: 1
      Np: 1
```

### Integrators

**`explicit`** (default): Standard Runge-Kutta via Diffrax (e.g., Dopri8). Adaptive
time-stepping resolves light waves and free-streaming stiffness.

**`exponential`** (Lawson-RK4): Factors out the linear part (free-streaming, Maxwell
curls, collision) into exact matrix exponentials, removing CFL stiffness. Must be
paired with `adaptive_time_step: false` and a suitable fixed `dt`.

---

## drivers

External electromagnetic field drivers.

```yaml
drivers:
  ex:                  # drives Ex component
    '0':
      k0: 6.2832       # wavenumber (rad/L)
      w0: 1.104        # frequency (ωpe)
      a0: 1.0e-6       # amplitude
      t_center: 35.0   # pulse center time
      t_width: 30.0    # pulse width
      t_rise: 14.0     # rise/fall time
      x_center: 0.5    # spatial center (normalized)
      x_width: 1000.0  # spatial width (large = uniform)
      x_rise: 0.1
      dw0: 0.0         # frequency detuning
  ey: {}
  ez: {}
```

### density

Stochastic density noise injected into the `(0,0,0)` Hermite mode each timestep.

```yaml
density:
  noise:
    enabled: true
    type: uniform        # "uniform" or "normal"
    amplitude: 1.0e-12
    seed: 42
    electrons:
      enabled: true
      amplitude: 1.0e-12
    ions:
      enabled: false
```

### hermite_filter

Hou-Li exponential damping of high Hermite modes (prevents filamentation).

Applies σ(h) = exp(-strength * (h / h_max)^order) to all modes, where h is the
Euclidean norm of the mode index triple (n, m, p) and h_max is the norm at the
highest-index corner of the mode grid.

```yaml
drivers:
  hermite_filter:
    enabled: true
    strength: 4.0              # filter strength (higher = more aggressive damping)
    order: 4                   # filter order (higher = sharper roll-off near h_max)
```

---

## save

Quantities to save during the simulation via Diffrax `SubSaveAt`.

```yaml
save:
  fields:           # electromagnetic fields Fk in Fourier space
    t:
      tmin: 0.0
      tmax: 200.0
      nt: 801

  hermite:          # full Hermite-Fourier distribution Ck (both species)
    t:
      tmin: 0.0
      tmax: 200.0
      nt: 101

  moments:          # real-space density/velocity/temperature moments
    t:
      tmin: 0.0
      tmax: 200.0
      nt: 201
```

A `default` save (scalar diagnostics: EM energy, peak fields, etc.) is always added
automatically at every grid timestep.
