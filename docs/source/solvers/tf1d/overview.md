# Two-Fluid 1D Solver

Example decks live in `configs/tf-1d/`. To run one:

```bash
uv run run.py --cfg configs/tf-1d/epw
```

`solver: tf-1d` selects this module.

A 1D warm two-fluid Poisson solver. Where the kinetic solvers evolve a distribution function, this
one evolves only the fluid moments — density, velocity, and pressure — and closes the hierarchy with
a prescribed (or machine-learned) closure. That makes it dramatically cheaper than
[Vlasov-1D](../vlasov1d/overview.md) and is the module used for the closure-learning work in the
ADEPT references.

## Equations and Quantities

The normalized 1D fluid-Poisson system, as implemented, is

$$
\partial_t n_s + \partial_x (n_s u_s) = 0
$$

$$
\partial_t u_s + u_s \partial_x u_s
= \frac{q_s}{m_s} (E + E_D)
- \frac{1}{n_s} \mathcal{R}\!\left[ \partial_x \frac{P_s}{m_s} \right]
+ \mathcal{L}[u_s]
$$

$$
\partial_t P_s + u_s \partial_x P_s + \gamma_s P_s \partial_x u_s = 2 n_s u_s \frac{q_s}{m_s} E
$$

$$
\partial_x E = \sum_s q_s n_s
$$

where $n_s$, $u_s$, and $P_s$ are the density, fluid velocity, and pressure of species $s$, and
$E_D$ is an optional external driver. Spatial derivatives are evaluated spectrally on a periodic
domain.

Two operators in the momentum equation carry the kinetic corrections, and both reduce to nothing in
the plain fluid limit:

- $\mathcal{R}$ is a wavenumber-space correction applied to the pressure-gradient (restoring force)
  term. It is the identity unless `gamma: kinetic`, in which case it rescales the restoring force so
  the wave frequency matches the kinetic dispersion relation at each $k$.
- $\mathcal{L}$ is the Landau damping operator, zero unless `landau_damping` is enabled, and
  modulated by the trapping model when that is on.

```{note}
The pressure equation carries a field-work term $2 n u (q/m) E$ on the right-hand side, so `p` is
not a pure pressure — it absorbs the work done by the self-consistent field. Note also that the
external driver $E_D$ enters the momentum equation only; the pressure equation sees the
self-consistent $E$ alone.
```

Both an electron and an ion species are available. Each can be switched off independently with
`physics.<species>.is_on`; a species that is off acts as a static neutralizing background.

## Closures

The fluid hierarchy has to be truncated somewhere, and the choice of closure is what determines how
much kinetic physics survives.

### Adiabatic index (`physics.<species>.gamma`)

1. **A number** (e.g. `3`) — a conventional adiabatic closure, $P \propto n^\gamma$. Use `3` for
   1D adiabatic compression, `1` for isothermal.
2. **`kinetic`** — the pressure equation is closed against the *kinetic* dispersion relation:
   $\gamma$ is set to 1 and the real frequency is interpolated from a precomputed table of the
   kinetic EPW dispersion, so the fluid model reproduces the correct Bohm-Gross frequency across
   $k \lambda_D$ rather than only in the long-wavelength limit.

### Landau damping (`physics.<species>.landau_damping`)

Fluid equations have no Landau damping — it is a kinetic, phase-mixing effect. Setting this flag
adds a damping term whose rate is interpolated from a table of the kinetic imaginary frequency
$\omega_i(k \lambda_D)$, restoring the correct linear damping to the fluid model.

### Trapping (`physics.<species>.trapping`)

At finite amplitude, particles trapped in the wave flatten the distribution at the phase velocity
and the damping is reduced. Enabling `trapping` modulates the Landau damping term to capture this:

- **`zk`** — the Zakharov-Karpman-style reduction, driven by the trapping frequency computed from
  the local field amplitude at wavenumber `kld`
- **`delta`** — a reduction proportional to an evolved $\delta$ variable
- **`none`** — no modification

`nuee` sets the electron-electron collision rate that competes with trapping (collisions refill the
flattened region and restore damping), and `kld` sets the wavenumber at which the trapping
diagnostic is evaluated.

## Machine-Learned Closures

The `models` block attaches trainable `equinox` modules to the closure terms — this is the
differentiable-simulation use case described in reference [2] of the
[README](https://github.com/ergodicio/adept), where a neural network learns the closure that best
reproduces kinetic results. Because the whole solver is written in JAX, gradients propagate through
the entire time integration, so the network can be trained on a metric evaluated at the end of a
simulation. See `adept/_tf1d/train_damping.py` for a worked training loop.

## Things You Might Care About

1. Infinite-length (single mode) plasma waves — Landau damping, trapping
2. Finite-length plasma waves — the above plus wavepackets
3. Wave dynamics on density gradients
4. Machine-learned fluid closures

## Boundary Conditions

| Axis / quantity | Condition | Notes |
|---|---|---|
| $x$ | **Periodic** | All spatial derivatives are spectral (FFT), so the domain is periodic by construction. There is no velocity axis to give boundary conditions to — that is the whole point of a fluid model. |

Because there is no velocity grid, `nx` need only resolve the spatial modes of interest; the example
decks run with `nx: 16`.

## Forcing and Drivers

| Block | What it does |
|---|---|
| `drivers.ex` | A prescribed longitudinal field added to the momentum equation, keyed by pulse index. Each pulse is a travelling wave set by `k0`, `w0`, `dw0`, and `a0`, with tanh envelopes in space (`x_c`, `x_w`, `x_r`) and time (`t_c`, `t_w`, `t_r`). Set `x_w` very large for a spatially uniform drive. |
| `physics.<species>.landau_damping` | Not an external driver but the other term that injects or removes wave energy: a damping rate interpolated from the tabulated kinetic $\omega_i(k\lambda_D)$. |

Note that the driver enters the momentum equation only — the pressure equation sees the
self-consistent $E$ alone.

```{note}
The field names here are the short forms (`t_c`, `t_w`, …), not the nested `params`/`envelope`
structure used by Vlasov-1D and PIC-1D.
```

## What Gets Saved

**`binary/`**:

| File | Contents |
|---|---|
| `state_vs_<label>.nc` | The fluid state — $n$, $u$, $P$, and the trapping variable $\delta$ per species, plus the field — on each configured output axis. `save` takes three axes (`t`, `x`, `kx`), so you get a real-space and a wavenumber-space view. |
| `ground_truth.nc` | Written for the training workflows, which compare against a reference run. |

**`plots/`** — one subdirectory per output axis, with a PNG per state variable.

## Practical Notes

**Density profile.** Uniform is easy. For a non-uniform profile the density can be parameterized as a
sinusoidal perturbation or a tanh flat top — see
[initialization](../../usage/initialization.md) for the tanh flat-top parameters.

**Static ions.** The ions are static by default (`physics.ion.is_on: false`) but all the functionality
is in place if someone wants to get them to move.

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
