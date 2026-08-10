# Spectrax 1D Solver

Example decks live in `configs/spectrax-1d/`. To run one:

```bash
uv run run.py --cfg configs/spectrax-1d/landau-damping
```

A Hermite-Fourier spectral solver for the Vlasov-Maxwell system. Unlike the grid-based
[Vlasov-1D](../vlasov1d/overview.md) solver, velocity space is not discretized onto a mesh — it is
expanded in a Hermite basis, so the kinetic state is a set of spectral coefficients rather than a
sampled $f(x, v)$.

## Equations and Quantities

We solve the Vlasov equation for each species $s$ coupled to Maxwell's equations:

$$
\frac{\partial f_s}{\partial t} + \mathbf{v} \cdot \nabla_x f_s
+ \frac{q_s}{m_s} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right) \cdot \nabla_v f_s = C[f_s]
$$

Normalization is time in $\omega_{pe}^{-1}$, length in $c/\omega_{pe}$, so that $c = 1$ and
$\omega_{pe} = 1$.

The distribution function is expanded in an asymmetrically-weighted Hermite basis in each velocity
direction and a Fourier basis in space:

$$
f_s(\mathbf{x}, \mathbf{v}, t) = \sum_{n,m,p} \sum_{\mathbf{k}}
C^s_{nmp}(\mathbf{k}, t) \, \psi_n(v_x) \, \psi_m(v_y) \, \psi_p(v_z) \, e^{i \mathbf{k} \cdot \mathbf{x}}
$$

where each $\psi$ is scaled by a per-species thermal velocity $\alpha_s$ and shifted by a drift
$u_s$ (`alpha_s` and `u_s` in the config). Substituting the expansion turns the Vlasov equation into
a coupled set of ODEs for the coefficients $C^s_{nmp}(\mathbf{k}, t)$. Free-streaming and the
Lorentz force each couple only to neighbouring Hermite indices — the three-term recurrence of the
Hermite ladder operators $\sqrt{n}$, $\sqrt{n+1}$ — so the right-hand side stays sparse in mode
index.

Maxwell's equations are advanced in the same Fourier space, with the plasma current obtained
directly from the first-order Hermite coefficients.

Note that although the module is named "1D", the state carries $(N_x, N_y, N_z)$ Fourier modes and
$(N_n, N_m, N_p)$ Hermite modes. 1D problems are the case where the transverse mode counts are set
to 1; the velocity space remains three-dimensional.

### Closure and Filtering

The Hermite hierarchy has to be truncated, and a sharp truncation reflects energy back down the
spectrum as filamentation in velocity space. Two mechanisms control this:

- **Hypercollisions** — a damping rate proportional to mode index (`physics.nu`), which acts most
  strongly on the highest Hermite modes and leaves the low-order moments essentially untouched.
- **Hou-Li filter** — an exponential mask $\sigma(h) = \exp(-\text{strength} \cdot (h/h_{max})^{\text{order}})$
  applied across the Hermite indices (`drivers.hermite_filter`).

## Solver Options

### Time Integration

1. **`exponential`** (default, recommended) — Lawson-RK4. The linear part of the system
   (free-streaming, the Maxwell curls, and the collision operator) is factored out into exact matrix
   exponentials, which removes the CFL stiffness those terms would otherwise impose. Only the
   nonlinear terms (Lorentz force, plasma current) are integrated explicitly.
2. **`explicit`** — a plain Runge-Kutta method through `diffrax` (`Dopri8` by default). Simpler, but
   the timestep is limited by the stiffest linear term.

Both support adaptive stepping via `grid.adaptive_time_step`. For the exponential integrator this
uses an embedded 2nd-order Lawson-Heun companion to control error on the nonlinear term alone.

### Field Solver

Maxwell's equations are solved spectrally. Setting `physics.static_ions: true` freezes the ion
distribution at its initial Maxwellian — ion current is dropped from Ampère's law and ion
free-streaming is bypassed. At electron plasma wave frequencies $\omega / k v_{th,i} \gg 1$, so this
leaves the dispersion unchanged while cutting the cost roughly in half.

## Entry Points

Three `solver:` keys share this module and differ only in post-processing:

| `solver:` key | Purpose |
| --- | --- |
| `spectrax-1d` | Full Hermite-Fourier Vlasov-Maxwell |
| `hermite-epw-1d` | Adds electron plasma wave diagnostics — first-mode amplitude, instantaneous frequency from the phase evolution, and Hermite coefficient spectra |
| `hermite-maxwell-1d` | Adds electromagnetic dispersion diagnostics — measures the $k=1$ transverse mode frequency against $\omega^2 = \xi^2 + 1$, and an absorption ratio for checking sponge layers |

## Boundary Conditions

| Axis / quantity | Condition | Notes |
|---|---|---|
| $x, y, z$ | **Periodic** | The spatial representation is Fourier, so periodicity is structural. A 2/3 dealiasing mask is applied in Fourier space. |
| Velocity | **None** — spectral | There is no velocity grid and therefore no velocity boundary. The truncation of the Hermite hierarchy plays the role a velocity boundary plays elsewhere, which is why the closure and filtering below matter so much. |
| $x$ (optional) | **Sponge layer** | An absorbing layer can be applied through spatial damping profiles — separate coefficients for the EM fields, the electrons, and the ions. Sponge layers make the linear operator stiff, which is the case the split-step exponential integrator exists to handle. |

## Forcing and Drivers

| Block | What it does |
|---|---|
| `drivers.ex`, `ey`, `ez` | Prescribed field drivers, one per component, keyed by pulse index. Each has `k0`, `w0`, `a0`, `dw0`, plus tanh envelopes in space and time. |
| `density.noise` | Stochastic density noise injected into the $(0,0,0)$ Hermite mode every timestep, configurable per species with independent amplitudes and a seed. This is what seeds an instability from a clean initial condition. |
| `drivers.hermite_filter` | Not forcing but the counterpart to it: Hou-Li exponential damping of the high Hermite modes. |

## What Gets Saved

**`binary/`**:

| File | Contents |
|---|---|
| `fields-t=<t>.nc`, `fields-<axis>-t=<t>.nc` | EM fields in Fourier space, per axis |
| `field_diagnostics-t=<t>.nc` | Derived field diagnostics |
| `moments-t=<t>.nc` | Real-space density, velocity, and temperature moments |
| `distribution_<species>_timeseries-t=<t>.nc` | The Hermite-Fourier coefficients $C_k$ |
| `scalars-t=<t>.nc` | Scalar time series — EM energy, peak fields |
| `energies.nc`, `fields.nc`, `distribution.nc` | Consolidated views |

**`plots/`**:

| File | Contents |
|---|---|
| `field_mode_amplitudes.png`, `magnetic_field_mode_amplitudes.png` | Mode amplitude histories |
| `spacetime-<field>.png`, `<field>.png` | Space-time plots and lineouts per field |
| `Ck_amplitude_facets.png`, `hermite_mode_amplitudes_kx1.png` | Hermite spectra — the diagnostic to watch for filamentation |
| `plots/fields/`, scalars PNGs | Per-field and per-scalar plots |

The `hermite-epw-1d` entry point adds `epw_mode1_diagnostics.png` and
`epw_hermite_coefficients_2x2.png`; `hermite-maxwell-1d` adds `em_wave_k1_diagnostics.png`.

A `default` save (scalar diagnostics) is always added automatically at every grid timestep, so you
get scalars even with an empty `save` block.

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
