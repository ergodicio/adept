# Vlasov 1D2V Solver

Example decks live in `configs/vlasov-1d2v/`. To run one:

```bash
uv run run.py --cfg configs/vlasov-1d2v/epw
```

`solver: vlasov-1d2v` selects this module.

| Deck | What it shows |
|---|---|
| `epw.yaml` | Driven EPW with the marginal-coefficient `dougherty` operator |
| `epw-cylindrical-landau.yaml` | The same wave with the full-geometry `cylindrical_landau` operator and per-channel weights |

This solver exists to run the same electrostatic problems as
[Vlasov-1D](../vlasov1d/overview.md) but with a velocity space that can support a genuinely
3V-geometry collision operator — pitch-angle scattering has no meaning in 1V, and the anisotropy it
produces is exactly what distinguishes a real Coulomb operator from a 1D drift-diffusion caricature.

## Things You Might Care About

1. Infinite length (single mode) plasma waves — Landau damping, trapping
2. Finite length plasma waves — everything in 1. plus wavepackets
3. Wave dynamics on density gradients — 2. plus density gradients
4. How pitch-angle scattering versus speed diffusion modifies a trapped-particle wave, using the
   `cylindrical_landau` channel weights

## Equations and Quantities

The distribution is $f = f(x, v_\parallel, v_\perp)$, with $v_\parallel$ along the single spatial
direction and $v_\perp$ the magnitude of the two-dimensional perpendicular velocity, so the
phase-space measure is

$$
d^3v = 2\pi v_\perp \, dv_\perp \, dv_\parallel,
\qquad
w_\perp = 2\pi v_\perp \, \Delta v_\perp
$$

Only $v_\parallel$ couples to the electric field:

$$
\frac{\partial f}{\partial t}
+ v_\parallel \frac{\partial f}{\partial x}
+ \frac{q}{m} (E + E_D) \frac{\partial f}{\partial v_\parallel}
= C[f]
$$

$$
\partial_x^2 \phi = 1 - \int f \, d^3v, \qquad E = -\partial_x \phi
$$

The ions are static.

### Reuse of the 1D field machinery

The field solver only ever consumes velocity *moments*, and the moments of $f$ under the cylindrical
weight are exactly the 1D moments of the **marginal**

$$
F(x, v_\parallel) = \int f \, w_\perp \, dv_\perp
$$

So the electrostatic and EM field machinery from Vlasov-1D is reused verbatim, fed the marginal. The
marginal is the bridge to the 1D solver in every respect: it is initialized to exactly the
`vlasov-1d` initialization, saved in exactly the `vlasov-1d` layout, and — for the
marginal-coefficient collision operators — evolves with exactly the `vlasov-1d` dynamics. That is
what the `test_1d_limit.py` suite checks.

## Collisions

Two families of collision operator are available through `terms.fokker_planck.type`.

### Marginal-coefficient operators

`dougherty`, `dougherty_nodrag`, and `lenard_bernstein` act along $v_\parallel$ only, with drift and
diffusion coefficients computed from the marginal $F$:

$$
C[f] = \nu \frac{\partial}{\partial v_\parallel}
\left[ (v_\parallel - \bar v) f + \frac{1}{2\beta} \frac{\partial f}{\partial v_\parallel} \right],
\qquad (\bar v, \beta) \ \text{from} \ F
$$

This is deliberate rather than approximate bookkeeping: the discrete energy-flux condition is linear
in $f$ and the marginal is a $w_\perp$-weighted sum of slices, so computing the coefficients from the
marginal and applying the same tridiagonal operator to every $v_\perp$ slice conserves $n$,
$P_\parallel$, and $E_\parallel$ to the same standard as the 1D operator. For a separable
$f = F(v_\parallel) M(v_\perp)$ the marginal dynamics are *exactly* the 1D operator's.

### Cylindrical Landau

`cylindrical_landau` is the full-velocity-geometry linearized electron-electron operator:

$$
C[f] = \nu \, \nabla_v \cdot \left[ \mathbf{D}(v) \cdot M \nabla_v (f/M) \right],
\qquad M = e^{-|\mathbf{v} - \mathbf{u}|^2 / 2T}
$$

with the anisotropic test-particle tensor built about the bulk frame,

$$
\mathbf{D} = \alpha_{\text{speed}} \hat{D}_\parallel(s) \, \hat{s}\hat{s}
+ \alpha_{\text{lorentz}} \hat{D}_\perp(s) \, (\mathbf{I} - \hat{s}\hat{s}),
\qquad s = |\mathbf{v} - \mathbf{u}| / \sqrt{T}
$$

where $\hat{D}_\parallel = \psi(x)/s^3$ and $\hat{D}_\perp = [(1 - 1/s^2)\psi + \psi']/(2s)$ (with
$x = s^2/2$) are the erf-exact Rosenbluth-potential coefficients of a Maxwellian field species. The
two channel weights are independently configurable, and that is the main reason to reach for this
operator: it lets you attribute an effect on the wave to pitch-angle scattering or to speed diffusion
separately.

Writing the flux in the Einstein-relation form $M \nabla_v (f/M)$ means each channel annihilates the
bulk Maxwellian separately. Discretizing as $D \, M_{\text{edge}} (g_{i+1} - g_i)/dv$ with $g = f/M$
and $M_{\text{edge}}$ the geometric mean makes the *sampled* Maxwellian an exact discrete fixed point
of any channel mix, at any timestep.

**Moment restoration.** Momentum and energy exchanged with the implicit field particles are restored
each step by projecting onto shifted and heated bulk-Maxwellian modes — the discrete analogue of the
field-particle back-reaction. This is required for long-run stability, not just bookkeeping: the
operator re-linearizes about the live $(n, u, T)$ every call, and without restoration that loop
chases the $O(dv_\perp^2)$ midpoint-rule bias in the measured $T$ and drifts secularly. Leave
`moment_restoration: true` unless you are deliberately measuring the drift.

```{note}
The Krook operator is **not** implemented for this solver and raises `NotImplementedError`.
```

## Boundary Conditions

| Axis / quantity | Condition | Notes |
|---|---|---|
| $x$ | **Periodic** | Inherited from the Vlasov-1D field solvers, which are spectral. |
| $v_\parallel$ (advection) | **Periodic** | `edfdv: exponential` is the only supported push, so the same caveat as Vlasov-1D applies — keep $f \approx 0$ at both parallel edges. |
| $v_\parallel$, $v_\perp$ (`cylindrical_landau`) | **Zero-flux** | Finite-volume on the cylindrical grid with zero-flux outer boundaries, so density is conserved to solver precision. |
| $v_\perp$ (lower edge) | Cylindrical axis | $v_\perp \in (0, v_{\perp,\text{max}})$ with cell-centered points; there is no lower half to the axis. |
| Legendre/Dirichlet | — | Not applicable to this solver. |

## Forcing and Drivers

Forcing is inherited from Vlasov-1D and enters only the $v_\parallel$ force term:

- **`drivers.ex`** — a prescribed longitudinal field with tanh envelopes in space and time. The
  wavenumber, frequency, and amplitude set the wave; see the
  [configuration reference](config.md) for the schema.
- **`terms.fokker_planck.time.baseline`** sets $\nu$, and the `time`/`space` envelopes shape it
  exactly as the driver envelopes shape the driver. Collisions will modify the dynamics substantially
  depending on how far the distribution is driven from Maxwellian.

## What Gets Saved

**`binary/`**:

| File | Contents |
|---|---|
| `scalars-t=<t>.nc` | Scalar time series. Saved moments include `pperp` $= \int f v_\perp^2 w_\perp \, dv_\perp dv_\parallel$ alongside the parallel `p`; with two perpendicular degrees of freedom, $T_\perp = p_\perp / 2n$. |
| `dist-<save_key>.nc` | One file per species save block. A `{t, x, v}` save writes the rank-3 marginal $F(x, v_\parallel)$ in the `vlasov-1d` layout; a `{t}`-only save writes the full rank-4 $f(x, v_\parallel, v_\perp)$. |

**`plots/`**:

| Path | Contents |
|---|---|
| `plots/fields/spacetime_<field>.png` | Space-time plots of the shared fields, with a `logplots/` variant |
| `plots/fields/<species>/spacetime_<field>.png` | Per-species moments |
| `plots/scalars/<name>.png` | Scalar time series |
| `plots/dists/<species>/phase_space.png` | Phase-space snapshots |

`postprocess_time_min` is logged to MLflow as a metric.

### Cumulative diagnostics

`diagnostics.diag-vlasov-cumulative` and `diag-fp-cumulative` accumulate each term's contribution to
$\partial F / \partial t$ as marginal `(nx, nv)` arrays holding a running **time integral**, not a
sampled rate. Difference them between save points to recover exact interval-averaged rates — sampling
a rate instead would alias the $2\omega$ wave-particle energy exchange. Enable them under
`diagnostics`, then add matching entries under `save` to write them out.

## Solver Options

### Time Integration (`terms.time`)

1. **`leapfrog`** — 2nd-order.
2. **`sixth`** — 6th-order Hamiltonian splitting (Crouseilles et al.). Worth the extra cost when the
   collision time is long compared to the wave period and time-integration error would otherwise
   dominate the measurement.

### Velocity Advection (`terms.edfdv`)

Only `exponential` is supported; anything else raises `NotImplementedError`.

## Practical Notes

**Density profile.** Uniform is easy. For a non-uniform profile the density can be parameterized as a
sinusoidal perturbation or a tanh flat top; see
[initialization](../../usage/initialization.md) for the tanh flat-top parameters. The $v_\perp$
dependence is always a Maxwellian at the component's `T0` — super-Gaussian shapes apply to
$v_\parallel$ only.

**Velocity grid.** Beyond the 1D `nv`/`vmax` you must set `nvperp` and `vperp_max`; they have no
defaults. How fine the perpendicular axis needs to be depends on the collision operator — see
[choosing `nvperp`](config.md#choosing-nvperp).

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
