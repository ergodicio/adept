# Mixed Hermite-Legendre 1D Solver

Example decks live in `configs/hermite-legendre-1d/`. To run one:

```bash
uv run run.py --cfg configs/hermite-legendre-1d/bump-on-tail
```

`solver: hermite-legendre-1d` selects this module.

A 1D-1V electrostatic Vlasov-Poisson solver that uses **two** velocity-space bases at once,
following Issan, Delzanno & Roytershteyn ([arXiv:2606.12322](https://arxiv.org/abs/2606.12322)).

The motivation is that a pure Hermite expansion is very efficient for a near-Maxwellian plasma and
very inefficient for anything else: resolving a beam, a plateau, or a sharp velocity-space feature
takes a large number of Hermite modes, because those features are poorly represented by
Maxwellian-weighted polynomials. Splitting the distribution lets each basis do what it is good at.

## Equations and Quantities

The electron distribution is split into a near-Maxwellian bulk and a correction:

$$
f(x, v, t) = f_0(x, v, t) + \delta f(x, v, t)
$$

The bulk is expanded in an asymmetrically-weighted (AW) Hermite basis, and the correction in a
Legendre basis on a bounded velocity window $[v_a, v_b]$:

$$
f_0 = \sum_{n=0}^{N_h - 1} C_n(x, t) \, \psi_n(v; \alpha, u), \qquad
\delta f = \sum_{m=0}^{N_l - 1} B_m(x, t) \, \xi_m(v; v_a, v_b)
$$

These evolve under the electrostatic Vlasov-Poisson system

$$
\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} - E \frac{\partial f}{\partial v} = 0,
\qquad \partial_x E = 1 - \int f \, dv
$$

with a single electron species against an immobile neutralizing ion background of density 1.
Normalization is time by $1/\omega_{pe}$, space by the Debye length $\lambda_D$, and velocity by the
electron thermal velocity $v_{th,e}$.

The coupling between the two bases is one-way: the highest Hermite coefficient $C_{N_h - 1}$ feeds
the Legendre modes, and both sets of coefficients feed the self-consistent field through Poisson.
The method pays off when the non-Maxwellian features are *localized* in velocity — a bump-on-tail
beam is the canonical case — since then a modest Legendre window resolves what would otherwise cost
many Hermite modes.

## Numerics

Space is treated spectrally on a periodic domain (Fourier, $\partial_x \to i k_x$). Both
free-streaming operators are symmetric-tridiagonal in mode index and are integrated **exactly** via
prediagonalized matrix exponentials. The recommended **split** integrator uses exact half linear
steps around a local velocity-force update. The Hermite force is solved by a bidiagonal recurrence;
the Legendre implicit-midpoint Cayley transform uses the derivative matrix's lower-triangular plus
rank-2 structure. This removes the global Newton/GMRES solve and its Krylov-vector memory cost.

```{note}
With `gamma=0.5` on every Legendre mode, the force generator is skew-symmetric to roundoff and the
Cayley update preserves its coefficient norm. A minimum-norm correction of the six low `k=0`
Hermite/Legendre coefficients restores total mass, momentum, and energy after each split step.
```

Artificial collision rates `nu_H` and `nu_L` damp the highest modes of each basis to control
filamentation, in the same spirit as the hypercollisions in
[Spectrax-1D](../spectrax1d/overview.md).

For collisional physics, `physics.collisions` offers two total-distribution models:
`bgk` is a linear-cost relaxation to the same-moment Maxwellian, while `dougherty`
is the higher-fidelity Fokker--Planck option with quadratic modal cost. Both use
native Hermite/Legendre updates with no conversion or least-squares projection and
restore density, momentum, and kinetic energy exactly. These remain 1D model
operators; full Coulomb pitch-angle and perpendicular-energy scattering require the
`vlasov-1d2v` solver's cylindrical Landau operator.

## Initialization

The `initialization.type` key selects the initial condition:

- **`linear-advection`** — free-streaming with no field, for verifying the streaming operators
- **`two-stream`** — counter-propagating beams, for the two-stream instability
- **`bump-on-tail`** — a Maxwellian bulk plus a fast beam, the case the mixed basis is built for
- **`custom`** — user-specified

An external longitudinal driver `drivers.ex` can be applied. It enters only the $E \cdot \partial_v f$
force term and never the Poisson solve, so the self-consistent field-energy diagnostic excludes the
driver — useful for a clean Landau-damping measurement.

## Boundary Conditions

| Axis / quantity | Condition | Notes |
|---|---|---|
| $x$ | **Periodic** | Spatial dependence is carried in Fourier space, $\partial_x \to i k_x$. |
| $v$ (Hermite part $f_0$) | **None** — spectral | The AW-Hermite basis is defined on the whole line; the truncation at $N_h$ is the effective closure. |
| $v$ (Legendre part $\delta f$) | **Weak Dirichlet** | $\delta f(v_a) = \delta f(v_b) = 0$, enforced by a rank-2 penalty term $P[m,j] = (\gamma_m/\text{width})(\xi_b[m]\xi_b[j] - \xi_a[m]\xi_a[j])$ with strength `gamma`. This is what confines the Legendre correction to its velocity window instead of letting it leak. |

In the split integrator the penalty is part of the structured local Cayley solve; both free-streaming
operators are integrated exactly.

## Forcing and Drivers

| Block | What it does |
|---|---|
| `initialization.type` | The dominant "forcing" here is the initial condition: `linear-advection`, `two-stream`, `bump-on-tail`, or `custom`, each with its own parameters. |
| `drivers.ex` | A prescribed longitudinal field $E_\text{drive}(x,t) = \sum \text{env}(x,t)(\omega_0 + \delta\omega_0) a_0 \sin(k_0 x - (\omega_0+\delta\omega_0)t)$. It enters **only** the $E \cdot \partial_v f$ force term and never the Poisson solve, so the self-consistent field-energy diagnostic excludes the driver — which is what makes a clean Landau-damping measurement possible. |
| `physics.nu_H`, `nu_L` | Artificial collision rates damping the highest modes of each basis, to control filamentation. |
| `physics.collisions` | Optional conservative `bgk` or `dougherty` total-distribution collision model. |

## What Gets Saved

**`binary/`**:

| File | Contents |
|---|---|
| `fields-t=<t>.nc` | Field quantities on the requested time grid |
| `<name>-t=<t>.nc` | One file per configured save stream |
| `distribution-f_xv.nc` | The reconstructed $f(x, v)$ — both bases evaluated back onto a velocity grid |

**`plots/`**:

| File | Contents |
|---|---|
| `spacetime-<field>.png` | Space-time plots per field |
| `scalar-<name>.png` | Scalar time series, including the conservation diagnostics |
| `coefficients-facets.png` | Hermite and Legendre coefficient spectra |
| `phase-space-f_xv.png` | Reconstructed phase space |

The scalar diagnostics directly report the largest Legendre-window boundary value, the high-mode
Legendre energy fraction, the step residual, and the low-mode conservation-correction norm. Together
with the coefficient-facet plot, these distinguish boundary contamination from spectral recurrence
or loss of predictor/corrector convergence.

## Configuration Reference

See the [Configuration Reference](config.md) for complete YAML schema documentation.
