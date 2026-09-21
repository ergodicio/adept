# Driven interaction-region reservoirs

VFP2D can maintain prescribed upstream plasma conditions in smooth bands inside
its periodic box. This lets a reduced interaction-region calculation replenish
magnetized inflow and damp plasma entering the remote buffer. It is an externally
forced periodic problem. It does **not** impose open electromagnetic or kinetic
boundary conditions, simulate a wire array, or guarantee that an outflow leaves
without reflection.

Use a separate configuration without reservoirs for conservation tests. Before
interpreting an interaction-region observable, enlarge the box and vary the band
width and relaxation time until that observable is insensitive to the buffer.
Keep the diagnostic upstream points and current sheet outside the driven bands.

## Configuration

`configs/vfp-2d/magpie-carbon-driven.yaml` supplies a two-step carbon smoke
configuration. Its upstream reference, numerical resolution, cutoff and cost
limitations are the same as the [periodic geometry example](magpie_geometry.md).

The target is a copy of the fully initialized ion state, electron distribution,
and magnetic field. Use spatial initial conditions to define the target inflows.

```yaml
terms:
  field_solver: {mode: kinetic-ohm}
  ion_fluid: {active: true, frozen: false}
drivers:
  reservoir:
    active: true
    target: initial_state
    relaxation_time: 1ns
    x_width: 2mm
    y_width: 1mm
    magnetic: true
```

Widths include the entire smooth transition and must be strictly smaller than
the corresponding half-box. Omit a width or set it to zero to disable that pair
of bands. At least one width must be positive. The bands use a quintic
smoothstep; their particle source is exactly zero in the interior. Overlap is
combined as `1-(1-mask_x)*(1-mask_y)`, so a corner is not driven twice as fast.
This interface requires evolving ions, kinetic Ohm, and unsharded spectral
spatial derivatives. Physical strings have their stated units; numeric values
are normalized code units.

The drive is always on. For a pulsed source, use a distinct future time-dependent
target implementation; changing the simulation save interval does not gate the
reservoir.

## Source map and invariant accounting

Each full step is a reservoir half-step, a coupled plasma step, and a second
reservoir half-step. For a source duration `h`,

\[
\alpha(\mathbf{x})=1-\exp[-h\,m(\mathbf{x})/\tau_R],\qquad
U_i'=(1-\alpha)U_i+\alpha U_{i,R}.
\]

Both electron distributions are translated into the new ion velocity frame
before mixing. This avoids interpreting a change of frame as injected heat.
Ion conservative-state mixing preserves positive density and internal energy
for valid endpoint states. Electron remaps and harmonic/current projections do
not guarantee positivity of the reconstructed angular distribution; retain the
kinetic positivity and resolution checks.

A spatially weighted relaxation of `B` would create magnetic divergence. Instead,
the source reconstructs the resolved periodic Coulomb-gauge potential difference
and applies

\[
\Delta\mathbf B=\nabla_h\times[\alpha(\mathbf A_R-\mathbf A)].
\]

The discrete curl is the same as Faraday's law. This preserves magnetic
divergence and the periodic mean flux. Mean and unresolved Nyquist components
are not driven; this source cannot inject arbitrary net flux through the box.
The derivatives of the source envelope contribute to the applied field. The
spectral potential construction is nonlocal and is another reason to perform
buffer-distance tests.

The current target includes the changed Ampere current. Existing current
residuals are damped only in the driven bands; a zero drive does not silently
repair a physical solver residual elsewhere. The source's current correction
belongs to its external momentum and energy budget.

Output retains the physical totals and saves cumulative measured additions:

- `reservoir_electron_number`, `reservoir_ion_number`;
- `reservoir_total_momentum`;
- `reservoir_electron_energy`, `reservoir_ion_energy`,
  `reservoir_magnetic_energy`, and `reservoir_total_energy`.

`reservoir_magnetic_field_change(t,x,y,component)` saves the cumulative actual
field increment from the source. Subtract its line-integrated flux when checking
Faraday residuals in driven runs; retain the raw flux history as well. This
records the discrete source without assigning it a fictitious physical EMF.

`source_accounted_total_energy` is physical total energy minus the existing
kinetic-current-projection work and measured reservoir energy injection.
`source_accounted_total_momentum`, `source_accounted_electron_number`, and
`source_accounted_ion_number` subtract their corresponding external injection.
These account for this source only: laser heating, filtering and discretization
errors are not removed. A flat budget is necessary but does not establish that
the imposed boundary model is physically appropriate.

## Verification

The source tests compare particle and momentum injection with the prescribed
reservoir, verify quasineutral mixtures and exact ion relaxation, check magnetic
divergence and mean flux, and verify the ledger survives the physical step.
These establish the implementation of the source, not a calibrated model of
MAGPIE inflow or outflow.
