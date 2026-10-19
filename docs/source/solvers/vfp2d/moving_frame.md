# Ion-frame electron operators

The moving-ion model represents the electron distribution in peculiar velocity
$\mathbf c=\mathbf v-\mathbf u_i(\mathbf x,t)$ while retaining laboratory position
and time. If $F(\mathbf x,\mathbf v,t)=f(\mathbf x,\mathbf c,t)$ and
$A_{ij}=\partial_j u_{i}$, the non-relativistic mixed-frame Vlasov equation can be
written in conservative phase-space form as

$$
\partial_t f
+\nabla_x\!\cdot[(\mathbf u_i+\mathbf c)f]
+\nabla_c\!\cdot\left[
  \left(\mathbf a_e-D_t\mathbf u_i-A\mathbf c\right)f
\right]
=C[f].
$$

Thus the terms added to the stationary-ion `TzoufrasVlasov` right-hand side are

$$
-\nabla_x\!\cdot(\mathbf u_i f)
+\nabla_c\!\cdot(A\mathbf c f)
+D_t\mathbf u_i\!\cdot\nabla_c f.
$$

The laboratory Lorentz force is evaluated with
$\mathbf E+\mathbf u_i\times\mathbf B$ before the existing peculiar-velocity
electric and magnetic operators are applied. This is the non-relativistic limit of
the mixed-frame fictitious-force terms derived by Schween and Reville,
[*MNRAS* **529**, 1970 (2024)](https://doi.org/10.1093/mnras/stae596), especially
their equations (5), (43), and (48).

## Gate 1a discretization

`IonFrameVlasov` supplies four equation-level operators:

- conservative periodic bulk transport of every $(\ell,m,v)$ cell;
- $\partial_j u_i$ computed with the same spatial derivative backend as VFP-2D;
- a conservative radial-velocity flux and angular Galerkin projection for
  $\nabla_c\cdot(A\mathbf c f)$; and
- frame acceleration through the existing arbitrary-harmonic electric-force operator.

The angular Galerkin construction uses ADEPT's unnormalized associated-Legendre convention and
projects the continuous product before truncating back to the configured harmonic
layout. This avoids the top-mode product error that occurs when two already-truncated
direction matrices are multiplied.

Gate 1a tests establish:

- angular reconstruction/projection round trips;
- conservative bulk transport and the $\partial_j u_i$ index convention;
- exact invariance of a uniform Maxwellian under constant Galilean translation;
- particle-conserving isotropic compression with $T_e\propto n_e^{2/3}$;
- the analytic pressure-anisotropy rate under prescribed trace-free strain; and
- cancellation between a uniform force and an oppositely accelerating frame.

Gates 2a/2b wire this implementation into the opt-in `CoupledIonKineticStep`. The
production deformation operator uses precomputed sparse angular couplings. Dense angular
quadrature remains a verification reference for the sparse operator.

## Gate 1b moment-exchange reference

`ElectronIonExchange` supplies the local finite-ion-mass relaxation check required by
Gate 1. It acts only on the moments needed by the acceptance tests:

$$
\frac{d\mathbf P_e}{dt}=-\nu_m\mathbf P_e,
\qquad
\frac{dT_e}{dt}=-\nu_T(T_e-T_i).
$$

The momentum correction is projected onto $f_1$ without changing $f_0$ or higher
harmonics. The thermal correction changes the $f_0$ energy moment while having zero
discrete density moment. Rather than trusting the requested rates analytically, the
operator measures the resulting electron momentum and energy rates with the same
quadrature used by VFP-2D and writes their exact negatives into the ion conserved
state. Tests cover both directions of temperature relaxation, all three momentum
components, density preservation, JIT execution, and machine-small exchange
residuals.

This remains a weak-drift moment-relaxation model, not a replacement for the full
finite-mass Landau collision operator. The source measures electron energy transfer in
the lab frame, including ion velocity dotted with the relative momentum transfer.
Both momentum and thermal exchange are enabled in the coupled split when their
configured rates are nonzero.

## Gates 2a/2b coupled evolution

`CoupledIonKineticStep` composes hydro half-step, coupled-source half-step, full kinetic
step at midpoint ion kinematics, coupled-source half-step, and hydro half-step. Each
source stage remaps the electron distribution into the updated ion frame with
`VelocityFrameRemap`; discrete corrections enforce the Galilean density, momentum,
and energy transforms. Remapping requires the (1,0) and (1,1) modes, so coupled runs
require `lmax >= 1` and `mmax >= 1`.

For evolving ions, each hydro half-step transports the electron distribution
with the **same HLLC mass flux and SSPRK2 stages** as ion mass. Face values of
`flm / rho_i` use a MUSCL reconstruction whose limiter is shared across all
real and imaginary harmonic/radial components. Thus a constant discrete
number moment of `flm / rho_i` remains constant at faces: initially neutral
cells receive matching electron and ion number fluxes. The periodic transport
conserves every globally integrated coefficient and transports existing charge
errors; it does not reset electron density to `Z * ni` after a step.

The kinetic stage disables its spectral bulk advection in this split. Relative
streaming, frame deformation and acceleration, collisions, and the Ampere
current constraint remain active. An enabled spatial filter acts on the field
and anisotropic harmonics while leaving the full `f00` radial spectrum
unchanged; filtering electron density independently would violate the shared
continuity update. Standalone and frozen-ion stepping retain their original
bulk-advection and filtering behavior. Finite velocity-domain losses and
non-number-conserving collision choices remain separate sources of charge error.

Spectral first derivatives also use a zero Nyquist multiplier on even grids,
matching the real-field curl operator. A complex harmonic coefficient stores
two real angular components; differentiating its unresolved Nyquist mode with
an imaginary multiplier would rotate those components and generate a spurious
density rate even when the projected current is a discrete curl. Odd-grid and
resolved Fourier modes retain their usual spectral derivatives.

The sources include full `f0 + f2` electron-pressure feedback and magnetic force/work.
Ion pressure work is $-\mathbf u_i\cdot\nabla\cdot\mathbf P_e$, while electron peculiar
pressure work $-\mathbf P_e:\nabla\mathbf u_i$ is supplied once by deformation in the
kinetic step. Their sum is a pressure-flux divergence; there is no extra local `f00`
pressure-energy correction. The lab electric field contains $-\mathbf u_i\times\mathbf B$.

Tests cover finite-mass frame invariants, coupled temperature behavior under a common
boost with a pressure gradient, local Spitzer–Härm/Epperlein–Haines and Biermann limits,
and nonlinear coupled energy under timestep, spatial, and radial refinement. The
Ampere projection's lab-frame work is recorded in `current_projection_energy` and
subtracted in `accounted_total_energy`.

The coupled path is periodic, quasineutral, non-relativistic, unsharded, and restricted
to `kinetic-ohm`. Active hidden density gradients are rejected because the ion equations
include only x/y derivatives. `frozen: true` suppresses hydro and coupled ion sources.
Production-scale validation, nonperiodic coupled boundaries, and a full finite-mass
collision operator remain future work; see the [configuration reference](config.md).

## Magnetic force and the energy budget

The quasistatic normalization is

$$
\mathbf J=c^2\nabla\times\mathbf B,\qquad
\mathcal E_B=\tfrac12c^2|\mathbf B|^2.
$$

`IonMagneticCoupling` uses the same discrete curl as Faraday's law and adds
$\mathbf J\times\mathbf B$ to ion momentum and
$\mathbf u_i\cdot(\mathbf J\times\mathbf B)$ to ion total energy. Magnetic
source half-kicks use the old and advanced fields, with midpoint mechanical work
and frame remaps. The electron step between them includes
$\mathbf E_{\rm bulk}=-\mathbf u_i\times\mathbf B$. On a periodic grid the ideal
magnetic and mechanical work cancel under the discrete curl summation identity.
There is no opposite electron heating term for magnetic ion work: its reservoir
is the magnetic energy evolved by induction.

Collision half-steps receive the evolving midpoint ion number density. With
`frozen: true`, both the coupled call and its exchange map skip all ion sources;
the coupled call also skips ion transport.

The returned distribution satisfies the Ampere current constraint after the final
frame change. This projection's lab energy is added to
`current_projection_energy`. Diagnostics retain both `total_energy` and
`accounted_total_energy = total_energy - current_projection_energy`; the latter
does not imply that the projection is physically energy conserving. Number and
quasineutrality diagnostics are measured without repairing density.

## Verification scope and remaining energy gate

`tests/test_vfp2d/test_magnetic.py` verifies magnetic pressure and tension, periodic
work cancellation for both derivative backends, ion internal-energy preservation
under a magnetic kick, unchanged electron lab energy under the accompanying frame
translation, and an actual nonzero ion response in the coupled solver. A separate
ideal magnetic subsystem test propagates a circularly polarized Alfvén wave for
one period; it verifies the force/induction split without claiming a full kinetic
Alfvén benchmark.

The original centered-electric-operator diagnostic, retained as a controlled
comparison with `conserve_electric_work=False`, initializes
$\mathbf B=(0.2,0.01\cos x,0.01\sin x)$ on a $12\times4$ periodic grid with
$c=5$, uniform density and pressure, and runs to normalized $t=0.5$. Its energy
scale is the transverse magnetic perturbation, not the much larger background
thermal or guide-field energy. At $n_v=32$, halving $\Delta t$ from 0.01 to 0.005
leaves the accounted defect close to 6.74% of that perturbation. Increasing
$n_v$ to 64 reduces it to 1.69%. The declared regression gate uses $n_v=96$,
$\Delta t=0.005$, requires less than 1% accounted defect and at least eightfold
improvement over $n_v=32$, and separately checks projection work and
quasineutrality. At that resolution the measured raw, accounted, and projection
contributions are 2.954%, 0.742%, and 2.212% of the perturbation energy,
respectively. This is a bounded radial convergence requirement, not a universal
resolution recommendation or long-time energy acceptance criterion. The end time
is only about 0.8% of a guide-field Alfvén period for this initial state; the
full-period wave test above applies to the ideal magnetic subsystem alone.
The default nonrelativistic operator now removes the analytically identified
interior electric-work defect; see the [derivation and longer coupled check](electric_work.md).

Longer finite-field kinetic runs must establish temporal, radial, spatial, and
angular convergence on the relevant energy-transfer and flow timescales. Driven
boundaries also need an explicit source/escape budget. Current projection,
finite-radial-grid work errors, positivity, and the fluid-ion approximation remain
material limitations for experiment design. The existing initial ion CFL check
uses Euler sound/flow speeds; it does not establish stability for Alfvén,
magnetosonic, or Hall induction timescales. Those require explicit timestep
verification for each magnetized configuration.
