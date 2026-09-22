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
