# Vlasov-Fokker-Planck 2D Solver

VFP-2D is a 2D3P electron Vlasov-Maxwell-Fokker-Planck solver. Configuration space is a periodic Cartesian $(x,y)$ mesh; momentum space is represented by its radial coordinate and an arbitrary spherical-harmonic expansion,

$$
f(\mathbf r,p,t)=\sum_{\ell=0}^{\ell_{\max}}\sum_{m=-\ell}^{\ell}
f_\ell^m(\mathbf r,p,t)P_\ell^{|m|}(\cos\theta)e^{im\phi},
\qquad f_\ell^{-m}=(f_\ell^m)^*.
$$

The implementation follows equations (5)-(24) of Tzoufras et al., *Journal of Computational Physics* **230** (2011), 6475-6494, and equations (28)-(39) of Bell et al., *Plasma Physics and Controlled Fusion* **48** (2006), R37-R57 (the KALOS review).

## Harmonic storage

Only $m\geq0$ is stored. The retained modes satisfy

$$0\leq\ell\leq\ell_{\max},\qquad 0\leq m\leq\min(\ell,m_{\max}).$$

They occupy one packed harmonic axis rather than a nested Python dictionary. Output includes `ell(harmonic)` and `m(harmonic)` coordinates. Internally, Diffrax sees a real-valued final axis of length two; the operator restores complex values for the angular algebra. This avoids relying on experimental complex-state support while remaining compatible with JIT, `vmap`, autodiff, and array sharding.

## Physics

The solver includes:

- all $x$ and $y$ streaming couplings between $(\ell,m)$ modes;
- all three electric-force components, with the radial $G_\ell^m$ and $H_\ell^m$ operators;
- all three magnetic-field components, algebraic and tridiagonal in $m$ at fixed $\ell$;
- all three components of Maxwell's equations with $\partial_z=0$;
- a spectral initial Poisson solve;
- density-conserving implicit isotropic electron-electron collisions;
- the linearized Tzoufras anisotropic electron-electron and electron-ion operator for every retained $(\ell,m)$;
- spatially shaped inverse-bremsstrahlung or Maxwellian heating;
- distribution-function diagnostics for the scalar, vector, $f_2$ tensor, and Nernst moments used in kinetic Ohm's law.
- opt-in ideal-fluid ions coupled to ion-frame electrons through pressure feedback,
  finite-mass moment exchange, and magnetic force and mechanical work.

The default is non-relativistic, matching VFP-1D. With `grid.relativistic: true`, the radial coordinate is momentum in $m_ec$ units, streaming uses $v=p/\sqrt{1+p^2}$, current moments use $p^2v$, and initialization uses a Maxwell-Juttner distribution. The current collision operator is non-relativistic, so relativistic mode presently requires `terms.fokker_planck.active: false`.

## Time integration

Stationary-ion steps use collision half-steps around the kinetic/field update. The
field-solver hierarchy is:

- `maxwell`: physical explicit Vlasov–Maxwell with midpoint stepping.
- `ampere`: the same explicit update with the complete Ampere residual divided by a
  required `relative_permittivity >= 1`; Faraday remains unchanged.
- `oshun-implicit`: explicit Faraday and midpoint non-electric transport, followed by
  a local discrete kinetic-current response solve for the electric field. It updates
  the distribution through the electric-force operator without an `f1` projection.
- `kinetic-ohm`: an RK4 kinetic/Faraday step with the inertia-free generalized Ohm law
  and current-moment projection onto quasistatic Ampere's law.

The OSHUN-style response is not a fully implicit Maxwell integrator. Coupled-ion runs
use the symmetric hydro/source/kinetic/source/hydro composition described in the
[configuration reference](config.md). The origin derivative enforces the KALOS
regularity condition $f_\ell^m\sim p^\ell$. Unsharded spatial derivatives and field
curls are spectral on a periodic box; the sharded x derivative uses finite differences.

## Current limitations

- Coupled spatial boundaries are periodic; standalone `IonEuler2D` also supports outflow.
- Moving ions are opt in and require non-relativistic, unsharded, quasineutral
  `kinetic-ohm` evolution, no active hidden density gradient, and `lmax >= 1`, `mmax >= 1`.
- `oshun-implicit` is stationary-ion only and rejects spatial sharding.
- `kinetic-ohm` is inertia-free and uses a current-moment projection; OSHUN's implicit
  current response still leaves Faraday and transport explicit.
- Local electron–ion exchange is a weak-drift moment model, not a full finite-mass Landau operator.
- Atomic kinetics, ionization, and relativistic collisions are not implemented.
- Positivity of the full distribution is not guaranteed by a truncated harmonic expansion.
- Long-duration and production-scale moving-ion validation remain outstanding.

## Ion-fluid verification gates

`IonEuler2D` advances cell averages of
$(\rho_i,\rho_i u_x,\rho_i u_y,\rho_i u_z,\mathcal E_i)$ with MUSCL reconstruction,
HLLC fluxes, periodic or outflow boundaries, and SSP-RK2 time stepping. The following
components and acceptance tests are implemented:

1. **Gate 0a:** conservative Euler core; uniform-flow, smooth-advection, contact,
   conservation, and coordinate-rotated Sod tests.
2. **Gate 0b:** quantitative strong-shock, Sedov, translating isentropic-vortex, and
   magnetic-divergence benchmarks.
3. **Gate 1a:** conservative bulk advection, compression, shear, and frame acceleration
   for arbitrary spherical harmonics, with sparse angular couplings and a dense
   verification reference. See [ion-frame operators](moving_frame.md).
4. **Gate 1b:** finite-mass electron–ion temperature and momentum relaxation with
   measured equal-and-opposite ion updates, within the local weak-drift model.
5. **Gates 2a/2b:** opt-in coupled split, finite-mass frame remapping, full electron-pressure
   feedback, magnetic force/work, coupled invariant histories, frozen-ion regression,
   and quantitative Spitzer–Härm/Epperlein–Haines and Biermann local-limit tests.
6. **Nonlinear energy gate:** accounted-energy tolerance and timestep, spatial, and radial
   refinement tests for the periodic coupled benchmark. Projection work is saved separately.
7. **Magnetic flow milestone:** $\mathbf J\times\mathbf B$ force and
   $\mathbf u_i\cdot(\mathbf J\times\mathbf B)$ ion work, with source half-kicks
   around induction. Tests cover pressure and tension, discrete ideal-work balance,
   one-period Alfvén propagation of the ideal magnetic subsystem, the full coupled
   initial tension response, and finite-field radial energy convergence.

The default `vfp-2d` time loop uses stationary ions. Moving ions must be enabled
explicitly. Passing these local and nonlinear tests does not complete production-scale
validation; moving-ion parameter scans remain a future gate. The nonrelativistic
electric operator corrects its known interior radial work defect while retaining the
upper-tail flux and separate current-projection
accounting. A finite-field coupled check requires less than 0.1% **accounted** energy
defect relative to the transverse magnetic perturbation at its specified end time.
The raw energy and projection work remain separately visible. See the
[electric-work derivation and limits](electric_work.md) and [coupling details](moving_frame.md)
before choosing a resolution. These tests do not close the long-time energy gate.
Sustained driven/open boundaries, cooling/ionization, and convergence over experimental
flow times remain separate development gates.

See the [configuration reference](config.md), the [Joglekar 2014 reconstruction design](joglekar2014.md), and [`configs/vfp-2d/landau-damping.yaml`](../../../../configs/vfp-2d/landau-damping.yaml).
