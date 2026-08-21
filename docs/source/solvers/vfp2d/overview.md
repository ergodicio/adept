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
- the linearized Tzoufras anisotropic electron-electron and electron-ion operator for every retained $(\ell,m)$.
- spatially shaped inverse-bremsstrahlung or Maxwellian heating;
- distribution-function diagnostics for the scalar, vector, $f_2$ tensor, and Nernst moments used in kinetic Ohm's law.

The default is non-relativistic, matching VFP-1D. With `grid.relativistic: true`, the radial coordinate is momentum in $m_ec$ units, streaming uses $v=p/\sqrt{1+p^2}$, current moments use $p^2v$, and initialization uses a Maxwell-Juttner distribution. The current collision operator is non-relativistic, so relativistic mode presently requires `terms.fokker_planck.active: false`.

## Time integration

Each step uses collision half-step / midpoint kinetic-field step / collision half-step. In
`maxwell` mode the middle step advances explicit Vlasov--Maxwell. In `kinetic-ohm` mode it
evaluates the inertia-free generalized Ohm law, advances Faraday's law, and projects the
current moment onto quasistatic Ampere's law. The origin derivative enforces the KALOS
regularity condition $f_\ell^m\sim p^\ell$. Spatial derivatives and field curls are spectral
on a periodic box.

## Current limitations

- periodic spatial boundaries only;
- stationary ions represented by a prescribed neutralizing background;
- no atomic kinetics or ionization;
- relativistic collisions are not yet implemented;
- positivity is not guaranteed by a truncated spherical-harmonic expansion.
- `kinetic-ohm` is inertia-free and uses a current-moment projection; a fully implicit kinetic-current response is not yet implemented;
- moving-ion fluid coupling is not yet implemented.

## Ion-fluid development phases

The first moving-ion component is available as the standalone `IonEuler2D` finite-volume
operator. It advances cell averages of
$(\rho_i,\rho_i u_x,\rho_i u_y,\rho_i u_z,\mathcal E_i)$ with MUSCL reconstruction,
HLLC fluxes, periodic or outflow boundaries, and SSP-RK2 time stepping. Keeping the
operator separate from the kinetic split initially makes its conservation and shock
tests auditable before electron pressure work or collisional exchange are introduced.

Development follows the verification gates in the kinetic-electron / ion-fluid research
plan:

1. **Gate 0a (implemented):** conservative Euler core; uniform-flow, smooth-advection,
   contact, conservation, and coordinate-rotated Sod tests.
2. **Gate 0b:** strong-shock, Sedov, isentropic-vortex, and magnetic-divergence tests,
   followed by configuration and diagnostic integration.
3. **Gate 1:** conservative bulk advection and the compression, shear, and inertial
   terms for spherical harmonics in the ion peculiar-velocity frame.
4. **Gate 2:** equal-and-opposite kinetic/fluid exchange and coupled local-limit tests.

The production `vfp-2d` time loop still uses stationary ions until Gates 0b--2 land.

See the [configuration reference](config.md), the [Joglekar 2014 reconstruction design](joglekar2014.md), and [`configs/vfp-2d/landau-damping.yaml`](../../../../configs/vfp-2d/landau-damping.yaml).
