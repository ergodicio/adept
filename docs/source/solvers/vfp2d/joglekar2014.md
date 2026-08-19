# Joglekar 2014 reconstruction and hydro coupling

This page is both a benchmark specification and an implementation boundary. The target is
A. S. Joglekar *et al.*, *Physical Review Letters* **112**, 105004 (2014),
[doi:10.1103/PhysRevLett.112.105004](https://doi.org/10.1103/PhysRevLett.112.105004).
The reduced heating configuration in `configs/vfp-2d/joglekar-2014-prl.yaml` exercises the
parts that are implemented today. It is not yet the fully implicit PRL reproduction.

## What the benchmark requires

The paper retained the Cartesian-tensor equivalent of all spherical harmonics through
$\ell=2$. Its generalized kinetic Ohm law was

$$
\mathbf E=\bar\eta\mathbf j+\frac{\mathbf j\times\mathbf B}{e n_e}
-\mathbf v_T\times\mathbf B
-\frac{\nabla(n_em_e\langle v^5\rangle)}{6en_e\langle v^3\rangle}
-\frac{\nabla\cdot(n_em_e\langle\mathbf{vv}v^3\rangle)}{2en_e\langle v^3\rangle},
$$

with

$$
\mathbf v_T=\frac{\langle\mathbf v v^3\rangle}{2\langle v^3\rangle}
+\frac{\mathbf j}{e n_e}.
$$

The last term is an $f_2$ pressure-anisotropy contribution and is essential at the X point;
an $f_0+f_1$ model is therefore insufficient. VFP2D now emits the exact scalar, vector, and
traceless tensor moments in these equations, together with $\mathbf j$, $T_e$, and
$\mathbf v_T$.

The published setup used:

- $T_{e0}=1.6$ keV and $n_{e0}=2.5\times10^{22}\,\mathrm{cm}^{-3}$;
- $v_{th}/c=0.08$, $\omega_{pe}\tau_n=125$, and $\lambda_{mfp}=0.34\,\mu$m;
- $-1500<x/\lambda_{mfp}<1500$ and $-100<y/\lambda_{mfp}<100$, with a uniform central
  region $-400<x/\lambda_{mfp}<400$ and stretched cells outside it;
- $\Delta x=13.3333\lambda_{mfp}$, $\Delta y=3.125\lambda_{mfp}$, and
  $\Delta v=0.0625v_{th}$;
- two Gaussian inverse-bremsstrahlung heating spots with $r_0=50\lambda_{mfp}$,
  $H_0=0.5$, and laser intensity $2.5\times10^{14}\,\mathrm{W/cm^2}$;
- a prescribed hidden density derivative $\partial_z n=(n_0/L_n)$ times the same spatial
  envelope, with $L_n=50\lambda_{mfp}$, switched off at $800\tau_n$;
- stationary ions, with the main comparison at $19000\tau_n$ and the final state at
  $27000\tau_n\simeq0.6$ ns.

The Letter does not state the spot separation or ionization in its five pages. Those values
must come from an original input deck, a longer manuscript, or a controlled inference; they
must not silently become “reference” values in ADEPT.

## Long-timescale field modes

The current explicit Vlasov--Maxwell integrator is useful for equation tests and kinetic
wave problems. It is not a viable route to $27000\tau_n$: its timestep is constrained by the
light-wave CFL condition and electron plasma oscillations, neither of which is part of the
slow collisional physics being studied. The PRL used a fully implicit VFP code.

The production deck uses Chang--Cooper differencing for $f_{00}$ because laser heating
drives the distribution away from a Maxwellian. The nominal 2 fs Strang step applies the
collision operator in 1 fs half-steps, about $0.045\tau_n$ for the published normalization.
The alternative log-mean flux has a zero semidiscrete energy derivative at the frozen state,
not exact energy conservation for a finite nonlinear implicit step. Both the collision
timestep and velocity grid must therefore be included in convergence scans.

The implemented `kinetic-ohm` mode takes the following quasineutral path:

1. enforce $n_e=Z n_i$;
2. obtain the required current from Ampere's law without displacement current,
   $\mathbf j=\nabla\times\mathbf B/\mu_0$;
3. evaluate the electric field from the complete moment equation above;
4. advance $\mathbf B$ with Faraday's law;
5. minimally project the bulk-current moment of $f_1$ onto the Ampere current while retaining
   its velocity-dependent heat-flux structure;
6. retain all five moment contributions as output diagnostics.

This removes the light-wave and plasma-frequency timestep restrictions and is an appropriate
first long-timescale benchmark model. It assumes negligible electron inertia and the small
contracted $\langle\mathbf{vvv}\rangle:\mathbf E$ term used to derive the displayed Ohm law.
The current projection also exchanges a small amount of kinetic energy with the unresolved
constraint system. Both effects must be measured in convergence tests.

The higher-fidelity `implicit-current` mode remains to be implemented. It should solve the
electric field as a Lagrange multiplier that makes the implicit kinetic current response agree
with Ampere's current. That retains electron inertia when requested and avoids using the
displayed Ohm law as an evolution closure. The solve should use matrix-free JAX
Jacobian-vector products and a Krylov method, with a custom VJP or implicit differentiation
rather than differentiating through every iteration.

## The 2.5D Biermann source

For gradients lying in the evolved $x$--$y$ plane, the Biermann field should emerge from the
quasineutral electric solve. The PRL geometry is different: it evolves $x,y$ but prescribes
$\partial_z n$. The implementation therefore needs an explicit `hidden_gradients.dndz`
field which enters the pressure-gradient part of the same Ohm residual. It must not be
implemented as an arbitrary magnetic source because doing so would lose the corresponding
electric field and electron response.

The source gate is smooth and differentiable during optimization but can reproduce the
published switch-off at $800\tau_n$ in validation mode. A direct $\nabla n\times\nabla T$
diagnostic should be saved in both modes.

## Recommended ion-fluid coupling

Use a quasineutral ion fluid and keep electrons kinetic in the local ion center-of-mass
frame. The minimum useful fluid state is $(n_i,n_i\mathbf u_i,\mathcal E_i)$:

$$
\partial_t n_i+\nabla\cdot(n_i\mathbf u_i)=0,
$$

$$
\rho_i D_t\mathbf u_i=\mathbf j\times\mathbf B
-\nabla\cdot\mathbf P_e-\nabla p_i+\mathbf R_{ext},
$$

with electron pressure and heat transport taken directly from $f_{\ell m}$, not from a
second electron-energy equation. The electron distribution must receive the equal and
opposite work and momentum exchange. Its update splits naturally into:

- conservative bulk advection by $\mathbf u_i$;
- relative electron streaming and electromagnetic harmonic coupling;
- velocity-space compression/inertial terms caused by $\nabla\mathbf u_i$ and acceleration
  of the ion frame;
- implicit collisions and laser heating.

For the first coupled milestone, an isothermal or adiabatic ion pressure law is adequate.
The acceptance tests should verify total particle number, total momentum, total energy, and
$\nabla\cdot\mathbf B$ across the kinetic--fluid exchange. A prescribed bulk velocity is a
useful operator test, but it should not be presented as the physical coupling.

## Delivery sequence

1. Frozen-ion, moment-projected kinetic-Ohm solve on a periodic uniform mesh. **Implemented.**
2. Published 2.5D hidden density gradient, IB profile, and five-term diagnostics. **Implemented
   as a reduced benchmark; full-resolution validation remains.** Reproduce magnetic generation,
   Nernst inflow, the five Ohm-law traces, and reconnection-rate history.
3. Fully implicit kinetic-current response and mapped/stretched $x$ mesh with non-periodic
   thermal boundaries for the full published box.
4. Conservative ion-fluid coupling and moving-ion comparison.
5. Higher $\ell_{max}$, convergence scans, and differentiable parameter inference.

The benchmark should first lock the published $\ell_{max}=2$ result, then demonstrate that
increasing $\ell_{max}$ changes observables by less than a stated tolerance. That distinguishes
reproduction from extension.
