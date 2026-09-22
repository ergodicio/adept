# Periodic MAGPIE carbon-flow initialization

`configs/vfp-2d/magpie-carbon-periodic.yaml` initializes a reduced, fixed-charge
carbon plasma with antiparallel magnetic fields and counterstreaming ion flows.
It is a starting point for reconnection analysis and boundary-sensitivity studies,
not a reproduction of a MAGPIE shot or a replacement for the three-dimensional
GORGON experiment model.

Hare et al., *Phys. Rev. Lett.* **118**, 085001 (2017), Table I, report upstream
$n_e=3\times10^{17}\,\mathrm{cm}^{-3}$, $T_e=15$ eV, $\bar Z=4$,
$B=3$ T and $V_{in}=50$ km/s. The text gives an upper bound $T_i\lesssim50$ eV;
the example uses that upper bound. Their measured sheet half-width is
$\delta=0.6$ mm. The sheet itself has a different temperature, density and charge
state: those are observations to compare against, not imposed predictions.
[Primary paper and Table I](https://arxiv.org/pdf/1609.09234).
The measured half-length $L=7$ mm is described in
[Hare et al., *Phys. Plasmas* **25**, 055703 (2018)](https://arxiv.org/pdf/1711.06534).

Here **x is outflow, y is inflow and z is current**, swapping the in-plane labels
used in those papers. The 28 mm by 12 mm box, perturbation and numerical resolution
are modeling choices. The example runs only two 0.1 ps steps as an input/solver
smoke check. Extending it to the experimental hundreds of nanoseconds requires a
measured stability and cost study, timestep and velocity-space convergence, and
checks that the central analysis region is insensitive to periodic images.

## Spatial fields and units

The optional `initial_conditions` block accepts ion velocity and temperature:

```yaml
initial_conditions:
  ion_velocity:
    - 0km/s
    - {basis: periodic_tanh, axis: y, baseline: 0,
       amplitude: -50km/s, center: 0mm, width: 0.6mm, wavelength: 12mm}
    - 0km/s
  ion_temperature: 50eV
```

Velocity components are in x/y/z order; a mapping with `x`, `y`, `z` keys is also
accepted, and omitted components are zero. Numeric velocities retain the existing
code-velocity convention (normalized to $c$ for laser normalization). Numeric ion
temperatures multiply `units.reference ion temperature`. Physical strings are
converted directly. The old `terms.ion_fluid.initial_velocity` three-number list
continues to work; the new `initial_conditions.ion_velocity` overrides it.

Each scalar field accepts a uniform number/string, the existing profile syntax,
or `{scale: <physical amplitude>, profile: <dimensionless shape>}`. This also
applies to the electron `density.species-*.n` and `.T` fields. Their numeric
amplitudes multiply the reference electron density and temperature respectively.
Use `scale` for a dimensional separable x/y shape; physical amplitude strings
in both children are rejected. `sine` and `cosine` retain their
multiplicative convention $b[1+a\sin(kx)]$. The added `periodic_tanh` basis uses
an **additive** amplitude,

$$q(s)=b+a\tanh\left[\frac{\sin(k(s-s_0))}{kw}\right],\quad k=2\pi/\lambda,$$

so `baseline: 0` and a negative velocity amplitude give symmetric inward flows.
Near the central sheet it approaches $-V_{in}\tanh(y/w)$. Across the periodic seam
there is another transition, so this is not a pair of isolated inflow boundaries.

## Divergence-free magnetic fields and initial current

Specify a Cartesian vector potential and an optional uniform magnetic field:

```yaml
initial_conditions:
  magnetic_field:
    uniform: [0T, 0T, 1T]
    vector_potential:
      z:
        scale: 0.03T*mm
        profile:
          x: {basis: cosine, baseline: 1, amplitude: 1, wavelength: 28mm}
          y: {basis: cosine, baseline: 1, amplitude: 1, wavelength: 12mm}
```

The solver sets $\mathbf B=\nabla_h\times\mathbf A+\mathbf B_0$ using the same
periodic derivative as field evolution. Discrete $\nabla_h\cdot\mathbf B$ therefore
vanishes to floating-point precision. Both spectral and sharded finite-difference
stencils are supported for initialization; ion coupling still requires the
unsharded solver. Numbers are code units, with
$B_0=m_e/(e\tau)$ and $A_0=B_0L_0$; physical strings may use T and T*m.
Direct spatial B-component input is intentionally excluded so interpolated fields
cannot silently introduce magnetic divergence.

The carbon example instead uses this convenience initializer:

```yaml
initial_conditions:
  magnetic_field:
    periodic_sheet:
      field: 3T
      width: 0.6mm
      center: 0mm
      perturbation: 0.03T*mm
```

It constructs
$B_x=B_{up}\tanh[\sin(k_y(y-y_0))/(k_y\delta)]$, integrates it to periodic $A_z$,
adds $\delta A\cos[k_x(x-x_0)]\cos[k_y(y-y_0)]$, and takes the discrete curl.
Positive `perturbation` with positive `field` seeds a central X point. The optional
`x_center` defaults to the box center. `width` must span at least two y cells;
this rejection prevents a clearly unresolved initial sheet, but is not a
convergence claim. On the finite-difference path B differs from the analytic
profile by the derivative truncation error. Vector potential and `periodic_sheet`
are mutually exclusive. A uniform guide field may accompany either.

When `init_diffeqsolve` configures `kinetic-ohm`, it projects the initial electron
$f_1$ current onto the same quasistatic Ampere constraint used during evolution,
then recomputes the Ohm electric field. A field requiring transverse current
(`Jy` or `Jz`) needs `mmax >= 1`; initialization rejects a missing `(1,1)`
harmonic. Uniform or purely `Jx`-carrying fields pass this initial-current check
with `mmax=0`; subsequent angular dynamics still require convergence.
The coupled distribution is in the local
ion frame, so its relative electron current is the total current under
quasineutrality. This initial projection defines the initial state; the cumulative
current-projection work budget starts at zero. It preserves the isotropic
density and temperature but requires sufficiently small drift relative to the
resolved electron thermal distribution; check angular positivity and harmonic
convergence when imposing a stronger/narrower sheet.

## Importing an experimental or GORGON-derived plane

Export scalar fields to a numeric NPZ archive containing one-dimensional,
strictly increasing coordinate arrays `x`, `y`, and two-dimensional values with
shape **(len(x), len(y))**. No axis inference, coordinate sorting or extrapolation
is performed. Coordinate and value units are required in the config. Example
conversion after choosing the physical plane and axis mapping in the source:

```python
import numpy as np

# These arrays must already be in the stated physical units and x/y order.
np.savez("plane.npz", x=x_mm, y=y_mm, ne=ne_cm3, Te=te_eV,
         Ti=ti_eV, ux=ux_kms, uy=uy_kms, Az=az_tesla_metre)
```

```yaml
density:
  quasineutrality: true
  species-electron:
    m: 2
    n: {basis: file_xy, path: plane.npz, x_unit: mm, y_unit: mm,
        value_key: ne, value_unit: cm^-3}
    T: {basis: file_xy, path: plane.npz, x_unit: mm, y_unit: mm,
        value_key: Te, value_unit: eV}
initial_conditions:
  ion_temperature: {basis: file_xy, path: plane.npz, x_unit: mm, y_unit: mm,
                    value_key: Ti, value_unit: eV}
  ion_velocity:
    x: {basis: file_xy, path: plane.npz, x_unit: mm, y_unit: mm,
        value_key: ux, value_unit: km/s}
    y: {basis: file_xy, path: plane.npz, x_unit: mm, y_unit: mm,
        value_key: uy, value_unit: km/s}
  magnetic_field:
    vector_potential:
      z: {basis: file_xy, path: plane.npz, x_unit: mm, y_unit: mm,
          value_key: Az, value_unit: T*m}
```

`x_key`, `y_key` and `value_key` default to `x`, `y` and `values`. Paths resolve
against the process working directory. Bilinear interpolation requires the
source coordinates to cover all target cell centers. Incompatible units,
nonfinite data, shape mismatches and coordinates outside that coverage are
rejected. This explicit exchange format does not parse native GORGON output.
A spatially uniform B component belongs in `uniform`, since a strictly periodic
vector potential has zero mean curl. Ensure the supplied fields are compatible
with a periodic box or explicitly embed them within a modeled buffer region;
interpolation does not convert a measured open system into a periodic one.

Fixed Z, ideal single-fluid ions, no radiation/ionization model and the periodic
domain limit comparisons with the experiment. In particular, the observed change
from upstream Z=4 to sheet Z=6 cannot occur in this example. Use matched charge,
geometry and material assumptions when comparing with GORGON, and distinguish
assumed initial profiles from simulated observables.

## Outputs and reconnection-rate normalization

The existing postprocessing saves fields, current, ion primitives and sheet/topology
diagnostics in `binary/moments.nc`. Its `normalized_reconnection_rate` uses
$E_z(X)/(B_{up}v_{N,in})$, where `upstream_v_nernst_y` measures inward Nernst
transport. This is not a normalization by the imposed ion inflow: finite
counterstreams alone do not make this rate a bulk-flow MAGPIE observable.

When inward Nernst transport vanishes, the saved rate is NaN. The scalar
`reconnection_metrics` summary exports an unavailable final rate as zero (and a
zero peak if no finite rates exist), alongside validity metrics. A zero summary
therefore does not establish zero reconnection. Inspect `reconnection_valid`,
`rate_normalization_valid`, the rate's finiteness and the denominator together;
the normalization flag alone does not exclude a zero Nernst denominator.

For a bulk-inflow-normalized analysis, use the saved `ion_velocity`, `b` and `e`
to evaluate $E_z(X)/(B_{up}u_{i,in})$ with an explicitly chosen upstream sampling
region and inward-flow sign convention. Check the topology validity and a
nonzero bulk-inflow denominator separately from the Nernst validity flag. The
saved fields support this analysis; the existing normalized-rate summary does
not compute it.

## Initial numerical scales and a path to convergence

The following are **setup-only** calculations for the checked-in carbon deck
(`nx=ny=64`, `nv=48`, `lmax=mmax=2`). They use the initialized arrays and the
actual field/ion operators; they are not a stability result for a later heated,
compressed or plasmoid-containing state. The physical cell sizes are
$\Delta x=0.4375$ mm and $\Delta y=0.1875$ mm. The radial cutoff is
$v_{max}=1.2994\times10^7$ m/s, and $\Delta v=2.7071\times10^5$ m/s.

For the table, $k_{max}=\sqrt{(\pi/\Delta x)^2+(\pi/\Delta y)^2}
=1.8229\times10^4$ m$^{-1}$, $v_{last}$ is the largest sampled radial velocity,
and the initialized peak field is 3.0053 T including the seed perturbation.

| Initial scale | Value at $\Delta t=0.1$ ps | Interpretation |
| --- | ---: | --- |
| Spectral streaming $v_{last}k_{max}\Delta t$ | 0.02344 | Fastest spatial mode estimate for the sampled electron speeds |
| Electron gyromotion $\Omega_{ce}\Delta t$ | 0.05286 | $\Omega_{ce}=eB_{max}/m_e=5.286\times10^{11}$ s$^{-1}$ |
| Retained angular gyromotion $l_{max}\Omega_{ce}\Delta t$ | 0.1057 | Largest rotation frequency within the retained harmonics |
| Hall envelope $D_H k_{max}^2\Delta t$ | 0.001653 | $D_H=B/(\mu_0 e n_e)=49.76$ m$^2$/s at peak B |
| Resistive diffusion $D_\eta k_{max}^2\Delta t$ | 0.0003072 | $D_\eta=9.245$ m$^2$/s from the initialized kinetic-Ohm coefficient |
| Electric radial displacement $e|E|_{max}\Delta t/(m_e\Delta v)$ | 0.009730 | Includes the initialized bulk, pressure and other Ohm terms |
| Ion Euler half-step CFL limit at `cfl=0.4` | 0.8640 ns | The actual hydro half-step is only 0.00005 ns |

The Hall estimate is the uniform-background whistler envelope
$|\omega|\le D_H k^2$; it does not bound all modes of an inhomogeneous coupled
state. The resistive value is the frozen local coefficient
$c_{norm}^2\eta_{norm}L_0^2/\tau$ with
$\eta_{norm}=\texttt{resistivity_coefficient}/\langle v^3\rangle$.
The ion Euler CFL check includes ion sound and advection, not all coupled
magnetic/electron-pressure waves. Initial ion sound speed is 25.89 km/s;
including isothermal electron pressure gives the estimate
$\sqrt{(\gamma_iT_i+ZT_e)/m_i}=33.95$ km/s, and the peak Alfvén speed is
69.35 km/s. Evaluate these indicators again as B, density, pressure and velocity
change, and establish timestep convergence against a smaller step.

Collisions add an independent accuracy check. For the implemented Lorentz
$l=1$ electron-ion diagonal, evaluated at
$v=\sqrt{T_e/m_e}$, $\nu_{ei}=1.254\times10^{12}$ s$^{-1}$ and the collision
half-step has $\nu_{ei}\Delta t/2=0.06269$. At the first radial node that factor
is 108.3 because the diagonal scales as $v^{-3}$. Its implicit treatment removes
that diagonal's explicit stability restriction, but does not establish temporal
accuracy; the anisotropic electron-electron off-diagonal terms remain explicit.
Refine the collision timestep as well as the radial grid.

The 0.2 ps horizon covers only 0.0168 electron gyro-orbits. Ions at 50 km/s move
10 nm, compared with the 600 micrometre sheet half-width. It is therefore a
plumbing test: the sheet inflow time $\delta/V_{in}=12$ ns is 60,000 times longer.
Finite two-step output cannot demonstrate hydrodynamic reconnection or agreement
with the experimental outflow speed.

The spatial mesh has only 3.2 cells per measured sheet half-width and 16 per
measured half-length. The x domain extends to twice the experimental half-length
on each side, whereas the periodic y reversal is 6 mm from the central sheet.
Those box choices are not experimental boundary conditions. Compare successively
finer grids (for example 64, 128 and 256 cells per axis), then increase domain
sizes at fixed physical cell sizes to separate resolution effects from periodic
image effects. Compare sheet width, flux transfer, outflow and heating in a
fixed central physical region. The initial two-cell width rejection is much
weaker than this convergence requirement.

Velocity extent also needs attention before heating studies. The current cutoff
is $8\sqrt{15\,\mathrm{eV}/m_e}$, but only
$3.098\sqrt{100\,\mathrm{eV}/m_e}$ at the sheet electron temperature reported in
[Hare et al., Table I](https://arxiv.org/pdf/1609.09234). An untruncated isotropic
100 eV Maxwellian has **2.23% of its particle number and 8.74% of its kinetic
energy above this cutoff** (respectively the upper incomplete gamma fractions
$Q(3/2,a^2/2)$ and $Q(5/2,a^2/2)$, with $a=3.098$). A normalized truncated
distribution cannot recover those missing tails. To retain eight thermal
standard deviations at 100 eV, raise the config's `vmax` to at least
$8\sqrt{100/15}=20.66$; keeping the initial $\Delta v$ then needs at least 124
radial cells. This is an example extent calculation, not a converged choice.
Refine extent and spacing separately, and increase `lmax=mmax` through a
convergence sequence: 2, 4 and 6 retain 6, 15 and 28 packed complex harmonics.

At the current timestep, 100 ns requires one million steps and 250 ns requires
2.5 million. The initial arrays occupy 18.375 MiB in float64 real storage,
including the coupled state; 101 full snapshots alone occupy about 1.81 GiB
before file overhead. This excludes RK stages, transforms, collision workspaces,
compiler memory and diagnostic arrays. Dominant distribution storage scales as
$16N_xN_yN_vN_h$ bytes, and a rough workload count scales with
$N_tN_xN_yN_vN_h$ plus transform/collision costs. Benchmark a compiled short run
on the intended hardware before committing to a long run. Refinement in space,
velocity and harmonic order compounds the cost; no wall-clock claim follows
from these setup calculations.
# Review trigger: this page defines the physical-unit geometry contract for MAGPIE inputs.
