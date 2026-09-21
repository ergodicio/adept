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
Use `scale` for a dimensional separable x/y shape. `sine` and `cosine` retain their
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
then recomputes the Ohm electric field. The coupled distribution is in the local
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
