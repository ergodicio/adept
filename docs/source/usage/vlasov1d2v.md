# Vlasov-Poisson-Fokker-Planck 1D2V

To run the 1D2V solver, use the configs in `configs/vlasov-1d2v/`:

```bash
uv run run.py --cfg configs/vlasov-1d2v/epw.yaml
```

`solver: vlasov-1d2v` in the config file is what selects this module. The full
list of options is in the [configuration reference](../solvers/vlasov1d2v/config.md).

## Equations

The velocity space is **cylindrical**: $f = f(x, v_\parallel, v_\perp)$, with
$v_\parallel$ along the single spatial direction and $v_\perp$ the magnitude of the
two-dimensional perpendicular velocity, so the phase-space measure is
$d^3v = 2\pi v_\perp \, dv_\perp \, dv_\parallel$. Only $v_\parallel$ couples to the
electric field:

$$
\frac{\partial f}{\partial t}
+ v_\parallel \frac{\partial f}{\partial x}
+ E_x \frac{\partial f}{\partial v_\parallel}
= C[f]
$$

$$
\partial_x^2 \phi = 1 - \int f \, d^3 v, \qquad E_x = -\partial_x \phi
$$

The ions are static.

Much of the analysis is done on the **marginal**

$$
F(x, v_\parallel) = \int f \, 2\pi v_\perp \, dv_\perp ,
$$

which is initialized, saved, and (for the marginal-coefficient collision operators)
evolved identically to the corresponding `vlasov-1d` run.

### Collisions

Two families of collision operator are available through
`terms.fokker_planck.type`.

**Marginal-coefficient operators** (`dougherty`, `dougherty_nodrag`,
`lenard_bernstein`) act along $v_\parallel$ only, with drift and diffusion
coefficients computed from the marginal $F$:

$$
C[f] = \nu \frac{\partial}{\partial v_\parallel}
\left[ (v_\parallel - \bar v) f + \frac{1}{2\beta} \frac{\partial f}{\partial v_\parallel} \right],
\qquad (\bar v, \beta) \ \text{from} \ F .
$$

These conserve $n$, $P_\parallel$, and $E_\parallel$ to the same standard as the 1D
operator, and reproduce the 1D dynamics exactly for a separable
$f = F(v_\parallel) M(v_\perp)$.

**The full-geometry operator** (`cylindrical_landau`) is the linearized
Landau/Coulomb operator on the full $(v_\parallel, v_\perp)$ plane:

$$
C[f] = \nu \, \nabla_v \cdot \left[ \mathbf{D}(v) \cdot M \nabla_v (f / M) \right],
\qquad
M = \exp\!\left(-|\mathbf{v} - \mathbf{u}|^2 / 2T\right),
$$

with the anisotropic test-particle tensor split into a speed-diffusion channel and
a pitch-angle (Lorentz) channel,

$$
\mathbf{D} = \alpha_\text{speed}\, \hat D_\parallel(s)\, \hat{\mathbf{s}}\hat{\mathbf{s}}
           + \alpha_\text{lorentz}\, \hat D_\perp(s) \left(\mathbf{I} - \hat{\mathbf{s}}\hat{\mathbf{s}}\right),
\qquad
s = |\mathbf{v} - \mathbf{u}| / \sqrt{T},
$$

whose coefficients are the erf-exact Rosenbluth-potential coefficients of a
Maxwellian field species. The two channel weights are independently
configurable, which is the main reason to reach for this operator: it lets you
attribute an effect on the wave to pitch-angle scattering or to speed diffusion
separately. See `configs/vlasov-1d2v/epw-cylindrical-landau.yaml`.

The Krook operator is **not** implemented for this solver.

## Things You Might Care About

1. Infinite length (single mode) plasma waves — Landau damping, trapping
2. Finite length plasma waves — everything in 1. plus wavepackets
3. Wave dynamics on density gradients — 2. plus density gradients
4. How pitch-angle scattering versus speed diffusion modifies a trapped-particle
   wave, using the `cylindrical_landau` channel weights

---

## Configuration Options

### Density Profile

Uniform is easy. For a non-uniform profile, you have to specify the parameters of
the profile. The density profile can be parameterized as a sinusoidal perturbation
or a tanh flat top; see [initialization](initialization.md) for details on the tanh
flat-top parameters. The $v_\perp$ dependence is always a Maxwellian at the
component's `T0` (super-Gaussian shapes apply to $v_\parallel$ only).

### Velocity Grid

Beyond the 1D `nv`/`vmax`, you must set `nvperp` and `vperp_max` — they have no
defaults. How fine the perpendicular axis needs to be depends on the collision
operator; see
[choosing `nvperp`](../solvers/vlasov1d2v/config.md#choosing-nvperp).

### Ponderomotive Driver

You may want a driver to drive up a wave. The envelope is specified via a tanh
profile in space and in time, and the wave itself by its wavenumber, frequency, and
amplitude. Refer to the config files for more details.

### Collision Frequency

`terms.fokker_planck.time.baseline` sets $\nu$, and the `time`/`space` envelopes
shape it exactly as the driver envelopes shape the driver. Collisions will modify
the dynamics substantially depending on how far the distribution is driven from
Maxwellian.

### Diagnostics

`diag-vlasov-cumulative` and `diag-fp-cumulative` accumulate each term's
contribution to $\partial F / \partial t$ as a running **time integral**. Difference
them between save points to get exact interval-averaged rates — sampling a rate
instead would alias the $2\omega$ wave-particle energy exchange.
