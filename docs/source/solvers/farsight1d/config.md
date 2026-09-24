# FARSIGHT-1D configuration

Use `solver: farsight-1d` with the [explicit runner](overview.md). Unknown fields,
nonfinite values, nonintegral counters and invalid ranges are rejected. Float64
must be enabled before preparation (the module CLI does this). Quantities use
normalized electron plasma units with `q = -1`, `m = 1`, mean density one.

| Section / key | Default | Meaning |
| --- | --- | --- |
| `grid.nx` | required | Even number of x intervals, at least 4 |
| `grid.nv` | required | Even number of v intervals, at least 2 |
| `grid.xmin` | 0 | Left periodic boundary |
| `grid.xmax` | required | Right boundary, greater than xmin |
| `grid.vmin`, `grid.vmax` | required | Increasing velocity bounds; zero inflow on remesh |
| `time.tmin` | 0 | Starting time |
| `time.tmax` | required | Ending time, greater than tmin |
| `time.dt` | required | Positive timestep; the time span must be a positive integer multiple |
| `initial.kind` | `maxwellian` | `maxwellian` or symmetric `two-stream` |
| `initial.thermal_speed` | 1 | Positive Gaussian standard deviation in v |
| `initial.drift` | 0 | Maxwellian mean; two-stream means ±drift, requiring drift > 0 |
| `initial.amplitude` | 0.01 | Density modulation A in `1 + A*cos(2*pi*mode*(x-xmin)/L)`, between -1 and 1 |
| `initial.mode` | 1 | Positive integer strictly below nx/2 |
| `numerical.epsilon` | required | Positive regularization length, in x units |
| `numerical.quadrature` | `trapezoid` | `trapezoid` or `simpson`; both assemble shared panel weights |
| `numerical.remesh_every` | 1 | Remesh every this many steps; 0 disables remeshing for diagnostic experiments |
| `numerical.chunk_size` | 64 | Positive target batch size for field and panel lookup |
| `save.scalars.every_steps` | 1 | Scalar observation stride |
| `save.fields.every_steps` | 10 | Fixed-grid electric field stride; `fields: null` disables |
| `save.distribution` | null | Enable with `{every_steps: N}` to save moving x, v and f |

All enabled schedules include step zero and the final step, even if the stride
does not divide the step count. Saves are at integer steps; observations never
interpolate between two different remeshed representations. The full final
state is retained independently of all observation strides. A final step not
divisible by `remesh_every` retains moving coordinates. `tmin` labels the initial
state supplied by the chosen initial condition; it does not evolve it beforehand.

The sampled initial distribution is normalized once to `sum(W*f) = L`; no
normalization is applied during evolution. Two-stream initialization uses equal
Gaussian beams. This normalization cannot resolve a narrow beam on an inadequate
velocity grid or restore truncated tails. There are no collisions, multiple
species, external fields, limiters, AMR or treecode configuration fields.

## Saved data

`scalars` is a dataset with dimension `t`:

| Variable | Definition |
| --- | --- |
| `mass`, `c2` | Reference/material quadrature sums `sum(W*f)`, `sum(W*f*f)` |
| `momentum`, `kinetic_energy` | Unit-mass sums `sum(W*f*v)`, `sum(W*f*v*v)/2` |
| `electric_energy`, `total_energy` | Fixed-x `L*mean(E*E)/2`, and its sum with kinetic energy |
| `min_f`, `negative_mass` | Minimum nodal value and `sum(W*max(-f,0))` |
| `remesh_count` | Number of executed remeshes |
| `remap_mass_change`, `remap_c2_change` | Sum of signed after-minus-before remesh changes |
| `remap_mass_abs_change`, `remap_c2_abs_change` | Sum of absolute per-remesh changes |
| `remap_uncovered_nodes` | Cumulative target-node count outside all panels, including legitimate zero inflow and the duplicated x endpoint |
| `invalid_panels` | Cumulative invalid panel count at remeshes |
| `max_panel_area_error` | Maximum absolute relative corner-polygon area error observed at any remesh |
| `valid` | Persistent validity flag; a false value makes host analysis fail |

`fields.electric_field` has dimensions `(t,x)` on nx unique periodic locations.
`distribution.x`, `.v`, and `.f` have dimensions `(t,x_node,v_node)` with
`(nx+1,nv+1)` nodes. x and v are **data variables** holding the current moving
coordinates. Neither node index is generally a physical coordinate. The x
endpoint is a duplicate with quadrature endpoint weighting.

See [conservation and resolution](overview.md#conservation-and-resolution) before
interpreting C2, energy or geometry diagnostics.
