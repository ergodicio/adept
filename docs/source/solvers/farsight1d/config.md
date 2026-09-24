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
| `numerical.positivity_limiter` | `none` | `none` or experimental `bernstein`; the latter requires AMR and Simpson quadrature |
| `numerical.remesh_every` | 1 | Remesh every this many steps; 0 disables remeshing for diagnostic experiments |
| `numerical.chunk_size` | 64 | Positive target batch size for field and panel lookup |
| `numerical.field_solver` | `direct` | `direct` or `treecode`, used for all RK stages and field observations |
| `numerical.treecode.degree` | 8 | Interpolation polynomial degree, 1–32 (degree+1 nodes) |
| `numerical.treecode.theta` | 0.5 | Opening ratio in [0,1); 0 dispatches to exact direct evaluation |
| `numerical.treecode.leaf_size` | 32 | Positive source capacity per direct leaf |
| `amr.enabled` | false | Use packed adaptive leaves; fixed-grid path is unchanged when false |
| `amr.max_level` | 1 | Maximum quadtree depth, 0–4; 0 is the fixed root partition |
| `amr.min_level` | 0 | Force refinement to at least this depth, no greater than max_level |
| `amr.max_panels` | 256 | Static leaf-slot capacity; overflow fails explicitly |
| `amr.atol`, `amr.rtol` | 0.05, 0 | Nonnegative range threshold `atol + rtol*max(abs(f))` |
| `amr.max_gap_fraction` | 0.01 | Positive maximum interior extension distance in `(x/L,v/velocity_span)` |
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
species or external fields. Positivity limiting is opt-in, as described below.

Treecode options are validated even when the direct evaluator is selected, but
only affect evaluation for `field_solver: treecode`. A positive-theta tree walk
is not reverse-mode differentiable; its prepared capability says so. Use direct
evaluation or theta=0 for reverse-mode objectives. Force approximation can add
momentum drift; compare fields and trajectories against direct sums while
varying degree/theta, independently of AMR tolerances and softening.

AMR initially samples all candidate nodes, normalizing their finest-level
quadrature before threshold decisions, then normalizes the selected partition
once to mass L. Later remeshes never normalize. At least
`(nx/2)*(nv/2)*4**min_level` slots are required; a more refined initial selection
can require more and is checked before execution. The full candidate hierarchy
is limited to 32768 panels. These bounds are allocation guards, not accuracy
guarantees. Refinement and balancing iterate to a fixed point; newly created
leaves are checked against the same range criterion.

## Experimental positivity limiting

Set `numerical.positivity_limiter: bernstein` together with `amr.enabled: true`
and `numerical.quadrature: simpson`. The default `none` retains the original
method. The selected initial AMR distribution is limited **after** its one-time
mass normalization; initialization defects are recorded separately from all
evolution budgets. No normalization is performed after limiting.

At every remesh, two distinct stages are applied:

1. Each deformed source polynomial is affinely rescaled around its material
   Simpson mean, using the minimum over **all actual candidate query locations**
   assigned to that source. These locations include allowed interior-gap
   extensions; exterior zero-inflow samples remain zero. This makes the candidate
   samples nonnegative to roundoff. It does not certify the polynomial between
   queries or its advected physical-coordinate fit.
2. After selecting the new leaf partition, each rectangular destination
   biquadratic is rescaled around its mean until all tensor Bernstein coefficients
   are nonnegative to roundoff. Bernstein coefficients bound the polynomial
   throughout that rectangle, not merely at its nine stored nodes. This
   sufficient condition can limit an already nonnegative polynomial, so it may
   be more dissipative than an exact minimum test.

For either stage, the affine form is `p_limited = mean + theta*(p - mean)`,
with `0 <= theta <= 1`. On a destination rectangle, Simpson quadrature integrates
the biquadratic exactly, so this preserves both its panel mean and native panel
mass to roundoff. Negative or nonfinite panel means cause an explicit failure:
they cannot be made positive while preserving mass. These guarantees do not
extend to trapezoid quadrature, hence the validation restriction. Independently
limited panels can have different traces at a shared geometric edge.

This is **not a fully conservative remap**: interpolation and AMR partition
changes can still change mass. The source material mean is not generally the
physical mean of a deformed fitted polynomial. Nor is this a C2 repair:
destination limiting cannot increase either its native nodal C2 or its exact
rectangular-polynomial C2, to roundoff. Source rescaling cannot increase the
source **material** C2 norm, but the reported `source_limiter_c2_change` is measured
on resampled previous-layout candidate values and can have **either sign**.
Native nodal C2 differs from the exact integral of the squared polynomial; the
destination defects in both measures are reported separately. Positivity is certified for the
initial and freshly remeshed rectangular representation, not for all intermediate
advected fits, particularly when `remesh_every` exceeds one or is zero.

## Saved data

`scalars` is a dataset with dimension `t`:

| Variable | Definition |
| --- | --- |
| `mass`, `c2` | Reference/material quadrature sums `sum(W*f)`, `sum(W*f*f)` |
| `positive_mass`, `negative_mass` | Nodal sums `sum(W*max(f,0))`, `sum(W*max(-f,0))`; their difference is mass |
| `c2_positive`, `c2_negative` | Nodal sums `sum(W*max(f,0)^2)`, `sum(W*min(f,0)^2)`; their **sum** is C2 |
| `momentum`, `kinetic_energy` | Unit-mass sums `sum(W*f*v)`, `sum(W*f*v*v)/2` |
| `electric_energy`, `total_energy` | Fixed-x `L*mean(E*E)/2`, and its sum with kinetic energy |
| `min_f`, `negative_node_count` | Minimum active nodal value and unweighted count of negative stored nodes; edge duplicates count separately |
| `remesh_count` | Number of executed remeshes |
| `remap_mass_change`, `remap_c2_change` | Sum of signed after-minus-before remesh changes |
| `remap_mass_abs_change`, `remap_c2_abs_change` | Sum of absolute per-remesh changes |
| `remap_uncovered_nodes` | Cumulative target-node count outside all panels, including legitimate zero inflow and the duplicated x endpoint |
| `invalid_panels` | Cumulative invalid panel count at remeshes |
| `max_panel_area_error` | Maximum absolute relative corner-polygon area error observed at any remesh |
| `valid` | Persistent validity flag; a false value makes host analysis fail |

With AMR enabled, additional scalar variables are saved:

| Variable | Definition |
| --- | --- |
| `active_panels`, `requested_panels` | Packed leaf count and requested count before capacity check |
| `capacity_exceeded` | Persistent overflow flag; invalidates the result |
| `refinement_limited_panels` | Current maximum-level leaves still exceeding the range threshold |
| `regrid_mass_change`, `regrid_c2_change` | Cumulative quadrature-change portion of the total `remap_*` budget |
| `remap_gap_nodes` | Cumulative candidate-node evaluations extended across interior polygon gaps |
| `max_gap_fraction` | Largest recorded interior extension in the fixed normalized phase-space metric |

AMR uncovered/gap counts include every candidate node evaluation, including
duplicates and candidates not ultimately selected. They are diagnostics of
reconstruction work, not unique-node counts or lost mass. `min_f` excludes
inactive slots. Changing AMR quadrature changes C2; it is part of the reported
total remap defect, not an invisible correction.

The sign-resolved diagnostics are native **nodal** quadrature measurements, not
integrals over the positive and negative regions of the interpolating polynomial.
Nonnegative stored values do not rule out undershoot between nodes. Since the
negative contribution to C2 is positive, nearly unchanged total C2 can hide
simultaneous smoothing and negative undershoot; it does not establish positivity.

With `positivity_limiter: bernstein`, the following additional scalars separate
limiter effects from the existing remap defect:

| Variable | Definition |
| --- | --- |
| `initial_positivity_mass_change`, `initial_positivity_c2_change` | Native after-minus-before initial limiting; excluded from evolution budgets |
| `initial_positivity_polynomial_c2_change` | Exact rectangular-polynomial C2 change from initial limiting |
| `initial_positivity_limited_panels`, `initial_positivity_failed_panels`, `initial_positivity_min_theta` | Initial limiter activity, failures, and minimum rescaling factor |
| `interpolation_mass_change`, `interpolation_c2_change` | Cumulative raw candidate interpolation defect measured on the previous leaf layout |
| `source_limiter_mass_change`, `source_limiter_c2_change` | Cumulative limited-minus-raw candidate defect on the previous leaf layout; not the physical source integral |
| `destination_limiter_mass_change`, `destination_limiter_c2_change` | Cumulative native defect from limiting the selected destination rectangles |
| `destination_limiter_polynomial_c2_change` | Cumulative exact rectangular-polynomial C2 change from destination limiting |
| `source_limiter_panels`, `destination_limiter_panels`, `positivity_failed_panels` | Cumulative panel activity/failure counts during evolution |
| `source_limiter_min_theta`, `destination_limiter_min_theta` | Smallest rescaling factors during evolution; one means unchanged |
| `min_bernstein_coefficient` | Latest initial/remeshed rectangular coefficient minimum; not a certificate for a subsequently advected fit |

For each native moment (`mass` or `c2`), the cumulative signed budget is
`remap = interpolation + source_limiter + regrid + destination_limiter`, up to
roundoff. Here `regrid` compares the old and new leaf layouts using the already
source-limited samples. The separate exact polynomial C2 quantities are **not**
additional terms in this native nodal identity. All initial limiter quantities
remain distinct because step-zero observations already contain the limited state.
Analysis rejects an invalid limiter result instead of silently clipping values,
renormalizing mass, or treating its diagnostics as a successful solve.

`fields.electric_field` has dimensions `(t,x)` on nx unique periodic locations.
`distribution.x`, `.v`, and `.f` have dimensions `(t,x_node,v_node)` with
`(nx+1,nv+1)` nodes. x and v are **data variables** holding the current moving
coordinates. Neither node index is generally a physical coordinate. The x
endpoint is a duplicate with quadrature endpoint weighting.

With AMR, distribution variables `x`, `v`, `f`, `weights` instead have dimensions
`(t,panel,node)`, with `panel=max_panels` and nine nodes. `active`, `panel_id`,
and `level` have dimensions `(t,panel)`. Slots can change identity at remesh;
use `panel_id` (the hierarchy index) and the mask when comparing frames. Inactive
IDs/levels are -1. The complete final NPZ state includes this topology and
quadrature information even when distribution snapshots are disabled.

See [conservation and resolution](overview.md#conservation-and-resolution) before
interpreting C2, energy or geometry diagnostics.
