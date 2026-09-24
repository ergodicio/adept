# FARSIGHT / Eulerian phase-space comparisons

Exploratory collisionless two-stream and NLEPW comparisons. Eulerian advection is
**spectral in x, cubic-spline in v**, with Strang time splitting. FARSIGHT uses
fixed or adaptive biquadratic panels, coupled RK4, and configurable remeshing.
The default remains fixed panels/direct fields/remeshing every step. The
production Eulerian solver is unchanged; the benchmark adapter explicitly sets
the analytic IC, uniform ions, initial electric field, and step-aligned saves.

All simulation runs require MLflow and upload/verify their configuration,
source snapshot, dependency versions, timing, distributions, and scalar
diagnostics. Credentials come from the existing environment, never CLI flags.
The prepared runtime is used for FARSIGHT; Eulerian retains its normal `ergoExo`
lifecycle. The campaign experiment is `farsight-comparison`.

## Force model and conservation

FARSIGHT uses a softened periodic kernel, whereas ordinary Eulerian Poisson does
not. Each case therefore has two Eulerian controls:

- `eulerian-poisson`: the ordinary Poisson model requested as the baseline.
- `eulerian-softened`: continuum Fourier coefficients of the exact FARSIGHT
  kernel, independently integrated to checked tolerances, to isolate numerical
  differences from force-model differences.

Neither an equal grid count nor a matching force model establishes convergence
or equal computational cost. `nx,nv` denote intervals for FARSIGHT and cells for
Eulerian. Velocity boundaries also differ; inspect tails, mass, negative mass,
and C2. Both solvers normalize their discrete initial mass once; neither clips
or renormalizes during evolution.

Native quadrature C2 is reported separately from display-grid quadrature and
the exactly integrated saved biquadratic polynomial. The renderer evaluates
the actual fixed-panel or adaptive-leaf interpolant on Eulerian cell centers,
with no scatter interpolation or time interpolation. For AMR it validates the
saved hierarchy IDs, active mask and complete disjoint partition, choosing
the finest leaf at shared boundaries. Moving/deformed saved frames are still
rejected: arrange frame saves to coincide with remeshes.

The polynomial mass/C2 diagnostic uses three-point Gaussian quadrature in each
coordinate, which exactly integrates a biquadratic and its square on rectangular
leaves. It is not the material/nodal quadrature invariant, a positivity check,
or a physical convergence estimate. Refinement/coarsening can change native
quadrature even when the represented polynomial is unchanged.

## Benchmark and run

Activate an existing ADEPT environment. From the repository root:

```bash
python -m examples.farsight_comparison.run \
  --case two-stream --solver farsight --phase pilot --benchmark \
  --nx 128 --nv 256 --tmax 1 --dt .05 --frame-dt .5 --epsilon .375 \
  --output /absolute/new/pilot-directory
```

The benchmark executes the same initial-value problem three times, separating
the compilation-inclusive first execution from two synchronized warmed solves.
Device memory is the process allocator high-water mark, not an isolated profiler
measurement. `timing.json` records exactly what was measured.

The scan uses **one Parsl LocalProvider worker**, preserving the allocation's GPU
visibility. On NERSC it must run on a compute node, not a login node. Both driver
and worker enforce that check. It does not request another allocation:

```bash
python -m examples.farsight_comparison.scan \
  --nx 128 --nv 256 --tmax 40 --dt .05 --frame-dt .5 --epsilon .375 \
  --output /absolute/new/movie-runs
```

Default cases: `two-stream nlepw`; solvers:
`farsight eulerian-softened eulerian-poisson`. A one-GPU development session can
run this through `session.sh exec`; use its isolated `outputs/` directory.
Measure cost at actual shapes before extending duration or resolution. This is
an interactive development workflow, not an unattended production launch.

### Explicit AMR/direct–treecode pair

```bash
python -m examples.farsight_comparison.scan \
  --task-file examples/farsight_comparison/amr-pilot.json \
  --output /absolute/new/amr-pilot
```

This preregistered short timing pilot uses base32x64, up to two refinement levels
(finest equivalent128x256), capacity4096, range threshold .05, fixed epsilon.375,
and tmax1. The direct and treecode tasks share every other numerical setting.
It is not a recommended production turbulence configuration. Capacity4096 can
overflow during nonlinear evolution; the full hierarchy can request8192 leaves.
Padding affects actual computation even with fewer active leaves. Measure at
the final capacity before choosing a duration.

Task files accept a nonempty list or `{ "defaults": {...}, "tasks": [...] }`.
Tasks execute with one Parsl worker, unique campaign-local output directories,
and distinct method names. `--amr-pair` generates a direct/tree pair per case;
`--eulerian-nx` and `--eulerian-nv` set independent reference resolutions.
AMR/treecode/remesh/quadrature options are available on both CLIs; see `--help`
and the [solver configuration reference](../../docs/source/solvers/farsight1d/config.md).

Inspect active/requested panels, capacity exhaustion, maximum-level saturation,
gap extension, geometry validity, negativity, and remap/regrid budgets. A finished
run is not necessarily tolerance-resolved. On analyzer rejection the tracked run
stays FAILED while `failed_final_state.npz`, `failed_scalars.nc` and
`failure_diagnostics.json` retain the invalid evidence.

The native field observation grid uses the **base** nx, not the finest AMR nx.
Compare spectra and electric-field energies only after evaluating both states
on a common sufficiently resolved x grid. Compare direct and treecode fields on
the **same** source state to isolate force approximation from trajectory and
refinement differences.

```bash
python -m examples.farsight_comparison.field_audit \
  --source /absolute/completed-amr-run --output /absolute/new/field-audit \
  --nx 128 --probe-count 257 --chunk-size 512
```

This creates a separate verified MLflow artifact run, recomputing direct and
configured treecode fields on identical saved source states at all frame times.
Off-grid probes at initial/final saves guard against an accidentally favorable
target grid. It records absolute and charge-normalized errors, relative errors
only above a documented reference-norm floor, source hashes, common-grid fields
and field-energy quadrature. There is no trajectory integration in this audit.

### Conditional uniform-grid control

The planned `amr-uniform-control.json` keeps the direct task's settings from
`amr-two-stream-movies.json`, changing only `amr_min_level` to 2 and the tracking
phase. It retains all 8192 finest leaves (uniform 128x256) in the same AMR code
path, with unchanged capacity and 10752 candidate panels. This tests sensitivity
to the adaptive pipeline at the same maximum resolution, not convergence or
regridding alone: coarse coverage, hanging interfaces and initial quadrature
normalization also change. Compare native and exact polynomial C2 separately;
the native regrid budget should vanish, but interpolation defects can remain.
Launch only after preregistration and checking runtime plus artifact-collection
margin in the existing allocation; match the original runtime/compiler environment:

```bash
python -m examples.farsight_comparison.scan \
  --task-file examples/farsight_comparison/amr-uniform-control.json \
  --output /absolute/new/amr-uniform-control
```

The candidate limit is 32768: raising the original hierarchy to level 3 would
request 43520 and is rejected. Possible subsequent directional controls are
uniform 128x512 or 256x256 with AMR `min_level=max_level=0`, capacity 16384 and
base intervals equal to the uniform resolution. Each doubles source slots and
roughly quadruples direct-field arithmetic; measure cost before launch. These
are proposed controls, not results or automatic follow-up runs.

### Experimental positivity control

`positivity-pilot.json` benchmarks the opt-in AMR Bernstein limiter at the actual
8192-panel capacity for tmax1. `positivity-two-stream.json` defines a matched
tmax40 **Simpson unlimited / Simpson limited** pair: every physical and numerical
option except `positivity_limiter` is identical. Do not compare directly against
the earlier trapezoid run and attribute all differences to limiting. Initial
polynomial limiting can also alter the sampled IC; its defects are logged apart
from remap defects. Both tasks use the direct field, leaving force approximation
out of this control.

```bash
python -m examples.farsight_comparison.scan \
  --task-file examples/farsight_comparison/positivity-pilot.json \
  --output /absolute/new/positivity-pilot
```

Run the full pair only after checking pilot validity, timing and memory against
the approved allocation. Inspect positive/negative native C2, negative mass,
initial and cumulative limiter defects, remap mass error, partition saturation
and gaps. Destination Bernstein scaling preserves panel mass with Simpson
quadrature but intentionally dissipates C2; the complete remap still need not
conserve mass. A positive solution or a smaller C2 drift alone is not a fidelity
or turbulence-readiness result. The Eulerian reference remains spectral-x and
cubic-spline-v.

Rendered diagnostics include a sufficient Bernstein positivity bound for every
saved rectangular panel. Nonnegative bounds certify the full polynomial to the
reported roundoff tolerance; negative bounds are inconclusive, not proof of
negative f. Nodal sign-split C2 is likewise a native quadrature diagnostic, not an
exact integral over the negative region of the polynomial.

## Render and log movies

Pull completed run directories locally or render where an existing ffmpeg is
available. Each rendering has its own MLflow run linking both simulation run IDs:

```bash
python -m examples.farsight_comparison.render \
  --eulerian /absolute/movie-runs/two-stream-eulerian-poisson \
  --farsight /absolute/movie-runs/two-stream-farsight \
  --output /absolute/new/two-stream-poisson-movie
```

Outputs: H.264 MP4, initial/midpoint/final contact sheet, conservation plot, and
JSON diagnostics. Shared color limits remain fixed across frames; the difference
panel is labeled a distribution difference, not an accuracy estimator.

## Validation and limits

```bash
python -m pytest tests/test_farsight_comparison -q
```

Tests cover common ICs/background/initial field, actual selected pushers, exact
times, softened Fourier coefficients against direct convolution, reconstruction,
encoder output, failure handling, provenance, and tracking verification.

Bump-on-tail and KEEN are not implemented by this example: FARSIGHT needs a
mixture initializer and stage-consistent external driving first. AMR/treecode
performance and late-time convergence require measured comparisons. Cold,
long-domain turbulence additionally needs its own initialization, spectral and
resolution tests; these warm single-mode movies do not establish that readiness.
