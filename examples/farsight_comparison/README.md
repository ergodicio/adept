# FARSIGHT / Eulerian phase-space comparisons

Exploratory collisionless two-stream and NLEPW comparisons. Eulerian advection is
**spectral in x, cubic-spline in v**, with Strang time splitting. FARSIGHT uses
fixed biquadratic panels, coupled RK4, and remeshing after every step. The
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

Native quadrature C2 is reported separately from display-grid quadrature. The
renderer evaluates the actual fixed-panel biquadratic interpolant on Eulerian
cell centers, with no scatter interpolation or time interpolation. It rejects
AMR and moving/deformed saved frames; those need a separate reconstruction.

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

## Render and log movies

Pull completed run directories locally or render where an existing ffmpeg is
available. Each rendering has its own MLflow run linking both simulation run IDs:

```bash
python -m examples.farsight_comparison.render \
  --eulerian /absolute/movie-runs/two-stream-eulerian-poisson \
  --farsight /absolute/movie-runs/two-stream-farsight \
  --output /absolute/new/two-stream-poisson-movie
```

Outputs: H.264 MP4, initial/midpoint/final contact sheet, native mass/C2 plot, and
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
performance and late-time convergence require separate comparisons.
