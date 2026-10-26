# CUDA/MPI Vlasov split-step prototype

A standalone, FP64, 1D1V Vlasov–Poisson benchmark targeting A100 GPUs. It does
not require AD or import ADEPT. The optional FFTW backend exists for numerical
validation; it is not a threaded CPU performance implementation.

The numerical implementation now lives in
[`adept/_vlasov1d/native`](../../adept/_vlasov1d/native/README.md). Its optional
ADEPT backend retains Python initialization, custom diagnostics, MLflow and
xarray/NetCDF output. The standalone benchmark shares its C++ core. Controlled
ADEPT/JAX versus CUDA measurements are preserved in
[`results/2026-10-24-adept`](results/2026-10-24-adept/summary.json).

## Latest turbulence target and halo mode

The reproduction target is the **23 October Nv4096 campaign**, not the older
L8000 cases: Nx134400 and Nx268800, Nv4096, L12600, vmax15, dt0.02, sixth-order
Hamiltonian splitting, exponential/spectral x advection, `cubic-spline` velocity
advection, and Chang-Cooper Dougherty collisions at nu1e-12. The pinned runtime
is ADEPT `ddd70b957dc3df85f0b5ec84af927c25d09d24e0` and JAX 0.10.2. The exact
member configs and source identifiers are under
[reference/latest-nv4096](reference/latest-nv4096/target.json).

**Correction to the original benchmark's method assumption:** ADEPT's
`VelocityCubicSpline` calls interpax's default `method="cubic"`, a **local C1
Hermite interpolant**, with slopes averaged from adjacent secants and one-sided
endpoint slopes. It does not use the global natural C2 spline implemented in
the first benchmark. Consequently, preserving the actual velocity method does
not require full velocity lines to reside on one rank.

`--velocity cubic --layout halos` keeps the velocity-slab layout for all six
Vlasov kicks. It communicates just the velocity rows required by the departure
points and the four-point local stencil. The active width is computed from the
maximum field, kick duration, and dv, with two extra rows for the stencil.
Requests may span several neighbouring ranks. If a departure exceeds the
allocated `--halo-capacity`, the program fails explicitly. Both signed values
and the original extrapolation value `1e-30` are retained; no positivity limiter
or mass normalization changes the interpolant.

Each rank owns a velocity slab and the complete x axis, so the batched x FFTs
stay local. The Poisson solve reduces the density array across velocity ranks.

`--integrator sixth` uses ADEPT's actual force-gradient coefficients and dt
corrections, with its six velocity/five streaming stages. `--cells` selects the
production cell-centred velocity grid and rectangular velocity quadrature.
`--nu` enables the production Dougherty moment closure, Chang-Cooper weights,
zero-flux endpoints, and delta-form implicit solve. Its self-consistent-beta
iteration is disabled, matching the target configs. **One transpose pair per
timestep remains for collisions**; the Vlasov halo path removes the other six
pairs. Spectral x streaming is retained.

For an exact native-state comparison, import raw float64 arrays with
`--input` (v,x order), `--v-grid`, and `--background`. Import the original ion
background; replacing its seeded density profile with a constant would change
the simulation. `checkpoint_reference.py` verifies and exports a committed
native checkpoint and advances the pinned ADEPT operators. It writes only to
its own output directory. `compare_checkpoint.py` checks full-resolution f,
Poisson fields and the latest whole-box N observer (periodic Gaussian sigma3,
potential depths/vb², strict cuts0.02 and0.5). Neither script modifies an
existing production controller or MLflow parent.

`distributed_reference.py` advances the same verified prepared state with
eight JAX processes on two nodes. This is needed for Nx268800 on 40 GB GPUs:
the pinned JAX collision solve could not allocate its gtsv2 workspace on four.
The eight-process reference uses one GPU per rank and the production NCCL
Libfabric plugin; its provenance and network log are preserved with the results.
This reference uses all numerical operators, including nu1e-12 collisions.

Omitting `--checkpoint` from the reference exporter uses the pinned project's
original t0 initializer, including both seeded, band-limited beams. This provides
an exact initial-state input for a future full-duration reproduction; the C++
benchmark's default Maxwellian initializer is not the turbulence initializer.

`validate_local.py` checks interpax C1 parity, signed values, physical endpoints,
multi-neighbour shifts, the sixth-order step and collisions on both layouts:

```sh
python benchmarks/vlasov_cuda/validate_local.py /path/to/vlasov_bench

# Example latest-case invocation after checkpoint_reference.py exports input:
mpiexec -n 8 /path/to/vlasov_bench --nx 134400 --nv 4096 \
  --length 12600 --vmax 15 --dt .02 --cells --integrator sixth \
  --velocity cubic --layout halos --nu 1e-12 --steps 100 --warmup 0 \
  --input /path/to/reference/initial.bin --v-grid /path/to/reference/v.bin \
  --background /path/to/reference/background.bin
```

Short-window agreement is an implementation check. It does not establish the
full t0–2000 decay trajectory, resolution convergence, or equivalence of another
spatial/velocity discretization. The earlier natural-spline/Strang results below
are preserved as the initial experiment, not a matched turbulence performance
comparison. Their exact source is archived in
[baseline-source.tar.gz](results/2026-10-24/baseline-source.tar.gz).

## Results using the latest turbulence operators

On Perlmutter A100-SXM4-40GB GPUs, the halo path gives the following medians
over three repeats, each with 10 measured steps after 3 warmup steps. Both
layouts use the same native t600 input and all operators listed above. Timings
exclude initialization, diagnostics and I/O; communication/computation do not
overlap. These compare two CUDA layouts, not CUDA against JAX.

| Nx × Nv | Nodes | GPUs | Full transpose (ms/step) | Halos (ms/step) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 134400 × 4096 | 1 | 4 | 323.5 | 174.3 | 1.86× |
| 134400 × 4096 | 2 | 8 | 490.5 | 140.0 | 3.50× |
| 268800 × 4096 | 1 | 4 | 649.6 | 342.5 | 1.90× |
| 268800 × 4096 | 2 | 8 | 965.0 | 275.0 | 3.51× |

Maximum per-rank distribution bytes sent fall by 85.1–85.3%. One full transpose
pair remains for collisions. On eight GPUs, MPI still takes about 60 ms on the
smaller grid and 119 ms on the larger grid, so that remaining communication
limits further strong scaling.

All 20 native-state comparisons pass across both layouts and GPU placements.
Through 100 steps (t600→602), maximum absolute f error is `6.73e-13`, relative
L2 f error is at most `9.52e-14`, and relative L2 E error is at most `5.64e-14`.
The exact latest whole-box hole counts match at both thresholds in every case:
at t602, 129/47 for Nx134400 and 129/50 for Nx268800. Another 84 CPU/CUDA/MPI
operator and initial natural-spline regression checks pass. Conservation diagnostics
are recorded without clipping f or renormalizing mass.

Reports, commands, runtime versions, source/executable hashes and the pinned
reference provenance are in
[results/2026-10-24-halos](results/2026-10-24-halos/summary.json).
`results/2026-10-24-halos/summarize.py` rebuilds the compact report from the
adjacent raw JSONL and comparison files. Native distributions stay on scratch;
their exact locations and manifest hashes are in the reference provenance.
The owned development allocations were released after saving the results.
This is a short-window implementation check; the full t0–2000 decay trajectory
has not been reproduced yet.

## Initial natural-C2/Strang benchmark: 24 October 2026

On Perlmutter A100-SXM4-40GB GPUs, a `32768 x 8192` FP64 distribution
(2 GiB of state), with 20 timesteps after 3 warmup steps, gave the following
medians over three runs. All runs use the same global problem size.

| Nodes | GPUs | Step (ms) | Streaming (ms) | Spline (ms) | Pack/unpack (ms) | Slab MPI (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 68.91 | 49.02 | 17.81 | 0.00 | 0.00 | 1.00x |
| 1 | 2 | 58.70 | 24.54 | 12.88 | 6.34 | 13.78 | 1.17x |
| 1 | 4 | 36.95 | 12.33 | 10.98 | 3.17 | 9.75 | 1.87x |
| 2 | 2 | 121.00 | 24.54 | 12.88 | 6.34 | 76.10 | 0.57x |
| 2 | 4 | 65.88 | 12.32 | 10.97 | 3.17 | 38.55 | 1.05x |

The field stage adds about 0.8–2.1 ms. Owned device allocations, including
cuFFT workspaces, are about 20 GiB on one GPU, 10 GiB per GPU on two, and
5 GiB per GPU on four. GPU runtime/MPI allocations are additional.

The full distribution agrees with NumPy FFTs and SciPy natural splines to
less than `4e-15` maximum absolute error in 28 checks: 12 CPU/MPI checks,
12 CUDA checks on one node, and 4 CUDA checks spanning two nodes. For the
larger timing runs, maximum relative mass change is `5.1e-12` and maximum
relative total-energy change is `3.4e-7` over physical time 2.0. The smallest
f is `-2.4e-16` to the stated precision.

This establishes a working initial implementation. It also shows that the
two full-state redistributions limit multinode strong scaling: they consume
about 59% of the two-node/four-GPU timestep. Streaming is the largest component
on a single GPU; the spline recurrence also scales less than ideally. These
measurements do not establish a performance comparison with ADEPT/JAX.

Raw measurements, source hashes, and correctness reports are in
[results/2026-10-24](results/2026-10-24/summary.json). All final validation and
measurement Slurm steps completed with exit status `0:0`. The development
allocation was released after downloading the results.

## Original natural-spline baseline

The normalized model has electrons with acceleration `-E`, an immobile uniform
ion background, and `dE/dx = 1 - integral(f dv)`. The initial condition is a
discretely normalized Maxwellian times `1 + amplitude*cos(2*pi*x/length)`.
Defaults are `length=4*pi`, `vmax=8`, `amplitude=0.01`, and `dt=0.1`.

Each Strang step performs:

1. Half a streaming step using batched real FFTs in x and spectral phase shifts.
2. A velocity-moment MPI Allreduce and a replicated 1D FFT Poisson solve.
3. An MPI Alltoall from velocity slabs to spatial slabs.
4. A full velocity step using natural cubic splines, including coefficient
   construction and evaluation at the backward characteristic.
5. An MPI Alltoall back to velocity slabs, then half a streaming step.

Velocity slabs have shape `(nv/ranks, nx)`; spatial slabs have shape
`(nv, nx/ranks)`. Both keep x contiguous. Spline construction uses a Thomas
solve per spatial line with precomputed LU factors. Threads run across x so
memory accesses in each recurrence sweep are coalesced. The recurrence along
v is serial within each thread; this is a baseline to profile before investing
in a parallel tridiagonal solver. One-rank runs swap layout pointers and skip
the redistribution copies and MPI Alltoall.

Distribution, packing buffers, coefficients, FFT plans, and FFT workspaces
remain allocated throughout the run. GPU-aware MPI is the default. `--host-mpi`
stages MPI buffers through host memory for comparison or systems without device
pointer support. FFT streaming and the spline solve still execute on the GPU.

## Build

CUDA, cuFFT, MPI, and CMake >= 3.24 are required. The CUDA target defaults to
SM80. Use the system's MPI compiler wrapper when required by its toolchain:

```sh
cmake -S benchmarks/vlasov_cuda -B /tmp/vlasov-cuda-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx
cmake --build /tmp/vlasov-cuda-build -j 4
mpiexec -n 4 /tmp/vlasov-cuda-build/vlasov_bench --nx 32768 --nv 8192
```

Each local MPI rank selects its GPU by local rank before MPI initialization,
using the launcher's local-rank environment variable. This is needed because
Cray MPI initializes its GPU transport in `MPI_Init`. If the launcher exposes
just one GPU per rank, each rank uses its visible device 0. Do not launch
multiple ranks bound to the same physical GPU.

On Perlmutter, use `CC` instead of `mpicxx`, load an available CMake module,
and set `MPICH_GPU_SUPPORT_ENABLED=1`. Expose peer GPUs with
`--gpus-per-node=4 --gpu-bind=none`; the program assigns one rank to each GPU.
The tested CUDA 13.2 / Cray MPICH 9.1 setup reported invalid CUDA IPC handles
when Slurm restricted visibility to one GPU per rank. A development-session
build is:

```sh
~/.ergodic-agent-workflows/ops/session.sh exec -- bash -lc '
  set -euo pipefail
  if [ "${SLURM_PROCID:-0}" -ne 0 ]; then exit 0; fi
  module load cmake/3.30.2
  cmake -S benchmarks/vlasov_cuda -B "$EC_SESSION_WORKDIR/cuda-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=CC
  cmake --build "$EC_SESSION_WORKDIR/cuda-build" -j 4
'
```

Use an isolated development session and verify the account, modules, resource
counts, and launcher on the actual system before allocating. The benchmark
does not allocate or cancel jobs.

## Validate

The reference checks require NumPy >= 2 and SciPy. They compare the entire
distribution against NumPy FFTs and SciPy's natural `CubicSpline`, verify
analytic streaming, test exact redistribution, and check rank independence.
They also check mass and short-time energy error for the coupled step.

```sh
# CPU build on a machine with FFTW and MPI:
cmake -S benchmarks/vlasov_cuda -B /tmp/vlasov-cpu-build \
  -DVLASOV_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vlasov-cpu-build -j 4
python benchmarks/vlasov_cuda/validate.py /tmp/vlasov-cpu-build/vlasov_bench

# CUDA build on an allocated GPU node:
python benchmarks/vlasov_cuda/validate.py /path/to/cuda-build/vlasov_bench \
  --launcher 'srun --jobid=JOBID --overlap --nodes=1 --ntasks={ranks} --ntasks-per-node={ranks} --gpus-per-node=4 --gpu-bind=none'
```

For validation across nodes, put `TMPDIR` on shared scratch so the driver can
read rank 0's dump, and use an appropriate explicit node/task layout.

## Measure

`measure.py` runs serial repeats of the same fixed problem on different MPI
placements. It verifies the reported placement and saves JSONL with timings,
diagnostics, commands, rank mapping, and optional source-state provenance:

```sh
python benchmarks/vlasov_cuda/measure.py /path/to/cuda-build/vlasov_bench \
  --launcher 'srun --jobid=JOBID --overlap --nodes={nodes} --ntasks={ranks} --ntasks-per-node={ranks_per_node} --gpus-per-node=4 --gpu-bind=none' \
  --placements 1:1 1:2 1:4 2:2 2:4 --expect-cuda \
  --nx 32768 --nv 8192 --steps 20 --warmup 3 --repeats 3 \
  --output /path/to/outputs/scaling.jsonl
```

Run the lightweight driver on a compute node, with overlapping inner Slurm
steps if using a development-session outer step. The source directory remains
the local worktree; synchronize edits before rebuilding. Keep builds under
the session `workdir` and results under `outputs` so source sync preserves them.

Each executable prints one JSON object. `step_ms` excludes initialization,
planning, warmup, diagnostics, downloads, and optional dumps. Warmup is followed
by reinitialization so reported physical time is exactly `steps*dt`. Component
timings explicitly synchronize the device and do not overlap communication
with computation. Times are the maximum across ranks of each rank's mean;
the component maxima need not sum to `step_ms`. `mpi_ms` covers the slab/halo
exchanges and halo-width coordination; `field_ms` includes density integration,
its Allreduce, and Poisson. `pack_ms` includes packing and unpacking for all
full-state redistributions.

## Scope

This is an initial solver/communication benchmark, not a replacement for ADEPT.
It supports one species, a uniform 1D1V grid and periodic x. The original natural
spline uses zero extrapolation; local cubic mode uses ADEPT's `1e-30`. Remapping
does not guarantee positivity or exact discrete conservation; no clipping or
mass correction hides that error. Keep velocity tails negligible and inspect
the reported mass, energy, and minimum f. The validation is short-time and
does not establish nonlinear or long-time fidelity.

Both dimensions must divide evenly by the MPI rank count. Local arrays must
fit MPI's int count limit. Memory reporting includes owned device buffers and
cuFFT workspaces, but excludes CUDA/MPI runtime overhead. The current storage
uses roughly eight copies of the distribution per rank plus FFT workspace when
full-line solves are enabled. Unused transpose/coefficient buffers are omitted
for collisionless local-cubic halo runs.
Packing copies, scratch reuse, spline batching, and communication overlap are
deliberately available optimization targets once measurements identify the
limiting operation. No speedup over ADEPT/JAX is claimed without an equivalent
comparison run.
