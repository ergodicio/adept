# Native Vlasov backend

Select `backend: native` in the usual `solver: vlasov-1d` configuration and run
through `ergoExo`. See the [configuration and build instructions](../../../docs/source/solvers/vlasov1d/config.md#execution-backend).
An existing `BaseVlasov1D` subclass can still supply initialization, save
callbacks and postprocessing. C++/CUDA/MPI advances the distribution; JAX runs
the original save callbacks, without automatic differentiation of the solve.

`core/` is the canonical numerical implementation. The standalone program in
`benchmarks/vlasov_cuda` includes these same sources. Build the shared library
on the target system with CMake; `VLASOV_CUDA=OFF` builds the FFTW validation
implementation. The library is loaded only by MPI workers, not the outer
Python driver.

The outer driver performs initialization on the CPU and stages the initial
state in bounded blocks. One persistent worker owns the solver on each GPU.
At a save point, DLPack exposes synchronized native buffers as JAX arrays on
a velocity-slab mesh. CUDA views require zero-copy import. The CPU validation
build permits a copy when FFTW's host alignment is insufficient for JAX.
Callbacks complete and saved outputs reach the host before the next native
mutation. No borrowed state view may survive `av_close`; the library checks
this lifetime rule explicitly.

Workers stream saved leaves into host `.npy` memory maps. The driver assembles
distributed saved outputs and returns the usual Diffrax `Solution` containing
NumPy arrays. Existing xarray/NetCDF writers consume it. The module retains
the backing workspace while its returned saves are in use. MLflow artifacts
include actual save schedules, worker statistics/logs and the library hash.

Standard saves round to actual timestep boundaries and merge duplicates.
Scalars default to every step; `native.scalar_stride` can reduce the cadence.
Custom callbacks preserve their requested times by default because some
project postprocessors reconstruct those axes. Off-step custom saves require
a previous-state copy and interpolation. `custom_save_times: steps` avoids
that work when the postprocessor consumes actual result times. Postprocessors
that reconstruct uniform scalar axes also need actual result times if the
last scalar interval is shorter than the requested stride.

The validated physics is spectral spatial streaming, ADEPT's local cubic C1
velocity interpolation, sixth-order splitting, Poisson fields, and optional
constant-rate Chang-Cooper Dougherty collisions. Kinetic ions use the existing
`terms.species` entries, with independent velocity grids, charges and masses.
Each stage sums every species' charge density into one MPI field reduction,
then advances every species under that shared field. Collisions act on the
reference species, as in ADEPT's JAX path; inter-species collisional exchange
is not included. Rebuild the library when updating ADEPT (current ABI: 3).
Unsupported physics and
trainable modules fail explicitly. Performance and reproduction evidence is
under [the benchmark results](../../../benchmarks/vlasov_cuda/results/2026-10-24-adept).
