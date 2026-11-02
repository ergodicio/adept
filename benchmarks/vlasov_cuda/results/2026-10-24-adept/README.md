# ADEPT/JAX versus native CUDA on A100s

The native backend is integrated into ADEPT (`backend: native`). It retains
`ergoExo`, ordinary/custom module initialization, original save callbacks and
xarray/NetCDF postprocessing. Standard saves use actual timestep boundaries;
outputs are streamed to host memory maps. The C++ physics core is shared with
the standalone timing executable.

| Grid | Nodes / A100s | ADEPT/JAX ms/step | Native ms/step | Speedup |
|---|---:|---:|---:|---:|
| 134400 × 4096 | 1 / 4 | 223.54 | 174.24 | 1.283× |
| 134400 × 4096 | 2 / 8 | 383.94 | 140.07 | 2.741× |
| 268800 × 4096 | 2 / 8 | 756.70 | 275.99 | 2.742× |

FP64, spectral x, local cubic C1 v, sixth-order splitting, dt=0.02,
Chang-Cooper Dougherty nu=1e-12. Both paths reset to the same mature t600
checkpoint before each repeat. Median of three repeats, ten timed steps,
three warmup steps. Timings exclude compilation, preprocessing, save callbacks,
I/O and postprocessing; these are physics-step speedups, not complete-run
wall-clock speedups. No concurrent benchmark workloads were run.

Allocation 60221398 used nid001600 and nid003745, four A100-SXM4-40GB cards
per node. ADEPT was pinned to ddd70b957dc3df85f0b5ec84af927c25d09d24e0,
JAX/JAXlib 0.10.2, interpax 0.3.14+10.g84aaf30, Diffrax 0.7.2.
The four-GPU pair was rerun with both solvers pinned to nid001600 after an
audit found the initial pair used different nodes. Excluded measurements are
retained under `unmatched-node/`; the final pair's provenance is under
`pinned-one-node/`. Both eight-GPU pairs used the same physical nodes/GPUs.

Five native CPU integration tests and 33 existing configuration/save/logging
regressions passed. Seven GPU integration cases passed: ordinary ADEPT and
VPTwoStream1D at 1/4/8 ranks, plus the latest native velocity-spectrum and
cascade callbacks at eight ranks. Each compares original save callbacks at
the actual native save times (rtol=1e-10, atol=1e-11). The largest absolute
error was 1.16e-11 across all saved diagnostics; ordinary ADEPT saves were
within 1.32e-13. `observer-netcdf/` contains outputs from the unchanged
VPTwoStream1D postprocessor; `observer-native/` contains its MLflow provenance.
The initial `integration-v3.log` also records a subsequently fixed callback
constant serialization failure; its three ordinary ADEPT cases passed.

`summary.json` is rebuilt by `python summarize.py`, which verifies source
archive hashes, matched hosts, finite outputs, endpoint aggregate checks and
all seven integration results. Raw repeated measurements, software versions,
checkpoint provenance, command lines and executable hashes are adjacent.
Pointwise full-grid t600-to-t602 validation of the shared numerical core is
in `../2026-10-24-halos/`; a full t0-to-2000 trajectory has not been tested.
The NERSC session's files remain under
`/pscratch/sd/a/archis/adept-sessions/20261024T190159Z-99199-22637/`.
