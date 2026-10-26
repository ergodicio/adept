# Native kinetic-species integration validation

All five CUDA integration cases passed on allocation 60336491, nodes
nid003048/nid003744 (A100-SXM4-40GB, driver 580.178.04). The allocation was
released after preserving the outputs; all validation steps completed with
exit code 0. This is correctness evidence, not a production-scale ion timing.
Only this written report is published. Raw source archives, NetCDF outputs,
configs and MLflow metadata remain in the matching local result directory
and the isolated NERSC session recorded below.

| Case | GPUs | Steps | Maximum absolute saved-value error vs ADEPT/JAX |
|---|---:|---:|---:|
| Kinetic electrons + protons | 1 | 100 | 1.12266e-12 |
| Kinetic electrons + protons | 4 | 100 | 9.30811e-13 |
| Kinetic electrons + protons | 8 / 2 nodes | 100 | 9.66338e-13 |
| Electron two-stream regression | 4 | 5 | 1.49214e-13 |
| Current turbulence custom module | 8 / 2 nodes | 5 | 1.01110e-11 |

Each saved leaf and time was compared with rtol 1e-10, atol 1e-11, including
full distributions, fields, species moments and scalar diagnostics. The
proton case has nx=32, electron nv=64/vmax=8, ion nv=32,
ion vmax=6*sqrt(0.1/1836), qi=1, mi=1836, Ti=0.1 and dt=0.02. Only the
ions have the initial sine density perturbation. Both species change: max
|f(t=2)-f(t=0)| is approximately 0.02271 for electrons and 0.22539 for ions.
Constant-rate analytic-beta Chang-Cooper Dougherty collisions (nu=0.001)
act on electrons, matching the existing ADEPT reference-species convention.
No inter-species collision operator is claimed.

The turbulence case uses VPTwoStream1D from vp-turbulence main commit
94df45d0172f861ae42bae5297bacc957294ba32, with no project edits. Its original
f0, velocity-spectrum and cascade callbacks and postprocessor ran through
ergoExo. `vp-artifacts/binary/` contains the actual postprocessed NetCDF
outputs; `vp-artifacts/native/` contains the request, statistics and library
hash logged by the run. `vp-source.tar.gz` preserves its source snapshot.

ADEPT source: ddc4dd14082a60f9fe8a8f9dfdb80044f5ab91f0 plus the recorded
`run_integration.sh` launcher, archived in source.tar.gz. The compiled C++
physics is unchanged in the final PR except for trailing blank-line cleanup.
Subsequent Python changes add float64/null-default guards and ensure every
registered Vlasov module honors backend selection. Runtime: JAX/JAXlib
0.10.2, interpax 0.3.14+10.g84aaf30, diffrax 0.7.2, cloudpickle 3.1.2,
CUDA 13.2.78, GNU 14 / Cray MPI 9.1. Existing environments were unchanged.

Build/launch commands: `../../run_integration.sh` and
`../../integration_check.py`. The turbulence check additionally passes
`--module vp --ranks 8 --vp-source <snapshot>`. Machine paths in hashes,
requests and accounting refer to the isolated session at
/pscratch/sd/a/archis/adept-sessions/20261026T213621Z-5332-10833/.

Local validation: 8 native/entropy tests, including 1/2-rank MPI and three
kinetic species with independent grids and charges/masses, plus 211 existing
lifecycle/core/builder/configuration regressions. The wheel contains the
CMake/C++ sources needed to build this backend from downstream projects.

## Review corrections and masked-GPU regression

ABI3 uses the CUDA ordinal selected at allocation time for both JAX and
DLPack, separately from the MPI local rank. Native synchronization restores
that allocation's device. Collision guards and the constant frequency use
the resolved ADEPT envelopes, so an omitted bump height defaults to one and
is rejected as a nonconstant profile; an omitted constant baseline retains
ADEPT's zero default. Eleven focused CPU tests pass, including both omitted
time/space pulse defaults, constant defaults and MPI/NetCDF parity.

Supplemental ABI3 checks on shared allocation 60339880, nid200301:

| Case | GPUs | Steps | Maximum absolute saved-value error |
|---|---:|---:|---:|
| Kinetic electrons + protons | 1 | 100 | 1.12266e-12 |
| Same, one GPU visible per worker | 2 | 100 | 8.31335e-13 |
| Current turbulence callbacks, one GPU visible per worker | 2 | 5 | 5.74119e-12 |

Both MPI local ranks 0/1 use CUDA ordinal zero and report
CUDA_VISIBLE_DEVICES=0. A separate CUDA probe confirmed distinct physical PCI
devices 0000:41:00.0 and 0000:C1:00.0. Comparison tolerances remain unchanged.
The earlier 1/4/8-GPU tests above use the normal all-visible production path.

Two launcher/transport failures were retained in the local evidence: shared
Slurm steps with gpu-bind=none exposed the same GPU to both workers, causing
NCCL to reject duplicate devices; explicit single-GPU binding exposed distinct
GPUs, but NCCL SHM imports failed with an invalid device ordinal. The masked
regression passed with host-staged MPI and socket NCCL, disabling P2P/SHM and
GPU-direct registration. These transport settings are for the regression,
not a new throughput benchmark. Reproduction: ../../run_visibility_check.sh.
The shared lease was released after preserving reports and source.
