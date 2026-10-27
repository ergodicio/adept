#!/usr/bin/env bash
# ABI3 regression on an owned two-GPU shared allocation.
set -euo pipefail
if [ "${SLURM_PROCID:-0}" -ne 0 ]; then exit 0; fi
: "${EC_SESSION_WORKDIR:?}" "${EC_SESSION_OUTPUTS:?}" "${SLURM_JOB_ID:?}"
report_root="$EC_SESSION_OUTPUTS/native-visibility"
mkdir "$report_root"
export MPICH_GPU_SUPPORT_ENABLED=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=2 NCCL_NET=Socket NCCL_NET_GDR_LEVEL=0 FI_CXI_DISABLE_HOST_REGISTER=1
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
cmake -S adept/_vlasov1d/native -B "$EC_SESSION_WORKDIR/native-build" \
    -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Release > "$report_root/build.log" 2>&1
cmake --build "$EC_SESSION_WORKDIR/native-build" -j4 >> "$report_root/build.log" 2>&1
library="$EC_SESSION_WORKDIR/native-build/libadept_vlasov.so"
sha256sum "$library" > "$report_root/library.sha256"
sha256sum adept/_vlasov1d/native/api.cpp adept/_vlasov1d/native/core/* > "$report_root/source.sha256"
python -u benchmarks/vlasov_cuda/integration_check.py --library "$library" \
    --output "$report_root/shared" --ranks 1 --module base --ions --steps 100 \
    --gpus-per-node 2 --host-mpi > "$report_root/shared.log" 2>&1
python -u benchmarks/vlasov_cuda/integration_check.py --library "$library" \
    --output "$report_root/bound" --ranks 2 --module base --ions --steps 100 \
    --gpus-per-node 2 --per-rank-gpu > "$report_root/bound.log" 2>&1
python -u benchmarks/vlasov_cuda/integration_check.py --library "$library" \
    --output "$report_root/vp-main" --ranks 2 --module vp \
    --vp-source "$EC_SESSION_OUTPUTS/vp-source" --gpus-per-node 2 --per-rank-gpu \
    > "$report_root/vp-main.log" 2>&1
echo "Single-worker and per-worker GPU visibility checks passed: $report_root"
