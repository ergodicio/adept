#!/usr/bin/env bash
# Run inside an owned two-node A100 session; preserves reports beside source.
set -euo pipefail
if [ "${SLURM_PROCID:-0}" -ne 0 ]; then exit 0; fi
: "${EC_SESSION_ROOT:?}" "${EC_SESSION_OUTPUTS:?}" "${SLURM_JOB_ID:?}"
report_root="$EC_SESSION_OUTPUTS/native-ions"
mkdir "$report_root"
export MPICH_GPU_SUPPORT_ENABLED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=2 NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=LOC FI_CXI_DISABLE_HOST_REGISTER=1
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
cmake -S adept/_vlasov1d/native -B "$EC_SESSION_WORKDIR/native-build" \
    -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Release > "$report_root/build.log" 2>&1
cmake --build "$EC_SESSION_WORKDIR/native-build" -j4 >> "$report_root/build.log" 2>&1
library="$EC_SESSION_WORKDIR/native-build/libadept_vlasov.so"
sha256sum "$library" > "$report_root/library.sha256"
sha256sum adept/_vlasov1d/native/api.cpp adept/_vlasov1d/native/core/* > "$report_root/source.sha256"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$report_root/gpus.csv"
scontrol show job "$SLURM_JOB_ID" > "$report_root/allocation.txt"
python -u benchmarks/vlasov_cuda/integration_check.py --library "$library" \
    --output "$report_root/coupled" --ranks 1 4 8 --module base --ions --steps 100 \
    > "$report_root/coupled.log" 2>&1
python -u benchmarks/vlasov_cuda/integration_check.py --library "$library" \
    --output "$report_root/electron" --ranks 4 --module base \
    > "$report_root/electron.log" 2>&1
echo "Native electron/ion A100 integration checks passed: $report_root"
