set -euo pipefail
source /global/common/software/m4490/archis/venvs/vp-turbulence/bin/activate
export PYTHONPATH=/pscratch/sd/a/archis/vp-turbulence-sessions/20261023T081121Z-nv4096-nx268800-93a418d7/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=2 NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=LOC FI_CXI_DISABLE_HOST_REGISTER=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
export JAX_ENABLE_X64=true
root=/pscratch/sd/a/archis/adept-sessions/20261024T103224Z-71019-3473
cd "$root/outputs"
srun --jobid=60205333 --overlap --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=16 --cpu-bind=none --gpus-per-node=4 --gpu-bind=none --mpi=none --kill-on-bad-exit=1 python "$root/src/benchmarks/vlasov_cuda/distributed_reference.py" --prepared "$root/outputs/native-nx268800" --config "$root/src/benchmarks/vlasov_cuda/reference/latest-nv4096/nx268800.yaml" --vp-source "$PYTHONPATH" --output "$root/outputs/native-nx268800-distributed" --steps 1 100 > "$root/workdir/reference8.log" 2>&1
tail -n 8 "$root/workdir/reference8.log"
