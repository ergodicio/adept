#!/usr/bin/env bash
# Run only inside an owned two-node allocation, after building both CUDA targets.
set -euo pipefail
if [ "${SLURM_PROCID:-0}" -ne 0 ]; then exit 0; fi
: "${EC_SESSION_ROOT:?}" "${SLURM_JOB_ID:?}"
source_root="$EC_SESSION_ROOT/src"
report_root="$EC_SESSION_OUTPUTS/adept-comparison"
mkdir "$report_root"
prepared_root=/pscratch/sd/a/archis/adept-sessions/20261024T103224Z-71019-3473/outputs
vp_small=/pscratch/sd/a/archis/vp-turbulence-sessions/20261023T081121Z-nv4096-nx134400-5ce4fe14/src
vp_large=/pscratch/sd/a/archis/vp-turbulence-sessions/20261023T081121Z-nv4096-nx268800-93a418d7/src
export MPICH_GPU_SUPPORT_ENABLED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=2 NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=LOC FI_CXI_DISABLE_HOST_REGISTER=1
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
for case in 134400:1:4 134400:2:8 268800:2:8; do
    IFS=: read -r nx nodes ranks <<< "$case"
    prepared="$prepared_root/native-nx$nx"
    vp_source="$vp_small"
    if [ "$nx" = 268800 ]; then
        prepared="$prepared_root/native-nx268800-distributed"
        vp_source="$vp_large"
    fi
    # The pinned environment's own ADEPT must win over this development checkout.
    # JAX CUDA wheels supply their runtime; unloading cudatoolkit matches the reference runs.
    (
        module unload cudatoolkit 2>/dev/null || true
        source /global/common/software/m4490/archis/venvs/vp-turbulence/bin/activate
        export PYTHONPATH="$vp_source"
        srun --jobid="$SLURM_JOB_ID" --overlap --nodes="$nodes" --ntasks="$ranks" \
            --ntasks-per-node=$((ranks / nodes)) --gpus-per-node=4 --gpu-bind=none \
            --cpu-bind=none --mpi=none --kill-on-bad-exit=1 \
            python -u "$source_root/benchmarks/vlasov_cuda/adept_timing.py" \
            --prepared "$prepared" --config "$source_root/benchmarks/vlasov_cuda/reference/latest-nv4096/nx$nx.yaml" \
            --vp-source "$vp_source" --output "$report_root/jax-nx$nx-g$ranks.json" \
            > "$report_root/jax-nx$nx-g$ranks.log" 2>&1
    )
    python -u benchmarks/vlasov_cuda/measure.py "$EC_SESSION_WORKDIR/cuda-build/vlasov_bench" \
        --launcher "srun --jobid=$SLURM_JOB_ID --overlap --nodes={nodes} --ntasks={ranks} --ntasks-per-node={ranks_per_node} --gpus-per-node=4 --gpu-bind=none --cpu-bind=none" \
        --placements "$nodes:$ranks" --nx "$nx" --nv 4096 --steps 10 --warmup 3 --repeats 3 \
        --expect-cuda --velocity cubic --layout halos --integrator sixth --cells --nu 1e-12 \
        --length 12600 --vmax 15 --dt .02 --input "$prepared/initial.bin" \
        --v-grid "$prepared/v.bin" --background "$prepared/background.bin" --max-mass-change -1 \
        --source-state "$EC_SESSION_ROOT/source-state" --output "$report_root/cuda-nx$nx-g$ranks.jsonl"
done
sha256sum "$EC_SESSION_WORKDIR/cuda-build/vlasov_bench" "$EC_SESSION_WORKDIR/native-build/libadept_vlasov.so" > "$report_root/binaries.sha256"
sha256sum adept/_vlasov1d/native/api.cpp adept/_vlasov1d/native/core/* benchmarks/vlasov_cuda/adept_timing.py > "$report_root/source.sha256"
cp "$EC_SESSION_ROOT/source-state" "$report_root/source-state"
cp benchmarks/vlasov_cuda/run_adept_comparison.sh "$report_root/launch.sh"
