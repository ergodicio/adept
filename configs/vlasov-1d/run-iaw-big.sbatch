#!/bin/bash
# Single-node 4-GPU sbatch launcher for the Te/Ti = 0.05 production IAW
# turbulence run (configs/vlasov-1d/iaw-turbulence-big.yaml).
#
# Benchmarked at ~45 ms/step with time: sixth on 4x A100-40GB (10k-step bench,
# 2026-08-05); the production config now uses time: leapfrog (validated against
# sixth on the Te/Ti=1 vortex test: identical energy history, heating, and
# n(k) through k*lambda_De ~ 0.6), which is ~6x cheaper -> ~2 h for 772,800
# steps. That fits an interactive 4-GPU node; use this sbatch when the queue
# or a longer variant makes batch preferable. grid.parallel = [x, v] shards
# the pushers across the 4 GPUs via shard_map.
#
# Submit from the repo root with:
#     ~/.claude/scripts/ergodic/submit-batch.sh configs/vlasov-1d/run-iaw-big.sbatch

#SBATCH --account=m4490_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=workdir/%x-%j.out

set -e

REPO="$(basename "$SLURM_SUBMIT_DIR")"
VENV="/global/common/software/m4490/$USER/venvs/$REPO"

source ~/.bash_profile.ext
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "[run-iaw-big] job=$SLURM_JOB_ID host=$(hostname) gpus=$SLURM_GPUS_PER_NODE"
python -u run.py --cfg configs/vlasov-1d/iaw-turbulence-big
