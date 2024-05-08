#!/bin/bash
#SBATCH -A m4490_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 20:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="$PSCRATCH/mlflow"
export MLFLOW_EXPORT=True

# copy job stuff over
source /pscratch/sd/a/archis/venvs/adept-gpu/bin/activate
module load cudnn/8.9.3_cuda12.lua

cd /global/u2/a/archis/adept/
srun python3 tpd_learn.py