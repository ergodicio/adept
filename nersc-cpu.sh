#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu

export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="$PSCRATCH/mlflow"

source /pscratch/sd/a/archis/venvs/adept-cpu/bin/activate
cd /global/u2/a/archis/adept/
srun python3 tpd_opt.py