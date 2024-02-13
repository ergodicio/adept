#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu

export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="$PSCRATCH/mlflow"

module load conda
mamba activate adept
cd /global/u2/a/archis/adept/