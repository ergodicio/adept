#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu

export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="$PSCRATCH/mlflow"

source /global/u2/a/archis/adept/venv/bin/activate
cd /global/u2/a/archis/adept/