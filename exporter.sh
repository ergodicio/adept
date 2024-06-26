#!/bin/bash
#SBATCH --qos=regular
#SBATCH -A m4490
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu

# export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'
export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'
export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow'
# export JAX_ENABLE_X64=True
# export MLFLOW_EXPORT=True
# export MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=False

source /pscratch/sd/a/archis/venvs/adept-cpu/bin/activate
cd /global/u2/a/archis/adept/
srun python export_runs.py