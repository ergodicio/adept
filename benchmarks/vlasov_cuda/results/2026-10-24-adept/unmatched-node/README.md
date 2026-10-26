These initial four-GPU measurements are excluded from the reported comparison.
Slurm placed ADEPT/JAX on nid001600 and CUDA on nid003745. The audit caught
this before publication; the final pair was repeated with an explicit node
list, using identical physical GPUs for both solvers.
