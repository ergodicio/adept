# ADEPT
ADEPT is an **A**utomatic **D**ifferentation **E**nabled **P**lasma **T**ransport code.

It solves the equations of motion for a plasma. 

Currently, it is tested to reproduce

1. Two fluid - Poisson system in 1D
2. Vlasov-Poisson system in 2D

### What is novel about it?
- Automatic Differentiation (AD) Enabled (bc of JAX, Diffrax)
- GPU-capable (bc of JAX, XLA)
- Experiment manager enabled (bc of mlflow)
- Pythonic

### What does AD do for us?
AD enables the calculation of derivatives of entire simulations, pieces of it, or anything in between. This can be used for 
- sensitivity analyses
- parameter estimation
- parameter optimization
- model training

A couple of implemented examples are

#### Find the resonant frequency given a density and temperature
This is provided as a test in `tests/test_resonance_search.py`. Also see ref. [1]  

#### Fit a parameteric model for unresolved / unsolved microphysics
The gist is that there is a discrepancy in the observable between a "first-principles" and "approximate" simulation.
You would like for that discrepancy to decrease. To do so, you add a neural network to your "approximate" solver in a smart fashion.
Then, you calibrate the results of your "approximate" simulation against the "ground-truth" from the "first-principles" simulation.
After doing that enough times, over a diverse enough set of physical parameters, the neural network is able to help your "approximate" solver
recover the "correct" answer.

See ref. [2] for details and an application

### What does an experiment manager do for us?
An experiment manager, namely `mlflow` here, simplifies management of simulation configurations and artifacts.
We run `mlflow` on the cloud so we have a central, web-accessible store for all simulation objects. This saves all the 
data and simulation management effort and just let `mlflow` manage everything. To see for yourself,
just run a simulation and type `mlflow ui` into your terminal and see what happens :)

## Current (Tested) Implementation:
1D Electrostatic Plasma - 3 Moments - $$n(t, x), u(t, x), p(t, x)$$

- Gradients are calculated spectrally using FFTs
- 4th order explicit time integrator (using Diffrax)

Depending on the flags in the config, it can support
- Bohm-Gross oscillation frequency (Fluid dispersion relation)
- Landau damping (through a momentum damping term and a tabulated damping coefficient)
- Kinetic oscillation frequency (Kinetic dispersion relation by modifying adiabatic index)

## Installation
### Conda
1. Install `conda` (we recommend `mamba`)
2. `mamba env create -f env.yaml` or `mamba env create -f env_gpu.yaml`
3. `mamba activate adept`

### pip
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip3 install -r requirements.txt`

## Tests
The package is tested against 
- `test_resonance.py` - recover the Bohm-Gross dispersion relation and kinetic dispersion relation just using the forward pass
- `test_resonance_search.py` - recover the Bohm-Gross dispersion relation and kinetic dispersion relation using the backward pass
- `test_landau_damping.py` - recover the Landau damping rate according to the kinetic dispersion relation using a phenomenological term
- `test_against_vlasov.py` - recover a driven warm plasma wave simulation that is nearly identical to a Vlasov-Boltzmann simulation

### Run the tests
1. `pip install pytest`
2. `pytest` or `pytest tests/<test_filename>`

## Run custom simulations
Take one of the `config`s in the `/configs` directory and modify it as you wish. Then use `run.py` to run the simulation
You will find the results using the mlflow ui. You can find the binary run information will be stored using mlflow as well.

## Contributing guide
The contributing guide is in development but for now, just make an issue / pull request and we can go from there :)

References:
[1] A. S. Joglekar and A. G. R. Thomas, “Machine learning of hidden variables in multiscale fluid simulation,” Mach. Learn.: Sci. Technol., vol. 4, no. 3, p. 035049, Sep. 2023, doi: 10.1088/2632-2153/acf81a.
[2] A. S. Joglekar and A. G. R. Thomas, “Unsupervised discovery of nonlinear plasma physics using differentiable kinetic simulations,” J. Plasma Phys., vol. 88, no. 6, p. 905880608, Dec. 2022, doi: 10.1017/S0022377822000939.
 

Citation:
[1] A. S. Joglekar and A. G. R. Thomas, “Machine learning of hidden variables in multiscale fluid simulation,” Mach. Learn.: Sci. Technol., vol. 4, no. 3, p. 035049, Sep. 2023, doi: 10.1088/2632-2153/acf81a.