# ADEPT
ADEPT is an [A]utomatic [D]ifferentation [E]nabled [P]lasma [T]ransport code.

It solves the fluid equations of motion for a plasma. These are derived from moments of the Vlasov-Boltzmann equation.
In 1D, this equation is

$$ \int \partial_t f + v \partial_x f + E \partial_v f = C(f) $$

The first moment gives the continunity equation

$$ \partial_t n + \partial_x (n~u) $$

The second moment gives the continunity equation

$$ \partial_t u + u \partial_x u = e E - \partial_x p$$

The third moment can be considered to be static which gives something like the ideal gas law

$$ p = \gamma n k_B T $$ 

or can be considered dynamic.

### What is novel about it?
- Automatic Differentiation (AD) Enabled
- GPU-capable
- Experiment manager enabled (mlflow)
- Pythonic

### What does AD do for us?
AD enables the calculation of derivatives of entire simulations. This can be used for 
- sensitivity analyses
- parameter estimation
- parameter optimization
- model training

A couple of implemented examples are

#### Find the resonant frequency given a density and temperature
This is provided as a test in `tests/test_resonance_search.py`  

#### Fit a parameteric model for unresolved / unsolved microphysics
This work is described in an upcoming publication

### What does an experiment manager do for us?
An experiment manager, namely `mlflow` here, simplifies management of simulation configurations and artifacts.
We run `mlflow` on the cloud so we have a central, web-accessible store for all simulation objects. This saves all the 
data and simulation management effort and just let `mlflow` manage everything. To see for yourself,
just run a simulation and type `mlflow ui` into your terminal and see what happens :)

## Current (Tested) Implementation:
1D Electrostatic Plasma - 3 Moments - $$n(t, x), u(t, x), p(t, x)$$

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

## Train physics models in-situ using differentiable plasma simulations
In development. Please reach out if you'd like to learn about this before I get a chance to write it down here.

## Contributing guide
The contributing guide is in development but for now, just make an issue / pull request and we can go from there :)