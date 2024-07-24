# ADEPT
ADEPT is an **A**utomatic **D**ifferentation **E**nabled **P**lasma **T**ransport code.

## Installation
### Conda
1. Install `conda` (we recommend `mamba`)
2. `mamba env create -f env.yaml` or `mamba env create -f env_gpu.yaml`
3. `mamba activate adept`

### pip
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip3 install -r requirements.txt`

## Docs
https://adept.readthedocs.io/en/latest/

## Examples
https://github.com/ergodicio/adept-notebooks

## Usage
`python3 run.py --cfg {config_path}` without the `.yaml` extension

This runs the simulation defined in the config and stores the output to an `mlflow` server.

Unless you have separately deployed an `mlflow` server somewhere, it simply writes files using the mlflow specification to the current working directory. 

To access and visualize the results, it is easiest to use the UI from the browser by typing `mlflow ui` in the command line from the same directory.


## Contributing guide
The contributing guide is in development but for now, just make an issue / pull request and we can go from there :) 

## Citation

[1] A. S. Joglekar and A. G. R. Thomas, “Machine learning of hidden variables in multiscale fluid simulation,” Mach. Learn.: Sci. Technol., vol. 4, no. 3, p. 035049, Sep. 2023, doi: 10.1088/2632-2153/acf81a.
