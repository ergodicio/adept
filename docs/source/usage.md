# Usage

## Installation

To use ADEPT, first install the requirements using pip:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

or using conda:

```bash
mamba env create -f env.yaml
mamba activate adept
```

---

## Run an Example

The most common use case for ADEPT is a simple forward simulation that can be run from the command line. For example, to run a 1D1V Vlasov simulation of a driven electron plasma wave:

```bash
python3 run.py --cfg configs/vlasov-1d/epw
```

The input parameters are provided in `configs/vlasov-1d/epw.yaml`.

### Access the Output

The output will be saved and made accessible via MLFlow. To access it:

1. Launch an mlflow server via running `mlflow ui` from the command line
2. Open a web browser and navigate to http://localhost:5000
3. Click on the experiment name to see the results

---

## Solver-Specific Guides

```{toctree}
:maxdepth: 2

usage/initialization
usage/vlasov1d
usage/vlasov1d2v
usage/tf1d
```
