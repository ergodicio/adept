#!/usr/bin/env python

import os, sys, subprocess

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)

# import versioneer  # noqa: E402

# get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# get the current git commit hash
def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        return "unknown"


setup(
    # metadata
    name="adept",
    description="Automatic Differentiation Enabled Plasma Transport",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ergodicio/adept",
    author="Archis Joglekar",
    author_email="archis@ergodic.io",
    version="0.0.7",
    # cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "diffrax",
        "matplotlib",
        "scipy",
        "numpy",
        "tqdm",
        "xarray",
        "mlflow",
        "flatdict",
        "h5netcdf",
        "optax",
        "boto3",
        "pint",
        "plasmapy",
        "tabulate",
        "interpax",
        "tabulate",
        "pydantic",
        "scienceplots",
    ],
    extras_require={
        "cpu": ["jax"],
        "gpu": ["jax[cuda12]"],
    },
)
