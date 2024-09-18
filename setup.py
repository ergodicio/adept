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
    version=get_git_commit_hash(),
    # cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "jax[cuda12]",
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
        "mlflow_export_import",
        "plasmapy",
        "tabulate",
        "interpax",
        "tabulate",
    ],
)
