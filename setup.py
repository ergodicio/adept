#!/usr/bin/env python

import os
import sys

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)

# import versioneer  # noqa: E402

# get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    # metadata
    name="adept",
    description="Automatic Differentiation Enabled Plasma Transport",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ergodicio/adept",
    author="Archis Joglekar",
    author_email="archis@ergodic.io",
    version=1.0,  # versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    packages=["adept"],
    python_requires=">=3.8",
    install_requires=[
        "jax",
        "jaxlib",
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
        "jaxopt",
        "boto3",
        "pint",
        "mlflow_export_import",
        "plasmapy",
        "tabulate",
        "interpax",
    ],
    # extras_require={
    #     "dev": [
    #         "fastapi",
    #         "httpx",  # required by fastapi test client
    #         "requests",
    #         "numpy",
    #         "pre-commit",
    #         "pytest",
    #         "pytest-cov",
    #         "sphinx",
    #     ],
    # },
    # package_data={
    #     "tesseract": [
    #         "templates/**/*",
    #         # Ensure tesseract_runtime folder is copied to site-packages when installing
    #         "../tesseract_runtime/**/*",
    #     ],
    # },
    # zip_safe=False,
    # entry_points={
    #     "console_scripts": [
    #         "tesseract=tesseract.cli:entrypoint",
    #     ],
    # },
)
