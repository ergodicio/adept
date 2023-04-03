# base image
FROM python:3.10-slim

# making directory of app
WORKDIR /workdir

# copying all files over
COPY dockerfiles/cpu/requirements-cpu.txt .
COPY continuum continuum
COPY run_job.py .

ENV JAX_ENABLE_X64=True

RUN apt update && apt install --yes --no-install-recommends git

# install pip then packages
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements-cpu.txt