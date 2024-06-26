#!/bin/bash
#SCRON -q workflow
#SCRON -A m4490
#SCRON -t 30-00:00:00
#SCRON -o output-%j.out
#SCRON --open-mode=append

0 0 */5 * * /pscratch/sd/a/archis/venvs/adept-cpu/bin/python3 /global/u2/a/archis/adept/tpd_learn.py

