"""Plasma normalization for VFP-1D simulations.

VFP-1D normalizes to a fixed reference density n0 = 9.0663e21 cm^-3,
with length unit x0 = c/wp0 and time unit tp0 = 1/wp0.
"""

from dataclasses import dataclass

import numpy as np
from astropy import constants as csts
from astropy import units as u
from astropy.units import Quantity as _Q

from adept.vfp1d.helpers import calc_logLambda


@dataclass
class PlasmaNorm:
    """Core normalization quantities for VFP-1D.

    Defines what "1" means in simulation units for length, time, and velocity.
    """

    n0: _Q  # reference density (fixed)
    wp0: _Q  # plasma frequency
    tp0: _Q  # time unit = 1/wp0
    x0: _Q  # length unit = c/wp0
    vth: _Q  # thermal velocity sqrt(2 Te / me)
    beta: float  # vth / c (dimensionless)
    Te: _Q  # reference electron temperature
    ne: _Q  # electron density
    logLambda_ei: float
    logLambda_ee: float

    @staticmethod
    def from_config(cfg_units: dict) -> "PlasmaNorm":
        """Construct from the raw ``cfg["units"]`` dict."""
        n0 = _Q("9.0663e21/cm^3")
        ne = _Q(cfg_units["reference electron density"]).to("1/cm^3")
        Te = _Q(cfg_units["reference electron temperature"]).to("eV")
        Z = cfg_units["Z"]
        ion_species = cfg_units["Ion"]

        wp0 = np.sqrt(n0 * csts.e.to("C") ** 2.0 / (csts.m_e * csts.eps0)).to("Hz")
        tp0 = (1 / wp0).to("fs")
        x0 = (csts.c / wp0).to("nm")
        vth = np.sqrt(2 * Te / csts.m_e).to("m/s")
        beta = float((vth / csts.c).to("").value)

        logLambda_ei, logLambda_ee = calc_logLambda(
            {"units": cfg_units}, ne, Te, Z, ion_species, force_ee_equal_ei=True
        )

        return PlasmaNorm(
            n0=n0,
            wp0=wp0,
            tp0=tp0,
            x0=x0,
            vth=vth,
            beta=beta,
            Te=Te,
            ne=ne,
            logLambda_ei=logLambda_ei,
            logLambda_ee=logLambda_ee,
        )
