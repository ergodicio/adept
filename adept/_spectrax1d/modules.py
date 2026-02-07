"""
Spectrax-1D Module: Wrapper for spectrax-based Hermite-Fourier Vlasov-Maxwell solver

This module wraps the external spectrax library to solve the Vlasov-Maxwell equations
using Hermite-Fourier decomposition. It provides ADEPT integration including MLflow
logging, unit management, and configuration handling.

The wrapper approach calls spectrax.simulation() directly rather than reimplementing
the solver in diffrax. This maintains compatibility with the external library while
leveraging ADEPT's experiment tracking infrastructure.

This file serves as the main entry point and re-exports components from submodules
for backward compatibility.
"""

# Re-export all public classes for backward compatibility
from adept._spectrax1d.base_module import BaseSpectrax1D
from adept._spectrax1d.driver import Driver
from adept._spectrax1d.epw_module import EPW1D
from adept._spectrax1d.vector_field import SpectraxVectorField

__all__ = [
    "EPW1D",
    "BaseSpectrax1D",
    "Driver",
    "SpectraxVectorField",
]
