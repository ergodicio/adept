"""
Spectrax-1D Module

Wrapper for the spectrax library to solve Vlasov-Maxwell equations using
Hermite-Fourier decomposition with ADEPT's MLflow logging infrastructure.
"""

from adept._spectrax1d.modules import EPW1D, BaseSpectrax1D, HermiteSRS1D

__all__ = ["EPW1D", "BaseSpectrax1D", "HermiteSRS1D"]
