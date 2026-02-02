"""
Spectrax-1D Module

Wrapper for the spectrax library to solve Vlasov-Maxwell equations using
Hermite-Fourier decomposition with ADEPT's MLflow logging infrastructure.
"""

from adept._spectrax1d.modules import BaseSpectrax1D

__all__ = ["BaseSpectrax1D"]
