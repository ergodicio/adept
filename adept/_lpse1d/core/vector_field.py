from jax import numpy as jnp

from adept._lpse1d.core.epw import SpectralEPWSolver1D
from adept._lpse1d.core.laser import LaserSolver1D


class SplitStep1D:
    """
    1D split-step solver for coupled laser-plasma system.

    This implements the split-step time integration matching MATLAB's
    spectralEpwUpdate() and evalLaserFieldUpdate() functions.

    The system consists of:
    - Plasma waves (EPW): phi_k in k-space
    - Pump laser: E0 in real space
    - Seed laser: E1 in real space (for SRS)

    Matches MATLAB split-step approach (lines 1377-1427 and 1966-2118).
    """

    def __init__(self, cfg: dict):
        """
        Initialize the 1D split-step solver.

        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.epw_solver = SpectralEPWSolver1D(cfg)
        self.laser_solver = LaserSolver1D(cfg)

        # Check what physics is enabled
        self.evolve_epw = cfg["terms"]["epw"]["active"]
        self.evolve_laser = cfg["terms"].get("laser", {}).get("active", False)
        self.evolve_raman = cfg["terms"]["epw"]["source"]["srs"]

    def __call__(self, t: float, y: dict, args: dict) -> dict:
        """
        Compute time derivatives for all fields.

        This is called by diffrax's ODE solver.

        Args:
            t: Current time
            y: State dictionary with keys:
                - "epw": plasma wave potential in k-space (shape: nx) [complex as float64 view]
                - "E0": pump laser field in real space (shape: nx) [complex as float64 view]
                - "E1": seed laser field in real space (shape: nx) [complex as float64 view]
            args: Arguments dictionary

        Returns:
            Dictionary of time derivatives with same structure as y
        """
        # Convert views back to complex for computation
        epw_complex = y["epw"].view(jnp.complex128)
        E0_complex = y["E0"].view(jnp.complex128)
        E1_complex = y["E1"].view(jnp.complex128)

        # Prepare state dict with complex arrays
        state = {
            "epw": epw_complex,
            "E0": E0_complex,
            "E1": E1_complex,
        }

        # Initialize derivatives
        dydt = {}

        # ====================================================================
        # EPW update
        # ====================================================================
        if self.evolve_epw:
            # Call spectral EPW solver
            # This returns the updated phi_k directly (not a derivative)
            # because it's a split-step operator
            epw_new = self.epw_solver(t, state, args)
            # Convert back to time derivative for ODE solver
            depw_dt = (epw_new - epw_complex) / self.cfg["grid"]["dt"]
        else:
            depw_dt = jnp.zeros_like(epw_complex)

        # ====================================================================
        # Laser field updates (E0 and E1)
        # ====================================================================
        if self.evolve_laser:
            dE0_dt = self.laser_solver(t, state, "E0")
        else:
            dE0_dt = jnp.zeros_like(E0_complex)

        if self.evolve_raman:
            dE1_dt = self.laser_solver(t, state, "E1")
        else:
            dE1_dt = jnp.zeros_like(E1_complex)

        # ====================================================================
        # Convert back to float64 view for diffrax
        # ====================================================================
        dydt["epw"] = depw_dt.view(jnp.float64)
        dydt["E0"] = dE0_dt.view(jnp.float64)
        dydt["E1"] = dE1_dt.view(jnp.float64)

        return dydt
