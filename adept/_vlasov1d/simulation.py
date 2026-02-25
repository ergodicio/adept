from adept._vlasov1d.grid import Grid
from adept._vlasov1d.normalization import PlasmaNormalization
from adept.functions import SpaceTimeEnvelopeFunction


class Vlasov1DSimulation:
    """
    Domain object representing a Vlasov-1D simulation setup.
    Holds the physical parameters computed from config.
    """

    def __init__(
        self,
        plasma_norm: PlasmaNormalization,
        grid: Grid,
        nu_fp_prof: SpaceTimeEnvelopeFunction | None = None,
        nu_K_prof: SpaceTimeEnvelopeFunction | None = None,
    ):
        self.plasma_norm = plasma_norm
        self.grid = grid
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof
