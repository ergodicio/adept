from adept._vlasov1d.normalization import PlasmaNormalization


class Vlasov1DSimulation:
    """
    Domain object representing a Vlasov-1D simulation setup.
    Holds the physical parameters computed from config.
    """

    def __init__(self, plasma_norm: PlasmaNormalization):
        self.plasma_norm = plasma_norm
