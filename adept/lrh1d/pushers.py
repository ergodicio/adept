import equinox as eqx


class IdealGas(eqx.Module):
    sigma_sb: float
    c: float

    def __init__(self, cfg):
        self.sigma_sb = 4.0
        self.c = 4.0

    def __call__(self, ne, ni, Te, Ti, Tr):
        pi = ni * Ti
        pe = ne * Te
        pr = 4.0 / 3.0 * self.sigma_sb / self.c * Tr**4.0

        return pi, pe, pr
