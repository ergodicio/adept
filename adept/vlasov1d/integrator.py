import diffrax

from adept.vlasov1d.pushers import Vlasov


class Stepper(diffrax.Euler):
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful


class VectorField:
    """
    This class contains the function that updates the state

    All the pushers are chosen and initialized here and a single time-step is defined here.

    :param cfg:
    :return:
    """

    def __init__(self, cfg):
        self.vlasov = Vlasov(cfg)

    def __call__(self, t, y, args):
        return self.vlasov(t, y, args)
