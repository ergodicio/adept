import equinox as eqx
import numpy as np
from diffrax import ODETerm, diffeqsolve
from jax import jit

from adept._base_ import Stepper
from adept._vlasov1d.simulation import EMDriver
from adept._vlasov1d.solvers.pushers.field import TransverseCurrentSourceDriver, WaveSolver
from adept.functions import EnvelopeFunction, SpaceTimeEnvelopeFunction


class VectorField(eqx.Module):
    wave_solver: WaveSolver
    driver: TransverseCurrentSourceDriver
    """
    This class contains the function that updates the state

    All the pushers are chosen and initialized here and a single time-step is defined here.

    :param cfg:
    :return:
    """

    def __init__(self, c_light, dx, dt, xax, drivers):
        self.wave_solver = WaveSolver(c=c_light, dx=dx, dt=dt)
        self.driver = TransverseCurrentSourceDriver(xax=xax, drivers=drivers)

    def __call__(self, t, y, args):
        djy = self.driver(t, args)
        return self.wave_solver(y["a"], y["prev_a"], djy, 0.0)


def test_absorbing_boundaries():
    xmax = 1000
    xmin = 0
    nx = 1024
    tmax = 200
    c_light = 11.3

    dx = (xmax - xmin) / nx
    dt = 0.95 * dx / c_light
    nt = int(tmax / dt)

    xax = np.linspace(xmin - dx / 2.0, xmax + dx / 2.0, nx + 2)

    a_old, a, charge, djy = np.zeros_like(xax), np.zeros_like(xax), np.zeros_like(xax)[1:-1], np.zeros_like(xax)

    # Create EMDriver with SpaceTimeEnvelopeFunction
    time_env = EnvelopeFunction(center=40.0, width=30.0, rise=5.0, baseline=0.0, bump_height=1.0, is_trough=False)
    space_env = EnvelopeFunction(center=800.0, width=50.0, rise=10.0, baseline=0.0, bump_height=1.0, is_trough=False)
    envelope = SpaceTimeEnvelopeFunction(time_envelope=time_env, space_envelope=space_env)
    ey_driver = EMDriver(a0=1.0e-4, k0=-1.4, w0=15.82, dw0=0.0, envelope=envelope)
    drivers = [ey_driver]

    args = {}

    @jit
    def _run_(y, args):
        return diffeqsolve(
            terms=ODETerm(VectorField(c_light, dx, dt, xax, drivers)),
            solver=Stepper(),
            t0=0.0,
            t1=tmax,
            max_steps=nt + 5,
            dt0=dt,
            y0=y,
            args=args,
        )

    test_a = _run_({"a": a, "prev_a": a_old}, args)

    np.testing.assert_almost_equal(np.sum(np.square(test_a.ys["a"])), 0.0, decimal=8)


if __name__ == "__main__":
    test_absorbing_boundaries()
