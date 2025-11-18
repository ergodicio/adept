import numpy as np
from jax import Array
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._lpse2d.core import epw, laser


class SplitStep:
    """
    This class contains the function that updates the state

    All the pushers are chosen and initialized here and a single time-step is defined here.

    :param cfg:
    :return:
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg["grid"]["dt"]
        self.wp0 = cfg["units"]["derived"]["wp0"]
        # self.epw = epw.SpectralPotential(cfg)
        self.epw = epw.SpectralEPWSolver(cfg)
        self.light = laser.Light(cfg)
        self.complex_state_vars = ["E0", "epw", "E1"]
        self.boundary_envelope = cfg["grid"]["absorbing_boundaries"]
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]
        self.zero_mask = cfg["grid"]["zero_mask"]
        self.low_pass_filter = cfg["grid"]["low_pass_filter"]
        self.k_sq = cfg["grid"]["kx"][:, None] ** 2 + cfg["grid"]["ky"][None, :] ** 2
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]

        self.envelope_density = cfg["units"]["envelope density"]
        self.e = cfg["units"]["derived"]["e"]
        self.me = cfg["units"]["derived"]["me"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.phi_laplacian = "spectral"  # hard coded for now, can be implemented to config if ever necessary
        self.background_density = cfg["grid"]["background_density"]

    def _unpack_y_(self, y: dict[str, Array]) -> dict[str, Array]:
        new_y = {}
        for k in y.keys():
            if k in self.complex_state_vars:
                new_y[k] = y[k].view(jnp.complex128)
            else:
                new_y[k] = y[k].view(jnp.float64)
        return new_y

    def _pack_y_(self, y: dict[str, Array], new_y: dict[str, Array]) -> tuple[dict[str, Array], dict[str, Array]]:
        for k in y.keys():
            y[k] = y[k].view(jnp.float64)
            new_y[k] = new_y[k].view(jnp.float64)

        return y, new_y

    def get_envelope_coefficient(self, envelope_args, t):
        return get_envelope(
            envelope_args["tr"],
            envelope_args["tr"],
            envelope_args["tc"] - envelope_args["tw"] / 2,
            envelope_args["tc"] + envelope_args["tw"] / 2,
            t,
        )

    def light_split_step(self, t, y, driver_args):
        if "E0" in driver_args:
            t_coeff = self.get_envelope_coefficient(driver_args["E0"], t)
            y["E0"] = t_coeff * self.light.laser_update(t, y, driver_args["E0"])

        y["E1"] *= self.boundary_envelope[..., None]

        return y

    def __call__(self, t, y, args):
        # unpack y into complex128
        new_y = self._unpack_y_(y)

        # light split step
        new_y = self.light_split_step(t, new_y, args["drivers"])

        if "E2" in args["drivers"]:
            new_y["epw"] += jnp.fft.fft2(self.dt * self.epw.driver(args["drivers"]["E2"], t))
        # epw split step
        new_y["epw"] = self.epw(t, new_y, args)

        # pack y into float64
        y, new_y = self._pack_y_(y, new_y)

        return new_y
