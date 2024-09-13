from typing import Dict

from jax import numpy as jnp, Array
import numpy as np

from adept import get_envelope
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
        self.epw = epw.SpectralPotential(cfg)
        self.light = laser.Light(cfg)
        self.complex_state_vars = ["E0", "epw"]
        self.boundary_envelope = cfg["grid"]["absorbing_boundaries"]
        self.one_over_ksq = cfg["grid"]["one_over_ksq"]
        self.zero_mask = cfg["grid"]["zero_mask"]
        self.low_pass_filter = cfg["grid"]["low_pass_filter"]

        self.nu_coll = cfg["units"]["derived"]["nu_coll"]

    def _unpack_y_(self, y: Dict[str, Array]) -> Dict[str, Array]:
        new_y = {}
        for k in y.keys():
            if k in self.complex_state_vars:
                new_y[k] = y[k].view(jnp.complex128)
            else:
                new_y[k] = y[k].view(jnp.float64)
        return new_y

    def _pack_y_(self, y: Dict[str, Array], new_y: Dict[str, Array]) -> tuple[Dict[str, Array], Dict[str, Array]]:
        for k in y.keys():
            y[k] = y[k].view(jnp.float64)
            new_y[k] = new_y[k].view(jnp.float64)

        return y, new_y

    def light_split_step(self, t, y, driver_args):
        if "E0" in driver_args:
            t_coeff = get_envelope(
                driver_args["E0"]["tr"],
                driver_args["E0"]["tr"],
                driver_args["E0"]["tc"] - driver_args["E0"]["tw"] / 2,
                driver_args["E0"]["tc"] + driver_args["E0"]["tw"] / 2,
                t,
            )
            y["E0"] = self.boundary_envelope[..., None] * t_coeff * self.light.laser_update(t, y, driver_args["E0"])

        # if self.cfg["terms"]["light"]["update"]:
        # y["E0"] = y["E0"] + self.dt * jnp.real(k1_E0)

        # t_coeff = get_envelope(0.1, 0.1, 0.2, 100.0, t + 0.5 * self.dt)
        # y["E0"] = t_coeff * self.light.laser_update(t + 0.5 * self.dt, y, args["E0"])
        # if self.cfg["terms"]["light"]["update"]:
        # y["E0"] = y["E0"] + 1j * self.dt * jnp.imag(k1_E0)

        return y

    def landau_damping(self, epw: Array, vte_sq: float):
        gammaLandauEpw = (
            np.sqrt(np.pi / 8)
            * self.wp0**4
            * self.one_over_ksq**1.5
            / (vte_sq**1.5)
            * jnp.exp(-self.wp0**2.0 * self.one_over_ksq / (2 * vte_sq))
        )

        return jnp.fft.ifft2(jnp.fft.fft2(epw) * jnp.exp(-(gammaLandauEpw + self.nu_coll) * self.dt))

    def __call__(self, t, y, args):
        # unpack y into complex128
        new_y = self._unpack_y_(y)

        # split step
        new_y = self.light_split_step(t, new_y, args["drivers"])

        if "E2" in args["drivers"]:
            new_y["epw"] += self.dt * self.epw.driver(args["drivers"]["E2"], t)
        new_y["epw"] = self.epw(t, new_y, args)

        # landau and collisional damping
        if self.cfg["terms"]["epw"]["damping"]["landau"]:
            new_y["epw"] = self.landau_damping(epw=new_y["epw"], vte_sq=y["vte_sq"])

        # boundary damping
        ex, ey = self.epw.calc_fields_from_phi(new_y["epw"])
        ex = ex * self.boundary_envelope
        ey = ey * self.boundary_envelope
        new_y["epw"] = self.epw.calc_phi_from_fields(ex, ey)
        new_y["epw"] = jnp.fft.ifft2(self.zero_mask * self.low_pass_filter * jnp.fft.fft2(new_y["epw"]))

        # pack y into float64
        y, new_y = self._pack_y_(y, new_y)

        return new_y
