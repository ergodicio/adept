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
        self.epw = epw.SpectralPotential(cfg)
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
        self.nu_coll = cfg["units"]["derived"]["nu_coll"]
        self.phi_laplacian = "spectral"  # hard coded for now, can be implemented to config if ever necessary

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

    def scattered_light_update(self, t, y):
        E0 = y["E0"]
        E0_x = E0[..., 0]
        E0_y = E0[..., 1]
        E1 = y["E1"]
        E1x = jnp.pad(E1[..., 0], ((1, 1), (1, 1)), mode="wrap")
        E1y = jnp.pad(E1[..., 1], ((1, 1), (1, 1)), mode="wrap")
        phi = y["epw"]
        c = self.cfg["units"]["derived"]["c"]
        w1 = self.cfg["units"]["derived"]["w1"]
        dx = self.cfg["grid"]["dx"]
        dy = self.cfg["grid"]["dy"]
        backgroundDensityPerturbation = y["background_density"] / self.envelope_density - 1

        Nelf = 0.0  # self.cfg["Nelf"]

        # Implement the scattered light update logic here
        # these are derivative calculations, rewrite using numpy slicing and broadcasting
        ixc = iyc = slice(1, -1)
        ixp = iyp = slice(2, None)
        ixm = iym = slice(None, -2)

        k_E1_x = (
            1j * c**2 / (2 * w1) * (E1x[ixc, iyp] - 2 * E1x[ixc, iyc] + E1x[ixc, iym]) / dy**2
            - 1j * c**2 / (2 * w1) * (E1y[ixp, iyp] - E1y[ixm, iyp] - E1y[ixp, iym] + E1y[ixm, iym]) / (4 * dy * dx)
            + 1j * w1 / 2 * (1 - self.wp0**2 / w1**2 * (1 + backgroundDensityPerturbation + Nelf)) * E1x[ixc, iyc]
        )

        k_E1_y = (
            1j * c**2 / (2 * w1) * (E1y[ixp, iyc] - 2 * E1y[ixc, iyc] + E1y[ixm, iyc]) / dx**2
            - 1j * c**2 / (2 * w1) * (E1x[ixp, iyp] - E1x[ixm, iyp] - E1x[ixp, iym] + E1x[ixm, iym]) / (4 * dy * dx)
            + 1j * w1 / 2 * (1 - self.wp0**2 / w1**2 * (1 + backgroundDensityPerturbation + Nelf)) * E1y[ixc, iyc]
        )

        # calculate 2d laplacian

        if self.phi_laplacian == "fd":
            padded_phi = jnp.pad(phi, ((1, 1), (1, 1)), mode="wrap")
            laplacianPhi = (padded_phi[ixp, iyc] + padded_phi[ixm, iyc] - 2 * padded_phi[ixc, iyc]) / dx**2.0
            laplacianPhi += (padded_phi[ixc, iyp] + padded_phi[ixc, iym] - 2 * padded_phi[ixc, iyc]) / dy**2.0
        elif self.phi_laplacian == "spectral":
            laplacianPhi = (
                jnp.fft.ifft2(
                    -self.k_sq * jnp.fft.fft2(phi)  # * self.low_pass_filter * self.zero_mask
                )
            ).real
        else:
            raise NotImplementedError("phi_laplacian method not implemented")

        coeff = 1j * self.e / (4 * self.w0 * self.me) * jnp.conj(laplacianPhi)
        k_E1_x -= coeff * E0_x
        k_E1_y -= coeff * E0_y

        return None, jnp.concatenate([k_E1_x[..., None], k_E1_y[..., None]], axis=-1)

    def light_split_step(self, t, y, driver_args):
        if "E0" in driver_args:
            t_coeff = get_envelope(
                driver_args["E0"]["tr"],
                driver_args["E0"]["tr"],
                driver_args["E0"]["tc"] - driver_args["E0"]["tw"] / 2,
                driver_args["E0"]["tc"] + driver_args["E0"]["tw"] / 2,
                t,
            )
            y["E0"] = t_coeff * self.light.laser_update(t, y, driver_args["E0"])

        if self.cfg["terms"]["epw"]["source"]["srs"]:
            [k1_E0, k1_E1] = self.scattered_light_update(t, y)
            # y["E0"] += self.dt * jnp.real(k1_E0)
            y["E1"] += self.dt * jnp.real(k1_E1)

            yE0t = y["E0"]

            t_coeff = get_envelope(
                driver_args["E0"]["tr"],
                driver_args["E0"]["tr"],
                driver_args["E0"]["tc"] - driver_args["E0"]["tw"] / 2,
                driver_args["E0"]["tc"] + driver_args["E0"]["tw"] / 2,
                t + self.dt / 2.0,
            )
            y["E0"] = t_coeff * self.light.laser_update(t + self.dt / 2.0, y, driver_args["E0"])

            [k1_E0, k1_E1] = self.scattered_light_update(t + self.dt / 2, y)
            # y["E0"] += self.dt * jnp.imag(k1_E0)
            y["E1"] += self.dt * 1j * jnp.imag(k1_E1)

        # if self.cfg["terms"]["light"]["update"]:
        # y["E0"] = y["E0"] + self.dt * jnp.real(k1_E0)

        # t_coeff = get_envelope(0.1, 0.1, 0.2, 100.0, t + 0.5 * self.dt)
        # y["E0"] = t_coeff * self.light.laser_update(t + 0.5 * self.dt, y, args["E0"])
        # if self.cfg["terms"]["light"]["update"]:
        # y["E0"] = y["E0"] + 1j * self.dt * jnp.imag(k1_E0)

        return y

    def landau_damping(self, epw: Array, vte_sq: float):
        gammaLandauEpw = (
            jnp.sqrt(np.pi / 8)
            * (1.0 + 1.5 * self.k_sq * (vte_sq / self.wp0**2))
            * self.wp0**4
            * self.one_over_ksq**1.5
            / vte_sq**1.5
            * jnp.exp(-(1.5 + 0.5 * self.wp0**2 * self.one_over_ksq / vte_sq))
        )

        return jnp.fft.ifft2(
            jnp.fft.fft2(epw) * jnp.exp(-(gammaLandauEpw + self.nu_coll) * self.dt) * self.low_pass_filter
        )

    def __call__(self, t, y, args):
        # unpack y into complex128
        new_y = self._unpack_y_(y)

        if self.cfg["terms"]["epw"]["damping"]["landau"]:
            new_y["epw"] = self.landau_damping(epw=new_y["epw"], vte_sq=y["vte_sq"])

        # split step
        new_y = self.light_split_step(t, new_y, args["drivers"])

        if "E2" in args["drivers"]:
            new_y["epw"] += self.dt * self.epw.driver(args["drivers"]["E2"], t)
        new_y["epw"] = self.epw(t, new_y, args)

        # landau and collisional damping

        # boundary damping
        ex, ey = self.epw.calc_fields_from_phi(new_y["epw"])
        ex = ex * self.boundary_envelope
        ey = ey * self.boundary_envelope
        new_y["E1"] *= self.boundary_envelope[..., None]
        new_y["epw"] = self.epw.calc_phi_from_fields(ex, ey)
        # new_y["epw"] = jnp.fft.ifft2(self.zero_mask * self.low_pass_filter * jnp.fft.fft2(new_y["epw"]))

        # pack y into float64
        y, new_y = self._pack_y_(y, new_y)

        return new_y
