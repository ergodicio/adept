from jax import numpy as jnp


class IdealGas:
    def __init__(self, cfg):
        self.sigma_sb = 4.0
        self.c = 4.0

    def __call__(self, ne, ni, Te, Ti, Tr):
        pi = ni * Ti
        pe = ne * Te
        pr = 4.0 / 3.0 * self.sigma_sb / self.c * Tr**4.0

        return pi, pe, pr


class Temperature:
    def __init__(self, cfg):
        self.dt = cfg["grid"]["dt"]

    def get_artifical_viscosity(self, rho, u):
        return jnp.where(rho * jnp.abs(du) * (cq * jnp.abs(du) + cL * cs), du < 0, 0.0)

    def __call__(self, rho, u, r, Ti, Te, Tr, pi, pe, pr):
        artificial_viscosity = self.get_artifical_viscosity(rho, u)
        delta_r = r[1:] - r[:-1]
        new_Ti = Ti - self.dt / delta_r[:-1] * (self.gamma - 1) * new_vol / ni * (
            0.5 * pi + artificial_viscosity + self.eta0 * (u[2:] - u[1:-1]) / new_delta_r
        ) / ((r**2.0 * u)[1:-1] - (r**2.0 * u)[:-2])

        new_Te = Te - self.dt / delta_r[:-1] * (self.gamma - 1) * new_vol / ne * (0.5 * pe) / (
            (r**2.0 * u)[1:-1] - (r**2.0 * u)[:-2]
        )

        new_Tr = Tr - self.dt / delta_r[:-1] * self.c / (self.sigma_sb * Tr**3.0) * (0.5 * pr) / (
            (r**2.0 * u)[1:-1] - (r**2.0 * u)[:-2]
        )

        new_Ti = self.diffusion_solve(kappa_i, new_Ti)
        new_Te = self.diffusion_solve(kappa_e, new_Te)
        new_Tr = self.diffusion_solve(kappa_r, new_Tr)

        return self.couple_temperatures(new_Ti, new_Te, new_Tr)
