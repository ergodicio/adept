from typing import Dict, Callable
import jax.numpy as jnp
import equinox as eqx

from adept._tf1d.solvers import pushers


class VF(eqx.Module):
    """
    This function returns the function that defines $d_state / dt$

    All the pushers are chosen and initialized here and a single time-step is defined here.

    We use the time-integrators provided by diffrax, and therefore, only need $d_state / dt$ here

    :param cfg:
    :return:
    """

    cfg: Dict
    pusher_dict: Dict
    push_driver: Callable
    poisson_solver: Callable

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pusher_dict = {"ion": {}, "electron": {}}
        for species_name in ["ion", "electron"]:
            self.pusher_dict[species_name]["push_n"] = pushers.DensityStepper(cfg["grid"]["kx"])
            self.pusher_dict[species_name]["push_u"] = pushers.VelocityStepper(
                cfg["grid"]["kx"], cfg["grid"]["kxr"], cfg["grid"]["one_over_kxr"], cfg["physics"][species_name]
            )
            self.pusher_dict[species_name]["push_e"] = pushers.EnergyStepper(
                cfg["grid"]["kx"], cfg["physics"][species_name]
            )
            if cfg["physics"][species_name]["trapping"]["is_on"]:
                self.pusher_dict[species_name]["particle_trapper"] = pushers.ParticleTrapper(cfg, species_name)

        self.push_driver = pushers.Driver(cfg["grid"]["x"])
        # if "ey" in self.cfg["drivers"]:
        #     self.wave_solver = pushers.WaveSolver(cfg["grid"]["c"], cfg["grid"]["dx"], cfg["grid"]["dt"])
        self.poisson_solver = pushers.PoissonSolver(cfg["grid"]["one_over_kx"])

    def __call__(self, t: float, y: Dict, args: Dict):
        """
        This function is used by the time integrators specified in diffrax

        :param t:
        :param y:
        :param args:
        :return:
        """
        e = self.poisson_solver(
            self.cfg["physics"]["ion"]["charge"] * y["ion"]["n"]
            + self.cfg["physics"]["electron"]["charge"] * y["electron"]["n"]
        )
        ed = 0.0

        for p_ind in self.cfg["drivers"]["ex"].keys():
            ed += self.push_driver(args["drivers"]["ex"][p_ind], t)

        # if "ey" in self.cfg["drivers"]:
        #     ad = 0.0
        #     for p_ind in self.cfg["drivers"]["ey"].keys():
        #         ad += self.push_driver(args["pulse"]["ey"][p_ind], t)
        #     a = self.wave_solver(a, aold, djy_array, charge)
        #     total_a = y["a"] + ad
        #     ponderomotive_force = -0.5 * jnp.gradient(jnp.square(total_a), self.cfg["grid"]["dx"])[1:-1]

        total_e = e + ed  # + ponderomotive_force

        dstate_dt = {"ion": {}, "electron": {}}
        for species_name in ["ion", "electron"]:
            n = y[species_name]["n"]
            u = y[species_name]["u"]
            p = y[species_name]["p"]
            delta = y[species_name]["delta"]
            if self.cfg["physics"][species_name]["is_on"]:
                q_over_m = self.cfg["physics"][species_name]["charge"] / self.cfg["physics"][species_name]["mass"]
                p_over_m = p / self.cfg["physics"][species_name]["mass"]

                dstate_dt[species_name]["n"] = self.pusher_dict[species_name]["push_n"](n, u)
                dstate_dt[species_name]["u"] = self.pusher_dict[species_name]["push_u"](
                    n, u, p_over_m, q_over_m * total_e, delta
                )
                dstate_dt[species_name]["p"] = self.pusher_dict[species_name]["push_e"](n, u, p_over_m, q_over_m * e)
            else:
                dstate_dt[species_name]["n"] = jnp.zeros(self.cfg["grid"]["nx"])
                dstate_dt[species_name]["u"] = jnp.zeros(self.cfg["grid"]["nx"])
                dstate_dt[species_name]["p"] = jnp.zeros(self.cfg["grid"]["nx"])

            if self.cfg["physics"][species_name]["trapping"]["is_on"]:
                dstate_dt[species_name]["delta"] = self.pusher_dict[species_name]["particle_trapper"](e, delta, args)
            else:
                dstate_dt[species_name]["delta"] = jnp.zeros(self.cfg["grid"]["nx"])

        return dstate_dt
