from typing import Dict


import os

import diffrax
import numpy as np
import equinox as eqx
from equinox.internal import ω
from diffrax import Euler

from jax import numpy as jnp

from adept.sh2d.solvers import vlasov, field, fokker_planck


def get_derived_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dy"] = cfg_grid["ymax"] / cfg_grid["ny"]
    cfg_grid["dv"] = cfg_grid["vmax"] / cfg_grid["nv"]
    cfg_grid["dt"] = cfg_grid["tmax"] / cfg_grid["nt"]
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    return cfg_grid


def get_solver_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are arrays

    This is run after the log params step

    :param cfg_grid:
    :return:
    """

    cfg_grid = {
        **cfg_grid,
        **{
            "x": jnp.linspace(
                cfg_grid["xmin"] + cfg_grid["dx"] / 2, cfg_grid["xmax"] - cfg_grid["dx"] / 2, cfg_grid["nx"]
            ),
            "y": jnp.linspace(
                cfg_grid["ymin"] + cfg_grid["dy"] / 2, cfg_grid["ymax"] - cfg_grid["dy"] / 2, cfg_grid["ny"]
            ),
            "v": jnp.linspace(cfg_grid["dv"] / 2, cfg_grid["vmax"] - cfg_grid["dv"] / 2, cfg_grid["nv"]),
            "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "ky": jnp.fft.fftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
            "kyr": jnp.fft.rfftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
            "kv": jnp.fft.fftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
            "kvr": jnp.fft.rfftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
        },
    }

    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

    one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

    one_over_ky = np.zeros_like(cfg_grid["ky"])
    one_over_ky[1:] = 1.0 / cfg_grid["ky"][1:]
    cfg_grid["one_over_ky"] = jnp.array(one_over_ky)

    one_over_kyr = np.zeros_like(cfg_grid["kyr"])
    one_over_kyr[1:] = 1.0 / cfg_grid["kyr"][1:]
    cfg_grid["one_over_kyr"] = jnp.array(one_over_kyr)

    one_over_kv = np.zeros_like(cfg_grid["kv"])
    one_over_kv[1:] = 1.0 / cfg_grid["kv"][1:]
    cfg_grid["one_over_kv"] = jnp.array(one_over_kv)

    one_over_kvr = np.zeros_like(cfg_grid["kvr"])
    one_over_kvr[1:] = 1.0 / cfg_grid["kvr"][1:]
    cfg_grid["one_over_kvr"] = jnp.array(one_over_kvr)

    return cfg_grid


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    # cfg["save"]["func"] = {**cfg["save"]["func"], **{"callable": get_save_func(cfg)}}
    cfg["save"]["t"]["ax"] = jnp.linspace(cfg["save"]["t"]["tmin"], cfg["save"]["t"]["tmax"], cfg["save"]["t"]["nt"])
    # cfg["save"]["x"]["ax"] = jnp.linspace(cfg["save"]["x"]["xmin"], cfg["save"]["x"]["xmax"], cfg["save"]["x"]["nx"])
    # cfg["save"]["y"]["ax"] = jnp.linspace(cfg["save"]["y"]["ymin"], cfg["save"]["y"]["ymax"], cfg["save"]["y"]["ny"])

    return cfg


def init_state(cfg: Dict) -> tuple[Dict, Dict]:
    """
    This function initializes the state

    :param cfg:
    :return:
    """

    nx = cfg["grid"]["nx"]
    ny = cfg["grid"]["ny"]
    nv = cfg["grid"]["nv"]
    norm = 1.0 / (
        4.0 * jnp.pi * cfg["grid"]["dv"] * jnp.sum(cfg["grid"]["v"] ** 2.0 * jnp.exp(-cfg["grid"]["v"] ** 2.0 / 2.0))
    )

    density_profile = jnp.ones((nx, ny))  # cfg["profile"]["density"]
    temperature_profile = jnp.ones((nx, ny))  # cfg["profile"]["temperature"]
    state = {}
    state["flm"] = {}
    state["flm"][0] = {}
    state["flm"][0][0] = (
        norm
        * density_profile[:, :, None]
        * jnp.exp(-cfg["grid"]["v"][None, None, :] ** 2.0 / 2.0 / temperature_profile[:, :, None])
    )
    state["flm"][0][0] = jnp.array(state["flm"][0][0], dtype=jnp.complex128).view(dtype=jnp.float64)

    for il in range(1, cfg["grid"]["nl"] + 1):
        state["flm"][il] = {}
        for im in range(0, il + 1):
            state["flm"][il][im] = jnp.zeros((nx, ny, nv), dtype=jnp.complex128).view(dtype=jnp.float64)

    state["e"] = jnp.zeros((nx, ny, 3))
    state["b"] = jnp.zeros((nx, ny, 3))
    state["de"] = jnp.zeros((nx, ny, 3))
    state["db"] = jnp.zeros((nx, ny, 3))

    return state, {"drivers": cfg["drivers"]}


class FokkerPlanckVectorField(eqx.Module):
    """
    :param cfg:
    :return:
    """

    cfg: Dict
    fp_flm: eqx.Module
    fp_f00: eqx.Module

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp_flm = fokker_planck.AnisotropicCollisions(cfg)
        self.fp_f00 = fokker_planck.IsotropicCollisions(cfg)

    def __call__(self, t: float, y: Dict, args: Dict):
        """
        :param t:
        :param y:
        :param args:
        :return:
        """
        if self.cfg["terms"]["fokker-planck"]["active"]:
            y["flm"][0][0] = y["flm"][0][0].view(dtype=jnp.complex128)
            y["flm"][0][0] = self.fp_f00(t, y["flm"][0][0], args)

            for il in range(1, self.cfg["grid"]["nl"] + 1):
                for im in range(0, il + 1):
                    y["flm"][il][im] = y["flm"][il][im].view(dtype=jnp.complex128)
            y["flm"] = self.fp_flm(t, y["flm"], args)

            for il in range(0, self.cfg["grid"]["nl"] + 1):
                for im in range(0, il + 1):
                    y["flm"][il][im] = y["flm"][il][im].view(dtype=jnp.float64)

        return y


class VlasovVectorField(eqx.Module):
    """
    This function returns the function that defines $d_state / dt$

    All the pushers are chosen and initialized here and a single time-step is defined here.

    We use the time-integrators provided by diffrax, and therefore, only need $d_state / dt$ here

    :param cfg:
    :return:
    """

    cfg: Dict

    push_vlasov: eqx.Module
    push_driver: eqx.Module
    # poisson_solver: eqx.Module
    ampere_solver: eqx.Module

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.push_vlasov = vlasov.Vlasov(cfg)
        self.push_driver = vlasov.Driver(cfg["grid"]["x"], cfg["grid"]["y"])
        # cfg["profiles"]["ion_charge"]
        # self.poisson_solver = field.SpectralPoissonSolver(
        #     jnp.ones((cfg["grid"]["nx"], cfg["grid"]["ny"])),
        #     cfg["grid"]["one_over_kx"],
        #     cfg["grid"]["one_over_ky"],
        #     cfg["grid"]["dv"],
        #     cfg["grid"]["v"],
        # )
        self.ampere_solver = field.AmpereSolver(cfg)

    def __call__(self, t: float, y: Dict, args: Dict):
        """
        This function is used by the time integrators specified in diffrax

        :param t:
        :param y:
        :param args:
        :return:
        """
        for il in range(0, self.cfg["grid"]["nl"] + 1):
            for im in range(0, il + 1):
                y["flm"][il][im] = y["flm"][il][im].view(dtype=jnp.complex128)

        # y["e"] = self.poisson_solver(y["flm"][0][0])
        this_j = -jnp.real(self.ampere_solver(t, y, args))
        # y["b"] = self.b_solver(y)

        ed = 0.0

        for p_ind in self.cfg["drivers"]["ex"].keys():
            ed += self.push_driver(args["driver"]["ex"][p_ind], t)

        ed = jnp.concatenate([ed[:, :, None], jnp.zeros_like(ed[:, :, None]), jnp.zeros_like(ed[:, :, None])], axis=-1)

        total_e = y["e"] + ed
        total_b = y["b"] + args["b_ext"]

        dydt = {
            "flm": self.push_vlasov(y["flm"], total_e, total_b),
            "e": this_j,
            "b": jnp.zeros_like(total_e),
            "de": jnp.zeros_like(total_e),
            "db": jnp.zeros_like(total_e),
        }

        for il in range(0, self.cfg["grid"]["nl"] + 1):
            for im in range(0, il + 1):
                y["flm"][il][im] = y["flm"][il][im].view(dtype=jnp.float64)
                dydt["flm"][il][im] = dydt["flm"][il][im].view(dtype=jnp.float64)

        return dydt


class ExplicitEStepper(Euler):
    def step(self, term: diffrax.MultiTerm, t0, t1, y0, args, solver_state, made_jump):
        vlasov_term, collision_term = term.terms

        # control1 = vlasov_term.contr(t0, t1)
        # control2 = collision_term.contr(t0, t1)
        # y1, _, dense_info, _, success = self.vlasov_stepper.step(
        #     vlasov_term, t0, t1, y0, args, solver_state, made_jump
        # )
        y1 = (y0**ω + vlasov_term.vf_prod(t0, y0, args, vlasov_term.contr(t0, t1)) ** ω).ω
        # y1, _, dense_info, _, success = self.vlasov_stepper.step(term_2, t0, t1, y0, args, solver_state, made_jump)
        y1 = collision_term.vf(t0, y1, args)

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful


class ImplicitEStepper(Euler):
    def __init__(self, cfg):
        self.vlasov_stepper = diffrax.Tsit5()
        self.field_solver = None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        vlasov_without_e, vlasov_e, collisions, implicit_e_solve = terms
        y_after_vlasov_part_a = vlasov_without_e.vf_prod(t0, y0, args, vlasov.contr(t0, t1))

        de = self.calc_de(y_after_vlasov_part_a)

        # get j0
        j0 = self.calc_j0(y_after_vlasov_part_a)

        # get jdEx
        y_after_vlasov_part_a["e"] = jnp.concatenate([de[..., :1], self.zeros, self.zeros])
        e_dex = vlasov_e.vf_prod(t0, y_after_vlasov_part_a, args, vlasov_e.contr(t0, t1))

        # get jdEy
        y_after_vlasov_part_a["e"] = jnp.concatenate([self.zeros, de[..., 1:2], self.zeros])
        jdey = vlasov_e.vf_prod(t0, y_after_vlasov_part_a, args, vlasov_e.contr(t0, t1))

        # get jdEz
        y_after_vlasov_part_a["e"] = jnp.concatenate([self.zeros, self.zeros, de[..., 2:]])
        jdez = vlasov_e.vf_prod(t0, y_after_vlasov_part_a, args, vlasov_e.contr(t0, t1))

        # get JN

        # solve 3x3 systems
        # y["e"] = implicit_e_solve.vf_prod()

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful
