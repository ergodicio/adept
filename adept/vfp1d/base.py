import numpy as np
from diffrax import ODETerm, SaveAt, SubSaveAt, diffeqsolve
from jax import numpy as jnp

from adept._base_ import ADEPTModule, Stepper
from adept.normalization import UREG, laser_normalization
from adept.utils import filter_scalars
from adept.vfp1d.grid import Grid
from adept.vfp1d.helpers import _initialize_total_distribution_, calc_logLambda
from adept.vfp1d.storage import get_save_quantities, post_process
from adept.vfp1d.vector_field import OSHUN1D


class BaseVFP1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.plasma_norm = laser_normalization(
            cfg["units"]["laser_wavelength"], cfg["units"]["reference electron temperature"]
        )
        self.grid = Grid.from_config(cfg["grid"], self.plasma_norm)

    def post_process(self, solver_result: dict, td: str) -> dict:
        return post_process(solver_result["solver result"], cfg=self.cfg, td=td, args=self.args)

    def write_units(self) -> dict:
        norm = self.plasma_norm
        Z = self.cfg["units"]["Z"]
        ne = UREG.Quantity(self.cfg["units"]["reference electron density"]).to("1/cc")
        Te_eV = norm.T0.to("eV")
        logLambda_ei, logLambda_ee = calc_logLambda(
            self.cfg, ne, Te_eV, Z, self.cfg["units"]["Ion"], force_ee_equal_ei=True
        )

        # collision frequency nu_ee0 of electron moving at speed of light
        # normalised to background plasma frequency ω_p0
        #    nuee0 = 4π n0 r_e^2 c logΛ_ee normalised to plasma frequency ω_p0 = √(4πn0 r_e))
        # => nuee0/ω_p0 = r_e ω_p0 logΛ_ee / c = k_p0 r_e logΛ_ee, where k_p0 = ω_p/c
        r_e = 2.8179403205e-13 * UREG.cm  # Classical electron radius (CODATA 2022)
        kpre = (r_e * np.sqrt(4 * np.pi * norm.n0 * r_e)).to("").magnitude
        nuee_coeff = float(kpre * logLambda_ee)

        # Physical plasma frequency at ne (not the normalization frequency 1/tau)
        wp0 = ((ne * UREG.e**2 / (UREG.m_e * UREG.epsilon_0)) ** 0.5).to("rad/s")
        vth = ((2.0 * norm.T0 / UREG.m_e) ** 0.5).to("m/s")
        vth_norm = norm.vth_norm()
        # Elementary charge in Gaussian CGS (pint cannot convert SI↔Gaussian charge dimensions)
        e_gauss = UREG.Quantity(4.803204712570263e-10, "Fr")

        nD_NRL = 1.72e9 * (Te_eV / UREG.electronvolt) ** 1.5 / np.sqrt(ne * UREG.cc)

        nuei_shk = np.sqrt(2.0 / np.pi) * wp0 * logLambda_ei / np.exp(logLambda_ei)
        # JPB - Maybe comment which page/eq this is from? There are lots of collision times in NRL
        # For example, nu_ei on page 32 does include Z^2
        nuei_nrl = np.sqrt(2.0 / np.pi) * wp0 * logLambda_ei / nD_NRL

        nuei_epphaines = (
            1
            / (
                0.75
                * UREG.m_e**0.5
                * Te_eV**1.5
                / (np.sqrt(2 * np.pi) * (ne / Z) * Z**2.0 * e_gauss**4.0 * logLambda_ei)
            )
        ).to("Hz")

        # Normalised v_osc^2 per unit intensity:
        #   v_osc^2_norm = K * I * lam^2 / (alpha * T)
        # where K = e^2 / (m_e * 2 * 4*pi^2 * c^3 * eps_0)
        #         = 0.093373  [at I in 1e15 W/cm^2, lam in um, T in keV]
        # Derived from fundamental constants (CODATA 2022, see Wikipedia
        # "Ponderomotive energy").  Ridgers (2011) uses 0.091 (older constants).
        lam0 = UREG.Quantity(self.cfg["units"]["laser_wavelength"]).to("um")
        ib_cfg = self.cfg.get("drivers", {}).get("ib", {})
        polarisation = ib_cfg.get("polarisation", "linear")
        if polarisation == "linear":
            alpha_pol = 1.0
        elif polarisation == "circular":
            alpha_pol = 0.5
        else:
            alpha_pol = float(polarisation)
        vosc2_per_intensity = float(
            (0.093373 * (lam0 / UREG.um) ** 2 / (alpha_pol * (Te_eV / UREG.keV))).to("").magnitude
        )  # per 10¹⁵ W/cm²

        # Normalised laser frequency ω₀ = ω_L / ω_p (derived from laser_wavelength)
        # ω_L = 2πc/λ₀, ω_p = 1/τ
        w0_norm = float((2 * np.pi * UREG.c / lam0 * norm.tau).to(""))

        all_quantities = {
            "wp0": wp0,
            "n0": norm.n0.to("1/cc"),
            "tp0": norm.tau.to("fs"),
            "ne": ne,
            "vth": vth,
            "Te": Te_eV,
            "Ti": UREG.Quantity(self.cfg["units"]["reference ion temperature"]).to("eV"),
            "logLambda_ei": logLambda_ei,
            "logLambda_ee": logLambda_ee,
            "vth_norm": vth_norm,
            "x0": norm.L0.to("nm"),
            "nuei_shk": nuei_shk,
            "nuei_nrl": nuei_nrl,
            "nuei_epphaines": nuei_epphaines,
            "nuei_shk_norm": (nuei_shk * norm.tau).to(""),
            "nuei_nrl_norm": (nuei_nrl * norm.tau).to(""),
            "nuei_epphaines_norm": (nuei_epphaines * norm.tau).to(""),
            "lambda_mfp_shk": (vth / nuei_shk).to("micron"),
            "lambda_mfp_nrl": (vth / nuei_nrl).to("micron"),
            "lambda_mfp_epphaines": (vth / nuei_epphaines).to("micron"),
            "nD_NRL": nD_NRL,
            "nD_Shkarofsky": np.exp(logLambda_ei) * Z / 9,
            "nuee_coeff": nuee_coeff,
            "logLam_ratio": logLambda_ei / logLambda_ee,
            "vosc2_per_intensity": vosc2_per_intensity,
            "w0_norm": w0_norm,
        }

        self.cfg["units"]["derived"] = all_quantities
        self.cfg["grid"]["vth_norm"] = vth_norm

        return {k: str(v) for k, v in all_quantities.items()}

    def get_derived_quantities(self):
        """Sync scalar grid values into cfg for logging.

        This is run prior to the log params step.
        """
        cfg_grid = self.cfg["grid"]
        grid = self.grid

        # Default save.*.t.tmin/tmax to computed grid values
        for save_type in self.cfg.get("save", {}).keys():
            if "t" in self.cfg["save"][save_type]:
                t_cfg = self.cfg["save"][save_type]["t"]
                t_cfg.setdefault("tmin", cfg_grid.get("tmin", "0ps"))
                t_cfg.setdefault("tmax", cfg_grid["tmax"])

        # Merge scalar grid values into cfg for logging (arrays come later in get_solver_quantities)
        cfg_grid.update(filter_scalars(grid.as_dict()))

        print("tmax", grid.tmax, "dt", grid.dt)
        print("xmax", grid.xmax, "dx", grid.dx)

        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self):
        """Merge all grid values (including arrays) into cfg['grid'] for backward compatibility."""
        self.cfg["grid"].update(self.grid.as_dict())

    def init_state_and_args(self) -> dict:
        grid = self.grid
        f0, f10, ne_prof = _initialize_total_distribution_(
            self.cfg, grid, self.plasma_norm.vth_norm(), self.plasma_norm
        )

        # Scale density to physical ne (f10 is invariant in the big-dt collision limit)
        ne = UREG.Quantity(self.cfg["units"]["reference electron density"])
        ne_over_n0 = (ne / self.plasma_norm.n0).to("").magnitude
        f0 *= ne_over_n0
        ne_prof *= ne_over_n0

        state = {"f0": f0}
        # not currently necessary but kept for completeness
        for il in range(1, grid.nl + 1):
            for im in range(0, il + 1):
                state[f"f{il}{im}"] = jnp.zeros((grid.nx + 1, grid.nv))

        state["f10"] = f10

        for field in ["e", "b"]:
            state[field] = jnp.zeros(grid.nx + 1)

        state["Z"] = jnp.ones(grid.nx)
        state["ni"] = ne_prof / self.cfg["units"]["Z"]

        self.state = state
        self.args = {"drivers": self.cfg["drivers"]}

    def init_diffeqsolve(self):
        grid = self.grid
        self.cfg = get_save_quantities(self.cfg, self.plasma_norm)
        self.time_quantities = {
            "t0": 0.0,
            "t1": grid.tmax,
            "max_steps": grid.max_steps,
        }
        self.diffeqsolve_quants = dict(
            terms=ODETerm(OSHUN1D(self.cfg, grid=grid)),
            solver=Stepper(),
            saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in self.cfg["save"].items()}),
        )

    def __call__(self, trainable_modules: dict, args: dict):
        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=self.grid.max_steps,
            dt0=self.grid.dt,
            y0=self.state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return {"solver result": solver_result}
