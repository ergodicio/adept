"""ADEPTModule entry point for the arbitrary-harmonic VFP-2D solver."""

from __future__ import annotations

from dataclasses import asdict

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import xarray as xr
from diffrax import ODETerm, SaveAt, diffeqsolve

from adept._base_ import ADEPTModule, Stepper
from adept.normalization import UREG, laser_normalization, normalize
from adept.utils import filter_scalars
from adept.vfp1d.fokker_planck import (
    F0Collisions,
    FLMCollisions,
    SelfConsistentBetaConfig,
    get_model,
    get_scheme,
    inverse_bremsstrahlung_resonance_ratio,
)
from adept.vfp1d.grid import Grid as CollisionGrid
from adept.vfp1d.helpers import _initialize_distribution_, calc_logLambda, load_profile_on_grid
from adept.vfp2d.collisions import AnisotropicCollisions, CollisionStep
from adept.vfp2d.distributed import create_spatial_sharding
from adept.vfp2d.grid import Grid
from adept.vfp2d.harmonics import (
    HarmonicLayout,
    HouLiFilter2D,
    TzoufrasVlasov,
    complex_to_real,
    current,
    density,
    nernst_velocity,
    real_to_complex,
    scalar_velocity_moment,
    tensor_velocity_moment,
)
from adept.vfp2d.ohm import KineticOhm2D
from adept.vfp2d.plotting import add_reconnection_diagnostics, reconnection_metrics, save_artifacts
from adept.vfp2d.vector_field import (
    KineticOhmStep,
    Maxwell2D,
    SpectralPoisson2D,
    SplitStepVFP2D,
    VlasovMaxwell,
)


def _profile_1d(profile: dict, axis, norm, reference=None) -> jnp.ndarray:
    basis = profile.get("basis", "uniform")
    baseline = float(profile.get("baseline", profile.get("value", 1.0)))
    if basis == "uniform":
        return baseline * jnp.ones_like(axis)
    if basis in ("sine", "cosine"):
        amplitude = float(profile.get("amplitude", 0.0))
        wavelength = normalize(profile["wavelength"], norm, dim="x")
        trig = jnp.sin if basis == "sine" else jnp.cos
        return baseline * (1.0 + amplitude * trig(2.0 * jnp.pi * axis / wavelength))
    if basis == "tanh":
        center = normalize(profile["center"], norm, dim="x")
        width = normalize(profile["width"], norm, dim="x")
        rise = normalize(profile["rise"], norm, dim="x")
        left, right = center - 0.5 * width, center + 0.5 * width
        envelope = 0.5 * (jnp.tanh((axis - left) / rise) - jnp.tanh((axis - right) / rise))
        if profile.get("bump_or_trough", "bump") == "trough":
            envelope = 1.0 - envelope
        return baseline + float(profile.get("bump_height", 0.0)) * envelope
    if basis == "file":
        loaded = load_profile_on_grid(profile, axis, norm)
        if reference is None:
            raise ValueError("A physical reference quantity is required for file profiles")
        return jnp.asarray((loaded / reference).to("").magnitude)
    raise NotImplementedError(f"Unsupported VFP-2D profile basis: {basis}")


def _profile_2d(profile: dict, grid: Grid, norm, reference=None) -> jnp.ndarray:
    """Build a separable 2D profile while accepting VFP-1D profile syntax."""

    if profile.get("basis") == "gaussian_spots":
        x_center = normalize(profile.get("x_center", 0.0), norm, dim="x")
        x_radius = normalize(profile["x_radius"], norm, dim="x")
        y_radius = normalize(profile.get("y_radius", profile["x_radius"]), norm, dim="x")
        y_centers = profile.get("y_centers", [profile.get("y_center", 0.0)])
        y_centers = jnp.asarray([normalize(center, norm, dim="x") for center in y_centers])
        x_envelope = jnp.exp(-(((grid.x - x_center) / x_radius) ** 2))
        y_envelope = jnp.sum(jnp.exp(-(((grid.y[:, None] - y_centers[None, :]) / y_radius) ** 2)), axis=1)
        return float(profile.get("amplitude", 1.0)) * x_envelope[:, None] * y_envelope[None, :]

    if "x" in profile or "y" in profile:
        px = _profile_1d(profile.get("x", {"basis": "uniform", "baseline": 1.0}), grid.x, norm, reference)
        py = _profile_1d(profile.get("y", {"basis": "uniform", "baseline": 1.0}), grid.y, norm, reference)
        return px[:, None] * py[None, :]
    target_axis = profile.get("axis", "x")
    if target_axis == "y":
        return jnp.broadcast_to(_profile_1d(profile, grid.y, norm, reference)[None, :], (grid.nx, grid.ny))
    return jnp.broadcast_to(_profile_1d(profile, grid.x, norm, reference)[:, None], (grid.nx, grid.ny))


class BaseVFP2D(ADEPTModule):
    """2D3P VFP solver with arbitrary packed complex spherical harmonics."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.plasma_norm = laser_normalization(
            cfg["units"]["laser_wavelength"], cfg["units"]["reference electron temperature"]
        )
        g = cfg["grid"]
        l_max = int(g.get("lmax", g.get("nl", 1)))
        m_max = int(g.get("mmax", l_max))
        if g.get("vmax_is_normalized", False):
            vmax = float(g["vmax"])
        else:
            vmax = float(g.get("vmax", 8.0)) * self.plasma_norm.vth_norm() / np.sqrt(2.0)
        self.grid = Grid(
            xmin=normalize(g["xmin"], self.plasma_norm, dim="x"),
            xmax=normalize(g["xmax"], self.plasma_norm, dim="x"),
            nx=int(g["nx"]),
            ymin=normalize(g["ymin"], self.plasma_norm, dim="x"),
            ymax=normalize(g["ymax"], self.plasma_norm, dim="x"),
            ny=int(g["ny"]),
            vmax=vmax,
            nv=int(g["nv"]),
            dt=normalize(g["dt"], self.plasma_norm, dim="t"),
            l_max=l_max,
            m_max=m_max,
        )
        self.layout = HarmonicLayout(l_max, m_max)
        self.tmin = normalize(g.get("tmin", 0.0), self.plasma_norm, dim="t")
        requested_tmax = normalize(g["tmax"], self.plasma_norm, dim="t")
        self.nt = int(np.ceil((requested_tmax - self.tmin) / self.grid.dt))
        self.tmax = self.tmin + self.nt * self.grid.dt
        self.max_steps = self.nt + 4
        self._density = None
        field_cfg = cfg.get("terms", {}).get("field_solver", {})
        self.field_mode = field_cfg if isinstance(field_cfg, str) else field_cfg.get("mode", "maxwell")
        self._kinetic_ohm = None
        self._maxwell = None
        self.spatial_sharding = create_spatial_sharding(g.get("sharding"), self.grid.nx)

    def write_units(self) -> dict:
        norm = self.plasma_norm
        z = self.cfg["units"]["Z"]
        ne = UREG.Quantity(self.cfg["units"]["reference electron density"]).to("1/cc")
        log_ei, log_ee = calc_logLambda(
            self.cfg, ne, norm.T0.to("eV"), z, self.cfg["units"]["Ion"], force_ee_equal_ei=True
        )
        r_e = 2.8179403205e-13 * UREG.cm
        nuee_coeff = float(
            (4 * jnp.pi * norm.n0 * r_e**2 * UREG.c**4 * log_ee * norm.tau / norm.v0**3).to("").magnitude
        )
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
            (0.093373 * (lam0 / UREG.um) ** 2 / (alpha_pol * (norm.T0 / UREG.keV))).to("").magnitude
        )
        w0_norm = float((2 * np.pi * UREG.c / lam0 * norm.tau).to(""))
        derived = {
            "n0": norm.n0.to("1/cc"),
            "T0": norm.T0.to("eV"),
            "x0": norm.L0.to("nm"),
            "t0": norm.tau.to("fs"),
            "vth_norm": norm.vth_norm(),
            "c_norm": norm.speed_of_light_norm(),
            "logLambda_ei": log_ei,
            "logLambda_ee": log_ee,
            "nuee_coeff": nuee_coeff,
            "logLam_ratio": log_ei / log_ee,
            "vosc2_per_intensity": vosc2_per_intensity,
            "w0_norm": w0_norm,
        }
        self.cfg["units"]["derived"] = derived
        return {key: str(value) for key, value in derived.items()}

    def get_derived_quantities(self):
        values = filter_scalars(asdict(self.grid))
        values.update({"tmin": self.tmin, "tmax": self.tmax, "nt": self.nt, "max_steps": self.max_steps})
        self.cfg["grid"].update(values)

    def get_solver_quantities(self):
        self.cfg["grid"].update(asdict(self.grid))
        self.cfg["grid"].update({"harmonic_pairs": self.layout.pairs})

    def init_state_and_args(self):
        f00 = jnp.zeros((self.grid.nx, self.grid.ny, self.grid.nv))
        n_total = jnp.zeros((self.grid.nx, self.grid.ny))
        found = False
        for name, component in self.cfg["density"].items():
            if not name.startswith("species-"):
                continue
            n_prof = _profile_2d(
                component["n"],
                self.grid,
                self.plasma_norm,
                reference=UREG.Quantity(self.cfg["units"]["reference electron density"]),
            )
            t_prof = _profile_2d(component["T"], self.grid, self.plasma_norm, reference=self.plasma_norm.T0)
            if self.cfg["grid"].get("relativistic", False):
                theta0 = float((self.plasma_norm.T0 / (UREG.m_e * UREG.c**2)).to("").magnitude)
                theta = theta0 * t_prof[..., None]
                gamma = jnp.sqrt(1.0 + self.grid.v**2)
                local_f = jnp.exp(-(gamma[None, None, :] - 1.0) / theta)
                norm = 4.0 * jnp.pi * jnp.sum(local_f * self.grid.v**2, axis=-1) * self.grid.dv
                local_f = n_prof[..., None] * local_f / norm[..., None]
            else:
                local_f, _ = _initialize_distribution_(
                    nv=self.grid.nv,
                    m=float(component.get("m", 2.0)),
                    vth=self.plasma_norm.vth_norm(),
                    vmax=self.grid.vmax,
                    n_prof=n_prof.reshape(-1),
                    T_prof=t_prof.reshape(-1),
                )
                local_f = local_f.reshape((self.grid.nx, self.grid.ny, self.grid.nv))
            f00 = f00 + local_f
            n_total = n_total + n_prof
            found = True
        if not found:
            raise ValueError("VFP-2D density must contain at least one 'species-*' component")

        ne_over_n0 = float(
            (UREG.Quantity(self.cfg["units"]["reference electron density"]) / self.plasma_norm.n0).to("").magnitude
        )
        f00 = f00 * ne_over_n0
        n_total = n_total * ne_over_n0
        flm = (
            jnp.zeros((self.grid.nx, self.grid.ny, self.layout.size, self.grid.nv), dtype=jnp.complex128)
            .at[..., self.layout.index(0, 0), :]
            .set(f00)
        )

        zref = float(self.cfg["units"]["Z"])
        ion_charge = n_total if self.cfg["density"].get("quasineutrality", True) else jnp.mean(n_total)
        charge_density = ion_charge - density(flm, self.layout, self.grid.v, self.grid.dv)
        e = SpectralPoisson2D(self.grid.kx, self.grid.ky)(charge_density)
        # Diffrax currently warns that complex state support is experimental.
        # Keep its PyTree purely real while retaining complex arithmetic inside
        # the harmonic operator.
        self.state = {"flm": complex_to_real(flm), "e": e, "b": jnp.zeros_like(e)}
        self._density = n_total
        self.args = {"Z": jnp.ones_like(n_total), "ni": n_total / zref}
        drivers = self.cfg.get("drivers", {})
        maxwellian = drivers.get("maxwellian_heating", {})
        if "D0" in maxwellian:
            profile = _profile_2d(
                maxwellian.get("profile", {"basis": "uniform", "baseline": 1.0}),
                self.grid,
                self.plasma_norm,
            )
            self.args["D0_heating"] = float(maxwellian["D0"]) * profile

        ib = drivers.get("ib", {})
        intensity = float(ib.get("intensity_1e15_Wcm2", 0.0))
        if intensity > 0.0:
            profile = _profile_2d(
                ib.get("profile", {"basis": "uniform", "baseline": 1.0}),
                self.grid,
                self.plasma_norm,
            )
            self.args["ib_vosc2"] = self.cfg["units"]["derived"]["vosc2_per_intensity"] * intensity * profile
            derived = self.cfg["units"]["derived"]
            self.args["ib_Z2ni_w0"] = inverse_bremsstrahlung_resonance_ratio(
                self.args["Z"],
                self.args["ni"],
                derived["nuee_coeff"],
                derived["logLam_ratio"],
                derived["w0_norm"],
            )
            for source_key, arg_key in (
                ("switch_on", "ib_t_on"),
                ("switch_off", "ib_t_off"),
                ("switch_width", "ib_switch_width"),
            ):
                if source_key in ib:
                    self.args[arg_key] = normalize(ib[source_key], self.plasma_norm, dim="t")

        field_cfg = self.cfg.get("terms", {}).get("field_solver", {})
        if isinstance(field_cfg, dict) and self.field_mode == "kinetic-ohm":
            hidden = field_cfg.get("hidden_density_gradient", {})
            if hidden.get("active", False):
                profile = _profile_2d(
                    hidden.get("profile", {"basis": "uniform", "baseline": 1.0}),
                    self.grid,
                    self.plasma_norm,
                )
                scale_length = normalize(hidden["scale_length"], self.plasma_norm, dim="x")
                reference_density = float(
                    (UREG.Quantity(self.cfg["units"]["reference electron density"]) / self.plasma_norm.n0)
                    .to("")
                    .magnitude
                )
                self.args["hidden_dndz"] = reference_density * profile / scale_length
                if "switch_off" in hidden:
                    self.args["hidden_gradient_t_off"] = normalize(hidden["switch_off"], self.plasma_norm, dim="t")
                if "switch_width" in hidden:
                    self.args["hidden_gradient_switch_width"] = normalize(
                        hidden["switch_width"], self.plasma_norm, dim="t"
                    )

    def _collision_step(self) -> CollisionStep | None:
        fp = self.cfg.get("terms", {}).get("fokker_planck", {})
        if not fp.get("active", True):
            return None
        if self.cfg["grid"].get("relativistic", False):
            raise NotImplementedError(
                "The Tzoufras linearized collision operator is non-relativistic; "
                "set grid.relativistic=false when Fokker-Planck collisions are active."
            )
        collision_grid = CollisionGrid(
            xmin=0.0,
            xmax=1.0,
            nx=self.grid.nx * self.grid.ny,
            tmin=0.0,
            tmax=self.grid.dt,
            dt=self.grid.dt,
            nv=self.grid.nv,
            vmax=self.grid.vmax,
            nl=self.layout.l_max,
        )
        f00_cfg = fp.get("f00", {})
        model = get_model(f00_cfg.get("model", "CoulombianKernel"), collision_grid.v, collision_grid.dv)
        scheme = get_scheme(f00_cfg.get("scheme", "central"), collision_grid.dv)
        sc = fp.get("self_consistent_beta", {})
        isotropic = F0Collisions(
            nuee_coeff=self.cfg["units"]["derived"]["nuee_coeff"],
            grid=collision_grid,
            model=model,
            scheme=scheme,
            sc_beta=SelfConsistentBetaConfig(
                max_steps=sc.get("max_steps", 3) if sc.get("enabled", False) else 0,
                rtol=sc.get("rtol", 1e-8),
                atol=sc.get("atol", 1e-12),
            ),
        )
        flm_operator = FLMCollisions(
            Z=float(self.cfg["units"]["Z"]),
            nuee_coeff=self.cfg["units"]["derived"]["nuee_coeff"],
            grid=collision_grid,
            logLam_ratio=self.cfg["units"]["derived"]["logLam_ratio"],
            full_aniso_ee=fp.get("flm", {}).get("ee", True),
        )
        return CollisionStep(
            self.layout,
            isotropic,
            AnisotropicCollisions(flm_operator, self.layout),
            mesh=None if self.spatial_sharding is None else self.spatial_sharding.mesh,
        )

    def init_diffeqsolve(self):
        if self.spatial_sharding is not None:
            self.state = jtu.tree_map(self.spatial_sharding.put, self.state)
            self.args = jtu.tree_map(self.spatial_sharding.put, self.args)
        relativistic = bool(self.cfg["grid"].get("relativistic", False))
        streaming_speed = self.grid.v / jnp.sqrt(1.0 + self.grid.v**2) if relativistic else self.grid.v
        partitioned_dx = self.grid.dx if self.spatial_sharding is not None else None
        partitioned_dy = self.grid.dy if self.spatial_sharding is not None else None
        vlasov = TzoufrasVlasov(
            self.layout,
            self.grid.v,
            self.grid.dv,
            self.grid.kx,
            self.grid.ky,
            streaming_speed=streaming_speed,
            dx=partitioned_dx,
            dy=partitioned_dy,
            mesh=None if self.spatial_sharding is None else self.spatial_sharding.mesh,
        )
        maxwell = Maxwell2D(
            self.grid.kx,
            self.grid.ky,
            c=self.plasma_norm.speed_of_light_norm(),
            dx=partitioned_dx,
            dy=partitioned_dy,
            mesh=None if self.spatial_sharding is None else self.spatial_sharding.mesh,
        )
        self._maxwell = maxwell
        collisions = self._collision_step()
        if self.field_mode == "maxwell":
            rhs = VlasovMaxwell(
                vlasov,
                maxwell,
                self.layout,
                self.grid.v,
                self.grid.dv,
                real_storage=True,
                streaming_speed=streaming_speed,
            )
            step = SplitStepVFP2D(rhs, self.grid.dt, collisions=collisions)
        elif self.field_mode == "kinetic-ohm":
            zref = float(self.cfg["units"]["Z"])
            resistivity_coefficient = (
                0.5 * zref * self.cfg["units"]["derived"]["nuee_coeff"] * self.cfg["units"]["derived"]["logLam_ratio"]
            )
            self._kinetic_ohm = KineticOhm2D(
                self.layout,
                self.grid.v,
                self.grid.dv,
                self.grid.kx,
                self.grid.ky,
                resistivity_coefficient=resistivity_coefficient,
                dx=partitioned_dx,
                dy=partitioned_dy,
                mesh=None if self.spatial_sharding is None else self.spatial_sharding.mesh,
            )
            filter_cfg = self.cfg.get("terms", {}).get("hou_li_filter", {})
            spatial_filter = None
            if filter_cfg.get("is_on", False):
                dimensions = set(filter_cfg.get("dimensions", ["x", "y"]))
                if not dimensions or not dimensions <= {"x", "y"}:
                    raise ValueError("VFP-2D Hou-Li filtering dimensions must be a nonempty subset of [x, y]")
                if self.spatial_sharding is not None and "x" in dimensions:
                    raise ValueError("x-sharded VFP-2D must omit x from spectral Hou-Li filter dimensions")
                spatial_filter = HouLiFilter2D(
                    self.grid.nx,
                    self.grid.ny,
                    alpha=float(filter_cfg.get("alpha", 36.0)),
                    order=int(filter_cfg.get("order", 36)),
                    dimensions=tuple(sorted(dimensions)),
                    mesh=None if self.spatial_sharding is None else self.spatial_sharding.mesh,
                )
            step = KineticOhmStep(
                vlasov,
                maxwell,
                self._kinetic_ohm,
                self.layout,
                self.grid.v,
                self.grid.dv,
                self.grid.dt,
                collisions=collisions,
                real_storage=True,
                enforce_f00_positivity=(
                    self.cfg.get("terms", {}).get("fokker_planck", {}).get("f00", {}).get("positivity", "none")
                    == "conservative"
                ),
                spatial_filter=spatial_filter,
            )
            initial_flm = real_to_complex(self.state["flm"])
            initial_current = maxwell.c2 * maxwell.curl(self.state["b"])
            initial_hidden_dndz = KineticOhmStep._hidden_dndz(self.tmin, self.args, self.state["b"][..., 0])
            initial_e, _terms = self._kinetic_ohm(
                initial_flm,
                self.state["b"],
                plasma_current=initial_current,
                hidden_dndz=initial_hidden_dndz,
            )
            self.state = {**self.state, "e": initial_e}
        else:
            raise ValueError(
                f"Unsupported VFP-2D field solver mode {self.field_mode!r}; expected 'maxwell' or 'kinetic-ohm'"
            )
        save_cfg = self.cfg.get("save", {}).get("t", {})
        save_tmin = normalize(save_cfg.get("tmin", self.tmin), self.plasma_norm, dim="t")
        save_tmax = normalize(save_cfg.get("tmax", self.tmax), self.plasma_norm, dim="t")
        save_nt = int(save_cfg.get("nt", min(self.nt + 1, 101)))
        self.save_times = jnp.linspace(save_tmin, save_tmax, save_nt)
        self.time_quantities = {"t0": self.tmin, "t1": self.tmax, "max_steps": self.max_steps}
        if self.spatial_sharding is not None:

            def save_fn(_t, state, _args):
                return jtu.tree_map(self.spatial_sharding.replicate, state)

            saveat = SaveAt(ts=self.save_times, fn=save_fn)
        else:
            saveat = SaveAt(ts=self.save_times)
        self.diffeqsolve_quants = {
            "terms": ODETerm(step),
            "solver": Stepper(),
            "saveat": saveat,
        }

    def __call__(self, trainable_modules: dict | None, args: dict | None):
        def solve():
            return diffeqsolve(
                terms=self.diffeqsolve_quants["terms"],
                solver=self.diffeqsolve_quants["solver"],
                t0=self.tmin,
                t1=self.tmax,
                dt0=self.grid.dt,
                max_steps=self.max_steps,
                y0=self.state,
                args=self.args if args is None else args,
                saveat=self.diffeqsolve_quants["saveat"],
            )

        if self.spatial_sharding is not None:
            with self.spatial_sharding.mesh:
                result = solve()
        else:
            result = solve()
        return {"solver result": result}

    def post_process(self, run_output: dict, td: str) -> dict:
        result = run_output["solver result"]
        flm_jax = real_to_complex(result.ys["flm"])
        flm = np.asarray(flm_jax)
        ne = density(flm_jax, self.layout, self.grid.v, self.grid.dv)
        plasma_current = current(flm_jax, self.layout, self.grid.v, self.grid.dv)
        mean_v2 = scalar_velocity_moment(flm_jax, self.layout, self.grid.v, self.grid.dv, power=2)
        temperature_normalized = (2.0 / 3.0) * mean_v2 / self.plasma_norm.vth_norm() ** 2
        pressure_anisotropy = tensor_velocity_moment(flm_jax, self.layout, self.grid.v, self.grid.dv, power=0)
        v_nernst = nernst_velocity(flm_jax, self.layout, self.grid.v, self.grid.dv, plasma_current=plasma_current)
        coords = {
            "t": np.asarray(result.ts),
            "x": np.asarray(self.grid.x),
            "y": np.asarray(self.grid.y),
            "harmonic": np.arange(self.layout.size),
            "v": np.asarray(self.grid.v),
            "component": ["x", "y", "z"],
            "ell": ("harmonic", self.layout.ell),
            "m": ("harmonic", self.layout.m),
        }
        data_vars = {
            "flm_real": (("t", "x", "y", "harmonic", "v"), flm.real),
            "flm_imag": (("t", "x", "y", "harmonic", "v"), flm.imag),
            "e": (("t", "x", "y", "component"), np.asarray(result.ys["e"])),
            "b": (("t", "x", "y", "component"), np.asarray(result.ys["b"])),
            "ne": (("t", "x", "y"), np.asarray(ne)),
            "temperature": (("t", "x", "y"), np.asarray(temperature_normalized)),
            "current": (("t", "x", "y", "component"), np.asarray(plasma_current)),
            "v_nernst": (("t", "x", "y", "component"), np.asarray(v_nernst)),
            "pressure_anisotropy": (
                ("t", "x", "y", "component", "component_2"),
                np.asarray(pressure_anisotropy),
            ),
        }
        if self._kinetic_ohm is not None and self._maxwell is not None:
            ohm_history = {key: [] for key in ("resistive", "hall", "nernst", "scalar_pressure", "tensor_pressure")}
            for index, time in enumerate(np.asarray(result.ts)):
                target_current = self._maxwell.c2 * self._maxwell.curl(result.ys["b"][index])
                hidden_dndz = KineticOhmStep._hidden_dndz(float(time), self.args, result.ys["b"][index, ..., 0])
                _electric, terms = self._kinetic_ohm(
                    flm_jax[index],
                    result.ys["b"][index],
                    plasma_current=target_current,
                    hidden_dndz=hidden_dndz,
                )
                for key, value in terms.items():
                    ohm_history[key].append(value)
            for key, values in ohm_history.items():
                data_vars[f"ohm_{key}"] = (
                    ("t", "x", "y", "component"),
                    np.asarray(jnp.stack(values)),
                )

        ds = xr.Dataset(
            data_vars,
            coords={**coords, "component_2": ["x", "y", "z"]},
            attrs={
                "solver": "vfp-2d",
                "harmonic_convention": "Tzoufras JCP 230 (2011)",
                "length_unit_um": float(self.plasma_norm.L0.to("um").magnitude),
                "time_unit_ps": float(self.plasma_norm.tau.to("ps").magnitude),
            },
        )
        ds = add_reconnection_diagnostics(ds)
        if td:
            n_panels = int(self.cfg.get("output", {}).get("n_panels", 9))
            save_artifacts(ds, td, n_panels=n_panels)
        return {"vfp2d": ds, "metrics": reconnection_metrics(ds)}
