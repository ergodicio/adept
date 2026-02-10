"""HermiteSRS1D module with real-unit normalization and laser Ey driver support."""

import math
import os

import numpy as np
import pint
from jax import numpy as jnp

from adept._spectrax1d.base_module import BaseSpectrax1D


class HermiteSRS1D(BaseSpectrax1D):
    """Spectrax-1D module with real-unit normalization and SRS laser drivers."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.ureg = pint.UnitRegistry()

    def _require_quantity(self, value, field_name: str):
        if isinstance(value, str):
            return self.ureg.Quantity(value)
        if isinstance(value, pint.Quantity):
            return value
        raise ValueError(f"{field_name} must be a unit string (e.g., '1 ps', '10 micron'); got {value!r}")

    def write_units(self) -> dict:
        _Q = self.ureg.Quantity

        n0 = self._require_quantity(self.cfg["units"]["normalizing_density"], "units.normalizing_density").to("1/cc")
        Te = self._require_quantity(self.cfg["units"]["normalizing_temperature"], "units.normalizing_temperature").to(
            "eV"
        )
        Ti = self._require_quantity(self.cfg["units"]["ion_temperature"], "units.ion_temperature").to("eV")

        wp0 = np.sqrt(n0 * self.ureg.e**2.0 / (self.ureg.m_e * self.ureg.epsilon_0)).to("rad/s")
        tp0 = (1 / wp0).to("s")
        x0 = (self.ureg.c / wp0).to("m")

        mi_me = float(self.cfg["physics"]["mi_me"])
        vth_e = np.sqrt(2.0 * Te / self.ureg.m_e).to("m/s")
        vth_i = np.sqrt(2.0 * Ti / (self.ureg.m_e * mi_me)).to("m/s")

        alpha_e = (vth_e / self.ureg.c).to("dimensionless")
        alpha_i = (vth_i / self.ureg.c).to("dimensionless")

        lambda_D = (vth_e / wp0).to("m")

        all_quantities = {
            "wp0": wp0,
            "tp0": tp0,
            "n0": n0,
            "Te": Te,
            "Ti": Ti,
            "x0": x0,
            "skin_depth": x0,
            "lambda_D": lambda_D,
            "alpha_e": alpha_e,
            "alpha_i": alpha_i,
        }

        self.cfg["units"]["derived"] = all_quantities
        return all_quantities

    def _convert_driver_envelopes(self, pulse: dict, tp0, x0) -> None:
        for key in ["t_center", "t_width", "t_rise"]:
            if key in pulse and not isinstance(pulse[key], (int, float)):
                tval = self._require_quantity(pulse[key], f"drivers.*.{key}")
                pulse[key] = (tval / tp0).to("").magnitude

        for key in ["x_center", "x_width", "x_rise"]:
            if key in pulse and not isinstance(pulse[key], (int, float)):
                xval = self._require_quantity(pulse[key], f"drivers.*.{key}")
                pulse[key] = (xval / x0).to("").magnitude

    def _get_boundary_config(self, Lx_real):
        if not isinstance(Lx_real, pint.Quantity):
            Lx_real = Lx_real * self.ureg.meter
        bcfg = self.cfg.get("boundaries", None)
        if not isinstance(bcfg, dict):
            return None

        side = bcfg.get("side", "left")
        if side not in {"left", "right"}:
            raise ValueError("boundaries.side must be 'left' or 'right'")

        vac = bcfg.get("vacuum_length", None)
        if vac is None:
            vac = 0.1 * Lx_real
        else:
            vac = self._require_quantity(vac, "boundaries.vacuum_length").to("m")

        absorb = bcfg.get("absorb_length", None)
        if absorb is None:
            absorb = 0.1 * Lx_real
        else:
            absorb = self._require_quantity(absorb, "boundaries.absorb_length").to("m")

        if (vac + absorb) >= Lx_real:
            raise ValueError("boundaries.vacuum_length + boundaries.absorb_length must be < Lx")

        return {
            "side": side,
            "vacuum_length": vac,
            "absorb_length": absorb,
        }

    def _get_sponge_config(self, Lx_real):
        bcfg = self.cfg.get("boundaries", None)
        if not isinstance(bcfg, dict):
            return None

        sponge = bcfg.get("sponge", {})
        if sponge is False:
            return None
        if isinstance(sponge, dict):
            enabled = sponge.get("enabled", True)
            if not enabled:
                return None
        else:
            sponge = {}

        fields = sponge.get("sigma_max_fields", "0.2/ps")
        plasma = sponge.get("sigma_max_plasma", "0.1/ps")
        sigma_fields = self._require_quantity(fields, "boundaries.sponge.sigma_max_fields").to("1/s")
        sigma_plasma = self._require_quantity(plasma, "boundaries.sponge.sigma_max_plasma").to("1/s")

        return {
            "sigma_max_fields": sigma_fields,
            "sigma_max_plasma": sigma_plasma,
        }

    def _build_density_profile(self, x_real: jnp.ndarray, Lx_real, side: str, vacuum_len, absorb_len):
        if side == "left":
            x0 = vacuum_len
            x1 = vacuum_len + absorb_len
            s = (x_real - x0) / absorb_len
            ramp = jnp.where(
                x_real < x0,
                0.0,
                jnp.where(x_real > x1, 1.0, 0.5 * (1.0 - jnp.cos(jnp.pi * s))),
            )
        else:
            x0 = Lx_real - vacuum_len - absorb_len
            x1 = Lx_real - vacuum_len
            s = (x_real - x0) / absorb_len
            ramp = jnp.where(
                x_real < x0,
                1.0,
                jnp.where(x_real > x1, 0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * s))),
            )
        return ramp

    def _build_sponge_profile(self, x_real: jnp.ndarray, Lx_real, side: str, vacuum_len, absorb_len):
        if side == "left":
            x0 = vacuum_len
            x1 = vacuum_len + absorb_len
            s = (x_real - x0) / absorb_len
            ramp = jnp.where(
                x_real < x0,
                0.0,
                jnp.where(x_real > x1, 1.0, jnp.sin(0.5 * jnp.pi * s) ** 2),
            )
        else:
            x0 = Lx_real - vacuum_len - absorb_len
            x1 = Lx_real - vacuum_len
            s = (x_real - x0) / absorb_len
            ramp = jnp.where(
                x_real < x0,
                1.0,
                jnp.where(x_real > x1, 0.0, jnp.sin(0.5 * jnp.pi * (1.0 - s)) ** 2),
            )
        return ramp

    def _get_plasma_bounds_norm(self):
        if "boundaries" not in self.cfg:
            return None
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]
        Lx_norm = float(self.cfg["physics"]["Lx"])
        Lx_real = Lx_norm * x0

        boundary_cfg = self._get_boundary_config(Lx_real)
        if boundary_cfg is None:
            return None

        vac_norm = (boundary_cfg["vacuum_length"] / x0).to("").magnitude
        absorb_norm = (boundary_cfg["absorb_length"] / x0).to("").magnitude
        side = boundary_cfg["side"]

        if side == "left":
            xmin = vac_norm + absorb_norm
            xmax = Lx_norm
        else:
            xmin = 0.0
            xmax = Lx_norm - vac_norm - absorb_norm

        if xmax <= xmin:
            return None
        return xmin, xmax

    def _get_vacuum_bounds_norm(self):
        if "boundaries" not in self.cfg:
            return None
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]
        Lx_norm = float(self.cfg["physics"]["Lx"])
        Lx_real = Lx_norm * x0

        boundary_cfg = self._get_boundary_config(Lx_real)
        if boundary_cfg is None:
            return None

        vac_norm = (boundary_cfg["vacuum_length"] / x0).to("").magnitude
        side = boundary_cfg["side"]

        if side == "left":
            xmin = 0.0
            xmax = vac_norm
        else:
            xmin = Lx_norm - vac_norm
            xmax = Lx_norm

        if xmax <= xmin:
            return None
        return xmin, xmax

    def _plot_fields_spacetime(self, fields_xr, td: str) -> None:
        super()._plot_fields_spacetime(fields_xr, td)

        bounds = self._get_plasma_bounds_norm()
        if bounds is None:
            return

        xmin, xmax = bounds
        fields_plasma = fields_xr.sel(x=slice(xmin, xmax))
        if fields_plasma.coords["x"].size < 2:
            return

        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "fields", "plasma")
        os.makedirs(plots_dir, exist_ok=True)

        for field_name, field_data in fields_plasma.items():
            field_data.plot()
            plt.title(f"{field_name} Spacetime (Plasma Region)")
            plt.savefig(os.path.join(plots_dir, f"spacetime-{field_name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

        vac_bounds = self._get_vacuum_bounds_norm()
        if vac_bounds is None:
            return

        xmin, xmax = vac_bounds
        fields_vacuum = fields_xr.sel(x=slice(xmin, xmax))
        if fields_vacuum.coords["x"].size < 2:
            return

        plots_dir = os.path.join(td, "plots", "fields", "vacuum")
        os.makedirs(plots_dir, exist_ok=True)

        for field_name, field_data in fields_vacuum.items():
            field_data.plot()
            plt.title(f"{field_name} Spacetime (Vacuum Region)")
            plt.savefig(os.path.join(plots_dir, f"spacetime-{field_name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

    def _plot_fields_lineouts(self, fields_xr, td: str, n_slices: int = 6) -> None:
        super()._plot_fields_lineouts(fields_xr, td, n_slices=n_slices)

        bounds = self._get_plasma_bounds_norm()
        if bounds is None:
            return

        xmin, xmax = bounds
        fields_plasma = fields_xr.sel(x=slice(xmin, xmax))
        if fields_plasma.coords["x"].size < 2:
            return

        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "fields", "lineouts-plasma")
        os.makedirs(plots_dir, exist_ok=True)

        for field_name, field_data in fields_plasma.items():
            nt = field_data.coords["t"].size
            t_skip = max(1, nt // n_slices)
            tslice = slice(0, None, t_skip)

            field_data[tslice].T.plot(col="t", col_wrap=3)
            plt.savefig(os.path.join(plots_dir, f"{field_name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

    def _plot_moments_spacetime(self, moments_xr, td: str) -> None:
        super()._plot_moments_spacetime(moments_xr, td)

        bounds = self._get_plasma_bounds_norm()
        if bounds is None:
            return

        xmin, xmax = bounds
        moments_plasma = moments_xr.sel(x=slice(xmin, xmax))
        if moments_plasma.coords["x"].size < 2:
            return

        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "moments", "plasma")
        os.makedirs(plots_dir, exist_ok=True)

        for name, data in moments_plasma.items():
            if "x" not in data.dims:
                continue
            data.plot()
            plt.title(f"{name} Spacetime (Plasma Region)")
            plt.savefig(os.path.join(plots_dir, f"spacetime-{name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

        vac_bounds = self._get_vacuum_bounds_norm()
        if vac_bounds is None:
            return

        xmin, xmax = vac_bounds
        moments_vacuum = moments_xr.sel(x=slice(xmin, xmax))
        if moments_vacuum.coords["x"].size < 2:
            return

        plots_dir = os.path.join(td, "plots", "moments", "vacuum")
        os.makedirs(plots_dir, exist_ok=True)

        for name, data in moments_vacuum.items():
            if "x" not in data.dims:
                continue
            data.plot()
            plt.title(f"{name} Spacetime (Vacuum Region)")
            plt.savefig(os.path.join(plots_dir, f"spacetime-{name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

    def _plot_moments_lineouts(self, moments_xr, td: str, n_slices: int = 6) -> None:
        super()._plot_moments_lineouts(moments_xr, td, n_slices=n_slices)

        bounds = self._get_plasma_bounds_norm()
        if bounds is None:
            return

        xmin, xmax = bounds
        moments_plasma = moments_xr.sel(x=slice(xmin, xmax))
        if moments_plasma.coords["x"].size < 2:
            return

        import matplotlib.pyplot as plt

        plots_dir = os.path.join(td, "plots", "moments", "lineouts-plasma")
        os.makedirs(plots_dir, exist_ok=True)

        for name, data in moments_plasma.items():
            if "x" not in data.dims:
                continue
            nt = data.coords["t"].size
            t_skip = max(1, nt // n_slices)
            tslice = slice(0, None, t_skip)

            data[tslice].T.plot(col="t", col_wrap=3)
            plt.savefig(os.path.join(plots_dir, f"{name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

        vac_bounds = self._get_vacuum_bounds_norm()
        if vac_bounds is None:
            return

        xmin, xmax = vac_bounds
        moments_vacuum = moments_xr.sel(x=slice(xmin, xmax))
        if moments_vacuum.coords["x"].size < 2:
            return

        plots_dir = os.path.join(td, "plots", "moments", "lineouts-vacuum")
        os.makedirs(plots_dir, exist_ok=True)

        for name, data in moments_vacuum.items():
            if "x" not in data.dims:
                continue
            nt = data.coords["t"].size
            t_skip = max(1, nt // n_slices)
            tslice = slice(0, None, t_skip)

            data[tslice].T.plot(col="t", col_wrap=3)
            plt.savefig(os.path.join(plots_dir, f"{name}.png"), bbox_inches="tight", dpi=150)
            plt.close()

    def _convert_ey_laser_pulse(self, pulse: dict, x0, wp0) -> None:
        if "dw0" in pulse:
            raise ValueError("drivers.ey: detuning (dw0) is disabled when wavelength drives w0")
        if "k0" in pulse or "w0" in pulse:
            raise ValueError("drivers.ey: k0/w0 must not be provided when using intensity+wavelength inputs")

        if "intensity" not in pulse or "wavelength" not in pulse:
            raise ValueError("drivers.ey: intensity and wavelength are required for Ey laser drivers")

        intensity = self._require_quantity(pulse["intensity"], "drivers.ey.intensity").to("W/m^2")
        wavelength = self._require_quantity(pulse["wavelength"], "drivers.ey.wavelength").to("m")

        e = self.ureg.e
        m_e = self.ureg.m_e
        eps0 = self.ureg.epsilon_0
        c = self.ureg.c

        a0 = (e * wavelength / (m_e * math.pi)) * (intensity / (2 * eps0 * c**5)) ** 0.5
        k0 = (2 * math.pi / wavelength).to("1/m")
        w0 = (2 * math.pi * c / wavelength).to("1/s")

        pulse["a0"] = float(a0.to("").magnitude)
        pulse["k0"] = float((k0 * x0).to("").magnitude)
        pulse["w0"] = float((w0 / wp0).to("").magnitude)

        pulse.pop("intensity", None)
        pulse.pop("wavelength", None)

    def _apply_real_units(self) -> None:
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]
        tp0 = derived["tp0"]
        wp0 = derived["wp0"]

        physics = self.cfg["physics"]
        for key in ["Lx", "Ly", "Lz"]:
            if key in physics:
                val = self._require_quantity(physics[key], f"physics.{key}")
                physics[key] = (val / x0).to("").magnitude

        grid = self.cfg["grid"]
        if "tmax" in grid:
            tmax = self._require_quantity(grid["tmax"], "grid.tmax")
            grid["tmax"] = (tmax / tp0).to("").magnitude

        if "save" in self.cfg:
            for save_key, save_cfg in self.cfg["save"].items():
                if isinstance(save_cfg, dict) and isinstance(save_cfg.get("t"), dict):
                    t_cfg = save_cfg["t"]
                    for t_key in ["tmin", "tmax"]:
                        if t_key in t_cfg:
                            tval = self._require_quantity(t_cfg[t_key], f"save.{save_key}.t.{t_key}")
                            t_cfg[t_key] = (tval / tp0).to("").magnitude
                if isinstance(save_cfg, dict) and isinstance(save_cfg.get("x"), dict):
                    x_cfg = save_cfg["x"]
                    for x_key in ["xmin", "xmax"]:
                        if x_key in x_cfg:
                            xval = self._require_quantity(x_cfg[x_key], f"save.{save_key}.x.{x_key}")
                            x_cfg[x_key] = (xval / x0).to("").magnitude
                if isinstance(save_cfg, dict) and isinstance(save_cfg.get("kx"), dict):
                    kx_cfg = save_cfg["kx"]
                    for kx_key in ["kxmin", "kxmax"]:
                        if kx_key in kx_cfg:
                            kval = self._require_quantity(kx_cfg[kx_key], f"save.{save_key}.kx.{kx_key}")
                            kx_cfg[kx_key] = (kval * x0).to("").magnitude

        drivers = self.cfg.get("drivers", {})
        Lx_real = physics["Lx"] * x0
        boundary_cfg = self._get_boundary_config(Lx_real) if "boundaries" in self.cfg else None
        for driver_key, driver_group in drivers.items():
            if not isinstance(driver_group, dict):
                continue
            for _, pulse in driver_group.items():
                if not isinstance(pulse, dict):
                    continue
                if driver_key == "ey" and boundary_cfg is not None:
                    if not any(k in pulse for k in ["x_center", "x_width", "x_rise"]):
                        vac = boundary_cfg["vacuum_length"]
                        side = boundary_cfg["side"]
                        if side == "left":
                            pulse["x_center"] = 0.5 * vac
                        else:
                            pulse["x_center"] = Lx_real - 0.5 * vac
                        pulse["x_width"] = vac
                        pulse["x_rise"] = 0.1 * vac

                self._convert_driver_envelopes(pulse, tp0, x0)

                if driver_key == "ey":
                    if "intensity" in pulse or "wavelength" in pulse:
                        self._convert_ey_laser_pulse(pulse, x0, wp0)
                        continue
                    if pulse:
                        raise ValueError("drivers.ey: intensity and wavelength are required for Ey laser drivers")

                for k_key in ["k0"]:
                    if k_key in pulse and not isinstance(pulse[k_key], (int, float)):
                        kval = self._require_quantity(pulse[k_key], f"drivers.{driver_key}.{k_key}")
                        pulse[k_key] = (kval * x0).to("").magnitude

                for w_key in ["w0", "dw0"]:
                    if w_key in pulse and not isinstance(pulse[w_key], (int, float)):
                        wval = self._require_quantity(pulse[w_key], f"drivers.{driver_key}.{w_key}")
                        pulse[w_key] = (wval / wp0).to("").magnitude

    def get_derived_quantities(self) -> None:
        self._apply_real_units()

        derived = self.cfg["units"]["derived"]
        alpha_e = float(derived["alpha_e"].to("").magnitude)
        alpha_i = float(derived["alpha_i"].to("").magnitude)
        Te = derived["Te"]
        Ti = derived["Ti"]

        physics = self.cfg["physics"]
        physics["dn1"] = 0.0
        physics["alpha_e"] = [alpha_e, alpha_e, alpha_e]
        physics["alpha_s"] = [alpha_e, alpha_e, alpha_e, alpha_i, alpha_i, alpha_i]
        physics["Ti_Te"] = float((Ti / Te).to("").magnitude)

        grid = self.cfg["grid"]
        x0 = derived["x0"]
        lambda_D = derived["lambda_D"]
        Lx_real = physics["Lx"] * x0
        dx_target = 0.95 * lambda_D
        Nx = int(np.ceil((Lx_real / dx_target).to("").magnitude))
        if Nx < 3:
            Nx = 3
        if Nx % 2 == 0:
            Nx += 1
        grid["Nx"] = Nx
        grid["Ny"] = 1
        grid["Nz"] = 1
        grid["dx"] = float(physics["Lx"] / Nx)

        n0_e = 1.0
        if "n0_s" in physics:
            n0_s = physics["n0_s"]
            if isinstance(n0_s, (list, tuple)) and len(n0_s) >= 2:
                n0_e = float(n0_s[0])
        n0_e = float(physics.get("n0_e", n0_e))
        if n0_e <= 0.0:
            raise ValueError("n0_e must be > 0 to compute dt from omega_pe")

        n_min = n0_e
        Lx_real_m = (physics["Lx"] * x0).to("m").magnitude
        boundary_cfg = self._get_boundary_config(Lx_real_m) if "boundaries" in self.cfg else None
        if boundary_cfg is not None:
            vac = boundary_cfg["vacuum_length"].to("m").magnitude
            absorb = boundary_cfg["absorb_length"].to("m").magnitude
            x_real = np.linspace(0.0, Lx_real_m, Nx, endpoint=False)
            ramp = np.asarray(
                self._build_density_profile(
                    jnp.asarray(x_real),
                    Lx_real_m,
                    boundary_cfg["side"],
                    vac,
                    absorb,
                )
            )
            if boundary_cfg["side"] == "left":
                plasma_mask = x_real >= (vac + absorb)
            else:
                plasma_mask = x_real <= (Lx_real_m - vac - absorb)

            if np.any(plasma_mask):
                n_min = float(np.min(n0_e * ramp[plasma_mask]))

        if n_min <= 0.0:
            raise ValueError("Computed minimum plasma density must be > 0 for dt calculation")

        dt_plasma = 0.1 / np.sqrt(n_min)
        cfl = float(grid.get("cfl", 0.5))
        dt_light = cfl * grid["dx"] / np.pi
        dt = min(dt_plasma, dt_light)
        grid["dt"] = dt
        tmax = grid["tmax"]
        nt = int(tmax / dt) + 1
        grid["nt"] = nt
        grid["tmax"] = dt * nt

        if nt > 1e6:
            grid["max_steps"] = int(1e6)
            print(r"Only running $10^6$ steps")
        else:
            grid["max_steps"] = nt + 4

        self.cfg["grid"] = grid

    def get_solver_quantities(self) -> None:
        super().get_solver_quantities()

        physics = self.cfg["physics"]
        grid = self.cfg["grid"]
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]
        Nx = int(grid["Nx"])

        n0_e = 1.0
        n0_i = 1.0
        if "n0_s" in physics:
            n0_s = physics["n0_s"]
            if isinstance(n0_s, (list, tuple)) and len(n0_s) >= 2:
                n0_e = float(n0_s[0])
                n0_i = float(n0_s[1])
        n0_e = float(physics.get("n0_e", n0_e))
        n0_i = float(physics.get("n0_i", n0_i))

        input_parameters = self.cfg["spectrax_input"]
        alpha_e = float(input_parameters["alpha_s"][0])
        alpha_i = float(input_parameters["alpha_s"][3])

        Ck_0_electrons = input_parameters["Ck_0_electrons"]
        Ck_0_ions = input_parameters["Ck_0_ions"]

        Lx_real = (float(physics["Lx"]) * x0).to("m").magnitude
        boundary_cfg = self._get_boundary_config(Lx_real * self.ureg.meter) if "boundaries" in self.cfg else None
        sponge_cfg = self._get_sponge_config(Lx_real * self.ureg.meter)
        if sponge_cfg is not None:
            tp0 = derived["tp0"]
            sigma_fields = (sponge_cfg["sigma_max_fields"] * tp0).to("").magnitude
            sigma_plasma = (sponge_cfg["sigma_max_plasma"] * tp0).to("").magnitude
            x_real = jnp.linspace(0.0, Lx_real, Nx, endpoint=False)
            if boundary_cfg is None:
                ramp = jnp.zeros_like(x_real)
            else:
                ramp = self._build_sponge_profile(
                    x_real,
                    Lx_real,
                    boundary_cfg["side"],
                    boundary_cfg["vacuum_length"].to("m").magnitude,
                    boundary_cfg["absorb_length"].to("m").magnitude,
                )
            grid["sponge_fields"] = jnp.asarray(ramp * sigma_fields)
            grid["sponge_plasma"] = jnp.asarray(ramp * sigma_plasma)

        if boundary_cfg is None:
            Ck_0_electrons = Ck_0_electrons.at[0, 0, 0, 0, 0, 0].set(n0_e / (alpha_e**3))
            Ck_0_ions = Ck_0_ions.at[0, 0, 0, 0, 0, 0].set(n0_i / (alpha_i**3))
        else:
            x_real = jnp.linspace(0.0, Lx_real, Nx, endpoint=False)
            ramp = self._build_density_profile(
                x_real,
                Lx_real,
                boundary_cfg["side"],
                boundary_cfg["vacuum_length"].to("m").magnitude,
                boundary_cfg["absorb_length"].to("m").magnitude,
            )
            ne_real = n0_e * ramp
            ni_real = n0_i * ramp

            ne_3d = ne_real[None, :, None]
            ni_3d = ni_real[None, :, None]

            ne_k = jnp.fft.fftn(ne_3d, axes=(-3, -2, -1), norm="forward")
            ni_k = jnp.fft.fftn(ni_3d, axes=(-3, -2, -1), norm="forward")

            Ck_0_electrons = Ck_0_electrons.at[0, 0, 0, :, :, :].set(ne_k / (alpha_e**3))
            Ck_0_ions = Ck_0_ions.at[0, 0, 0, :, :, :].set(ni_k / (alpha_i**3))

        input_parameters["Ck_0_electrons"] = Ck_0_electrons
        input_parameters["Ck_0_ions"] = Ck_0_ions
        self.cfg["spectrax_input"] = input_parameters

    def init_state_and_args(self) -> None:
        super().init_state_and_args()

        grid = self.cfg["grid"]
        sponge_fields = grid.get("sponge_fields", None)
        sponge_plasma = grid.get("sponge_plasma", None)
        if sponge_fields is not None:
            self.grid_quantities_electrons["sponge_fields"] = sponge_fields
            self.grid_quantities_ions["sponge_fields"] = sponge_fields
        if sponge_plasma is not None:
            self.grid_quantities_electrons["sponge_plasma"] = sponge_plasma
            self.grid_quantities_ions["sponge_plasma"] = sponge_plasma

    def init_diffeqsolve(self) -> None:
        if "save" not in self.cfg:
            self.cfg["save"] = {}
        if "moments" not in self.cfg["save"]:
            self.cfg["save"]["moments"] = {"t": {}}
        super().init_diffeqsolve()
