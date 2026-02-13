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
        """Parse boundary configuration with separate field and plasma absorbing lengths.

        Supports two formats:
        1. Legacy: absorb_length (applies to both fields and plasma)
        2. New: field_absorb_length and plasma_absorb_length (independent control)

        Args:
            Lx_real: Domain length in meters (pint Quantity)

        Returns:
            dict with "field_absorb_length" and "plasma_absorb_length" keys
        """
        if not isinstance(Lx_real, pint.Quantity):
            Lx_real = Lx_real * self.ureg.meter
        bcfg = self.cfg.get("boundaries", None)
        if not isinstance(bcfg, dict):
            return None

        # Check for new format first
        field_absorb = bcfg.get("field_absorb_length", None)
        plasma_absorb = bcfg.get("plasma_absorb_length", None)

        # Backward compatibility: if only absorb_length provided, use for both
        if field_absorb is None and plasma_absorb is None:
            absorb = bcfg.get("absorb_length", None)
            if absorb is None:
                absorb = 0.1 * Lx_real
            else:
                absorb = self._require_quantity(absorb, "boundaries.absorb_length").to("m")
            field_absorb = absorb
            plasma_absorb = absorb
        else:
            # New format: parse separately
            if field_absorb is None:
                field_absorb = 0.1 * Lx_real
            else:
                field_absorb = self._require_quantity(field_absorb, "boundaries.field_absorb_length").to("m")

            if plasma_absorb is None:
                plasma_absorb = 0.1 * Lx_real
            else:
                plasma_absorb = self._require_quantity(plasma_absorb, "boundaries.plasma_absorb_length").to("m")

        # Validate that absorbing regions don't overlap (use wider of the two for check)
        max_absorb = max(field_absorb, plasma_absorb)
        if 2 * max_absorb >= Lx_real:
            raise ValueError(
                f"2 * max(field_absorb_length, plasma_absorb_length) = {2 * max_absorb.to('micron').magnitude:.1f} μm "
                f"must be < Lx = {Lx_real.to('micron').magnitude:.1f} μm (need space for both boundaries)"
            )

        return {
            "field_absorb_length": field_absorb,
            "plasma_absorb_length": plasma_absorb,
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

    def _get_density_config(self, Lx_real):
        """Parse density profile configuration from YAML.

        Returns None for homogeneous plasma, or dict with profile parameters.

        Config format:
            density:
                basis: linear  # or exponential
                min: 0.19  # minimum density at boundaries and in flat regions
                max: 0.27  # maximum density in gradient region
                edge_length: 5um  # length of flat regions at boundaries
                rise: 1um  # smoothing length for transitions
                # For exponential only:
                gradient_scale_length: 100um
                center: 0.5  # normalized position for center density
        """
        dcfg = self.cfg.get("density", None)
        if not isinstance(dcfg, dict):
            return None

        if "basis" not in dcfg:
            return None

        basis = dcfg["basis"]
        if basis == "homogeneous":
            return None

        # Convert lengths to meters
        if "edge_length" in dcfg:
            edge_length = self._require_quantity(dcfg["edge_length"], "density.edge_length").to("m")
        else:
            edge_length = None

        if "rise" in dcfg:
            rise = self._require_quantity(dcfg["rise"], "density.rise").to("m")
        else:
            rise = 0.1 * Lx_real if edge_length is not None else None

        result = {
            "basis": basis,
            "max": float(dcfg.get("max", 0.25)),
            "min": float(dcfg.get("min", 0.19)),
            "edge_length": edge_length,
            "rise": rise,
        }

        if basis == "linear":
            # For linear gradients, just need max/min
            pass
        elif basis == "exponential":
            # For exponential, need scale length
            if "gradient_scale_length" not in dcfg:
                raise ValueError("density.gradient_scale_length required for exponential basis")
            scale_length = self._require_quantity(dcfg["gradient_scale_length"], "density.gradient_scale_length").to(
                "m"
            )
            result["gradient_scale_length"] = scale_length
            result["center"] = float(dcfg.get("center", 0.5))  # Normalized position
        else:
            raise ValueError(f"Unsupported density.basis: {basis} (use 'linear' or 'exponential')")

        return result

    def _build_sponge_profile(self, x_real: jnp.ndarray, Lx_real, absorb_len):
        """Build sponge damping profile with absorbing boundaries on both sides.

        Damping is maximum at x=0 and x=Lx, zero in the interior.
        """
        # Left boundary: damping from x=0 to x=absorb_len
        s_left = x_real / absorb_len
        ramp_left = jnp.where(x_real < absorb_len, jnp.sin(0.5 * jnp.pi * (1.0 - s_left)) ** 2, 0.0)

        # Right boundary: damping from x=Lx-absorb_len to x=Lx
        x_right = Lx_real - x_real
        s_right = x_right / absorb_len
        ramp_right = jnp.where(x_right < absorb_len, jnp.sin(0.5 * jnp.pi * (1.0 - s_right)) ** 2, 0.0)

        # Combine: max of left and right (they shouldn't overlap if validation is correct)
        return jnp.maximum(ramp_left, ramp_right)

    def _build_density_profile(self, x_norm: jnp.ndarray, Lx_norm: float, density_cfg):
        """Build density profile with flat edge regions and gradient in middle.

        Structure: [flat at nmin] - [smooth transition] - [gradient nmin→nmax] - [smooth transition] - [flat at nmin]
        - Flat regions at boundaries stay at nmin (no vacuum)
        - Gradient region in middle goes from nmin to nmax
        - Smooth envelope transitions connect flat to gradient regions

        Args:
            x_norm: Normalized x coordinates (in units of x0)
            Lx_norm: Domain length in normalized units
            density_cfg: Density configuration dict from _get_density_config()

        Returns:
            jnp.ndarray: Density profile (as multiplier of n0_e, n0_i)
        """
        from adept._base_ import get_envelope

        n_min = density_cfg["min"]  # Min density at boundaries and in flat regions
        n_max = density_cfg["max"]  # Max density in gradient region
        basis = density_cfg["basis"]

        # Get physical units for conversions
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]

        # Define gradient region (where density varies from min to max)
        if density_cfg["edge_length"] is not None:
            # Convert edge_length to normalized units
            edge_len_norm = (density_cfg["edge_length"] / x0).to("").magnitude
            rise_norm = (density_cfg["rise"] / x0).to("").magnitude

            # Gradient region is in the middle, excluding flat edge regions
            gradient_left = edge_len_norm
            gradient_right = Lx_norm - edge_len_norm

            if gradient_right <= gradient_left:
                raise ValueError("Domain too small for edge regions")

            gradient_center = 0.5 * (gradient_left + gradient_right)
            gradient_width = gradient_right - gradient_left
        else:
            # No edge regions: gradient fills entire domain
            gradient_center = 0.5 * Lx_norm
            gradient_width = Lx_norm
            rise_norm = 0.1 * Lx_norm

        # Build gradient profile in gradient region
        if basis == "linear":
            # Linear gradient: density increases from n_min to n_max left to right
            x_normalized = (x_norm - (gradient_center - gradient_width * 0.5)) / gradient_width
            gradient = n_min + (n_max - n_min) * x_normalized
        elif basis == "exponential":
            # Exponential gradient with scale length
            scale_length_norm = (density_cfg["gradient_scale_length"] / x0).to("").magnitude
            center_frac = density_cfg.get("center", 0.5)
            x_center = (gradient_center - gradient_width * 0.5) + center_frac * gradient_width
            n_center = n_min + (n_max - n_min) * center_frac
            gradient = n_center * jnp.exp((x_norm - x_center) / scale_length_norm)
        else:
            gradient = 0.5 * (n_min + n_max) * jnp.ones_like(x_norm)

        # Create envelope: smooth transition from 0 to 1 across gradient region
        # get_envelope returns 0 outside [g_L, g_R] and 1 inside, with smooth transitions
        g_L = gradient_center - gradient_width * 0.5
        g_R = gradient_center + gradient_width * 0.5
        envelope = get_envelope(rise_norm, rise_norm, g_L, g_R, x_norm)

        # Apply envelope:
        # - In flat edge regions (envelope=0): density = n_min
        # - In gradient region (envelope=1): density = gradient (n_min to n_max)
        # - In transition: smooth interpolation
        density = n_min + envelope * (gradient - n_min)

        return density

    def _get_plasma_bounds_norm(self):
        """Get the interior plasma region bounds.

        If density gradient is configured, returns bounds excluding flat edge regions.
        Otherwise, returns bounds excluding absorbing boundaries.
        This defines the "plasma" region for focused plotting.
        """
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]
        Lx_norm = float(self.cfg["physics"]["Lx"])
        Lx_real = Lx_norm * x0

        # Check if density gradient with edge regions is configured
        density_cfg = self._get_density_config(Lx_real)
        if density_cfg is not None and density_cfg["edge_length"] is not None:
            # Use edge length to define plasma bounds
            edge_len_norm = (density_cfg["edge_length"] / x0).to("").magnitude
            xmin = edge_len_norm
            xmax = Lx_norm - edge_len_norm

            if xmax <= xmin:
                return None
            return xmin, xmax

        # Fall back to using plasma absorbing boundaries
        if "boundaries" not in self.cfg:
            return None

        boundary_cfg = self._get_boundary_config(Lx_real)
        if boundary_cfg is None:
            return None

        # Use plasma absorb length to define plasma region (wider than field absorb)
        plasma_absorb_norm = (boundary_cfg["plasma_absorb_length"] / x0).to("").magnitude

        # Interior region: between both plasma absorbing boundaries
        xmin = plasma_absorb_norm
        xmax = Lx_norm - plasma_absorb_norm

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
            plt.savefig(os.path.join(plots_dir, f"spacetime-{field_name}.png"), bbox_inches="tight")
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
            plt.savefig(os.path.join(plots_dir, f"{field_name}.png"), bbox_inches="tight")
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
            plt.savefig(os.path.join(plots_dir, f"spacetime-{name}.png"), bbox_inches="tight")
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
            plt.savefig(os.path.join(plots_dir, f"{name}.png"), bbox_inches="tight")
            plt.close()

    def _plot_density_profile(self, td: str) -> None:
        """Plot the density profile and region boundaries for diagnostics."""
        grid = self.cfg["grid"]
        if "density_profile" not in grid:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        density_profile = np.array(grid["density_profile"])
        physics = self.cfg["physics"]
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]

        # Get x-axis in physical units
        Lx_norm = float(physics["Lx"])
        Nx = len(density_profile)
        x_norm = np.linspace(0, Lx_norm, Nx, endpoint=False)
        x_micron = (x_norm * x0).to("micron").magnitude

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_micron, density_profile, "b-", linewidth=2, label="Density profile")
        ax.set_xlabel("x (μm)", fontsize=12)
        ax.set_ylabel("Density (normalized to $n_0$)", fontsize=12)
        ax.set_title("Density Profile with Absorbing Regions", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Get region boundaries
        Lx_real = Lx_norm * x0
        density_cfg = self._get_density_config(Lx_real)

        # Mark absorbing boundaries if configured
        if "boundaries" in self.cfg:
            boundary_cfg = self._get_boundary_config(Lx_real)
            if boundary_cfg is not None:
                # Field absorbing boundaries (narrow, solid green)
                field_absorb_micron = boundary_cfg["field_absorb_length"].to("micron").magnitude
                ax.axvline(
                    field_absorb_micron, color="g", linestyle="-", alpha=0.7, linewidth=1.5, label="Field absorb"
                )
                ax.axvline(x_micron[-1] - field_absorb_micron, color="g", linestyle="-", alpha=0.7, linewidth=1.5)

                # Plasma absorbing boundaries (wider, dashed orange)
                plasma_absorb_micron = boundary_cfg["plasma_absorb_length"].to("micron").magnitude
                ax.axvline(
                    plasma_absorb_micron,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label="Plasma absorb",
                )
                ax.axvline(
                    x_micron[-1] - plasma_absorb_micron, color="orange", linestyle="--", alpha=0.7, linewidth=1.5
                )

                # Shade absorbing regions
                ax.axvspan(0, plasma_absorb_micron, alpha=0.15, color="orange", label="Absorbing region")
                ax.axvspan(x_micron[-1] - plasma_absorb_micron, x_micron[-1], alpha=0.15, color="orange")

        # Add horizontal lines for density levels
        if density_cfg is not None:
            n_min = density_cfg["min"]
            n_max = density_cfg["max"]
            ax.axhline(n_min, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax.axhline(n_max, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax.text(
                0.02,
                n_min,
                f"n_min={n_min:.2f}",
                transform=ax.get_yaxis_transform(),
                fontsize=10,
                verticalalignment="bottom",
            )
            ax.text(
                0.02,
                n_max,
                f"n_max={n_max:.2f}",
                transform=ax.get_yaxis_transform(),
                fontsize=10,
                verticalalignment="bottom",
            )

        ax.legend(loc="best", fontsize=10)
        ax.set_ylim(bottom=0)

        # Save plot
        plots_dir = os.path.join(td, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "density_profile.png"), bbox_inches="tight")
        plt.close()

    def post_process(self, run_output: dict, td: str) -> dict:
        """Post-process with density profile plotting."""
        # Plot density profile first (if it exists)
        self._plot_density_profile(td)

        # Call parent post-processing
        return super().post_process(run_output, td)

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
        for driver_key, driver_group in drivers.items():
            if not isinstance(driver_group, dict):
                continue
            for _, pulse in driver_group.items():
                if not isinstance(pulse, dict):
                    continue

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

        # Use homogeneous plasma density for dt calculation
        if n0_e <= 0.0:
            raise ValueError("n0_e must be > 0 for dt calculation")

        dt_plasma = 0.1 / np.sqrt(n0_e)
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
                # No boundaries configured - no damping
                ramp_fields = jnp.zeros_like(x_real)
                ramp_plasma = jnp.zeros_like(x_real)
            else:
                # Build separate spatial profiles for fields and plasma
                field_absorb_len = boundary_cfg["field_absorb_length"].to("m").magnitude
                plasma_absorb_len = boundary_cfg["plasma_absorb_length"].to("m").magnitude

                ramp_fields = self._build_sponge_profile(x_real, Lx_real, field_absorb_len)
                ramp_plasma = self._build_sponge_profile(x_real, Lx_real, plasma_absorb_len)

            grid["sponge_fields"] = jnp.asarray(ramp_fields * sigma_fields)
            grid["sponge_plasma"] = jnp.asarray(ramp_plasma * sigma_plasma)

        # Initialize with homogeneous density - will be modified in init_state_and_args if gradient is specified
        Ck_0_electrons = Ck_0_electrons.at[0, 0, 0, 0, 0, 0].set(n0_e / (alpha_e**3))
        Ck_0_ions = Ck_0_ions.at[0, 0, 0, 0, 0, 0].set(n0_i / (alpha_i**3))

        input_parameters["Ck_0_electrons"] = Ck_0_electrons
        input_parameters["Ck_0_ions"] = Ck_0_ions
        self.cfg["spectrax_input"] = input_parameters

    def init_state_and_args(self) -> None:
        super().init_state_and_args()

        grid = self.cfg["grid"]
        physics = self.cfg["physics"]
        derived = self.cfg["units"]["derived"]
        x0 = derived["x0"]

        # Apply sponge profiles if configured
        sponge_fields = grid.get("sponge_fields", None)
        sponge_plasma = grid.get("sponge_plasma", None)
        if sponge_fields is not None:
            self.grid_quantities_electrons["sponge_fields"] = sponge_fields
            self.grid_quantities_ions["sponge_fields"] = sponge_fields
        if sponge_plasma is not None:
            self.grid_quantities_electrons["sponge_plasma"] = sponge_plasma
            self.grid_quantities_ions["sponge_plasma"] = sponge_plasma

        # Apply density gradient if configured
        Lx_norm = float(physics["Lx"])
        Lx_real = Lx_norm * x0
        density_cfg = self._get_density_config(Lx_real)

        if density_cfg is not None:
            # Build density profile
            Nx = int(grid["Nx"])
            x_norm = jnp.linspace(0.0, Lx_norm, Nx, endpoint=False)
            density_profile = self._build_density_profile(x_norm, Lx_norm, density_cfg)

            # Get reference densities and thermal velocities
            n0_e = float(physics.get("n0_e", 1.0))
            n0_i = float(physics.get("n0_i", 1.0))
            input_parameters = self.cfg["spectrax_input"]
            alpha_e = float(input_parameters["alpha_s"][0])
            alpha_i = float(input_parameters["alpha_s"][3])

            # Apply density profile to equilibrium Hermite coefficient in state
            # For a Maxwellian: C_000(x) = n(x) / alpha^3
            n_e_profile = n0_e * density_profile / (alpha_e**3)
            n_i_profile = n0_i * density_profile / (alpha_i**3)

            # Modify state directly (state stores float64 views of complex arrays)
            # Need to convert complex profiles to float64 view for state storage
            Ck_e_complex = self.state["Ck_electrons"].view(jnp.complex128)
            Ck_i_complex = self.state["Ck_ions"].view(jnp.complex128)

            # Set spatially varying density for all x-locations
            Ck_e_complex = Ck_e_complex.at[0, 0, 0, 0, :, 0].set(n_e_profile)
            Ck_i_complex = Ck_i_complex.at[0, 0, 0, 0, :, 0].set(n_i_profile)

            # Update state with modified coefficients (as float64 views)
            self.state["Ck_electrons"] = Ck_e_complex.view(jnp.float64)
            self.state["Ck_ions"] = Ck_i_complex.view(jnp.float64)

            # Store density profile for diagnostics
            grid["density_profile"] = jnp.asarray(density_profile)

    def init_diffeqsolve(self) -> None:
        if "save" not in self.cfg:
            self.cfg["save"] = {}
        if "moments" not in self.cfg["save"]:
            self.cfg["save"]["moments"] = {"t": {}}
        super().init_diffeqsolve()
