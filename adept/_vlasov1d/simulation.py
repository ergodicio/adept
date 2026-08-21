"""Domain objects that represent a configured Vlasov-1D simulation."""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu
from jaxtyping import Array

from adept._vlasov1d.datamodel import (
    AKWDriverConfig,
    BroadbandConfig,
    EMDriverConfig,
    EMDriverSetConfig,
    IntensityWavelengthDriverConfig,
    SpeciesComponentConfig,
    SpeciesConfig,
    StochasticDriverConfig,
)
from adept._vlasov1d.grid import Grid
from adept.functions import (
    EnvelopeConfig,
    EnvelopeFunction,
    ExponentialFunction,
    GradientDensityConfig,
    LinearFunction,
    NoiseConfig,
    NoiseProfile,
    SineDensityConfig,
    SineFunction,
    SpaceTimeEnvelopeFunction,
    UniformFunction,
)
from adept.normalization import PlasmaNormalization


class EMDriver(eqx.Module):
    """Normalized electromagnetic driver used by longitudinal or transverse sources."""

    a0: float
    k0: float
    w0: float
    phase: float
    dw0: float
    envelope: SpaceTimeEnvelopeFunction
    is_point_source: bool = False

    @staticmethod
    def from_config(cfg: EMDriverConfig, norm: PlasmaNormalization | None = None) -> "list[EMDriver] | BroadbandDriver":
        """Convert user driver configuration into normalized solver parameters."""
        envelope = SpaceTimeEnvelopeFunction.from_config(cfg.envelope, norm)

        params = cfg.params
        # local import avoids the simulation <-> helpers module cycle
        from adept._vlasov1d.helpers import get_akw_from_intensity_wavelength

        match cfg.params:
            case AKWDriverConfig():
                c = norm.speed_of_light_norm()
                if params.k0 is None:
                    w0 = c * params.k0
                elif params.w0 is None:
                    k0 = params.w0 / c
                else:
                    k0, w0 = params.k0, params.w0

                is_point = cfg.source_type == "point"
                return [EMDriver(params.a0, k0, w0, params.phase, params.dw0, envelope, is_point_source=is_point)]

            case IntensityWavelengthDriverConfig(intensity=intensity, wavelength=wavelength, leftgoing=leftgoing):
                a0, k0, w0 = get_akw_from_intensity_wavelength(intensity, wavelength, leftgoing, norm)

                dw0 = params.dw0
                is_point = cfg.source_type == "point"
                return [EMDriver(a0, k0, w0, params.phase, dw0, envelope, is_point_source=is_point)]

            case BroadbandConfig(intensities=intensities, wavelength=wavelength, leftgoing=leftgoing):
                a0, k0, w0 = get_akw_from_intensity_wavelength(intensities.base_intensity, wavelength, leftgoing, norm)

                is_point = cfg.source_type == "point"
                broadband_driver = BroadbandDriver(params.model_dump(), a0, k0, w0, envelope, is_point)
                return broadband_driver

            case _:
                raise NotImplementedError(f"Unsupported driver params type: {type(cfg.params).__name__}")


class BroadbandDriver(eqx.Module):
    """Multi-color (broadband) ey driver carrying per-line parameter arrays.

    Given the monochromatic amplitude ``a0`` for ``intensities.base_intensity`` and
    per-line weights ``w_j`` (uniform or seeded-random), the line amplitudes are
    ``a_j = a0 * sqrt(w_j / sum_k w_k)`` so that ``sum_j a_j^2 = a0^2`` -- the comb
    carries the same *time-averaged* power as the single line (a phase-locked comb
    still peaks at ``a0 * sqrt(N)`` at recurrence). Line frequencies are
    ``w_j = w0 * (1 + d_j)``, ``d_j`` uniform on ``[-delta_omega, +delta_omega]``
    (``delta_omega`` is the half-width; ``num_colors == 1`` sits exactly at ``w0``).

    All lines share the carrier ``k0``: the source is
    ``-env * w_j^2 * a_j * sin(k0 x - w_j t + phi_j)``, so an off-center line is not
    placed on its own dispersion branch. This is accurate for a localized antenna
    (``source_type: point`` / narrow spatial envelope, where ``dk * L_antenna`` is
    negligible) and NOT for a source extended across the box.
    """

    params: dict
    a0: float
    k0: float
    w0: float
    intensity_weights: Array  # raw per-line weights w_j (dimensionless)
    amplitudes: Array  # per-line a_j = a0 * sqrt(w_j / sum w)
    delta_omega: Array
    phases: Array
    envelope: SpaceTimeEnvelopeFunction
    is_point_source: bool = False

    def __init__(self, cfg: dict, a0, k0, w0, envelope, is_point):
        self.params = cfg
        self.a0 = a0
        self.k0 = k0
        self.w0 = w0
        self.envelope = envelope
        self.is_point_source = is_point

        n_colors = int(self.params["num_colors"])

        # per-line intensity weights w_j
        if self.params["intensities"]["init"] == "random":
            int_lo, int_hi = self.params["intensities"].get("range", (0.0, 2.0))
            int_rng = np.random.default_rng(seed=self.params["intensities"]["seed"])
            self.intensity_weights = jnp.array(int_rng.uniform(int_lo, int_hi, n_colors))
        elif self.params["intensities"]["init"] == "uniform":
            self.intensity_weights = jnp.ones(n_colors)
        else:
            raise NotImplementedError(f"Initialization type -- {self.params['intensities']['init']} -- not implemented")
        # amplitudes: sqrt normalization so sum_j a_j^2 = a0^2 (same time-averaged power as
        # the monochromatic line; otherwise a uniform comb would carry N x the power)
        self.amplitudes = self.a0 * jnp.sqrt(self.intensity_weights / jnp.sum(self.intensity_weights))

        # frequency shifts:
        if n_colors == 1:
            # a single line must sit exactly at w0 (this is what makes num_colors: 1 the monochromatic driver)
            self.delta_omega = jnp.zeros(1)
        else:
            self.delta_omega = jnp.linspace(-self.params["delta_omega"], self.params["delta_omega"], n_colors) * self.w0

        if self.params["phases"]["init"] == "random":
            # Spectral phases drawn uniformly over (0, 2*pi) -- the default;
            # Override via phases.range
            phase_lo, phase_hi = self.params["phases"].get("range", (0.0, 2.0 * np.pi))
            phase_rng = np.random.default_rng(seed=self.params["phases"]["seed"])
            self.phases = jnp.array(phase_rng.uniform(phase_lo, phase_hi, self.params["num_colors"]))
        elif self.params["phases"]["init"] == "uniform":
            self.phases = jnp.ones(self.params["num_colors"]) * self.params["phases"]["base_phase"]
        else:
            raise NotImplementedError(f"Initialization type -- {self.params['phases']['init']} -- not implemented")

    def scale_intensities(self, intensities):
        if self.params["intensities"]["activation"] == "linear":
            ints = 0.5 * (jnp.tanh(intensities) + 1.0)
        elif self.params["intensities"]["activation"] == "log":
            ints = 3 * (jnp.tanh(intensities) + 1.0) - 3
            ints = 10**ints
        elif self.params["intensities"]["activation"] == "log-3wide":
            ints = -1.5 * (jnp.tanh(intensities) + 1.0)  # from 0 to -3
            ints = 10**ints
        else:
            raise NotImplementedError(
                f"Amplitude Output type -- {self.params['intensities']['activation']} -- not implemented"
            )

        return ints

    def get_partition_spec(self):
        """
        Get the partition spec for the model

        Depends what is learned based on the driver being passed in

        Returns
        -------
        filter_spec : pytree with the same structure as the model

        """
        filter_spec = jtu.tree_map(lambda _: False, self)

        if self.params["intensities"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.intensities, filter_spec, replace=True)

        if self.params["phases"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.phases, filter_spec, replace=True)

        return filter_spec

    # def __call__(self, state: dict, args: dict) -> tuple:
    #     # intensities = self.scale_intensities(self.intensity_weights)
    #     # intensities = intensities / jnp.sum(intensities)

    #     args["drivers"]["ey"] = {
    #         "delta_omega": self.delta_omega,
    #         "phases": self.phases,
    #         "amplitudes": self.amplitudes,
    #     } | self.envelope

    #     return state, args


class StochasticDriver(eqx.Module):
    """Band-limited, time-correlated (Ornstein-Uhlenbeck) longitudinal field driver.

    Drives the plasma with an external electric field built from a set of box
    Fourier modes whose complex amplitudes evolve as independent OU processes
    with correlation time ``tau`` and stationary RMS ``amplitude``:

        dE(x, t) = sum_m Re[a_m(t) exp(i k_m x)],   k_m = 2 pi m / L

    The realization is drawn once at construction from ``seed`` on a uniform
    time grid, so the forcing is reproducible, deterministic inside the solve,
    and independent of the solver timestep; the amplitudes are linearly
    interpolated in time when evaluated.
    """

    t_grid: jnp.ndarray
    amp_real: jnp.ndarray
    amp_imag: jnp.ndarray
    k_modes: jnp.ndarray

    def __init__(self, cfg: StochasticDriverConfig, grid: Grid):
        """Precompute the OU amplitude time series for each driven mode."""
        length = grid.xmax - grid.xmin
        modes = np.asarray(cfg.modes, dtype=np.float64)
        n_modes = len(modes)

        dt_update = cfg.dt_update if cfg.dt_update is not None else cfg.tau / 10.0
        dt_update = min(dt_update, cfg.tau / 2.0)  # never undersample the OU process
        nt = int(np.ceil((grid.tmax - grid.tmin) / dt_update)) + 2

        rng = np.random.default_rng(cfg.seed)
        theta = np.exp(-dt_update / cfg.tau)
        kick = cfg.amplitude * np.sqrt(1.0 - theta**2)

        amps = np.zeros((nt, n_modes), dtype=np.complex128)
        amps[0] = cfg.amplitude * (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)) / np.sqrt(2.0)
        for it in range(1, nt):
            xi = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)) / np.sqrt(2.0)
            amps[it] = theta * amps[it - 1] + kick * xi

        self.t_grid = jnp.array(grid.tmin + dt_update * np.arange(nt))
        self.amp_real = jnp.array(amps.real)
        self.amp_imag = jnp.array(amps.imag)
        self.k_modes = jnp.array(2.0 * np.pi * modes / length)

    def __call__(self, t: float, x: jax.Array) -> jax.Array:
        """Evaluate the stochastic forcing field at time t on the spatial grid x."""
        ar = jax.vmap(lambda col: jnp.interp(t, self.t_grid, col), in_axes=1)(self.amp_real)
        ai = jax.vmap(lambda col: jnp.interp(t, self.t_grid, col), in_axes=1)(self.amp_imag)
        phase = self.k_modes[:, None] * x[None, :]
        return jnp.sum(ar[:, None] * jnp.cos(phase) - ai[:, None] * jnp.sin(phase), axis=0)


class EMDriverSet(eqx.Module):
    """Container for longitudinal (Ex) and transverse (Ey) driver lists."""

    ex: list[EMDriver]
    ey: list[EMDriver | BroadbandDriver]
    ex_stochastic: StochasticDriver | None = None

    @staticmethod
    def from_config(
        cfg: EMDriverSetConfig, norm: PlasmaNormalization | None = None, grid: Grid | None = None
    ) -> "EMDriverSet":
        """Build normalized Ex and Ey driver lists (and optional stochastic forcing) from configuration."""
        ex = []
        for ex_cfg in cfg.ex.values():
            obj = EMDriver.from_config(ex_cfg, norm)
            ex.extend(obj if isinstance(obj, list) else [obj])

        ey = []
        for ey_cfg in cfg.ey.values():
            obj = EMDriver.from_config(ey_cfg, norm)
            ey.extend(obj if isinstance(obj, list) else [obj])

        ex_stochastic = None
        if cfg.ex_stochastic is not None:
            if grid is None:
                raise ValueError("A Grid is required to construct the stochastic Ex driver")
            ex_stochastic = StochasticDriver(cfg.ex_stochastic, grid)
        return EMDriverSet(ex, ey, ex_stochastic)


class Species(eqx.Module):
    """Specification for a single species"""

    name: str
    mass: float
    charge: float
    vmax: float
    vmin: float
    nv: int
    density_components: list[str]

    @staticmethod
    def from_config(cfg: SpeciesConfig) -> "Species":
        """Convert a species config into the immutable simulation species model.

        ``vmin`` defaults to ``-vmax`` (the historical symmetric velocity grid)
        when it is not specified in the config.
        """
        vmax = float(cfg.vmax)
        vmin = float(cfg.vmin) if cfg.vmin is not None else -vmax
        return Species(
            name=cfg.name,
            mass=float(cfg.mass),
            charge=float(cfg.charge),
            vmax=vmax,
            vmin=vmin,
            nv=int(cfg.nv),
            density_components=cfg.density_components,
        )


# Type alias for density profile functions
DensityFunction = UniformFunction | LinearFunction | ExponentialFunction | SineFunction


class SubspeciesDensityProfile(eqx.Module):
    """Complete density profile for a subspecies component: envelope * density * (1 + noise)"""

    density: DensityFunction
    envelope: EnvelopeFunction | None  # Spatial mask (None for sine/uniform without mask)
    noise_profile: NoiseProfile

    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the noisy, optionally enveloped density profile at positions x."""
        profile = self.density(x)
        if self.envelope is not None:
            profile = self.envelope(x) * profile
        return profile * (1.0 + self.noise_profile(profile.shape))

    @staticmethod
    def from_config(cfg: SpeciesComponentConfig, norm: PlasmaNormalization | None = None) -> "SubspeciesDensityProfile":
        """Parse config into domain model."""
        basis = cfg.basis

        if basis == "uniform":
            density = UniformFunction(float(cfg.baseline) if cfg.baseline is not None else 1.0)
            envelope = None
        elif basis == "tanh":
            density = UniformFunction()
            envelope_cfg = EnvelopeConfig.model_validate(cfg.model_dump(exclude_none=True))
            envelope = EnvelopeFunction.from_config(envelope_cfg, norm, dim="x")
        elif basis == "linear":
            grad_cfg = GradientDensityConfig(
                center=cfg.center,
                gradient_scale_length=cfg.gradient_scale_length,
                val_at_center=cfg.val_at_center,
            )
            density = LinearFunction.from_config(grad_cfg, norm)
            envelope_cfg = EnvelopeConfig.model_validate(cfg.model_dump(exclude_none=True))
            envelope = EnvelopeFunction.from_config(envelope_cfg, norm, dim="x")
        elif basis == "exponential":
            grad_cfg = GradientDensityConfig(
                center=cfg.center,
                gradient_scale_length=cfg.gradient_scale_length,
                val_at_center=cfg.val_at_center,
            )
            density = ExponentialFunction.from_config(grad_cfg, norm)
            envelope_cfg = EnvelopeConfig.model_validate(cfg.model_dump(exclude_none=True))
            envelope = EnvelopeFunction.from_config(envelope_cfg, norm, dim="x")
        elif basis == "sine":
            sine_cfg = SineDensityConfig(
                baseline=cfg.baseline,
                amplitude=cfg.amplitude,
                wavenumber=cfg.wavenumber,
            )
            density = SineFunction.from_config(sine_cfg, norm)
            envelope = None
        else:
            raise NotImplementedError(f"Unknown density basis: {basis}")

        noise_cfg = NoiseConfig(
            noise_type=cfg.noise_type,
            noise_val=cfg.noise_val,
            noise_seed=cfg.noise_seed,
        )
        noise = NoiseProfile.from_config(noise_cfg)
        return SubspeciesDensityProfile(density=density, envelope=envelope, noise_profile=noise)


class SubspeciesDistributionSpec(eqx.Module):
    """Specification for a single subspecies distribution component."""

    density_profile: SubspeciesDensityProfile
    v0: float  # Drift velocity (dimensionless)
    T0: float  # Temperature (dimensionless)
    supergaussian_order: float  # 2.0 = Maxwellian

    @staticmethod
    def from_config(
        cfg: SpeciesComponentConfig, norm: PlasmaNormalization | None = None
    ) -> "SubspeciesDistributionSpec":
        """Parse density component config into domain model."""
        return SubspeciesDistributionSpec(
            density_profile=SubspeciesDensityProfile.from_config(cfg, norm),
            v0=float(cfg.v0),
            T0=float(cfg.T0),
            supergaussian_order=float(cfg.m),
        )


class Vlasov1DSimulation:
    """
    Domain object representing a Vlasov-1D simulation setup.
    Holds the physical parameters computed from config.
    """

    def __init__(
        self,
        plasma_norm: PlasmaNormalization,
        grid: Grid,
        species: list[Species],
        species_distributions: dict[str, list[SubspeciesDistributionSpec]],
        drivers: EMDriverSet,
        nu_fp_prof: SpaceTimeEnvelopeFunction | None = None,
        nu_K_prof: SpaceTimeEnvelopeFunction | None = None,
    ):
        """Store normalized simulation inputs needed by the solver and module wrapper."""
        self.plasma_norm = plasma_norm
        self.grid = grid
        self.species = species
        self.species_distributions = species_distributions
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof
        self.drivers = drivers

    @property
    def species_dict(self) -> dict[str, Species]:
        """Map species names to their simulation species definitions."""
        return {s.name: s for s in self.species}
