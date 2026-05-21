"""Domain-model dataclasses for a Vlasov-2D simulation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from adept._vlasov2d.datamodel import (
    AKWDriverConfig,
    EMDriverConfig,
    EMDriverSetConfig,
    Space2DTimeEnvelopeConfig,
    SpeciesComponentConfig,
    SpeciesConfig,
)
from adept._vlasov2d.grid import Grid
from adept.functions import EnvelopeConfig, EnvelopeFunction, NoiseConfig, NoiseProfile
from adept.normalization import PlasmaNormalization, normalize


class Space2DTimeEnvelope(eqx.Module):
    """Separable envelope env_x(x) * env_y(y) * env_t(t)."""

    time_env: EnvelopeFunction
    x_env: EnvelopeFunction
    y_env: EnvelopeFunction

    def __call__(self, x: jax.Array, y: jax.Array, t: float) -> jax.Array:
        return self.time_env(t) * self.x_env(x)[:, None] * self.y_env(y)[None, :]

    @staticmethod
    def from_config(cfg: Space2DTimeEnvelopeConfig, norm: PlasmaNormalization | None = None) -> "Space2DTimeEnvelope":
        return Space2DTimeEnvelope(
            time_env=EnvelopeFunction.from_config(cfg.time, norm, dim="t"),
            x_env=EnvelopeFunction.from_config(cfg.space_x, norm, dim="x"),
            y_env=EnvelopeFunction.from_config(cfg.space_y, norm, dim="x"),
        )


class EMDriver(eqx.Module):
    a0: float
    k0x: float
    k0y: float
    w0: float
    dw0: float
    envelope: Space2DTimeEnvelope
    polarization: str  # "x" or "y" — direction of the current source

    @staticmethod
    def from_config(cfg: EMDriverConfig, norm: PlasmaNormalization | None = None) -> "EMDriver":
        params: AKWDriverConfig = cfg.params
        envelope = Space2DTimeEnvelope.from_config(cfg.envelope, norm)
        return EMDriver(
            a0=float(params.a0),
            k0x=float(params.k0x),
            k0y=float(params.k0y),
            w0=float(params.w0),
            dw0=float(params.dw0),
            envelope=envelope,
            polarization=cfg.polarization,
        )


class EMDriverSet(eqx.Module):
    ex: list[EMDriver]
    ey: list[EMDriver]

    @staticmethod
    def from_config(cfg: EMDriverSetConfig, norm: PlasmaNormalization | None = None) -> "EMDriverSet":
        ex = [EMDriver.from_config(v, norm) for v in cfg.ex.values()]
        ey = [EMDriver.from_config(v, norm) for v in cfg.ey.values()]
        return EMDriverSet(ex=ex, ey=ey)


class Species(eqx.Module):
    name: str
    mass: float
    charge: float
    vmax: float
    nvx: int
    nvy: int
    density_components: list[str]

    @staticmethod
    def from_config(cfg: SpeciesConfig) -> "Species":
        return Species(
            name=cfg.name,
            mass=float(cfg.mass),
            charge=float(cfg.charge),
            vmax=float(cfg.vmax),
            nvx=int(cfg.nvx),
            nvy=int(cfg.nvy),
            density_components=cfg.density_components,
        )


class SubspeciesDensityProfile(eqx.Module):
    """2D density profile: (1 + amp * cos(kx*x + ky*y)) * x_mask(x) * y_mask(y) * (1 + noise)."""

    baseline: float
    amplitude: float
    wavenumber_x: float
    wavenumber_y: float
    x_envelope: EnvelopeFunction | None
    y_envelope: EnvelopeFunction | None
    noise_profile: NoiseProfile

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        xx = x[:, None]
        yy = y[None, :]
        n = self.baseline + self.amplitude * jnp.cos(self.wavenumber_x * xx + self.wavenumber_y * yy)
        if self.x_envelope is not None:
            n = n * self.x_envelope(x)[:, None]
        if self.y_envelope is not None:
            n = n * self.y_envelope(y)[None, :]
        return n * (1.0 + self.noise_profile(n.shape))

    @staticmethod
    def from_config(cfg: SpeciesComponentConfig, norm: PlasmaNormalization | None = None) -> "SubspeciesDensityProfile":
        basis = cfg.basis
        if basis not in ("uniform", "sine"):
            raise NotImplementedError(f"Unknown density basis for vlasov-2d: {basis}")

        baseline = float(cfg.baseline) if cfg.baseline is not None else 1.0
        amplitude = float(cfg.amplitude) if cfg.amplitude is not None else 0.0
        kx = float(cfg.wavenumber_x) if cfg.wavenumber_x is not None else 0.0
        ky = float(cfg.wavenumber_y) if cfg.wavenumber_y is not None else 0.0

        x_env = EnvelopeFunction.from_config(cfg.space_x, norm, dim="x") if cfg.space_x is not None else None
        y_env = EnvelopeFunction.from_config(cfg.space_y, norm, dim="x") if cfg.space_y is not None else None

        noise = NoiseProfile.from_config(
            NoiseConfig(noise_type=cfg.noise_type, noise_val=cfg.noise_val, noise_seed=cfg.noise_seed)
        )
        return SubspeciesDensityProfile(
            baseline=baseline,
            amplitude=amplitude,
            wavenumber_x=kx,
            wavenumber_y=ky,
            x_envelope=x_env,
            y_envelope=y_env,
            noise_profile=noise,
        )


class SubspeciesDistributionSpec(eqx.Module):
    """A single additive component of a species' distribution."""

    density_profile: SubspeciesDensityProfile
    v0x: float
    v0y: float
    T0: float
    supergaussian_order: float

    @staticmethod
    def from_config(
        cfg: SpeciesComponentConfig, norm: PlasmaNormalization | None = None
    ) -> "SubspeciesDistributionSpec":
        return SubspeciesDistributionSpec(
            density_profile=SubspeciesDensityProfile.from_config(cfg, norm),
            v0x=float(cfg.v0x),
            v0y=float(cfg.v0y),
            T0=float(cfg.T0),
            supergaussian_order=float(cfg.m),
        )


class CollisionProfile2D(eqx.Module):
    """nu(x,y,t) = time_env(t) * x_env(x) * y_env(y) for collision frequencies."""

    time_env: EnvelopeFunction | None
    x_env: EnvelopeFunction | None
    y_env: EnvelopeFunction | None

    def __call__(self, x: jax.Array, y: jax.Array, t: float) -> jax.Array:
        out = jnp.ones((x.shape[0], y.shape[0]))
        if self.time_env is not None:
            out = out * self.time_env(t)
        if self.x_env is not None:
            out = out * self.x_env(x)[:, None]
        if self.y_env is not None:
            out = out * self.y_env(y)[None, :]
        return out

    @staticmethod
    def from_envs(
        time_cfg: EnvelopeConfig | None,
        space_x_cfg: EnvelopeConfig | None,
        space_y_cfg: EnvelopeConfig | None,
        norm: PlasmaNormalization | None,
    ) -> "CollisionProfile2D":
        return CollisionProfile2D(
            time_env=EnvelopeFunction.from_config(time_cfg, norm, dim="t") if time_cfg is not None else None,
            x_env=EnvelopeFunction.from_config(space_x_cfg, norm, dim="x") if space_x_cfg is not None else None,
            y_env=EnvelopeFunction.from_config(space_y_cfg, norm, dim="x") if space_y_cfg is not None else None,
        )


class Vlasov2DSimulation:
    """Domain object collecting the physical setup for a single 2D2V run."""

    def __init__(
        self,
        plasma_norm: PlasmaNormalization,
        grid: Grid,
        species: list[Species],
        species_distributions: dict[str, list[SubspeciesDistributionSpec]],
        drivers: EMDriverSet,
        nu_fp_prof: CollisionProfile2D | None = None,
        nu_K_prof: CollisionProfile2D | None = None,
    ):
        self.plasma_norm = plasma_norm
        self.grid = grid
        self.species = species
        self.species_distributions = species_distributions
        self.drivers = drivers
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof

    @property
    def species_dict(self) -> dict[str, Species]:
        return {s.name: s for s in self.species}
