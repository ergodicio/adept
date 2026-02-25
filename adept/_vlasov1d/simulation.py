import equinox as eqx
import jax

from adept._vlasov1d.grid import Grid
from adept.functions import (
    EnvelopeFunction,
    ExponentialFunction,
    LinearFunction,
    NoiseProfile,
    SineFunction,
    SpaceTimeEnvelopeFunction,
    UniformFunction,
)
from adept.normalization import PlasmaNormalization
from adept.utils import SpaceTimeEnvelopeFunction


class Species(eqx.Module):
    """Specification for a single species"""

    name: str
    mass: float
    charge: float
    vmax: float
    nv: int
    density_components: list[str]

    @staticmethod
    def from_config(cfg: dict, norm: PlasmaNormalization | None = None) -> "Species":
        return Species(
            name=cfg["name"],
            mass=float(cfg["mass"]),
            charge=float(cfg["charge"]),
            vmax=cfg["vmax"],
            nv=int(cfg["nv"]),
            density_components=cfg["density_components"],
        )


# Type alias for density profile functions
DensityFunction = UniformFunction | LinearFunction | ExponentialFunction | SineFunction


class SubspeciesDensityProfile(eqx.Module):
    """Complete density profile for a subspecies component: envelope * density * (1 + noise)"""

    density: DensityFunction
    envelope: EnvelopeFunction | None  # Spatial mask (None for sine/uniform without mask)
    noise_profile: NoiseProfile

    def __call__(self, x: jax.Array) -> jax.Array:
        profile = self.density(x)
        if self.envelope is not None:
            profile = self.envelope(x) * profile
        return profile * (1.0 + self.noise_profile(profile.shape))

    @staticmethod
    def from_config(cfg: dict, norm: PlasmaNormalization | None = None) -> "SubspeciesDensityProfile":
        """Parse existing config format into domain model."""
        basis = cfg["basis"]

        if basis == "uniform":
            density = UniformFunction()
            envelope = None
        elif basis == "tanh":
            density = UniformFunction()
            envelope = EnvelopeFunction.from_config(cfg)
        elif basis == "linear":
            density = LinearFunction.from_physical_units_cfg(cfg, norm)
            # Uses defaults: baseline=0.0, bump_height=1.0 (masking envelope)
            envelope = EnvelopeFunction.from_config(cfg)
        elif basis == "exponential":
            density = ExponentialFunction.from_physical_units_cfg(cfg, norm)
            envelope = EnvelopeFunction.from_config(cfg)
        elif basis == "sine":
            density = SineFunction.from_config(cfg)
            envelope = None
        else:
            raise NotImplementedError(f"Unknown density basis: {basis}")

        noise = NoiseProfile.from_config(cfg)
        return SubspeciesDensityProfile(density=density, envelope=envelope, noise_profile=noise)


class SubspeciesDistributionSpec(eqx.Module):
    """Specification for a single subspecies distribution component."""

    density_profile: SubspeciesDensityProfile
    v0: float  # Drift velocity (dimensionless)
    T0: float  # Temperature (dimensionless)
    supergaussian_order: float  # 2.0 = Maxwellian

    @staticmethod
    def from_config(cfg: dict, norm: PlasmaNormalization | None = None) -> "SubspeciesDistributionSpec":
        """Parse density component config into domain model."""
        return SubspeciesDistributionSpec(
            density_profile=SubspeciesDensityProfile.from_config(cfg, norm),
            v0=cfg["v0"],
            T0=cfg["T0"],
            supergaussian_order=cfg.get("m", 2.0),
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
        nu_fp_prof: SpaceTimeEnvelopeFunction | None = None,
        nu_K_prof: SpaceTimeEnvelopeFunction | None = None,
    ):
        self.plasma_norm = plasma_norm
        self.grid = grid
        self.species = species
        self.species_distributions = species_distributions
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof

    @property
    def species_dict(self) -> dict[str, Species]:
        return {s.name: s for s in self.species}
