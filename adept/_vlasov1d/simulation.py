import equinox as eqx
import jax

from adept._vlasov1d.datamodel import (
    EMDriverAKWParametrization,
    EMDriverIntensityWavelengthParametrization,
    EMDriverModel,
    EMDriverSetModel,
)
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
from adept.normalization import UREG, PlasmaNormalization, normalize


class EMDriver(eqx.Module):
    a0: float
    k0: float
    w0: float
    dw0: float
    envelope: SpaceTimeEnvelopeFunction

    @staticmethod
    def from_model(model: EMDriverModel, norm: PlasmaNormalization | None = None) -> "EMDriver":
        envelope = SpaceTimeEnvelopeFunction.from_config(model.envelope.model_dump(), norm)

        params = model.params

        match model.params:
            case EMDriverAKWParametrization():
                c = norm.speed_of_light_norm()
                if params.k0 is None:
                    w0 = c * params.k0
                elif params.w0 is None:
                    k0 = params.w0 / c
                else:
                    k0, w0 = params.k0, params.w0

                return EMDriver(params.a0, k0, w0, params.dw0, envelope)

            case EMDriverIntensityWavelengthParametrization(intensity=intensity, wavelength=wavelength):
                I = UREG.Quantity(intensity).to("W/m^2")
                L = UREG.Quantity(wavelength).to("nm")

                e = UREG.e
                m_e = UREG.m_e
                eps0 = UREG.epsilon_0
                c = UREG.c

                # Standard a0 = eE0/(m_e c w0) — identical to HermiteSRS1D formula
                a0_std = ((e * L / (m_e * math.pi)) * (intensity / (2 * eps0 * c**5)) ** 0.5).to("").magnitude
                # Vlasov normalization: a0_vlasov = a0_std / β  (β = v0/c)
                a0 = a0 * norm.speed_of_light_norm()

                # k0 in Debye-length units: k0_vlasov = k_phys x v0/wp0
                k0_phys = (2 * math.pi / wavelength).to("1/m")
                k0 = float((k0_phys * norm.L0).to("").magnitude)

                # w0 normalized to wp0 (same normalization as Hermite)
                w0_phys = (2 * math.pi * c / wavelength).to("1/s")
                w0 = float((w0_phys * norm.tau).to("").magnitude)

                dw0 = 0.0  # ???

                return EMDriver(a0, k0, w0, dw0, envelope)


class EMDriverSet(eqx.Module):
    ex: list[EMDriver]
    ey: list[EMDriver]

    @staticmethod
    def from_model(model: EMDriverSetModel, norm: PlasmaNormalization | None = None) -> "EMDriverSet":
        ex = [EMDriver.from_model(ex_model, norm) for ex_model in model.ex.values()]
        ey = [EMDriver.from_model(ey_model, norm) for ey_model in model.ey.values()]
        return EMDriverSet(ex, ey)



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
