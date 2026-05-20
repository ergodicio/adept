"""PIC-1D simulation domain model.

This module composes the PIC-1D simulation from validated config (datamodel.py)
and the shared Vlasov-1D pieces (EM drivers, density profiles, supergaussian
distribution helpers) so that PIC and Vlasov reuse the same input deck.
"""

from adept._pic1d.datamodel import PIC1DConfig, PICSpeciesConfig
from adept._vlasov1d.grid import Grid
from adept._vlasov1d.simulation import (
    EMDriverSet,
    SubspeciesDistributionSpec,
)
from adept.normalization import PlasmaNormalization, electron_debye_normalization


class PICSpecies:
    """Physical/loading parameters for one species in PIC-1D."""

    def __init__(
        self,
        name: str,
        mass: float,
        charge: float,
        density_components: list[str],
        loading: str,
        vmax_load: float,
    ):
        self.name = name
        self.mass = float(mass)
        self.charge = float(charge)
        self.density_components = list(density_components)
        self.loading = loading
        self.vmax_load = float(vmax_load)

    @staticmethod
    def from_config(cfg: PICSpeciesConfig, default_components: list[str]) -> "PICSpecies":
        components = cfg.density_components if cfg.density_components else default_components
        return PICSpecies(
            name=cfg.name,
            mass=cfg.mass,
            charge=cfg.charge,
            density_components=components,
            loading=cfg.loading,
            vmax_load=cfg.vmax_load,
        )


class PIC1DSimulation:
    def __init__(
        self,
        plasma_norm: PlasmaNormalization,
        grid: Grid,
        species: list[PICSpecies],
        species_distributions: dict[str, list[SubspeciesDistributionSpec]],
        drivers: EMDriverSet,
        ppc: int,
        particle_shape: str,
    ):
        self.plasma_norm = plasma_norm
        self.grid = grid
        self.species = species
        self.species_distributions = species_distributions
        self.drivers = drivers
        self.ppc = ppc
        self.particle_shape = particle_shape

    @property
    def species_dict(self) -> dict[str, PICSpecies]:
        return {s.name: s for s in self.species}


def _density_component_names(cfg: PIC1DConfig) -> list[str]:
    return [name for name in cfg.density.model_extra.keys() if name.startswith("species-")]


def sim_from_config(cfg: PIC1DConfig) -> PIC1DSimulation:
    plasma_norm = electron_debye_normalization(
        cfg.units.normalizing_density,
        cfg.units.normalizing_temperature,
    )
    # If a transverse EM driver is configured we also evolve a vector potential
    # ``a(x)`` via a wave equation; in that case dt must satisfy the EM CFL.
    has_ey_driver = len(cfg.drivers.ey) > 0
    beta = 1.0 / plasma_norm.speed_of_light_norm() if has_ey_driver else 1.0
    grid = Grid.from_config(
        cfg.grid, beta=beta, should_override_dt_for_em_waves=has_ey_driver, norm=plasma_norm
    )

    default_components = _density_component_names(cfg)
    if not default_components:
        raise ValueError("No density components found (expected keys starting with 'species-')")

    if cfg.terms.species:
        species = [PICSpecies.from_config(s, default_components) for s in cfg.terms.species]
    else:
        species = [
            PICSpecies(
                name="electron",
                mass=1.0,
                charge=-1.0,
                density_components=default_components,
                loading="quiet",
                vmax_load=8.0,
            )
        ]

    species_distribution_specs = {
        s.name: [
            SubspeciesDistributionSpec.from_config(cfg.density.get_component(component_name), norm=plasma_norm)
            for component_name in s.density_components
        ]
        for s in species
    }

    drivers = EMDriverSet.from_config(cfg.drivers, norm=plasma_norm)

    return PIC1DSimulation(
        plasma_norm=plasma_norm,
        grid=grid,
        species=species,
        species_distributions=species_distribution_specs,
        drivers=drivers,
        ppc=int(cfg.grid.ppc),
        particle_shape=cfg.grid.particle_shape,
    )
