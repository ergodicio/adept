"""Pydantic configuration models for Vlasov-1D simulations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adept.functions import EnvelopeConfig, SpaceTimeEnvelopeConfig
from adept.normalization import PlasmaNormalization, electron_debye_normalization, ion_debye_normalization


# TODO(gh-250): refactor to use a nested SpaceTimeEnvelopeConfig instead of inlining
# envelope fields. See https://github.com/ergodicio/adept/issues/250
class SpeciesComponentConfig(BaseModel):
    """Density and velocity-space parameters for one species density component."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    noise_seed: int
    noise_type: str
    noise_val: float
    v0: float
    T0: float
    m: float = 2.0
    basis: str
    # Sine basis
    baseline: float | None = None
    amplitude: float | None = None
    wavenumber: float | None = None
    # Gradient basis (linear/exponential)
    center: float | str | None = None
    gradient_scale_length: float | str | None = Field(default=None, alias="gradient scale length")
    val_at_center: float | str | None = Field(default=None, alias="val at center")
    rise: float | str | None = None
    width: float | str | None = None
    # Envelope fields (tanh basis, or uniform top-level envelope)
    bump_or_trough: str | None = None
    bump_height: float | None = None
    slope: float | None = None
    # Space profile (uniform basis with nested envelope)
    space_profile: EnvelopeConfig | None = Field(default=None, alias="space-profile")


class DensityConfig(BaseModel):
    """Collection of species density components and quasineutrality setting."""

    model_config = ConfigDict(extra="allow")

    quasineutrality: bool

    def get_component(self, name: str) -> SpeciesComponentConfig:
        """Return a named species density component as a validated config model."""
        return SpeciesComponentConfig.model_validate(self.model_extra[name])


class UnitsConfig(BaseModel):
    """Physical units used to construct the plasma normalization."""

    normalizing_temperature: str
    normalizing_density: str
    # Reference species for the normalization. The default "electron" gives
    # electron units (t in 1/omega_pe, v in sqrt(T0/m_e), L0 = lambda_De).
    # "ion" gives the ion analogue (t in 1/omega_pi, v in v_ti = sqrt(T0/m_i))
    # for kinetic-ion runs such as `terms.field: poisson-boltzmann`; there
    # normalizing_density and normalizing_temperature are the *ion* density
    # and temperature.
    reference: Literal["electron", "ion"] = "electron"
    # Ion mass number (m_i = A m_p) and charge state (q_i = Z e).
    # Only used when reference == "ion".
    A: float = Field(default=1.0, gt=0)
    Z: float = Field(default=1.0, gt=0)

    def make_normalization(self) -> PlasmaNormalization:
        """Build the reference-species normalization described by this config."""
        if self.reference == "ion":
            return ion_debye_normalization(self.normalizing_density, self.normalizing_temperature, A=self.A, Z=self.Z)
        return electron_debye_normalization(self.normalizing_density, self.normalizing_temperature)


class GridConfig(BaseModel):
    """Configuration-space and velocity-space grid parameters."""

    dt: float | str
    nx: int
    tmin: float | str = 0.0
    tmax: float | str
    xmax: float | str
    xmin: float | str
    nv: int | None = None
    vmax: float | None = None
    vmin: float | None = None
    parallel: tuple[str, ...] | bool = False

    @field_validator("parallel", mode="before")
    @classmethod
    def validate_parallel(cls, v):
        """Normalize and validate the optional parallelization axis list."""
        if v is False or v is None:
            return False
        if isinstance(v, (list, tuple)):
            valid = {"x", "v"}
            invalid = set(v) - valid
            if invalid:
                raise ValueError(f"parallel axes must be 'x' and/or 'v', got unknown: {invalid}")
            return tuple(v)
        raise ValueError(f"parallel must be False or a sequence of axes ('x', 'v'), got: {v!r}")


class TimeSaveConfig(BaseModel):
    """Temporal sampling configuration for saved quantities."""

    tmin: float | None = None
    tmax: float | None = None
    nt: int


class SaveConfig(BaseModel):
    """Top-level save configuration keyed by field, diagnostic, or species name."""

    model_config = ConfigDict(extra="allow")

    fields: dict[str, TimeSaveConfig]


class IntensityWavelengthDriverConfig(BaseModel):
    """Laser driver parameters specified by physical intensity and wavelength."""

    intensity: str
    wavelength: str
    leftgoing: bool = False
    dw0: float = 0.0
    phase: float = 0.0


class BroadbandIntensitiesConfig(BaseModel):
    """Per-line intensity weights of a broadband (multi-color) driver.

    The weights ``w_j`` set the line *amplitudes* as ``a_j = a0 * sqrt(w_j / sum_k w_k)``,
    where ``a0`` is the monochromatic amplitude for ``base_intensity``; hence
    ``sum_j a_j^2 = a0^2`` (same time-averaged power as a single line at
    ``base_intensity``) and the per-line intensity is ``I_j = base_intensity * w_j / sum_k w_k``.
    """

    base_intensity: str  # e.g. "2.378e+14 W/cm^2"; the total (monochromatic-equivalent) intensity
    init: Literal["uniform", "random"] = "uniform"
    seed: int | None = None  # required for init: random
    range: tuple[float, float] = (0.0, 2.0)  # uniform draw bounds for init: random

    @model_validator(mode="after")
    def check_seed(self) -> "BroadbandIntensitiesConfig":
        """``init: random`` must be reproducible, so it needs a seed."""
        if self.init == "random" and self.seed is None:
            raise ValueError("intensities.init == 'random' requires intensities.seed")
        return self


class BroadbandPhasesConfig(BaseModel):
    """Per-line spectral phases of a broadband (multi-color) driver."""

    init: Literal["uniform", "random"] = "random"
    seed: int | None = None  # required for init: random
    range: tuple[float, float] = (0.0, 2.0 * 3.141592653589793)  # uniform draw bounds, radians
    base_phase: float = 0.0  # the common phase for init: uniform

    @model_validator(mode="after")
    def check_seed(self) -> "BroadbandPhasesConfig":
        """``init: random`` must be reproducible, so it needs a seed."""
        if self.init == "random" and self.seed is None:
            raise ValueError("phases.init == 'random' requires phases.seed")
        return self


class BroadbandConfig(BaseModel):
    """Broadband (multi-color) laser driver: a comb of ``num_colors`` lines.

    Line frequencies are ``w_j = w0 * (1 + d_j)`` with ``d_j`` spaced uniformly on
    ``[-delta_omega, +delta_omega]`` (``num_colors: 1`` puts the single line exactly at
    ``w0``). **``delta_omega`` is the HALF-width of the comb as a fraction of ``w0``;
    the full bandwidth is ``2 * delta_omega``** -- a "0.25% bandwidth" run is
    ``delta_omega: 0.00125``. All lines share the carrier wavenumber ``k0`` of
    ``wavelength`` (an off-center line is not launched on its own dispersion branch),
    which is accurate for a localized antenna (``source_type: point`` / a narrow
    spatial envelope) and not for a source extended across the box.
    """

    num_colors: int = Field(ge=1)
    delta_omega: float = Field(ge=0.0)  # half-width, fraction of w0
    wavelength: str
    intensities: BroadbandIntensitiesConfig
    phases: BroadbandPhasesConfig
    leftgoing: bool = False


class AKWDriverConfig(BaseModel):
    """Laser driver parameters specified directly as amplitude, wavenumber, and frequency."""

    a0: float
    k0: float | None = None
    w0: float | None = None
    dw0: float
    phase: float = 0.0

    @model_validator(mode="after")
    def check_w_or_k(self) -> "AKWDriverConfig":
        """Require enough information to infer both driver wavenumber and frequency."""
        if self.k0 is None and self.w0 is None:
            raise ValueError("You must specify at least one of k0 or w0.")
        return self


class EMDriverConfig(BaseModel):
    """One electromagnetic driver with parameters, envelope, and source geometry."""

    params: IntensityWavelengthDriverConfig | AKWDriverConfig | BroadbandConfig
    envelope: SpaceTimeEnvelopeConfig
    source_type: Literal["extended", "point"] = "extended"


class StochasticDriverConfig(BaseModel):
    """Time-correlated (Ornstein-Uhlenbeck) stochastic forcing of the longitudinal field.

    Each entry of ``modes`` is an integer mode number m of the periodic box
    (k_m = 2 pi m / L). The complex amplitude of each mode evolves as an
    independent OU process with correlation time ``tau`` and stationary RMS
    ``amplitude``, so the forcing field is

        dE(x, t) = sum_m Re[a_m(t) exp(i k_m x)]

    The realization is precomputed from ``seed`` on a uniform time grid of
    spacing ``dt_update`` (default tau/10) and linearly interpolated in time,
    so a given seed is reproducible and independent of the solver timestep.
    """

    modes: list[int] = [1]
    amplitude: float
    tau: float
    seed: int = 42
    dt_update: float | None = None

    @field_validator("modes")
    @classmethod
    def _positive_modes(cls, v: list[int]) -> list[int]:
        if not v or any(m < 1 for m in v):
            raise ValueError(f"stochastic driver modes must be positive integers, got {v}")
        return v

    @field_validator("tau")
    @classmethod
    def _positive_tau(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"stochastic driver correlation time tau must be > 0, got {v}")
        return v


class EMDriverSetConfig(BaseModel):
    """Longitudinal and transverse electromagnetic driver collections."""

    ex: dict[str, EMDriverConfig]
    ey: dict[str, EMDriverConfig]
    ex_stochastic: StochasticDriverConfig | None = None


class HouLiFilterConfig(BaseModel):
    """Configuration for optional Hou-Li spectral filtering (configuration space only)."""

    is_on: bool
    alpha: float = 36.0
    order: int = 36
    dimensions: list[str] = ["x"]

    @field_validator("dimensions")
    @classmethod
    def _no_velocity_space(cls, v: list[str]) -> list[str]:
        bad = [d for d in v if d != "x"]
        if bad:
            raise ValueError(
                "velocity-space Hou-Li filtering has been removed: the FFT filter is "
                f"periodic in v and corrupts f(v). Only dimensions=['x'] is allowed; got {v}."
            )
        return v


class FokkerPlanckConfig(BaseModel):
    """Configuration for optional Fokker-Planck collisions."""

    is_on: bool
    type: str
    time: EnvelopeConfig
    space: EnvelopeConfig
    # Super-Gaussian exponent of the operator's equilibrium (only used by
    # type: super_gaussian; m=2 is Maxwellian)
    m: float = Field(default=2.0, ge=1.0)


class KrookConfig(BaseModel):
    """Configuration for optional Krook relaxation."""

    is_on: bool
    time: EnvelopeConfig
    space: EnvelopeConfig


class SpeciesConfig(BaseModel):
    """Configuration for a physical species in multi-species simulations."""

    name: str
    charge: float
    mass: float
    vmax: float
    vmin: float | None = None
    nv: int
    density_components: list[str]


class BoltzmannElectronsConfig(BaseModel):
    """Linearized Boltzmann electron closure parameters (``field: poisson-boltzmann``).

    The closure is e phi / T_e = delta n_i / n_0 + lambda_De^2 d^2/dx^2 (e phi / T_e),
    i.e. a screened Poisson solve where the electrons respond adiabatically.
    """

    # Electron temperature in code units (units of the normalizing temperature)
    Te: float = 1.0
    # Screening length in code units. None -> the consistent sqrt(Te / rho_0);
    # 0.0 -> quasineutral closure e phi / T_e = delta n / n_0
    lambda_De: float | None = None


class TermsConfig(BaseModel):
    """Numerical term selections and optional physics operators."""

    field: str
    edfdv: str
    time: str
    species: list[SpeciesConfig] | None = None
    boltzmann_electrons: BoltzmannElectronsConfig | None = None
    fokker_planck: FokkerPlanckConfig
    krook: KrookConfig
    hou_li_filter: HouLiFilterConfig = HouLiFilterConfig(is_on=False)
    diags: bool | None = None

    @model_validator(mode="after")
    def check_boltzmann_electrons(self) -> "TermsConfig":
        """Require an explicit, ion-only species list for the Boltzmann-electron field solver."""
        if self.field == "poisson-boltzmann":
            if not self.species:
                raise ValueError(
                    "field: poisson-boltzmann requires an explicit terms.species list "
                    "(the legacy single-electron default is incompatible with a Boltzmann electron closure)"
                )
            negative = [s.name for s in self.species if s.charge < 0]
            if negative:
                raise ValueError(
                    "field: poisson-boltzmann treats electrons with an adiabatic (Boltzmann) closure; "
                    f"all kinetic species must have positive charge but got negative charge for {negative}"
                )
        return self


class MLFlowConfig(BaseModel):
    """MLflow experiment and run naming configuration."""

    experiment: str
    run: str


class DiagnosticsConfig(BaseModel):
    """Optional diagnostic distribution-function save toggles."""

    model_config = ConfigDict(populate_by_name=True)

    diag_vlasov_dfdt: bool = Field(default=False, alias="diag-vlasov-dfdt")
    diag_fp_dfdt: bool = Field(default=False, alias="diag-fp-dfdt")


class IAWDiagnosticsConfig(BaseModel):
    """Options for the IAW-turbulence module's nk save stream and spectrum plots.

    Only used by ``solver: vlasov-1d-iaw`` (see adept/_vlasov1d/iaw.py).
    """

    # Number of box modes in the saved charge-density spectrum (clipped to nx // 2)
    nk_modes: int = 1024
    # Number of time samples of the nk stream
    nk_nt: int = 2001
    # Averaging window for the late-time spectrum plot, as fractions of tmax
    spectrum_window: tuple[float, float] = (0.5, 1.0)

    @field_validator("spectrum_window")
    @classmethod
    def _valid_window(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not (0.0 <= v[0] < v[1] <= 1.0):
            raise ValueError(f"spectrum_window must satisfy 0 <= t0 < t1 <= 1, got {v}")
        return v


class Vlasov1DConfig(BaseModel):
    """Validated top-level configuration for a Vlasov-1D ADEPT run."""

    units: UnitsConfig
    density: DensityConfig
    grid: GridConfig
    save: SaveConfig
    solver: str
    mlflow: MLFlowConfig
    drivers: EMDriverSetConfig
    terms: TermsConfig
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    iaw_diagnostics: IAWDiagnosticsConfig | None = None
