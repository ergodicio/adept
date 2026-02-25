import equinox as eqx
import jax
import jax.numpy as jnp

from adept.normalization import UREG, PlasmaNormalization


class EnvelopeFunction(eqx.Module):
    """A 1D tanh-based envelope function.

    Evaluates: baseline + bump_height * envelope(x)
    where envelope is a smooth tanh step from 0 to 1 (or 1 to 0 for trough).

    Parameters:
        center: The midpoint of the envelope region.
        width: The full width of the "on" region. The envelope transitions from
               0 to 1 at (center - width/2) and back to 0 at (center + width/2).
        rise: Controls the smoothness of the tanh transitions. Smaller values
              give sharper edges; larger values give more gradual transitions.
        baseline: The minimum value of the envelope (when envelope=0).
        bump_height: The amplitude added to baseline when the envelope is active.
                     Final value ranges from baseline to (baseline + bump_height).
        is_trough: If True, inverts the envelope (1 - envelope), creating a
                   trough (dip) instead of a bump (peak).
    """

    center: float
    width: float
    rise: float
    baseline: float
    bump_height: float
    is_trough: bool

    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the envelope at position(s) x."""
        left = self.center - self.width * 0.5
        right = self.center + self.width * 0.5
        # Inline tanh envelope: 0.5 * (tanh((x - left) / rise) - tanh((x - right) / rise))
        env = 0.5 * (jnp.tanh((x - left) / self.rise) - jnp.tanh((x - right) / self.rise))
        if self.is_trough:
            env = 1 - env
        return self.baseline + self.bump_height * env

    @staticmethod
    def from_config(cfg: dict) -> "EnvelopeFunction":
        """Construct an EnvelopeFunction from a config dict.

        Args:
            cfg: Dict containing center, width, rise, and optionally
                 baseline (default 0.0), bump_height (default 1.0), bump_or_trough

        Returns:
            EnvelopeFunction instance
        """
        return EnvelopeFunction(
            center=cfg["center"],
            width=cfg["width"],
            rise=cfg["rise"],
            baseline=cfg.get("baseline", 0.0),
            bump_height=cfg.get("bump_height", 1.0),
            is_trough=(cfg.get("bump_or_trough", "bump") == "trough"),
        )


class SpaceTimeEnvelopeFunction(eqx.Module):
    """A space-time envelope composed of separate time and space envelopes.

    Evaluates: time_envelope(t) * space_envelope(x)
    """

    time_envelope: EnvelopeFunction
    space_envelope: EnvelopeFunction

    def __call__(self, x: jax.Array, t: float) -> jax.Array:
        """Evaluate the envelope at positions x and time t.

        Returns an array of shape (nx,) representing the spatial profile
        at the given time.
        """
        return self.time_envelope(t) * self.space_envelope(x)

    @staticmethod
    def from_config(term_config: dict) -> "SpaceTimeEnvelopeFunction":
        """Construct a SpaceTimeEnvelopeFunction from a fokker_planck or krook config dict.

        Args:
            term_config: Dict with 'time' and 'space' sub-dicts, each containing
                         center, width, rise, baseline, bump_height, bump_or_trough

        Returns:
            SpaceTimeEnvelopeFunction ready to be called with (x, t)
        """
        time_env = EnvelopeFunction.from_config(term_config["time"])
        space_env = EnvelopeFunction.from_config(term_config["space"])
        return SpaceTimeEnvelopeFunction(time_envelope=time_env, space_envelope=space_env)


class UniformFunction(eqx.Module):
    """Uniform density profile: n(x) = value"""

    value: float = 1.0

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.value * jnp.ones_like(x)


class LinearFunction(eqx.Module):
    """Linear density profile: n(x) = val_at_center + (x - center) / L"""

    center: float
    gradient_scale_length: float  # L in simulation units
    val_at_center: float

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.val_at_center + (x - self.center) / self.gradient_scale_length

    @staticmethod
    def from_physical_units_cfg(cfg: dict, norm: PlasmaNormalization) -> "LinearFunction":
        L = UREG.Quantity(cfg["gradient scale length"]) / norm.L0
        return LinearFunction(
            center=cfg["center"],
            gradient_scale_length=L.to("").magnitude,
            val_at_center=cfg["val at center"],
        )


class ExponentialFunction(eqx.Module):
    """Exponential density profile: n(x) = val_at_center * exp((x - center) / L)"""

    center: float
    gradient_scale_length: float
    val_at_center: float

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.val_at_center * jnp.exp((x - self.center) / self.gradient_scale_length)

    @staticmethod
    def from_physical_units_cfg(cfg: dict, norm: PlasmaNormalization) -> "ExponentialFunction":
        L = UREG.Quantity(cfg["gradient scale length"]) / norm.L0
        return ExponentialFunction(
            center=cfg["center"],
            gradient_scale_length=L.to("").magnitude,
            val_at_center=cfg["val at center"],
        )


class SineFunction(eqx.Module):
    """Sinusoidal density profile: n(x) = baseline * (1 + amplitude * sin(k * x))"""

    baseline: float
    amplitude: float
    wavenumber: float

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.baseline * (1.0 + self.amplitude * jnp.sin(self.wavenumber * x))

    @staticmethod
    def from_config(cfg: dict) -> "SineFunction":
        return SineFunction(
            baseline=cfg["baseline"],
            amplitude=cfg["amplitude"],
            wavenumber=cfg["wavenumber"],
        )


class NoiseProfile(eqx.Module):
    """Noise profile for density perturbations."""

    noise_type: str  # "uniform", "gaussian", or "none"
    noise_val: float
    noise_seed: int

    def __call__(self, shape: tuple) -> jax.Array:
        """Returns a noise profile centered at 0."""
        key = jax.random.PRNGKey(self.noise_seed)
        # noise_val=0 will naturally produce zero noise via multiplication
        if self.noise_type.casefold() == "uniform":
            return jax.random.uniform(key, shape, minval=-self.noise_val, maxval=self.noise_val)
        elif self.noise_type.casefold() == "gaussian":
            return jax.random.normal(key, shape) * self.noise_val
        return jnp.zeros(shape)

    @staticmethod
    def from_config(cfg: dict) -> "NoiseProfile":
        return NoiseProfile(
            noise_type=cfg.get("noise_type", "uniform"),
            noise_val=cfg.get("noise_val", 0.0),
            noise_seed=int(cfg.get("noise_seed", 42)),
        )
