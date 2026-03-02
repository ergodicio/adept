import equinox as eqx
import jax


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
        import jax.numpy as jnp

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
            cfg: Dict containing center, width, rise, baseline, bump_height, bump_or_trough

        Returns:
            EnvelopeFunction instance
        """
        return EnvelopeFunction(
            center=cfg["center"],
            width=cfg["width"],
            rise=cfg["rise"],
            baseline=cfg["baseline"],
            bump_height=cfg["bump_height"],
            is_trough=(cfg["bump_or_trough"] == "trough"),
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
