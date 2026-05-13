"""Unit tests for EnvelopeFunction and SpaceTimeEnvelopeFunction."""

import jax.numpy as jnp

from adept.functions import EnvelopeConfig, EnvelopeFunction, SpaceTimeEnvelopeConfig, SpaceTimeEnvelopeFunction


class TestEnvelopeFunction:
    """Tests for the EnvelopeFunction class."""

    def test_bump_envelope(self):
        """Test that bump envelope has high value at center, baseline at edges."""
        env = EnvelopeFunction(
            center=50.0,
            width=20.0,
            rise=5.0,
            baseline=0.1,
            bump_height=0.9,
            is_trough=False,
        )
        x = jnp.linspace(0, 100, 101)
        result = env(x)

        # At center, envelope should be close to baseline + bump_height = 1.0
        assert abs(float(result[50]) - 1.0) < 0.05
        # Far from center, envelope should be close to baseline
        assert abs(float(result[0]) - 0.1) < 0.01
        assert abs(float(result[100]) - 0.1) < 0.01

    def test_trough_envelope(self):
        """Test that trough envelope has low value at center, max at edges."""
        env = EnvelopeFunction(
            center=50.0,
            width=20.0,
            rise=5.0,
            baseline=0.1,
            bump_height=0.9,
            is_trough=True,
        )
        x = jnp.linspace(0, 100, 101)
        result = env(x)

        # At center, trough envelope should be close to baseline
        assert abs(float(result[50]) - 0.1) < 0.05
        # Far from center, trough envelope should be baseline + bump_height = 1.0
        assert abs(float(result[0]) - 1.0) < 0.01
        assert abs(float(result[100]) - 1.0) < 0.01

    def test_from_config(self):
        """Test constructing EnvelopeFunction from EnvelopeConfig."""
        cfg = EnvelopeConfig(
            center=50.0,
            width=20.0,
            rise=5.0,
            baseline=0.1,
            bump_height=0.9,
            bump_or_trough="bump",
        )
        env_bump = EnvelopeFunction.from_config(cfg)
        env_trough = EnvelopeFunction.from_config(cfg.model_copy(update={"bump_or_trough": "trough"}))

        assert env_bump.center == 50.0
        assert env_bump.width == 20.0
        assert env_bump.rise == 5.0
        assert env_bump.baseline == 0.1
        assert env_bump.bump_height == 0.9
        assert env_bump.is_trough is False
        assert env_trough.is_trough is True

    def test_scalar_input(self):
        """Test that EnvelopeFunction works with scalar input."""
        env = EnvelopeFunction(
            center=50.0,
            width=20.0,
            rise=5.0,
            baseline=0.1,
            bump_height=0.9,
            is_trough=False,
        )
        result = env(50.0)
        assert abs(float(result) - 1.0) < 0.05


class TestSpaceTimeEnvelopeFunction:
    """Tests for the SpaceTimeEnvelopeFunction class."""

    def test_from_config_and_factorization(self):
        """Test from_config and that result = time_envelope(t) * space_envelope(x)."""
        time_cfg = EnvelopeConfig(
            center=10.0,
            width=5.0,
            rise=1.0,
            baseline=0.5,
            bump_height=0.5,
            bump_or_trough="bump",
        )
        space_cfg = EnvelopeConfig(
            center=50.0,
            width=20.0,
            rise=5.0,
            baseline=0.1,
            bump_height=0.9,
            bump_or_trough="bump",
        )
        config = SpaceTimeEnvelopeConfig(time=time_cfg, space=space_cfg)
        st_env = SpaceTimeEnvelopeFunction.from_config(config)
        time_env = EnvelopeFunction.from_config(time_cfg)
        space_env = EnvelopeFunction.from_config(space_cfg)

        x = jnp.linspace(0, 100, 101)
        t = 10.0

        result = st_env(x, t)
        expected = time_env(t) * space_env(x)

        assert jnp.allclose(result, expected)
