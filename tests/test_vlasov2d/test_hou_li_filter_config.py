"""Tests that vlasov-2d Hou-Li filtering stays confined to configuration space.

The Hou-Li filter is FFT-based, hence periodic in whichever axis it filters. x and y are
periodic so filtering there is correct; vx/vy are bounded domains (f -> 0 at the velocity
edges), so an FFT filter in velocity wraps the forward tail onto the opposite edge and
corrupts f(v). Velocity-space filtering was removed outright, and the config validator is
what keeps it from being reintroduced from a deck. These tests guard that validator.
"""

import pytest
from pydantic import ValidationError

from adept._vlasov2d.datamodel import HouLiFilterConfig


@pytest.mark.parametrize(
    "dimensions",
    [["vx"], ["vy"], ["vx", "vy"], ["x", "y", "vx", "vy"], ["x", "vx"]],
)
def test_hou_li_filter_rejects_velocity_dimensions(dimensions: list[str]):
    """Velocity-space Hou-Li filtering must be rejected at config validation."""
    with pytest.raises(ValidationError, match="velocity-space Hou-Li filtering has been removed"):
        HouLiFilterConfig(is_on=True, dimensions=dimensions)


@pytest.mark.parametrize("dimensions", [["x"], ["y"], ["x", "y"]])
def test_hou_li_filter_accepts_configuration_space(dimensions: list[str]):
    """Filtering in x and/or y is the supported case and must still validate."""
    assert HouLiFilterConfig(is_on=True, dimensions=dimensions).dimensions == dimensions


def test_hou_li_filter_defaults_to_configuration_space():
    """The default must be spatial-only so an unspecified deck cannot filter in velocity."""
    assert HouLiFilterConfig().dimensions == ["x", "y"]
