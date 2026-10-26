"""Entropy diagnostics use the continuous value 0 log(0) = 0."""

import jax.numpy as jnp
import numpy as np

from adept._vlasov1d.observations import FieldsObservation, ScalarsObservation


def test_entropy_with_empty_velocity_cells():
    grids = {"electron": {"v": jnp.array([-1.0, 0.0, 1.0]), "dv": 0.5}}
    state = {"electron": jnp.tile(jnp.array([0.0, 1.0, np.exp(-1.0)]), (4, 1))}
    state.update({name: jnp.zeros(6 if name in {"a", "prev_a"} else 4) for name in ["e", "de", "a", "prev_a"]})
    expected = 0.5 * np.exp(-1.0)
    fields = FieldsObservation(grids, 1.0)(0.0, state, {})
    scalars = ScalarsObservation(grids, {"electron": {"mass": 1.0}}, 1.0)(0.0, state, {})
    np.testing.assert_allclose(fields["electron"]["-flogf"], expected)
    np.testing.assert_allclose(scalars["mean_-flogf_electron"], expected)
