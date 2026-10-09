"""LPSE's Kubo-Anderson (KAP) bandwidth on the pump beams.

``LightSolver::computeKapTransitionTime`` / ``updateInjectorPhases``: every beam -- each its own
group unless ``laser.N.group`` says otherwise -- holds a uniform random phase for a dwell time
``2 X / dW`` with ``X ~ Exp(1)`` and ``dW = bandwidth * w0`` (``KAP.frequency`` is ``dW / W0``), then
jumps to a new uniform phase. The mean dwell is ``2 / dW``. The transition times and phases are
drawn once, from ``kap_seed``, for the length of the run.
"""

import numpy as np
from jax import numpy as jnp


class KapPhases:
    """The KAP phase of each beam as a function of time (0 everywhere without bandwidth)."""

    def __init__(self, bandwidth: float, w0: float, n_beams: int, t_end: float, seed: int = 0) -> None:
        self.active = bandwidth > 0.0
        if not self.active:
            return
        rate = float(bandwidth) * float(w0)  # dW, rad / ps
        rng = np.random.default_rng(int(seed))
        n = int(np.ceil(3.0 * float(t_end) * rate / 2.0)) + 64  # 3x the expected number of jumps
        dwell = 2.0 * rng.exponential(1.0 / rate, size=(n_beams, n))  # 2 Exp(dW), LPSE
        times = np.cumsum(dwell, axis=1)
        if np.any(times[:, -1] < t_end):
            raise RuntimeError("KapPhases: the drawn transition table does not cover the run")
        self.times = jnp.asarray(times)
        self.phases = jnp.asarray(2.0 * np.pi * rng.uniform(size=(n_beams, n + 1)))

    def phase(self, t, beam: int):
        if not self.active:
            return 0.0
        return self.phases[beam, jnp.searchsorted(self.times[beam], t, side="right")]
