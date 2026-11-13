"""LPSE's Kubo-Anderson (KAP) bandwidth on the pump beams.

``LightSolver::computeKapTransitionTime`` / ``updateInjectorPhases``: every beam -- each its own
group unless ``laser.N.group`` says otherwise -- holds a uniform random phase for a dwell time
``2 X / dW`` with ``X ~ Exp(1)`` and ``dW = bandwidth * w0`` (``KAP.frequency`` is ``dW / W0``), then
jumps to a new uniform phase. The mean dwell is ``2 / dW``. The transition times and phases are
drawn once, from ``kap_seed``, for the length of the run: each beam draws its dwell times and its
phases from its own streams (``SeedSequence(kap_seed).spawn``), so a beam's history depends on the
seed and the beam index only and a longer run only appends to it -- a run resumed from a
checkpoint with a larger ``tmax`` sees the same pump phases as the unbroken run.
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
        n = int(np.ceil(3.0 * float(t_end) * rate / 2.0)) + 64  # 3x the expected number of jumps
        times, phases = [], []
        # one dwell stream and one phase stream per beam: draw k of either depends on (seed, beam, k)
        # only, so the table is prefix-stable as n (the horizon) grows
        for beam_seq in np.random.SeedSequence(int(seed)).spawn(n_beams):
            dwell_seq, phase_seq = beam_seq.spawn(2)
            dwell = 2.0 * np.random.default_rng(dwell_seq).exponential(1.0 / rate, size=n)  # 2 Exp(dW), LPSE
            times.append(np.cumsum(dwell))
            phases.append(2.0 * np.pi * np.random.default_rng(phase_seq).uniform(size=n + 1))
        times = np.stack(times)
        if np.any(times[:, -1] < t_end):
            raise RuntimeError("KapPhases: the drawn transition table does not cover the run")
        self.times = jnp.asarray(times)
        self.phases = jnp.asarray(np.stack(phases))

    def phase(self, t, beam: int):
        if not self.active:
            return 0.0
        return self.phases[beam, jnp.searchsorted(self.times[beam], t, side="right")]
