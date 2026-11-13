"""Solver inputs that vary linearly across an EPW step (LPSE ``interpolateSourcesInTime``).

LPSE advances the ion-acoustic waves, then the Langmuir waves, then the light
(``ZakharovSolver::evolve``). Each solver reads the others' fields linearly interpolated in time
between their old and new values (``zs->linearInterp``): the light at the middle of each light
step (``LightSolver::computeDynamicE0`` builds its scattering potential and sources at
``time + dt / 2``: the EPW potential ``rho`` and the ion density ``Nelf``), the Langmuir wave at
the middle of its step (``densityUpdateOfE_fft``: ``Nelf``). With ``interpolateSourcesInTime``
off a solver reads the new value.

``Linear`` holds the two ends of such an interval and the fractions of it at the start and the
end of the current EPW step (an IAW step spans ``stride`` EPW steps); ``at(s)`` is the value at
the fraction ``s`` of the EPW step and ``substep(i, n)`` the value at the middle of light sub-step
``i`` of ``n``. An array (or ``None``) passed where a ``Linear`` is expected is constant in time.
"""

from collections.abc import Callable

from jax import Array


class Linear:
    """``old + f (new - old)``, ``f`` running from ``f_start`` to ``f_end`` over the EPW step."""

    def __init__(self, old, new=None, f_start=0.0, f_end=1.0, interpolate: bool = True) -> None:
        self.old = old
        self.new = old if new is None else new
        self.constant = old is None or new is None or new is old
        self.f_start, self.f_end = f_start, f_end
        self.interpolate = interpolate

    def at(self, s):
        """The value at the fraction ``s`` of the EPW step (the new value without interpolation)."""
        if self.constant:
            return self.new
        if not self.interpolate:
            return self.new
        f = self.f_start + s * (self.f_end - self.f_start)
        return self.old + f * (self.new - self.old)

    def substep(self, i, n_sub: int):
        """The value at the middle of light sub-step ``i`` of ``n_sub`` (LPSE ``time + dt / 2``)."""
        return self.at((i + 0.5) / n_sub)

    def map(self, fn: Callable[[Array], Array]) -> "Linear":
        """Apply a *linear* map to both ends (e.g. the Laplacian of the potential)."""
        if self.old is None:
            return self
        if self.constant:
            return Linear(fn(self.new))
        return Linear(fn(self.old), fn(self.new), self.f_start, self.f_end, self.interpolate)


def as_linear(value) -> Linear:
    return value if isinstance(value, Linear) else Linear(value)
