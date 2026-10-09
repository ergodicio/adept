"""The pump's temporal pulse shape, LPSE ``laser.pulseShape`` (``LightSolver::pulseShape``).

LPSE's shape is a *power* factor. The injectors multiply the launched field by its square root
(``SchrodingerSolver3::addInjectorSources``: "Take square root because this multiplies the
fields, but pulseShape is for the power"), for analytic beams and injector files alike, and a
static pump's field is rescaled by the square root of the shape, floored at ``1e-12``
(``LightSolver::applyPulseShapeStatic``). The three shapes:

- ``file``: a table of ``(t_ps, scale)``, linearly interpolated and held at its end values;
- ``square``: ``1 / duty_cycle`` for the first ``duty_cycle`` of every ``period``, else 0 (unit
  time average);
- ``sin``: ``2 sin^2(pi t / period)`` (unit time average).
"""

import numpy as np
from jax import numpy as jnp

# LightSolver::applyPulseShapeStatic: the static pump's power factor never drops below this
STATIC_FLOOR = 1.0e-12


def load_pulse_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an LPSE pulse-shape file (``LightSolver::loadPulseShapingData``): two columns
    ``t_ps scale``, ``#`` and ``//`` comment lines, times non-negative and strictly increasing,
    scales in ``[0, 10]``."""
    table = np.loadtxt(path, dtype=np.float64, comments=("#", "//"), ndmin=2)
    if table.shape[1] != 2:
        raise ValueError(f"pulse file {path}: needs two columns (t_ps, power scale), got {table.shape[1]}")
    t, scale = table[:, 0], table[:, 1]
    if t.size == 0:
        raise ValueError(f"pulse file {path} holds no data")
    if np.any(t < 0.0) or np.any(np.diff(t) <= 0.0):
        raise ValueError(f"pulse file {path}: times must be non-negative and strictly increasing (LPSE)")
    if np.any(scale < 0.0) or np.any(scale > 10.0):
        raise ValueError(f"pulse file {path}: power scale factors must lie in [0, 10] (LPSE)")
    return t, scale


class PulseShape:
    """The pulse shape of one light driver, from its ``derived`` block (``pulse_shape`` and, per
    shape, ``pulse_t`` / ``pulse_power`` or ``pulse_period`` / ``pulse_duty_cycle``, times in ps).
    Inactive (every factor 1) when ``pulse_shape`` is absent."""

    def __init__(self, derived: dict) -> None:
        self.shape = derived.get("pulse_shape")
        if self.shape == "file":
            self.t = jnp.asarray(derived["pulse_t"])
            self.table = jnp.asarray(derived["pulse_power"])
        elif self.shape in ("square", "sin"):
            self.period = float(derived["pulse_period"])
            self.duty_cycle = float(derived.get("pulse_duty_cycle", 0.5))
        elif self.shape is not None:
            raise ValueError(f"pulse shape must be file, square or sin (LPSE), got {self.shape!r}")

    @property
    def active(self) -> bool:
        return self.shape is not None

    def power(self, t_ps):
        """LPSE ``LightSolver::pulseShape``: the power factor at time ``t_ps``."""
        if self.shape == "file":
            return jnp.interp(t_ps, self.t, self.table)
        if self.shape == "square":
            on = jnp.mod(t_ps, self.period) <= self.duty_cycle * self.period
            return jnp.where(on, 1.0 / self.duty_cycle, 0.0)
        if self.shape == "sin":
            return 2.0 * jnp.sin(jnp.pi * t_ps / self.period) ** 2
        return 1.0

    def field_factor(self, t_ps):
        """``sqrt(power)``, the injectors' field multiplier; exactly zero where the power is, with
        a finite gradient there."""
        if not self.active:
            return 1.0
        p = self.power(t_ps)
        positive = p > 0.0
        return jnp.where(positive, jnp.sqrt(jnp.where(positive, p, 1.0)), 0.0)

    def static_field_factor(self, t_ps):
        """``sqrt(max(power, 1e-12))``, the static pump's field multiplier."""
        if not self.active:
            return 1.0
        return jnp.sqrt(jnp.maximum(self.power(t_ps), STATIC_FLOOR))
