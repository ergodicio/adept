from dataclasses import dataclass

import jax.numpy as jnp
import jpu

UREG = jpu.UnitRegistry()


@dataclass
class PlasmaNormalization:
    """
    A plasma normalization is nothing more than a collection of dimensional
    quantities. Each dimensional quantity indicates what "one" simulation quantity
    refers to. E.g. a distance of "1" in simulation means a physical length of L0.
    """

    # Unit mass
    m0: UREG.Quantity

    # Unit charge
    q0: UREG.Quantity

    # Reference volumetric number density [particles/m^3]
    n0: UREG.Quantity
    # Reference temperature [eV]
    T0: UREG.Quantity
    # Reference distance [m]
    L0: UREG.Quantity
    # Reference velocity [m/s]
    v0: UREG.Quantity
    # Reference time [s]
    tau: UREG.Quantity

    def logLambda_ee(self) -> float:
        n0_cc = self.n0.to("1/cc").magnitude
        T0_eV = self.T0.to("eV").magnitude
        logLambda_ee = 23.5 - jnp.log(n0_cc**0.5 / T0_eV**-1.25)
        logLambda_ee -= (1e-5 + (jnp.log(T0_eV) - 2) ** 2.0 / 16) ** 0.5
        return logLambda_ee

    def approximate_ee_collision_frequency(self) -> UREG.Quantity:
        n0_cc = self.n0.to("1/cc").magnitude
        T0_eV = self.T0.to("eV").magnitude
        logLambda_ee = self.logLambda_ee()
        nu_ee = UREG.Quantity(2.91e-6 * n0_cc * logLambda_ee / T0_eV**1.5, "Hz")
        return nu_ee

    def speed_of_light_norm(self) -> float:
        return (UREG.c / self.v0).to("").magnitude


def normalize(s: float | int | str, norm: PlasmaNormalization | None = None, dim: str = "x") -> float:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)

    if norm is None:
        raise ValueError(f"No PlasmaNormalization was supplied to normalize quantity `{s}`")

    if dim == "x":
        return (UREG.Quantity(s) / norm.L0).to("").magnitude
    elif dim == "t":
        return (UREG.Quantity(s) / norm.tau).to("").magnitude
    elif dim == "v":
        return (UREG.Quantity(s) / norm.v0).to("").magnitude
    elif dim == "temp":
        return (UREG.Quantity(s) / norm.T0).to("").magnitude
    elif dim == "k":
        return (UREG.Quantity(s) * norm.L0).to("").magnitude
    else:
        raise ValueError(f"Don't know how to normalize dimensional quantities for dimension {dim}")


def electron_debye_normalization(n0_str, T0_str):
    """
    Returns the electron thermal normalization for the given density and temperature.
    Unit quantities are:
        - Debye length
        - Electron thermal velocity
        - Langmuir oscillation frequency
    """
    n0 = UREG.Quantity(n0_str)
    T0 = UREG.Quantity(T0_str)

    wp0 = ((n0 * UREG.e**2.0 / (UREG.m_e * UREG.epsilon_0)) ** 0.5).to("rad/s")
    tau = 1 / wp0

    v0 = ((2.0 * T0 / UREG.m_e) ** 0.5).to("m/s")
    x0 = (v0 / wp0).to("nm")

    return PlasmaNormalization(m0=UREG.m_e, q0=UREG.e, n0=n0, T0=T0, L0=x0, v0=v0, tau=tau)


def skin_depth_normalization(n0_str, T0_str):
    """
    Returns the VFP-1D normalization.
    Unit quantities are:
        - c/wp0 (collisionless skin depth)
        - Electron thermal velocity
        - 1/wp0
    """
    n0 = UREG.Quantity(n0_str)
    T0 = UREG.Quantity(T0_str)

    wp0 = ((n0 * UREG.e**2.0 / (UREG.m_e * UREG.epsilon_0)) ** 0.5).to("rad/s")
    tau = 1 / wp0

    v0 = ((2.0 * T0 / UREG.m_e) ** 0.5).to("m/s")
    x0 = (UREG.c / wp0).to("nm")

    return PlasmaNormalization(m0=UREG.m_e, q0=UREG.e, n0=n0, T0=T0, L0=x0, v0=v0, tau=tau)
