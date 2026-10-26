"""Validation gates for the cylindrical Landau operator (PRL-plan D2, rung A).

The decisive test is the projection theorem: acting on the wave's separable
dipole deviation, the marginal response of the full 2V operator — and of
each Maxwellian-preserving channel alone — must match the 1D effective
operator d/dv[ Dbar_c(v) (dF/dv + v F) ] with

    Dbar_c(v_par) = sum_j [a_s Dhat_par s_par^2/s^2 + a_L Dhat_perp s_perp^2/s^2]
                    M_perp(j) w_perp(j),

the same Dbar the campaign's theory (derivation_3v_sympy.py, PRL plan
Sec 2b) uses for its predictions — including the ~47/53 Lorentz/speed
attribution split at the EPW resonance.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from adept._vlasov1d2v.solvers.pushers.fokker_planck import Collisions, CylindricalLandau

NV, NVPERP, VMAX = 512, 64, 6.4
DV = 2 * VMAX / NV
DVPERP = VMAX / NVPERP
V = np.linspace(-VMAX + DV / 2, VMAX - DV / 2, NV)
VPERP = np.linspace(DVPERP / 2, VMAX - DVPERP / 2, NVPERP)
WPERP = 2 * np.pi * VPERP * DVPERP


def make_collisions(alpha_speed=1.0, alpha_lorentz=1.0, restore=True, substeps=1):
    cfg = {
        "grid": {
            "species_grids": {
                "electron": {
                    "v": jnp.array(V),
                    "dv": DV,
                    "vperp": jnp.array(VPERP),
                    "dvperp": DVPERP,
                    "wperp": jnp.array(WPERP),
                }
            }
        },
        "terms": {
            "fokker_planck": {
                "is_on": True,
                "type": "cylindrical_landau",
                "channels": {"speed": alpha_speed, "lorentz": alpha_lorentz},
                "moment_restoration": restore,
                "explicit_substeps": substeps,
            },
            "krook": {"is_on": False},
        },
    }
    return Collisions(cfg)


def maxwellian(u=0.0, T=1.0, n=1.0):
    s2 = (V[:, None] - u) ** 2 + VPERP[None, :] ** 2
    f = np.exp(-s2 / (2 * T))
    f = f / np.sum(f * DV * WPERP[None, :])
    return (n * f)[None, ...]  # (nx=1, nv, nvperp)


def full_moments(f):
    w = DV * WPERP[None, :]
    n = np.sum(f[0] * w)
    j = np.sum(f[0] * V[:, None] * w)
    e = np.sum(f[0] * (V[:, None] ** 2 + VPERP[None, :] ** 2) * w)
    return n, j, e


def freeze_moments(coll, n=1.0, u=0.0, T=1.0):
    """Pin the operator's bulk frame (unit-test mode: no quasi-static back-reaction)."""

    def _frozen(f):
        nx = f.shape[0]
        return (jnp.full((nx,), n), jnp.full((nx,), u), jnp.full((nx,), T))

    coll.cyl._moments = _frozen
    return coll


def test_maxwellian_exact_discrete_fixed_point():
    """With the bulk frame pinned, the sampled Maxwellian is machine-stationary.

    The M-weighted flux form makes g = f/M exactly constant for the sampled
    Maxwellian, so every channel flux vanishes identically at ANY nu dt.
    """
    # nu dt = 0.1 needs ~32 explicit substeps (perp stability nu dt D/dvperp^2 < 1/4)
    coll = freeze_moments(make_collisions(restore=False, substeps=32), n=1.0, u=0.3, T=1.2)
    f0 = maxwellian(u=0.3, T=1.2)
    f = jnp.array(f0)
    for _ in range(10):
        f = coll(jnp.array([0.1]), None, f, 1.0)  # nu dt = 0.1, large on purpose
    drift = np.abs(np.asarray(f) - f0).max() / f0.max()
    assert drift < 1e-12, f"Maxwellian drifted {drift:.2e}"


def test_self_consistent_fixed_point_no_secular_drift():
    """Production config (live moments + restoration) settles: no secular drift.

    Without restoration the live-moment loop chases its own O(dvperp^2)
    quadrature bias (the midpoint rule on the cylindrical v_perp grid biases
    the measured T of the sampled Maxwellian) and T drifts secularly at
    ~5e-5 per collision time. Restoration pins P_par and E exactly each
    step, closing the loop — it is REQUIRED for long-run bulk stability,
    not just conservation bookkeeping.
    """
    coll = make_collisions(restore=True, substeps=4)
    f0 = maxwellian(u=0.3, T=1.2)
    f = jnp.array(f0)
    for _ in range(50):
        f = coll(jnp.array([0.2]), None, f, 1.0)
    f50 = f
    for _ in range(50):
        f = coll(jnp.array([0.2]), None, f, 1.0)
    settle = np.abs(np.asarray(f50) - f0).max() / f0.max()
    late = np.abs(np.asarray(f) - np.asarray(f50)).max() / f0.max()
    assert settle < 5e-4, f"fixed point too far from sampled Maxwellian: {settle:.2e}"
    assert late < 0.15 * settle, f"secular drift: late {late:.2e} vs settle {settle:.2e}"


def _dbar_channel(alpha_s, alpha_l):
    """Projected marginal coefficient Dbar(v) on the discrete v_perp grid."""
    mperp = np.exp(-(VPERP**2) / 2)
    mperp /= np.sum(mperp * WPERP)
    s2 = V[:, None] ** 2 + VPERP[None, :] ** 2
    s = np.sqrt(s2)
    dpar = np.asarray(CylindricalLandau._dhat_par(jnp.array(s)))
    dperp = np.asarray(CylindricalLandau._dhat_perp(jnp.array(s)))
    dtil = alpha_s * dpar * V[:, None] ** 2 / s2 + alpha_l * dperp * VPERP[None, :] ** 2 / s2
    return np.sum(dtil * (mperp * WPERP)[None, :], axis=1)


@pytest.mark.parametrize(
    "alpha_s, alpha_l, label",
    [(1.0, 1.0, "full"), (0.0, 1.0, "lorentz-only"), (1.0, 0.0, "speed-only")],
)
def test_projection_theorem_marginal_response(alpha_s, alpha_l, label):
    """Marginal response to the dipole deviation == d/dv[Dbar (F' + vF)] per channel."""
    coll = make_collisions(alpha_speed=alpha_s, alpha_lorentz=alpha_l, restore=False)

    vphi, sigma, p = 3.86, 0.31, 1e-4
    fM = maxwellian()
    mperp = np.exp(-(VPERP**2) / 2)
    mperp /= np.sum(mperp * WPERP)
    dF = p * (V - vphi) * np.exp(-((V - vphi) ** 2) / (2 * sigma**2))
    f = fM + (dF[:, None] * mperp[None, :])[None, ...]

    nu, dt = 1.0, 1e-3
    # linear response: subtract the operator's action on f_M alone, which
    # carries the sampled Maxwellian's tiny discrete-fixed-point flux
    f_new = np.asarray(coll(jnp.array([nu]), None, jnp.array(f), dt))
    fM_new = np.asarray(coll(jnp.array([nu]), None, jnp.array(fM), dt))
    df = (f_new - f) - (fM_new - fM)
    dFdt_meas = np.sum(df[0] * WPERP[None, :], axis=1) / (nu * dt)

    # theory: d/dv [ Dbar (dF/dv + v F) ], evaluated with the same edge-flux
    # discretization as the operator's implicit sweep
    dbar = _dbar_channel(alpha_s, alpha_l)
    Mv = np.exp(-(V**2) / 2)
    g = dF / Mv
    dbar_e = 0.5 * (dbar[1:] + dbar[:-1])
    M_e = np.sqrt(Mv[1:] * Mv[:-1])
    flux = dbar_e * M_e * np.diff(g) / DV
    dFdt_th = (np.pad(flux, (0, 1)) - np.pad(flux, (1, 0))) / DV

    mask = np.abs(V - vphi) < 4 * sigma
    err = np.linalg.norm(dFdt_meas[mask] - dFdt_th[mask]) / np.linalg.norm(dFdt_th[mask])
    assert err < 0.05, f"{label}: marginal response off theory by {err:.1%}"


def test_channel_attribution_split():
    """Lorentz share of Dbar at the k=0.3 resonance ~ 47% (PRL plan Sec 2b P3)."""
    vphi = 3.86
    i = np.argmin(np.abs(V - vphi))
    d_l = _dbar_channel(0.0, 1.0)[i]
    d_s = _dbar_channel(1.0, 0.0)[i]
    share = d_l / (d_l + d_s)
    assert 0.42 < share < 0.52, f"Lorentz share {share:.1%}"


def test_bump_drift_at_delta_particle_rate():
    """A narrow tail bump's <v_par> drifts at the test-particle Ito rate.

    The Ito rate is defined against a FIXED Maxwellian background, so the
    bulk frame is pinned: with live moments the bump's momentum shifts the
    measured u and the operator's (physical) back-reaction on the bulk
    contaminates the bump-mass-normalized drift, independent of amplitude.
    """
    coll = freeze_moments(make_collisions(restore=False))
    fM = maxwellian()
    v0, sb, amp = 3.0, 0.25, 1e-4
    bump = amp * np.exp(-((V[:, None] - v0) ** 2 + VPERP[None, :] ** 2) / (2 * sb**2))
    f = fM + bump[None, ...]

    nu, dt = 1.0, 5e-4
    # frozen frame => C[f_M] = 0 exactly, so the response is the bump's alone
    f_new = np.asarray(coll(jnp.array([nu]), None, jnp.array(f), dt))
    df = f_new - f
    w = DV * WPERP[None, :]
    dvpar_dt = np.sum(df[0] * V[:, None] * w) / (nu * dt) / np.sum(bump * w)

    # Ito drift of the full tensor: vhat_par [ -D_par s + D_par' + 2 (D_par - D_perp)/s ]
    s = np.sqrt(V[:, None] ** 2 + VPERP[None, :] ** 2)
    dpar = np.asarray(CylindricalLandau._dhat_par(jnp.array(s)))
    dperp = np.asarray(CylindricalLandau._dhat_perp(jnp.array(s)))
    ds = 1e-5
    dpar_p = (
        np.asarray(CylindricalLandau._dhat_par(jnp.array(s + ds)))
        - np.asarray(CylindricalLandau._dhat_par(jnp.array(s - ds)))
    ) / (2 * ds)
    drift_field = (V[:, None] / s) * (-dpar * s + dpar_p + 2 * (dpar - dperp) / s)
    pred = np.sum(bump * drift_field * w) / np.sum(bump * w)

    assert abs(dvpar_dt - pred) / abs(pred) < 0.05, f"bump drift {dvpar_dt:.4e} vs Ito {pred:.4e}"


def test_lorentz_channel_conserves_energy():
    """Pitch-angle scattering moves negligible energy compared to the speed channel."""
    fM = maxwellian()
    mperp = np.exp(-(VPERP**2) / 2)
    mperp /= np.sum(mperp * WPERP)
    dF = 1e-4 * (V - 3.86) * np.exp(-((V - 3.86) ** 2) / (2 * 0.31**2))
    f = jnp.array(fM + (dF[:, None] * mperp[None, :])[None, ...])

    de = {}
    for label, (a_s, a_l) in {"lorentz": (0.0, 1.0), "speed": (1.0, 0.0)}.items():
        coll = make_collisions(alpha_speed=a_s, alpha_lorentz=a_l, restore=False)
        f_new = np.asarray(coll(jnp.array([1.0]), None, f, 1e-3))
        fM_new = np.asarray(coll(jnp.array([1.0]), None, jnp.array(fM), 1e-3))
        df = (f_new - np.asarray(f)) - (fM_new - fM)
        w = DV * WPERP[None, :]
        de[label] = abs(np.sum(df[0] * (V[:, None] ** 2 + VPERP[None, :] ** 2) * w))

    assert de["lorentz"] < 2e-2 * de["speed"], f"Lorentz dE {de['lorentz']:.2e} vs speed {de['speed']:.2e}"


def test_conservation_with_restoration():
    """n, P_par, E drift at solver precision with moment restoration on."""
    coll = make_collisions(restore=True)
    fM = maxwellian()
    bump = 1e-3 * np.exp(-((V[:, None] - 3.0) ** 2 + (VPERP[None, :] - 1.0) ** 2) / (2 * 0.3**2))
    f = jnp.array(fM + bump[None, ...])
    n0, j0, e0 = full_moments(np.asarray(f))

    for _ in range(50):
        f = coll(jnp.array([1.0]), None, f, 2e-3)  # nu t = 0.1 total

    n1, j1, e1 = full_moments(np.asarray(f))
    assert abs(n1 - n0) / n0 < 1e-12, f"dn/n {abs(n1 - n0) / n0:.2e}"
    assert abs(j1 - j0) < 1e-12, f"dj {abs(j1 - j0):.2e}"
    assert abs(e1 - e0) / e0 < 1e-12, f"dE/E {abs(e1 - e0) / e0:.2e}"
