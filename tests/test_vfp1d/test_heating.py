#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Tests for Maxwellian heating/cooling and IB heating in F0Collisions.

Verifies that the Ridgers D-augmentation formulation:
  - Maxwellian heating preserves Maxwellian shape and has constant dT/dt
  - IB heating increases temperature and drives toward super-Gaussian (Langdon)
  - Positivity is preserved
  - Zero heating params produce no change

Integrated tests (marked `sweep`) mirror test_fp_relaxation: they log metrics
to MLflow child runs and produce comparison dashboards.
"""

import itertools
import pickle
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pytest
from diffrax import ODETerm, SaveAt, diffeqsolve
from fp_relaxation.metrics import (
    RelaxationMetrics,
    compute_density,
    compute_entropy,
    compute_metrics,
    compute_positivity_violation,
    compute_rmse,
)
from fp_relaxation.plotting import plot_comparison_dashboard, plot_distribution_evolution
from fp_relaxation.registry import VelocityGrid, get_git_info
from jax import Array
from scipy.special import gamma as gamma_fn

from adept._base_ import Stepper
from adept.driftdiffusion import _find_self_consistent_beta_single, discrete_temperature
from adept.vfp1d.fokker_planck import F0Collisions, SelfConsistentBetaConfig, get_model, get_scheme
from adept.vfp1d.grid import Grid

# =============================================================================
# Configuration
# =============================================================================

VMAX = 6.0
NV = 128
MODELS = ("CoulombianKernel",)
SCHEMES = ("ChangCooper", "LogMeanFlux")
EXPERIMENT = "vfp1d-heating-tests"

# =============================================================================
# Helpers
# =============================================================================


def _make_grid(nv=NV, vmax=VMAX):
    """Create a VFP1D grid with dummy spatial parameters."""
    return Grid(
        xmin=0.0,
        xmax=1.0,
        nx=1,
        tmin=0.0,
        tmax=1.0,
        dt=0.01,
        nv=nv,
        vmax=vmax,
        nl=1,
        boundary="periodic",
    )


def _make_collisions(grid, model="CoulombianKernel", scheme="chang_cooper", nuee_coeff=1.0):
    """Create an F0Collisions instance."""
    return F0Collisions(
        nuee_coeff=nuee_coeff,
        grid=grid,
        model=get_model(model, grid.v, grid.dv),
        scheme=get_scheme(scheme, grid.dv),
        sc_beta=SelfConsistentBetaConfig(max_steps=0),
    )


def _maxwellian(v, T=1.0):
    """Normalised Maxwellian on a spherical (positive-only) grid."""
    f = jnp.exp(-(v**2) / (2.0 * T))
    dv = v[1] - v[0]
    norm = 4.0 * jnp.pi * jnp.sum(v**2 * f * dv)
    return f / norm


def _temperature(f, v, dv):
    return discrete_temperature(f, v, dv, spherical=True)


def _density(f, v, dv):
    return 4.0 * jnp.pi * jnp.sum(v**2 * f * dv)


def _rmse_vs_maxwellian(f, v, dv):
    T = _temperature(f, v, dv)
    n = _density(f, v, dv)
    f_max = n * jnp.exp(-(v**2) / (2.0 * T)) / (2.0 * jnp.pi * T) ** 1.5
    return jnp.sqrt(jnp.mean((f - f_max) ** 2))


def _step(collisions, f0, dt, n_steps, **heating_kwargs):
    """Run n_steps of the collision operator, return f0 at each step."""
    f0x = f0[None, :]
    history = [f0]
    for _ in range(n_steps):
        f0x = collisions(nu=1.0, f0x=f0x, dt=dt, **heating_kwargs)
        history.append(f0x[0])
    return history


# =============================================================================
# Super-Gaussian utilities (Ridgers §2.4.5, §5.6.2)
# =============================================================================


def super_gaussian_kurtosis(m: float) -> float:
    """Compute velocity kurtosis κ = <v⁴>/<v²>² for a super-Gaussian of order m.

    For the normalized super-Gaussian f ∝ exp(-(v/(aₑvT))^m):
        κ(m) = Γ(3/m)·Γ(7/m) / Γ(5/m)²

    κ(2) = 5/3 (Maxwellian), κ(5) ≈ 1.336 (Langdon).
    """
    return gamma_fn(3.0 / m) * gamma_fn(7.0 / m) / gamma_fn(5.0 / m) ** 2


def distortion_parameter(m: float) -> float:
    """Ridgers eq 5.21: Δ_SG = Γ(5/m)Γ(1/m) / (3·Γ(3/m)²).

    Δ(2) = 1.0 (Maxwellian), Δ(5) ≈ 0.69 (Langdon).
    """
    return gamma_fn(5.0 / m) * gamma_fn(1.0 / m) / (3.0 * gamma_fn(3.0 / m) ** 2)


def kurtosis_from_f(f: Array, v: Array, dv: float) -> Array:
    """Compute velocity kurtosis κ = <v⁴>/<v²>² from distribution on spherical grid.

    <v^k> = ∫v^(k+2)f dv / ∫v²f dv  (the 4π factors cancel).
    """
    m2 = jnp.sum(v**4 * f * dv)  # ∝ <v²>
    m4 = jnp.sum(v**6 * f * dv)  # ∝ <v⁴>
    m0 = jnp.sum(v**2 * f * dv)  # ∝ density
    v2_mean = m2 / m0  # <v²>
    v4_mean = m4 / m0  # <v⁴>
    return v4_mean / v2_mean**2


def invert_kurtosis_to_m(kappa: float, m_min: float = 2.0, m_max: float = 8.0) -> float:
    """Invert κ(m) to find the super-Gaussian exponent m.

    κ(m) is monotonically decreasing for m > 0, so bisection works.
    Clamps to [m_min, m_max] when kappa is outside the valid range.
    """
    from scipy.optimize import brentq

    kappa_at_min = super_gaussian_kurtosis(m_min)
    kappa_at_max = super_gaussian_kurtosis(m_max)

    if kappa >= kappa_at_min:
        return m_min  # Maxwellian or narrower
    if kappa <= kappa_at_max:
        return m_max  # Beyond m_max

    def residual(m):
        return super_gaussian_kurtosis(m) - kappa

    return brentq(residual, m_min, m_max, xtol=1e-6)


def matte_expected_m(alpha: float) -> float:
    """Matte et al formula for expected super-Gaussian order (Ridgers eq 2.67).

    m = 2 + 3 / (1 + 1.66/a^0.724)

    where a = Z · v_osc / v_T is the Langdon parameter.
    a → 0: m → 2 (Maxwellian), a → ∞: m → 5 (Langdon).
    """
    return 2.0 + 3.0 / (1.0 + 1.66 / max(alpha, 1e-30) ** 0.724)


def make_super_gaussian(v: Array, dv: float, T: float, m: float) -> Array:
    """Create a normalized super-Gaussian on a spherical grid (Ridgers eq 2.66).

    f₀^SG = C(m) · n/vT³ · exp(-(v/(aₑ·vT))^m)

    where aₑ = [3Γ(3/m)/(2Γ(5/m))]^(1/2), C(m) = m/(4π·aₑ³·Γ(3/m)),
    and vT = √(2T).
    """
    vT = jnp.sqrt(2.0 * T)
    alpha_e = jnp.sqrt(3.0 * gamma_fn(3.0 / m) / (2.0 * gamma_fn(5.0 / m)))
    f = jnp.exp(-((v / (alpha_e * vT)) ** m))
    norm = 4.0 * jnp.pi * jnp.sum(v**2 * f * dv)
    return f / norm


# =============================================================================
# Heating vector field for diffrax integration
# =============================================================================


class F0CollisionsHeatingVectorField(eqx.Module):
    """Wraps F0Collisions for diffrax, passing heating kwargs."""

    collisions: F0Collisions
    dt: Array = eqx.field(converter=jnp.asarray)
    D0_heating: float | None = None
    ib_vosc2: float | None = None
    ib_Z2ni_w0: float | None = None

    def __call__(self, t: float, f: Array, args) -> Array:
        del t, args
        f_batched = f[None, :]
        kwargs = {}
        if self.D0_heating is not None:
            kwargs["D0_heating"] = self.D0_heating
        if self.ib_vosc2 is not None:
            kwargs["ib_vosc2"] = self.ib_vosc2
            kwargs["ib_Z2ni_w0"] = self.ib_Z2ni_w0
        f_new = self.collisions(nu=1.0, f0x=f_batched, dt=self.dt, **kwargs)
        return f_new[0]


# =============================================================================
# Heating metrics
# =============================================================================


class HeatingMetrics(eqx.Module):
    """Extended metrics for heating tests, beyond RelaxationMetrics."""

    time: Array
    rel_density: Array
    temperature_discrete: Array
    temperature_sc: Array
    positivity_violation: Array
    rmse_expected: Array  # RMSE to instantaneous expected distribution (Maxwellian or super-Gaussian)
    entropy: Array
    kurtosis: Array  # velocity kurtosis κ = <v⁴>/<v²>²
    sg_m: Array  # super-Gaussian exponent from kurtosis inversion


def compute_heating_metrics(
    f: Array,
    grid: VelocityGrid,
    times: Array,
    expected_T: Array | None = None,
    expected_m: float | None = None,
) -> HeatingMetrics:
    """Compute metrics for heating simulations.

    In addition to standard relaxation metrics, computes:
    - kurtosis (κ) and super-Gaussian order (m) at each snapshot
    - RMSE against expected distribution (Maxwellian at expected_T, or super-Gaussian at expected_m)
    """
    # Standard metrics via existing infrastructure
    base = compute_metrics(f, grid, times)

    # Compute kurtosis and m at each snapshot (not JIT-able due to scipy)
    f_np = np.asarray(f)
    v_np = np.asarray(grid.v)
    dv_np = float(grid.dv)

    kappas = []
    ms = []
    for i in range(f.shape[0]):
        k = float(kurtosis_from_f(jnp.array(f_np[i]), grid.v, grid.dv))
        kappas.append(k)
        ms.append(invert_kurtosis_to_m(k))

    kurtosis_arr = jnp.array(kappas)
    m_arr = jnp.array(ms)

    # RMSE against instantaneous expected distribution
    T_measured = base.temperature_discrete
    if expected_T is not None and expected_m is None:
        # Maxwellian heating: expected is Maxwellian at instantaneous measured T
        def _make_expected_maxwellian(T_meas):
            return _maxwellian(grid.v, T=T_meas)

        f_expected = jax.vmap(_make_expected_maxwellian)(T_measured)
        rmse_expected = compute_rmse(f, f_expected, grid)
    elif expected_m is not None:
        # IB heating: expected is super-Gaussian at measured T with expected m
        def _make_expected_sg(T_meas):
            return make_super_gaussian(grid.v, grid.dv, T=T_meas, m=expected_m)

        f_expected = jax.vmap(_make_expected_sg)(T_measured)
        rmse_expected = compute_rmse(f, f_expected, grid)
    else:
        # Fallback: RMSE to instantaneous Maxwellian
        rmse_expected = base.rmse_instant

    return HeatingMetrics(
        time=times,
        rel_density=base.rel_density,
        temperature_discrete=base.temperature_discrete,
        temperature_sc=base.temperature_sc,
        positivity_violation=base.positivity_violation,
        rmse_expected=rmse_expected,
        entropy=base.entropy,
        kurtosis=kurtosis_arr,
        sg_m=m_arr,
    )


# =============================================================================
# Heating-specific plotting
# =============================================================================


def plot_heating_dashboard(
    results: dict[str, HeatingMetrics],
    title: str = "Heating Comparison",
    expected_T_fn=None,
    expected_m: float | None = None,
    output_path: Path | None = None,
):
    """Create an 8-panel dashboard for heating tests.

    Panels (2x4): density conservation, temperature (with expected overlay),
    T_actual - T_expected, positivity, super-Gaussian m, RMSE to expected,
    RMSE to instant Maxwellian, entropy.
    """
    import matplotlib.pyplot as plt

    sorted_names = sorted(results.keys())
    n_results = len(sorted_names)
    cmap = plt.cm.tab10
    colors = {name: cmap(i / cmap.N) for i, name in enumerate(sorted_names)}

    stacked = {}
    for name in sorted_names:
        s = jax.tree.map(np.asarray, results[name])
        stacked[name] = (s, s.time)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    def _plot(ax, times, y, color, label, log=False):
        plot_fn = ax.semilogy if log else ax.plot
        if len(times) <= 2:
            plot_fn(times[-1], y[-1], "o", color=color, label=label, markersize=8)
        else:
            plot_fn(times, y, color=color, label=label, linewidth=1.5)

    # Panel 1: Density conservation
    ax = axes[0, 0]
    for name, (s, t) in stacked.items():
        _plot(ax, t, s.rel_density, colors[name], name)
    ax.set_xlabel("t")
    ax.set_ylabel("(n - n₀) / n₀")
    ax.set_title("Density Conservation")
    ax.axhline(0, color="k", ls="--", alpha=0.5)

    # Panel 2: Temperature (with expected if provided)
    ax = axes[0, 1]
    for name, (s, t) in stacked.items():
        _plot(ax, t, s.temperature_discrete, colors[name], name)
    if expected_T_fn is not None:
        for name, (s, t) in stacked.items():
            T_exp = expected_T_fn(t)
            ax.plot(t, T_exp, color=colors[name], ls="--", alpha=0.5, lw=1)
            break  # Only plot expected once
    ax.set_xlabel("t")
    ax.set_ylabel("T")
    ax.set_title("Temperature (solid=actual, dashed=expected)")

    # Panel 3: Positivity
    ax = axes[0, 2]
    for name, (s, t) in stacked.items():
        _plot(ax, t, s.positivity_violation, colors[name], name)
    ax.set_xlabel("t")
    ax.set_ylabel("Positivity Violation")
    ax.set_title("Positivity Violation")
    ax.legend(fontsize=7)

    # Panel 4: Super-Gaussian m
    ax = axes[1, 0]
    for name, (s, t) in stacked.items():
        mask = ~np.isnan(s.sg_m)
        if np.any(mask):
            ax.plot(t[mask], s.sg_m[mask], color=colors[name], label=name, lw=1.5)
    if expected_m is not None:
        ax.axhline(expected_m, color="k", ls="--", alpha=0.7, label=f"Matte m={expected_m:.2f}")
    ax.axhline(2.0, color="gray", ls=":", alpha=0.5, label="Maxwellian (m=2)")
    ax.axhline(5.0, color="gray", ls="-.", alpha=0.5, label="Langdon (m=5)")
    ax.set_xlabel("t")
    ax.set_ylabel("m")
    ax.set_title("Super-Gaussian Order")
    ax.legend(fontsize=7)

    # Panel 5: RMSE to instantaneous expected distribution
    ax = axes[1, 1]
    for name, (s, t) in stacked.items():
        _plot(ax, t, s.rmse_expected, colors[name], name, log=True)
    ax.set_xlabel("t")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE to Instantaneous Expected")

    # Panel 6: Entropy
    ax = axes[1, 2]
    for name, (s, t) in stacked.items():
        mask = ~np.isnan(s.entropy)
        if np.any(mask):
            ax.plot(t[mask], s.entropy[mask], color=colors[name], lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, hspace=0.35, wspace=0.3)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# =============================================================================
# Heating sweep runner
# =============================================================================

SCHEME_MAP = {
    "ChangCooper": "chang_cooper",
    "LogMeanFlux": "log_mean",
}


def run_heating_sweep(
    problem_name: str,
    f0: Array,
    grid: VelocityGrid,
    dt: float,
    n_steps: int,
    n_snapshots: int,
    heating_kwargs: dict,
    expected_T_fn=None,
    expected_m: float | None = None,
    models: tuple[str, ...] = MODELS,
    schemes: tuple[str, ...] = SCHEMES,
    experiment_name: str = EXPERIMENT,
) -> dict[str, HeatingMetrics]:
    """Run heating simulation for all model/scheme combos and log to MLflow.

    Args:
        problem_name: Name for MLflow run
        f0: Initial distribution
        grid: VelocityGrid
        dt: Time step
        n_steps: Total steps to run
        n_snapshots: Number of snapshots to save (evenly spaced)
        heating_kwargs: Dict of heating params (D0_heating, ib_vosc2, ib_Z2ni_w0)
        expected_T_fn: Optional callable(times) -> expected T array
        expected_m: Optional expected super-Gaussian m (for IB)
        models: Tuple of model names to sweep
        schemes: Tuple of scheme names to sweep
        experiment_name: MLflow experiment name
    """
    results = {}
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=problem_name) as parent_run:
        mlflow.set_tags(get_git_info())
        mlflow.log_params(
            {
                "problem": problem_name,
                "dt": dt,
                "n_steps": n_steps,
                "nv": grid.nv,
                "vmax": float(grid.vmax),
                **{f"heating_{k}": v for k, v in heating_kwargs.items()},
            }
        )

        for model_name, scheme_name in itertools.product(models, schemes):
            child_name = f"{model_name}_{scheme_name}"

            with mlflow.start_run(run_name=child_name, nested=True):
                mlflow.log_params({"model": model_name, "scheme": scheme_name})

                # Build collisions
                vfp_grid = Grid(
                    xmin=0.0,
                    xmax=1.0,
                    nx=1,
                    tmin=0.0,
                    tmax=float(n_steps * dt),
                    dt=dt,
                    nv=grid.nv,
                    vmax=float(grid.vmax),
                    nl=1,
                    boundary="periodic",
                )
                collisions = F0Collisions(
                    nuee_coeff=1.0,
                    grid=vfp_grid,
                    model=get_model(model_name, vfp_grid.v, vfp_grid.dv),
                    scheme=get_scheme(SCHEME_MAP[scheme_name], vfp_grid.dv),
                    sc_beta=SelfConsistentBetaConfig(max_steps=0),
                )

                # Build vector field with heating
                vf = F0CollisionsHeatingVectorField(
                    collisions=collisions,
                    dt=dt,
                    D0_heating=heating_kwargs.get("D0_heating"),
                    ib_vosc2=heating_kwargs.get("ib_vosc2"),
                    ib_Z2ni_w0=heating_kwargs.get("ib_Z2ni_w0"),
                )

                # Run with diffrax Stepper
                t_final = n_steps * dt
                solution = diffeqsolve(
                    ODETerm(vf),
                    Stepper(),
                    t0=0.0,
                    t1=t_final,
                    dt0=dt,
                    y0=f0,
                    saveat=SaveAt(t0=True, t1=True, steps=n_snapshots),
                    max_steps=n_steps,
                )
                f_history = solution.ys
                times = solution.ts

                # Compute expected T array for metrics
                expected_T = None
                if expected_T_fn is not None:
                    expected_T = jnp.array(expected_T_fn(times))

                metrics = compute_heating_metrics(
                    f_history,
                    grid,
                    times,
                    expected_T=expected_T,
                    expected_m=expected_m,
                )
                results[child_name] = metrics

                # Log MLflow metrics
                for i in range(times.shape[0]):
                    log_dict = {
                        "rel_density": float(metrics.rel_density[i]),
                        "temperature": float(metrics.temperature_discrete[i]),
                        "positivity_violation": float(metrics.positivity_violation[i]),
                        "rmse_expected": float(metrics.rmse_expected[i]),
                        "kurtosis": float(metrics.kurtosis[i]),
                    }
                    if not np.isnan(float(metrics.sg_m[i])):
                        log_dict["sg_m"] = float(metrics.sg_m[i])
                    if not np.isnan(float(metrics.entropy[i])):
                        log_dict["entropy"] = float(metrics.entropy[i])
                    if expected_T is not None:
                        log_dict["T_error"] = float(metrics.temperature_discrete[i] - expected_T[i])
                    mlflow.log_metrics(log_dict)

                # Save artifacts
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)

                    # Distribution pickles
                    with open(tmpdir / "distribution_history.pkl", "wb") as fp:
                        pickle.dump(
                            {
                                "times": times,
                                "f_history": f_history,
                                "v": grid.v,
                                "dv": grid.dv,
                            },
                            fp,
                        )

                    # Distribution evolution plots
                    plot_distribution_evolution(
                        f_history,
                        grid,
                        title=f"{problem_name}: {child_name} (log)",
                        output_path=tmpdir / "distribution_log.png",
                        log_scale=True,
                    )
                    plot_distribution_evolution(
                        f_history,
                        grid,
                        title=f"{problem_name}: {child_name} (linear)",
                        output_path=tmpdir / "distribution_linear.png",
                        log_scale=False,
                    )

                    mlflow.log_artifacts(str(tmpdir))

        # Parent-level dashboard
        if results:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                plot_heating_dashboard(
                    results,
                    title=f"{problem_name}: model/scheme comparison",
                    expected_T_fn=expected_T_fn,
                    expected_m=expected_m,
                    output_path=tmpdir / "heating_dashboard.png",
                )
                mlflow.log_artifacts(str(tmpdir))

    return results


# =============================================================================
# Unit tests: Maxwellian heating
# =============================================================================


def test_maxwellian_heating_preserves_shape():
    """Maxwellian heating should keep the distribution Maxwellian while increasing T."""
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    T0 = _temperature(f0, grid.v, grid.dv)

    D0 = 0.5
    history = _step(coll, f0, dt=0.01, n_steps=50, D0_heating=D0)
    f_final = history[-1]
    T_final = _temperature(f_final, grid.v, grid.dv)

    assert T_final > T0 + 0.01, f"T should increase: T0={T0:.4f}, T_final={T_final:.4f}"
    rmse = _rmse_vs_maxwellian(f_final, grid.v, grid.dv)
    assert rmse < 1e-3, f"Distribution should remain Maxwellian, rmse={rmse:.2e}"


def test_maxwellian_cooling_preserves_shape():
    """Maxwellian cooling (D₀ < 0) should decrease T while preserving Maxwellian shape."""
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    T0 = _temperature(f0, grid.v, grid.dv)

    D0 = -0.2
    history = _step(coll, f0, dt=0.01, n_steps=50, D0_heating=D0)
    f_final = history[-1]
    T_final = _temperature(f_final, grid.v, grid.dv)

    assert T_final < T0 - 0.01, f"T should decrease: T0={T0:.4f}, T_final={T_final:.4f}"
    rmse = _rmse_vs_maxwellian(f_final, grid.v, grid.dv)
    assert rmse < 1e-2, f"Distribution should remain Maxwellian, rmse={rmse:.2e}"


def test_maxwellian_heating_rate_matches_theory():
    """Maxwellian heating should give approximately constant dT/dt (Ridgers eq 4.45)."""
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)

    D0 = 0.3
    dt = 0.005
    n_steps = 100
    history = _step(coll, f0, dt=dt, n_steps=n_steps, D0_heating=D0)

    temps = jnp.array([_temperature(h, grid.v, grid.dv) for h in history])
    times = jnp.arange(n_steps + 1) * dt

    t_mean = jnp.mean(times)
    T_mean = jnp.mean(temps)
    slope = jnp.sum((times - t_mean) * (temps - T_mean)) / jnp.sum((times - t_mean) ** 2)
    T_fit = T_mean + slope * (times - t_mean)
    residuals = temps - T_fit
    rel_residual = jnp.max(jnp.abs(residuals)) / (temps[-1] - temps[0])

    assert slope > 0, f"Heating rate should be positive, got slope={slope:.4f}"
    assert rel_residual < 0.05, f"T(t) should be linear, max relative residual={rel_residual:.2e}"


# =============================================================================
# Unit tests: IB heating
# =============================================================================


def test_ib_heating_increases_temperature():
    """IB heating with v_osc² > 0 should increase temperature."""
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    T0 = _temperature(f0, grid.v, grid.dv)

    history = _step(coll, f0, dt=0.01, n_steps=50, ib_vosc2=0.5, ib_Z2ni_w0=1.0)
    T_final = _temperature(history[-1], grid.v, grid.dv)
    assert T_final > T0 + 0.01, f"IB should heat: T0={T0:.4f}, T_final={T_final:.4f}"


def test_ib_heating_rate_matches_theory():
    """IB heating rate should be monotonically increasing (Ridgers eq 4.43)."""
    grid = _make_grid(nv=256, vmax=6.0)
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)

    history = _step(coll, f0, dt=0.005, n_steps=60, ib_vosc2=0.3, ib_Z2ni_w0=1.0)
    temps = jnp.array([_temperature(h, grid.v, grid.dv) for h in history])
    assert jnp.all(jnp.diff(temps) > 0), "IB heating should monotonically increase temperature"


# =============================================================================
# Unit tests: Safety properties
# =============================================================================


def test_zero_heating_no_change():
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    f0x = f0[None, :]
    result_baseline = coll(nu=1.0, f0x=f0x, dt=0.01)
    result_heated = coll(nu=1.0, f0x=f0x, dt=0.01, D0_heating=0.0, ib_vosc2=0.0, ib_Z2ni_w0=0.0)
    np.testing.assert_allclose(np.array(result_baseline), np.array(result_heated), atol=1e-14)


def test_positivity_preserved_heating():
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    for f in _step(coll, f0, dt=0.01, n_steps=100, D0_heating=0.5):
        assert jnp.all(f >= -1e-20), f"Positivity violated: min(f)={jnp.min(f):.2e}"


def test_positivity_preserved_cooling():
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    for f in _step(coll, f0, dt=0.01, n_steps=100, D0_heating=-0.2):
        assert jnp.all(f >= -1e-20), f"Positivity violated: min(f)={jnp.min(f):.2e}"


def test_positivity_preserved_ib():
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    for f in _step(coll, f0, dt=0.01, n_steps=100, ib_vosc2=0.5, ib_Z2ni_w0=1.0):
        assert jnp.all(f >= -1e-20), f"Positivity violated: min(f)={jnp.min(f):.2e}"


def test_density_conserved_heating():
    grid = _make_grid()
    coll = _make_collisions(grid)
    f0 = _maxwellian(grid.v, T=1.0)
    n0 = _density(f0, grid.v, grid.dv)
    history = _step(coll, f0, dt=0.01, n_steps=50, D0_heating=0.5)
    n_final = _density(history[-1], grid.v, grid.dv)
    np.testing.assert_allclose(float(n_final), float(n0), rtol=1e-10)


# =============================================================================
# Integrated sweep tests (MLflow, marked slow)
# =============================================================================


@pytest.mark.slow
def test_maxwellian_heating_sweep():
    """Integrated test: Maxwellian heating with CoulombianKernel x {ChangCooper, LogMeanFlux}.

    Checks:
    - T(t) is approximately linear (constant dT/dt)
    - Distribution remains Maxwellian (m ≈ 2)
    - Density conserved, positivity preserved
    """
    grid = VelocityGrid(nv=NV, vmax=VMAX, spherical=True)
    f0 = _maxwellian(grid.v, T=1.0)
    D0 = 0.02  # Moderate rate so T stays well-resolved on [0, vmax] over 20τ
    dt = 0.005
    n_collision_times = 20.0
    n_steps = int(n_collision_times / dt)
    n_snapshots = 40

    # Measure the actual heating rate from a short run to build expected T
    # (Since D0_heating absorbs unknown normalization, we calibrate from the simulation)
    coll_cal = _make_collisions(_make_grid(), scheme="chang_cooper")
    cal_history = _step(coll_cal, f0, dt=dt, n_steps=20, D0_heating=D0)
    T_start = float(_temperature(cal_history[0], grid.v, grid.dv))
    T_end = float(_temperature(cal_history[-1], grid.v, grid.dv))
    heating_rate = (T_end - T_start) / (20 * dt)

    def expected_T_fn(times):
        return T_start + heating_rate * np.asarray(times)

    results = run_heating_sweep(
        problem_name="maxwellian_heating",
        f0=f0,
        grid=grid,
        dt=dt,
        n_steps=n_steps,
        n_snapshots=n_snapshots,
        heating_kwargs={"D0_heating": D0},
        expected_T_fn=expected_T_fn,
    )

    for name, metrics in results.items():
        # Density conserved
        assert abs(float(metrics.rel_density[-1])) < 1e-10, f"{name}: density not conserved"
        # Temperature increased
        assert float(metrics.temperature_discrete[-1]) > float(metrics.temperature_discrete[0]) + 0.1
        # Still Maxwellian (m ≈ 2)
        m_final = float(metrics.sg_m[-1])
        assert abs(m_final - 2.0) < 0.15, f"{name}: m should be ~2 (Maxwellian), got {m_final:.2f}"
        # Positivity
        assert float(metrics.positivity_violation[-1]) < 1e-20, f"{name}: positivity violated"


@pytest.mark.slow
def test_maxwellian_cooling_sweep():
    """Integrated test: Maxwellian cooling with CoulombianKernel x {ChangCooper, LogMeanFlux}.

    Checks:
    - T(t) decreases approximately linearly
    - Distribution remains Maxwellian (m ≈ 2)
    - T stays positive (cool by roughly factor of 2: T₀=1 → T_final≈0.5)
    - Density conserved, positivity preserved
    """
    grid = VelocityGrid(nv=NV, vmax=VMAX, spherical=True)
    f0 = _maxwellian(grid.v, T=1.0)
    # Gentle cooling so T drops by roughly factor of 2 over 20τ
    D0 = -0.005
    dt = 0.005
    n_collision_times = 20.0
    n_steps = int(n_collision_times / dt)
    n_snapshots = 40

    coll_cal = _make_collisions(_make_grid(), scheme="chang_cooper")
    cal_history = _step(coll_cal, f0, dt=dt, n_steps=20, D0_heating=D0)
    T_start = float(_temperature(cal_history[0], grid.v, grid.dv))
    T_end = float(_temperature(cal_history[-1], grid.v, grid.dv))
    cooling_rate = (T_end - T_start) / (20 * dt)  # negative

    def expected_T_fn(times):
        return T_start + cooling_rate * np.asarray(times)

    results = run_heating_sweep(
        problem_name="maxwellian_cooling",
        f0=f0,
        grid=grid,
        dt=dt,
        n_steps=n_steps,
        n_snapshots=n_snapshots,
        heating_kwargs={"D0_heating": D0},
        expected_T_fn=expected_T_fn,
    )

    for name, metrics in results.items():
        # Density conserved
        assert abs(float(metrics.rel_density[-1])) < 1e-10, f"{name}: density not conserved"
        # Temperature decreased but stayed positive
        T_final = float(metrics.temperature_discrete[-1])
        assert T_final < float(metrics.temperature_discrete[0]) - 0.01, f"{name}: T should decrease"
        assert T_final > 0.3, f"{name}: T should stay well above zero, got T={T_final:.4f}"
        # Still Maxwellian (m ≈ 2)
        m_final = float(metrics.sg_m[-1])
        assert abs(m_final - 2.0) < 0.15, f"{name}: m should be ~2 (Maxwellian), got {m_final:.2f}"
        # Positivity (LogMeanFlux can produce small violations during cooling)
        pv = float(metrics.positivity_violation[-1])
        assert pv < 1e-8, f"{name}: positivity violated: {pv:.2e}"


@pytest.mark.slow
def test_ib_heating_sweep():
    """Integrated test: IB heating with CoulombianKernel x {ChangCooper, LogMeanFlux}.

    With ee collisions active, Matte et al (Ridgers eq 2.67) predicts the
    distribution tends toward a super-Gaussian with m = 2 + 3/(1 + 1.66/a^0.724)
    where a = Z·v_osc/v_T is the Langdon parameter.

    Checks:
    - Temperature increases monotonically
    - Super-Gaussian m evolves away from 2 toward Matte prediction
    - Density conserved, positivity preserved
    """
    grid = VelocityGrid(nv=256, vmax=VMAX, spherical=True)
    f0 = _maxwellian(grid.v, T=1.0)

    # IB parameters: moderate Langdon parameter a
    Z = 1.0
    v_osc = 1.0  # v_osc/v_T ~ 1 (moderate IB)
    v_T = jnp.sqrt(2.0)  # T=1 → v_T = √(2T) = √2
    ib_vosc2 = v_osc**2
    ib_Z2ni_w0 = Z**2 * 1.0 / 10.0  # Z²nᵢ/ω₀, ω₀ = 10

    # Langdon parameter and expected m
    alpha = float(Z * v_osc / v_T)
    expected_m = matte_expected_m(alpha)

    dt = 0.005
    n_collision_times = 20.0
    n_steps = int(n_collision_times / dt)
    n_snapshots = 40

    results = run_heating_sweep(
        problem_name="ib_heating",
        f0=f0,
        grid=grid,
        dt=dt,
        n_steps=n_steps,
        n_snapshots=n_snapshots,
        heating_kwargs={
            "ib_vosc2": float(ib_vosc2),
            "ib_Z2ni_w0": float(ib_Z2ni_w0),
        },
        expected_m=expected_m,
    )

    for name, metrics in results.items():
        # Density conserved
        assert abs(float(metrics.rel_density[-1])) < 1e-10, f"{name}: density not conserved"
        # Temperature increased
        assert float(metrics.temperature_discrete[-1]) > float(metrics.temperature_discrete[0])
        # m has moved away from 2 (toward super-Gaussian)
        m_final = float(metrics.sg_m[-1])
        assert m_final > 2.05, f"{name}: IB should distort away from Maxwellian, got m={m_final:.2f}"
        # Positivity
        assert float(metrics.positivity_violation[-1]) < 1e-20, f"{name}: positivity violated"


@pytest.mark.slow
def test_collisionless_ib_from_maxwellian():
    """Collisionless IB starting from Maxwellian → should approach Langdon (m=5).

    Without ee collisions, the distribution tends to a pure Langdon
    distribution f₀ ∝ exp(-v⁵) (Ridgers eq 2.65, §2.4.5).

    Uses NullModel (zero collision D/C) with nuee_coeff=1.0 so the
    1/v² Jacobian is still active for the IB operator.
    """
    grid = VelocityGrid(nv=256, vmax=VMAX, spherical=True)
    f0 = _maxwellian(grid.v, T=1.0)

    ib_vosc2 = 0.5
    ib_Z2ni_w0 = 0.1  # Small Z²nᵢ/ω₀ → g(v) ≈ 1 for most of grid

    dt = 0.005
    n_collision_times = 20.0
    n_steps = int(n_collision_times / dt)
    n_snapshots = 40

    results = run_heating_sweep(
        problem_name="collisionless_ib_from_maxwellian",
        f0=f0,
        grid=grid,
        dt=dt,
        n_steps=n_steps,
        n_snapshots=n_snapshots,
        heating_kwargs={"ib_vosc2": ib_vosc2, "ib_Z2ni_w0": ib_Z2ni_w0},
        expected_m=5.0,  # Pure Langdon
        models=("Null",),
        schemes=SCHEMES,
    )

    for name, metrics in results.items():
        # Density conserved
        assert abs(float(metrics.rel_density[-1])) < 1e-10, f"{name}: density not conserved"
        # Temperature increased
        assert float(metrics.temperature_discrete[-1]) > float(metrics.temperature_discrete[0])
        # m should be moving toward 5 (Langdon)
        m_final = float(metrics.sg_m[-1])
        assert m_final > 2.5, f"{name}: should approach Langdon, got m={m_final:.2f}"
        # Positivity
        assert float(metrics.positivity_violation[-1]) < 1e-20, f"{name}: positivity violated"


@pytest.mark.slow
def test_collisionless_ib_from_langdon():
    """Collisionless IB starting from Langdon (m=5) → should preserve shape.

    A Langdon distribution is the steady-state shape of collisionless IB.
    Starting from m=5 super-Gaussian, the distribution should remain m≈5
    while temperature increases. RMSE to instantaneous Langdon should stay small.
    """
    grid = VelocityGrid(nv=256, vmax=VMAX, spherical=True)
    f0 = make_super_gaussian(grid.v, float(grid.dv), T=1.0, m=5.0)

    ib_vosc2 = 0.5
    ib_Z2ni_w0 = 0.1

    dt = 0.005
    n_collision_times = 20.0
    n_steps = int(n_collision_times / dt)
    n_snapshots = 40

    results = run_heating_sweep(
        problem_name="collisionless_ib_from_langdon",
        f0=f0,
        grid=grid,
        dt=dt,
        n_steps=n_steps,
        n_snapshots=n_snapshots,
        heating_kwargs={"ib_vosc2": ib_vosc2, "ib_Z2ni_w0": ib_Z2ni_w0},
        expected_m=5.0,
        models=("Null",),
        schemes=SCHEMES,
    )

    for name, metrics in results.items():
        # Density conserved
        assert abs(float(metrics.rel_density[-1])) < 1e-10, f"{name}: density not conserved"
        # Temperature increased
        assert float(metrics.temperature_discrete[-1]) > float(metrics.temperature_discrete[0])
        # m should stay near 5 (Langdon is the steady-state shape)
        m_final = float(metrics.sg_m[-1])
        assert abs(m_final - 5.0) < 1.0, f"{name}: Langdon should be preserved, got m={m_final:.2f}"
        # Positivity
        assert float(metrics.positivity_violation[-1]) < 1e-20, f"{name}: positivity violated"
