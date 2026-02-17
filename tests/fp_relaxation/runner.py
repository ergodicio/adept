#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
MLflow run management for Fokker-Planck relaxation tests.

Provides proper nesting: parent run per (geometry, problem), child runs per (model, scheme).
"""

from __future__ import annotations

import pickle
import tempfile
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import mlflow
from diffrax import ODETerm, SaveAt, diffeqsolve
from jax import Array

from adept._base_ import Stepper

from .factories import AbstractFPRelaxationVectorFieldFactory
from .metrics import RelaxationMetrics, compute_metrics, compute_momentum, compute_rmse
from .plotting import plot_comparison_dashboard, plot_distribution_evolution
from .registry import VelocityGrid, get_git_info


def problem_name(ic_fn) -> str:
    """Derive problem name from ic_fn (partial or bare function)."""
    if isinstance(ic_fn, partial):
        name = ic_fn.func.__name__
        if name == "bump_on_tail":
            name += "_narrow" if ic_fn.keywords.get("narrow") else "_wide"
        elif name == "shifted_maxwellian" and ic_fn.keywords.get("v_shift", 1.0) != 1.0:
            name += "_beam"
        return name
    return ic_fn.__name__


def run_relaxation_sweep_and_assert(
    factory: AbstractFPRelaxationVectorFieldFactory,
    experiment_name: str,
    problem: dict,
    slow: bool,
    slow_extra_combos: list[dict],
    nv: int,
    vmax: float,
    temperature_tol: float,
) -> None:
    """Run relaxation sweep and verify conservation properties.

    Combined entry point: calls run_relaxation_sweep, verifies conservation,
    and runs any extra checks (rmse, momentum).

    Args:
        factory: Vector field factory that provides model/scheme combinations
            and creates the appropriate vector fields for the solver.
    """
    ic_fn = problem["ic_fn"]
    grid = VelocityGrid(nv=nv, vmax=vmax, spherical=factory.spherical)
    f0 = ic_fn(grid)

    results = run_relaxation_sweep(
        problem_name=problem_name(ic_fn),
        factory=factory,
        f0=f0,
        grid=grid,
        experiment_name=experiment_name,
        extra_params=dict(ic_fn.keywords) if isinstance(ic_fn, partial) else {},
        extra_param_combos=slow_extra_combos if slow else None,
    )
    assert results, f"No results for {problem_name(ic_fn)}"

    for name, (metrics, f_history) in results.items():
        # Skip assertions for CentralDifferencing (only log results)
        if "CentralDifferencing" in name:
            continue

        # Density conservation
        assert abs(metrics.rel_density[-1]) < 1e-6, (
            f"{name}: Density not conserved: rel_density={metrics.rel_density[-1]:.2e}"
        )

        # Temperature (total energy) conservation
        assert jnp.isclose(
            metrics.temperature_discrete[-1], metrics.temperature_discrete[0], atol=0.0, rtol=temperature_tol
        ), f"{name}: Temperature changed: rel_T={
            metrics.temperature_discrete[-1] / metrics.temperature_discrete[0] - 1.0:.2e}"

        # Relaxation toward equilibrium
        if not problem.get("equilibrium", False):
            assert metrics.rmse_instant[-1] < metrics.rmse_instant[0], (
                f"{name}: Distribution did not relax toward Maxwellian"
            )

        # Extra: equilibrium RMSE check
        if problem.get("extra_checks") == "rmse":
            rmse = compute_rmse(f0, f_history[-1], grid)
            assert rmse < 1e-2, f"{name}: Equilibrium drifted: RMSE={rmse}"

    # Extra: momentum conservation check (Dougherty vs LB)
    if problem.get("extra_checks") == "momentum":
        p0 = float(compute_momentum(f0, grid))
        d_key = "Dougherty_ChangCooper"
        lb_key = "LenardBernstein_ChangCooper"
        if d_key in results and lb_key in results:
            _, f_history_d = results[d_key]
            _, f_history_lb = results[lb_key]
            drift_d = abs(float(compute_momentum(f_history_d[-1], grid)) - p0) / abs(p0)
            drift_lb = abs(float(compute_momentum(f_history_lb[-1], grid)) - p0) / abs(p0)
            assert drift_d < drift_lb, f"Dougherty drift ({drift_d:.3f}) not better than LB ({drift_lb:.3f})"


def run_relaxation_sweep(
    problem_name: str,
    factory: AbstractFPRelaxationVectorFieldFactory,
    f0: Array,
    grid: VelocityGrid,
    experiment_name: str = "fokker-planck-relaxation-tests",
    dt_over_tau: float = 1.0,
    sc_iterations: int = 2,
    n_collision_times: float = 10.0,
    extra_params: dict | None = None,
    extra_param_combos: list[dict] | None = None,
) -> dict[str, tuple[RelaxationMetrics, Array]]:
    """
    Run relaxation for all model/scheme combinations and log to MLflow.

    Creates a parent run named `{problem}` with child runs for each
    model/scheme combination.

    Args:
        problem_name: Name of the problem (e.g., "supergaussian")
        factory: Vector field factory that provides model/scheme combinations
            and creates the appropriate vector fields for the solver.
        f0: Initial distribution function
        grid: VelocityGrid instance
        experiment_name: MLflow experiment name
        dt_over_tau: Time step in units of collision time (base sim)
        sc_iterations: Number of self-consistency iterations (base sim)
        n_collision_times: How many collision times to run
        extra_params: Additional parameters to log
        extra_param_combos: If provided (slow test), each child runs the base sim
            plus these extra (dt_over_tau, sc_iterations) combos. Each dict must
            have keys "dt_over_tau" and "sc_iterations".

    Returns:
        Dict mapping "{model}_{scheme}" to (RelaxationMetrics, f_history) tuple.
        Extra param combos (slow) are used internally for artifacts but not returned.
    """
    geometry = "spherical" if grid.spherical else "cartesian"
    nu = 1.0  # Collision frequency (tau = 1/nu)

    base_label = f"dt={dt_over_tau}/sc={sc_iterations}"

    # Build list of all param combos to run per child
    # Each entry: (label, dt_over_tau, sc_iterations, is_base)
    all_combos = [(base_label, dt_over_tau, sc_iterations, True)]
    if extra_param_combos is not None:
        for combo in extra_param_combos:
            combo_dt = combo["dt_over_tau"]
            combo_sc = combo["sc_iterations"]
            all_combos.append((f"dt={combo_dt}/sc={combo_sc}", combo_dt, combo_sc, False))

    results = {}
    mlflow.set_experiment(experiment_name)

    # Parent run for this problem
    parent_run_name = f"{problem_name}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # Log git info and common params on parent
        mlflow.set_tags(get_git_info())
        mlflow.log_params(
            {
                "problem": problem_name,
                "geometry": geometry,
                "dt_over_tau": dt_over_tau,
                "sc_iterations": sc_iterations,
                "n_collision_times": n_collision_times,
                "nv": grid.nv,
                "vmax": float(grid.vmax),
            }
            | extra_params
        )

        # Child runs for each model/scheme combination
        for model_name in factory.model_names:
            for scheme_name in factory.scheme_names:
                child_run_name = f"{model_name}_{scheme_name}"

                with mlflow.start_run(run_name=child_run_name, nested=True):
                    mlflow.log_params({"model": model_name, "scheme": scheme_name})

                    # Run all combos for this child
                    child_results = {}
                    for label, combo_dt, combo_sc, is_base in all_combos:
                        dt = combo_dt / nu

                        vector_field = factory.make_vector_field(
                            grid=grid,
                            model_name=model_name,
                            scheme_name=scheme_name,
                            dt=dt,
                            nu=nu,
                            sc_iterations=combo_sc,
                        )

                        solution = diffeqsolve(
                            ODETerm(vector_field),
                            Stepper(),
                            t0=0.0,
                            t1=n_collision_times / nu,
                            dt0=dt,
                            y0=f0,
                            saveat=SaveAt(t0=True, t1=True),
                            max_steps=int(n_collision_times / combo_dt) + 4,
                        )
                        f_history = solution.ys

                        metrics = compute_metrics(f_history, grid, solution.ts)
                        child_results[label] = (metrics, f_history)

                        # Log MLflow metrics only for base sim
                        if is_base:
                            s = metrics
                            for i in range(s.time.shape[0]):
                                mlflow.log_metrics(
                                    {
                                        "rel_density": float(s.rel_density[i]),
                                        "rel_temperature_discrete": float(
                                            s.temperature_discrete[i] / s.temperature_discrete[0] - 1.0
                                        ),
                                        "rel_temperature_sc": float(s.temperature_sc[i] / s.temperature_sc[0] - 1.0),
                                        "rel_entropy": float(s.entropy[i] / s.entropy[0] - 1.0),
                                        "positivity_violation": float(s.positivity_violation[i]),
                                        "rmse_expected": float(s.rmse_expected[i]),
                                        "rmse_instant": float(s.rmse_instant[i]),
                                        "momentum_drift": float(s.momentum_drift[i]),
                                    }
                                )

                    base_metrics, base_f_history = child_results[base_label]
                    results[child_run_name] = (base_metrics, base_f_history)

                    # Save artifacts
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir = Path(tmpdir)

                        # Pickle distribution histories
                        for label, (m, fh) in child_results.items():
                            suffix = f"_{label.replace('/', '_').replace('=', '')}" if len(child_results) > 1 else ""
                            dist_data = {
                                "times": m.time,
                                "f_history": fh,
                                "f_initial": fh[0],
                                "f_final": fh[-1],
                                "v": grid.v,
                                "dv": grid.dv,
                                "spherical": grid.spherical,
                            }
                            with open(tmpdir / f"distribution_history{suffix}.pkl", "wb") as f:
                                pickle.dump(dist_data, f)

                        # Distribution evolution plots (base sim only)
                        plot_distribution_evolution(
                            base_f_history,
                            grid,
                            title=f"{problem_name}: {model_name}/{scheme_name} (log)",
                            output_path=tmpdir / "distribution_log.png",
                            log_scale=True,
                        )
                        plot_distribution_evolution(
                            base_f_history,
                            grid,
                            title=f"{problem_name}: {model_name}/{scheme_name} (linear)",
                            output_path=tmpdir / "distribution_linear.png",
                            log_scale=False,
                        )

                        # Child-level comparison dashboard (multiple combos overlaid)
                        if len(child_results) > 1:
                            plot_comparison_dashboard(
                                {label: m for label, (m, _) in child_results.items()},
                                title=f"{problem_name}: {model_name}/{scheme_name} sweep",
                                output_path=tmpdir / "comparison_dashboard.png",
                            )

                        mlflow.log_artifacts(str(tmpdir))

        # Parent-level comparison dashboard (base sim across model/scheme)
        if results:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                plot_comparison_dashboard(
                    {name: m for name, (m, _) in results.items()},
                    title=f"{parent_run_name}: model/scheme comparison",
                    output_path=tmpdir / "comparison_dashboard.png",
                )
                mlflow.log_artifacts(str(tmpdir))

    return results
