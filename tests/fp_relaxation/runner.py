#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
MLflow run management for Fokker-Planck relaxation tests.

Provides proper nesting: parent run per (geometry, problem), child runs per (model, scheme).
"""

from __future__ import annotations

import itertools
import pickle
import tempfile
from functools import partial
from pathlib import Path

import mlflow
from diffrax import ODETerm, SaveAt, diffeqsolve
from jax import Array

from adept._base_ import Stepper

from .factories import AbstractFPRelaxationVectorFieldFactory
from .metrics import RelaxationMetrics, compute_metrics
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
) -> dict[str, RelaxationMetrics]:
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
        Dict mapping "{model}_{scheme}" to RelaxationMetrics.
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
        for model_name, scheme_name in itertools.product(factory.model_names, factory.scheme_names):
            child_run_name = f"{model_name}_{scheme_name}"

            with mlflow.start_run(run_name=child_run_name, nested=True):
                mlflow.log_params({"model": model_name, "scheme": scheme_name})

                # Run all combos for this child
                child_results = {}
                for label, combo_dt, combo_sc, is_base in all_combos:
                    dt = combo_dt / nu

                    vector_field = factory.make_vector_field(
                        grid=grid, model_name=model_name, scheme_name=scheme_name, dt=dt, nu=nu, sc_iterations=combo_sc
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
                results[child_run_name] = base_metrics

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
                    results,
                    title=f"{parent_run_name}: model/scheme comparison",
                    output_path=tmpdir / "comparison_dashboard.png",
                )
                mlflow.log_artifacts(str(tmpdir))

    return results
