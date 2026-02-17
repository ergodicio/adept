#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
MLflow run management for Fokker-Planck relaxation tests.

Provides proper nesting: parent run per (geometry, problem), child runs per (model, scheme).
"""

from __future__ import annotations

import pickle
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import mlflow
from jax import Array

from .factories import AbstractFPRelaxationVectorFieldFactory
from .metrics import compute_momentum, compute_rmse
from .plotting import plot_comparison_dashboard, plot_distribution_evolution
from .registry import VelocityGrid, get_git_info
from .time_stepper import RelaxationResult, TimeStepperConfig, run_relaxation


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


def problem_params(ic_fn) -> dict:
    """Extract loggable parameters from ic_fn."""
    if isinstance(ic_fn, partial):
        return dict(ic_fn.keywords)
    return {}


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
    name = problem_name(ic_fn)
    grid = VelocityGrid(nv=nv, vmax=vmax, spherical=factory.spherical)
    f0 = ic_fn(grid)

    extra_combos = slow_extra_combos if slow else None
    geometry = "spherical" if factory.spherical else "cartesian"
    results = run_relaxation_sweep(
        problem_name=name,
        factory=factory,
        f0=f0,
        experiment_name=experiment_name,
        extra_params=problem_params(ic_fn),
        nv=nv,
        vmax=vmax,
        extra_param_combos=extra_combos,
    )
    assert results, f"No results for {name} ({geometry})"

    for name, result in results.items():
        # Skip assertions for CentralDifferencing (only log results)
        if "CentralDifferencing" in name:
            continue

        s = result.snapshots  # Batched metrics

        # Density conservation
        assert abs(s.rel_density[-1]) < 1e-6, f"{name}: Density not conserved: rel_density={s.rel_density[-1]:.2e}"

        # Temperature (total energy) conservation
        assert jnp.isclose(s.temperature_discrete[-1], s.temperature_discrete[0], atol=0.0, rtol=temperature_tol), (
            f"{name}: Temperature changed: rel_T={s.temperature_discrete[-1] / s.temperature_discrete[0] - 1.0:.2e}"
        )

        # Relaxation toward equilibrium
        if not problem.get("equilibrium", False):
            assert s.rmse_instant[-1] < s.rmse_instant[0], f"{name}: Distribution did not relax toward Maxwellian"

        # Extra: equilibrium RMSE check
        if problem.get("extra_checks") == "rmse":
            rmse = compute_rmse(f0, result.f_final, grid)
            assert rmse < 1e-2, f"{name}: Equilibrium drifted: RMSE={rmse}"

    # Extra: momentum conservation check (Dougherty vs LB)
    if problem.get("extra_checks") == "momentum":
        p0 = float(compute_momentum(f0, grid))
        d_key = "Dougherty_ChangCooper"
        lb_key = "LenardBernstein_ChangCooper"
        if d_key in results and lb_key in results:
            drift_d = abs(float(compute_momentum(results[d_key].f_final, grid)) - p0) / abs(p0)
            drift_lb = abs(float(compute_momentum(results[lb_key].f_final, grid)) - p0) / abs(p0)
            assert drift_d < drift_lb, f"Dougherty drift ({drift_d:.3f}) not better than LB ({drift_lb:.3f})"


def _pickle_result(result: RelaxationResult, grid: VelocityGrid, path: Path) -> None:
    """Pickle a RelaxationResult's distribution history to disk."""
    dist_data = {
        "times": result.times,
        "f_history": result.f_history,
        "f_initial": result.f_initial,
        "f_final": result.f_final,
        "v": grid.v,
        "dv": grid.dv,
        "spherical": grid.spherical,
    }
    with open(path, "wb") as f:
        pickle.dump(dist_data, f)


def run_relaxation_sweep(
    problem_name: str,
    factory: AbstractFPRelaxationVectorFieldFactory,
    f0: Array,
    experiment_name: str = "fokker-planck-relaxation-tests",
    dt_over_tau: float = 1.0,
    sc_iterations: int = 2,
    n_collision_times: float = 10.0,
    extra_params: dict | None = None,
    nv: int = 128,
    vmax: float = 8.0,
    extra_param_combos: list[dict] | None = None,
) -> dict[str, RelaxationResult]:
    """
    Run relaxation for all model/scheme combinations and log to MLflow.

    Creates a parent run named `{problem}` with child runs for each
    model/scheme combination.

    Args:
        problem_name: Name of the problem (e.g., "supergaussian")
        factory: Vector field factory that provides model/scheme combinations
            and creates the appropriate vector fields for the solver.
        initial_condition_fn: Function that takes grid and returns f0
        experiment_name: MLflow experiment name
        dt_over_tau: Time step in units of collision time (base sim)
        sc_iterations: Number of self-consistency iterations (base sim)
        n_collision_times: How many collision times to run
        extra_params: Additional parameters to log
        nv: Number of velocity grid points
        vmax: Maximum velocity
        extra_param_combos: If provided (slow test), each child runs the base sim
            plus these extra (dt_over_tau, sc_iterations) combos. Each dict must
            have keys "dt_over_tau" and "sc_iterations".

    Returns:
        Dict mapping "{model}_{scheme}" to RelaxationResult (base sim only).
        Extra param combos (slow) are used internally for artifacts but not returned.
    """
    # Factory knows its grid geometry
    grid = VelocityGrid(nv=nv, vmax=vmax, spherical=factory.spherical)
    geometry = "spherical" if factory.spherical else "cartesian"

    # Base config
    base_config = TimeStepperConfig(
        n_collision_times=n_collision_times,
        nu=1.0,
        dt_over_tau=dt_over_tau,
    )
    base_label = "dt={dt_over_tau}/sc={sc_iterations}"

    # Build list of all param combos to run per child
    # Each entry: (label, config, sc_iterations, is_base)
    all_combos = [(base_label, base_config, sc_iterations, True)]
    if extra_param_combos is not None:
        for combo in extra_param_combos:
            combo_dt = combo["dt_over_tau"]
            combo_sc = combo["sc_iterations"]
            label = "dt={combo_dt}/sc={combo_sc}"
            cfg = TimeStepperConfig(
                n_collision_times=n_collision_times,
                nu=1.0,
                dt_over_tau=combo_dt,
            )
            all_combos.append((label, cfg, combo_sc, False))

    is_slow = extra_param_combos is not None
    results = {}
    mlflow.set_experiment(experiment_name)

    # Parent run for this problem
    parent_run_name = f"{problem_name}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # Log git info and common params on parent
        for key, value in get_git_info().items():
            mlflow.set_tag(key, value)
        params = {
            "problem": problem_name,
            "geometry": geometry,
            "dt_over_tau": dt_over_tau,
            "sc_iterations": sc_iterations,
            "n_collision_times": n_collision_times,
            "nv": grid.nv,
            "vmax": float(grid.vmax),
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        # Child runs for each model/scheme combination
        for model_name in factory.model_names:
            for scheme_name in factory.scheme_names:
                child_run_name = f"{model_name}_{scheme_name}"

                with mlflow.start_run(run_name=child_run_name, nested=True):
                    mlflow.log_params({"model": model_name, "scheme": scheme_name})

                    # Run all combos for this child
                    child_results = {}
                    for label, cfg, sc_iter, is_base in all_combos:
                        # Compute dt for this combo
                        tau = 1.0 / cfg.nu
                        dt = cfg.dt_over_tau * tau

                        # Factory creates the vector field
                        vector_field = factory.make_vector_field(
                            grid=grid,
                            model_name=model_name,
                            scheme_name=scheme_name,
                            dt=dt,
                            nu=cfg.nu,
                            sc_iterations=sc_iter,
                        )

                        result = run_relaxation(
                            f0,
                            grid,
                            vector_field=vector_field,
                            config=cfg,
                        )
                        child_results[label] = result

                        # Log MLflow metrics only for base sim
                        if is_base:
                            s = result.snapshots  # Batched pytree, each leaf shape (n_snapshots,)
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

                    base_result = child_results[base_label]
                    results[child_run_name] = base_result

                    # Save artifacts
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir = Path(tmpdir)

                        # Pickle: all sims for slow, base only for fast
                        if is_slow:
                            for label, res in child_results.items():
                                safe_label = label.replace("/", "_").replace("=", "")
                                _pickle_result(res, grid, tmpdir / f"distribution_history_{safe_label}.pkl")
                        else:
                            _pickle_result(base_result, grid, tmpdir / "distribution_history.pkl")

                        # Distribution evolution plots (base sim only)
                        plot_distribution_evolution(
                            base_result,
                            grid,
                            title=f"{problem_name}: {model_name}/{scheme_name} (log)",
                            output_path=tmpdir / "distribution_log.png",
                            log_scale=True,
                        )
                        plot_distribution_evolution(
                            base_result,
                            grid,
                            title=f"{problem_name}: {model_name}/{scheme_name} (linear)",
                            output_path=tmpdir / "distribution_linear.png",
                            log_scale=False,
                        )

                        # Child-level comparison dashboard (slow only: all combos overlaid)
                        if is_slow:
                            plot_comparison_dashboard(
                                child_results,
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
