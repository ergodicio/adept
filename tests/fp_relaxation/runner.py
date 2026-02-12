#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
MLflow run management for Fokker-Planck relaxation tests.

Provides proper nesting: parent run per (geometry, problem), child runs per (model, scheme).
"""

import math
import pickle
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import mlflow

from .metrics import compute_momentum, compute_rmse
from .plotting import plot_comparison_dashboard, plot_distribution_evolution
from .registry import (
    MODELS,
    SCHEMES,
    VelocityGrid,
    get_git_info,
)
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
    geometry: str,
    model_names: list[str],
    experiment_name: str,
    problem: dict,
    slow: bool,
    slow_extra_combos: list[dict],
    nv: int,
    vmax: float,
    temperature_tol: float,
) -> None:
    """Run relaxation sweep and verify conservation properties.

    Combined entry point: calls run_relaxation_sweep, extracts base results,
    calls verify_relaxation_results, and runs any extra checks (rmse, momentum).
    """
    ic_fn = problem["ic_fn"]
    name = problem_name(ic_fn)
    grid = VelocityGrid(nv=nv, vmax=vmax, spherical=geometry == "spherical")
    f0 = ic_fn(grid)

    extra_combos = slow_extra_combos if slow else None
    results = run_relaxation_sweep(
        problem_name=name,
        geometry=geometry,
        model_names=model_names,
        initial_condition_fn=ic_fn,
        experiment_name=experiment_name,
        extra_params=problem_params(ic_fn),
        nv=nv,
        vmax=vmax,
        extra_param_combos=extra_combos,
    )
    assert results, f"No results for {name} ({geometry})"

    if slow:
        base_results = {k: v["dt=1.0/sc=2"] for k, v in results.items()}
    else:
        base_results = results

    verify_relaxation_results(
        base_results,
        grid,
        f0,
        temperature_tol=temperature_tol,
        check_rmse_improvement=not problem.get("equilibrium", False),
    )

    # Extra: equilibrium RMSE check
    if problem.get("extra_checks") == "rmse":
        for child_name, result in base_results.items():
            if "CentralDifferencing" in child_name:
                continue
            rmse = compute_rmse(f0, result.f_final, grid)
            assert rmse < 1e-2, f"{child_name}: Equilibrium drifted: RMSE={rmse}"

    # Extra: momentum conservation check (Dougherty vs LB)
    if problem.get("extra_checks") == "momentum":
        p0 = float(compute_momentum(f0, grid))
        d_key = "Dougherty_ChangCooper"
        lb_key = "LenardBernstein_ChangCooper"
        if d_key in base_results and lb_key in base_results:
            drift_d = abs(float(compute_momentum(base_results[d_key].f_final, grid)) - p0) / abs(p0)
            drift_lb = abs(float(compute_momentum(base_results[lb_key].f_final, grid)) - p0) / abs(p0)
            assert drift_d < drift_lb, f"Dougherty drift ({drift_d:.3f}) not better than LB ({drift_lb:.3f})"


def _rel(val, ref):
    """Compute (val - ref) / |ref|, guarding against near-zero reference."""
    return (val - ref) / abs(ref) if abs(ref) > 1e-30 else val - ref


def _log_metrics_for_result(result: RelaxationResult, config: TimeStepperConfig) -> None:
    """Log per-snapshot metrics to MLflow for a single result."""
    s = result.snapshots  # Batched pytree, each leaf shape (n_snapshots,)
    n_snapshots = s.time.shape[0]

    # Reference values (index 0) for relative metrics
    T_d0 = float(s.temperature_discrete[0])
    T_sc0 = float(s.temperature_sc[0])
    S0 = float(s.entropy[0])

    for i in range(n_snapshots):
        step = int(float(s.time[i]) * config.nu * 100)
        metrics_dict = {
            "rel_density": float(s.rel_density[i]),
            "rel_temperature_discrete": _rel(float(s.temperature_discrete[i]), T_d0),
            "rel_temperature_sc": _rel(float(s.temperature_sc[i]), T_sc0),
            "positivity_violation": float(s.positivity_violation[i]),
            "rmse_expected": float(s.rmse_expected[i]),
            "rmse_instant": float(s.rmse_instant[i]),
        }
        S = float(s.entropy[i])
        if not math.isnan(S) and not math.isnan(S0):
            metrics_dict["rel_entropy"] = _rel(S, S0)
        momentum_val = float(s.momentum_drift[i])
        if not math.isnan(momentum_val):
            metrics_dict["momentum_drift"] = momentum_val
        mlflow.log_metrics(metrics_dict, step=step)


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


def _make_param_label(dt_over_tau: float, sc_iterations: int) -> str:
    """Create a human-readable label for a (dt, sc) parameter combo."""
    return f"dt={dt_over_tau}/sc={sc_iterations}"


def run_relaxation_sweep(
    problem_name: str,
    geometry: str,
    model_names: list[str],
    initial_condition_fn: Callable[[VelocityGrid], any],
    experiment_name: str = "fokker-planck-relaxation-tests",
    dt_over_tau: float = 1.0,
    sc_iterations: int = 2,
    n_collision_times: float = 10.0,
    extra_params: dict | None = None,
    nv: int = 128,
    vmax: float = 8.0,
    extra_param_combos: list[dict] | None = None,
) -> dict[str, RelaxationResult] | dict[str, dict[str, RelaxationResult]]:
    """
    Run relaxation for all model/scheme combinations and log to MLflow.

    Creates a parent run named `{problem}` with child runs for each
    model/scheme combination.

    Args:
        problem_name: Name of the problem (e.g., "supergaussian")
        geometry: "cartesian" or "spherical"
        model_names: List of model names to run (e.g., ["LenardBernstein", "Dougherty"])
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
        When extra_param_combos is None (fast):
            Dict mapping "{model}_{scheme}" to RelaxationResult (base sim only)
        When extra_param_combos is provided (slow):
            Dict mapping "{model}_{scheme}" to dict mapping param label to RelaxationResult
    """
    spherical = geometry == "spherical"
    grid = VelocityGrid(nv=nv, vmax=vmax, spherical=spherical)
    models = model_names
    schemes = list(SCHEMES)

    if not models:
        return {}

    # Create initial condition
    f0 = initial_condition_fn(grid)

    # Base config
    base_config = TimeStepperConfig(
        n_collision_times=n_collision_times,
        n_snapshots=100,
        nu=1.0,
        dt_over_tau=dt_over_tau,
    )
    base_label = _make_param_label(dt_over_tau, sc_iterations)

    # Build list of all param combos to run per child
    # Each entry: (label, config, sc_iterations, is_base)
    all_combos = [(base_label, base_config, sc_iterations, True)]
    if extra_param_combos is not None:
        for combo in extra_param_combos:
            combo_dt = combo["dt_over_tau"]
            combo_sc = combo["sc_iterations"]
            label = _make_param_label(combo_dt, combo_sc)
            cfg = TimeStepperConfig(
                n_collision_times=n_collision_times,
                n_snapshots=100,
                nu=1.0,
                dt_over_tau=combo_dt,
            )
            all_combos.append((label, cfg, combo_sc, False))

    is_slow = extra_param_combos is not None
    results = {}
    mlflow.set_experiment(experiment_name)
    git_info = get_git_info()

    # Parent run for this problem
    suffix = "_sweep" if is_slow else ""
    parent_run_name = f"{problem_name}{suffix}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # Log git info and common params on parent
        for key, value in git_info.items():
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
        for model_type in models:
            for scheme_type in schemes:
                child_run_name = f"{model_type}_{scheme_type}"

                with mlflow.start_run(run_name=child_run_name, nested=True):
                    mlflow.log_params({"model": model_type, "scheme": scheme_type})

                    model = MODELS[model_type](v=grid.v, dv=grid.dv)
                    scheme = SCHEMES[scheme_type](dv=grid.dv)

                    # Run all combos for this child
                    child_results = {}
                    for label, cfg, sc_iter, is_base in all_combos:
                        result = run_relaxation(
                            f0,
                            grid,
                            model,
                            scheme,
                            cfg,
                            sc_iter,
                            store_f_history=True,
                        )
                        child_results[label] = result

                        # Log MLflow metrics only for base sim
                        if is_base:
                            _log_metrics_for_result(result, cfg)

                    base_result = child_results[base_label]

                    # Store results
                    if is_slow:
                        results[child_run_name] = child_results
                    else:
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
                            title=f"{problem_name}: {model_type}/{scheme_type} (log)",
                            output_path=tmpdir / "distribution_log.png",
                            log_scale=True,
                        )
                        plot_distribution_evolution(
                            base_result,
                            grid,
                            title=f"{problem_name}: {model_type}/{scheme_type} (linear)",
                            output_path=tmpdir / "distribution_linear.png",
                            log_scale=False,
                        )

                        # Child-level comparison dashboard (slow only: all combos overlaid)
                        if is_slow:
                            plot_comparison_dashboard(
                                child_results,
                                title=f"{problem_name}: {model_type}/{scheme_type} sweep",
                                output_path=tmpdir / "comparison_dashboard.png",
                            )

                        mlflow.log_artifacts(str(tmpdir))

        # Parent-level comparison dashboard (base sim across model/scheme)
        base_results_for_dashboard = {}
        for child_name in results:
            if is_slow:
                base_results_for_dashboard[child_name] = results[child_name][base_label]
            else:
                base_results_for_dashboard[child_name] = results[child_name]

        if base_results_for_dashboard:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                plot_comparison_dashboard(
                    base_results_for_dashboard,
                    title=f"{parent_run_name}: model/scheme comparison",
                    output_path=tmpdir / "comparison_dashboard.png",
                )
                mlflow.log_artifacts(str(tmpdir))

    return results


def verify_relaxation_results(
    results: dict[str, RelaxationResult],
    grid: VelocityGrid,
    f0,
    check_temperature: bool = True,
    temperature_tol: float = 5e-2,
    check_rmse_improvement: bool = True,
) -> None:
    """
    Verify basic conservation properties for ChangCooper results only.

    CentralDifferencing results are logged but not asserted (less stable scheme).
    Uses the pre-computed discrepancy metrics from the final snapshot.

    Args:
        results: Dict from run_relaxation_sweep (flat: name -> RelaxationResult)
        grid: VelocityGrid used for the runs
        f0: Initial distribution
        check_temperature: Whether to check temperature conservation
        temperature_tol: Tolerance for temperature conservation
        check_rmse_improvement: Whether to check RMSE improves (skip for equilibrium)
    """
    for name, result in results.items():
        # Skip assertions for CentralDifferencing (only log results)
        if "CentralDifferencing" in name:
            continue

        s = result.snapshots  # Batched metrics

        # Density conservation
        rel_n = float(s.rel_density[-1])
        assert abs(rel_n) < 1e-6, f"{name}: Density not conserved: rel_density={rel_n:.2e}"

        # Temperature (total energy) conservation
        if check_temperature:
            T0 = float(s.temperature_discrete[0])
            rel_T = _rel(float(s.temperature_discrete[-1]), T0)
            assert abs(rel_T) < temperature_tol, f"{name}: Temperature changed: rel_T={rel_T:.2e}"

        # Relaxation toward equilibrium
        if check_rmse_improvement:
            initial_rmse = float(s.rmse_instant[0])
            final_rmse = float(s.rmse_instant[-1])
            assert final_rmse < initial_rmse, f"{name}: Distribution did not relax toward Maxwellian"
