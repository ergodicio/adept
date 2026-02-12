#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Plotting utilities for Fokker-Planck relaxation test artifacts.

Generates publication-quality figures for MLflow logging.
"""

import re
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .registry import VelocityGrid
from .time_stepper import RelaxationResult


def _setup_plot_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


_setup_plot_style()


def plot_distribution_evolution(
    result: RelaxationResult,
    grid: VelocityGrid,
    title: str = "Distribution Evolution",
    output_path: Path | None = None,
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot the evolution of the distribution function.

    Shows initial, intermediate, and final distributions.

    Args:
        result: RelaxationResult from run_relaxation
        grid: VelocityGrid instance
        title: Figure title
        output_path: Optional path to save figure
        log_scale: If True, use semilogy; if False, use linear scale
    """
    fig, ax = plt.subplots()

    if result.f_history is not None:
        n_snaps = result.f_history.shape[0]
        n_curves = min(10, n_snaps)
        indices = np.linspace(0, n_snaps - 1, n_curves, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_curves))

        for i, idx in enumerate(indices):
            t = float(result.times[idx])
            label = f"t/tau = {t * result.config.nu:.1f}"
            if log_scale:
                ax.semilogy(grid.v, result.f_history[idx], color=colors[i], label=label)
            else:
                ax.plot(grid.v, result.f_history[idx], color=colors[i], label=label)
    else:
        # Only initial and final
        if log_scale:
            ax.semilogy(grid.v, result.f_initial, "b-", label="Initial", linewidth=2)
            ax.semilogy(grid.v, result.f_final, "r--", label="Final", linewidth=2)
        else:
            ax.plot(grid.v, result.f_initial, "b-", label="Initial", linewidth=2)
            ax.plot(grid.v, result.f_final, "r--", label="Final", linewidth=2)

    ax.set_xlabel("v / vth")
    ax.set_ylabel("f(v)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if log_scale:
        ax.set_ylim(bottom=1e-10)

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_comparison_dashboard(
    results: dict[str, RelaxationResult],
    title: str = "Comparison",
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Create a 6-panel comparison dashboard showing multiple results overlaid.

    Panels (2x3): density conservation, temperature, positivity,
    RMSE to final Maxwellian, RMSE to instantaneous Maxwellian, entropy.

    Args:
        results: Dict mapping label to RelaxationResult
        title: Figure title
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """

    # Sort results: param labels (dt=X/sc=Y) in canonical order, others alphabetical
    def _sort_key(label):
        m = re.match(r"dt=([\d.]+)/sc=(\d+)", label)
        if not m:
            return (2, label, 0)
        dt, sc = float(m.group(1)), int(m.group(2))
        # Vary-sc group (dt=1.0) first sorted by sc, then vary-dt group sorted by dt
        if dt == 1.0:
            return (0, sc, 0)
        return (1, dt, 0)

    sorted_names = sorted(results.keys(), key=_sort_key)

    n_results = len(sorted_names)
    cmap = plt.cm.tab20 if n_results > 10 else plt.cm.tab10
    colors = {name: cmap(i / cmap.N) for i, name in enumerate(sorted_names)}

    # Transfer batched snapshots to numpy (one transfer per result)
    stacked_results = {}
    for name in sorted_names:
        result = results[name]
        s = jax.tree.map(np.asarray, result.snapshots)
        stacked_results[name] = (s, s.time * result.config.nu)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=16)

    def _plot(ax, times, y, color, label, log=False):
        """Plot a line, or a single final-value marker when only 2 snapshots."""
        plot_fn = ax.semilogy if log else ax.plot
        if len(times) <= 2:
            plot_fn(times[-1], y[-1], "o", color=color, label=label, markersize=8)
        else:
            plot_fn(times, y, color=color, label=label, linewidth=1.5)

    # Density conservation
    ax = axes[0, 0]
    for name, (s, times) in stacked_results.items():
        _plot(ax, times, s.rel_density, colors[name], name)
    ax.set_xlabel("t / collision times")
    ax.set_ylabel("(n - n0) / n0")
    ax.set_title("Density Conservation")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)

    # Temperature (raw total-energy T, line-style legend)
    ax = axes[0, 1]
    for name, (s, times) in stacked_results.items():
        if len(times) <= 2:
            ax.plot(times[-1], s.temperature_discrete[-1], "o", color=colors[name], markersize=8)
            ax.plot(times[-1], s.temperature_sc[-1], "s", color=colors[name], markersize=6, alpha=0.7)
        else:
            ax.plot(times, s.temperature_discrete, color=colors[name], linewidth=1.5)
            ax.plot(times, s.temperature_sc, color=colors[name], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("t / collision times")
    ax.set_ylabel(r"$T$ (total energy)")
    ax.set_title("Total Energy")
    ax.legend(
        handles=[
            Line2D([], [], color="k", linestyle="--", linewidth=1.0, label="self-consistent"),
            Line2D([], [], color="k", linestyle="-", linewidth=1.5, label="discrete"),
        ],
        fontsize=7,
    )

    # Positivity violation (only panel with the full per-result legend)
    ax = axes[0, 2]
    for name, (s, times) in stacked_results.items():
        _plot(ax, times, s.positivity_violation, colors[name], name)
    ax.set_xlabel("t / collision times")
    ax.set_ylabel("Positivity Violation")
    ax.set_title("Positivity Violation")
    ax.legend(fontsize=7)

    # RMSE to expected equilibrium
    ax = axes[1, 0]
    for name, (s, times) in stacked_results.items():
        _plot(ax, times, s.rmse_expected, colors[name], name, log=True)
    ax.set_xlabel("t / collision times")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE to Expected Maxwellian")

    # RMSE to instantaneous Maxwellian
    ax = axes[1, 1]
    for name, (s, times) in stacked_results.items():
        _plot(ax, times, s.rmse_instant, colors[name], name, log=True)
    ax.set_xlabel("t / collision times")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE to Instantaneous Maxwellian")

    # Entropy (raw)
    ax = axes[1, 2]
    for name, (s, times) in stacked_results.items():
        mask = ~np.isnan(s.entropy)
        if np.any(mask):
            if len(times) <= 2:
                ax.plot(times[mask][-1], s.entropy[mask][-1], "o", color=colors[name], markersize=8)
            else:
                ax.plot(times[mask], s.entropy[mask], color=colors[name], linewidth=1.5)
    ax.set_xlabel("t / collision times")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, hspace=0.35, wspace=0.3)

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
