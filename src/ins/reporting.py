"""Publication-ready visualization and output helpers for Monte Carlo runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ins-monte-carlo-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from ins.config import SimulationConfig, save_config
from ins.monte_carlo import MonteCarloResult


AXIS_NAMES = ("North", "East", "Down")


def _time_axis(time_s: np.ndarray) -> tuple[np.ndarray, str]:
    if time_s[-1] >= 120.0:
        return time_s / 60.0, "Time (min)"
    return time_s, "Time (s)"


def _plot_error_panel(
    axis: plt.Axes,
    time: np.ndarray,
    runs: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    component: int,
    title: str,
    ylabel: str,
) -> None:
    for run_index in range(runs.shape[0]):
        axis.plot(
            time,
            runs[run_index, :, component],
            color="tab:blue",
            alpha=0.16,
            linewidth=0.7,
            label="Monte Carlo runs" if run_index == 0 else None,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9, label="Truth")
    axis.plot(time, mean[:, component], color="tab:orange", linewidth=1.5, label="Ensemble mean")
    axis.plot(time, lower[:, component], color="tab:green", linestyle="--", linewidth=1.1, label="Mean ± 2σ")
    axis.plot(time, upper[:, component], color="tab:green", linestyle="--", linewidth=1.1)
    axis.set_title(f"{title}: {AXIS_NAMES[component]}")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.28)


def build_monte_carlo_report(result: MonteCarloResult, config: SimulationConfig) -> plt.Figure:
    """Create one figure containing nine error panels and a 2-D trajectory."""

    figure = plt.figure(figsize=(16, 17))
    grid = figure.add_gridspec(4, 3, height_ratios=(1.0, 1.0, 1.0, 1.25))
    time, time_label = _time_axis(result.truth.time_s)

    panel_groups = (
        (
            "Position error",
            result.position_error_ned_m,
            result.position_mean_m,
            result.position_lower_2sigma_m,
            result.position_upper_2sigma_m,
            "Error (m)",
        ),
        (
            "Velocity error",
            result.velocity_error_ned_mps,
            result.velocity_mean_mps,
            result.velocity_lower_2sigma_mps,
            result.velocity_upper_2sigma_mps,
            "Error (m/s)",
        ),
        (
            "Attitude error",
            np.rad2deg(result.attitude_error_ned_rad),
            np.rad2deg(result.attitude_mean_rad),
            np.rad2deg(result.attitude_lower_2sigma_rad),
            np.rad2deg(result.attitude_upper_2sigma_rad),
            "Error (deg)",
        ),
    )

    first_axis = None
    for row, (title, runs, mean, lower, upper, ylabel) in enumerate(panel_groups):
        for component in range(3):
            axis = figure.add_subplot(grid[row, component])
            _plot_error_panel(axis, time, runs, mean, lower, upper, component, title, ylabel)
            if row == 2:
                axis.set_xlabel(time_label)
            if first_axis is None:
                first_axis = axis

    trajectory_axis = figure.add_subplot(grid[3, :])
    for run_index in range(result.estimated_position_ned_m.shape[0]):
        trajectory_axis.plot(
            result.estimated_position_ned_m[run_index, :, 1],
            result.estimated_position_ned_m[run_index, :, 0],
            color="tab:blue",
            alpha=0.16,
            linewidth=0.7,
        )
    trajectory_axis.plot(
        result.truth.position_ned[:, 1],
        result.truth.position_ned[:, 0],
        color="black",
        linewidth=1.5,
        label="Truth",
    )
    trajectory_axis.set_title("2-D Trajectory")
    trajectory_axis.set_xlabel("East (m)")
    trajectory_axis.set_ylabel("North (m)")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.grid(True, alpha=0.28)

    handles, labels = first_axis.get_legend_handles_labels()
    first_axis.legend(handles, labels, loc="upper left", fontsize=8, frameon=True)
    aiding = "ideal altitude aiding" if config.altitude_aiding else "no altitude aiding"
    figure.suptitle(
        f"{config.name} — {config.runs} Monte Carlo Runs ({aiding})",
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.04, top=0.92, hspace=0.32, wspace=0.20)
    return figure


def save_results(
    result: MonteCarloResult,
    config: SimulationConfig,
) -> Path:
    """Write the report, resolved configuration, aggregates, and run metadata."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_monte_carlo_report(result, config)
    report.savefig(output_dir / "monte_carlo_summary.pdf", bbox_inches="tight")
    report.savefig(output_dir / "monte_carlo_summary.png", dpi=220, bbox_inches="tight")
    plt.close(report)

    save_config(config, output_dir / "config.toml")
    np.savez_compressed(
        output_dir / "ensemble_statistics.npz",
        time_s=result.truth.time_s,
        position_mean_m=result.position_mean_m,
        position_lower_2sigma_m=result.position_lower_2sigma_m,
        position_upper_2sigma_m=result.position_upper_2sigma_m,
        velocity_mean_mps=result.velocity_mean_mps,
        velocity_lower_2sigma_mps=result.velocity_lower_2sigma_mps,
        velocity_upper_2sigma_mps=result.velocity_upper_2sigma_mps,
        attitude_mean_rad=result.attitude_mean_rad,
        attitude_lower_2sigma_rad=result.attitude_lower_2sigma_rad,
        attitude_upper_2sigma_rad=result.attitude_upper_2sigma_rad,
    )
    metadata = {
        "name": config.name,
        "runs": config.runs,
        "seed": config.seed,
        "sample_period_s": config.sample_period_s,
        "duration_s": config.duration_s,
        "scenario": config.scenario.name,
        "altitude_aiding": config.altitude_aiding,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir
