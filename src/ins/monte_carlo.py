"""Monte Carlo orchestration for strapdown INS error-propagation studies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ins.config import SimulationConfig
from ins.coordinates import curvilinear_history_to_geodetic, geodetic_history_to_curvilinear
from ins.mechanization import forward_mechanize
from ins.metrics import dcm_error_vector_ned, ensemble_statistics
from ins.rotations import rodrigues_rotation
from ins.scenarios import TruthTrajectory, build_truth_trajectory
from ins.noise import generate_imu_error


@dataclass(frozen=True)
class MonteCarloResult:
    """Full ensemble and aggregate statistics for one simulation."""

    truth: TruthTrajectory
    position_error_ned_m: np.ndarray
    velocity_error_ned_mps: np.ndarray
    attitude_error_ned_rad: np.ndarray
    estimated_position_ned_m: np.ndarray
    position_mean_m: np.ndarray
    position_lower_2sigma_m: np.ndarray
    position_upper_2sigma_m: np.ndarray
    velocity_mean_mps: np.ndarray
    velocity_lower_2sigma_mps: np.ndarray
    velocity_upper_2sigma_mps: np.ndarray
    attitude_mean_rad: np.ndarray
    attitude_lower_2sigma_rad: np.ndarray
    attitude_upper_2sigma_rad: np.ndarray


def _dcm_to_rpy(Cnb: np.ndarray) -> np.ndarray:
    """Convert a body-to-navigation DCM to roll, pitch, yaw angles."""

    return np.array(
        [
            np.arctan2(Cnb[2, 1], Cnb[2, 2]),
            -np.arcsin(np.clip(Cnb[2, 0], -1.0, 1.0)),
            np.arctan2(Cnb[1, 0], Cnb[0, 0]),
        ]
    )


def _initial_state(
    config: SimulationConfig,
    truth: TruthTrajectory,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample and construct one perturbed initial navigation state."""

    position_error_ned_m = rng.normal(
        config.initial_error.position.mean,
        config.initial_error.position.std,
    )
    initial_llh = curvilinear_history_to_geodetic(position_error_ned_m[np.newaxis, :], truth.llh[0])[0]
    velocity_error_ned_mps = rng.normal(
        config.initial_error.velocity.mean,
        config.initial_error.velocity.std,
    )
    initial_velocity = truth.velocity_ned[0] + velocity_error_ned_mps

    attitude_error_ned = rng.normal(
        config.initial_error.attitude_ned_rad.mean,
        config.initial_error.attitude_ned_rad.std,
    )
    initial_Cnb = rodrigues_rotation(attitude_error_ned, 1.0) @ truth.Cnb[0]
    initial_rpy = _dcm_to_rpy(initial_Cnb)
    return initial_llh, initial_velocity, initial_rpy


def _run_one(
    config: SimulationConfig,
    truth: TruthTrajectory,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one sampled INS realization and return navigation errors."""

    accel_error = generate_imu_error(
        config.accel,
        len(truth.time_s),
        config.sample_period_s,
        rng,
    )
    gyro_error = generate_imu_error(
        config.gyro,
        len(truth.time_s),
        config.sample_period_s,
        rng,
    )
    initial_llh, initial_velocity, initial_rpy = _initial_state(config, truth, rng)

    llh_est, velocity_est, _, Cnb_est = forward_mechanize(
        truth.accel + accel_error.total,
        truth.gyro + gyro_error.total,
        truth.velocity_ned,
        truth.llh,
        initial_rpy,
        config.sample_period_s,
        initial_llh=initial_llh,
        initial_velocity_ned=initial_velocity,
        altitude_aiding=config.altitude_aiding,
    )
    estimated_position_ned = geodetic_history_to_curvilinear(
        llh_est,
        truth.position_reference_llh,
    )
    position_error = estimated_position_ned - truth.position_ned
    velocity_error = velocity_est - truth.velocity_ned
    attitude_error = dcm_error_vector_ned(Cnb_est, truth.Cnb)
    return position_error, velocity_error, attitude_error, estimated_position_ned


def run_monte_carlo(
    config: SimulationConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MonteCarloResult:
    """Generate truth, propagate all runs, and calculate ensemble statistics.

    ``progress_callback`` receives the completed and total run counts after
    each realization. It is optional so the CLI and tests retain their current
    behavior while interactive frontends can report progress.
    """

    truth = build_truth_trajectory(
        config.scenario,
        config.sample_period_s,
        config.duration_s,
    )
    samples = len(truth.time_s)
    position_error = np.empty((config.runs, samples, 3))
    velocity_error = np.empty((config.runs, samples, 3))
    attitude_error = np.empty((config.runs, samples, 3))
    estimated_position = np.empty((config.runs, samples, 3))
    rng = np.random.default_rng(config.seed)

    for run_index in range(config.runs):
        (
            position_error[run_index],
            velocity_error[run_index],
            attitude_error[run_index],
            estimated_position[run_index],
        ) = _run_one(config, truth, rng)
        if progress_callback is not None:
            progress_callback(run_index + 1, config.runs)

    position_mean, position_lower, position_upper = ensemble_statistics(position_error)
    velocity_mean, velocity_lower, velocity_upper = ensemble_statistics(velocity_error)
    attitude_mean, attitude_lower, attitude_upper = ensemble_statistics(attitude_error)
    return MonteCarloResult(
        truth=truth,
        position_error_ned_m=position_error,
        velocity_error_ned_mps=velocity_error,
        attitude_error_ned_rad=attitude_error,
        estimated_position_ned_m=estimated_position,
        position_mean_m=position_mean,
        position_lower_2sigma_m=position_lower,
        position_upper_2sigma_m=position_upper,
        velocity_mean_mps=velocity_mean,
        velocity_lower_2sigma_mps=velocity_lower,
        velocity_upper_2sigma_mps=velocity_upper,
        attitude_mean_rad=attitude_mean,
        attitude_lower_2sigma_rad=attitude_lower,
        attitude_upper_2sigma_rad=attitude_upper,
    )
