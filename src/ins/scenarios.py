"""Synthetic INS truth scenarios for Monte Carlo experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import r3f

from ins.config import ScenarioConfig
from ins.coordinates import curvilinear_history_to_geodetic
from ins.earth import somigliana
from ins.mechanization import inverse_mechanize
from ins.trajectories import circular_trajectory, figure_eight_attitude, figure_eight_curvilinear


@dataclass(frozen=True)
class TruthTrajectory:
    """Truth navigation state and ideal IMU data for an INS scenario."""

    time_s: np.ndarray
    position_reference_llh: np.ndarray
    llh: np.ndarray
    velocity_ned: np.ndarray
    rpy: np.ndarray
    Cnb: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    position_ned: np.ndarray


def _origin_llh(config: ScenarioConfig) -> np.ndarray:
    return np.array(
        [
            np.deg2rad(config.origin_lat_deg),
            np.deg2rad(config.origin_lon_deg),
            config.origin_hae_m,
        ]
    )


def _dcm_history(rpy: np.ndarray) -> np.ndarray:
    return np.asarray([r3f.rpy_to_dcm(attitude).T for attitude in rpy])


def _stationary_truth(config: ScenarioConfig, time_s: np.ndarray) -> TruthTrajectory:
    llh0 = _origin_llh(config)
    samples = len(time_s)
    llh = np.tile(llh0, (samples, 1))
    velocity_ned = np.zeros((samples, 3))
    rpy = np.zeros((samples, 3))
    accel = np.tile([0.0, 0.0, -somigliana(llh0[0], llh0[2])], (samples, 1))
    gyro = np.tile(
        [r3f.W_EI * np.cos(llh0[0]), 0.0, -r3f.W_EI * np.sin(llh0[0])],
        (samples, 1),
    )
    return TruthTrajectory(
        time_s=time_s,
        position_reference_llh=llh0,
        llh=llh,
        velocity_ned=velocity_ned,
        rpy=rpy,
        Cnb=_dcm_history(rpy),
        accel=accel,
        gyro=gyro,
        position_ned=np.zeros((samples, 3)),
    )


def _circle_attitude(theta: np.ndarray, config: ScenarioConfig) -> np.ndarray:
    """Create a tangent-heading circular attitude profile.

    The body x-axis is aligned to the horizontal trajectory tangent.  Bank and
    pitch remain explicit scenario inputs rather than imposing a coordinated-
    turn model that may not match the vehicle under study.
    """

    yaw = np.unwrap(np.arctan2(np.cos(theta), -np.sin(theta)))
    return np.column_stack(
        (
            np.full_like(theta, config.bank_rad),
            np.full_like(theta, config.pitch_rad),
            yaw,
        )
    )


def _moving_truth(
    config: ScenarioConfig,
    time_s: np.ndarray,
    sample_period_s: float,
    duration_s: float,
) -> TruthTrajectory:
    theta = np.linspace(0.0, 2.0 * np.pi, len(time_s))
    position_reference_llh = _origin_llh(config)
    if config.name == "figure_eight":
        position_ned = figure_eight_curvilinear(
            theta,
            radius_m=config.radius_m,
            delta_h_m=config.altitude_amplitude_m,
        ).T
        rpy = figure_eight_attitude(theta)
    else:
        position_ned = circular_trajectory(
            theta,
            config.radius_m,
            config.altitude_amplitude_m,
        ).T
        rpy = _circle_attitude(theta, config)

    llh = curvilinear_history_to_geodetic(position_ned, position_reference_llh)
    accel, gyro, velocity_ned = inverse_mechanize(llh, rpy, sample_period_s, duration_s)
    return TruthTrajectory(
        time_s=time_s,
        position_reference_llh=position_reference_llh,
        llh=llh,
        velocity_ned=velocity_ned,
        rpy=rpy,
        Cnb=_dcm_history(rpy),
        accel=accel,
        gyro=gyro,
        position_ned=position_ned,
    )


def build_truth_trajectory(
    config: ScenarioConfig,
    sample_period_s: float,
    duration_s: float,
) -> TruthTrajectory:
    """Build ideal truth data for one named scenario."""

    samples = round(duration_s / sample_period_s) + 1
    time_s = np.arange(samples) * sample_period_s
    if config.name == "stationary":
        return _stationary_truth(config, time_s)
    return _moving_truth(config, time_s, sample_period_s, duration_s)
