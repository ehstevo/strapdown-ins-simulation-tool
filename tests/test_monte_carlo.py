from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("r3f")

from ins.config import GaussianVector, SensorErrorModel, load_config
from ins.monte_carlo import run_monte_carlo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _perfect_sensor() -> SensorErrorModel:
    return SensorErrorModel(
        constant_bias=GaussianVector(mean=np.zeros(3), std=np.zeros(3)),
        fogm_sigma=np.zeros(3),
        fogm_tau_s=np.ones(3),
        white_measurement_noise_density=np.zeros(3),
        bias_rate_random_walk_intensity=np.zeros(3),
    )


def test_stationary_perfect_ins_remains_at_truth() -> None:
    base = load_config(REPOSITORY_ROOT / "configs" / "stationary_schuler.toml")
    no_error = GaussianVector(mean=np.zeros(3), std=np.zeros(3))
    config = replace(
        base,
        runs=2,
        duration_s=2.0,
        initial_error=replace(
            base.initial_error,
            position=no_error,
            velocity=no_error,
            attitude_ned_rad=no_error,
        ),
        gyro=_perfect_sensor(),
        accel=_perfect_sensor(),
    )

    result = run_monte_carlo(config)

    np.testing.assert_allclose(result.position_error_ned_m, 0.0, atol=1e-8)
    np.testing.assert_allclose(result.velocity_error_ned_mps, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.attitude_error_ned_rad, 0.0, atol=1e-10)


def test_circle_perfect_ins_remains_at_truth_in_scenario_reference_frame() -> None:
    base = load_config(REPOSITORY_ROOT / "configs" / "circle.toml")
    no_error = GaussianVector(mean=np.zeros(3), std=np.zeros(3))
    config = replace(
        base,
        runs=1,
        duration_s=2.0,
        initial_error=replace(
            base.initial_error,
            position=no_error,
            velocity=no_error,
            attitude_ned_rad=no_error,
        ),
        gyro=_perfect_sensor(),
        accel=_perfect_sensor(),
    )

    result = run_monte_carlo(config)

    np.testing.assert_allclose(result.position_error_ned_m, 0.0, atol=1e-8)
    np.testing.assert_allclose(result.velocity_error_ned_mps, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.attitude_error_ned_rad, 0.0, atol=1e-10)
