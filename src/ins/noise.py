"""IMU stochastic error processes for Monte Carlo simulation.

The Monte Carlo simulator applies the following measurement-error model to
each gyro and accelerometer body axis:

``error[k] = constant_bias + fogm_bias[k] + white_noise[k] + bias_rrw[k]``

``constant_bias`` is sampled once per Monte Carlo run and is then held fixed.
``fogm_bias`` is a stationary first-order Gauss-Markov process representing a
finite-correlation bias drift. ``white_noise`` is gyro angle random walk (ARW)
or accelerometer velocity random walk (VRW). ``bias_rrw`` is a nonstationary
bias rate random walk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ins.config import GaussianVector, SensorErrorModel


@dataclass(frozen=True)
class ImuErrorRealization:
    """One sampled three-axis IMU measurement-error history.

    All arrays have shape ``(samples, 3)`` and are expressed in the sensor's
    body axes. ``total`` is the error added to ideal IMU measurements.
    """

    total: np.ndarray
    constant_bias: np.ndarray
    fogm_bias: np.ndarray
    white_measurement_noise: np.ndarray
    bias_rate_random_walk: np.ndarray


def _sample_gaussian_vector(
    distribution: GaussianVector,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one independent three-axis sample from a configured distribution."""

    return rng.normal(distribution.mean, distribution.std)


def generate_stationary_fogm(
    sigma: np.ndarray,
    tau_s: np.ndarray,
    samples: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stationary first-order Gauss-Markov bias drift.

    Each axis follows

    ``b[k + 1] = phi*b[k] + sigma*sqrt(1 - phi^2)*w[k]``

    where ``phi = exp(-dt/tau_s)`` and ``w[k]`` is standard normal. The
    initial state is sampled from ``N(0, sigma^2)`` so the requested
    steady-state standard deviation applies from the first sample.
    """

    phi = np.exp(-dt / tau_s)
    innovation_std = sigma * np.sqrt(1.0 - phi**2)
    fogm_bias = np.empty((samples, 3))
    fogm_bias[0] = rng.normal(0.0, sigma)

    for sample_index in range(1, samples):
        fogm_bias[sample_index] = (
            phi * fogm_bias[sample_index - 1]
            + rng.normal(0.0, innovation_std)
        )

    return fogm_bias


def generate_white_measurement_noise(
    density: np.ndarray,
    samples: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate ARW/VRW white measurement noise for three IMU axes.

    The discrete sample standard deviation is ``density / sqrt(dt)``. For a
    gyro, density is angle random walk in ``rad/sqrt(s)``. For an
    accelerometer, density is velocity random walk in ``m/s/sqrt(s)``.
    """

    return rng.normal(0.0, density / np.sqrt(dt), size=(samples, 3))


def generate_bias_rate_random_walk(
    intensity: np.ndarray,
    samples: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a nonstationary bias rate random walk for three IMU axes.

    The process starts at zero and follows

    ``b[k + 1] = b[k] + intensity*sqrt(dt)*w[k]``.

    For a gyro, intensity has units ``rad/s/sqrt(s)``. For an accelerometer,
    it has units ``m/s^2/sqrt(s)``.
    """

    increments = rng.normal(0.0, intensity * np.sqrt(dt), size=(samples, 3))
    return np.cumsum(increments, axis=0)


def generate_imu_error(
    model: SensorErrorModel,
    samples: int,
    dt: float,
    rng: np.random.Generator,
) -> ImuErrorRealization:
    """Generate one complete three-axis IMU error realization.

    A new constant bias is drawn for each call, which corresponds to one
    Monte Carlo run. The remaining terms evolve sample by sample.
    """

    constant_bias_vector = _sample_gaussian_vector(model.constant_bias, rng)
    constant_bias = np.broadcast_to(constant_bias_vector, (samples, 3)).copy()
    fogm_bias = generate_stationary_fogm(model.fogm_sigma, model.fogm_tau_s, samples, dt, rng)
    white_measurement_noise = generate_white_measurement_noise(
        model.white_measurement_noise_density,
        samples,
        dt,
        rng,
    )
    bias_rate_random_walk = generate_bias_rate_random_walk(
        model.bias_rate_random_walk_intensity,
        samples,
        dt,
        rng,
    )

    return ImuErrorRealization(
        total=constant_bias + fogm_bias + white_measurement_noise + bias_rate_random_walk,
        constant_bias=constant_bias,
        fogm_bias=fogm_bias,
        white_measurement_noise=white_measurement_noise,
        bias_rate_random_walk=bias_rate_random_walk,
    )
