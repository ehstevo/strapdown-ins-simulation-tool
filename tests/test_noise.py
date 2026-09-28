import numpy as np

from ins.config import GaussianVector, SensorErrorModel
from ins.noise import generate_imu_error, generate_stationary_fogm


def _zero_stochastic_model() -> SensorErrorModel:
    return SensorErrorModel(
        constant_bias=GaussianVector(
            mean=np.array([1.0, -2.0, 3.0]),
            std=np.zeros(3),
        ),
        fogm_sigma=np.zeros(3),
        fogm_tau_s=np.ones(3),
        white_measurement_noise_density=np.zeros(3),
        bias_rate_random_walk_intensity=np.zeros(3),
    )


def test_imu_error_combines_deterministic_constant_bias() -> None:
    realization = generate_imu_error(
        _zero_stochastic_model(),
        samples=4,
        dt=0.1,
        rng=np.random.default_rng(4),
    )

    np.testing.assert_allclose(realization.total, [[1.0, -2.0, 3.0]] * 4)
    np.testing.assert_allclose(realization.fogm_bias, 0.0)
    np.testing.assert_allclose(realization.white_measurement_noise, 0.0)
    np.testing.assert_allclose(realization.bias_rate_random_walk, 0.0)


def test_stationary_fogm_has_requested_long_run_standard_deviation() -> None:
    sigma = np.array([0.4, 0.4, 0.4])
    process = generate_stationary_fogm(
        sigma=sigma,
        tau_s=np.array([1.0, 1.0, 1.0]),
        samples=40_000,
        dt=0.05,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(np.mean(process, axis=0), 0.0, atol=0.035)
    np.testing.assert_allclose(np.std(process, axis=0), sigma, rtol=0.06)
