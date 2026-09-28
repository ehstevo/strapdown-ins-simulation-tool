import numpy as np

from ins.metrics import dcm_error_vector_ned, ensemble_statistics
from ins.rotations import rodrigues_rotation


def test_dcm_error_vector_returns_zero_for_identical_attitudes() -> None:
    identity = np.eye(3)[np.newaxis, :, :]

    error = dcm_error_vector_ned(identity, identity)

    np.testing.assert_allclose(error, np.zeros((1, 3)), atol=1e-15)


def test_dcm_error_vector_recovers_small_north_rotation() -> None:
    estimate = rodrigues_rotation(np.array([0.01, 0.0, 0.0]), 1.0)[np.newaxis, :, :]
    truth = np.eye(3)[np.newaxis, :, :]

    error = dcm_error_vector_ned(estimate, truth)

    np.testing.assert_allclose(error, [[0.01, 0.0, 0.0]], atol=1e-12)


def test_ensemble_statistics_use_sample_standard_deviation() -> None:
    values = np.array([[[0.0]], [[2.0]]])

    mean, lower, upper = ensemble_statistics(values)

    np.testing.assert_allclose(mean, [[1.0]])
    np.testing.assert_allclose(lower, [[1.0 - 2.0 * np.sqrt(2.0)]])
    np.testing.assert_allclose(upper, [[1.0 + 2.0 * np.sqrt(2.0)]])
