"""Navigation error metrics for Monte Carlo INS analyses."""

from __future__ import annotations

import numpy as np


def dcm_error_vector_ned(Cnb_est: np.ndarray, Cnb_truth: np.ndarray) -> np.ndarray:
    """Return the DCM-derived attitude error vector resolved in NED.

    The error DCM ``C_nb_est @ C_nb_truth.T`` maps a truth-navigation vector
    into the estimated-navigation frame.  Its rotation-vector logarithm gives
    the three-axis small-angle attitude error in navigation coordinates.
    """

    if Cnb_est.shape != Cnb_truth.shape or Cnb_est.shape[-2:] != (3, 3):
        raise ValueError("Both DCM histories must have matching shape (..., 3, 3).")

    C_error = Cnb_est @ np.swapaxes(Cnb_truth, -1, -2)
    vector = np.stack(
        (
            C_error[..., 2, 1] - C_error[..., 1, 2],
            C_error[..., 0, 2] - C_error[..., 2, 0],
            C_error[..., 1, 0] - C_error[..., 0, 1],
        ),
        axis=-1,
    )
    sine = 0.5 * np.linalg.norm(vector, axis=-1)
    cosine = np.clip(0.5 * (np.trace(C_error, axis1=-2, axis2=-1) - 1.0), -1.0, 1.0)
    angle = np.arctan2(sine, cosine)

    scale = np.ones_like(angle)
    nonzero = sine > 1e-12
    scale[nonzero] = angle[nonzero] / (2.0 * sine[nonzero])
    return vector * scale[..., np.newaxis]


def ensemble_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ensemble mean and mean-plus/minus two sample deviations."""

    if values.ndim < 2:
        raise ValueError("Monte Carlo values must have a leading run dimension.")

    mean = np.mean(values, axis=0)
    if values.shape[0] == 1:
        std = np.zeros_like(mean)
    else:
        std = np.std(values, axis=0, ddof=1)
    return mean, mean - 2.0 * std, mean + 2.0 * std
