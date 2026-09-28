import numpy as np

def rodrigues_rotation(omega, dt):
    """
    Convert a constant angular-rate vector into an incremental rotation matrix.

    This function uses Rodrigues' rotation formula to compute the direction
    cosine matrix corresponding to angular velocity `omega` applied over one
    time step `dt`.

    Parameters
    ----------
    omega : ndarray, shape (3,)
        Angular-rate vector (rad/s).

    dt : float
        Time step (s).

    Returns
    -------
    delta : ndarray, shape (3, 3)
        Incremental rotation matrix over the time interval `dt`.
    """

    # Magnitude of the angular-rate vector.
    omega_norm = np.linalg.norm(omega)

    # For small rotations, use limiting approximations to avoid numerical
    # issues caused by division by very small angular rates.
    if omega_norm * dt < 4e-8:
        k_s = dt
    else:
        k_s = np.sin(omega_norm * dt) / omega_norm

    if omega_norm * dt < 2e-4:
        k_c = 0.5 * dt**2
    else:
        k_c = (1.0 - np.cos(omega_norm * dt)) / omega_norm**2

    # Rodrigues' formula written component-wise for the skew-symmetric
    # angular-rate matrix.
    delta_1 = np.array([
        1.0 - k_c * (omega[1]**2 + omega[2]**2),
        k_c * omega[0] * omega[1] + k_s * omega[2],
        k_c * omega[0] * omega[2] - k_s * omega[1],
    ])

    delta_2 = np.array([
        k_c * omega[0] * omega[1] - k_s * omega[2],
        1.0 - k_c * (omega[0]**2 + omega[2]**2),
        k_c * omega[1] * omega[2] + k_s * omega[0],
    ])

    delta_3 = np.array([
        k_c * omega[0] * omega[2] + k_s * omega[1],
        k_c * omega[1] * omega[2] - k_s * omega[0],
        1.0 - k_c * (omega[0]**2 + omega[1]**2),
    ])

    return np.column_stack((delta_1, delta_2, delta_3))


def orthonormalize_dcm(Cnb):
    """
    Orthonormalize a direction cosine matrix (DCM) using the Modified
    Gram-Schmidt (MGS) algorithm.

    Numerical integration and finite-precision arithmetic cause the columns
    of a propagated DCM to gradually lose orthogonality and unit length.
    Left uncorrected, this numerical drift can eventually produce an invalid
    rotation matrix.

    This function applies Modified Gram-Schmidt orthonormalization to
    restore the defining properties of a DCM:

    - orthogonal columns,
    - unit-length columns,
    - and a valid rotation matrix representation.

    This operation is commonly performed after attitude propagation in
    strapdown inertial navigation systems to prevent accumulated numerical
    drift from violating DCM orthogonality.

    Parameters
    ----------
    Cnb : ndarray, shape (3, 3)
        Body-to-navigation direction cosine matrix.

    Returns
    -------
    Cnb : ndarray, shape (3, 3)
        Orthonormalized body-to-navigation DCM.
    """

    # Remove the component of column 1 along column 0.
    Cnb[:, 1] = Cnb[:, 1] - Cnb[:, 0].dot(Cnb[:, 1]) * Cnb[:, 0]

    # Remove the component of column 2 along column 0.
    Cnb[:, 2] = Cnb[:, 2] - Cnb[:, 0].dot(Cnb[:, 2]) * Cnb[:, 0]

    # Remove the component of column 2 along column 1.
    Cnb[:, 2] = Cnb[:, 2] - Cnb[:, 1].dot(Cnb[:, 2]) * Cnb[:, 1]

    # Normalize each basis vector to unit length.
    Cnb[:, 0] /= np.linalg.norm(Cnb[:, 0])
    Cnb[:, 1] /= np.linalg.norm(Cnb[:, 1])
    Cnb[:, 2] /= np.linalg.norm(Cnb[:, 2])

    return Cnb