import numpy as np
import r3f

def circular_trajectory(theta, R, delta_h):
    """
    Generate a large circular aircraft trajectory in a local curvilinear frame.

    The trajectory is parameterized such that:

    - the horizontal motion follows a circular path,
    - the aircraft returns to its starting position after one revolution,
    - and the altitude varies smoothly along the path.

    The trajectory equations are:

    x_c = R * (1 + cos(theta))
    y_c = R * sin(theta)
    z_c = -(cos(theta) + 1) * (delta_h / 2)

    where:

    - R controls the horizontal trajectory radius,
    - delta_h controls the peak altitude variation,
    - and theta parameterizes the trajectory angle.

    Parameters
    ----------
    theta : ndarray
        Trajectory parameter vector (rad).

    R : float
        Circular trajectory radius (m).

    delta_h : float
        Peak altitude variation above the trajectory origin (m).

    Returns
    -------
    curvilinear_traj : ndarray, shape (3, N)
        Curvilinear trajectory:
            [x_c,
             y_c,
             z_c]
    """

    # Horizontal circular motion in the local curvilinear frame.
    x_curv = R * (1 + np.cos(theta))
    y_curv = R * np.sin(theta)

    # Smooth altitude variation along the trajectory.
    #
    # Negative z values correspond to upward motion because the local
    # curvilinear frame defines positive vertical motion downward.
    z_curv = -(np.cos(theta) + 1) * (delta_h / 2)

    # Stack trajectory components into a single (3, N) trajectory matrix.
    curvilinear_traj = np.vstack((x_curv, y_curv, z_curv))

    return curvilinear_traj


def figure_eight_curvilinear(theta, radius_m=1000.0, delta_h_m=100.0):
    """
    Generate a figure-eight aircraft trajectory in a local curvilinear frame.

    The trajectory is designed to provide a smooth maneuvering flight path
    containing:

    - left and right turns,
    - changing heading,
    - horizontal acceleration,
    - and altitude variation.

    This trajectory is used throughout the repository as a truth path for:

    - inverse mechanization,
    - forward mechanization,
    - sensor-noise studies,
    - and INS error analysis.

    Parameters
    ----------
    theta : ndarray
        Trajectory parameter vector (rad).

    radius_m : float, optional
        Characteristic horizontal trajectory scale (m).

    delta_h_m : float, optional
        Peak altitude variation along the trajectory (m).

    Returns
    -------
    xyz_curvilinear : ndarray, shape (3, N)
        Curvilinear trajectory coordinates:

            x_c : north displacement (m)
            y_c : east displacement (m)
            z_c : down displacement (m)

        expressed in the local curvilinear frame.
    """

    # Generate a horizontal figure-eight path.
    #
    # The x-component oscillates at twice the frequency of the y-component,
    # creating the characteristic figure-eight geometry.
    x_c = radius_m / 4.0 * np.sin(2.0 * theta)
    y_c = radius_m * (np.cos(theta) - 1.0)

    # Add smooth altitude variation along the trajectory.
    #
    # Positive z is downward in the local curvilinear frame.
    z_c = delta_h_m / 2.0 * (np.cos(theta) - 1.0)

    return np.vstack((x_c, y_c, z_c))


def figure_eight_attitude(theta):
    """
    Generate a smooth aircraft attitude profile for the figure-eight trajectory.

    The attitude profile is designed to produce realistic aircraft motion
    while avoiding abrupt changes in orientation that would generate
    unrealistic IMU measurements.

    The resulting roll, pitch, and yaw histories are used throughout the
    repository as truth attitude inputs for:

    - inverse mechanization,
    - forward mechanization,
    - sensor-noise studies,
    - and INS error analysis.

    Parameters
    ----------
    theta : ndarray
        Trajectory parameter vector (rad).

    Returns
    -------
    rpy : ndarray, shape (N, 3)
        Aircraft attitude history.

        Columns are:

            roll  (rad)
            pitch (rad)
            yaw   (rad)
    """

    # Small roll oscillations approximate the aircraft banking as it
    # transitions through the figure-eight maneuver.
    roll = -1.0 / 27.0 * np.cos(theta)

    # Pitch oscillations introduce gentle climb and descent behavior
    # throughout the trajectory.
    pitch = (
        -1.0 / 48.0 * np.sin(2.0 * theta)
        + 1.0 / 143.0 * np.sin(4.0 * theta)
    )

    # Yaw follows the direction of travel along the figure-eight path,
    # producing continuous heading changes throughout the maneuver.
    yaw = np.arctan2(
        -2.0 * np.sin(theta),
        np.cos(2.0 * theta)
    )

    return np.vstack((roll, pitch, yaw)).T