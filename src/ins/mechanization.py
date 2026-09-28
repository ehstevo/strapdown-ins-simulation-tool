import numpy as np
import r3f
from ins.earth import somigliana
from ins.rotations import rodrigues_rotation, orthonormalize_dcm

def inverse_mechanize(llh_t, rpy_t, T, t_dur):
    """
    Generate ideal IMU measurements from a prescribed trajectory and attitude profile.

    This function performs inverse inertial mechanization. Instead of using
    accelerometer and gyroscope measurements to estimate position, velocity,
    and attitude, it starts from a known trajectory and attitude history and
    computes the ideal IMU signals that would produce that motion.

    Parameters
    ----------
    llh_t : ndarray, shape (K, 3)
        Truth geodetic trajectory.

        Columns are:
            latitude  (rad)
            longitude (rad)
            height above ellipsoid (m)

    rpy_t : ndarray, shape (K, 3)
        Truth attitude trajectory.

        Columns are:
            roll  (rad)
            pitch (rad)
            yaw   (rad)

    T : float
        Sampling period (s).

    t_dur : float
        Total simulation duration (s).

    Returns
    -------
    fbbi_t : ndarray, shape (K, 3)
        Ideal body-frame specific force measurements (m/s^2).

    wbbi_t : ndarray, shape (K, 3)
        Ideal body-frame angular-rate measurements (rad/s).

    vne_t : ndarray, shape (K, 3)
        Navigation-frame velocity truth (m/s).
    """

    def extrapolate_two(y):
        """
        Extrapolate two additional samples using the final three samples.

        The inverse mechanization loop uses current, next-step, and
        next-next-step quantities to compute finite-difference derivatives.
        Two extrapolated samples avoid shortening the returned trajectory.
        """
        if y.ndim == 1:
            y_1 = 3 * y[-1] - 3 * y[-2] + y[-3]
            y_2 = 6 * y[-1] - 8 * y[-2] + 3 * y[-3]
            z = np.concatenate((y, y_1, y_2))

        elif y.ndim == 2:
            y_1 = 3 * y[-1, :] - 3 * y[-2, :] + y[-3, :]
            y_2 = 6 * y[-1, :] - 8 * y[-2, :] + 3 * y[-3, :]
            z = np.vstack((y, y_1, y_2))

        return z

    # Number of output samples.
    #
    # One additional sample is included so the trajectory contains both:
    #
    # - the initial state at t = 0
    # - and the final state at t = t_dur
    K = round(t_dur / T) + 1

    # Allocate output arrays.
    fbbi_t = np.zeros((K, 3))   # ideal accelerometer measurements
    wbbi_t = np.zeros((K, 3))   # ideal gyroscope measurements
    vne_t = np.zeros((K, 3))    # reconstructed navigation-frame velocity

    # Add two extrapolated samples so finite differences can be evaluated
    # through the final requested output time.
    llh_t = extrapolate_two(llh_t)
    rpy_t = extrapolate_two(rpy_t)

    for i in range(K):
        # Current and next-step body-to-navigation DCMs.
        #
        # These are used to determine how the body frame rotates relative
        # to the navigation frame over one sample interval.
        C_nb = r3f.rpy_to_dcm(rpy_t[i]).T
        C_nbp = r3f.rpy_to_dcm(rpy_t[i + 1]).T

        # Incremental attitude change from the current body frame to the
        # next body frame, expressed through the navigation-frame DCMs.
        S = C_nb.T @ C_nbp

        # Recover the principal rotation angle from the incremental
        # rotation matrix.
        q = np.trace(S)
        ang = np.arccos((q - 1) / 2)

        # Numerically stable scale factor used to convert the
        # skew-symmetric part of S into a rotation vector.
        #
        # The polynomial branch avoids numerical issues for very small
        # incremental rotations, where the standard expression becomes
        # poorly conditioned.
        if q <= 2.9995:
            s = ang / np.sqrt(3 + 2 * q - q**2)
        else:
            s = (q**2 - 11 * q + 54) / 60

        # Body angular rate relative to the navigation frame, expressed
        # in the body frame.
        wbbn = (s / T) * np.array([
            S[2, 1] - S[1, 2],
            S[0, 2] - S[2, 0],
            S[1, 0] - S[0, 1],
        ])

        # Finite-difference geodetic rates at the current and next step.
        #
        # These are used to compute transport rates and reconstruct
        # navigation-frame velocity.
        d_llh = (llh_t[i + 1] - llh_t[i]) / T
        d_llh_p = (llh_t[i + 2] - llh_t[i + 1]) / T

        # Transport rate of the navigation frame with respect to the Earth.
        #
        # The local navigation frame rotates as the vehicle moves across
        # the curved Earth surface.
        wnne = np.array([
            np.cos(llh_t[i, 0]) * d_llh[1],
            -d_llh[0],
            -np.sin(llh_t[i, 0]) * d_llh[1],
        ])

        wnne_p = np.array([
            np.cos(llh_t[i + 1, 0]) * d_llh_p[1],
            -d_llh_p[0],
            -np.sin(llh_t[i + 1, 0]) * d_llh_p[1],
        ])

        # Earth rotation rate resolved in the local navigation frame.
        wnei = np.array([
            r3f.W_EI * np.cos(llh_t[i, 0]),
            0,
            -r3f.W_EI * np.sin(llh_t[i, 0]),
        ])

        # Inverse gyroscope equation.
        #
        # The gyro measures body angular rate with respect to inertial
        # space, expressed in the body frame. This includes:
        #
        # - body rotation relative to the navigation frame
        # - navigation-frame transport rate
        # - Earth rotation rate
        wbbi_t[i] = wbbn + C_nb.T @ (wnne + wnei)

        # Earth curvature radii at the current step.
        #
        # These convert angular latitude/longitude rates into linear
        # navigation-frame velocity components.
        klat = np.sqrt(1 - r3f.E2 * np.sin(llh_t[i, 0])**2)
        Rm = (r3f.A_E / klat**3) * (1 - r3f.E2)
        Rt = r3f.A_E / klat

        # Reconstruct navigation-frame velocity in NED coordinates.
        vne_t[i] = np.array([
            -wnne[1] * (Rm + llh_t[i, 2]),
            wnne[0] * (Rt + llh_t[i, 2]),
            -d_llh[2],
        ])

        # Earth curvature radii at the next step.
        #
        # These are needed to compute the next-step navigation velocity
        # and therefore approximate acceleration.
        klat_p = np.sqrt(1 - r3f.E2 * np.sin(llh_t[i + 1, 0])**2)
        Rm_p = (r3f.A_E / klat_p**3) * (1 - r3f.E2)
        Rt_p = r3f.A_E / klat_p

        # Reconstruct next-step navigation-frame velocity.
        vne_p = np.array([
            -wnne_p[1] * (Rm_p + llh_t[i + 1, 2]),
            wnne_p[0] * (Rt_p + llh_t[i + 1, 2]),
            -d_llh_p[2],
        ])

        # Approximate kinematic acceleration in the navigation frame.
        d_vne = (vne_p - vne_t[i]) / T

        # Gravity magnitude from the Somigliana model, resolved in the
        # local navigation frame.
        gamnl = somigliana(llh_t[i, 0], llh_t[i, 2])
        gamnl_ = np.array([0, 0, gamnl])

        # Inverse accelerometer equation.
        #
        # Accelerometers measure specific force, not gravity directly.
        # Therefore, the ideal body-frame accelerometer measurement is:
        #
        # navigation acceleration
        # + Coriolis/transport correction
        # - gravity
        #
        # rotated from navigation coordinates into body coordinates.
        fbbi_t[i] = C_nb.T @ (
            d_vne
            + np.cross(2 * wnei + wnne, vne_t[i])
            - gamnl_
        )

    return fbbi_t, wbbi_t, vne_t


def forward_mechanize(
    fbbi_t,
    wbbi_t,
    vne_t,
    llh_t,
    rpy0,
    dt,
    *,
    initial_llh=None,
    initial_velocity_ned=None,
    altitude_aiding=True,
):
    """
    Propagate a strapdown inertial navigation solution from IMU measurements.

    This function performs forward inertial mechanization. Starting from an
    initial geodetic position, velocity, and attitude, it integrates ideal or
    measured accelerometer and gyroscope data to estimate the vehicle's
    position, velocity, and attitude over time.

    The propagation includes:

    - Earth rotation,
    - navigation-frame transport rate,
    - Earth curvature,
    - Somigliana gravity,
    - Coriolis/transport corrections,
    - DCM attitude propagation,
    - DCM orthonormalization,
    - and ideal altitude aiding.

    Altitude Aiding
    ---------------
    The full truth geodetic trajectory `llh_t` is passed into this function,
    but only the height-above-ellipsoid component is used as an external
    altitude measurement during propagation.

    Conceptually, this acts like an ideal barometric altitude measurement.
    The truth latitude and longitude are not used to correct the horizontal
    navigation solution.

    Parameters
    ----------
    fbbi_t : ndarray, shape (K, 3)
        Body-frame specific force measurements from the accelerometers
        (m/s^2).

    wbbi_t : ndarray, shape (K, 3)
        Body-frame angular-rate measurements from the gyroscopes (rad/s).

    vne_t : ndarray, shape (K, 3)
        Navigation-frame velocity truth used only for initialization.

        The first sample provides the initial velocity. The remaining samples
        are not used as velocity measurements during propagation.

    llh_t : ndarray, shape (K, 3)
        Geodetic trajectory used for initialization and ideal altitude aiding.

        Columns are:
            latitude  (rad)
            longitude (rad)
            height above ellipsoid (m)

        By default, the first row provides the initial position. During
        propagation, only llh_t[i, 2] is used as an altitude aiding
        measurement.

    rpy0 : ndarray, shape (3,)
        Initial roll, pitch, and yaw attitude (rad).

    dt : float
        Sampling period (s).

    initial_llh : ndarray, shape (3,), optional
        Initial geodetic position. When omitted, ``llh_t[0]`` is used. This
        permits initial position-error simulations while retaining the truth
        altitude history as the external altitude measurement.

    initial_velocity_ned : ndarray, shape (3,), optional
        Initial navigation-frame velocity. When omitted, ``vne_t[0]`` is
        used.

    altitude_aiding : bool, optional
        Apply the existing ideal altitude-aiding feedback. Defaults to true
        for backward-compatible behavior. When false, the mechanization is
        fully inertial in the vertical channel.

    Returns
    -------
    llh_est : ndarray, shape (K, 3)
        Estimated geodetic position trajectory.

    vne_est : ndarray, shape (K, 3)
        Estimated navigation-frame velocity trajectory.

    rpy_est : ndarray, shape (K, 3)
        Estimated roll, pitch, and yaw trajectory.

    Cnb_est : ndarray, shape (K, 3, 3)
        Estimated body-to-navigation DCM trajectory.
    """

    K = fbbi_t.shape[0]

    # Altitude-aiding gains.
    #
    # These gains stabilize the vertical channel using an external altitude
    # measurement, analogous to an ideal barometric altimeter.
    lam = 0.01
    C1 = 3 * lam
    C2 = 4 * lam**2
    C3 = 2 * lam**3
    eI = 0.0

    # Allocate output arrays for the estimated navigation solution.
    llh_est = np.zeros((K, 3))
    vne_est = np.zeros((K, 3))
    rpy_est = np.zeros((K, 3))
    Cnb_est = np.zeros((K, 3, 3))

    # Initialize navigation state from the known initial conditions.
    llh = llh_t[0, :].copy() if initial_llh is None else np.asarray(initial_llh, dtype=float).copy()
    rpy = rpy0.copy()
    vne = vne_t[0, :].copy() if initial_velocity_ned is None else np.asarray(initial_velocity_ned, dtype=float).copy()

    # Initialize the body-to-navigation attitude matrix.
    Cnb = r3f.rpy_to_dcm(rpy).T

    def mech_step(fbbi, wbbi, dt, llh, vne, Cnb, hB, eI):
        """
        Propagate the navigation state forward by one time step.
        """

        # Earth rotation rate resolved in the navigation frame.
        wnei = np.array([
            r3f.W_EI * np.cos(llh[0]),
            0.0,
            -r3f.W_EI * np.sin(llh[0]),
        ])

        # Local Earth curvature radii.
        #
        # These convert between geodetic rates and linear navigation-frame
        # velocity components.
        klat = np.sqrt(1.0 - r3f.E2 * np.sin(llh[0])**2)
        Rm = (r3f.A_E / klat**3) * (1.0 - r3f.E2)
        Rt = r3f.A_E / klat

        # Navigation-frame transport rate.
        #
        # The local navigation frame rotates relative to the Earth as the
        # vehicle moves across the curved Earth surface.
        wnne = np.array([
            vne[1] / (Rt + llh[2]),
            -vne[0] / (Rm + llh[2]),
            -vne[1] * np.tan(llh[0]) / (Rt + llh[2]),
        ])

        # Geodetic position rate.
        #
        # Latitude and longitude rates are obtained from navigation velocity
        # and local curvature. Down velocity maps directly to height rate.
        d_llh = np.array([
            -wnne[1],
            wnne[0] / np.cos(llh[0]),
            -vne[2],
        ])

        # Altitude aiding error.
        #
        # hB acts as an external altitude measurement. In this demo it is
        # ideal, but it represents the role of a barometric altimeter.
        e = llh[2] - hB if altitude_aiding else 0.0
        if altitude_aiding:
            d_llh[2] -= C1 * e

        # Body angular rate relative to the navigation frame.
        #
        # Gyroscopes measure body angular rate relative to inertial space.
        # To propagate attitude relative to the navigation frame, remove
        # Earth rotation and navigation-frame transport rate.
        wbbn = wbbi - Cnb.T @ (wnne + wnei)

        # Local gravity resolved in the navigation frame.
        gamnl = somigliana(llh[0], llh[2])
        gamnl_ = np.array([0.0, 0.0, gamnl])

        # Navigation-frame velocity derivative.
        #
        # Accelerometer measurements are transformed into the navigation
        # frame, corrected for Coriolis/transport effects, and combined
        # with local gravity.
        d_vne = Cnb @ fbbi - np.cross(2 * wnei + wnne, vne) + gamnl_

        # Apply altitude-aiding feedback to the vertical channel.
        if altitude_aiding:
            d_vne[2] += C2 * e + C3 * eI

        # Euler integration of position and velocity.
        llh = llh + d_llh * dt
        vne = vne + d_vne * dt

        # Integrate altitude aiding error.
        if altitude_aiding:
            eI += e * dt

        # Attitude propagation using incremental Rodrigues rotation.
        delta = rodrigues_rotation(wbbn, dt)
        Cnb = Cnb @ delta

        # Re-orthonormalize the DCM to prevent numerical drift.
        Cnb = orthonormalize_dcm(Cnb)

        return llh, vne, Cnb, eI

    for i in range(K):
        # Current IMU measurements.
        fbbi = fbbi_t[i]
        wbbi = wbbi_t[i]

        # Recover Euler angles from the current DCM for storage and plotting.
        rpy[0] = np.arctan2(Cnb[2, 1], Cnb[2, 2])
        rpy[1] = -np.arcsin(Cnb[2, 0])
        rpy[2] = np.arctan2(Cnb[1, 0], Cnb[0, 0])

        # Store current navigation solution before propagating to the next step.
        llh_est[i] = llh
        vne_est[i] = vne
        rpy_est[i] = rpy
        Cnb_est[i] = Cnb

        # Propagate one time step using ideal altitude aiding.
        #
        # Only the height component llh_t[i, 2] is used as an external
        # measurement. Latitude and longitude are not used for correction.
        llh, vne, Cnb, eI = mech_step(
            fbbi,
            wbbi,
            dt,
            llh,
            vne,
            Cnb,
            llh_t[i, 2],
            eI,
        )

    return llh_est, vne_est, rpy_est, Cnb_est
