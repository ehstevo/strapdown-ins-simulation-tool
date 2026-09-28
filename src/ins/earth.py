import numpy as np

def somigliana(lat, hae):
    """
    Compute local gravity magnitude using the Somigliana gravity model.

    Parameters
    ----------
    lat : float or ndarray
        Geodetic latitude (rad).

    hae : float or ndarray
        Height above ellipsoid (m).

    Returns
    -------
    grav_l : float or ndarray
        Local gravity magnitude (m/s^2).
    """

    # Earth/gravity model constants
    A_E = 6378137.0
    E2 = 6.694379990141317e-3
    ge = 9.7803253359
    k = 1.93185265241e-3
    f = 3.35281066475e-3
    m = 3.44978650684e-3

    slat = np.sin(lat)
    klat = np.sqrt(1 - E2 * slat**2)

    # Somigliana gravity at the ellipsoid surface
    g0 = ge * (1 + k * slat**2) / klat

    # Altitude correction
    grav_l = g0*(1 + (3/A_E**2)*hae**2 - 2/A_E*(1 + f + m - 2*f*slat**2)*hae)

    return grav_l