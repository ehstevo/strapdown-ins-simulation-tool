"""Unambiguous coordinate-history adapters around :mod:`r3f`."""

from __future__ import annotations

import numpy as np
import r3f


def curvilinear_history_to_geodetic(position_ned_m: np.ndarray, origin_llh: np.ndarray) -> np.ndarray:
    """Convert a ``(K, 3)`` NED history to a matching LLH history.

    ``r3f`` supports both ``(K, 3)`` and ``(3, K)`` input forms.  A history
    with exactly three samples is ambiguous, so this adapter always passes the
    unambiguous component-major form to the external library.
    """

    position_ned_m = np.asarray(position_ned_m, dtype=float)
    if position_ned_m.ndim != 2 or position_ned_m.shape[1] != 3:
        raise ValueError("Position history must have shape (K, 3).")
    return np.asarray(r3f.curvilinear_to_geodetic(position_ned_m.T, origin_llh)).T


def geodetic_history_to_curvilinear(llh: np.ndarray, origin_llh: np.ndarray) -> np.ndarray:
    """Convert a ``(K, 3)`` LLH history to a matching NED history."""

    llh = np.asarray(llh, dtype=float)
    if llh.ndim != 2 or llh.shape[1] != 3:
        raise ValueError("LLH history must have shape (K, 3).")
    return np.asarray(r3f.geodetic_to_curvilinear(llh.T, origin_llh)).T
