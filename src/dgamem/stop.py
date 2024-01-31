import numpy as np


def forward_stop(d):
    """
    Find the first exit time from the domain.

    Parameters
    ----------
    d : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (N,) ndarray of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(d).

    """
    (t,) = np.nonzero(np.logical_not(d))
    t = np.concatenate([[-1], t, [len(d)]])
    return np.repeat(t[1:], np.diff(t))[:-1]


def backward_stop(d):
    """
    Find the last entry time into the domain.

    Parameters
    ----------
    d : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (N,) ndarray of int
        Last entry time into the domain for trajectories starting at
        each frame of the input trajectory. A last entry time not within
        the trajectory is indicated by -1.

    """
    (t,) = np.nonzero(np.logical_not(d))
    t = np.concatenate([[-1], t, [len(d)]])
    return np.repeat(t[:-1], np.diff(t))[1:]
