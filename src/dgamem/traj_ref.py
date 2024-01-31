"""Estimate statistics from long trajectories."""

import numpy as np

from dgamem.stop import backward_stop, forward_stop

__all__ = ["forecast", "aftcast"]


def forecast(in_domain, function, guess):
    """
    Estimate a forecast.

    Parameters
    ----------
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each point is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Function to integrate.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess of the forecast. Must obey boundary conditions.

    Returns
    -------
    list[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimated forecast.

    """
    out = []
    for d, f, g in zip(in_domain, function, guess, strict=True):
        n = d.shape[0]
        f = np.broadcast_to(f, (n - 1,))
        assert d.shape == (n,)
        assert f.shape == (n - 1,)
        assert g.shape == (n,)

        t0 = np.arange(len(d))
        t1 = forward_stop(d)
        mask = t1 != n
        t0 = t0[mask]
        t1 = t1[mask]
        assert np.all(t0 >= 0) and np.all(t0 <= t1) and np.all(t1 < n)
        assert not np.any(d[t1])

        intf = np.concatenate([np.zeros(1), np.cumsum(f)])
        u = np.full(n, np.nan)
        u[t0] = g[t1] + (intf[t1] - intf[t0])

        out.append(u)
    return out


def aftcast(in_domain, function, guess):
    """
    Estimate an aftcast.

    Parameters
    ----------
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each point is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Function to integrate.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess of the aftcast. Must obey boundary conditions.

    Returns
    -------
    list[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimated aftcast.

    """
    out = []
    for d, f, g in zip(in_domain, function, guess, strict=True):
        n = d.shape[0]
        f = np.broadcast_to(f, (n - 1,))
        assert d.shape == (n,)
        assert f.shape == (n - 1,)
        assert g.shape == (n,)

        t0 = np.arange(len(d))
        t1 = backward_stop(d)
        mask = t1 != -1
        t0 = t0[mask]
        t1 = t1[mask]
        assert np.all(t1 >= 0) and np.all(t1 <= t0) and np.all(t0 < n)
        assert not np.any(d[t1])

        intf = np.concatenate([np.zeros(1), np.cumsum(f)])
        u = np.full(n, np.nan)
        u[t0] = g[t1] + (intf[t0] - intf[t1])

        out.append(u)
    return out
