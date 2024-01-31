"""Compute statistics on a grid."""

import numpy as np
import scipy as sp

__all__ = ["stationary_distribution", "forecast", "aftcast"]


def stationary_distribution(generator):
    """
    Compute the stationary distribution.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.

    Returns
    -------
    NDArray[float], shape=(n_states,)
        Stationary distribution.

    """
    a = generator.T[:-1, :-1]
    b = generator.T[:-1, -1]

    pi = np.empty(generator.shape[0])
    pi[:-1] = -sp.linalg.solve(a, b)
    pi[-1] = 1
    pi /= np.sum(pi)
    return pi


def forecast(generator, in_domain, function, guess):
    """
    Compute a forecast.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each point is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the forecast. Must obey boundary conditions.

    Returns
    -------
    NDArray[float], shape=(n_states,)
        Forecast.

    """
    d, f, g = np.broadcast_arrays(in_domain, function, guess)
    d = np.ravel(d)
    f = np.ravel(f)
    g = np.ravel(g)

    a = generator[np.ix_(d, d)]
    b = generator[d] @ g + f[d]

    u = g.astype(float)
    u[d] -= sp.linalg.solve(a, b)
    return u.reshape(g.shape)


def aftcast(generator, weights, in_domain, function, guess):
    """
    Compute an aftcast.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    weights : NDArray[float], shape=(n_states,)
        Stationary distribution.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each point is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the aftcast. Must obey boundary conditions.

    Returns
    -------
    NDArray[float], shape=(n_states,)
        Aftcast.

    """
    pi, d, f, g = np.broadcast_arrays(weights, in_domain, function, guess)
    pi = np.ravel(pi)

    # generator for the time-reversed process
    generator_rev = generator.T * pi / pi[:, np.newaxis]

    return forecast(generator_rev, d, f, g)
