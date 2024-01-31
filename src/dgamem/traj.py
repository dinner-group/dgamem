"""Estimate statistics from trajectories using DGA with memory."""

import numpy as np
import scipy as sp

from dgamem.stop import backward_stop, forward_stop

__all__ = [
    "stationary_distribution",
    "stationary_distribution_matrices",
    "stationary_distribution_projection",
    "stationary_distribution_solution",
    "forecast",
    "forecast_matrices",
    "forecast_projection",
    "forecast_solution",
    "aftcast",
    "aftcast_matrices",
    "aftcast_projection",
    "aftcast_solution",
    "solve",
]


def stationary_distribution(
    basis,
    weights,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate the stationary distribution using DGA with memory.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis), optional
        Test basis against which to minimize the error. Must have the
        same shape as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the estimated projection.
    return_solution : bool, optional
        If True (default), return the estimated solution.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the projected stationary distribution.
    solution : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the stationary distribution.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.
    mem_coef : NDArray[float], shape=(mem, n_basis)
        Memory-correction coefficients.

    """
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = stationary_distribution_matrices(
        basis, weights, lag, mem, test_basis=test_basis
    )
    coef, mem_coef = solve(a, b, c0)
    out = []
    if return_projection:
        out.append(stationary_distribution_projection(basis, weights, coef))
    if return_solution:
        out.append(
            stationary_distribution_solution(
                basis, weights, lag, mem, coef, mem_coef
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def stationary_distribution_matrices(
    basis, weights, lag, mem, test_basis=None
):
    """
    Compute matrices for estimating the stationary distribution using
    DGA with memory.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis), optional
        Test basis against which to minimize the error. Must have the
        same shape as `basis`. If `None`, use `basis`.

    Returns
    -------
    a : NDArray[float], shape=(mem+1, n_basis, n_basis)
        Matrices for the homogeneous term.
    b : NDArray[float], shape=(mem+1, n_basis)
        Matrices for the nonhomogeneous term.
    c0 : NDArray[float], shape=(n_basis, n_basis)
        Matrix of inner products of basis functions.

    """
    if test_basis is None:
        test_basis = basis

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a = np.zeros((mem + 1, n_basis, n_basis))
    b = np.zeros((mem + 1, n_basis))
    c0 = np.zeros((n_basis, n_basis))

    for x, y, w in zip(test_basis, basis, weights, strict=True):
        n_frames = len(w)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        wy = w[:end, np.newaxis] * y[:end]
        for n in range(mem + 1):
            dx = (x[(n + 1) * dlag : end + (n + 1) * dlag] - x[:end]).T
            a[n] += dx @ wy
            b[n] += dx @ w[:end]
        c0[:] += x[:end].T @ wy

    return a, b, c0


def stationary_distribution_projection(basis, weights, coef):
    """
    Return the estimated projected stationary distribution

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.

    Returns
    -------
    Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the projected stationary distribution.

    """
    return [w * (y @ coef + 1.0) for y, w in zip(basis, weights, strict=True)]


def stationary_distribution_solution(basis, weights, lag, mem, coef, mem_coef):
    """
    Return the estimated stationary distribution.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.
    mem_coef : NDArray[float], shape=(mem, n_basis)
        Memory-correction coefficients.

    Returns
    -------
    Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the stationary distribution.

    """
    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    out = []
    for y, w in zip(basis, weights, strict=True):
        n_frames = y.shape[0]
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            out.append(np.zeros(n_frames))
            continue
        assert np.all(w[-lag:] == 0.0)

        pad = np.zeros(dlag)
        u = w * (y @ coef + 1.0)
        for v in mem_coef:
            u = np.concatenate([pad, u[:-dlag]])
            u -= w * (y @ v)
        u = np.concatenate([pad, u[:-dlag]])
        out.append(u)
    return out


def forecast(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate a forecast using DGA with memory.

    To solve for the forward committor, set `function=0`.
    To solve for the mean first passage time, set `function=dt` where
    `dt` is the elapsed time between frames.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each frame is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj]-1)
        Function to integrate. This is defined over *steps*, not frames.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis), optional
        Test basis against which to minimize the error. Must have the
        same shape as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the estimated projection.
    return_solution : bool, optional
        If True (default), return the estimated solution.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the projected forecast.
    solution : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the forecast.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.
    mem_coef : NDArray[float], shape=(mem, n_basis)
        Memory-correction coefficients.

    """
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = forecast_matrices(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis=test_basis,
    )
    coef, mem_coef = solve(a, b, c0)
    out = []
    if return_projection:
        out.append(forecast_projection(basis, guess, coef))
    if return_solution:
        out.append(
            forecast_solution(
                basis, in_domain, function, guess, lag, mem, coef, mem_coef
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def forecast_matrices(
    basis, weights, in_domain, function, guess, lag, mem, test_basis=None
):
    """
    Compute matrices for estimating a forecast using DGA with memory.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each frame is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj]-1)
        Function to integrate. This is defined over *steps*, not frames.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis), optional
        Test basis against which to minimize the error. Must have the
        same shape as `basis`. If `None`, use `basis`.

    Returns
    -------
    a : NDArray[float], shape=(mem+1, n_basis, n_basis)
        Matrices for the homogeneous term.
    b : NDArray[float], shape=(mem+1, n_basis)
        Matrices for the nonhomogeneous term.
    c0 : NDArray[float], shape=(n_basis, n_basis)
        Matrix of inner products of basis functions.

    """
    if test_basis is None:
        test_basis = basis
    if not np.iterable(function):
        function = [np.broadcast_to(function, g.shape[0] - 1) for g in guess]

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a = np.zeros((mem + 1, n_basis, n_basis))
    b = np.zeros((mem + 1, n_basis))
    c0 = np.zeros((n_basis, n_basis))

    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess, strict=True
    ):
        n_frames = len(w)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        ix = np.arange(end)
        stop = forward_stop(d)[:end]
        intf = np.insert(np.cumsum(f), 0, 0.0)
        xw = x[:end].T * w[:end]
        for n in range(mem + 1):
            iy = np.minimum(ix + (n + 1) * dlag, stop)
            a[n] += xw @ (y[iy] - y[:end])
            b[n] += xw @ ((g[iy] - g[:end]) + (intf[iy] - intf[:end]))
        c0[:] += xw @ y[:end]

    return a, b, c0


def forecast_projection(basis, guess, coef):
    """
    Return the estimated projected forecast.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.

    Returns
    -------
    Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the projected forecast.

    """
    return [y @ coef + g for y, g in zip(basis, guess, strict=True)]


def forecast_solution(
    basis, in_domain, function, guess, lag, mem, coef, mem_coef
):
    """
    Return the estimated forecast.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each frame is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj]-1)
        Function to integrate. This is defined over *steps*, not frames.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.
    mem_coef : NDArray[float], shape=(mem, n_basis)
        Memory-correction coefficients.

    Returns
    -------
    Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the forecast.

    """
    if not np.iterable(function):
        function = [np.broadcast_to(function, g.shape[0] - 1) for g in guess]

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    out = []
    for y, d, f, g in zip(basis, in_domain, function, guess, strict=True):
        n_frames = y.shape[0]
        assert y.shape == (n_frames, n_basis)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            out.append(np.full(n_frames, np.nan))
            continue

        stop = np.minimum(np.arange(dlag, len(d)), forward_stop(d)[:-dlag])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        r = intf[stop] - intf[:-dlag]
        pad = np.full(dlag, np.nan)
        u = y @ coef + g
        for v in mem_coef:
            u = np.concatenate([u[stop] + r, pad])
            u -= y @ v
        u = np.concatenate([u[stop] + r, pad])
        out.append(u)
    return out


def aftcast(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate an aftcast using DGA with memory.

    To solve for the backward committor, set `function=0`.
    To solve for the mean last passage time, set `function=dt` where
    `dt` is the elapsed time between frames.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Stationary distribution. The last `lag` frames of each trajectory
        must be zero.
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each frame is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj]-1)
        Function to integrate. This is defined over *steps*, not frames.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis), optional
        Test basis against which to minimize the error. Must have the
        same shape as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the estimated projection.
    return_solution : bool, optional
        If True (default), return the estimated solution.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the projected aftcast.
    solution : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the aftcast.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.
    mem_coef : NDArray[float], shape=(mem, n_basis)
        Memory-correction coefficients.

    """
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = aftcast_matrices(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis=test_basis,
    )
    coef, mem_coef = solve(a, b, c0)
    out = []
    if return_projection:
        out.append(aftcast_projection(basis, guess, coef))
    if return_solution:
        out.append(
            aftcast_solution(
                basis, in_domain, function, guess, lag, mem, coef, mem_coef
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def aftcast_matrices(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
):
    """
    Compute matrices for estimating an aftcast using DGA with memory.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Stationary distribution. The last `lag` frames of each trajectory
        must be zero.
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each frame is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj]-1)
        Function to integrate. This is defined over *steps*, not frames.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis), optional
        Test basis against which to minimize the error. Must have the
        same shape as `basis`. If `None`, use `basis`.

    Returns
    -------
    a : NDArray[float], shape=(mem+1, n_basis, n_basis)
        Matrices for the homogeneous term.
    b : NDArray[float], shape=(mem+1, n_basis)
        Matrices for the nonhomogeneous term.
    c0 : NDArray[float], shape=(n_basis, n_basis)
        Matrix of inner products of basis functions.

    """
    if test_basis is None:
        test_basis = basis
    if not np.iterable(function):
        function = [np.broadcast_to(function, g.shape[0] - 1) for g in guess]

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a = np.zeros((mem + 1, n_basis, n_basis))
    b = np.zeros((mem + 1, n_basis))
    c0 = np.zeros((n_basis, n_basis))

    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess, strict=True
    ):
        n_frames = len(w)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        ix = np.arange(lag, n_frames)
        stop = backward_stop(d)[lag:]
        intf = np.insert(np.cumsum(f), 0, 0.0)
        xw = x[lag:].T * w[:end]
        for n in range(mem + 1):
            iy = np.maximum(ix - (n + 1) * dlag, stop)
            a[n] += xw @ (y[iy] - y[lag:])
            b[n] += xw @ ((g[iy] - g[lag:]) + (intf[lag:] - intf[iy]))
        c0[:] += xw @ y[lag:]

    return a, b, c0


def aftcast_projection(basis, guess, coef):
    """
    Return the estimated projected aftcast.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.

    Returns
    -------
    Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the projected aftcast.

    """
    return [y @ coef + g for y, g in zip(basis, guess, strict=True)]


def aftcast_solution(
    basis, in_domain, function, guess, lag, mem, coef, mem_coef
):
    """
    Return the estimated aftcast.

    Parameters
    ----------
    basis : Sequence[NDArray[float] | SparseArray[float]], \
            shape=(n_trajs, n_frames[traj], n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    in_domain : Sequence[NDArray[bool]], shape=(n_trajs, n_frames[traj])
        Whether each frame is in the domain.
    function : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj]-1)
        Function to integrate. This is defined over *steps*, not frames.
    guess : Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.
    mem_coef : NDArray[float], shape=(mem, n_basis)
        Memory-correction coefficients.

    Returns
    -------
    Sequence[NDArray[float]], shape=(n_trajs, n_frames[traj])
        Estimate of the aftcast.

    """
    if not np.iterable(function):
        function = [np.broadcast_to(function, g.shape[0] - 1) for g in guess]

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    out = []
    for y, d, f, g in zip(basis, in_domain, function, guess, strict=True):
        n_frames = y.shape[0]
        assert y.shape == (n_frames, n_basis)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            out.append(np.full(n_frames, np.nan))
            continue

        stop = np.maximum(np.arange(len(d) - dlag), backward_stop(d)[dlag:])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        r = intf[dlag:] - intf[stop]
        pad = np.full(dlag, np.nan)
        u = y @ coef + g
        for v in mem_coef:
            u = np.concatenate([pad, u[stop] + r])
            u -= y @ v
        u = np.concatenate([pad, u[stop] + r])
        out.append(u)
    return out


def solve(a, b, c0):
    """
    Solve DGA with memory for projection and memory-correction
    coefficients.

    Parameters
    ----------
    a : NDArray[float], shape=(mem+1, n_basis, n_basis)
        Matrices for the homogeneous term.
    b : NDArray[float], shape=(mem+1, n_basis)
        Matrices for the nonhomogeneous term.
    c0 : NDArray[float], shape=(n_basis, n_basis)
        Matrix of inner products of basis functions.

    Returns
    -------
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.
    mem_coef : NDArray[float], shape=(mem, n_basis)
        Memory-correction coefficients.

    """
    mem = a.shape[0] - 1
    n_basis = a.shape[1]
    assert a.shape == (mem + 1, n_basis, n_basis)
    assert b.shape == (mem + 1, n_basis)

    b = b[..., np.newaxis]

    inv = sp.linalg.inv(c0)
    a = inv @ a
    b = inv @ b
    c = a[::-1] + np.identity(n_basis)
    for n in range(1, mem + 1):
        a[n] -= np.sum(c[-n:] @ a[:n], axis=0)
        b[n] -= np.sum(c[-n:] @ b[:n], axis=0)

    b = b.reshape(b.shape[:2])

    coef = sp.linalg.solve(a[-1], -b[-1])
    mem_coef = a[:-1] @ coef + b[:-1]
    return coef, mem_coef
