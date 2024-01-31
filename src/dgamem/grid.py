"""Estimate statistics on a grid using DGA with memory."""

import numpy as np
import scipy as sp

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
    generator,
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
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : NDArray[float], shape=(n_states,)
        Weight of each frame.
    lag : float
        Maximum lag time.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : NDArray[float], shape=(n_states, n_basis), optional
        Test basis against which to minimize the error. Must have the
        same shape as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the estimated projection.
    return_solution : bool, optional
        If True (default), return the estimated solution.
        distribution.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : NDArray[float], shape=(n_states,)
        Estimate of the projected stationary distribution.
    solution : NDArray[float], shape=(n_states,)
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
        generator, basis, weights, lag, mem, test_basis=test_basis
    )
    coef, mem_coef = solve(a, b, c0)
    out = []
    if return_projection:
        out.append(stationary_distribution_projection(basis, weights, coef))
    if return_solution:
        out.append(
            stationary_distribution_solution(
                generator, basis, weights, lag, mem, coef, mem_coef
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
    generator, basis, weights, lag, mem, test_basis=None
):
    """
    Compute matrices for estimating the stationary distribution using
    DGA with memory.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : NDArray[float], shape=(n_states,)
        Weight of each frame.
    lag : float
        Maximum lag time.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : NDArray[float], shape=(n_states, n_basis), optional
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

    _, [x, y], [w] = _to_flat([test_basis, basis], [weights])
    _, k = y.shape
    dlag = lag / (mem + 1)

    L = generator.T
    T = sp.linalg.expm(L * dlag)  # adjoint transition operator
    wy = w[:, None] * y

    a = np.zeros((mem + 1, k, k))
    b = np.zeros((mem + 1, k))
    wy_t = wy
    w_t = w
    for m in range(mem + 1):
        wy_t = T @ wy_t
        w_t = T @ w_t
        a[m] = x.T @ (wy_t - wy)
        b[m] = x.T @ (w_t - w)
    c0 = x.T @ wy
    return a, b, c0


def stationary_distribution_projection(basis, weights, coef):
    """
    Return the estimated projected stationary distribution

    Parameters
    ----------
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : NDArray[float], shape=(n_states,)
        Weight of each frame.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.

    Returns
    -------
    NDArray[float], shape=(n_states,)
        Estimate of the projected stationary distribution.

    """
    shape, [y], [w] = _to_flat([basis], [weights])
    u = w * (y @ coef + 1.0)
    return u.reshape(shape)


def stationary_distribution_solution(
    generator, basis, weights, lag, mem, coef, mem_coef
):
    """
    Return the estimated stationary distribution.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the stationary distribution. Each basis
        function must have zero mean with respect to `weights`.
    weights : NDArray[float], shape=(n_states,)
        Weight of each frame.
    lag : float
        Maximum lag time.
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
    NDArray[float], shape=(n_states,)
        Estimate of the stationary distribution.

    """
    shape, [y], [w] = _to_flat([basis], [weights])
    dlag = lag / (mem + 1)

    L = generator.T
    T = sp.linalg.expm(L * dlag)  # adjoint transition operator

    u = w * (y @ coef + 1.0)  # dga projection
    for m in range(mem):
        u = T @ u - w * (y @ mem_coef[m])
    u = T @ u  # dga solution
    return u.reshape(shape)


def forecast(
    generator,
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
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : NDArray[float], shape=(n_states,)
        Weight of each frame.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each frame is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    lag : float
        Maximum lag time.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : NDArray[float], shape=(n_states, n_basis), optional
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
    projection : NDArray[float], shape=(n_states,)
        Estimate of the projected forecast.
    solution : NDArray[float], shape=(n_states,)
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
        generator,
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
                generator,
                basis,
                in_domain,
                function,
                guess,
                lag,
                mem,
                coef,
                mem_coef,
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
    generator,
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
    Compute matrices for estimating a forecast using DGA with memory.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : NDArray[float], shape=(n_states,)
        Weight of each frame.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each frame is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    lag : float
        Maximum lag time.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : NDArray[float], shape=(n_states, n_basis), optional
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

    _, [x, y], [w, d, f, g] = _to_flat(
        [test_basis, basis], [weights, in_domain, function, guess]
    )
    _, k = y.shape
    dlag = lag / (mem + 1)

    L = generator
    Ld = L[np.ix_(d, d)]
    xwd = x[d].T * w[d]
    yd = y[d]
    rd = sp.linalg.solve(Ld, L[d] @ g + f[d])  # guess - solution
    Sd = sp.linalg.expm(dlag * Ld)  # stopped transition operator

    a = np.empty((mem + 1, k, k))
    b = np.empty((mem + 1, k))
    yd_t = yd
    rd_t = rd
    for m in range(mem + 1):
        yd_t = Sd @ yd_t
        rd_t = Sd @ rd_t
        a[m] = xwd @ (yd_t - yd)
        b[m] = xwd @ (rd_t - rd)
    c0 = xwd @ yd
    return a, b, c0


def forecast_projection(basis, guess, coef):
    """
    Return the estimated projected forecast.

    Parameters
    ----------
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.

    Returns
    -------
    NDArray[float], shape=(n_states,)
        Estimate of the projected forecast.

    """
    shape, [y], [g] = _to_flat([basis], [guess])
    u = y @ coef + g
    return u.reshape(shape)


def forecast_solution(
    generator, basis, in_domain, function, guess, lag, mem, coef, mem_coef
):
    """
    Return the estimated forecast.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each frame is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    lag : float
        Maximum lag time.
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
    NDArray[float], shape=(n_states,)
        Estimate of the forecast.

    """
    shape, [y], [d, f, g] = _to_flat([basis], [in_domain, function, guess])
    dlag = lag / (mem + 1)

    L = generator
    Ld = L[np.ix_(d, d)]
    yd = y[d]
    rd = sp.linalg.solve(Ld, L[d] @ g + f[d])  # guess - true solution
    Sd = sp.linalg.expm(dlag * Ld)  # stopped transition operator

    du = yd @ coef + rd  # dga projection - true solution
    for m in range(mem):
        du = Sd @ du - yd @ mem_coef[m]
    du = Sd @ du  # dga solution - true solution

    u = g.astype(float)
    u[d] -= rd  # true solution
    u[d] += du  # dga solution
    return u.reshape(shape)


def aftcast(
    generator,
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
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : NDArray[float], shape=(n_states,)
        Stationary distribution.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each frame is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    lag : float
        Maximum lag time.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : NDArray[float], shape=(n_states, n_basis), optional
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
    projection : NDArray[float], shape=(n_states,)
        Estimate of the projected aftcast.
    solution : NDArray[float], shape=(n_states,)
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
        generator,
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
                generator,
                basis,
                weights,
                in_domain,
                function,
                guess,
                lag,
                mem,
                coef,
                mem_coef,
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
    generator,
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
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : NDArray[float], shape=(n_states,)
        Stationary distribution.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each frame is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    lag : float
        Maximum lag time.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : NDArray[float], shape=(n_states, n_basis), optional
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

    _, [x, y], [w, d, f, g] = _to_flat(
        [test_basis, basis], [weights, in_domain, function, guess]
    )
    _, k = y.shape
    dlag = lag / (mem + 1)

    L = generator.T
    Ld = L[np.ix_(d, d)]
    rd = L[d] * g - g[d, None] * L[d] + np.diag(f)[d]

    n = L.shape[0]
    nd = Ld.shape[0]

    A = np.block([[Ld, rd], [np.zeros((n, nd)), L]])
    eA = sp.linalg.expm(dlag * A)

    T = eA[nd:, nd:]  # time-reversed transition operator
    Sd = eA[:nd, :nd]  # time-reversed stopped transition operator
    Rd = eA[:nd, nd:]

    a = np.empty((mem + 1, k, k))
    b = np.empty((mem + 1, k))

    xd = x[d].T
    yd = y[d]

    w_t = np.empty((mem + 2, n))
    w_t[0] = w
    for m in range(1, mem + 2):
        w_t[m] = T @ w_t[m - 1]
    wyd_0 = w_t[mem + 1, d, None] * yd
    for m in range(mem + 1):
        wyd_t = w_t[mem - m, d, None] * yd
        wr = np.zeros(nd)
        for i in range(mem - m, mem + 1):
            wyd_t = Sd @ wyd_t
            wr = Sd @ wr + Rd @ w_t[i]
        a[m] = xd @ (wyd_t - wyd_0)
        b[m] = xd @ wr
    c0 = xd @ wyd_0
    return a, b, c0


def aftcast_projection(basis, guess, coef):
    """
    Return the estimated projected aftcast.

    Parameters
    ----------
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    coef : NDArray[float], shape=(n_basis,)
        Projection coefficients.

    Returns
    -------
    NDArray[float], shape=(n_states,)
        Estimate of the projected aftcast.

    """
    shape, [y], [g] = _to_flat([basis], [guess])
    u = y @ coef + g
    return u.reshape(shape)


def aftcast_solution(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    coef,
    mem_coef,
):
    """
    Return the estimated aftcast.

    Parameters
    ----------
    generator : NDArray[float], shape=(n_states, n_states)
        Generator matrix.
    basis : NDArray[float], shape=(n_states, n_basis)
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : NDArray[float], shape=(n_states,)
        Stationary distribution.
    in_domain : NDArray[bool], shape=(n_states,)
        Whether each frame is in the domain.
    function : NDArray[float], shape=(n_states,)
        Function to integrate.
    guess : NDArray[float], shape=(n_states,)
        Guess of the solution. Must satisfy boundary conditions.
    lag : float
        Maximum lag time.
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
    NDArray[float], shape=(n_states,)
        Estimate of the aftcast.

    """
    shape, [y], [w, d, f, g] = _to_flat(
        [basis], [weights, in_domain, function, guess]
    )
    dlag = lag / (mem + 1)

    L = generator.T
    Ld = L[np.ix_(d, d)]
    rd = L[d] * g - g[d, None] * L[d] + np.diag(f)[d]

    n = L.shape[0]
    nd = Ld.shape[0]

    A = np.block([[Ld, rd], [np.zeros((n, nd)), L]])
    eA = sp.linalg.expm(dlag * A)

    T = eA[nd:, nd:]  # time-reversed transition operator
    Sd = eA[:nd, :nd]  # time-reversed stopped transition operator
    Rd = eA[:nd, nd:]

    yd = y[d]

    duw = w[d] * (yd @ coef)  # dga projection - guess
    for m in range(mem):
        duw = Sd @ duw + Rd @ w - w[d] * (yd @ mem_coef[m])
        w = T @ w
    duw = Sd @ duw + Rd @ w  # dga solution - guess
    w = T @ w

    uw = w * g
    uw[d] += duw  # dga solution
    u = uw / w

    return u.reshape(shape)


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

    b = b[..., None]

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


def _to_flat(bases, functions):
    """
    Flatten state space dimensions.

    This function broadcasts the state space dimensions of the input
    arrays against each other, and returns the shape of these dimensions
    and arrays with these dimensions flattened.

    Parameters
    ----------
    bases : Sequence[NDArray[Any]], shape=(n_i, *shape[i], n_basis[i])
        Vector-valued functions of the state space.
    functions : Sequence[NDArray[Any]], shape=(n_j, *shape[j])
        Scalar functions of the state space.

    Returns
    -------
    out_shape : tuple[int, ...]
        Broadcasted shape of the state space:
        ``out_shape = numpy.broadcast_shapes(*shape)``.
        The flattened size of the state space is
        ``size = numpy.product(out_shape)``.
    out_bases : tuple[ndarray, ...], shape=(n_i, size, n_basis[i])
        Flattened broadcasted bases. Note that the last dimension is
        preserved: ``bases[i].shape[-1] == out_bases[i].shape[-1]``.
    out_functions : tuple[ndarray, ...], shape=(n_j, size)
        Flattened broadcasted functions.

    """
    # determine out_shape
    shapes = []
    nbases = []
    for basis in bases:
        *shape, nbasis = np.shape(basis)
        shapes.append(shape)
        nbases.append(nbasis)
    for function in functions:
        shape = np.shape(function)
        shapes.append(shape)
    out_shape = np.broadcast_shapes(*shapes)

    # broadcast and flatten arrays
    out_bases = tuple(
        np.broadcast_to(
            basis,
            (*out_shape, nbasis),
        ).reshape((-1, nbasis))
        for basis, nbasis in zip(bases, nbases, strict=True)
    )
    out_functions = tuple(
        np.ravel(np.broadcast_to(function, out_shape))
        for function in functions
    )

    return out_shape, out_bases, out_functions
