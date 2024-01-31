# %% [markdown]
# # Preamble

# %%
import numpy as np
from joblib import Memory

import dgamem.grid as dga
import dgamem.grid_ref as ref

memory = Memory("data/cache")

# %% [markdown]
# # System

# %%
shape = (80, 80)  # grid shape
size = 80**2  # number of states
temperature = 0.5
sep = 0.05  # grid spacing
area = sep**2  # area of histogram bin


@memory.cache
def compute_bin_edges():
    edges1 = np.linspace(-2, 2, shape[0] + 1)
    edges2 = np.linspace(-1.5, 2.5, shape[1] + 1)
    assert np.allclose(np.outer(np.diff(edges1), np.diff(edges2)), area)
    return edges1, edges2


@memory.cache
def compute_coordinates():
    edges1 = np.linspace(-2, 2, shape[0] + 1)
    edges2 = np.linspace(-1.5, 2.5, shape[1] + 1)
    chi1, chi2 = np.meshgrid(
        (edges1[:-1] + edges1[1:]) / 2,
        (edges2[:-1] + edges2[1:]) / 2,
        indexing="ij",
    )
    return chi1, chi2


@memory.cache
def compute_potential():
    chi1, chi2 = compute_coordinates()
    return (
        3 * np.exp(-(chi1**2 + (chi2 - 1 / 3) ** 2))
        - 3 * np.exp(-(chi1**2 + (chi2 - 5 / 3) ** 2))
        - 5 * np.exp(-((chi1 - 1) ** 2 + chi2**2))
        - 5 * np.exp(-((chi1 + 1) ** 2 + chi2**2))
        + 1 / 5 * chi1**4
        + 1 / 5 * (chi2 - 1 / 3) ** 4
    )


@memory.cache
def compute_generator():
    potential = compute_potential()
    ind = np.arange(size).reshape(shape)

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :], np.s_[1:, :]),
        (np.s_[1:, :], np.s_[:-1, :]),
        (np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:, 1:], np.s_[:, :-1]),
    ]

    out = np.zeros((80 * 80, 80 * 80))
    for row, col in transitions:
        p = (2 * temperature / sep**2) / (
            1 + np.exp((potential[col] - potential[row]) / temperature)
        )
        out[ind[row], ind[col]] += p
        out[ind[row], ind[row]] -= p
    return out


# %% [markdown]
# # States

# %%
@memory.cache
def compute_states():
    chi1, chi2 = compute_coordinates()
    in_a = (chi1 + 1.05) ** 2 + (chi2 + 0.05) ** 2 <= 0.25**2
    in_b = (chi1 - 1.05) ** 2 + (chi2 + 0.05) ** 2 <= 0.25**2
    return in_a, in_b


# %% [markdown]
# # Basis

# %%
@memory.cache
def compute_basis():
    chi1, chi2 = compute_coordinates()
    edges1 = np.linspace(-2, 2, 8 + 1)[1:-1]
    edges2 = np.linspace(-1.5, 2.5, 8 + 1)[1:-1]
    i1, i2 = np.indices(shape)
    j1 = np.searchsorted(edges1, chi1)
    j2 = np.searchsorted(edges2, chi2)
    basis = np.zeros((*shape, 8, 8))
    basis[i1, i2, j1, j2] = 1
    basis = np.reshape(basis, (*shape, 8 * 8))
    return basis


@memory.cache
def compute_basis_pi():
    basis = compute_basis()
    means = np.mean(basis, axis=(0, 1))
    order = np.argsort(np.abs(means))
    basis = (
        basis[..., order[:-1]]
        - means[order[:-1]] / means[order[1:]] * basis[..., order[1:]]
    )
    assert np.allclose(np.sum(basis, axis=(0, 1)), 0)
    return basis


# %% [markdown]
# # Reference

# %%
@memory.cache
def compute_pi_ref():
    potential = compute_potential()
    pi = np.exp(-potential / temperature)
    pi /= np.sum(pi)
    pi /= area
    return pi


@memory.cache
def compute_m_ref():
    gen = compute_generator()
    _, in_b = compute_states()
    return ref.forecast(gen, ~in_b, np.ones(shape), np.zeros(shape))


@memory.cache
def compute_q_ref():
    gen = compute_generator()
    in_a, in_b = compute_states()
    return ref.forecast(gen, ~(in_a | in_b), np.zeros(shape), in_b.astype(float))


@memory.cache
def compute_qtilde_ref():
    gen = compute_generator()
    in_a, in_b = compute_states()
    return ref.aftcast(
        gen,
        compute_pi_ref(),
        ~(in_a | in_b),
        np.zeros(shape),
        in_a.astype(float),
    )


# %% [markdown]
# # DGA

# %%
@memory.cache
def compute_pi(sigma, tau):
    gen = compute_generator()
    basis = compute_basis_pi()
    mu = np.full(shape, 1 / size)
    pi = dga.stationary_distribution(gen, basis, mu, tau, _mem(sigma, tau))
    pi /= area
    return pi


@memory.cache
def compute_m(sigma, tau):
    gen = compute_generator()
    in_a, in_b = compute_states()
    basis = ~in_b[..., np.newaxis] * compute_basis()
    return dga.forecast(
        gen,
        basis,
        compute_pi(sigma, tau),
        ~in_b,
        np.ones(shape),
        np.zeros(shape),
        tau,
        _mem(sigma, tau),
    )


@memory.cache
def compute_q(sigma, tau):
    gen = compute_generator()
    in_a, in_b = compute_states()
    basis = ~(in_a | in_b)[..., np.newaxis] * compute_basis()
    return dga.forecast(
        gen,
        basis,
        compute_pi(sigma, tau),
        ~(in_a | in_b),
        np.zeros(shape),
        in_b.astype(float),
        tau,
        _mem(sigma, tau),
    )


@memory.cache
def compute_qtilde(sigma, tau):
    gen = compute_generator()
    in_a, in_b = compute_states()
    basis = ~(in_a | in_b)[..., np.newaxis] * compute_basis()
    return dga.aftcast(
        gen,
        basis,
        compute_pi(sigma, tau),
        ~(in_a | in_b),
        np.zeros(shape),
        in_a.astype(float),
        tau,
        _mem(sigma, tau),
    )


def _mem(sigma, tau):
    mem = int(np.rint(tau / sigma)) - 1
    assert np.isclose(sigma * (mem + 1), tau)
    return mem


# %% [markdown]
# # Inverse rate

# %%
@memory.cache
def compute_inv_rate_ref():
    m = compute_m_ref()
    pi = compute_pi_ref()
    in_a, _ = compute_states()
    return np.sum(pi[in_a] * m[in_a]) / np.sum(pi[in_a])


@memory.cache
def compute_inv_rate(sigma, tau):
    m = compute_m(sigma, tau)
    pi = compute_pi(sigma, tau)
    in_a, _ = compute_states()
    return np.sum(pi[in_a] * m[in_a]) / np.sum(pi[in_a])


# %% [markdown]
# # Relative error

# %%
@memory.cache
def compute_pi_error(sigma, tau):
    return _rel_error(compute_pi(sigma, tau), compute_pi_ref())


@memory.cache
def compute_m_error(sigma, tau):
    return _rel_error(compute_m(sigma, tau), compute_m_ref())


@memory.cache
def compute_q_error(sigma, tau):
    return _q_error(compute_q(sigma, tau), compute_q_ref())


@memory.cache
def compute_qtilde_error(sigma, tau):
    return _q_error(compute_qtilde(sigma, tau), compute_qtilde_ref())


def _rel_error(pred, true):
    assert np.all(true >= 0)
    error = np.where(pred == true, 0, (pred - true) / true)
    assert np.all(np.isfinite(error))
    return error


def _q_error(pred, true):
    assert np.all(true >= 0) and np.all(true <= 1)
    error = np.where(pred == true, 0, (pred - true) / (true * (1 - true)))
    assert np.all(np.isfinite(error))
    return error


# %% [markdown]
# # Mean absolute relative error

# %%
@memory.cache
def compute_pi_mrae(sigma, tau):
    return _mae(compute_pi_error(sigma, tau))


@memory.cache
def compute_m_mrae(sigma, tau):
    return _mae(compute_m_error(sigma, tau))


@memory.cache
def compute_q_mrae(sigma, tau):
    return _mae(compute_q_error(sigma, tau))


@memory.cache
def compute_qtilde_mrae(sigma, tau):
    return _mae(compute_qtilde_error(sigma, tau))


def _mae(error):
    u = compute_potential()
    return np.mean(np.abs(error[u <= 0]))
