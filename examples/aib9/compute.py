# %% [markdown]
# # Preamble

# %%
import numpy as np
import scipy as sp
from joblib import Memory

import dgamem.traj as dga
import dgamem.traj_ref as ref

memory = Memory("data/cache")

# %% [markdown]
# # System

# %%
dt = 1e-3  # sampling interval in ns
area = 0.1**2  # histogram bin area


@memory.cache
def compute_bin_edges():
    """Return histogram bin edges"""
    edges1 = np.linspace(-7.5, 7.5, 150 + 1)
    edges2 = np.linspace(-6, 6, 120 + 1)
    assert np.allclose(np.outer(np.diff(edges1), np.diff(edges2)), area)
    return edges1, edges2


# %% [markdown]
# # States


# %%
@memory.cache
def compute_states():
    phi = np.load("data/phi.npy")
    psi = np.load("data/psi.npy")
    return _compute_states(phi, psi)


@memory.cache
def compute_states_ref():
    phi = np.load("data/phi_ref.npy")
    psi = np.load("data/psi_ref.npy")
    return _compute_states(phi, psi)


def _compute_states(phi, psi):
    phi = phi[..., 2:-2]
    psi = psi[..., 2:-2]
    in_l = (phi - 41) ** 2 + (psi - 47) ** 2 <= 25**2
    in_r = (phi + 41) ** 2 + (psi + 47) ** 2 <= 25**2
    in_a = np.all(in_l, axis=-1)
    in_b = np.all(in_r, axis=-1)
    return in_a, in_b


# %% [markdown]
# # CVs


# %%
@memory.cache
def compute_cvs():
    phi = np.load("data/phi.npy")
    psi = np.load("data/psi.npy")
    return _compute_cvs(phi, psi)


@memory.cache
def compute_cvs_ref():
    phi = np.load("data/phi_ref.npy")
    psi = np.load("data/psi_ref.npy")
    return _compute_cvs(phi, psi)


def _compute_cvs(phi, psi):
    a = -0.8 * (np.sin(np.deg2rad(phi[..., 2:-2])) + np.sin(np.deg2rad(psi[..., 2:-2])))
    cv1 = a @ np.array([1, 1, 1, 1, 1])
    cv2 = a @ np.array([1, 1, 0, -1, -1])
    return cv1, cv2


# %% [markdown]
# # Basis


# %%
@memory.cache
def compute_centers():
    v1 = np.array([1, 1, 1, 1, 1])
    v2 = np.array([1, 1, 0, -1, -1])
    c1 = np.sum(np.reshape(np.meshgrid(*zip(v1, -v1), indexing="ij"), (5, -1)), axis=0)
    c2 = np.sum(np.reshape(np.meshgrid(*zip(v2, -v2), indexing="ij"), (5, -1)), axis=0)
    centers = np.unique(np.stack([c1, c2], axis=-1), axis=0)
    assert centers.shape == (18, 2)
    return centers


@memory.cache
def compute_basis():
    centers = compute_centers()
    cv1, cv2 = compute_cvs()
    features = np.stack([cv1, cv2], axis=-1)
    basis = []
    for x in features:
        indices = np.argmin(np.sum((x[:, np.newaxis] - centers) ** 2, axis=-1), axis=-1)
        cols = len(centers)
        rows = len(indices)
        row_ind = np.arange(rows)
        col_ind = indices
        assert np.all(col_ind >= 0) and np.all(col_ind < cols)
        basis.append(
            sp.sparse.csr_array(
                (np.ones(len(row_ind)), (row_ind, col_ind)), shape=(rows, cols)
            )
        )
    return basis


# %% [markdown]
# # Helper functions


# %%
def _compute_mask():
    eps = 1e-2
    edges1, edges2 = compute_bin_edges()
    centers1 = 0.5 * (edges1[:-1] + edges1[1:])
    centers2 = 0.5 * (edges2[:-1] + edges2[1:])
    xv, yv = np.meshgrid(centers1, centers2, indexing="ij")
    mask = (np.abs(xv) + np.abs(yv) <= 7.0 + eps) & (np.abs(yv) <= 5.5 + eps)
    return mask


_mask = _compute_mask()


def _apply_mask(hist):
    hist[~_mask] = np.nan
    assert np.all(np.isfinite(hist[_mask]))
    # print(np.nanmin(hist), np.nanmax(hist))


# %% [markdown]
# # Reference


# %%
@memory.cache
def compute_m_ref():
    _, in_b = compute_states_ref()
    in_d = ~in_b
    f = np.ones(in_d.shape)[:, :-1]
    g = np.full(in_d.shape, np.nan)
    g[in_b] = 0
    m = ref.forecast(in_d, f, g)
    m = dt * np.array(m)
    return m


@memory.cache
def compute_q_ref():
    in_a, in_b = compute_states_ref()
    in_d = ~(in_a | in_b)
    f = np.zeros(in_d.shape)[:, :-1]
    g = np.full(in_d.shape, np.nan)
    g[in_a] = 0
    g[in_b] = 1
    q = ref.forecast(in_d, f, g)
    q = np.array(q)
    return q


@memory.cache
def compute_qtilde_ref():
    in_a, in_b = compute_states_ref()
    in_d = ~(in_a | in_b)
    f = np.zeros(in_d.shape)[:, :-1]
    g = np.full(in_d.shape, np.nan)
    g[in_a] = 1
    g[in_b] = 0
    q = ref.aftcast(in_d, f, g)
    q = np.array(q)
    return q


# %% [markdown]
# # DGA


# %%
@memory.cache
def compute_pi(sigma, tau):
    basis = compute_basis()
    shape = (len(basis), basis[0].shape[0])
    mu = np.ones(shape)
    mu[:, -2 * tau :] = 0
    mu /= np.sum(mu)
    basis = _remove_constant_feature(basis, mu)
    pi = dga.stationary_distribution(basis, mu, tau, (tau // sigma) - 1)
    pi = np.array(pi)
    assert np.isclose(np.sum(pi), 1)
    return pi


def _remove_constant_feature(basis, weights):
    assert np.array_equal([phi.shape[0] for phi in basis], [len(w) for w in weights])
    phi = sp.sparse.vstack(basis)
    assert isinstance(phi, sp.sparse.sparray)
    w = np.concatenate(weights)

    means = (w @ phi) / np.sum(w)
    assert means.ndim == 1
    order = np.argsort(np.abs(means))
    phi = phi[:, order[:-1]] - means[order[:-1]] / means[order[1:]] * phi[:, order[1:]]
    phi = phi.tocsr()
    offsets = np.concatenate([[0], np.cumsum([len(w) for w in weights])])
    return [phi[i:j] for i, j in zip(offsets[:-1], offsets[1:])]


@memory.cache
def compute_m(sigma, tau):
    basis = compute_basis()
    shape = (len(basis), basis[0].shape[0])
    _, in_b = compute_states()
    pi = compute_pi(sigma, tau)
    basis = [(d[:, np.newaxis] * phi).tocsr() for phi, d in zip(basis, ~in_b)]
    m = dga.forecast(
        basis,
        pi,
        ~in_b,
        1.0,
        np.zeros(shape),
        tau,
        (tau // sigma) - 1,
    )
    m = dt * np.array(m)
    return m


@memory.cache
def compute_q(sigma, tau):
    basis = compute_basis()
    in_a, in_b = compute_states()
    pi = compute_pi(sigma, tau)
    basis = [(d[:, np.newaxis] * phi).tocsr() for phi, d in zip(basis, ~(in_a | in_b))]
    q = dga.forecast(
        basis,
        pi,
        ~(in_a | in_b),
        0.0,
        in_b.astype(float),
        tau,
        (tau // sigma) - 1,
    )
    q = np.array(q)
    return q


@memory.cache
def compute_qtilde(sigma, tau):
    basis = compute_basis()
    in_a, in_b = compute_states()
    pi = compute_pi(sigma, tau)
    basis = [(d[:, np.newaxis] * phi).tocsr() for phi, d in zip(basis, ~(in_a | in_b))]
    q = dga.aftcast(
        basis,
        pi,
        ~(in_a | in_b),
        0.0,
        in_a.astype(float),
        tau,
        (tau // sigma) - 1,
    )
    q = np.array(q)
    return q


# %% [markdown]
# # Reference histogram


# %%
@memory.cache
def compute_pi_ref_hist():
    w = _hist_ref_pi()
    _apply_mask(w)
    return w


@memory.cache
def compute_m_ref_hist():
    u = _hist_ref(compute_m_ref())
    _apply_mask(u)
    return u


@memory.cache
def compute_q_ref_hist():
    u = _hist_ref(compute_q_ref())
    _apply_mask(u)
    return u


@memory.cache
def compute_qtilde_ref_hist():
    u = _hist_ref(compute_qtilde_ref())
    _apply_mask(u)
    return u


@memory.cache
def _hist_ref_inds():
    cv1, cv2 = compute_cvs_ref()
    edges1, edges2 = compute_bin_edges()
    ind1 = np.searchsorted(edges1, cv1) - 1
    ind2 = np.searchsorted(edges2, cv2) - 1
    assert np.all(ind1 >= 0) and np.all(ind1 < len(edges1) - 1)
    assert np.all(ind2 >= 0) and np.all(ind2 < len(edges2) - 1)
    return ind1, ind2


def _hist_ref_pi():
    s = np.s_[:, 500000:-500000]
    ind1, ind2 = _hist_ref_inds()
    hist = np.zeros((15 * 10, 12 * 10))
    np.add.at(hist, (ind1[s], ind2[s]), 1)
    hist /= np.sum(hist)
    hist /= area
    hist = sp.ndimage.gaussian_filter(hist, sigma=2, mode="constant", cval=0.0)
    return hist


def _hist_ref(v):
    s = np.s_[:, 500000:-500000]
    ind1, ind2 = _hist_ref_inds()
    assert np.all(np.isfinite(v[s]))
    numer = np.zeros((15 * 10, 12 * 10))
    denom = np.zeros((15 * 10, 12 * 10))
    np.add.at(numer, (ind1[s], ind2[s]), v[s])
    np.add.at(denom, (ind1[s], ind2[s]), 1)
    numer = sp.ndimage.gaussian_filter(numer, sigma=2, mode="constant", cval=0.0)
    denom = sp.ndimage.gaussian_filter(denom, sigma=2, mode="constant", cval=0.0)
    hist = numer / denom
    return hist


# %% [markdown]
# # DGA histogram


# %%
@memory.cache
def compute_pi_hist(sigma, tau):
    w = compute_pi(sigma, tau)
    w = _hist(w) / area
    _apply_mask(w)
    return w


@memory.cache
def compute_m_hist(sigma, tau):
    w = compute_pi(sigma, tau)
    u = compute_m(sigma, tau)
    wu = np.where(w == 0, 0, w * u)
    assert not np.any(wu[:, -tau:])
    assert not np.any(w[:, -tau:])
    hist = _hist(wu) / _hist(w)
    _apply_mask(hist)
    return hist


@memory.cache
def compute_q_hist(sigma, tau):
    w = compute_pi(sigma, tau)
    u = compute_q(sigma, tau)
    wu = np.where(w == 0, 0, w * u)
    assert not np.any(wu[:, -tau:])
    assert not np.any(w[:, -tau:])
    hist = _hist(wu) / _hist(w)
    _apply_mask(hist)
    return hist


@memory.cache
def compute_qtilde_hist(sigma, tau):
    w = compute_pi(sigma, tau)
    u = compute_qtilde(sigma, tau)
    assert not np.any(w[:, -tau:])
    w = np.roll(w, tau, axis=1)
    wu = np.where(w == 0, 0, w * u)
    assert not np.any(wu[:, :tau])
    assert not np.any(w[:, :tau])
    hist = _hist(wu) / _hist(w)
    _apply_mask(hist)
    return hist


@memory.cache
def _hist_inds():
    cv1, cv2 = compute_cvs()
    edges1, edges2 = compute_bin_edges()
    ind1 = np.searchsorted(edges1, cv1) - 1
    ind2 = np.searchsorted(edges2, cv2) - 1
    assert np.all(ind1 >= 0) and np.all(ind1 < len(edges1) - 1)
    assert np.all(ind2 >= 0) and np.all(ind2 < len(edges2) - 1)
    return ind1, ind2


def _hist(v):
    ind1, ind2 = _hist_inds()
    v = np.asarray_chkfinite(v)
    hist = np.zeros((15 * 10, 12 * 10))
    np.add.at(hist, (ind1, ind2), v)
    hist = sp.ndimage.gaussian_filter(hist, sigma=2, mode="constant", cval=0.0)
    return hist


# %% [markdown]
# # Inverse rate


# %%
@memory.cache
def compute_inv_rate(sigma, tau):
    in_a, _ = compute_states()
    w = compute_pi(sigma, tau)
    mp = compute_m(sigma, tau)
    mask = in_a & (w != 0)
    inv_rate = np.sum(w[mask] * mp[mask]) / np.sum(w[mask])
    return inv_rate


# %% [markdown]
# # Relative error

# %%
@memory.cache
def compute_pi_error(sigma, tau):
    return _rel_error(compute_pi_hist(sigma, tau), compute_pi_ref_hist())


@memory.cache
def compute_m_error(sigma, tau):
    return _rel_error(compute_m_hist(sigma, tau), compute_m_ref_hist())


@memory.cache
def compute_q_error(sigma, tau):
    return _q_error(compute_q_hist(sigma, tau), compute_q_ref_hist())


@memory.cache
def compute_qtilde_error(sigma, tau):
    return _q_error(compute_qtilde_hist(sigma, tau), compute_qtilde_ref_hist())


def _rel_error(pred, true):
    pred = np.where(_mask, pred, np.nan)
    true = np.where(_mask, true, np.nan)
    assert np.all(np.isfinite(pred[_mask]))
    assert np.all(np.isfinite(true[_mask]))
    with np.errstate(divide="ignore"):
        error = np.where(pred == true, 0.0, (pred - true) / true)
    assert np.all(np.isfinite(error[_mask]))
    assert np.all(np.isnan(error[~_mask]))
    return error


def _q_error(pred, true):
    pred = np.where(_mask, pred, np.nan)
    true = np.where(_mask, true, np.nan)
    assert np.all(np.isfinite(pred[_mask]))
    assert np.all(np.isfinite(true[_mask]))
    with np.errstate(divide="ignore"):
        error = np.where(pred == true, 0.0, (pred - true) / (true * (1 - true)))
    assert np.all(np.isfinite(error[_mask]))
    assert np.all(np.isnan(error[~_mask]))
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
    return np.mean(np.abs(error[_mask]))
