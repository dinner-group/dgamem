# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preamble

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from compute import (
    compute_bin_edges,
    compute_centers,
    compute_inv_rate,
    compute_m_error,
    compute_m_hist,
    compute_m_mrae,
    compute_m_ref_hist,
    compute_pi_error,
    compute_pi_hist,
    compute_pi_mrae,
    compute_pi_ref_hist,
    compute_q_error,
    compute_q_hist,
    compute_q_mrae,
    compute_q_ref_hist,
    compute_qtilde_error,
    compute_qtilde_hist,
    compute_qtilde_mrae,
    compute_qtilde_ref_hist,
)

# %%
dt = 1e-3  # sampling interval in ns

# histogram bin edges
edges1 = np.linspace(-7.5, 7.5, 150 + 1)
edges2 = np.linspace(-6, 6, 120 + 1)


# %% [markdown]
# # Figure 8

# %% [markdown]
# ## Figure 8a

# %%
def plot():
    fig, ax = plt.subplots(figsize=(4, 4.5))
    edges1, edges2 = compute_bin_edges()
    chi1, chi2 = np.meshgrid(
        (edges1[:-1] + edges1[1:]) / 2, (edges2[:-1] + edges2[1:]) / 2, indexing="ij"
    )
    pi = compute_pi_ref_hist()
    pmf = -np.log(pi)
    pmf -= np.nanmin(pmf)

    pcm = ax.pcolormesh(edges1, edges2, pmf.T)
    cbar = fig.colorbar(pcm, ax=ax, location="top")
    cbar.set_label(r"$PMF/k_\mathrm{B}T$")

    ax.contour(
        chi1,
        chi2,
        pmf,
        np.arange(11),
        colors="w",
        linewidths=0.5,
        linestyles="-",
    )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")

    ax.text(-5.5, 0, r"$A$", c="w", ha="center", va="center")
    ax.text(5.5, 0, r"$B$", c="w", ha="center", va="center")

    plt.show()


plot()


# %% [markdown]
# ## Figure 8b

# %%
def plot():
    fig, ax = plt.subplots(figsize=(4, 4.5))
    edges1, edges2 = compute_bin_edges()
    chi1, chi2 = np.meshgrid(
        (edges1[:-1] + edges1[1:]) / 2, (edges2[:-1] + edges2[1:]) / 2, indexing="ij"
    )
    pi = compute_pi_ref_hist()
    pmf = -np.log(pi)
    pmf -= np.nanmin(pmf)

    pcm = ax.pcolormesh(edges1, edges2, pmf.T)
    cbar = fig.colorbar(pcm, ax=ax, location="top")
    cbar.set_label(r"$PMF/k_\mathrm{B}T$")

    lines = [
        [(-4, -1), (4, -1), (4, 1), (-4, 1), (-4, -1)],
        [(-2, -3), (-2, 3), (2, 3), (2, -3), (-2, -3)],
        [(0, -5.5), (0, 5.5)],
        [(2, 3), (3, 4)],
        [(4, 1), (5, 2)],
        [(-2, 3), (-3, 4)],
        [(-4, 1), (-5, 2)],
        [(2, -3), (3, -4)],
        [(4, -1), (5, -2)],
        [(-2, -3), (-3, -4)],
        [(-4, -1), (-5, -2)],
    ]
    ax.plot(*np.transpose(compute_centers()), "w.", markersize=2)
    for line in lines:
        ax.plot(*np.transpose(line), "w", lw=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")

    ax.text(-5.5, 0, r"$A$", c="w", ha="center", va="center")
    ax.text(5.5, 0, r"$B$", c="w", ha="center", va="center")

    plt.show()


plot()


# %% [markdown]
# # Figure 9

# %%
def plot():
    fig, axes = plt.subplots(
        1, 4, sharex=True, sharey=True, figsize=(6, 2), layout="constrained"
    )
    edges1, edges2 = compute_bin_edges()

    pcm = axes[0].pcolormesh(
        edges1, edges2, compute_pi_ref_hist().T, norm=mpl.colors.LogNorm(1e-5, 1e0)
    )
    cbar = fig.colorbar(pcm, ax=axes[0], location="top")
    cbar.set_label(r"$\pi$")

    pcm = axes[1].pcolormesh(edges1, edges2, compute_m_ref_hist().T, vmin=0, vmax=60)
    cbar = fig.colorbar(pcm, ax=axes[1], location="top")
    cbar.set_label(r"$m/\mathrm{ns}$")

    pcm = axes[2].pcolormesh(edges1, edges2, compute_q_ref_hist().T, vmin=0, vmax=1)
    cbar = fig.colorbar(pcm, ax=axes[2], location="top")
    cbar.set_label(r"$q$")

    pcm = axes[3].pcolormesh(
        edges1, edges2, compute_qtilde_ref_hist().T, vmin=0, vmax=1
    )
    cbar = fig.colorbar(pcm, ax=axes[3], location="top")
    cbar.set_label(r"$\tilde{q}$")

    for ax in axes:
        ax.set_xlabel(r"$\chi_1$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$\chi_2$")

    plt.show()


plot()


# %% [markdown]
# # Figure 10

# %%
def plot():
    fig, axes = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(6.5, 2.5), layout="constrained"
    )

    for i, tau_over_sigma in enumerate([1, 5]):
        for j, tau in enumerate([10, 50, 200, 1000]):
            sigma = tau // tau_over_sigma
            hist = compute_m_hist(sigma, tau)
            pcm = axes[i, j].pcolormesh(edges1, edges2, hist.T, vmin=0, vmax=60)

    cbar = fig.colorbar(pcm, ax=axes)
    cbar.set_label(r"$m^{\sigma,\tau}$")

    axes[0, 0].set_title(r"$\tau=0.01 \mathrm{ns}$", fontsize="medium")
    axes[0, 1].set_title(r"$\tau=0.05 \mathrm{ns}$", fontsize="medium")
    axes[0, 2].set_title(r"$\tau=0.2 \mathrm{ns}$", fontsize="medium")
    axes[0, 3].set_title(r"$\tau=1 \mathrm{ns}$", fontsize="medium")
    for ax in axes.flat:
        ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\chi_1$")
    axes[0, 0].set_ylabel("\n".join(["DGA", r"($\tau/\sigma=1$)", r"$\chi_2$"]))
    axes[1, 0].set_ylabel(
        "\n".join(["DGA with memory", r"($\tau/\sigma=5$)", r"$\chi_2$"])
    )

    plt.show()


plot()


# %% [markdown]
# # Figure 11

# %% [markdown]
# ## Figure 11a

# %%
def plot():
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.fill_between([1e-2, 5], 50, 60, color="lightgray")

    taus = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    for tau_over_sigma in [1, 2, 5, 10]:
        inv_rates = [compute_inv_rate(tau // tau_over_sigma, tau) for tau in taus]
        ax.plot([tau * dt for tau in taus], inv_rates, label=tau_over_sigma)

    ax.semilogx()
    ax.set_xlim(0.01, 5)
    ax.set_ylim(0, 65)
    ax.set_xlabel(r"$\tau/\mathrm{ns}$")
    ax.set_ylabel(r"$k_{AB}^{-1}/\mathrm{ns}$")
    ax.legend(title=r"$\tau/\sigma$")

    plt.show()


plot()


# %% [markdown]
# ## Figure 11b

# %%
def plot():
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.fill_between([5e-3, 5], 50, 60, color="lightgray")

    taus = [
        [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        [25, 50, 100, 250, 500, 1000, 2500, 5000],
        [100, 200, 500, 1000, 2000, 5000],
        [500, 1000, 2000, 5000],
    ]
    for i, sigma in enumerate([5, 25, 100, 500]):
        inv_rates = [compute_inv_rate(sigma, tau) for tau in taus[i]]
        ax.plot([tau * dt for tau in taus[i]], inv_rates, label=sigma * dt)

    ax.semilogx()
    ax.set_xlim(0.005, 5)
    ax.set_ylim(0, 65)
    ax.set_xlabel(r"$\tau/\mathrm{ns}$")
    ax.set_ylabel(r"$k_{AB}^{-1}/\mathrm{ns}$")
    ax.legend(title=r"$\sigma/\mathrm{ns}$")

    plt.show()


plot()


# %% [markdown]
# ## Figure 11c

# %%
def plot():
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.fill_between([1, 10], 50, 60, color="lightgray")

    for tau in [10, 50, 200, 1000]:
        tau_over_sigmas = [1, 2, 5, 10]
        inv_rate = [
            compute_inv_rate(tau // tau_over_sigma, tau)
            for tau_over_sigma in tau_over_sigmas
        ]
        ax.plot(tau_over_sigmas, inv_rate, label=tau * dt)

    ax.set_xlim(1, 10)
    ax.set_ylim(0, 65)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel(r"$\tau/\sigma$")
    ax.set_ylabel(r"$k_{AB}^{-1}/\mathrm{ns}$")
    ax.legend(title=r"$\tau/\mathrm{ns}$")

    plt.show()


plot()


# %% [markdown]
# # Figure 12

# %%
def plot():
    fig, axes = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(6, 3), layout="constrained"
    )

    tau = 50
    for i, tau_over_sigma in enumerate([1, 5]):
        sigma = tau // tau_over_sigma

        error = compute_pi_error(sigma, tau)
        pcm0 = axes[i, 0].pcolormesh(
            edges1, edges2, error.T, vmin=-0.5, vmax=0.5, cmap="coolwarm"
        )

        error = compute_m_error(sigma, tau)
        pcm1 = axes[i, 1].pcolormesh(
            edges1, edges2, error.T, vmin=-0.7, vmax=0.7, cmap="coolwarm"
        )

        error = compute_q_error(sigma, tau)
        pcm2 = axes[i, 2].pcolormesh(
            edges1, edges2, error.T, vmin=-0.7, vmax=0.7, cmap="coolwarm"
        )

        error = compute_qtilde_error(sigma, tau)
        pcm3 = axes[i, 3].pcolormesh(
            edges1, edges2, error.T, vmin=-0.7, vmax=0.7, cmap="coolwarm"
        )

    cbar = fig.colorbar(pcm0, ax=axes[:, 0], location="top")
    cbar.set_label(r"$\Delta \pi^{\sigma,\tau} / \pi$")

    cbar = fig.colorbar(pcm1, ax=axes[:, 1], location="top")
    cbar.set_label(r"$\Delta m^{\sigma,\tau} / m$")

    cbar = fig.colorbar(pcm2, ax=axes[:, 2], location="top")
    cbar.set_label(r"$\Delta q^{\sigma,\tau} / (q (1 - q))$")

    cbar = fig.colorbar(pcm3, ax=axes[:, 3], location="top")
    cbar.set_label(r"$\Delta \tilde{q}^{\sigma,\tau} / (\tilde{q} (1 - \tilde{q}))$")

    for ax in axes.flat:
        ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\chi_1$")
    axes[0, 0].set_ylabel("\n".join(["DGA", r"($\tau/\sigma=1$)", r"$\chi_2$"]))
    axes[1, 0].set_ylabel(
        "\n".join(["DGA with memory", r"($\tau/\sigma=5$)", r"$\chi_2$"])
    )

    plt.show()


plot()


# %% [markdown]
# # Figure 13

# %%
def plot():
    fig, axes = plt.subplots(1, 4, sharex=True, figsize=(8, 2.5), layout="constrained")

    taus = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    for tau_over_sigma in [1, 2, 5, 10]:
        mrae = [compute_pi_mrae(tau // tau_over_sigma, tau) for tau in taus]
        axes[0].plot([tau * dt for tau in taus], mrae, label=tau_over_sigma)

        mrae = [compute_m_mrae(tau // tau_over_sigma, tau) for tau in taus]
        axes[1].plot([tau * dt for tau in taus], mrae, label=tau_over_sigma)

        mrae = [compute_q_mrae(tau // tau_over_sigma, tau) for tau in taus]
        axes[2].plot([tau * dt for tau in taus], mrae, label=tau_over_sigma)

        mrae = [compute_qtilde_mrae(tau // tau_over_sigma, tau) for tau in taus]
        axes[3].plot([tau * dt for tau in taus], mrae, label=tau_over_sigma)

    for ax in axes:
        ax.semilogx()
        ax.set_xlim(0.01, 5)
        ax.set_ylim(0)
        ax.set_xlabel(r"$\tau/\mathrm{ns}$")
    axes[0].set_ylabel(r"$\overline{| \Delta \pi^{\sigma,\tau} / \pi |}$")
    axes[1].set_ylabel(r"$\overline{| \Delta m^{\sigma,\tau} / m |}$")
    axes[2].set_ylabel(r"$\overline{| \Delta q^{\sigma,\tau} / (q (1 - q)) |}$")
    axes[3].set_ylabel(
        r"$\overline{| \Delta \tilde{q}^{\sigma,\tau} / (\tilde{q} (1 - \tilde{q})) |}$"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=r"$\tau/\sigma$", loc="outside right")

    plt.show()


plot()
