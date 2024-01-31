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
    compute_coordinates,
    compute_inv_rate,
    compute_inv_rate_ref,
    compute_m,
    compute_m_error,
    compute_m_mrae,
    compute_m_ref,
    compute_pi,
    compute_pi_error,
    compute_pi_mrae,
    compute_pi_ref,
    compute_potential,
    compute_q,
    compute_q_error,
    compute_q_mrae,
    compute_q_ref,
    compute_qtilde,
    compute_qtilde_error,
    compute_qtilde_mrae,
    compute_qtilde_ref,
)


# %% [markdown]
# # Figure 1

# %% [markdown]
# ## Figure 1a

# %%
def plot():
    fig, ax = plt.subplots(figsize=(4, 5))
    edges1, edges2 = compute_bin_edges()
    chi1, chi2 = compute_coordinates()
    potential = compute_potential()

    pcm = ax.pcolormesh(edges1, edges2, potential.T)
    cbar = fig.colorbar(pcm, ax=ax, location="top")
    cbar.set_label(r"$V$")

    ax.contour(
        chi1,
        chi2,
        potential,
        np.linspace(-3.5, 7, 22),
        colors="w",
        linewidths=0.5,
        linestyles="-",
    )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")

    ax.text(-1.05, -0.05, r"$A$", c="w", ha="center", va="center")
    ax.add_patch(mpl.patches.Circle((-1.05, -0.05), 0.25, fc="none", ec="w"))

    ax.text(1.05, -0.05, r"$B$", c="w", ha="center", va="center")
    ax.add_patch(mpl.patches.Circle((1.05, -0.05), 0.25, fc="none", ec="w"))

    plt.show()


plot()


# %% [markdown]
# # Figure 1b

# %%
def plot():
    fig, ax = plt.subplots(figsize=(4, 5))
    edges1, edges2 = compute_bin_edges()
    potential = compute_potential()

    pcm = ax.pcolormesh(edges1, edges2, potential.T)
    cbar = fig.colorbar(pcm, ax=ax, location="top")
    cbar.set_label(r"$V$")

    for v in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
        ax.axvline(v, c="w", lw=0.5)
    for v in [-1, -0.5, 0, 0.5, 1, 1.5, 2]:
        ax.axhline(v, c="w", lw=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")

    ax.text(-1.05, -0.05, r"$A$", c="w", ha="center", va="center")
    ax.add_patch(mpl.patches.Circle((-1.05, -0.05), 0.25, fc="none", ec="w"))

    ax.text(1.05, -0.05, r"$B$", c="w", ha="center", va="center")
    ax.add_patch(mpl.patches.Circle((1.05, -0.05), 0.25, fc="none", ec="w"))

    plt.show()


plot()


# %% [markdown]
# ## Figure 2

# %%
def plot():
    fig, axes = plt.subplots(
        1, 4, sharex=True, sharey=True, figsize=(6, 2.5), layout="constrained"
    )
    edges1, edges2 = compute_bin_edges()

    pcm = axes[0].pcolormesh(
        edges1, edges2, compute_pi_ref().T, norm=mpl.colors.LogNorm(1e-5, 1e1)
    )
    cbar = fig.colorbar(pcm, ax=axes[0], location="top")
    cbar.set_label(r"$\pi$")

    pcm = axes[1].pcolormesh(edges1, edges2, compute_m_ref().T, vmin=0, vmax=60)
    cbar = fig.colorbar(pcm, ax=axes[1], location="top")
    cbar.set_label(r"$m$")

    pcm = axes[2].pcolormesh(edges1, edges2, compute_q_ref().T, vmin=0, vmax=1)
    cbar = fig.colorbar(pcm, ax=axes[2], location="top")
    cbar.set_label(r"$q$")

    pcm = axes[3].pcolormesh(edges1, edges2, compute_qtilde_ref().T, vmin=0, vmax=1)
    cbar = fig.colorbar(pcm, ax=axes[3], location="top")
    cbar.set_label(r"$\tilde{q}$")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\chi_1$")
    axes[0].set_ylabel(r"$\chi_2$")

    plt.show()


plot()


# %% [markdown]
# # Figure 3

# %%
def plot():
    fig, axes = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(6, 2.5), layout="constrained"
    )
    edges1, edges2 = compute_bin_edges()

    for i, tau_over_sigma in enumerate([1, 5]):
        for j, tau in enumerate([0.01, 0.05, 0.2, 1]):
            sigma = tau / tau_over_sigma
            hist = compute_m(sigma, tau)
            pcm = axes[i, j].pcolormesh(edges1, edges2, hist.T, vmin=0, vmax=60)
    cbar = fig.colorbar(pcm, ax=axes)
    cbar.set_label(r"$m^{\sigma,\tau}$")

    for ax in axes.flat:
        ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\chi_1$")
    axes[0, 0].set_ylabel("\n".join(["DGA", r"($\tau/\sigma=1$)", r"$\chi_2$"]))
    axes[1, 0].set_ylabel(
        "\n".join(["DGA with memory", r"($\tau/\sigma=5$)", r"$\chi_2$"])
    )
    axes[0, 0].set_title(r"$\tau=0.01$", fontsize="medium")
    axes[0, 1].set_title(r"$\tau=0.05$", fontsize="medium")
    axes[0, 2].set_title(r"$\tau=0.2$", fontsize="medium")
    axes[0, 3].set_title(r"$\tau=1$", fontsize="medium")

    plt.show()


plot()

# %% [markdown]
# # Figure 4

# %%
print(compute_inv_rate_ref())


# %% [markdown]
# ## Figure 4a

# %%
def plot():
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.axhline(compute_inv_rate_ref(), c="k", linestyle="--")

    taus = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for tau_over_sigma in [1, 2, 5, 10]:
        inv_rates = [compute_inv_rate(tau / tau_over_sigma, tau) for tau in taus]
        ax.plot(taus, inv_rates, label=tau_over_sigma)

    ax.semilogx()
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(0, 60)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$k_{AB}^{-1}$")
    ax.legend(title=r"$\tau/\sigma$")

    plt.show()


plot()


# %% [markdown]
# ## Figure 4b

# %%
def plot():
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.axhline(compute_inv_rate_ref(), c="k", linestyle="--")

    for sigma in [0.001, 0.01, 0.02, 0.05]:
        taus = [
            sigma * tau_over_sigma
            for tau_over_sigma in range(1, round(0.2 / sigma) + 1)
        ]
        inv_rates = [compute_inv_rate(sigma, tau) for tau in taus]
        ax.plot(taus, inv_rates, label=sigma)

    ax.semilogx()
    ax.set_xlim(1e-3, 2e-1)
    ax.set_ylim(0, 60)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$k_{AB}^{-1}$")
    ax.legend(title=r"$\sigma$")

    plt.show()


plot()


# %% [markdown]
# ## Figure 4c

# %%
def plot():
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.axhline(compute_inv_rate_ref(), c="k", linestyle="--")

    for tau in [0.01, 0.02, 0.05, 0.1]:
        tau_over_sigmas = list(range(1, 11))
        inv_rate = [
            compute_inv_rate(tau / tau_over_sigma, tau)
            for tau_over_sigma in tau_over_sigmas
        ]
        ax.plot(tau_over_sigmas, inv_rate, label=tau)

    ax.set_xlim(1, 10)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel(r"$\tau/\sigma$")
    ax.set_ylabel(r"$k_{AB}^{-1}$")
    ax.legend(title=r"$\tau$")

    plt.show()


plot()


# %% [markdown]
# # Figure 5

# %%
def plot():
    fig, axes = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(6, 3.5), layout="constrained"
    )
    edges1, edges2 = compute_bin_edges()

    tau = 0.05
    for i, tau_over_sigma in enumerate([1, 5]):
        sigma = tau / tau_over_sigma

        error = compute_pi_error(sigma, tau)
        pcm0 = axes[i, 0].pcolormesh(
            edges1, edges2, error.T, vmin=-10, vmax=10, cmap="coolwarm"
        )

        error = compute_m_error(sigma, tau)
        pcm1 = axes[i, 1].pcolormesh(
            edges1, edges2, error.T, vmin=-1, vmax=1, cmap="coolwarm"
        )

        error = compute_q_error(sigma, tau)
        pcm2 = axes[i, 2].pcolormesh(
            edges1, edges2, error.T, vmin=-5, vmax=5, cmap="coolwarm"
        )

        error = compute_qtilde_error(sigma, tau)
        pcm3 = axes[i, 3].pcolormesh(
            edges1, edges2, error.T, vmin=-50, vmax=50, cmap="coolwarm"
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
# # Figure 6

# %%
def plot():
    fig, axes = plt.subplots(1, 4, sharex=True, figsize=(8, 2.5), layout="constrained")

    taus = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    for tau_over_sigma in [1, 2, 5, 10]:
        mrae = [compute_pi_mrae(tau / tau_over_sigma, tau) for tau in taus]
        axes[0].plot(taus, mrae, label=tau_over_sigma)

        mrae = [compute_m_mrae(tau / tau_over_sigma, tau) for tau in taus]
        axes[1].plot(taus, mrae, label=tau_over_sigma)

        mrae = [compute_q_mrae(tau / tau_over_sigma, tau) for tau in taus]
        axes[2].plot(taus, mrae, label=tau_over_sigma)

        mrae = [compute_qtilde_mrae(tau / tau_over_sigma, tau) for tau in taus]
        axes[3].plot(taus, mrae, label=tau_over_sigma)

    for ax in axes:
        ax.semilogx()
        ax.set_xlim(0.01, 10)
        ax.set_ylim(0)
        ax.set_xlabel(r"$\tau$")
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
