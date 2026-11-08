"""Figure helpers. All figures are produced by scripts from results/*.json or tensors."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Black (no superposition) -> gold (fully superposed), as in the paper's bar charts.
SUPERPOS_CMAP = LinearSegmentedColormap.from_list("superpos", ["#111111", "#f2b134"])


def plot_phase1_grid(gram, bias, norms, superpos, sparsities, losses, path):
    """One column per sparsity: W^T W heatmap, bias, and ||W_i|| bars coloured by superposition."""
    k = len(sparsities)
    fig = plt.figure(figsize=(2.6 * k, 7.2))
    gs = fig.add_gridspec(3, k, height_ratios=[4, 0.5, 3.2], hspace=0.35, wspace=0.25)
    for j, S in enumerate(sparsities):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(gram[j], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"S = {S:g}\n1/(1-S) = {1/(1-S):.3g}\nloss {losses[j]:.4f}", fontsize=9)
        if j == 0:
            ax.set_ylabel("WᵀW", fontsize=10)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(bias[j][None, :], cmap="RdBu", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("b", rotation=0, labelpad=12, fontsize=10, va="center")
        ax = fig.add_subplot(gs[2, j])
        n = len(norms[j])
        colors = SUPERPOS_CMAP(np.clip(superpos[j], 0, 1))
        ax.barh(np.arange(n), norms[j], color=colors)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.3)
        ax.set_yticks([0, n - 1]); ax.set_yticklabels(["most imp.", "least imp."], fontsize=7)
        ax.set_xlabel("‖W_i‖", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=SUPERPOS_CMAP, norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.35, -0.04, 0.3, 0.015])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("superposition  Σ_{j≠i} (Ŵ_i·W_j)²   (bar colour)", fontsize=9)
    fig.text(0.5, 0.905, "WᵀW (blue = positive, red = negative)", ha="center", fontsize=9, color="#444")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_phase3(inv_density, dpf, feat_dims, path, seeds=None, n=None, m=None):
    """Top: dimensions-per-feature m/||W||_F^2 vs 1/(1-S). Bottom: per-feature dimensionality scatter."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    lines = {"1 (orthogonal)": 1.0, "3/4 (tetrahedron)": 0.75, "2/3 (triangle)": 2 / 3,
             "1/2 (digon / antipodal)": 0.5, "2/5 (pentagon)": 0.4, "3/8 (square antiprism)": 3 / 8,
             "1/3": 1 / 3}
    for ax in (ax1, ax2):
        for lab, y in lines.items():
            ax.axhline(y, color="#999", lw=0.7, ls="--")
            ax.text(1.02, y, lab, transform=ax.get_yaxis_transform(), fontsize=7, va="center", color="#666")
    inv_density = np.asarray(inv_density)
    dpf = np.asarray(dpf)
    seeds = np.zeros_like(inv_density) if seeds is None else np.asarray(seeds)
    for s in np.unique(seeds):
        sel = seeds == s
        order = np.argsort(inv_density[sel])
        ax1.plot(inv_density[sel][order], dpf[sel][order], "o-", ms=3, lw=0.8, alpha=0.8, label=f"seed {s}")
    ax1.set_ylabel("dimensions per feature  m / ‖W‖²_F")
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8, loc="upper right")
    ttl = f"n={n}, m={m}, uniform importance" if n else ""
    ax1.set_title(f"Feature geometry vs sparsity  {ttl}", fontsize=11)
    # Bottom: scatter of every feature's dimensionality D_i, jittered in x.
    fd = np.asarray(feat_dims)  # (I, n)
    xs = np.repeat(inv_density[:, None], fd.shape[1], axis=1)
    rng = np.random.default_rng(0)
    jitter = np.exp(rng.normal(0, 0.012, size=xs.shape))
    ax2.scatter((xs * jitter).ravel(), fd.ravel(), s=3, alpha=0.25, color="#1f77b4", edgecolors="none")
    ax2.set_xscale("log")
    ax2.set_ylim(-0.02, 1.1)
    ax2.set_xlabel("1 / (1 − S)   (feature density⁻¹)")
    ax2.set_ylabel("feature dimensionality  D_i")
    fig.subplots_adjust(right=0.80)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
