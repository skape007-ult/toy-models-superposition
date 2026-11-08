"""Plot Phase 2 phase diagrams from results/phase2.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "phase2.json").read_text())
cmap = ListedColormap(["#f0f0f0", "#4c72b0", "#dd8452"])
names = d["class_names"]
ri, dens = np.array(d["rel_importance"]), np.array(d["density"])
panels = [("Empirical (majority of %d seeds)" % d["seeds"], np.array(d["class_majority"]))] + \
         [(f"Empirical, seed {s}", np.array(c)) for s, c in enumerate(d["classes_per_seed"])] + \
         [("Theory (best of 3 solution families)", np.array(d["theory_class"]))]
fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8), sharey=True)
for ax, (title, cls) in zip(axes, panels):
    ax.pcolormesh(ri, dens, cls, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, fontsize=9); ax.set_xlabel("relative importance of feature 1")
axes[0].set_ylabel("feature density  1 − S")
axes[0].invert_yaxis()
fig.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names)], loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.08))
fig.suptitle(f"n=2, m=1 phase diagram  (agreement majority vs theory: {d['agreement_majority_vs_theory']:.2f})", fontsize=11)
fig.savefig(ROOT / "figures" / "phase2_phase_diagram.png", dpi=130, bbox_inches="tight"); plt.close(fig)
# Also show |W_1| and |W_0| of seed 0 as continuous maps
W = np.array(d["W_per_seed"][0])
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
for ax, k in zip(axes, [0, 1]):
    pm = ax.pcolormesh(ri, dens, np.abs(W[:, :, k]), cmap="viridis", vmin=0, vmax=1.2, shading="nearest")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"|W_{k}| (seed 0)"); ax.set_xlabel("relative importance of feature 1")
axes[0].set_ylabel("feature density 1 − S"); axes[0].invert_yaxis()
fig.colorbar(pm, ax=axes); fig.savefig(ROOT / "figures" / "phase2_weights.png", dpi=130, bbox_inches="tight")
print("wrote figures/phase2_phase_diagram.png, figures/phase2_weights.png")
