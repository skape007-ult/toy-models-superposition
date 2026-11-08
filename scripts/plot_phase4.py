"""Plot Phase 4: per-neuron input weights (stacked bars) coloured by polysemanticity."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms.plotting import SUPERPOS_CMAP  # noqa: E402
d = json.loads((ROOT / "results" / "phase4.json").read_text())
runs = [r for r in d["runs"] if r["W1"] is not None]
k = len(runs)
fig, axes = plt.subplots(1, k, figsize=(2.9 * k, 5.5), sharey=True)
for ax, r in zip(axes, runs):
    W1 = np.array(r["W1"])  # (m, n)
    poly = np.array(r["polysemanticity"])
    order = np.argsort(-np.abs(W1).max(1))  # show strongest neurons first
    for row, kk in enumerate(order):
        w = W1[kk]
        idx = np.argsort(-np.abs(w))[:8]  # top-8 features per neuron, stacked by |w|
        left = 0.0
        for i in idx:
            ax.barh(row, abs(w[i]), left=left, color=SUPERPOS_CMAP(min(poly[kk] * 2, 1.0)), edgecolor="white", lw=0.3)
            left += abs(w[i])
    ax.invert_yaxis(); ax.set_title(f"S = {r['sparsity']:g}\nmono {r['n_monosemantic']} / poly {r['n_polysemantic']}", fontsize=9)
    ax.set_xlabel("stacked |W1[k, i]| (top 8 features)", fontsize=8); ax.set_xlim(0, 3)
axes[0].set_ylabel("neuron (sorted by max |weight|)")
sm = plt.cm.ScalarMappable(cmap=SUPERPOS_CMAP, norm=plt.Normalize(0, 0.5))
cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.03, pad=0.12); cb.set_label("polysemanticity 1 − max_i w²/Σw²  (clipped at 0.5)")
fig.suptitle(f"Absolute-value task, n={d['n']}, m={d['m']} neurons, importance {d['importance']}", fontsize=11)
fig.savefig(ROOT / "figures" / "phase4_neurons.png", dpi=130, bbox_inches="tight")
print("wrote figures/phase4_neurons.png")
