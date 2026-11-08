import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "phase3_convergence.json").read_text())
p3 = json.loads((ROOT / "results" / "phase3.json").read_text())
inv = np.array(d["inv_density"])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
lines = {"1/2": .5, "2/5": .4, "3/8": .375, "1/3": 1/3, "2/3": 2/3, "3/4": .75}
for ax in (ax1, ax2):
    for lab, y in lines.items():
        ax.axhline(y, color="#999", lw=0.7, ls="--"); ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), fontsize=8, va="center", color="#666")
inv0 = np.array(p3["inv_density"]); sel = (inv0 >= 2.5) & (inv0 <= 14)
ax1.plot(inv0[sel], np.array(p3["dims_per_feature_mean_over_seeds"])[sel], "s-", color="#bbb", ms=5, label="first pass: 10k steps, mean of 3 seeds")
for r in d["per_seed"]:
    ax1.plot(inv, r["dims_per_feature"], "o-", ms=3, lw=0.8, alpha=0.8, label=f"50k steps, seed {r['seed']}")
ax1.set_ylabel("dimensions per feature  m/‖W‖²_F"); ax1.set_ylim(0.15, 0.6); ax1.legend(fontsize=7, loc="upper right")
ax1.set_title("Phase 3 convergence check: 1/(1−S) ∈ [3, 12], 5× steps, 5 seeds (n=200, m=20)", fontsize=10)
fd = np.array([r["feature_dimensionality"] for r in d["per_seed"]])  # (seeds, points, n)
xs = np.broadcast_to(inv[None, :, None], fd.shape)
rng = np.random.default_rng(0); jit = np.exp(rng.normal(0, 0.012, size=fd.shape))
ax2.scatter((xs * jit).ravel(), fd.ravel(), s=3, alpha=0.2, color="#1f77b4", edgecolors="none")
ax2.set_xscale("log"); ax2.set_ylim(-0.02, 0.85); ax2.set_xlabel("1 / (1 − S)"); ax2.set_ylabel("feature dimensionality D_i")
ax2.set_xticks([3, 4, 5, 6, 8, 10, 12]); ax2.set_xticklabels(["3", "4", "5", "6", "8", "10", "12"])
fig.subplots_adjust(right=0.9); fig.savefig(ROOT / "figures" / "phase3_convergence.png", dpi=130, bbox_inches="tight"); print("wrote figures/phase3_convergence.png")
