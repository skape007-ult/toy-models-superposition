"""Plot Phase 3 figures from results/phase3.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms.plotting import plot_phase3  # noqa: E402

d = json.loads((ROOT / "results" / "phase3.json").read_text())
inv = np.concatenate([r["inv_density"] for r in d["per_seed"]])
dpf = np.concatenate([r["dims_per_feature"] for r in d["per_seed"]])
fd = np.concatenate([r["feature_dimensionality"] for r in d["per_seed"]])
seeds = np.concatenate([[r["seed"]] * len(r["inv_density"]) for r in d["per_seed"]])
(ROOT / "figures").mkdir(exist_ok=True)
plot_phase3(inv, dpf, fd, ROOT / "figures" / "phase3_geometry.png", seeds=seeds, n=d["n"], m=d["m"])
# Histogram of per-feature dimensionality over the sweep
h = d["feature_dim_hist"]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(h["edges"][:-1], h["counts"], width=np.diff(h["edges"]), align="edge", color="#1f77b4")
for v, lab in [(1, "1"), (0.75, "3/4"), (2/3, "2/3"), (0.5, "1/2"), (0.4, "2/5"), (3/8, "3/8"), (1/3, "1/3")]:
    ax.axvline(v, color="#999", lw=0.7, ls="--"); ax.text(v, ax.get_ylim()[1]*0.95, lab, fontsize=7, ha="center")
ax.set_xlabel("feature dimensionality D_i (all features, all sparsities, all seeds)"); ax.set_ylabel("count")
ax.set_yscale("log")
fig.savefig(ROOT / "figures" / "phase3_dimensionality_hist.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("wrote figures/phase3_geometry.png, figures/phase3_dimensionality_hist.png")
