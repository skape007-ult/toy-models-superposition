import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "phase1_convergence.json").read_text())
steps = sorted({r["steps"] for r in d["runs"]})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
for seed in d["seeds"]:
    ax.plot(steps, [next(r["n_represented"] for r in d["runs"] if r["steps"] == s and r["seed"] == seed) for s in steps], "o-", label=f"seed {seed}, ‖W_i‖>0.5")
    ax.plot(steps, [next(r["n_norm_above_0.1"] for r in d["runs"] if r["steps"] == s and r["seed"] == seed) for s in steps], "s--", alpha=0.5, label=f"seed {seed}, ‖W_i‖>0.1")
ax.set_xscale("log"); ax.set_xlabel("training steps"); ax.set_ylabel("# features represented"); ax.set_ylim(0, 21); ax.axhline(20, color="#999", lw=0.7)
ax.legend(fontsize=7, ncol=2); ax.set_title(f"S = {d['sparsity']}, n=20, m=5")
ax = axes[1]
for s in steps:
    norms = np.mean([r["norms"] for r in d["runs"] if r["steps"] == s], axis=0)
    ax.plot(np.arange(20), norms, "o-", ms=3, label=f"{s//1000}k steps")
ax.axhline(0.5, color="#999", lw=0.7, ls="--"); ax.set_xlabel("feature index (importance 0.7^i)"); ax.set_ylabel("mean ‖W_i‖ over seeds"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "phase1_convergence.png", dpi=130, bbox_inches="tight"); print("wrote figures/phase1_convergence.png")
