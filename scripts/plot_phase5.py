"""Plot Phase 5 SAE recovery from results/phase5_sae.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "phase5_sae.json").read_text())
sps, lams = d["sparsities"], d["lams"]
rec = np.array([[next(r["n_recovered"] for r in d["runs"] if r["sparsity"] == S and r["lam"] == l) for l in lams] for S in sps])
rep = np.array([next(r["n_represented_true"] for r in d["runs"] if r["sparsity"] == S) for S in sps])
prec = np.array([[next(r["precision"] or 0 for r in d["runs"] if r["sparsity"] == S and r["lam"] == l) for l in lams] for S in sps])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
for i, S in enumerate(sps):
    ax.plot(lams, rec[i], "o-", label=f"S={S:g} (rep. {rep[i]})")
ax.set_xscale("log"); ax.set_xlabel("L1 coefficient λ"); ax.set_ylabel("# true features recovered (cos > 0.9)"); ax.legend(fontsize=7)
ax = axes[1]
for i, S in enumerate(sps):
    ax.plot(lams, prec[i], "o-", label=f"S={S:g}")
ax.set_xscale("log"); ax.set_xlabel("L1 coefficient λ"); ax.set_ylabel("precision of live dictionary elements"); ax.set_ylim(0, 1.05)
ax = axes[2]
best = d["best_per_sparsity"]
x = np.arange(len(sps))
ax.bar(x - 0.2, [b["n_represented_true"] for b in best], 0.4, label="represented by toy model", color="#bbb")
ax.bar(x + 0.2, [b["n_recovered"] for b in best], 0.4, label="recovered by SAE (best λ)", color="#4c72b0")
ax.set_xticks(x); ax.set_xticklabels([f"{S:g}" for S in sps]); ax.set_xlabel("sparsity S"); ax.set_ylabel("# features"); ax.legend(fontsize=8)
fig.suptitle(f"SAE (dict {d['dict_size']}) on 5-dim hidden activations of the Phase 1 models (n=20, m=5)", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "phase5_sae_recovery.png", dpi=130, bbox_inches="tight")
print("wrote figures/phase5_sae_recovery.png")
