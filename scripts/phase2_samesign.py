"""Item 2 follow-up: are the same-sign n=2,m=1 solutions (W ~ [1.05, 0.35]) lower-loss solutions
that the three-family theory misses, or higher-loss local minima? Reads results/phase2.json,
writes results/phase2_samesign.json."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "phase2.json").read_text())
W = np.array(d["W_per_seed"]); L = np.array(d["loss_per_seed"]); fam = np.array(d["theory_family_loss"])
cls = np.array(d["classes_per_seed"]); T = d["thresh"]
best = fam.min(0)
same = (np.sign(W[..., 0]) == np.sign(W[..., 1])) & (np.abs(W[..., 0]) > T) & (np.abs(W[..., 1]) > T)
anti = (np.sign(W[..., 0]) != np.sign(W[..., 1])) & (np.abs(W[..., 0]) > T) & (np.abs(W[..., 1]) > T)
single = (cls == 0) | (cls == 1)
gap = L - best[None]
def stats(mask):
    g = gap[mask]; rel = g / best[None].repeat(W.shape[0], 0)[mask]
    return {"n_models": int(mask.sum()), "gap_mean": float(g.mean()), "gap_median": float(np.median(g)),
            "gap_min": float(g.min()), "frac_gap_above_1e-3": float((g > 1e-3).mean()),
            "frac_gap_below_minus_1e-4": float((g < -1e-4).mean()), "relative_gap_median": float(np.median(rel))}
out = {"same_sign": stats(same), "antipodal": stats(anti), "single_feature": stats(single)}
cells = np.argwhere(same.any(0)); n = worse = 0
for r, c in cells:
    ss = L[same[:, r, c], r, c]; other = L[~same[:, r, c], r, c]
    if len(other):
        n += 1; worse += int(ss.min() > other.min() + 1e-4)
out["cells_with_same_sign_and_other_solution"] = n
out["cells_where_same_sign_is_worse"] = worse
ri = np.array(d["rel_importance"]); de = np.array(d["density"]); idx = np.argwhere(same)
out["same_sign_rel_importance_range"] = [float(ri[idx[:, 2]].min()), float(ri[idx[:, 2]].max())]
out["same_sign_density_range"] = [float(de[idx[:, 1]].min()), float(de[idx[:, 1]].max())]
out["verdict"] = ("higher-loss local minimum" if out["same_sign"]["frac_gap_above_1e-3"] > 0.8 and worse == n
                  else "inconclusive")
out["note"] = "gap = trained final loss (8192 fresh samples) minus best of the 3 theory families (20k MC samples) at the same cell"
(ROOT / "results" / "phase2_samesign.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
