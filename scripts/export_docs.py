"""Export precomputed JSON for docs/index.html (self-contained: written as a JS global)."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
p1 = json.loads((ROOT / "results" / "phase1.json").read_text())
runs = sorted([r for r in p1["runs"] if r["seed"] == p1["figure_seed"]], key=lambda r: r["sparsity"])
data = {"phase1": {"n": p1["n"], "m": p1["m"], "sparsities": [r["sparsity"] for r in runs],
                   "gram": [[[round(v, 3) for v in row] for row in g] for g in p1["gram_seed0"]],
                   "bias": [[round(v, 3) for v in r["bias"]] for r in runs],
                   "norms": [[round(v, 3) for v in r["norms"]] for r in runs],
                   "superposition": [[round(v, 3) for v in r["superposition"]] for r in runs],
                   "loss": [r["final_loss"] for r in runs], "n_represented": [r["n_represented"] for r in runs]}}
p3f = ROOT / "results" / "phase3.json"
if p3f.exists():
    p3 = json.loads(p3f.read_text())
    data["phase3"] = {"n": p3["n"], "m": p3["m"], "inv_density": [round(v, 3) for v in p3["inv_density"]],
                      "dpf_mean": [round(v, 4) for v in p3["dims_per_feature_mean_over_seeds"]],
                      "dpf_std": [round(v, 4) for v in p3["dims_per_feature_std_over_seeds"]],
                      "plateaus": p3["plateau_counts_within_0.02"]}
    # one seed's per-feature dimensionality for the slider (40 points x 200 features, rounded)
    data["phase3"]["feature_dims_seed0"] = [[round(v, 3) for v in row] for row in p3["per_seed"][0]["feature_dimensionality"]]
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "data.js").write_text("window.TMS_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("wrote docs/data.js", (ROOT / "docs" / "data.js").stat().st_size, "bytes")
