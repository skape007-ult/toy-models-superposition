"""Plot Phase 1 figure from results/phase1.json (no training)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms.plotting import plot_phase1_grid  # noqa: E402

d = json.loads((ROOT / "results" / "phase1.json").read_text())
seed = d["figure_seed"]
runs = [r for r in d["runs"] if r["seed"] == seed]
runs.sort(key=lambda r: d["sparsities"].index(r["sparsity"]))
(ROOT / "figures").mkdir(exist_ok=True)
plot_phase1_grid(np.array(d["gram_seed0"]), [np.array(r["bias"]) for r in runs],
                 [np.array(r["norms"]) for r in runs], [np.array(r["superposition"]) for r in runs],
                 d["sparsities"], [r["final_loss"] for r in runs], ROOT / "figures" / "phase1_superposition.png")
print("wrote figures/phase1_superposition.png")
