"""Phase 1: basic superposition. n=20, m=5, importance 0.7^i, six sparsities, several seeds."""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms import Config, train, get_device  # noqa: E402
from tms.analysis import feature_norms, superposition, gram, dims_per_feature  # noqa: E402

SPARSITIES = [0.0, 0.7, 0.9, 0.97, 0.99, 0.999]
SEEDS = [0, 1, 2]
N, M = 20, 5
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
REPRESENTED_THRESH = 0.5   # ||W_i|| above this counts as "represented"
SUPERPOS_THRESH = 0.2      # and superposition above this counts as "in superposition"


def main():
    device = get_device()
    cfg = Config(N, M, len(SPARSITIES) * len(SEEDS))
    # instance index = seed_idx * len(SPARSITIES) + sparsity_idx
    sparsity = torch.tensor([S for _ in SEEDS for S in SPARSITIES])
    importance = (0.7 ** torch.arange(N)).repeat(cfg.n_instances, 1)
    # Different seeds get different inits via a per-instance xavier draw; data noise is shared
    # across instances by construction of the batched sampler, so per-seed chunks are trained
    # separately to get truly independent data streams too.
    out = {"n": N, "m": M, "importance": "0.7^i", "sparsities": SPARSITIES, "seeds": SEEDS,
           "represented_thresh": REPRESENTED_THRESH, "superpos_thresh": SUPERPOS_THRESH,
           "device": str(device), "runs": []}
    t0 = time.time()
    all_W, all_b = [], []
    for seed in SEEDS:
        cfg_s = Config(N, M, len(SPARSITIES))
        model, info = train(cfg_s, importance[: len(SPARSITIES)], torch.tensor(SPARSITIES), steps=STEPS,
                            seed=seed, device=device, log_every=2000)
        W, b = model.W.detach().cpu().numpy(), model.b.detach().cpu().numpy()
        all_W.append(W); all_b.append(b)
        norms, sp = feature_norms(W), superposition(W)
        for j, S in enumerate(SPARSITIES):
            rep = norms[j] > REPRESENTED_THRESH
            out["runs"].append({
                "seed": seed, "sparsity": S, "final_loss": info["final_loss"][j],
                "norms": norms[j].tolist(), "superposition": sp[j].tolist(), "bias": b[j].tolist(),
                "n_represented": int(rep.sum()),
                "n_in_superposition": int((rep & (sp[j] > SUPERPOS_THRESH)).sum()),
                "dims_per_feature": float(dims_per_feature(W)[j]),
                "frobenius_sq": float((W[j] ** 2).sum()),
            })
        if seed == SEEDS[0]:
            out["figure_seed"] = seed
            out["gram_seed0"] = gram(W).tolist()
        out.setdefault("train_info", []).append({k: v for k, v in info.items() if k != "history"})
    out["seconds_total"] = time.time() - t0
    np.savez_compressed(ROOT / "results" / "phase1_W.npz", W=np.stack(all_W), b=np.stack(all_b),
                        seeds=np.array(SEEDS), sparsities=np.array(SPARSITIES))
    # Summary across seeds
    summ = []
    for S in SPARSITIES:
        rs = [r for r in out["runs"] if r["sparsity"] == S]
        summ.append({"sparsity": S,
                     "n_represented_mean": float(np.mean([r["n_represented"] for r in rs])),
                     "n_represented_all_seeds": [r["n_represented"] for r in rs],
                     "n_in_superposition_mean": float(np.mean([r["n_in_superposition"] for r in rs])),
                     "loss_mean": float(np.mean([r["final_loss"] for r in rs])),
                     "dims_per_feature_mean": float(np.mean([r["dims_per_feature"] for r in rs]))})
    out["summary"] = summ
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "phase1.json").write_text(json.dumps(out, indent=1))
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_phase1.py")], check=True)
    for s in summ:
        print(s)
    print(f"done in {out['seconds_total']:.0f}s on {device}")


if __name__ == "__main__":
    main()
