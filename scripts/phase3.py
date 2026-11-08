"""Phase 3: feature geometry. n=200, m=20, uniform importance, sweep sparsity.

Checkpointed per seed: results/phase3_chunks/seed{s}.json (+ .npz with W). Re-running skips
finished chunks. Final merge -> results/phase3.json, results/phase3_W.npz, figure.
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms import Config, train, get_device  # noqa: E402
from tms.analysis import dims_per_feature, feature_dimensionality, feature_norms  # noqa: E402

N, M = 200, 20
N_POINTS = 40
INV_DENSITY = np.logspace(0, 2, N_POINTS)        # 1/(1-S) from 1 to 100
SPARSITIES = 1 - 1 / INV_DENSITY
SEEDS = [0, 1, 2]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
CHUNKS = ROOT / "results" / "phase3_chunks"


def run_chunk(seed, device):
    f = CHUNKS / f"seed{seed}.json"
    if f.exists():
        print(f"chunk seed {seed} exists, skipping", flush=True)
        return
    cfg = Config(N, M, N_POINTS)
    importance = torch.ones(N_POINTS, N)
    model, info = train(cfg, importance, torch.tensor(SPARSITIES, dtype=torch.float32), steps=STEPS,
                        seed=seed, device=device, log_every=1000)
    W = model.W.detach().cpu().numpy()
    b = model.b.detach().cpu().numpy()
    np.savez_compressed(CHUNKS / f"seed{seed}.npz", W=W, b=b)
    rec = {"seed": seed, "steps": STEPS, "device": str(device), "seconds": info["seconds"],
           "inv_density": INV_DENSITY.tolist(), "sparsity": SPARSITIES.tolist(),
           "final_loss": info["final_loss"],
           "dims_per_feature": dims_per_feature(W).tolist(),
           "feature_dimensionality": feature_dimensionality(W).tolist(),
           "feature_norms": feature_norms(W).tolist()}
    f.write_text(json.dumps(rec))
    print(f"chunk seed {seed} done in {info['seconds']:.0f}s", flush=True)


def merge():
    recs = [json.loads((CHUNKS / f"seed{s}.json").read_text()) for s in SEEDS]
    Ws = [np.load(CHUNKS / f"seed{s}.npz")["W"] for s in SEEDS]
    bs = [np.load(CHUNKS / f"seed{s}.npz")["b"] for s in SEEDS]
    np.savez_compressed(ROOT / "results" / "phase3_W.npz", W=np.concatenate(Ws), b=np.concatenate(bs),
                        inv_density=np.tile(INV_DENSITY, len(SEEDS)), seed=np.repeat(SEEDS, N_POINTS))
    out = {"n": N, "m": M, "importance": "uniform", "steps": STEPS, "seeds": SEEDS,
           "inv_density": INV_DENSITY.tolist(), "sparsity": SPARSITIES.tolist(),
           "per_seed": recs}
    # Plateau analysis: for each target fraction, how many sweep points (per seed) sit within +-0.02.
    dpf = np.array([r["dims_per_feature"] for r in recs])  # (seeds, points)
    targets = {"1": 1.0, "3/4": 0.75, "2/3": 2 / 3, "1/2": 0.5, "2/5": 0.4, "3/8": 0.375, "1/3": 1 / 3}
    out["plateau_counts_within_0.02"] = {k: int((np.abs(dpf - v) < 0.02).sum()) for k, v in targets.items()}
    # Longest run of consecutive sweep points within +-0.02 of each target, per seed (plateau length).
    runs_out = {}
    for k, v in targets.items():
        runs = []
        for si, row in enumerate(dpf):
            m = np.abs(row - v) < 0.02
            best, cur, start = (0, None, None), 0, 0
            for i, flag in enumerate(m):
                if flag:
                    if cur == 0:
                        start = i
                    cur += 1
                    if cur > best[0]:
                        best = (cur, start, i)
                else:
                    cur = 0
            runs.append({"seed": SEEDS[si], "length": best[0],
                         "inv_density_range": [float(INV_DENSITY[best[1]]), float(INV_DENSITY[best[2]])] if best[0] else None})
        runs_out[k] = runs
    out["plateau_runs"] = runs_out
    out["dims_per_feature_mean_over_seeds"] = dpf.mean(0).tolist()
    out["dims_per_feature_std_over_seeds"] = dpf.std(0).tolist()
    # Histogram of per-feature dimensionality across the whole sweep, to see spikes at 1/2, 2/5 ...
    fd = np.array([r["feature_dimensionality"] for r in recs]).ravel()
    hist, edges = np.histogram(fd, bins=np.linspace(0, 1.05, 106))
    out["feature_dim_hist"] = {"counts": hist.tolist(), "edges": edges.tolist()}
    (ROOT / "results" / "phase3.json").write_text(json.dumps(out, indent=1))
    print("plateau counts:", out["plateau_counts_within_0.02"])


def main():
    CHUNKS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    t0 = time.time()
    for s in SEEDS:
        run_chunk(s, device)
    merge()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_phase3.py")], check=True)
    print(f"phase 3 total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
