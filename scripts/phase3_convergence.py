"""Item 4: Phase 3 rerun around 1/(1-S) in [3, 12] with 5x steps (50k) and 5 seeds.
Checkpointed per seed in results/phase3_conv_chunks/. Writes results/phase3_convergence.json."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms import Config, train, get_device
from tms.analysis import dims_per_feature, feature_dimensionality, feature_norms

N, M = 200, 20
INV = np.logspace(np.log10(3), np.log10(12), 10)
SP = 1 - 1 / INV
SEEDS = [0, 1, 2, 3, 4]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
CH = ROOT / "results" / "phase3_conv_chunks"

def run_chunk(seed, device):
    f = CH / f"seed{seed}.json"
    if f.exists():
        print("skip seed", seed); return
    model, info = train(Config(N, M, len(INV)), torch.ones(len(INV), N), torch.tensor(SP, dtype=torch.float32),
                        steps=STEPS, seed=seed, device=device, log_every=5000)
    W = model.W.detach().cpu().numpy()
    np.savez_compressed(CH / f"seed{seed}.npz", W=W, b=model.b.detach().cpu().numpy())
    f.write_text(json.dumps({"seed": seed, "steps": STEPS, "seconds": info["seconds"], "inv_density": INV.tolist(),
                             "final_loss": info["final_loss"], "dims_per_feature": dims_per_feature(W).tolist(),
                             "feature_dimensionality": feature_dimensionality(W).tolist(),
                             "feature_norms": feature_norms(W).tolist()}))
    print(f"seed {seed} done {info['seconds']:.0f}s", flush=True)

def longest_run(row, v, tol=0.02):
    m = np.abs(np.asarray(row) - v) < tol; best = cur = 0
    for flag in m:
        cur = cur + 1 if flag else 0; best = max(best, cur)
    return best

def merge():
    recs = [json.loads((CH / f"seed{s}.json").read_text()) for s in SEEDS]
    dpf = np.array([r["dims_per_feature"] for r in recs])
    targets = {"1/2": .5, "2/5": .4, "3/8": .375, "1/3": 1/3, "2/3": 2/3, "3/4": .75}
    fd = np.array([r["feature_dimensionality"] for r in recs])
    out = {"n": N, "m": M, "steps": STEPS, "seeds": SEEDS, "inv_density": INV.tolist(), "sparsity": SP.tolist(),
           "per_seed": recs, "dims_per_feature_mean": dpf.mean(0).tolist(), "dims_per_feature_std": dpf.std(0).tolist(),
           "plateau_counts_within_0.02": {k: int((np.abs(dpf - v) < 0.02).sum()) for k, v in targets.items()},
           "plateau_longest_run_per_seed": {k: [longest_run(r, v) for r in dpf] for k, v in targets.items()},
           "feature_dim_band_counts_within_0.02": {k: int((np.abs(fd - v) < 0.02).sum()) for k, v in targets.items()},
           "n_feature_values": int(fd.size)}
    # Comparison with the first pass at the nearest sweep points (10k steps, 3 seeds).
    p3 = json.loads((ROOT / "results" / "phase3.json").read_text())
    inv0 = np.array(p3["inv_density"]); m0 = np.array(p3["dims_per_feature_mean_over_seeds"])
    out["first_pass_interp_dims_per_feature"] = np.interp(np.log(INV), np.log(inv0), m0).tolist()
    (ROOT / "results" / "phase3_convergence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["plateau_counts_within_0.02", "plateau_longest_run_per_seed", "feature_dim_band_counts_within_0.02"]}, indent=1))

def main():
    CH.mkdir(parents=True, exist_ok=True); dev = get_device(); t0 = time.time()
    for s in SEEDS: run_chunk(s, dev)
    merge(); subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_phase3_convergence.py")], check=True)
    print(f"total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
