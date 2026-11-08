"""Item 3: Phase 1 at S=0.999 with 1x, 5x and 20x the steps (3 seeds). Writes results/phase1_convergence.json."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms import Config, train, get_device
from tms.analysis import feature_norms, superposition, dims_per_feature

S, N, M = 0.999, 20, 5
SEEDS = [0, 1, 2]
STEP_LIST = [int(a) for a in sys.argv[1:]] or [10_000, 50_000, 200_000]
REP, SUP = 0.5, 0.2
OUT = ROOT / "results" / "phase1_convergence.json"

def main():
    device = get_device()
    out = json.loads(OUT.read_text()) if OUT.exists() else {"sparsity": S, "n": N, "m": M, "seeds": SEEDS,
                                                             "represented_thresh": REP, "superpos_thresh": SUP, "runs": []}
    done = {(r["steps"], r["seed"]) for r in out["runs"]}
    importance = (0.7 ** torch.arange(N)).repeat(len(SEEDS), 1)
    for steps in STEP_LIST:
        if all((steps, s) in done for s in SEEDS):
            print("skip", steps); continue
        # one instance per seed, trained in one batch; per-instance init differs because the xavier draw
        # is per instance, and the data stream is shared (same as Phase 1 within a seed-chunk).
        cfg = Config(N, M, len(SEEDS))
        t0 = time.time()
        model, info = train(cfg, importance, torch.full((len(SEEDS),), S), steps=steps, seed=0, device=device, log_every=10_000)
        W, b = model.W.detach().cpu().numpy(), model.b.detach().cpu().numpy()
        norms, sp = feature_norms(W), superposition(W)
        for i, seed in enumerate(SEEDS):
            rep = norms[i] > REP
            out["runs"].append({"steps": steps, "seed": seed, "final_loss": info["final_loss"][i],
                                "n_represented": int(rep.sum()), "n_in_superposition": int((rep & (sp[i] > SUP)).sum()),
                                "n_norm_above_0.1": int((norms[i] > 0.1).sum()),
                                "norms": norms[i].tolist(), "bias": b[i].tolist(), "dims_per_feature": float(dims_per_feature(W)[i]),
                                "seconds": time.time() - t0})
        OUT.write_text(json.dumps(out, indent=1))
        print(steps, [r["n_represented"] for r in out["runs"] if r["steps"] == steps], f"{time.time()-t0:.0f}s", flush=True)
    out["summary"] = [{"steps": st, "n_represented": [r["n_represented"] for r in out["runs"] if r["steps"] == st],
                       "n_norm_above_0.1": [r["n_norm_above_0.1"] for r in out["runs"] if r["steps"] == st],
                       "loss_mean": float(np.mean([r["final_loss"] for r in out["runs"] if r["steps"] == st]))}
                      for st in sorted({r["steps"] for r in out["runs"]})]
    OUT.write_text(json.dumps(out, indent=1))
    for s in out["summary"]: print(s)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_phase1_convergence.py")], check=True)

if __name__ == "__main__":
    main()
