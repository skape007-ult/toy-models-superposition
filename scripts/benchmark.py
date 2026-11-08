"""Benchmark MPS vs CPU on one representative config and record the choice.

Also checks that MPS gives the same loss trajectory as CPU (within tolerance)
on identical data, so that a faster device is only chosen if it is also correct.
Writes results/benchmark.json.
"""
import json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tms import Config, ToyModel, generate_batch, weighted_mse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "benchmark.json"


def run(device, cfg, importance, sparsity, steps, batch, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ToyModel(cfg, device=device, seed=seed)
    imp = importance.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for step in range(steps):
        x = generate_batch(batch, cfg.n_instances, cfg.n_features, sparsity, gen, device)
        loss_i = weighted_mse(x, model(x), imp)
        loss = loss_i.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            losses.append(loss.item())
    if device == "mps":
        torch.mps.synchronize()
    return time.time() - t0, losses


def main():
    results = {"configs": []}
    configs = [
        ("phase1-like", Config(20, 5, 24), 300),
        ("phase3-like", Config(200, 20, 120), 100),
    ]
    for name, cfg, steps in configs:
        importance = (0.7 ** torch.arange(cfg.n_features)).repeat(cfg.n_instances, 1)
        sparsity = torch.linspace(0, 0.99, cfg.n_instances)
        entry = {"name": name, "n": cfg.n_features, "m": cfg.n_hidden, "instances": cfg.n_instances,
                 "steps": steps, "batch": 1024}
        t_cpu, l_cpu = run("cpu", cfg, importance, sparsity, steps, 1024)
        entry["cpu_seconds"] = t_cpu
        entry["cpu_steps_per_sec"] = steps / t_cpu
        if torch.backends.mps.is_available():
            t_mps, l_mps = run("mps", cfg, importance, sparsity, steps, 1024)
            entry["mps_seconds"] = t_mps
            entry["mps_steps_per_sec"] = steps / t_mps
            # Same seed => same data (sampled on CPU) and same init => trajectories should agree.
            diffs = [abs(a - b) / max(abs(a), 1e-8) for a, b in zip(l_cpu, l_mps)]
            entry["max_rel_loss_diff_cpu_vs_mps"] = max(diffs)
            entry["mps_correct"] = max(diffs) < 1e-2
        results["configs"].append(entry)
        print(json.dumps(entry, indent=1), flush=True)
    mps_ok = all(c.get("mps_correct", False) for c in results["configs"])
    speedups = [c["cpu_seconds"] / c["mps_seconds"] for c in results["configs"] if "mps_seconds" in c]
    results["mps_speedups"] = speedups
    # Use MPS only if it is correct and faster on the large (phase-3-like) config,
    # which dominates total compute.
    use_mps = mps_ok and len(speedups) > 0 and speedups[-1] > 1.1
    results["choice"] = "mps" if use_mps else "cpu"
    results["torch"] = torch.__version__
    OUT.write_text(json.dumps(results, indent=1))
    print("choice:", results["choice"])


if __name__ == "__main__":
    main()
