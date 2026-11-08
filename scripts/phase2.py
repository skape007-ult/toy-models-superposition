"""Phase 2: phase diagram for n=2, m=1.

Sweep relative importance of feature 1 (feature 0 has importance 1) and feature density
1-S on a log grid; several seeds per cell. Classify each trained model as
  0 = feature 1 not represented   (|W_1| small)
  1 = feature 1 gets the dimension (|W_0| small, |W_1| large)
  2 = superposition                (both large; antipodal)
Also computes a "theoretical" diagram: for each cell, the best loss achievable within each
of the three solution families (W=[a,0], W=[0,a], W=[a,-a], biases free) is found by
optimising on a large fixed Monte-Carlo sample of the data distribution; the family with the
lowest loss is the predicted phase. (The paper does the same comparison analytically.)

Usage: python scripts/phase2.py [GRID] [SEEDS] [STEPS]
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms import Config, train, get_device  # noqa: E402
from tms.analysis import classify_2x1  # noqa: E402

GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
REL_IMPORTANCE = np.logspace(-1, 1, GRID)   # importance of feature 1 relative to feature 0
DENSITY = np.logspace(0, -2, GRID)          # 1 - S, from dense (1) to sparse (0.01)
THRESH = 0.2


def empirical(device):
    cells = [(r, d) for d in DENSITY for r in REL_IMPORTANCE]  # row-major: density rows, importance cols
    I = len(cells)
    importance = torch.tensor([[1.0, r] for r, _ in cells])
    sparsity = torch.tensor([1 - d for _, d in cells], dtype=torch.float32)
    classes, Ws, bs, losses = [], [], [], []
    for seed in range(N_SEEDS):
        model, info = train(Config(2, 1, I), importance, sparsity, steps=STEPS, seed=seed, device=device,
                            log_every=2500)
        W = model.W.detach().cpu().numpy()
        classes.append(classify_2x1(W, None, THRESH).reshape(GRID, GRID))
        Ws.append(W[:, :, 0].reshape(GRID, GRID, 2)); bs.append(model.b.detach().cpu().numpy().reshape(GRID, GRID, 2))
        losses.append(np.array(info["final_loss"]).reshape(GRID, GRID))
        print(f"seed {seed} done ({info['seconds']:.0f}s)", flush=True)
    return np.array(classes), np.array(Ws), np.array(bs), np.array(losses)


def theoretical(n_samples=20_000, steps=1500, device=None, rel_importance=None, density=None):
    """Best loss per family per cell, by optimising (a, b0, b1) on a fixed MC sample.

    Biases start at +0.1, not 0: with b = 0 the output of an unrepresented feature is
    ReLU(0), whose gradient is 0 in PyTorch, so the bias could never move (first version of
    this script had that bug and under-estimated the single-feature families).
    """
    device = torch.device(device) if device is not None else get_device()
    torch.manual_seed(0)
    g = torch.Generator(device=device).manual_seed(123)
    rel_importance = REL_IMPORTANCE if rel_importance is None else np.asarray(rel_importance)
    density = DENSITY if density is None else np.asarray(density)
    cells = [(r, d) for d in density for r in rel_importance]
    I = len(cells)
    G = (len(density), len(rel_importance))
    imp = torch.tensor([[1.0, r] for r, _ in cells], dtype=torch.float32, device=device)  # (I, 2)
    dens = torch.tensor([d for _, d in cells], dtype=torch.float32, device=device)  # (I,)
    x = torch.rand(n_samples, I, 2, generator=g, device=device) * (torch.rand(n_samples, I, 2, generator=g, device=device) < dens[None, :, None])
    fam_dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], device=device)
    best = []
    for fd in fam_dirs:
        a = torch.ones(I, device=device, requires_grad=True)
        b = torch.full((I, 2), 0.1, device=device, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=2e-2)
        for _ in range(steps):
            W = a[:, None] * fd[None, :]                        # (I, 2)
            h = (x * W[None]).sum(-1)                            # (B, I)
            out = torch.relu(h[:, :, None] * W[None] + b[None])  # (B, I, 2)
            loss_i = (imp[None] * (x - out) ** 2).mean(dim=(0, 2))
            opt.zero_grad(); loss_i.sum().backward(); opt.step()
        best.append(loss_i.detach().cpu())
        print(f"  family {fd.tolist()} done", flush=True)
    best = torch.stack(best, 0)  # (3, I)
    return best.argmin(0).numpy().reshape(*G), best.numpy().reshape(3, *G)


def main():
    t0 = time.time()
    device = get_device()
    ckpt = ROOT / "results" / "phase2_empirical_ckpt.npz"
    if ckpt.exists():
        z = np.load(ckpt)
        classes, Ws, bs, losses = z["classes"], z["Ws"], z["bs"], z["losses"]
        print("loaded empirical checkpoint", flush=True)
    else:
        classes, Ws, bs, losses = empirical(device)
        np.savez_compressed(ckpt, classes=classes, Ws=Ws, bs=bs, losses=losses)
    print("computing theoretical diagram (20k MC samples, 3 families x 1500 steps)", flush=True)
    theo_cls, theo_loss = theoretical()
    ckpt.unlink(missing_ok=True)
    # majority vote over seeds; also agreement fraction with theory
    sys.path.insert(0, str(ROOT / "scripts")); from scipy_free_mode import mode_axis0  # noqa
    maj = mode_axis0(classes)
    agree = float((maj == theo_cls).mean())
    agree_per_seed = [float((c == theo_cls).mean()) for c in classes]
    # Extra diagnostics: seed instability, same-sign ("positive") superposition, and the gap between
    # the trained loss and the best family loss (positive gap => trained model in a worse local optimum).
    unanimous = (classes == classes[0]).all(0)
    same_sign = ((np.sign(Ws[..., 0]) == np.sign(Ws[..., 1])) & (np.abs(Ws[..., 0]) > THRESH) & (np.abs(Ws[..., 1]) > THRESH))
    theo_best = theo_loss.min(0)
    gap = losses - theo_best[None]
    diagnostics = {"frac_cells_all_seeds_agree": float(unanimous.mean()),
                   "frac_models_same_sign_superposition": float(same_sign.mean()),
                   "frac_models_loss_above_theory_by_1e-3": float((gap > 1e-3).mean()),
                   "mean_loss_gap": float(gap.mean()), "max_loss_gap": float(gap.max())}
    print(diagnostics)
    out = {"n": 2, "m": 1, "grid": GRID, "seeds": N_SEEDS, "steps": STEPS, "thresh": THRESH,
           "rel_importance": REL_IMPORTANCE.tolist(), "density": DENSITY.tolist(),
           "class_names": ["not represented", "dedicated dimension", "superposition"],
           "classes_per_seed": classes.tolist(), "class_majority": maj.tolist(),
           "theory_class": theo_cls.tolist(), "theory_family_loss": theo_loss.tolist(),
           "agreement_majority_vs_theory": agree, "agreement_per_seed": agree_per_seed,
           "diagnostics": diagnostics, "theory_note": "biases initialised at 0.1; 20k MC samples; 1500 Adam steps per family",
           "W_per_seed": Ws.tolist(), "b_per_seed": bs.tolist(), "loss_per_seed": losses.tolist(),
           "class_fractions_majority": {name: float((maj == k).mean()) for k, name in
                                        enumerate(["not represented", "dedicated dimension", "superposition"])},
           "seconds_total": time.time() - t0, "device": str(device)}
    (ROOT / "results" / "phase2.json").write_text(json.dumps(out, indent=1))
    print(f"agreement with theory: majority {agree:.3f}, per seed {agree_per_seed}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_phase2.py")], check=True)
    print(f"phase 2 total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
