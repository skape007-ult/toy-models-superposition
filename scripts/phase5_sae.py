"""Phase 5 (extension b): train sparse autoencoders on the hidden activations of the Phase 1
models and ask whether they recover the true features.

For each Phase 1 sparsity (seed 0 model, n=20, m=5) we sample x from the same distribution,
compute h = W x (5-dim), and train an SAE  f = ReLU(W_e h + b_e),  h_hat = W_d f + b_d  with
dictionary size D and loss ||h - h_hat||^2 + lam * ||f||_1 (decoder columns unit-norm).
Recovery metric: for every true feature that the toy model actually represents
(||W_i|| > 0.5), the max cosine similarity between its direction W_i/||W_i|| and any decoder
column; "recovered" if > 0.9. Also reports precision (fraction of live dictionary elements
whose best-matching true feature has cos > 0.9). All (sparsity, lam) pairs are trained at
once as batched instances.
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms.train import get_device, lr_at  # noqa: E402

LAMS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
DICT = 20
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
BATCH = 4096
REP_THRESH, COS_THRESH = 0.5, 0.9


def main():
    device = get_device()
    z = np.load(ROOT / "results" / "phase1_W.npz")
    W_all, sparsities, seeds = z["W"], z["sparsities"].tolist(), z["seeds"].tolist()
    W0 = torch.tensor(W_all[seeds.index(0)])  # (6, n, m), seed 0
    n_sp, n, m = W0.shape
    cells = [(j, lam) for j in range(n_sp) for lam in LAMS]
    I = len(cells)
    Wt = torch.stack([W0[j] for j, _ in cells]).to(device)                 # (I, n, m)
    sp = torch.tensor([sparsities[j] for j, _ in cells], device=device)
    lam = torch.tensor([l for _, l in cells], device=device)
    torch.manual_seed(0)
    g = torch.Generator(device=device).manual_seed(0)
    # SAE params (batched over I)
    W_e = torch.nn.Parameter(torch.randn(I, DICT, m, device=device) * 0.3)
    b_e = torch.nn.Parameter(torch.zeros(I, DICT, device=device))
    W_d = torch.nn.Parameter(F.normalize(torch.randn(I, m, DICT, device=device), dim=1))
    b_d = torch.nn.Parameter(torch.zeros(I, m, device=device))
    opt = torch.optim.Adam([W_e, b_e, W_d, b_d], lr=3e-3)

    def sample(B):
        shape = (B, I, n)
        x = torch.rand(shape, generator=g, device=device) * (torch.rand(shape, generator=g, device=device) >= sp[None, :, None])
        return torch.einsum("bin,inm->bim", x, Wt)

    t0 = time.time()
    for step in range(STEPS):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, STEPS, 3e-3)
        h = sample(BATCH)
        f = F.relu(torch.einsum("bim,idm->bid", h - b_d[None], W_e) + b_e[None])
        h_hat = torch.einsum("bid,imd->bim", f, W_d) + b_d[None]
        mse = ((h - h_hat) ** 2).sum(-1).mean(0)                                   # (I,)
        l1 = f.sum(-1).mean(0)
        loss = (mse + lam * l1).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            W_d.data = F.normalize(W_d.data, dim=1)
        if step % 1000 == 0:
            print(f"step {step} mse {mse.mean().item():.5f} l1 {l1.mean().item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    # Evaluation
    with torch.no_grad():
        h = sample(16384)
        f = F.relu(torch.einsum("bim,idm->bid", h - b_d[None], W_e) + b_e[None])
        h_hat = torch.einsum("bid,imd->bim", f, W_d) + b_d[None]
        mse = ((h - h_hat) ** 2).sum(-1).mean(0).cpu().numpy()
        var = h.var(0).sum(-1).cpu().numpy()
        l0 = (f > 0).float().sum(-1).mean(0).cpu().numpy()
        live = ((f > 0).float().mean(0) > 1e-4).cpu().numpy()                         # (I, D)
        Wd = W_d.detach().cpu().numpy()                                               # (I, m, D)
        Wtn = Wt.cpu().numpy()
    runs = []
    for c, (j, l) in enumerate(cells):
        Wtrue = Wtn[c]  # (n, m)
        norms = np.linalg.norm(Wtrue, axis=1)
        rep = norms > REP_THRESH
        dirs = Wtrue / np.maximum(norms[:, None], 1e-8)
        cos = dirs @ Wd[c]  # (n, D)
        best_per_feature = cos.max(1)
        best_per_dict = cos.max(0)
        rec = {"sparsity": sparsities[j], "lam": l, "mse": float(mse[c]), "frac_var_unexplained": float(mse[c] / var[c]),
               "l0": float(l0[c]), "n_live_dict": int(live[c].sum()),
               "n_represented_true": int(rep.sum()),
               "n_recovered": int(((best_per_feature > COS_THRESH) & rep).sum()),
               "mean_best_cos_represented": float(best_per_feature[rep].mean()) if rep.any() else None,
               "precision": float((best_per_dict[live[c]] > COS_THRESH).mean()) if live[c].any() else None,
               "best_cos_per_feature": best_per_feature.tolist(), "feature_norms": norms.tolist()}
        runs.append(rec)
    out = {"dict_size": DICT, "lams": LAMS, "steps": STEPS, "batch": BATCH, "sparsities": sparsities,
           "rep_thresh": REP_THRESH, "cos_thresh": COS_THRESH, "device": str(device), "seconds": time.time() - t0,
           "runs": runs}
    # Best lambda per sparsity by number recovered (tie -> lower unexplained variance)
    out["best_per_sparsity"] = []
    for j in range(n_sp):
        rs = [r for r in runs if r["sparsity"] == sparsities[j]]
        b = max(rs, key=lambda r: (r["n_recovered"], -r["frac_var_unexplained"]))
        out["best_per_sparsity"].append({k: b[k] for k in ["sparsity", "lam", "n_represented_true", "n_recovered",
                                                            "mean_best_cos_represented", "precision", "l0", "frac_var_unexplained"]})
        print(out["best_per_sparsity"][-1])
    (ROOT / "results" / "phase5_sae.json").write_text(json.dumps(out, indent=1))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_phase5.py")], check=True)


if __name__ == "__main__":
    main()
