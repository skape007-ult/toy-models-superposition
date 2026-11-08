"""Phase 4: computation in superposition — the absolute-value task.

Model: h = ReLU(W1 x + b1) (n -> m neurons), y' = ReLU(W2 h + b2) (m -> n), target y = |x|,
with x_i = 0 w.p. S else U[-1, 1]. Importance 0.8^i (assumption: paper decays importance but the
exact base for this figure is not recorded in the text; see notes/log.md). n=100, m=40.
For each sparsity, neurons are classified by polysemanticity = 1 - max_i W1[k,i]^2 / sum_i W1[k,i]^2.
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tms.train import get_device, lr_at  # noqa: E402

N, M = 100, 40
SPARSITIES = [0.0, 0.5, 0.8, 0.9, 0.97, 0.99]
SEEDS = [0, 1]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
MONO_THRESH = 0.1  # polysemanticity below this => monosemantic


class AbsModel(nn.Module):
    def __init__(self, I, n, m, device, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W1 = torch.empty(I, m, n); nn.init.xavier_normal_(W1, generator=g)
        W2 = torch.empty(I, n, m); nn.init.xavier_normal_(W2, generator=g)
        self.W1 = nn.Parameter(W1.to(device)); self.b1 = nn.Parameter(torch.zeros(I, m, device=device))
        self.W2 = nn.Parameter(W2.to(device)); self.b2 = nn.Parameter(torch.zeros(I, n, device=device))

    def forward(self, x):
        h = F.relu(torch.einsum("bin,imn->bim", x, self.W1) + self.b1)
        return F.relu(torch.einsum("bim,inm->bin", h, self.W2) + self.b2), h


def train_abs(sparsity, importance, seed, device, steps):
    I = len(sparsity)
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    model = AbsModel(I, N, M, device, seed)
    imp = importance.to(device); sp = sparsity.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, steps, 1e-3)
        shape = (1024, I, N)
        x = (torch.rand(shape, generator=g, device=device) * 2 - 1) * (torch.rand(shape, generator=g, device=device) >= sp[None, :, None])
        out, _ = model(x)
        loss_i = (imp[None] * (x.abs() - out) ** 2).mean(dim=(0, 2))
        opt.zero_grad(set_to_none=True); loss_i.sum().backward(); opt.step()
        if step % 2000 == 0:
            print(f"step {step} loss {loss_i.mean().item():.5f} ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        shape = (8192, I, N)
        x = (torch.rand(shape, generator=g, device=device) * 2 - 1) * (torch.rand(shape, generator=g, device=device) >= sp[None, :, None])
        out, _ = model(x)
        final = (imp[None] * (x.abs() - out) ** 2).mean(dim=(0, 2)).cpu().tolist()
    return model, final, time.time() - t0


def main():
    device = get_device()
    importance = (0.8 ** torch.arange(N)).repeat(len(SPARSITIES), 1)
    out = {"n": N, "m": M, "importance": "0.8^i", "sparsities": SPARSITIES, "seeds": SEEDS, "steps": STEPS,
           "mono_thresh": MONO_THRESH, "runs": []}
    for seed in SEEDS:
        model, final, secs = train_abs(torch.tensor(SPARSITIES), importance, seed, device, STEPS)
        W1 = model.W1.detach().cpu().numpy()  # (I, m, n)
        for j, S in enumerate(SPARSITIES):
            w = W1[j]
            sq = w ** 2
            poly = 1 - sq.max(1) / np.maximum(sq.sum(1), 1e-12)
            active = sq.sum(1) > 1e-3
            out["runs"].append({"seed": seed, "sparsity": S, "final_loss": final[j],
                                "polysemanticity": poly.tolist(), "neuron_norm2": sq.sum(1).tolist(),
                                "n_active_neurons": int(active.sum()),
                                "n_monosemantic": int((active & (poly < MONO_THRESH)).sum()),
                                "n_polysemantic": int((active & (poly >= MONO_THRESH)).sum()),
                                "W1": w.tolist() if seed == SEEDS[0] else None})
        print(f"seed {seed}: {secs:.0f}s", flush=True)
    out["summary"] = [{"sparsity": S,
                       "n_monosemantic_mean": float(np.mean([r["n_monosemantic"] for r in out["runs"] if r["sparsity"] == S])),
                       "n_polysemantic_mean": float(np.mean([r["n_polysemantic"] for r in out["runs"] if r["sparsity"] == S])),
                       "loss_mean": float(np.mean([r["final_loss"] for r in out["runs"] if r["sparsity"] == S]))}
                      for S in SPARSITIES]
    (ROOT / "results" / "phase4.json").write_text(json.dumps(out, indent=1))
    for s in out["summary"]:
        print(s)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_phase4.py")], check=True)


if __name__ == "__main__":
    main()
