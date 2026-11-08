"""Item 5: n=3, m=2 phase diagram for the third feature (relative importance r), 15x15 grid, 3 seeds.
Writes results/phase2_n3m2.json and figures/phase2_n3m2.png."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from tms import Config, train, get_device
from tms.analysis import feature_norms, superposition, feature_dimensionality
from scipy_free_mode import mode_axis0

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
GRID = int(ARGS[0]) if len(ARGS) > 0 else 15
N_SEEDS = int(ARGS[1]) if len(ARGS) > 1 else 3
STEPS = int(ARGS[2]) if len(ARGS) > 2 else 10_000
RI = np.logspace(-1, 1, GRID); DENS = np.logspace(0, -2, GRID)
NORM_T, SUP_T = 0.2, 0.1
NAMES = ["not represented", "dedicated dimension", "superposition"]

def classify(W):
    norms, sp = feature_norms(W), superposition(W)
    out = np.full(W.shape[0], 2); out[(norms[:, 2] >= NORM_T) & (sp[:, 2] < SUP_T)] = 1; out[norms[:, 2] < NORM_T] = 0
    return out

def geometry_tags(fdims):
    """fdims: (seeds, G, G, 3). triangle: all D_i ~ 2/3. pair_plus_one: sorted D ~ (1/2, 1/2, 1), i.e. an
    antipodal pair plus an orthogonal dedicated feature (interference-free under ReLU)."""
    tri = (np.abs(fdims - 2 / 3) < 0.05).all(-1)
    sd = np.sort(fdims, axis=-1)
    pair1 = (np.abs(sd[..., 0] - 0.5) < 0.05) & (np.abs(sd[..., 1] - 0.5) < 0.05) & (np.abs(sd[..., 2] - 1.0) < 0.05)
    return tri, pair1


def finish(out):
    fdims = np.array(out["feature_dims_per_seed"]); classes = np.array(out["classes_per_seed"]); maj = np.array(out["class_majority"])
    tri, pair1 = geometry_tags(fdims)
    out["frac_models_triangle"] = float(tri.mean()); out["triangle_per_cell_count"] = tri.sum(0).tolist()
    out["frac_models_pair_plus_one"] = float(pair1.mean()); out["pair_plus_one_per_cell_count"] = pair1.sum(0).tolist()
    sparse = np.array(out["density"]) < 0.1
    out["frac_models_pair_plus_one_density_below_0.1"] = float(pair1[:, sparse].mean())
    out["frac_models_triangle_density_below_0.1"] = float(tri[:, sparse].mean())
    (ROOT / "results" / "phase2_n3m2.json").write_text(json.dumps(out, indent=1))
    cmap = ListedColormap(["#f0f0f0", "#4c72b0", "#dd8452"])
    ns = len(out["classes_per_seed"])
    fig, axes = plt.subplots(1, ns + 3, figsize=(3.3 * (ns + 3), 3.8), sharey=True)
    panels = [("majority", maj)] + [(f"seed {s}", classes[s]) for s in range(ns)]
    for ax, (t, c) in zip(axes, panels):
        ax.pcolormesh(RI, DENS, c, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest"); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"feature 2: {t}", fontsize=9); ax.set_xlabel("relative importance of feature 2")
    for ax, (t, c) in zip(axes[ns + 1:], [("# seeds: antipodal pair + 1 orthogonal\n(D = ½, ½, 1)", pair1.sum(0)), ("# seeds: triangle (all D ≈ ⅔)", tri.sum(0))]):
        pm = ax.pcolormesh(RI, DENS, c, cmap="Purples", vmin=0, vmax=ns, shading="nearest")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t, fontsize=9); ax.set_xlabel("relative importance of feature 2")
    fig.colorbar(pm, ax=axes[-1])
    axes[0].set_ylabel("feature density 1 − S"); axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(NAMES)], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("n=3, m=2: features 0 and 1 have importance 1; feature 2 is swept", fontsize=11)
    fig.savefig(ROOT / "figures" / "phase2_n3m2.png", dpi=130, bbox_inches="tight")
    print({k: out[k] for k in ["class_fractions_majority", "frac_cells_all_seeds_agree", "frac_models_triangle",
                               "frac_models_pair_plus_one", "frac_models_pair_plus_one_density_below_0.1"]})


def main():
    if "--replot" in sys.argv:
        finish(json.loads((ROOT / "results" / "phase2_n3m2.json").read_text())); return
    t0 = time.time(); device = get_device()
    cells = [(r, d) for d in DENS for r in RI]; I = len(cells)
    imp = torch.tensor([[1.0, 1.0, r] for r, _ in cells], dtype=torch.float32)
    sp = torch.tensor([1 - d for _, d in cells], dtype=torch.float32)
    classes, Ws, losses, fdims = [], [], [], []
    for seed in range(N_SEEDS):
        model, info = train(Config(3, 2, I), imp, sp, steps=STEPS, seed=seed, device=device, log_every=2500)
        W = model.W.detach().cpu().numpy()
        classes.append(classify(W).reshape(GRID, GRID)); Ws.append(W.reshape(GRID, GRID, 3, 2))
        losses.append(np.array(info["final_loss"]).reshape(GRID, GRID)); fdims.append(feature_dimensionality(W).reshape(GRID, GRID, 3))
        print(f"seed {seed} done {info['seconds']:.0f}s", flush=True)
    classes, fdims = np.array(classes), np.array(fdims)
    maj = mode_axis0(classes)
    out = {"n": 3, "m": 2, "grid": GRID, "seeds": N_SEEDS, "steps": STEPS, "norm_thresh": NORM_T, "sup_thresh": SUP_T,
           "rel_importance": RI.tolist(), "density": DENS.tolist(), "class_names": NAMES,
           "classes_per_seed": classes.tolist(), "class_majority": maj.tolist(),
           "class_fractions_majority": {n: float((maj == k).mean()) for k, n in enumerate(NAMES)},
           "frac_cells_all_seeds_agree": float((classes == classes[0]).all(0).mean()),
           "W_per_seed": np.array(Ws).tolist(), "loss_per_seed": np.array(losses).tolist(),
           "feature_dims_per_seed": fdims.tolist(), "device": str(device), "seconds_total": time.time() - t0}
    finish(out)


if __name__ == "__main__":
    main()
