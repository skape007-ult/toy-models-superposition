"""Training loop (AdamW, linear warmup + cosine decay) and device selection."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

from .model import Config, ToyModel, generate_batch, weighted_mse

ROOT = Path(__file__).resolve().parent.parent
BENCH_FILE = ROOT / "results" / "benchmark.json"


def get_device() -> torch.device:
    """Device choice: $TMS_DEVICE overrides; else results/benchmark.json; else cpu."""
    env = os.environ.get("TMS_DEVICE")
    if env:
        return torch.device(env)
    if BENCH_FILE.exists():
        choice = json.loads(BENCH_FILE.read_text()).get("choice", "cpu")
        if choice == "mps" and not torch.backends.mps.is_available():
            choice = "cpu"
        return torch.device(choice)
    return torch.device("cpu")


def lr_at(step: int, steps: int, lr: float, warmup_frac: float = 0.05) -> float:
    warm = max(1, int(steps * warmup_frac))
    if step < warm:
        return lr * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    return lr * 0.5 * (1 + math.cos(math.pi * t))


def train(
    cfg: Config,
    importance: torch.Tensor,  # (I, n)
    sparsity: torch.Tensor,  # (I,) or (I, n)
    steps: int = 10_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: torch.device | str | None = None,
    log_every: int = 0,
    model: ToyModel | None = None,
) -> tuple[ToyModel, dict]:
    """Train I independent instances at once. Returns the model and a summary dict.

    The total loss is the *sum* over instances of each instance's mean loss, so
    each instance's gradient is exactly what it would be if trained alone.
    """
    device = torch.device(device) if device is not None else get_device()
    torch.manual_seed(seed)
    # Sample data on-device: on MPS, CPU sampling + copy costs more than the model step.
    gen = torch.Generator(device=device).manual_seed(seed)
    model = model if model is not None else ToyModel(cfg, device=device, seed=seed)
    importance = importance.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    t0 = time.time()
    history = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x = generate_batch(batch_size, cfg.n_instances, cfg.n_features, sparsity, gen, device)
        loss_per_inst = weighted_mse(x, model(x), importance)
        loss = loss_per_inst.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            history.append((step, loss_per_inst.detach().cpu().tolist()))
            print(f"step {step:6d}  mean loss {loss_per_inst.mean().item():.5f}  ({time.time()-t0:.0f}s)", flush=True)
    # Final evaluation on a fresh, larger batch.
    with torch.no_grad():
        x = generate_batch(8192, cfg.n_instances, cfg.n_features, sparsity, gen, device)
        final = weighted_mse(x, model(x), importance).cpu().tolist()
    info = {
        "steps": steps, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "seed": seed, "device": str(device), "seconds": time.time() - t0,
        "schedule": "linear warmup 5% then cosine decay to 0",
        "final_loss": final, "history": history,
    }
    return model, info
