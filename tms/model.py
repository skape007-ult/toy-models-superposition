"""Core toy model: x' = ReLU(W^T W x + b), batched over an *instance* dimension.

Shapes (I = n_instances, n = n_features, m = n_hidden, B = batch):
    W: (I, n, m)      b: (I, n)
    x: (B, I, n)  ->  h = x W : (B, I, m)  ->  x' = ReLU(h W^T + b) : (B, I, n)

Every instance is an independent model (its own importance / sparsity / seed);
batching them into one tensor is how the original work trained many models at
once, and is much faster than separate processes for models this small.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    n_features: int
    n_hidden: int
    n_instances: int


class ToyModel(nn.Module):
    def __init__(self, cfg: Config, device: torch.device | str = "cpu", seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        # Xavier-normal init as in the original notebook.
        W = torch.empty(cfg.n_instances, cfg.n_features, cfg.n_hidden)
        nn.init.xavier_normal_(W, generator=gen)
        self.W = nn.Parameter(W.to(device))
        self.b = nn.Parameter(torch.zeros(cfg.n_instances, cfg.n_features, device=device))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bin,inm->bim", x, self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        out = torch.einsum("bim,inm->bin", h, self.W) + self.b
        return F.relu(out)


def generate_batch(
    batch_size: int,
    n_instances: int,
    n_features: int,
    sparsity: torch.Tensor,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Synthetic features: each x_i is 0 with prob S (per instance), else U[0,1].

    `sparsity` has shape (I,) or (I, n). Sampling is done on CPU with the given
    generator for reproducibility, then moved to `device`.
    """
    shape = (batch_size, n_instances, n_features)
    vals = torch.rand(shape, generator=generator)
    mask = torch.rand(shape, generator=generator) >= sparsity.to("cpu").reshape(1, n_instances, -1)
    return (vals * mask).to(device)


def weighted_mse(x: torch.Tensor, x_hat: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    """Importance-weighted MSE, returned *per instance*: shape (I,).

    L_I = mean_{batch, features} importance[I, i] * (x - x')^2
    (Averaging over features instead of summing only rescales the loss; it keeps
    numbers comparable across different n.)
    """
    err = importance.reshape(1, *importance.shape) * (x - x_hat) ** 2
    return err.mean(dim=(0, 2))
