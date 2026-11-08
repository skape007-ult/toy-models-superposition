"""Metrics from the paper, computed on W of shape (I, n, m) (numpy or torch)."""
from __future__ import annotations

import numpy as np


def to_np(W):
    return W.detach().cpu().numpy() if hasattr(W, "detach") else np.asarray(W)


def feature_norms(W) -> np.ndarray:
    """||W_i|| per feature: (I, n)."""
    return np.linalg.norm(to_np(W), axis=-1)


def superposition(W, eps: float = 1e-12) -> np.ndarray:
    """Paper's 'superposition' measure for feature i: sum_{j != i} (W_i_hat . W_j)^2.  (I, n)"""
    W = to_np(W)
    norms = np.linalg.norm(W, axis=-1, keepdims=True)
    W_hat = W / np.maximum(norms, eps)
    dots = np.einsum("inm,ijm->inj", W_hat, W)  # (I, n, n): W_hat_i . W_j
    sq = dots ** 2
    idx = np.arange(W.shape[1])
    sq[:, idx, idx] = 0.0
    return sq.sum(-1)


def feature_dimensionality(W, eps: float = 1e-12) -> np.ndarray:
    """D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2   (sum includes j = i).  (I, n)"""
    W = to_np(W)
    norms2 = (W ** 2).sum(-1)
    W_hat = W / np.maximum(np.sqrt(norms2)[..., None], eps)
    denom = (np.einsum("inm,ijm->inj", W_hat, W) ** 2).sum(-1)
    return norms2 / np.maximum(denom, eps)


def dims_per_feature(W) -> np.ndarray:
    """m / ||W||_F^2 per instance: (I,). The paper's 'dimensions per feature'."""
    W = to_np(W)
    return W.shape[-1] / np.maximum((W ** 2).sum(axis=(1, 2)), 1e-12)


def gram(W) -> np.ndarray:
    """W^T W in the paper's convention: (I, n, n) matrix of feature dot products."""
    W = to_np(W)
    return np.einsum("inm,ijm->inj", W, W)


def classify_2x1(W, b, thresh: float = 0.2) -> np.ndarray:
    """Phase-2 classification for n=2, m=1 models. W: (I, 2, 1).

    Returns ints per instance: 0 = feature 1 (the sweep feature) not represented,
    1 = feature 1 has a dedicated dimension (feature 0 dropped), 2 = superposition
    (both features have non-negligible weight).
    """
    W = to_np(W)[:, :, 0]
    a, c = np.abs(W[:, 0]), np.abs(W[:, 1])
    out = np.full(W.shape[0], 2, dtype=int)
    out[c < thresh] = 0
    out[(a < thresh) & (c >= thresh)] = 1
    return out
