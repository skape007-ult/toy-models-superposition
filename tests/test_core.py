import math

import numpy as np
import torch

from tms import Config, ToyModel, generate_batch, weighted_mse, train
from tms.analysis import feature_norms, superposition, feature_dimensionality, dims_per_feature


def test_shapes():
    cfg = Config(n_features=7, n_hidden=3, n_instances=4)
    model = ToyModel(cfg, seed=0)
    assert model.W.shape == (4, 7, 3) and model.b.shape == (4, 7)
    x = generate_batch(16, 4, 7, torch.zeros(4), torch.Generator().manual_seed(0))
    assert x.shape == (16, 4, 7)
    out = model(x)
    assert out.shape == (16, 4, 7)
    assert (out >= 0).all()
    loss = weighted_mse(x, out, torch.ones(4, 7))
    assert loss.shape == (4,)


def test_sparsity_statistics():
    g = torch.Generator().manual_seed(1)
    S = torch.tensor([0.0, 0.5, 0.9])
    x = generate_batch(20000, 3, 10, S, g)
    zero_frac = (x == 0).float().mean(dim=(0, 2))
    assert torch.allclose(zero_frac, S, atol=0.02)
    nz = x[:, 0]  # dense instance: values ~ U[0,1]
    assert abs(nz.mean().item() - 0.5) < 0.01 and nz.min() >= 0 and nz.max() <= 1


def test_loss_matches_hand_computation():
    cfg = Config(n_features=2, n_hidden=1, n_instances=1)
    model = ToyModel(cfg, seed=0)
    with torch.no_grad():
        model.W[:] = torch.tensor([[[1.0], [-1.0]]])  # antipodal pair
        model.b[:] = torch.tensor([[0.0, -0.5]])
    x = torch.tensor([[[0.8, 0.3]], [[0.0, 0.6]]])  # (B=2, I=1, n=2)
    out = model(x)
    # hand: h = x0 - x1 ; x'0 = relu(h) ; x'1 = relu(-h - 0.5)
    exp = torch.tensor([[[0.5, 0.0]], [[0.0, 0.1]]])
    assert torch.allclose(out, exp, atol=1e-6)
    I = torch.tensor([[1.0, 0.5]])
    loss = weighted_mse(x, out, I)
    # per-sample weighted squared errors: sample 1: 1*(0.3)^2 + 0.5*(0.3)^2 = 0.135
    #                                     sample 2: 1*0 + 0.5*(0.5)^2 = 0.125
    hand = (0.135 + 0.125) / (2 * 2)  # mean over batch and over the 2 features
    assert math.isclose(loss.item(), hand, rel_tol=1e-6)


def test_metrics_on_known_weights():
    # Two orthogonal unit features and one feature exactly superposed on the first.
    W = np.array([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    assert np.allclose(feature_norms(W), [[1, 1, 1]])
    assert np.allclose(superposition(W), [[1, 0, 1]])
    assert np.allclose(feature_dimensionality(W), [[0.5, 1.0, 0.5]])
    assert np.allclose(dims_per_feature(W), [2 / 3])


def test_dense_model_learns_top_m_features():
    """With S=0 and m<n, the model should devote its m dims to the m most important features."""
    torch.manual_seed(0)
    n, m = 6, 3
    cfg = Config(n_features=n, n_hidden=m, n_instances=2)
    importance = (0.7 ** torch.arange(n)).repeat(2, 1)
    model, info = train(cfg, importance, torch.zeros(2), steps=10000, batch_size=1024, seed=0, device="cpu")
    norms = feature_norms(model.W)
    for i in range(2):
        top = set(np.argsort(-norms[i])[:m].tolist())
        assert top == {0, 1, 2}, norms[i]
        assert norms[i, :m].min() > 0.95 and norms[i, m:].max() < 0.1, norms[i]
        # Dense regime: the learned features should be close to orthogonal (little superposition).
        assert superposition(model.W)[i, :m].max() < 0.05
