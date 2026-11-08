# Work log

All times US/Eastern. Entries are appended as work happens; expected outcomes are
written *before* each phase is run.

## 2026-11-08 15:46 — Setup

- Machine: Apple M3, 8 cores, 8 GB RAM. Python 3.12 via uv venv; torch 2.15.0, numpy 2.5.4,
  matplotlib 3.11.2, pytest 9.2.0 (pinned in requirements.txt). MPS reports available.
- Design: all instances (sparsity / importance / seed combinations) are trained in one
  tensor with a leading instance dimension, exactly as the original notebook does. Loss is
  the sum over instances of each instance's mean importance-weighted squared error, so the
  gradient per instance is independent of the others.
- Optimizer: AdamW lr 1e-3, batch 1024, 10k steps, 5% linear warmup then cosine decay to 0
  (the original notebook used Adam(W) at lr 1e-3, batch 1024, 10k steps with constant /
  linear / cosine options). Xavier-normal init for W, zero init for b, as in the notebook.
- Deviation noted up front: my loss averages over features instead of summing. That is a
  constant rescaling by 1/n for a given n and does not change the optimum.

### Phase 0 expectations (written before running)
- Shape tests trivially pass. Loss hand-computation on a 2x1 antipodal model matches.
- Sanity check: n=6, m=3, S=0, importance 0.7^i: the model should learn the 3 most important
  features with near-unit norm and near-orthogonal directions, and drop the other 3.
  (Paper, "Demonstrating superposition": with dense features the model is essentially PCA /
  keeps the top-m features.)
