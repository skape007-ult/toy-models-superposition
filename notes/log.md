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

## 2026-11-08 16:00 — Benchmark

- results/benchmark.json: MPS vs CPU on identical data/init. Loss trajectories agree to
  ~1e-7 relative. Phase-1-sized config (n=20, m=5, 24 instances): identical speed (~114 steps/s).
  Phase-3-sized config (n=200, m=20, 120 instances): MPS 5.8 steps/s vs CPU 3.3 steps/s.
  **Choice: MPS.** `TMS_DEVICE=cpu` overrides.
- Profiling showed CPU-side sampling + host->device copy (0.18 s/step) cost more than the
  model step itself (0.08 s). Training now samples on the training device with a seeded
  device generator (0.06 s/step). Consequence: results are reproducible per device but
  CPU and MPS runs do not see identical data streams.

### Phase 1 expectations (written before running)
Paper, "Demonstrating superposition", n=20, m=5, importance 0.7^i:
- S=0: exactly the 5 most important features represented, orthogonally (W^T W is the
  identity on a 5x5 block, zero elsewhere); bias ~0 for those features, slightly negative
  for dropped ones is not required.
- S=0.7..0.9: a few extra features appear with small norm, antipodal pairs start to form
  (off-diagonal -1 entries in W^T W); bias becomes negative for superposed features.
- S=0.97..0.99: most of the 20 features are represented, heavily superposed; W^T W has
  many off-diagonal entries, feature norms spread; negative biases.
- S=0.999: essentially all 20 features represented in 5 dims.
Checking: count of features with ||W_i|| > 0.5 should rise monotonically from 5 to ~20.

### Phase 1 result (16:08)
- Ran 3 seeds x 6 sparsities, 10k steps, 80 s on MPS. results/phase1.json.
- Features with ||W_i|| > 0.5 (mean over seeds): S=0: 5.0, 0.7: 6.3, 0.9: 9.3, 0.97: 10.0,
  0.99: 11.0, 0.999: 13.3. Monotone increase as expected; W^T W goes identity-block ->
  antipodal pairs -> dense off-diagonal structure; bias turns negative for superposed features.
- **Differs**: at S=0.999 only ~13 of 20 features pass the 0.5-norm threshold (the least
  important ones, importance 0.7^14..0.7^19 ~ 0.007..0.001, stay at small norm). The paper's
  figure shows nearly all 20 bars. Likely cause: at S=0.999 each feature is active in ~1 of
  1024 samples per step, and the least important features carry ~1e-3 of the loss weight, so
  10k steps give them very little gradient. Not tuned away; noted as a deviation.
- Bias for *dropped* features at S=0 is +0.5 (= E[x]), the optimal constant prediction.

### Phase 3 expectations (written before running)
Paper, "Feature dimensionality" / "Geometry" (uniform importance): dimensions per feature
m/||W||_F^2 vs 1/(1-S) should fall from 1 (no superposition, dense) through a sharp drop to
~1/2 (antipodal digons), with "sticky" plateaus at 1/2, then ~2/5 (pentagons), with
tetrahedra (3/4) and triangles (2/3) also visible as per-feature dimensionality clusters,
and square antiprisms (3/8). The per-feature dimensionality scatter should show horizontal
bands at these fractions rather than a smooth spread. The paper noted the plateaus are
visible but noisy; plateaus at 1/2 and 2/5 are the clearest. Config here: n=200, m=20,
40 sparsity points logspaced 1/(1-S) in [1, 100], 3 seeds, 10k steps, batch 1024.
Benchmark says ~0.14 s/step for 120 instances => ~8 min per 40-instance chunk, ~25 min total.
