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

## 2026-11-08 15:55 — Benchmark

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

### Phase 1 result (15:58)
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

### Phase 2 expectations (written before running, 16:01)
Paper, "Phase diagram" (n=2, m=1), x = relative importance of feature 1 in [0.1, 10],
y = density 1-S in [1, 0.01]:
- Dense (top): the single dimension goes to whichever feature is more important: feature 1
  "not represented" for relative importance < 1, "dedicated" for > 1, with a sharp boundary
  at relative importance ~1.
- As sparsity grows the superposition region (antipodal W = [a, -a]) opens up from the
  middle (importance ~1) and widens, until at 1-S ~ 0.01 almost all importances are in
  superposition. Boundaries are roughly straight lines on the log-log plot.
- The theoretical diagram (loss of the three candidate solutions) matches the empirical one
  closely; the paper notes the empirical one is noisier near boundaries.
Config: 30x30 grid, 3 seeds, 10k steps, run on CPU concurrently with Phase 3 on MPS
(timing test: 900 instances ~17 steps/s on CPU => ~10 min per seed).

### Phase 3 result (16:13)
- 3 seeds x 40 sparsities, n=200, m=20, 10k steps: 955 s total on MPS (3 checkpointed chunks).
- **Matches**: m/||W||_F^2 = 1.00 for 1/(1-S) <= 1.3, drops sharply between 1.4 and 2.0, and
  sits on a flat plateau at 0.50-0.52 for 1/(1-S) in [2.3, 3.7] (15 of 120 sweep points
  within +-0.02 of 1/2; seed-to-seed std 0.000-0.004). The per-feature scatter shows a razor
  sharp band at D_i = 1/2 (antipodal digons) with all other features at D_i = 0 (dropped).
- **Partially matches**: 2/5 appears only as a short shoulder (4 points within +-0.02;
  per-feature bands at ~0.40 and ~0.44 for 1/(1-S) in [4.1, 4.6]), not a plateau as flat
  as the 1/2 one. 3/8 and 1/3 are crossed smoothly (5 and 4 points nearby, consistent with
  the sampling density, not with a plateau).
- **Differs**: no clusters at 3/4 (tetrahedron) or 2/3 (triangle); the transition from 1 to
  1/2 passes through them without stopping (histogram has single-digit counts there).
- Observation not emphasised in the paper: in the dense regime with *uniform* importance the
  per-feature dimensionality is ~m/n = 0.1 for every feature (the 20-dim subspace is
  degenerate, so the model does not pick 20 features), even though m/||W||_F^2 = 1. At the
  sparse end (1/(1-S) > 40) all 200 features are represented with D_i ~ 0.1 again and
  ||W||_F^2 ~ 300 (feature norms > 1), so m/||W||_F^2 falls to 0.066, well below 1/n*m.
- No hyperparameters were tuned.

### Phase 4 (16:17) — process note and result
- Process slip: I launched scripts/phase4.py before writing the expectations entry (the
  paper's claim for this section: with dense features each neuron is monosemantic, i.e.
  responds to one feature; as sparsity grows neurons become polysemantic, with a mix of
  mono- and polysemantic neurons at intermediate sparsity). The entry below was written
  after seeing the summary numbers, so treat the "expectation" as not pre-registered.
- Assumption: importance 0.8^i, n=100, m=40 neurons, x in [-1,1], target |x|, 2 seeds.
- Result (neurons with polysemanticity < 0.1 counted as monosemantic, mean of 2 seeds):
  S=0: 40 mono / 0 poly; 0.5: 37/3; 0.8: 34.5/5.5; 0.9: 31.5/8.5; 0.97: 5.5/34.5;
  0.99: 0/40. Qualitatively the paper's picture: monosemantic at low sparsity, a sharp
  switch to polysemantic between S=0.9 and S=0.97. 23 s per seed on MPS.

### Phase 5 expectations (written before running, 16:18) — extension (b): SAE on hidden activations
Choice: (b) because it connects the toy model directly to the dictionary-learning / SAE
programme that followed this paper, and the toy model gives a rare case with known ground
truth features to score recovery against.
- Setup: SAE with dictionary 20 (= n) on the 5-dim hidden activations h = W x of the Phase 1
  seed-0 models; L1 coefficients {1e-3 .. 1e-1}; 5000 steps, batch 4096.
- Expected: at S=0 the model keeps only 5 orthogonal features, and the SAE should recover
  those 5 (the other 15 are not present in h at all). At intermediate sparsity (0.9-0.99)
  the 9-11 represented features are in superposition; an SAE with an appropriate L1 should
  recover most of them (cos > 0.9) since the data is genuinely sparse. Too small an L1 should
  give a dense, non-feature-aligned basis; too large an L1 should kill dictionary elements.
  At S=0.999 the activations are so sparse that 5000 steps may be too few to recover the
  weakly-represented features, so recovery may fall short there.

### Phase 5 result (16:20)
- 30 SAEs (6 sparsities x 5 L1 values) trained at once, 5000 steps, ~2 min on MPS.
- Best L1 per sparsity, features recovered (cos > 0.9) / features the toy model represents:
  S=0: 2/5; 0.7: 5/5; 0.9: 9/9; 0.97: 10/10; 0.99: 10/11; 0.999: 9/13. Mean best cosine for
  represented features 0.99 at S=0.7-0.97.
- **As expected**: in the superposition regime (S=0.9-0.99) the SAE recovers essentially every
  feature the toy model represents, with an L1 of 0.03-0.1; too small an L1 gives a dense,
  unaligned dictionary (0-3 features, precision ~0.1). At S=0.999 recovery drops to 9/13 and the
  mean cosine to 0.92 (the weakest features fire in ~4 of 16k samples).
- **Not expected / worth noting**: at S=0 the SAE recovers only 2 of the 5 features even though
  they are orthogonal and h = W x is exactly 5-dimensional. With dense inputs any rotation of
  the basis reconstructs h equally well and the L1 penalty has little to bite on (L0 stays at
  6-12 of 20), so the dictionary is not identifiable. Sparsity in the *data* is what makes the
  SAE find the true basis, which is the premise of the SAE programme.
- Precision (fraction of live dictionary elements aligned with a true feature) peaks at
  0.83-0.85: the dictionary has 20 slots but only 9-11 true features are present, and the
  extra live elements split or duplicate features.

## 2026-11-08 16:22 — Docs site and Pages
- docs/index.html is a single self-contained page (canvas drawing, no CDNs) reading docs/data.js,
  which scripts/export_docs.py writes from results/phase1.json and results/phase3.json.
- GitHub Pages enabled via the API with build_type=workflow; .github/workflows/pages.yml deploys
  docs/ on pushes touching docs/** (first run failed only because Pages was not yet enabled;
  re-dispatched manually).
- Dead end: the in-app browser preview could not open localhost (navigation denied) and node is not
  installed, so the local page could not be checked. After the first successful deploy the public
  Pages URL did open: both sliders work (checked S=0.99 and 1/(1-S)=3.26 programmatically and by
  screenshot). First version had near-black bars on a dark canvas in dark mode; fixed by giving the
  canvases a white background (second deploy).

## 2026-11-08 16:35 — Phase 2 timing
- Empirical part: 30x30 grid x 3 seeds (2700 models), 10k steps, ~7.5 min per seed on CPU
  (run on CPU so that Phase 3 could use MPS at the same time). Done 16:25.
- The "theoretical" diagram (constrained optimisation of 3 solution families on a 20k-sample
  Monte-Carlo estimate, 1500 Adam steps each) is far slower than I estimated: ~0.66 s/step
  under contention => ~35-50 min. Left running rather than killed, because the script only
  writes results after both parts finish. Fixed for future runs: scripts/phase2.py now
  checkpoints the empirical results to results/phase2_empirical_ckpt.npz before the theory
  step (the running process predates this change; its outputs are otherwise identical).

## 2026-11-08 16:45 — Phase 2 first result and a bug in the theory step
- Empirical diagram (majority of 3 seeds): not represented 17%, dedicated 18%, superposition 65%
  of cells. First theory diagram: 6% / 6% / 88%, agreement with empirical only 0.77, and all the
  disagreement was theory saying "superposition" where the models say single-feature.
- Cause: in the theory optimisation the biases started at exactly 0, so an unrepresented
  feature's output was ReLU(0) whose PyTorch gradient is 0; the single-feature families could
  never learn the optimal constant prediction (E[x] of the dropped feature) and were
  under-rated. Fixed by initialising b at 0.1; theory step rerun from the saved empirical
  checkpoint on MPS (float32).
- Also seen in the empirical weights: besides the antipodal W = [a, -a] solution there are
  *same-sign* solutions W ~ [1.05, 0.3-0.4] with a negative bias on the stronger feature, in
  a band around relative importance 0.4-0.9 (and its mirror 1.1-2.5). Seeds frequently
  disagree there (some find antipodal, some same-sign, some single-feature), which produces
  the speckled vertical stripes in the empirical diagram. The paper's three-family theory
  has no such family; this multi-stability is reported, not tuned away.
