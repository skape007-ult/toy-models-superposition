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

### Phase 2 final result (17:02)
- Theory rerun with the bias fix (584 s on MPS): theory fractions 12% / 12% / 76%; agreement
  with the empirical majority 0.86 (0.94 on the 747 cells where all three seeds agree).
- Remaining disagreement is at the boundaries and in the striped multi-stable band; 26% of
  trained models have a loss >1e-3 above the best family's loss (mean gap 0.0019, max 0.023),
  i.e. they are in a worse local optimum, and 13% of models are same-sign solutions.
- Compared with the paper: the three regions and boundary shapes match; the stripes and the
  same-sign family are not in the paper's figure.

# Follow-up pass (branch exp/followup, worktree ../tms-followup)

## 2026-11-08 17:34 — Item 1: definitions check
What would change which claim: if D_i in tms/analysis.py did not sum over j = i, or if
dims_per_feature were not m/||W||_F^2, every Phase 3 number (the 1/2 plateau, the per-feature
bands, the histogram) and the Phase 1 "m/||W||_F^2" column would have to be recomputed and the
README's Phase 3 "Matches" paragraph rewritten. Expectation from reading the code: the sum in
feature_dimensionality runs over all j including i (the diagonal is not removed, unlike in
superposition()), so an antipodal pair gives D = 1/2 and a regular pentagon in 2D gives 2/5.
Result: both definitions are as stated; the diagonal term is included. New tests pass (antipodal
1/2, pentagon 2/5, triangle 2/3, tetrahedron 3/4, lone feature 1; m/||W||_F^2 cases). No README
change needed.

## 2026-11-08 17:38 — Item 2: Phase 2 theory spot-check and the same-sign family
What would change which claim: (a) if the theory's feature-0-only loss at density 1, r=0.16 is
not r/24 = 0.00667 (mean over 2 features of r * Var(U[0,1]) = r/12), the theory diagram and the
0.86 agreement number are wrong and would be recomputed. (b) For the same-sign models: if their
trained loss is *below* the best three-family loss at the same cell, the theory misses a real
solution and the README's "fourth kind of solution" claim becomes "a lower-loss solution the
three-family theory misses"; if it is *above*, they are local minima and the README should say
"a higher-loss local minimum", and the 0.86 disagreement is partly optimisation failure.
Expectation: (a) holds within Monte-Carlo error (~1%). (b) I expect them to be higher-loss
local minima, because at those cells other seeds found antipodal solutions and the mean loss
gap over all models was already positive (0.0019).

## 2026-11-08 17:36 — Item 3: Phase 1 S=0.999 convergence
What would change which claim: README Phase 1 "Differs" says only 13-14 of 20 features clear
the 0.5-norm threshold at S=0.999 and attributes it to too little gradient in 10k steps. If
5x / 20x steps bring the count to ~20, the claim becomes "a convergence effect, resolved with
longer training"; if the count stays at 13-14, the attribution is wrong and should be replaced
by "the optimum under this importance/sparsity genuinely leaves the weakest features out".
Expectation: the count rises with steps but may not reach 20 at 200k steps, because the
importance of the last features (~1e-3) times their activity (1e-3) is ~1e-6 of the loss.
Setup: scripts/phase1_convergence.py, S=0.999 only, 3 seeds, steps in {10k, 50k, 200k}, same
optimiser/schedule (cosine over the full run), same thresholds as Phase 1.
Result (results/phase2_samesign.json): 344 of 2700 trained models are same-sign solutions
(relative importance 0.49-2.04, all densities). Their loss exceeds the best three-family loss
in 93% of cases (median gap 0.0056, median relative gap 11.8x); in all 137 cells where another
seed found a different solution, the same-sign one is worse. Verdict: higher-loss local
minimum, not a solution the theory misses. README Phase 2 wording to change accordingly.
The r/24 spot check passes (new test; theory family-0 loss at density 1, r=0.16 within 2% of
0.00667, and the closed-form loss check on the model itself). Also: the three-family theory
only ever beats the trained antipodal models by MC noise (median gap -7e-6).

## 2026-11-08 17:37 — Item 4: Phase 3 convergence rerun around 1/(1-S) in [3, 12]
What would change which claim: README Phase 3 says 2/5 is "a shoulder rather than a plateau"
(1-2 consecutive points within +-0.02) and that there are "no clusters at 3/4 or 2/3". If, with
5x steps and 5 seeds, the run within +-0.02 of 2/5 grows to >= 3 consecutive points, the
"Partially matches" paragraph becomes "matches at 5x steps"; if 3/4 or 2/3 bands appear in the
per-feature scatter, the "Differs" paragraph is withdrawn. If nothing changes, the first-pass
wording stands and gains "robust to 5x longer training".
Expectation: the 2/5 shoulder sharpens a little (the 1/2 plateau was already converged, std 0),
but the 3/4 and 2/3 structures will not appear here because they belong to the transition
1 -> 1/2 at 1/(1-S) in [1.4, 2], which is outside the requested [3, 12] window.
Sizing (benchmark, MPS, n=200, m=20): 12 instances 0.020 s/step, 60 instances 0.086 s/step.
Choice: 10 log-spaced points in [3, 12] x 5 seeds, 50k steps, one checkpointed chunk per seed
(12 instances would be 17 min; 10 points ~14 min per chunk, ~70 min total, under the 90-min cap).

## 2026-11-08 17:38 — Item 5: n=3, m=2 phase diagram (15x15, 3 seeds)
What would change which claim: nothing in the README yet (this diagram is new); it adds a
Phase 2 subsection. Setup: features 0 and 1 have importance 1, feature 2 has relative
importance r in [0.1, 10] (x), density 1-S in [1, 0.01] (y), 15x15 grid, 3 seeds, 10k steps,
CPU (MPS is busy with items 3-4). Classification of feature 2 by its weights: not represented
(||W_2|| < 0.2); dedicated dimension (||W_2|| >= 0.2 and superposition measure
sum_{j!=2}(W_2_hat . W_j)^2 < 0.1); superposition otherwise. No theory diagram for this one
(the family enumeration is larger; noted as not done).
Expectation from the paper's n=3, m=2 figure: dense + r < 1 -> feature 2 dropped (features 0, 1
take the two dimensions); dense + r > 1 -> feature 2 dedicated and one of the equal features
dropped (ties broken by seed); as sparsity grows, superposition appears first near r ~ 1 and
then takes over, with triangle-like arrangements (three features in two dimensions, D_i = 2/3).

### Item 3 result (17:56) — results/phase1_convergence.json
- S=0.999, 3 seeds, features with ||W_i|| > 0.5: 10k steps [12, 13, 13]; 50k [16, 14, 13];
  200k [13, 14, 14]. Features with ||W_i|| > 0.1: [17, 17, 16] -> [17, 17, 17] -> [19, 18, 18].
  Final loss 1.23e-6 -> 1.08e-6 -> 1.01e-6.
- So the count above 0.5 does NOT rise with 20x more steps; what rises is the number of weakly
  represented features (norm 0.1-0.5). The first-pass attribution ("too little gradient in
  10k steps") is not supported: under this importance decay the optimum (as far as 200k steps
  reveal it) keeps the 6-7 least important features at partial norm. README Phase 1 "Differs"
  paragraph to be rewritten accordingly; the 10k numbers stay visible.
- Note the 10k baseline here ([12,13,13]) differs slightly from Phase 1's [13,13,14] because
  the batch composition (3 instances of the same S) changes the per-instance init/data draw.

### Item 5 result (17:57) — results/phase2_n3m2.json, figures/phase2_n3m2.png
- 15x15 x 3 seeds, 10k steps, CPU, 1035 s. Majority classes for feature 2: not represented
  15%, dedicated 33%, superposition 52%; only 55% of cells unanimous across seeds.
- Dense region matches the expectation: r < 1 -> feature 2 dropped, r > 1 -> feature 2 takes a
  dimension (one of the equal features dropped), boundary at r ~ 1.
- Sparse region: NOT the triangle. Zero models have all D_i ~ 2/3. Instead 70% of all models
  (95% of those at density < 0.1) are "antipodal pair + one orthogonal feature" (D = 1/2, 1/2, 1),
  which under ReLU is interference-free. Which feature is the odd one out is seed-dependent, so
  feature 2 is classified "dedicated" or "superposition" essentially at random in that region;
  that is the speckle, not a failure of convergence. This is consistent with the paper's remark
  that antipodal pairs carry no positive interference; the triangle (2/3) was my expectation,
  not an observation reported for this exact setup.

### Item 4 result (18:18) — results/phase3_convergence.json, figures/phase3_convergence.png
- 10 points in 1/(1-S) in [3, 12] x 5 seeds, 50k steps, 2409 s on MPS (5 checkpointed chunks).
- m/||W||_F^2 at 50k steps differs from the 10k first pass (interpolated) by -0.012 to +0.004;
  seed-to-seed std <= 0.011. The curve is slightly *lower* (more superposition) at 50k.
- Longest run of consecutive points within +-0.02 of 2/5, per seed: [0, 1, 1, 0, 0] (first pass: 1-2).
  The 2/5 shoulder does not widen; it is, if anything, weaker. 3/8: [1, 1, 1, 1, 1], 1/3: [1, 1, 1, 1, 1] (single
  crossings). 2/3 and 3/4: [0, 0, 0, 0, 0] / [0, 0, 0, 0, 0].
- Per-feature bands (values within +-0.02 of the fraction, per 1000 feature values, in this window):
  1/2: 48 (first pass 33), 2/5: 18 (20), 3/8: 23 (28), 1/3: 29 (41),
  2/3: 0 (0), 3/4: 0 (0). At 1/(1-S) ~ 4.1 the scatter shows discrete bands at 0.5, 0.44, 0.40 and 0.36,
  i.e. a mixture of digons and small polytopes rather than a single pentagon phase.
- Conclusion: the first-pass Phase 3 wording stands and is robust to 5x longer training and 5 seeds;
  the 2/5 plateau is not a convergence artefact of the first pass. README "Partially matches"
  and "Differs" paragraphs get this appended; the original figure and numbers stay.
