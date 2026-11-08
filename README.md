# Toy Models of Superposition — a reproduction

A from-scratch reproduction of Elhage et al. (2022), *Toy Models of Superposition*
([transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html)),
in PyTorch on an Apple M3 laptop. Every number below is read from `results/*.json`, every figure is
produced by a script in `scripts/`, and `scripts/run_all.sh` regenerates all of it (about one hour).
Interactive view of the Phase 1 and Phase 3 results: [docs/index.html](docs/index.html) (static, no CDNs).

## Summary

| Phase | What | Headline result | vs. paper |
|---|---|---|---|
| 1 | n=20, m=5, importance 0.7^i, S ∈ {0,…,0.999} | Features with ‖W_i‖>0.5 grow 5 → 13 as S goes 0 → 0.999; W^T W goes identity-block → antipodal pairs → dense | Matches qualitatively; fewer features than the paper at S=0.999 |
| 2 | n=2, m=1 phase diagram, 30×30 grid, 3 seeds | Three regions as in the paper; empirical vs three-family theory agree on 0.86 of cells (0.94 where seeds are unanimous) | Matches; extra multi-stability and a same-sign solution family not in the paper's theory |
| 3 | n=200, m=20, uniform importance, 40 sparsities × 3 seeds | Flat plateau at exactly 1/2 dims-per-feature for 1/(1−S) ∈ [2.3, 3.7]; 2/5 only a shoulder; no 3/4 or 2/3 | 1/2 plateau matches; higher-order polytopes do not |
| 4 | |x| task, n=100, m=40 neurons | 40/40 neurons monosemantic at S=0, 0/40 at S=0.99; switch between S=0.9 and 0.97 | Matches |
| 5 | SAE (dict 20) on the Phase 1 hidden activations | Recovers 9/9, 10/10, 10/11 true features at S=0.9/0.97/0.99 (cos>0.9); only 2/5 at S=0 (dense data not identifiable) | Extension; consistent with SAE premise |

Compute: MPS was chosen over CPU after a benchmark on identical data (loss trajectories agree to
1e-7 relative; 5.8 vs 3.3 steps/s on the Phase 3 config). Total training time for all phases is
under 45 minutes.

## Reproduction

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
pytest -q                      # 5 tests: shapes, sparsity statistics, hand-computed loss, metrics, top-m sanity check
scripts/run_all.sh             # benchmark + all phases + docs data
```

Individual phases: `python scripts/phase1.py`, `phase3.py`, `phase2.py [GRID] [SEEDS] [STEPS]`,
`phase4.py`, `phase5_sae.py`; each has a matching `plot_*.py` that only reads JSON. `TMS_DEVICE=cpu`
forces the CPU. Phase 3 is checkpointed per seed in `results/phase3_chunks/`.

## Setup common to all phases

Model: `x' = ReLU(W^T W x + b)`, `W ∈ R^{n×m}`. Features: `x_i = 0` with probability `S`, otherwise
`U[0,1]`. Loss: importance-weighted squared error, averaged over batch and features (the paper sums
over features; for fixed `n` that is a constant factor and does not change the optimum). Many models
are trained at once by giving every tensor a leading *instance* dimension, as the original notebook
does; the total loss is the sum over instances so gradients are independent.

Optimiser: AdamW, lr 1e-3, batch 1024, 10 000 steps, 5 % linear warm-up then cosine decay to 0,
Xavier-normal init, zero bias init. Seeds are fixed; three seeds per configuration unless stated.
No hyperparameters were tuned for any phase (`notes/log.md` records the expectations written before
each run and what came out).

## Phase 1 — basic superposition

![phase1](figures/phase1_superposition.png)

n=20, m=5, importance `0.7^i`. Top row: `W^T W`; strip: bias; bars: `‖W_i‖` coloured by the paper's
superposition measure `Σ_{j≠i}(Ŵ_i·W_j)²` (black = orthogonal, gold = fully superposed). Seed 0 shown;
counts are over 3 seeds (`results/phase1.json`).

| S | 1/(1−S) | features with ‖W_i‖ > 0.5 (3 seeds) | of which superposed | m/‖W‖²_F | loss |
|---|---|---|---|---|---|
| 0 | 1 | 5, 5, 5 | 0 | 1.00 | 2.3e-3 |
| 0.7 | 3.3 | 5, 7, 7 | 2.7 | 0.80 | 1.9e-3 |
| 0.9 | 10 | 9, 10, 9 | 8.7 | 0.54 | 4.0e-4 |
| 0.97 | 33 | 10, 10, 10 | 10 | 0.49 | 7.3e-5 |
| 0.99 | 100 | 11, 11, 11 | 11 | 0.41 | 2.0e-5 |
| 0.999 | 1000 | 13, 13, 14 | 13.3 | 0.34 | 1.3e-6 |

**Matches the paper.** Dense inputs give exactly the five most important features, orthogonal, with
the other fifteen at zero norm (and bias +0.5, the mean of a dropped feature). From S=0.9 the model
packs more features than dimensions, first as antipodal pairs (−1 off-diagonals in `W^T W`), then
densely; biases of superposed features go negative, as the paper explains (the negative bias
suppresses interference).

**Differs from the paper.** At S=0.999 only 13–14 of 20 features clear the 0.5-norm threshold; the
paper's figure shows nearly all 20. The least important features have importance 0.7^14…0.7^19 ≈
0.007…0.001 and are active in about one sample per batch, so 10k steps give them very little gradient.
I did not train longer to close the gap.

## Phase 2 — phase diagram (n=2, m=1)

![phase2](figures/phase2_phase_diagram.png)

Two features, one hidden dimension. Feature 0 has importance 1; the x-axis is the importance of
feature 1 relative to it (0.1–10, log), the y-axis the feature density 1−S (1 down to 0.01, log). 30×30
grid, 3 seeds per cell (2700 models trained in one batch, 10k steps each), `results/phase2.json`.
Each model is classified from its weights `W = [W_0, W_1]` with threshold 0.2: feature 1 *not
represented* (|W_1| < 0.2), *dedicated dimension* (|W_0| < 0.2 ≤ |W_1|), or *superposition* (both ≥ 0.2).
The right-most panel is the paper-style theoretical diagram: for every cell the best achievable loss
within each of the three solution families `W ∝ [1,0]`, `[0,1]`, `[1,−1]` (scale and both biases free)
is found by optimising on a fixed 20 000-sample Monte-Carlo estimate of the expected loss, and the
cheapest family wins.

| | not represented | dedicated | superposition |
|---|---|---|---|
| empirical (majority of 3 seeds), fraction of cells | 0.17 | 0.18 | 0.65 |
| theory, fraction of cells | 0.12 | 0.12 | 0.76 |

Agreement between the empirical majority and theory: **0.86** of cells (per seed
0.86, 0.87, 0.86); on the 747 cells where all three seeds
agree with each other, agreement with theory is 0.94.

**Matches the paper.** Dense inputs give a sharp vertical boundary at relative importance 1: the
single dimension goes to whichever feature matters more. Superposition opens as a wedge from
importance ≈ 1 as density falls and, below density ≈ 0.1, covers nearly the whole importance range.
The theory diagram has the same three regions with straight boundaries on log axes, as in the paper,
and the two diagrams agree everywhere except near the boundaries.

**Differs from the paper / caveats.** (i) The empirical diagram has speckled vertical stripes of
single-feature solutions inside the superposition region, around relative importance 0.4–0.6 and
its mirror 2–3. In those cells different seeds land in different local optima (17 % of all cells
are not unanimous), and 26 % of all trained models end with a loss more than 1e-3 above the best
family loss, so part of the disagreement is optimisation rather than the model's true preference.
(ii) 13 % of trained models sit in a fourth kind of solution the three-family theory does not
contain: both weights with the *same* sign (e.g. `W ≈ [1.05, 0.35]`) and a negative bias on the
stronger feature. They are counted as "superposition" above. (iii) The first version of the theory
computation initialised the biases at zero, where `ReLU(0)` has zero gradient, which under-rated the
single-feature families (agreement 0.77); the log records the fix. The theory itself is a numerical
optimisation on Monte-Carlo samples rather than the paper's closed-form expressions.

## Phase 3 — feature geometry

![phase3](figures/phase3_geometry.png)

n=200, m=20, uniform importance, 40 values of 1/(1−S) log-spaced in [1, 100], 3 seeds, 10k steps
(`results/phase3.json`, 16 min on MPS). Top: dimensions per feature `m/‖W‖²_F`. Bottom: every
feature's dimensionality `D_i = ‖W_i‖² / Σ_j(Ŵ_i·W_j)²`.

| target fraction | 1 | 3/4 | 2/3 | 1/2 | 2/5 | 3/8 | 1/3 |
|---|---|---|---|---|---|---|---|
| sweep points (of 120) within ±0.02 | 12 | 0 | 0 | 15 | 4 | 5 | 4 |

**Matches the paper.** The curve is 1.00 up to 1/(1−S) ≈ 1.3, drops steeply, and sits on a flat
plateau at 0.50–0.52 for 1/(1−S) ∈ [2.3, 3.7] in all three seeds (seed-to-seed std ≤ 0.004). The
per-feature scatter in that range is a single sharp band at D_i = 1/2: the model represents 40
features as 20 antipodal pairs and drops the other 160 (D_i = 0). This is the "sticky" digon plateau.

**Partially matches.** 2/5 (pentagons) is a shoulder rather than a plateau: for 1/(1−S) ∈ [4.1, 4.6]
the per-feature bands sit at ≈0.40 and ≈0.44 and the curve lingers near 0.39–0.43 for two sweep
points before continuing down. 3/8 and 1/3 are crossed at the speed the sampling grid implies, with
no visible stickiness.

**Differs from the paper.** There are no clusters at 3/4 (tetrahedra) or 2/3 (triangles): the
transition from 1 to 1/2 passes straight through them (single-digit histogram counts,
`figures/phase3_dimensionality_hist.png`). Two further observations: (i) in the dense regime with
*uniform* importance every feature has D_i ≈ m/n = 0.1 even though `m/‖W‖²_F = 1`, because any
20-dimensional subspace is an equally good solution and the model does not single out 20 features;
(ii) at the sparse end all 200 features are represented with norms above 1, so `m/‖W‖²_F` falls to
0.066. The paper's x-axis also stops at 100 and its curve likewise keeps falling.

## Phase 4 — computation in superposition (absolute value)

![phase4](figures/phase4_neurons.png)

`y = |x|` with `x_i ∈ U[−1,1]` (zero w.p. S), one hidden ReLU layer of m=40 neurons, n=100 features,
importance `0.8^i` (assumed; the paper decays importance but the exact base is not stated in the
text), 2 seeds. Polysemanticity of neuron k is `1 − max_i W1[k,i]² / Σ_i W1[k,i]²`; below 0.1 counts
as monosemantic (`results/phase4.json`).

| S | 0 | 0.5 | 0.8 | 0.9 | 0.97 | 0.99 |
|---|---|---|---|---|---|---|
| monosemantic neurons (mean of 2 seeds) | 40 | 37 | 34.5 | 31.5 | 5.5 | 0 |

**Matches the paper**: dense features give one feature per neuron (pairs of neurons compute
`ReLU(x)` and `ReLU(−x)`), intermediate sparsity gives a mix, and high sparsity makes every neuron
polysemantic. Process note: this phase was launched before its expectation entry was written in
`notes/log.md`; the entry says so.

## Phase 5 — extension: sparse autoencoder on the hidden activations

![phase5](figures/phase5_sae_recovery.png)

**Why this extension.** Sparse autoencoders (SAEs) are the main tool that grew out of this paper for
pulling superposed features out of real networks; the toy model is one of the few settings where the
ground-truth features are known, so recovery can be scored exactly.

**Setup.** For each Phase 1 seed-0 model (n=20, m=5) sample `x` from the same distribution, take the
5-dimensional hidden activations `h = W x`, and train an SAE `f = ReLU(W_e (h − b_d) + b_e)`,
`ĥ = W_d f + b_d` with 20 dictionary elements, unit-norm decoder columns, loss `‖h − ĥ‖² + λ‖f‖₁`,
λ ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, 5000 steps, batch 4096. A true feature counts as *recovered* if
some decoder column has cosine > 0.9 with its direction `W_i/‖W_i‖`; only features the toy model
represents (`‖W_i‖ > 0.5`) can be recovered. Precision is the fraction of live dictionary elements
whose best-matching true feature has cosine > 0.9. (`results/phase5_sae.json`)

| S | best λ | represented | recovered | mean best cos | precision | L0 | var. unexplained |
|---|---|---|---|---|---|---|---|
| 0 | 0.03 | 5 | 2 | 0.871 | 0.11 | 7.72 | 0.014 |
| 0.7 | 0.03 | 5 | 5 | 0.991 | 0.62 | 4.00 | 0.002 |
| 0.9 | 0.03 | 9 | 9 | 0.991 | 0.68 | 1.60 | 0.001 |
| 0.97 | 0.03 | 10 | 10 | 0.992 | 0.83 | 0.59 | 0.001 |
| 0.99 | 0.1 | 11 | 10 | 0.981 | 0.85 | 0.23 | 0.016 |
| 0.999 | 0.1 | 13 | 9 | 0.918 | 0.73 | 0.04 | 0.029 |

**Result.** In the superposition regime (S = 0.9–0.99) the SAE recovers all, or all but one, of the
9–11 features the toy model packs into 5 dimensions, with cosines of 0.98–0.99, at λ = 0.03–0.1. Too
small a λ gives a dense, unaligned dictionary (0–3 features, precision ≈ 0.1). Recovery degrades at
S = 0.999 (9 of 13, mean cosine 0.92), where the weakest features fire a handful of times per
evaluation batch. The instructive failure is S = 0: the five features are orthogonal and `h` is
exactly five-dimensional, yet only 2 of 5 are recovered. With dense inputs every rotation of the basis
reconstructs `h` equally well and the L1 term has little to select on (L0 stays at 6–12 of 20), so the
dictionary is not identifiable. Sparsity in the data, not merely a sparsity penalty, is what lets an
SAE find the true basis.

## Deviations and limitations

- Loss is averaged over features rather than summed (constant rescaling for fixed n; Adam is scale invariant).
- Training data is sampled on the training device with a seeded device generator, so CPU and MPS runs
  are each reproducible but do not see identical data streams. The benchmark compares them on identical CPU-sampled data.
- 10k steps for every phase. The Phase 1 S=0.999 shortfall and possibly the missing high-order
  polytopes in Phase 3 may be convergence effects; neither was investigated by training longer.
- Phase 3 uses n=200, m=20; the paper's figure uses a comparable but not identical configuration, and
  the exact numeric plateau extents are not expected to transfer.
- Phase 4 importance base (0.8) is an assumption. Phase 2 thresholds (|W| < 0.2) and Phase 1/5
  thresholds (0.5 norm, 0.9 cosine) are my choices, stated in the JSON.
- Three seeds per configuration (two for Phase 4); errors are seed ranges, not confidence intervals.

## Repository layout

```
tms/            model.py (ToyModel, data, loss), train.py (loop, device), analysis.py (metrics), plotting.py
scripts/        phase1.py … phase5_sae.py, plot_*.py, benchmark.py, export_docs.py, run_all.sh
tests/          pytest suite
results/        JSON written by the scripts (+ small .npz with weights)
figures/        PNGs written by plot_*.py
docs/           static site (index.html + data.js), deployed by .github/workflows/pages.yml
notes/          log.md (dated decisions, expectations vs. outcomes), progress.md (resume point)
```

## Citation

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R.,
Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., Olah, C. (2022).
*Toy Models of Superposition*. Transformer Circuits Thread.
https://transformer-circuits.pub/2022/toy_model/index.html
