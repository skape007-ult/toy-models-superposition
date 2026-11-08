# Progress (read this first when resuming)

Status key: [x] done, [~] in progress, [ ] next.

- [x] Setup: venv (.venv, Python 3.12), requirements.txt pinned, repo layout
- [x] Phase 0: core model/training/tests
- [x] Benchmark MPS vs CPU -> results/benchmark.json (choice: mps)
- [x] Phase 1 (n=20, m=5): results/phase1.json, figures/phase1_superposition.png
- [x] Phase 3 (geometry, plateaus): results/phase3.json, figures/phase3_*.png
- [x] Phase 2 (30x30 grid, 3 seeds): results/phase2.json, figures/phase2_*.png (n=3, m=2 variant NOT done)
- [x] Phase 4: results/phase4.json, figures/phase4_neurons.png
- [x] Phase 5 (SAE): results/phase5_sae.json, figures/phase5_sae_recovery.png
- [x] README.md complete for all phases
- [x] docs site deployed: https://skape007-ult.github.io/toy-models-superposition/ (Pages enabled, workflow green)

How to resume: `source .venv/bin/activate && pytest -q`, then run the next unchecked
script under scripts/ (each is idempotent / checkpointed; see notes/log.md for context).

## Possible next steps (not started)
- Phase 2 for n=3, m=2 (needs a classification scheme for the third feature and a theory with more families).
- Train Phase 1 at S=0.999 longer (or with larger batch) to test whether the missing least-important features appear.
- Add the same-sign solution family to the Phase 2 theory and see whether agreement rises.
- Phase 3 at the paper's larger size (e.g. n=400, m=30) and denser sampling around 1/(1-S) in [4, 8] to look for the 2/5 plateau.
