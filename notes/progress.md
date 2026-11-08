# Progress (read this first when resuming)

Status key: [x] done, [~] in progress, [ ] next.

- [x] Setup: venv (.venv, Python 3.12), requirements.txt pinned, repo layout
- [x] Phase 0: core model/training/tests
- [x] Benchmark MPS vs CPU -> results/benchmark.json (choice: mps)
- [x] Phase 1 (n=20, m=5): results/phase1.json, figures/phase1_superposition.png
- [x] Phase 3 (geometry, plateaus): results/phase3.json, figures/phase3_*.png
- [ ] README
- [~] Phase 2 (30x30 grid, 3 seeds): empirical done (results/phase2.json); theory step being rerun after a bias-init bug fix (see log 16:45)
- [x] Phase 4: results/phase4.json, figures/phase4_neurons.png
- [x] Phase 5 (SAE): results/phase5_sae.json, figures/phase5_sae_recovery.png
- [~] README.md drafted; Phase 2 section pending
- [x] docs site deployed: https://skape007-ult.github.io/toy-models-superposition/ (Pages enabled, workflow green)

How to resume: `source .venv/bin/activate && pytest -q`, then run the next unchecked
script under scripts/ (each is idempotent / checkpointed; see notes/log.md for context).
