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

## Follow-up pass (branch exp/followup)
- [x] Item 1: definition tests (tests/test_core.py) — no README change
- [x] Item 2: r/24 test; same-sign family = higher-loss local minimum (results/phase2_samesign.json); README updated
- [x] Item 3: Phase 1 S=0.999 at 10k/50k/200k (results/phase1_convergence.json, figures/phase1_convergence.png); README updated
- [x] Item 4: Phase 3 convergence rerun (results/phase3_convergence.json, figures/phase3_convergence.png); README updated
- [x] Item 5: n=3, m=2 15x15 (results/phase2_n3m2.json, figures/phase2_n3m2.png); README updated
- [ ] Open PR, merge gate

## Possible next steps (not started)
- Theory diagram for n=3, m=2 (needs enumeration of more solution families).
- Phase 1 at S=0.999 with a different importance curve (e.g. 0.9^i) to see whether all 20 features appear, as a check on the attribution in the README.
- Phase 3 at the paper's larger size (e.g. n=400, m=30); the 2/5 shoulder did not widen with 5x steps at n=200, m=20, so size is the remaining variable.
