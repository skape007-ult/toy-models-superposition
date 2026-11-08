# Progress (read this first when resuming)

Status key: [x] done, [~] in progress, [ ] next.

- [x] Setup: venv (.venv, Python 3.12), requirements.txt pinned, repo layout
- [x] Phase 0: core model/training/tests
- [x] Benchmark MPS vs CPU -> results/benchmark.json (choice: mps)
- [x] Phase 1 (n=20, m=5): results/phase1.json, figures/phase1_superposition.png
- [x] Phase 3 (geometry, plateaus): results/phase3.json, figures/phase3_*.png
- [ ] README
- [~] Phase 2 (30x30 grid, 3 seeds) running on CPU: scripts/phase2.py -> results/phase2.json
- [~] Phase 4 running (scripts/phase4.py); Phase 5 (SAE, scripts/phase5_sae.py) next; docs/index.html written, data via scripts/export_docs.py

How to resume: `source .venv/bin/activate && pytest -q`, then run the next unchecked
script under scripts/ (each is idempotent / checkpointed; see notes/log.md for context).
