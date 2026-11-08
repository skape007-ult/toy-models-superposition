# Progress (read this first when resuming)

Status key: [x] done, [~] in progress, [ ] next.

- [x] Setup: venv (.venv, Python 3.12), requirements.txt pinned, repo layout
- [x] Phase 0: core model/training/tests
- [x] Benchmark MPS vs CPU -> results/benchmark.json (choice: mps)
- [x] Phase 1 (n=20, m=5): results/phase1.json, figures/phase1_superposition.png
- [~] Phase 3 (geometry, plateaus): scripts/phase3.py running, chunks in results/phase3_chunks
- [ ] README
- [ ] Phase 2 (phase diagram, reduced grid)
- [ ] Phase 4, 5, docs site

How to resume: `source .venv/bin/activate && pytest -q`, then run the next unchecked
script under scripts/ (each is idempotent / checkpointed; see notes/log.md for context).
