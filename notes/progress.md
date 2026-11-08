# Progress (read this first when resuming)

Status key: [x] done, [~] in progress, [ ] next.

- [x] Setup: venv (.venv, Python 3.12), requirements.txt pinned, repo layout
- [~] Phase 0: core model/training/tests
- [ ] Benchmark MPS vs CPU -> results/benchmark.json
- [ ] Phase 1 (n=20, m=5)
- [ ] Phase 3 (geometry, plateaus)
- [ ] README
- [ ] Phase 2 (phase diagram, reduced grid)
- [ ] Phase 4, 5, docs site

How to resume: `source .venv/bin/activate && pytest -q`, then run the next unchecked
script under scripts/ (each is idempotent / checkpointed; see notes/log.md for context).
