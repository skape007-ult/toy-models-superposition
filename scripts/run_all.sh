#!/usr/bin/env bash
# Regenerates every result, figure and the docs data from scratch (~1 h on an M3 with MPS).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
pytest -q
python scripts/benchmark.py            # writes results/benchmark.json and picks the device
python scripts/phase1.py               # results/phase1.json, phase1_W.npz, figures/phase1_*.png
python scripts/phase3.py               # results/phase3.json (checkpointed per seed), figures/phase3_*.png
python scripts/phase2.py 30 3 10000    # results/phase2.json, figures/phase2_*.png
python scripts/phase4.py               # results/phase4.json, figures/phase4_neurons.png
python scripts/phase5_sae.py           # results/phase5_sae.json, figures/phase5_sae_recovery.png
python scripts/export_docs.py          # docs/data.js for the static site
