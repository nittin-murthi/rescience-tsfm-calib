#!/usr/bin/env bash
# N-BEATS baseline (slow on CPU: ~0.5-1.5 h per dataset per selection rule).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
for ds in ${DATASETS:-heartrate crime glucose}; do
  for sel in insample holdout; do
    venvs/nbeats/bin/python src/run_nbeats.py --dataset "$ds" --select "$sel"
  done
done
venvs/stats/bin/python src/evaluate.py
