#!/usr/bin/env bash
# Round 2: full paper N-BEATS grid ([Paper App. A.2, Table 3], 64 configs) x 3 seeds x 2 selection readings x
# 3 datasets, for a CUDA or Apple-silicon (MPS) machine. See RUN_ON_GPU.md.
# Replicates Adler et al., arXiv:2510.16060v2.
# Resumable: finished runs (forecast .npz present) are skipped; a half-finished run resumes from
# results/runlogs/partial/<dataset>__<tag>.jsonl.
# Env: PY (python of the venv, default .venv-nbeats/bin/python), DEVICE (auto|cuda|mps|cpu), SEEDS, DATASETS, SELECTS.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv-nbeats/bin/python}
DEVICE=${DEVICE:-auto}
SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"heartrate crime glucose"}
SELECTS=${SELECTS:-"insample holdout"}
export NBEATS_THREADS=${NBEATS_THREADS:-4}
for seed in $SEEDS; do for sel in $SELECTS; do for ds in $DATASETS; do
  tag="nbeats$([ "$sel" = holdout ] && echo -holdout)-full-seed$seed"
  if [ -f "results/forecasts/${ds}__${tag}.npz" ]; then echo "skip $ds $tag"; continue; fi
  echo "START $ds $tag $(date)"
  "$PY" src/run_nbeats.py --dataset "$ds" --select "$sel" --seed "$seed" --grid full --device "$DEVICE"
  echo "END $ds $tag rc=$? $(date)"
done; done; done
missing=0
for seed in $SEEDS; do for sel in $SELECTS; do for ds in $DATASETS; do
  tag="nbeats$([ "$sel" = holdout ] && echo -holdout)-full-seed$seed"
  [ -f "results/forecasts/${ds}__${tag}.npz" ] || { echo "MISSING $ds $tag"; missing=$((missing+1)); }
done; done; done
if [ "$missing" -eq 0 ]; then echo FULL_GRID_DONE; else echo "INCOMPLETE: $missing runs missing -- rerun the same command to resume"; fi
