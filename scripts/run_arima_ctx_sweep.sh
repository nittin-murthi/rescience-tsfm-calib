#!/usr/bin/env bash
# Round 4: ARIMA refit-context sensitivity sweep. W=64 ('arima') and W=T ('arima-ctxT') already exist.
# Resumable: skips any (dataset, W) whose runlog exists.
set -u
cd "$(dirname "$0")/.."
PY=venvs/stats/bin/python
for job in "crime 32" "crime 128" "crime 256" "glucose 32" "glucose 128" "glucose 256" "heartrate 32"; do
  set -- $job
  if [ -f "results/runlogs/$1__arima-ctx$2.json" ]; then echo "skip $1 $2"; continue; fi
  echo "START $1 ctx$2 $(date)"
  $PY src/run_arima.py --dataset "$1" --refit-context "$2" --procs 4; echo "END $1 ctx$2 rc=$? $(date)"
done
echo SWEEP_DONE
