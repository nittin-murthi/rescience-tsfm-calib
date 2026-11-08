#!/usr/bin/env bash
# DIAGNOSIS ONLY: run a subset of our pipeline on the authors' released Crime / Heart-Rate data.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for d in heartrate_authorsdata crime_authorsdata; do
  venvs/stats/bin/python src/run_arima.py --dataset $d --procs 3
  venvs/stats/bin/python src/run_arima.py --dataset $d --refit-context T --procs 3
  venvs/chronos/bin/python src/run_tsfm.py --model chronos-bolt-small --dataset $d
  venvs/timesfm/bin/python src/run_tsfm.py --model timesfm-2.5-200m --dataset $d
done
echo DIAG_DONE
