#!/usr/bin/env bash
# Reproduce every forecast used in the article (sequential, to keep peak memory low on a shared box).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
DATASETS="${DATASETS:-heartrate crime glucose}"
for ds in $DATASETS; do
  venvs/stats/bin/python src/run_arima.py --dataset "$ds" --refit-context 64
  venvs/stats/bin/python src/run_arima.py --dataset "$ds" --refit-context T
  venvs/chronos/bin/python src/run_tsfm.py --model chronos-bolt-small --dataset "$ds"
  venvs/chronos/bin/python src/run_tsfm.py --model chronos-bolt-base  --dataset "$ds"
  venvs/timesfm/bin/python src/run_tsfm.py --model timesfm-2.5-200m  --dataset "$ds"
  venvs/tirex/bin/python   src/run_tsfm.py --model tirex             --dataset "$ds"
done
venvs/stats/bin/python src/evaluate.py
