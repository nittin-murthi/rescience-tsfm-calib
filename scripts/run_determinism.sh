#!/usr/bin/env bash
# Round 2: repeat each TSFM once on Heart-Rate to check that inference is deterministic (no sampling).
# Replicates Adler et al., arXiv:2510.16060v2. Output: results/determinism/, compared by scripts/check_determinism.py.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for m in chronos-bolt-small chronos-bolt-base; do venvs/chronos/bin/python src/run_tsfm.py --model $m --dataset heartrate --repeat; done
venvs/timesfm/bin/python src/run_tsfm.py --model timesfm-2.5-200m --dataset heartrate --repeat
venvs/tirex/bin/python src/run_tsfm.py --model tirex --dataset heartrate --repeat
echo DET_DONE
