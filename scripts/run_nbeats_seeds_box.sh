#!/usr/bin/env bash
# Round 2 (box, CPU): reduced 8-config N-BEATS grid with seeds 0 and 2 (seed 1 = round-1 runs), both selection
# readings, all three datasets. Two streams in parallel, 3 threads each. Skips runs whose forecast already exists;
# a partly finished run resumes from results/runlogs/partial/. Replicates Adler et al., arXiv:2510.16060v2.
set -u
cd "$(dirname "$0")/.."
export NBEATS_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
stream() {  # $1 = selection
  for seed in 0 2; do for ds in heartrate crime glucose; do
    tag="nbeats$([ "$1" = holdout ] && echo -holdout)-seed$seed"
    if [ -f "results/forecasts/${ds}__${tag}.npz" ]; then echo "skip $ds $tag"; continue; fi
    echo "START $ds $tag $(date)"; venvs/nbeats/bin/python src/run_nbeats.py --dataset $ds --select $1 --seed $seed
    echo "END $ds $tag rc=$? $(date)"
  done; done
}
stream insample > results/runlogs_nbeats_seeds_insample.txt 2>&1 &
stream holdout  > results/runlogs_nbeats_seeds_holdout.txt 2>&1 &
wait
echo SEEDS_DONE
