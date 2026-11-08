#!/usr/bin/env bash
# Round 2: bundle full-grid N-BEATS outputs for copying back to the box (see RUN_ON_GPU.md).
set -eu
cd "$(dirname "$0")/.."
tar czf nbeats_full_results.tgz results/forecasts/*__nbeats*-full-seed*.npz results/runlogs/*__nbeats*-full-seed*.json \
  results/runlogs/partial/*__nbeats*-full-seed*.jsonl results/runlogs_nbeats_full*.txt 2>/dev/null || true
ls -la nbeats_full_results.tgz
