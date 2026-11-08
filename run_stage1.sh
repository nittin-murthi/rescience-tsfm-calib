#!/usr/bin/env bash
# Stage 1 of the 6-dataset extension: Patents + M5, all models (4 TSFMs, ARIMA W=64 and W=T, full-grid N-BEATS
# x 2 selection readings x 3 seeds). Usage: bash run_stage1.sh   (re-run to resume; SMOKE=1 for a quick test)
STAGE=1 DATASETS="patents m5" exec bash "$(dirname "$0")/scripts/stage_common.sh" "$@"
