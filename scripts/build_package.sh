#!/usr/bin/env bash
# Build the 6-dataset package: git archive of HEAD (no venvs, no .git) + data/processed/reviews.csv.gz (not in git).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/workspace/rescience-tsfm-calib-6ds.zip}
git archive --format=zip --prefix=rescience-tsfm-calib/ -o "$OUT" HEAD
venvs/stats/bin/python - "$OUT" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "a") as z:
    z.write("data/processed/reviews.csv.gz", "rescience-tsfm-calib/data/processed/reviews.csv.gz", zipfile.ZIP_STORED)
PY
ls -la "$OUT"
