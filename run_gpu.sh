#!/usr/bin/env bash
# One-command runner for the full paper N-BEATS grid on your own machine (Linux or macOS).
# Replicates Adler et al., arXiv:2510.16060v2.
# [Paper App. A.2, Table 3] 64-config grid; x 3 seeds x 2 selection readings x 3 datasets (see RUN_ON_GPU.md).
#
# Usage:   bash run_gpu.sh            (re-running the same command resumes where it stopped)
#          SMOKE=1 bash run_gpu.sh    (setup + 1 config on Heart-Rate, no full run; for testing)
# Env:     PYTHON=/path/to/python3.11 to choose the interpreter; DEVICE=cuda|mps|cpu to force a device.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
VENV=.venv-nbeats
mkdir -p results/runlogs

say() { printf '\n=== %s\n' "$*"; }

# ---- 1. Python 3.11/3.12 (the pinned numpy 2.4 / pandas 3.1 need >= 3.11) -------------------------------------
pick_python() {
  for c in ${PYTHON:-} python3.11 python3.12 python3 python; do
    [ -z "$c" ] && continue
    command -v "$c" >/dev/null 2>&1 || continue
    v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
    case "$v" in 3.11|3.12) echo "$c"; return 0;; esac
  done
  return 1
}
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating $VENV"
  if PY=$(pick_python); then
    "$PY" -m venv "$VENV"
  elif command -v uv >/dev/null 2>&1; then
    uv venv -p 3.11 "$VENV" && uv pip install -p "$VENV/bin/python" pip
  else
    echo "ERROR: Python 3.11 or 3.12 not found." >&2
    echo "  macOS:  brew install python@3.11      Linux: sudo apt install python3.11 python3.11-venv" >&2
    echo "  or install uv (https://docs.astral.sh/uv/) and rerun; it downloads Python 3.11 itself." >&2
    exit 1
  fi
fi
VPY="$ROOT/$VENV/bin/python"

# ---- 2. Dependencies (once; marker file) ---------------------------------------------------------------------
if [ ! -f "$VENV/.installed" ]; then
  say "Installing dependencies (envs/requirements-nbeats-gpu.txt)"
  "$VPY" -m pip install --upgrade pip >/dev/null
  if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    # NVIDIA GPU: CUDA 12.4 build of the pinned torch (needs driver >= 525)
    "$VPY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  elif [ "$(uname -s)" = "Linux" ]; then
    "$VPY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
  fi  # macOS: the default PyPI wheel includes MPS
  "$VPY" -m pip install -r envs/requirements-nbeats-gpu.txt
  touch "$VENV/.installed"
fi

# ---- 3. Environment record + device --------------------------------------------------------------------------
DEV=$("$VPY" -c "import sys; sys.path.insert(0,'src'); import run_nbeats as R; print(R.pick_device('${DEVICE:-auto}'))" 2>/dev/null | tail -1)
{
  echo "date: $(date)"; echo "os: $(uname -a)"; echo "device: $DEV"
  "$VPY" -c "import platform, torch, neuralforecast; print('python', platform.python_version()); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); print('mps_available', torch.backends.mps.is_available()); print('neuralforecast', neuralforecast.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi: not available"
  [ "$(uname -s)" = "Darwin" ] && sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null || true
} > results/runlogs/gpu_env.txt 2>&1
say "Device: $DEV   (environment recorded in results/runlogs/gpu_env.txt)"
if [ "$DEV" = "cpu" ]; then
  echo "##########################################################################"
  echo "# WARNING: no CUDA/MPS GPU detected -- running on CPU will take ~2 days.  #"
  echo "# Press Ctrl-C now to abort; continuing in 15 s.                          #"
  echo "##########################################################################"
  sleep 15
fi
export PY="$VPY" DEVICE="$DEV"

# ---- 4. Timing trial (two 1000-step configs; scores are reused by the full run) ------------------------------
TRIAL_IDS="32,63"; [ "${SMOKE:-0}" = "1" ] && TRIAL_IDS="0"
if [ -f results/forecasts/heartrate__nbeats-full-seed0.npz ] && [ "${SMOKE:-0}" != "1" ]; then
  say "Timing trial already covered by a finished run; skipping"
else
say "Timing trial on Heart-Rate (configs $TRIAL_IDS)"
"$VPY" src/run_nbeats.py --dataset heartrate --grid full --seed 0 --select insample --device "$DEV" \
  --config-ids "$TRIAL_IDS" 2>&1 | grep -E "config_id|tag=|stopped|Error|error" || true
"$VPY" - "$TRIAL_IDS" <<'PYEOF'
# Estimate: 32 long (1000-step) + 32 short configs per run, short ~ 1/10 of long, 18 runs, +20 % for larger datasets
import json, sys
ids = {int(x) for x in sys.argv[1].split(",")}
recs = [json.loads(l) for l in open("results/runlogs/partial/heartrate__nbeats-full-seed0.jsonl")]
sec = [r["seconds"] for r in recs if r["config_id"] in ids]
if sec:
    long_s = sum(sec) / len(sec) * (10 if ids == {0} else 1)
    tot = 18 * 1.2 * 32 * long_s * 1.1
    print(f"Trial: {sec} s per config -> estimated total {tot/3600:.1f} h (rough; resumable at any time)")
PYEOF
fi
if [ "${SMOKE:-0}" = "1" ]; then say "SMOKE=1: skipping the full grid"; else
  # ---- 5. Full grid, resumable ------------------------------------------------------------------------------------
  say "Full grid (log: results/runlogs_nbeats_full.txt). Safe to interrupt; rerun 'bash run_gpu.sh' to resume."
  bash scripts/run_nbeats_full_gpu.sh 2>&1 | tee -a results/runlogs_nbeats_full.txt
fi

# ---- 6. Pack -------------------------------------------------------------------------------------------------
bash scripts/pack_nbeats_full.sh || true
NAME=$(basename "$ROOT")
"$VPY" - "$ROOT" <<'PYEOF'
# zip the whole folder (code + artifacts) next to it, without the virtualenv
import os, sys, zipfile
root = sys.argv[1]; parent = os.path.dirname(root); name = os.path.basename(root)
out = os.path.join(parent, "RETURN_ME.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".venv-nbeats", "__pycache__", ".git")]
        for f in files:
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, parent))
print(f"Wrote {out} ({os.path.getsize(out)/1e6:.1f} MB). Send this file back.")
PYEOF
