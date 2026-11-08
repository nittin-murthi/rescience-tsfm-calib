#!/usr/bin/env bash
# Shared body of run_stage1.sh / run_stage2.sh (round 5, 6-dataset extension). Replicates Adler et al.,
# arXiv:2510.16060v2. Called with STAGE=<n> DATASETS="<names>". Re-running the same command resumes.
# Env knobs: SMOKE=1 (1 series, 1 N-BEATS config, in a throw-away .smoke/ copy), DEVICE=auto|mps|cuda|cpu,
#            TSFM_THREADS, ARIMA_PROCS, NBEATS_THREADS, TSFM_BATCH, FC_FLOAT16=1, PYTHON=/path/to/python3.12
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
: "${STAGE:?}" "${DATASETS:?}"
SMOKE=${SMOKE:-0}
export STAGE DATASETS SMOKE
# macOS: keep the machine awake for the whole run
if [ "$(uname -s)" = "Darwin" ] && [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export CAFFEINATED=1
  exec caffeinate -i bash "$0" "$@"
fi
mkdir -p results/runlogs
SUF=""; [ "$SMOKE" = "1" ] && SUF="_smoke"
LOG="$ROOT/results/runlogs_stage${STAGE}${SUF}.txt"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n=== [%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
say "Stage $STAGE ($DATASETS)  SMOKE=$SMOKE  log: $LOG"

# ---- 1. Python 3.12/3.11 + one venv per model family -------------------------------------------------------------
pick_python() {
  for c in ${PYTHON:-} python3.12 python3.11 python3 python; do
    [ -z "$c" ] && continue
    command -v "$c" >/dev/null 2>&1 || continue
    v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
    case "$v" in 3.11|3.12) echo "$c"; return 0;; esac
  done
  return 1
}
hashf() { python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:16])' "$1" 2>/dev/null \
          || shasum -a 256 "$1" | cut -c1-16; }
OS=$(uname -s)
mkvenv() {  # mkvenv <name> <requirements file> <needs torch: 0|1>
  local V="$ROOT/.venv-$1" REQ="$ROOT/$2" H
  if [ ! -x "$V/bin/python" ]; then
    say "Creating .venv-$1"
    if PY=$(pick_python); then "$PY" -m venv "$V"
    elif command -v uv >/dev/null 2>&1; then uv venv -q -p 3.12 "$V" && uv pip install -q -p "$V/bin/python" pip
    else
      echo "ERROR: Python 3.12 or 3.11 not found. macOS: brew install python@3.12 ; or install uv (docs.astral.sh/uv)"; exit 1
    fi
  fi
  H=$(hashf "$REQ")
  if [ "$(cat "$V/.installed" 2>/dev/null)" != "$H" ]; then
    say "Installing .venv-$1 ($2)"
    local PIP=("$V/bin/python" -m pip install -q)
    command -v uv >/dev/null 2>&1 && PIP=(uv pip install -q -p "$V/bin/python")
    if [ "$3" = "1" ] && [ "$OS" = "Linux" ]; then
      if command -v nvidia-smi >/dev/null 2>&1; then "${PIP[@]}" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
      else "${PIP[@]}" torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu; fi
    fi  # macOS: the default PyPI torch wheel has MPS
    "${PIP[@]}" -r "$REQ" || { echo "ERROR: pip install failed for .venv-$1"; exit 1; }
    echo "$H" > "$V/.installed"
  fi
}
mkvenv stats envs/requirements-stats-gpu.txt 0
mkvenv chronos envs/requirements-chronos-gpu.txt 1
mkvenv timesfm envs/requirements-timesfm-gpu.txt 1
mkvenv tirex envs/requirements-tirex-gpu.txt 1
mkvenv nbeats envs/requirements-nbeats-gpu.txt 1
VS="$ROOT/.venv-stats/bin/python"
vpy() { case "$1" in chronos*) echo "$ROOT/.venv-chronos/bin/python";; timesfm*) echo "$ROOT/.venv-timesfm/bin/python";;
        tirex) echo "$ROOT/.venv-tirex/bin/python";; esac; }

# ---- 2. Data -----------------------------------------------------------------------------------------------------
say "Data check"
"$VS" scripts/prepare_data.py $DATASETS || { echo "ERROR: data check failed"; exit 1; }

# ---- 3. Hardware, threads, device; environment record ------------------------------------------------------------
if [ "$OS" = "Darwin" ]; then
  NCPU=$(sysctl -n hw.ncpu); PCORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
else
  NCPU=$(nproc); PCORES=$NCPU
fi
export TSFM_THREADS=${TSFM_THREADS:-$PCORES}
export ARIMA_PROCS=${ARIMA_PROCS:-$(( NCPU > 3 ? NCPU - 2 : 1 ))}
export NBEATS_THREADS=${NBEATS_THREADS:-4}
export PYTORCH_ENABLE_MPS_FALLBACK=1
DEV=$("$ROOT/.venv-chronos/bin/python" -c "import sys; sys.path.insert(0,'src'); import run_tsfm as R; print(R.pick_device('${DEVICE:-auto}'))" 2>/dev/null | tail -1)
[ -z "$DEV" ] && DEV=cpu
BATCH=${TSFM_BATCH:-$([ "$DEV" = "cpu" ] && echo 64 || echo 256)}
ENVD="$ROOT/results/runlogs/env_stage$STAGE"; mkdir -p "$ENVD"
GPUENV="$ROOT/results/runlogs/gpu_env_stage${STAGE}${SUF}.txt"
{
  echo "date: $(date)"; echo "os: $(uname -a)"; echo "requested device: ${DEVICE:-auto} -> detected: $DEV"
  echo "cpus: $NCPU  performance cores: $PCORES  TSFM_THREADS=$TSFM_THREADS ARIMA_PROCS=$ARIMA_PROCS NBEATS_THREADS=$NBEATS_THREADS TSFM_BATCH=$BATCH FC_FLOAT16=${FC_FLOAT16:-0}"
  [ "$OS" = "Darwin" ] && { sysctl -n machdep.cpu.brand_string; echo "memory bytes: $(sysctl -n hw.memsize)"; }
  [ "$OS" = "Linux" ] && { grep -m1 "model name" /proc/cpuinfo; grep MemTotal /proc/meminfo; }
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi: not available"
  for v in stats chronos timesfm tirex nbeats; do
    echo "--- .venv-$v"
    "$ROOT/.venv-$v/bin/python" -c "import platform; print('python', platform.python_version())
try:
    import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'mps_available', torch.backends.mps.is_available())
except ImportError: pass" 2>&1
    "$ROOT/.venv-$v/bin/python" -m pip freeze > "$ENVD/freeze-$v.txt" 2>/dev/null || uv pip freeze -p "$ROOT/.venv-$v/bin/python" > "$ENVD/freeze-$v.txt" 2>/dev/null
  done
} > "$GPUENV" 2>&1
say "Device: $DEV  (record: $GPUENV)"

# ---- SMOKE: run everything in a throw-away copy with 1 series per dataset ----------------------------------------
if [ "$SMOKE" = "1" ]; then
  rm -rf "$ROOT/.smoke"; mkdir -p "$ROOT/.smoke/data"
  cp -R src scripts envs run_stage*.sh START_HERE.txt "$ROOT/.smoke/" 2>/dev/null
  cp -R data/processed "$ROOT/.smoke/data/"
  mkdir -p "$ROOT/.smoke/results/runlogs"; cp "$GPUENV" "$ROOT/.smoke/results/runlogs/"; cp -R "$ENVD" "$ROOT/.smoke/results/runlogs/"
  cd "$ROOT/.smoke"
  "$VS" scripts/make_smoke_data.py $DATASETS || exit 1
fi

TSFMS="chronos-bolt-small chronos-bolt-base timesfm-2.5-200m tirex"
SEEDS="0 1 2"; GRID=full; NBTRIAL="1,32,63"; TRIAL_TS=256; TRIAL_AR=400
if [ "$SMOKE" = "1" ]; then SEEDS="0"; GRID=smoke; TRIAL_TS=3; TRIAL_AR=3; fi
FAILED=()
run() {  # run <label> <cmd...>; failures are recorded, the stage continues
  local label=$1; shift
  say "$label"
  "$@"; local rc=$?
  [ $rc -ne 0 ] && { echo "FAILED ($rc): $label"; FAILED+=("$label"); }
  return 0
}
have() { [ -f "results/forecasts/$1__$2.npz" ]; }
nbtag() { echo "nbeats$([ "$1" = holdout ] && echo -holdout)-$GRID-seed$2"; }

# ---- 4. Timing trial + ETA ---------------------------------------------------------------------------------------
say "Timing trial"
for ds in $DATASETS; do
  for m in $TSFMS; do
    have "$ds" "$m" || [ -f "results/timing/${ds}__$m.json" ] || \
      run "trial $ds $m" "$(vpy $m)" src/run_tsfm.py --model "$m" --dataset "$ds" --device "$DEV" --batch "$BATCH" --trial $TRIAL_TS
  done
  for w in 64 T; do
    tag=$([ $w = 64 ] && echo arima || echo arima-ctxT)
    have "$ds" "$tag" || [ -f "results/timing/${ds}__$tag.json" ] || \
      run "trial $ds $tag" "$VS" src/run_arima.py --dataset "$ds" --refit-context $w --procs "$ARIMA_PROCS" --trial $TRIAL_AR
  done
  if [ "$SMOKE" != "1" ] && ! have "$ds" "$(nbtag insample 0)"; then
    run "trial $ds nbeats configs $NBTRIAL" "$ROOT/.venv-nbeats/bin/python" src/run_nbeats.py --dataset "$ds" \
        --grid full --seed 0 --select insample --device "$DEV" --config-ids "$NBTRIAL"
  fi
done
grep -h '"device_note"' results/timing/*.json 2>/dev/null | sort | uniq -c | sed 's/^/  device decisions: /' >> "$GPUENV"
for f in results/timing/*.json; do
  [ -f "$f" ] && "$VS" -c "import json,sys; d=json.load(open(sys.argv[1])); print('  trial', d['dataset'], d['model'], d.get('device', 'cpu'), d.get('device_note', ''))" "$f" >> "$GPUENV"
done
"$VS" scripts/eta.py $DATASETS | tee "results/eta_stage$STAGE.txt"
if [ "$DEV" = "cpu" ] && [ "$SMOKE" != "1" ]; then
  echo "WARNING: no GPU (CUDA/MPS) detected; everything runs on the CPU. Ctrl-C within 15 s to abort."; sleep 15
fi

# ---- 5. Full run (resumable; finished models are skipped) --------------------------------------------------------
for ds in $DATASETS; do
  for m in $TSFMS; do
    have "$ds" "$m" || run "$ds $m" "$(vpy $m)" src/run_tsfm.py --model "$m" --dataset "$ds" --device "$DEV" --batch "$BATCH"
  done
  for w in 64 T; do
    tag=$([ $w = 64 ] && echo arima || echo arima-ctxT)
    have "$ds" "$tag" || run "$ds $tag" "$VS" src/run_arima.py --dataset "$ds" --refit-context $w --procs "$ARIMA_PROCS"
  done
  for sel in insample holdout; do
    for s in $SEEDS; do
      have "$ds" "$(nbtag $sel $s)" || run "$ds $(nbtag $sel $s)" "$ROOT/.venv-nbeats/bin/python" src/run_nbeats.py \
          --dataset "$ds" --grid $GRID --seed "$s" --select "$sel" --device "$DEV"
    done
  done
  "$VS" scripts/eta.py $DATASETS | tail -1
done

# ---- 6. Evaluate, check completeness, pack -----------------------------------------------------------------------
run "evaluate" "$VS" src/evaluate.py --datasets $DATASETS --out-dir "results/stage$STAGE"
MISSING=()
for ds in $DATASETS; do
  for m in $TSFMS arima arima-ctxT; do have "$ds" "$m" || MISSING+=("$ds/$m"); done
  for sel in insample holdout; do for s in $SEEDS; do have "$ds" "$(nbtag $sel $s)" || MISSING+=("$ds/$(nbtag $sel $s)"); done; done
done
if [ "$SMOKE" = "1" ]; then OUTZIP="$PWD/RETURN_STAGE${STAGE}_SMOKE.zip"; else OUTZIP="$(dirname "$ROOT")/RETURN_STAGE$STAGE.zip"; fi
cp "$LOG" "results/runlogs_stage${STAGE}${SUF}.txt" 2>/dev/null || true
run "pack" "$VS" scripts/pack_stage.py "$STAGE" "$OUTZIP" $DATASETS
if [ ${#MISSING[@]} -eq 0 ] && [ ${#FAILED[@]} -eq 0 ]; then
  say "FULL_STAGE${STAGE}_DONE$([ "$SMOKE" = 1 ] && echo ' (SMOKE)')  ->  send $OUTZIP"
  RC=0
else
  say "INCOMPLETE: missing ${MISSING[*]:-none}; failed steps: ${FAILED[*]:-none}. Run the same command again to resume."
  RC=1
fi
if [ "$SMOKE" = "1" ] && [ "${KEEP_SMOKE:-0}" != "1" ]; then
  cd "$ROOT"; rm -rf "$ROOT/.smoke"; echo "(smoke copy .smoke/ removed; set KEEP_SMOKE=1 to keep it)"
fi
exit $RC
