# Running the full N-BEATS grid × 3 seeds on your own machine

**Shortcut:** `bash run_gpu.sh` (Linux/macOS) or `powershell -ExecutionPolicy Bypass -File run_gpu.ps1` (Windows,
untested) does steps 2–5 below in one command and is resumable; see `START_HERE.txt`. Smoke-tested on the Linux
box (CPU) with `SMOKE=1 bash run_gpu.sh`.

The paper's N-BEATS search ([Paper App. A.2, Table 3], arXiv:2510.16060v2) has 2^6 = 64 configurations:

| Hyperparameter | Search space (quoted from Table 3) |
|---|---|
| Epochs | {100, 1000} |
| Learning Rate | {0.001, 0.0001} |
| Early Stop Patience Steps | {-1, 2} |
| Number of Blocks | {(1,1,1), (3,3,3)} |
| Validation Check Steps | {10, 100} |
| Loss Function | {Normal Distribution Loss, Quantile Loss} |

We need 64 configs × 2 selection readings (in-sample, hold-out) × 3 seeds (0, 1, 2) × 3 datasets, which is
18 runs and 1,152 model fits. That is too much for the shared 8-core box. The box timings are in
`results/runlogs/timing_trial_box_heartrate__nbeats-full-seed0.jsonl`: on 2 threads under load, a 100-step fit took
12–47 s and a 1000-step fit took about 140 s with blocks (1,1,1). That works out to roughly 3 h per run, or about
50 h in total. The box therefore runs only the reduced 8-config grid with 3 seeds
(`scripts/run_nbeats_seeds_box.sh`). The full grid is packaged here.

## 1. Get the code (then build the data as in `REPRODUCE.md`, step 2)
Either clone the repository:
```bash
git clone https://github.com/nittin-murthi/rescience-tsfm-calib && cd rescience-tsfm-calib
```
or unpack the tarball copied from the box (`rescience-tsfm-calib-src.tgz`, made with `git archive`):
```bash
mkdir rescience-tsfm-calib && tar xzf rescience-tsfm-calib-src.tgz -C rescience-tsfm-calib && cd rescience-tsfm-calib
```

## 2. Environment (Python 3.11)
```bash
python3.11 -m venv .venv-nbeats            # or: uv venv -p 3.11 .venv-nbeats
.venv-nbeats/bin/pip install -r envs/requirements-nbeats-gpu.txt
# check the device; this prints cuda, mps or cpu
.venv-nbeats/bin/python -c "import sys; sys.path.insert(0,'src'); import run_nbeats as R; print(R.pick_device('auto'))"
```
On Linux, the default `torch==2.5.1` wheel bundles CUDA 12.4 and needs an NVIDIA driver ≥ 525. If you need a different
CUDA build: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` (before the other packages).
On the M2 MacBook the same pin installs the MPS build. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically.

## 3. Timing trial (about 5–10 min): two 1000-step configs on Heart-Rate
```bash
.venv-nbeats/bin/python src/run_nbeats.py --dataset heartrate --grid full --seed 0 --device auto --config-ids 32,63
```
Each scored config is printed with its `seconds`. Multiply the mean of the two by about 32 and add about 10 % for
the 32 100-step configs; that gives the time per run, and there are 18 runs. The trial's scores are kept in
`results/runlogs/partial/` and are reused when the full run starts, so no work is lost.

## 4. Full run (resumable; safe to interrupt and restart)
```bash
nohup env PY=.venv-nbeats/bin/python DEVICE=auto scripts/run_nbeats_full_gpu.sh > results/runlogs_nbeats_full.txt 2>&1 &
tail -f results/runlogs_nbeats_full.txt      # ends with FULL_GRID_DONE
```
Optional variables: `SEEDS="0"`, `DATASETS="heartrate"` and `SELECTS="insample"` run a subset first.
`NBEATS_THREADS` (default 4) sets CPU threads. On the M2, `DEVICE=cpu` is worth a comparison in step 3, because
these small MLPs are not always faster on MPS.

Expected runtime: **not measured**, so treat these as estimates to check against step 3. On a recent CUDA GPU,
expect roughly 2–6 h for all 18 runs. On the M2 (MPS or 8 CPU cores), expect roughly 10–25 h. Hold-out runs add
one refit each.

## 5. Copy the results back
```bash
scripts/pack_nbeats_full.sh                 # writes nbeats_full_results.tgz
```
Copy `nbeats_full_results.tgz` to the box's `/workspace/rescience-tsfm-calib/` and unpack it there with
`tar xzf nbeats_full_results.tgz`. Then, on the box:
```bash
OMP_NUM_THREADS=2 venvs/stats/bin/python src/evaluate.py
venvs/stats/bin/python scripts/seed_summary.py
venvs/stats/bin/python src/make_figures.py && venvs/stats/bin/python scripts/make_tables.py
cd article && PATH=/usr/bin:$PATH make
```
The outputs use the format `evaluate.py` reads: `results/forecasts/<dataset>__nbeats[-holdout]-full-seed<k>.npz`
(quantiles, target, series_id, origin) plus `results/runlogs/<dataset>__<tag>.json`.

Note: GPU training is not bit-for-bit deterministic, so a rerun can differ slightly from the first run. The seed
spread is what we report.
