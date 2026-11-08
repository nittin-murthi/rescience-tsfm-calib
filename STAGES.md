# 6-dataset extension: two-stage run (round 5)

Commands (run from the unzipped `rescience-tsfm-calib/`; re-run the same command to resume):

| Command | Datasets | Models |
|---|---|---|
| `bash run_stage1.sh` | Patents (83 series, 5,727 windows), M5 (70 series, 21,280 windows) | Chronos-Bolt small and base, TimesFM-2.5-200M, TiRex; ARIMA W=64 (primary) and W=T; N-BEATS full 64-config grid × {in-sample, hold-out} × seeds {0,1,2} |
| `bash run_stage2.sh` | Reviews (235 series, 358,610 windows) | the same 12 model runs |

Steps, all in `scripts/stage_common.sh`:
1. Venvs: `.venv-stats`, `.venv-chronos`, `.venv-timesfm`, `.venv-tirex`, `.venv-nbeats`. The requirement files are
   `envs/requirements-*-gpu.txt`, and a venv is reinstalled when its file changes.
2. Data check: `scripts/prepare_data.py` checks the sha256 values in `data/processed/CHECKSUMS.sha256`. A missing
   file is rebuilt from the pinned source in `src/data.py` `SOURCES`, and the source checksum is verified.
3. Device detection (`DEVICE=auto`: cuda > mps > cpu). The environment record goes to
   `results/runlogs/gpu_env_stageN.txt`, with pip freezes in `results/runlogs/env_stageN/`.
   - Each TSFM runs its first 16 windows on the device and on the CPU. If the device raises an error, or the
     relative difference is > 1e-2, that model runs on the CPU, and the reason is logged.
   - ARIMA is CPU-only, using `ARIMA_PROCS` processes (default: cores − 2).
   - N-BEATS uses the device (MPS worked in round 2).
4. Timing trial, with results in `results/timing/` and the ETA in `results/eta_stageN.txt`:
   - TSFMs: 256 windows each;
   - ARIMA: 400 windows each, plus AutoARIMA order selection, which is cached per dataset;
   - N-BEATS: configs 1, 32 and 63 of seed-0 in-sample. These scores are reused by the full run.
5. Full run, in the foreground, resumable:
   - TSFM chunks of 8,192 windows and ARIMA chunks of 20,000 go under `results/runlogs/partial_*`;
   - N-BEATS grid scores go in `results/runlogs/partial/*.jsonl`;
   - a failed model is recorded, and the stage continues.
6. `src/evaluate.py --datasets ... --out-dir results/stageN` writes `summary.csv`, `metrics_per_series.csv` and
   `pce_variants.csv`. The runner prints `FULL_STAGEN_DONE` or `INCOMPLETE`, then packs `../RETURN_STAGEN.zip`.
   - The zip holds the metrics, logs, timing, gpu_env, freezes, orders and code. There are no venvs and no .git.
   - Forecast `.npz` files are included if they total < 2 GB. Otherwise they go to `RETURN_STAGEN_forecasts.zip`.

`SMOKE=1` runs the whole pipeline in a throw-away `.smoke/` copy: 1 series, 3 windows per dataset, and only
N-BEATS config 0. It never writes real results.

Knobs:
- `DEVICE`, `TSFM_THREADS` (default: performance cores), `TSFM_BATCH` (default: 256 on a GPU, 64 on the CPU).
- `ARIMA_PROCS`, `NBEATS_THREADS` (default 4).
- `FC_FLOAT16=1`: float16 forecast storage, kept only if every coverage indicator 1{y ≤ q} is unchanged.

## Documented substitutions and assumptions (see TRACEABILITY.md D9)
- **Reviews:** `[FROM AUTHORS' DATA]`, the UCI deposit `y_amazon-google-large.csv`. It has 235 series, not the 239
  in Table 1, and all 235 are used. The window stride is d=8, giving 358,610 windows (Table 1: 360,490 forecasts).
- **M5:** we rebuild it from raw `sales_train_validation.csv` (dept × store, d_1..d_1912). It equals the authors'
  `y_m5.csv` exactly. With d=1 there are 21,280 full-target windows, vs 62,160 in Table 1; no d in {1,4,8}
  reproduces that number under the Table 2 split.
- **Patents:** `[FROM AUTHORS' DATA]`, `y_patents.csv` @ b60bfd9. Table 1's 10,956 = 83 × 132 counts origins with
  partial targets. We keep full 64-step targets, giving 5,727 windows.
- **TimesFM 2.5** is the instructed substitute for 2.0. The checkpoints are pinned to the HF revisions in
  `envs/MODEL_REVISIONS.md`.

## Smoke tests on the box (Linux, 8 CPU, no GPU, so device = cpu; fresh extract, fresh venvs on Python 3.12.15, fresh HF cache in stage 1 of the first smoke)
- Stage 1 (`SMOKE=1 bash run_stage1.sh`, code @ dfb3494): FULL_STAGE1_DONE (SMOKE), 16 forecasts (2 datasets x 8 model tags), evaluate and pack OK; about 11 min wall time on a loaded box (start [14:52:26], end 15:03:48 PDT).
- Stage 2 (`SMOKE=1 bash run_stage2.sh`, same extract and venvs): FULL_STAGE2_DONE (SMOKE), 8 forecasts, evaluate and pack OK; about 7 min (15:03:48 to 15:11:05 PDT).
- Not testable on the box: MPS/CUDA paths and the GPU-vs-CPU check (no GPU here; it decides the fallbacks at run time on the M2).

## Outcome (round 6)
- Stage 1 finished on the M2 Max on Oct 25, 2026 (16:31-19:23 PDT); stage 2 from Oct 26 15:21 to Oct 27 10:17 PDT.
  TimesFM-2.5 fell back to the CPU (MPS has no float64); the other TSFMs and N-BEATS used MPS.
- Both returns were checked (`scripts/verify_returned_forecasts.py`, `scripts/verify_returned_stage2.py`,
  outputs in `results/stage1_validity/`) and are valid. No rerun is needed.
