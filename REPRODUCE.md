# Reproducing every number in the article

All commands run from the repository root. Times are wall-clock times we observed; yours will differ.

## 0. Requirements

- Linux or macOS (tested: Linux x86-64 with 8 shared CPU cores and no GPU; Apple M2 Max, 32 GB, macOS, MPS).
- `uv` >= 0.4 (or `python -m venv` + `pip`), `curl`, `sha256sum` or `shasum`, `make`.
- For the article: either TeX Live (`xelatex`, `latexmk`, `biber` matching your biblatex; we used biber 2.20) or
  Tectonic 0.15 with biber 2.17 on `PATH` (used for the final PDF), plus Python 3 with PyYAML.
- Internet on first use: about 3 GB of Python packages and about 2 GB of model weights.

## 1. Environments (exact pins in `envs/`)

One environment per model family, because the libraries pin conflicting dependencies.

| Environment | Pins (CPU / GPU-capable) | Key packages |
|---|---|---|
| `stats` | `requirements-stats.txt` / `-stats-gpu.txt` | statsforecast 2.1.1, numpy 2.4.6, pandas 2.3.3 |
| `chronos` | `requirements-chronos.txt` / `-chronos-gpu.txt` | Chronos-Bolt, torch 2.5.1 |
| `timesfm` | `requirements-timesfm.txt` / `-timesfm-gpu.txt` | timesfm @ e51928e (git), torch 2.5.1 |
| `tirex` | `requirements-tirex.txt` / `-tirex-gpu.txt` | tirex-ts 1.4.2, torch 2.5.1 |
| `nbeats` | `requirements-nbeats.txt` / `-nbeats-gpu.txt` | neuralforecast 3.1.9, torch 2.5.1 |

Model checkpoints are pinned by Hugging Face revision in `envs/MODEL_REVISIONS.md` (Chronos-Bolt small/base,
TimesFM 2.5 200M, TiRex). The CPU machine used Python 3.11.17 with CPU torch wheels; the M2 Max used Python 3.12.11.

CPU environments (used by `scripts/run_all.sh` and `scripts/run_nbeats_all.sh`):
```bash
for v in chronos timesfm tirex stats nbeats; do uv venv -p 3.11 venvs/$v; done
CPU="--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
uv pip install -p venvs/chronos/bin/python -r envs/requirements-chronos.txt $CPU
uv pip install -p venvs/timesfm/bin/python -r envs/requirements-timesfm.txt $CPU
uv pip install -p venvs/tirex/bin/python   -r envs/requirements-tirex.txt   $CPU
uv pip install -p venvs/stats/bin/python   -r envs/requirements-stats.txt
uv pip install -p venvs/nbeats/bin/python  -r envs/requirements-nbeats.txt  $CPU
```
On macOS drop `$CPU` (PyPI torch wheels include MPS). The stage runners in step 4 create their own
`.venv-*` environments from the `*-gpu.txt` pins automatically.

## 2. Data (not shipped; downloaded and checksum-verified)

| Dataset | Source | How it is obtained | Series / windows (T, stride) |
|---|---|---|---|
| NYC Crime | NYC Open Data `qgea-i56i`, daily counts per borough by report date | `scripts/fetch_raw_data.sh` | 5 / 4,030 (512, 4) |
| Heart-Rate | PhysioNet *meditation* 1.0.0, `metron/M1..M14` | `scripts/fetch_raw_data.sh` | 6 / 1,794 (128, 1) |
| Glucose | PhysioNet *BIG IDEAs* 1.1.2, `Dexcom_001..016` | `scripts/fetch_raw_data.sh` | 16 / 13,040 (512, 1) |
| Reviews | authors' UCI deposit (doi:10.24432/C5SP7K) | `scripts/prepare_data.py reviews` | 235 / 358,610 (512, 8) |
| M5 | `sales_train_validation.csv` (Nixtla mirror), rebuilt dept × store | `scripts/prepare_data.py m5` | 70 / 21,280 (512, 1) |
| Patents | authors' `y_patents.csv` @ b60bfd9 | `scripts/prepare_data.py patents` | 83 / 5,727 (128, 1) |

Counts are from `data/processed/*.json`; the article's Methods section explains how they differ from the paper's Table 1.

```bash
bash scripts/fetch_raw_data.sh                         # prints RAW_DATA_OK when all checksums match
venvs/stats/bin/python src/data.py                     # builds crime, glucose, heartrate into data/processed/
venvs/stats/bin/python scripts/prepare_data.py patents m5 reviews   # downloads, verifies and builds
(cd data/processed && sha256sum -c CHECKSUMS.sha256)   # all processed files must match
```
The NYPD dataset is live; if a later download differs from ours, the checksum check fails and Crime results can
differ.

## 3. Tests
```bash
venvs/stats/bin/python tests/test_metrics.py
```

## 4. Forecasts

| Command | Datasets | Models | Hardware we used |
|---|---|---|---|
| `scripts/run_all.sh` | Heart-Rate, Crime, Glucose | ARIMA (64, T), Chronos-Bolt small/base, TimesFM 2.5, TiRex | CPU |
| `scripts/run_nbeats_all.sh` | Heart-Rate, Crime, Glucose | N-BEATS reduced grid, in-sample / hold-out (seed 1) | CPU |
| `scripts/run_nbeats_seeds_box.sh` | Heart-Rate, Crime, Glucose | N-BEATS reduced grid, seeds 0 and 2 | CPU |
| `bash run_gpu.sh` (= `scripts/run_nbeats_full_gpu.sh`) | Heart-Rate, Crime, Glucose | N-BEATS full 64-config grid × 2 selections × seeds 0–2 | M2 Max, MPS |
| `bash run_stage1.sh` | Patents, M5 | all 12 model runs above (4 TSFMs, 2 ARIMA, 6 N-BEATS full grid) | M2 Max |
| `bash run_stage2.sh` | Reviews | the same 12 model runs | M2 Max |
| `scripts/run_arima_ctx_sweep.sh`, `scripts/run_determinism.sh` | sensitivity: ARIMA context W ∈ {32,128,256}; TSFM repeat | | CPU |

The stage runners pick `DEVICE=auto` (cuda > mps > cpu), compare each TSFM on 16 windows against the CPU and fall
back to the CPU on an error or a relative difference > 1e-2. TimesFM 2.5 fell back to the CPU on MPS (no float64).
All runners are resumable; `SMOKE=1` runs a tiny throw-away version. Details: `STAGES.md`, `RUN_ON_GPU.md`.
Forecasts go to `results/forecasts/*.npz` (large, not in the repository); run logs with times and peak memory go
to `results/runlogs/`.

## 5. Metrics, verdicts, tables, article
```bash
venvs/stats/bin/python src/evaluate.py --include-returned results/stage2   # summary.csv, metrics_per_series.csv, pce_variants.csv
venvs/stats/bin/python scripts/arima_ctx_sweep.py                          # arima_ctx_sweep.csv, figures/arima_ctx.pdf
venvs/stats/bin/python src/make_figures.py                                 # figures/pce_cce.pdf, verdict.csv/json
venvs/stats/bin/python scripts/seed_summary.py                             # seed_summary.csv, seed_verdict.csv
venvs/stats/bin/python scripts/make_tables.py                              # article/tables/*.tex
python3 scripts/make_traceability.py                                       # TRACEABILITY.md tag index
cd article && make                                                         # article.pdf (TeX Live)
# or: cd article && python3 yaml-to-latex.py -i metadata.yaml -o metadata.tex && tectonic -X compile article.tex
```
Without the forecast arrays, `src/evaluate.py` cannot be re-run, but every later step can: it reads only the CSVs
shipped in `results/`. Reviews forecasts were not returned from the M2 Max, so its rows always come from
`results/stage2/` (checked by `scripts/verify_returned_stage2.py`); Patents and M5 forecasts were re-evaluated and
checked by `scripts/verify_returned_forecasts.py` (outputs in `results/stage1_validity/`).

## 6. Optional diagnosis (post hoc, not used for the verdicts)
```bash
git clone https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration /tmp/authors && git -C /tmp/authors checkout b60bfd9
venvs/stats/bin/python diagnosis/build_authors_data.py /tmp/authors
bash diagnosis/run_authors_data.sh
venvs/stats/bin/python src/evaluate.py && venvs/stats/bin/python diagnosis/authors_style_metrics.py
```

## Known differences from the paper (see the article)

- TimesFM 2.5 200M substitutes for TimesFM 2.0 500M; Moirai 2.0 and YingLong are not run.
- Window stride is an assumption (A-01); window counts differ from Table 1 on M5 and Patents; Reviews has 235 of 239 series.
- ARIMA refit context is ambiguous in the paper (64 vs all data); both are run, plus a sweep.
- N-BEATS early-stopping validation split is an assumption; MPS training is not bit-for-bit reproducible.
- TiRex ran on its torch backend (CPU/MPS), not the CUDA kernels.
