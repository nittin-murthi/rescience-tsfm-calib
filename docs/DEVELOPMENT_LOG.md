# [Re] Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated? — partial replication

Independent partial replication of Adler et al., *Beyond Accuracy: Are Time Series Foundation Models
Well-Calibrated?*, ICLR 2026, **arXiv:2510.16060v2** (PDF in `paper/`), prepared for ReScience C.
Author/owner: Nittin Murthi Dhekshinamoorthy (UIUC). Status: **draft, under owner review**; six datasets (round 6).

Claim tested (only): TSFMs are better calibrated (PCE, Eq. 1) than ARIMA / N-BEATS, with TSFM PCE close to or
below 5% (§4.1, Fig. 3, Table 4). See `PLAN.md`, `TRACEABILITY.md`, `DIAGNOSIS.md`, `article/`.

## Layout
```
paper/            arXiv v2 PDF + text dump
data/raw/         public raw data (NYPD daily counts via Socrata, PhysioNet Dexcom G6, PhysioNet meditation/metron)
data/processed/   built by src/data.py (CSV + protocol JSON)
src/              data.py, metrics.py, forecast_io.py, run_tsfm.py, run_arima.py, run_nbeats.py, evaluate.py, make_figures.py
tests/            metric property tests
scripts/          run_all.sh, run_nbeats_all.sh, make_tables.py (article tables/macros from CSVs), make_traceability.py,
                  verify_returned_forecasts.py, verify_returned_stage2.py (checks on the M2 Max runs)
diagnosis/        diagnosis-only scripts run after own results (authors' aggregation, authors' data); see DIAGNOSIS.md
results/          summary.csv, metrics_per_series.csv, pce_variants.csv, verdict.csv, paper_table4_subset.csv,
                  diagnosis_authors_style_metrics.csv, runlogs/ (forecasts/ is git-ignored)
envs/             exact package pins per venv + HF model revisions
article/          ReScience C LaTeX source (article.tex, content.tex, metadata.yaml, bibliography.bib, tables/, figures/) and article.pdf
                  (template vendored in article/rescience-template/ from rescience/template@0978d0e)
```

## Reproduce (Linux/macOS, CPU only; ~3-4 h total on 8 shared cores, N-BEATS adds ~3 h)
```bash
# 1. environments (one venv per model family; uv >= 0.4)
for v in chronos timesfm tirex stats nbeats; do uv venv -p 3.11 venvs/$v; done
uv pip install -p venvs/chronos/bin/python -r envs/requirements-chronos.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -p venvs/timesfm/bin/python -r envs/requirements-timesfm.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -p venvs/tirex/bin/python   -r envs/requirements-tirex.txt   --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -p venvs/stats/bin/python   -r envs/requirements-stats.txt
uv pip install -p venvs/nbeats/bin/python  -r envs/requirements-nbeats.txt  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# 2. data (raw files are committed; to re-download see "Data sources" below)
venvs/stats/bin/python src/data.py

# 3. tests
venvs/stats/bin/python tests/test_metrics.py

# 4. forecasts + metrics
scripts/run_all.sh            # ARIMA (2 refit windows) + 4 TSFMs on heartrate, crime, glucose; then evaluate
scripts/run_nbeats_all.sh     # N-BEATS (2 selection rules); then evaluate
venvs/stats/bin/python src/make_figures.py   # article figures + results/verdict.csv

# 5. diagnosis (optional; needs a clone of the authors' repo, used only for DIAGNOSIS.md)
git clone https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration /tmp/authors && git -C /tmp/authors checkout b60bfd9
venvs/stats/bin/python diagnosis/build_authors_data.py /tmp/authors
diagnosis/run_authors_data.sh
venvs/stats/bin/python src/evaluate.py && venvs/stats/bin/python diagnosis/authors_style_metrics.py

# 6. tables, traceability, article
venvs/stats/bin/python scripts/make_tables.py
python3 scripts/make_traceability.py
cd article && make            # needs xelatex, biber (version matching biblatex; we used 2.20), latexmk, python3-yaml
```
On an M-series Mac, the same commands work (torch CPU/MPS wheels from PyPI; drop the CPU index URL).

## Data sources (all public)
- Crime: NYC Open Data *NYPD Complaint Data Historic* (`qgea-i56i`), aggregated server-side:
  `curl -G https://data.cityofnewyork.us/resource/qgea-i56i.csv --data-urlencode '$select=boro_nm,date_trunc_ymd(rpt_dt) AS day,count(*) AS n' --data-urlencode '$group=boro_nm,day' --data-urlencode '$order=day,boro_nm' --data-urlencode '$limit=200000'`
  (downloaded 2026-10-23; the live dataset can change).
- Glucose: PhysioNet *BIG IDEAs Lab Glycemic Variability and Wearable Device Data* v1.1.2 (https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/), `NNN/Dexcom_NNN.csv`
  (SHA-256 verified against the release's SHA256SUMS.txt).
- Heart-Rate: PhysioNet *Heart Rate Oscillations during Meditation* v1.0.0 (https://physionet.org/content/meditation/1.0.0/), `data/metron/M1..M14`.

## Rules followed
Metrics and loop were written from the paper text before the authors' repository was opened; the repository was
consulted only afterwards for diagnosis (`DIAGNOSIS.md`) and no code was copied. Every number in the article comes
from `results/*.csv|json`; paper numbers are quoted with table/figure references.

## Status (2026-10-23, 23:55 PDT)
- Finished: Chronos-Bolt-small, Chronos-Bolt-base, TimesFM-2.5-200M, ARIMA (64 / T / guarded) and N-BEATS (in-sample / hold-out) on Crime, Glucose and Heart-Rate. TiRex is finished on all three datasets (Glucose added 2026-10-24).
- Results: `results/summary.csv` (main metrics), `results/verdict.csv` (claim checks), `results/diagnosis_authors_style_metrics.csv`, `results/runlogs/*.json` (time and peak RSS per run), and `article/article.pdf`.
- Typical CPU times (8 shared cores): TSFMs 5-15 min per dataset, except TiRex on Glucose, which took 70 min (4,214 s). N-BEATS takes 12-25 min per dataset and variant.

## Round 2 (2026-10-24): seeds, determinism, full N-BEATS grid
- TSFM determinism: `scripts/run_determinism.sh` then `scripts/check_determinism.py`, written to `results/determinism.csv`.
  Chronos-Bolt and TimesFM are bitwise identical on a repeat. TiRex (CPU) differs only at float level; PCE changes by 1.5e-5.
- N-BEATS seeds 0, 1, 2 (reduced 8-config grid; seed 1 = the round-1 runs): `scripts/run_nbeats_seeds_box.sh`, then
  `scripts/seed_summary.py`. Results are in `results/seed_summary.csv` and `results/seed_verdict.csv`. Each run took
  10–25 min on 3 threads.
- Full paper grid (64 configs × 3 seeds × 2 selections × 3 datasets): `scripts/run_nbeats_full_gpu.sh`, environment
  `envs/requirements-nbeats-gpu.txt`, instructions in `RUN_ON_GPU.md`. Status: **done (2026-10-25)**.
  - It ran on Nittin's Apple M2 Max (32 GB) on MPS, with torch 2.5.1, Python 3.12.11 and neuralforecast 3.1.9
    (`results/runlogs/gpu_env.txt`).
  - It took 3 h 14 min wall-clock for all 18 runs, 533–1313 s each (`results/runlogs_nbeats_full.txt`).
  - The returned codebase matched commit 3e54d0c with no modifications, so only the full-grid outputs were copied
    in: runlogs, partial jsonl and forecasts. The forecasts are gitignored.
  - On macOS, `peak_rss_mb` in these runlogs is actually in KB because ru_maxrss is reported in bytes; e.g.
    2861408 ≈ 2.86 GB.
  - Tags: `nbeats[-holdout]-full-seed{0,1,2}`. The full grid is now the primary N-BEATS baseline in the article;
    the reduced grid is a sensitivity check.

## Round 4 (2026-10-25)
- **(A) ARIMA refit-context sweep**, run on the box: W ∈ {32, 64, 128, 256, T}.
  - Run `scripts/run_arima_ctx_sweep.sh` (resumable; log in `results/runlogs_arima_ctx_sweep.txt`), then
    `src/evaluate.py` and `scripts/arima_ctx_sweep.py`.
  - Outputs: `results/arima_ctx_sweep.csv`, `article/figures/arima_ctx.pdf` and `article/tables/arima_ctx*.tex`.
  - The 7 new runs took 15–163 s each.
- **(B)** Limitations paragraph in the article (§ Deviations).
- **(C)** `FEASIBILITY_6DATASETS.md`: data provenance and an M2 Max runtime estimate for Reviews, M5 and Patents.
  No runs were started.

## Round 6 (2026-11-04): Reviews, M5 and Patents folded in
- Stage 1 (Patents, M5) and stage 2 (Reviews) ran on the owner's M2 Max (see `STAGES.md`, TRACEABILITY D9/D10).
  Returned metrics: `results/stage1/`, `results/stage2/`; logs: `results/runlogs/`, `results/timing/`,
  `results/runlogs_stage{1,2}.txt`.
- Patents/M5 forecasts (returned, gitignored under `results/forecasts/`) are re-evaluated here; Reviews forecasts
  were not returned, so its rows come from the returned per-series metrics:
  `venvs/stats/bin/python src/evaluate.py --include-returned results/stage2`.
- Validity checks: `scripts/verify_returned_forecasts.py` (Patents, M5) and `scripts/verify_returned_stage2.py`
  (Reviews) → `results/stage1_validity/`. All checks passed; no rerun is needed.
- ARIMA on Patents/M5 was also re-run in full on the box (tags `arima-box`, `arima-ctxT-box`, sensitivity only).
- Regenerate: `src/make_figures.py`, `scripts/seed_summary.py`, `scripts/make_tables.py`,
  `scripts/make_traceability.py`, then `cd article && PATH=/usr/bin:$PATH make` (biber 2.20 is needed).
