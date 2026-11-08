# [Re] Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated? A Partial Replication

Code, result summaries and ReScience C article for an independent **partial replication** of

> Coen Adler, Yuxin Chang, Felix Draxler, Samar Abdi and Padhraic Smyth. *Beyond Accuracy: Are Time Series
> Foundation Models Well-Calibrated?* ICLR 2026. arXiv:2510.16060v2 (https://arxiv.org/abs/2510.16060v2).

Author: Nittin Murthi Dhekshinamoorthy, University of Illinois Urbana-Champaign (ndhek2@illinois.edu).
Article: [`article/article.pdf`](article/article.pdf) (ReScience C template). Metadata: [`article/metadata.yaml`](article/metadata.yaml).

## What is tested

Only the main calibration claim (paper §4.1, Fig. 3, Table 4), split into two sub-claims per dataset:

- **C1a (ranking):** every tested TSFM has lower probabilistic calibration error (PCE) than the ARIMA and N-BEATS baselines.
- **C1b (level):** every tested TSFM has PCE close to or below 0.05 (fixed in advance as PCE ≤ 0.06, `PLAN.md`).

Datasets: all six real datasets of the paper (Reviews, M5, NYC Crime, Heart-Rate, Glucose, Patents).
Models: Chronos-Bolt small and base, TimesFM 2.5 200M (a **substitute** for the paper's TimesFM 2.0 500M), TiRex,
ARIMA (StatsForecast AutoARIMA; refit context 64 primary, T as sensitivity) and N-BEATS (NeuralForecast; the
paper's 64-configuration grid, in-sample and hold-out selection, seeds 0, 1, 2). Moirai 2.0 and YingLong were not run.

## Result (see the article for numbers and caveats)

Overall verdict: **partially reproduced.** C1b holds on the five datasets where the paper claims it; Patents,
which the paper exempts, stays high. C1a holds against in-sample-selected N-BEATS but depends on the baseline
otherwise: ARIMA (64) is better calibrated than TimesFM 2.5 on M5 and than every TSFM on Patents, and the
ARIMA refit on the TSFM context and hold-out-selected N-BEATS beat some TSFMs on several datasets. Per-dataset
checks are in `results/verdict.csv` and `results/seed_verdict.csv`; limitations are in the article's
"Deviations, limitations and incomplete items" section.

## Repository layout

```
src/          data.py (dataset builders), metrics.py (PCE, CCE, SIW, MASE), run_tsfm.py, run_arima.py,
              run_nbeats.py, forecast_io.py, evaluate.py, make_figures.py
scripts/      run_all.sh, run_nbeats_all.sh, run_nbeats_full_gpu.sh, stage_common.sh, fetch_raw_data.sh,
              prepare_data.py, make_tables.py, seed_summary.py, make_traceability.py, verify_returned_*.py, ...
run_stage1.sh, run_stage2.sh   one-command runners for Patents + M5 and for Reviews (CPU/MPS/CUDA)
tests/        metric property tests
diagnosis/    post-hoc, diagnosis-only scripts (comparison with the authors' aggregation and data)
envs/         exact package pins per environment; Hugging Face model revisions (MODEL_REVISIONS.md)
data/         SHA-256 manifests and protocol JSON files only (raw/processed data are downloaded, not shipped)
results/      summary.csv, metrics_per_series.csv, pce_variants.csv, verdict.csv, seed_*.csv, stage1/, stage2/,
              stage1_validity/, determinism.csv, arima_ctx_sweep.csv, timing/, runlogs/
article/      ReScience C LaTeX sources, tables/, figures/, article.pdf
docs/         DEVELOPMENT_LOG.md (chronological record of the work rounds)
PLAN.md, TRACEABILITY.md, DIAGNOSIS.md, STAGES.md, FEASIBILITY_6DATASETS.md, RUN_ON_GPU.md
```

`TRACEABILITY.md` maps every paper element to file:line. Code is tagged `[Paper §x, Eq. (n)]`,
`[ASSUMPTION: ...]`, `[FROM AUTHORS' DATA: ...]` or (diagnosis only) `[FROM AUTHORS' CODE: ...]`.

## Reproducing

See **[REPRODUCE.md](REPRODUCE.md)** for the exact environments, data downloads, commands per dataset and the
outputs each step writes. Short version:

```bash
bash scripts/fetch_raw_data.sh                       # Crime, Heart-Rate, Glucose raw data + checksums
# create venvs from envs/requirements-*.txt (REPRODUCE.md, step 1), then:
venvs/stats/bin/python src/data.py                   # build Crime, Heart-Rate, Glucose
venvs/stats/bin/python tests/test_metrics.py
scripts/run_all.sh && scripts/run_nbeats_all.sh      # CPU runs on the three original datasets
bash run_stage1.sh && bash run_stage2.sh             # Patents + M5, then Reviews (own venvs; MPS/CUDA if present)
```

## Rules followed

Metrics and the forecasting loop were written from the paper text before the authors' repository was opened; it
was consulted only afterwards for diagnosis (`DIAGNOSIS.md`) and no code was copied. Every number in the article
comes from files in `results/`; paper numbers are quoted with table or figure references. The original authors
were not contacted.

## License

Original code and documentation: MIT (`LICENSE`). The ReScience article template files keep GPL-3.0-or-later,
the fonts keep SIL OFL 1.1 / Apache 2.0, and the article text is CC BY 4.0; see `THIRD_PARTY_NOTICES.md`.
Data and model weights are not redistributed.

## AI disclosure

The author used Grok 4.7 in preparing this replication.
