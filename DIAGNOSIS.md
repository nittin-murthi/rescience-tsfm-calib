# DIAGNOSIS: our implementation compared with the authors' released code

Replicates: Adler et al., *Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated?*, arXiv:2510.16060**v2** (27 May 2026).

**When and how.** We opened the authors' repository only *after* our own results existed for all three datasets:
- HR and Crime were complete for every model.
- Glucose was complete for ARIMA, Chronos-Bolt and TimesFM.
- See commit `ceaa658` and the earlier history.

We used it **only to diagnose differences**.
- No authors' code was copied into `src/`.
- Two diagnosis-only scripts live in `diagnosis/`. They re-implement the authors' *aggregation definitions* (tagged `[FROM AUTHORS' CODE: file:line, commit]`) and load the authors' released CSV data. They never feed the primary results or the verdict.

- Repository: https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration
- Commit inspected: `b60bfd92ff836525773c71039a83ba2fe3d123bf` (2026-02-25 16:24 PST)
- Local clone, outside the project and not versioned: `/workspace/_authors_ref/repo`
- Diagnosis outputs:
  - `results/diagnosis_authors_style_metrics.csv`
  - `results/summary.csv`: rows with dataset `*_authorsdata`
  - `results/runlogs_diag_authorsdata.txt`

## 0. What the repo does and does not contain

| Component | In repo? | Consequence |
|---|---|---|
| Metric code (`src/metrics.py`) | yes | aggregation questions (A1, A10, MASE) can be resolved, §1 |
| TSFM inference (`src/forecaster.py`, `src/run_forecaster.py`) | yes | model IDs and windowing confirmed, §3 |
| Processed Crime / Glucose / Heart-Rate CSVs (`data/*/y_*.csv`) | yes | data construction can be compared, §2 |
| **ARIMA baseline code** | **no** (only `statsforecast==2.0.1` in `environments/main.requirements.txt:29`) | the ARIMA gap cannot be diagnosed from code, §4 |
| **N-BEATS baseline code** | **no** (only `neuralforecast==3.0.0`, line 28) | N-BEATS selection cannot be diagnosed, §4 |
| Per-dataset forecast date / stride configuration | no (README example only) | stride choice (A-01) stays an assumption |

## 1. Metric aggregation

### 1.1 PCE pooling: assumption A1 / A-09
`metric_pce`, `src/metrics.py:302-318`:
- The indicator mean is taken separately for each quantile level q, pooled over **all rows**: every series, every forecast origin and every horizon.
- The function then averages |q − mean| over q.
- This is the value the script prints and writes as "PCE" (lines 960, 1009).
- The per-series variant, `metric_pce_by_series` (lines 321-333), pools over the origins and horizons of one series.

In our terms, the paper's PCE is **dataset-level pooling** (`pce_pooled_dataset` in `results/pce_variants.csv`).
- Our primary reading, "pooled within series, then averaged over series", is the closest of our three readings and gives nearly identical numbers. Example for Crime TiRex: 0.0230 under our primary reading versus 0.0229 authors-style (`results/summary.csv`, `results/diagnosis_authors_style_metrics.csv`).
- The literal per-window reading (`pce_per_window`, 0.11–0.23) is **not** what the authors compute. That confirms the a-priori reason for A1.
- Effect on the verdict: none for Crime and Glucose.
  - For Heart-Rate the gap is larger (e.g. TimesFM 0.0356 primary vs 0.0152 authors-style). Pooling over 6 short series averages out per-series miscalibration.
  - Neither reading changes the TSFM-versus-baseline ordering for the primary ARIMA. Both readings are reported.

### 1.2 CCE
`metric_cce_multi_from_pivot`, `src/metrics.py:401-427`:
- For each horizon, coverage is pooled over all series and origins.
- CCE is s − coverage, averaged over S = {0.8, 0.6, 0.4, 0.2} and then over horizons.
- Our CCE pools within each series and then averages over series. With equal windows per series the two agree to 4 decimals: Crime TiRex −0.0485 in both files.

### 1.3 SIW denominator: assumption A-10
`metric_siw_multi_from_pivot`, `src/metrics.py:481-519`:
- The denominator is the empirical quantile range of **all observations in the obs file**: all series, train and test together. It is one global value per s.
- We used each series' own test values.
- This changes SIW's absolute level only, e.g. Crime TiRex 0.750 (ours) vs 0.277 (authors-style). SIW is not part of the tested claim.

### 1.4 MASE: explains our MASE gap
`compute_naive_mae`, lines 155-173, and `metric_mae_and_mase`, lines 260-281:
- The naive denominator is the in-sample one-step persistence MAE over the **whole obs file**, pooled across series. It is not the per-window in-sample naive error that we read Eq. (4) to mean.
- MASE_h is clipped to [0, 50] and averaged over horizons.

Recomputed authors-style (`results/diagnosis_authors_style_metrics.csv`):

| Dataset (our data) | TSFM MASE, ours | TSFM MASE, authors-style | Paper Table 4 TSFM MASE |
|---|---|---|---|
| Crime | 0.783–0.805 | 0.721–0.744 | 0.740–0.761 |
| Glucose | 6.57–7.29 | 6.12–6.60 | 5.84–6.36 |
| Heart-Rate | 1.46–1.55 | 1.17–1.26 | 1.10–1.26 |

On the authors' own Heart-Rate data, authors-style TSFM MASE is 1.086 (TimesFM-2.5) and 1.121 (Chronos-Bolt-S), against Table 4's 1.10–1.12. Our earlier MASE gap is therefore almost entirely a **definition difference**, not a forecasting difference.

The clip at 50 also hides explosive forecasts. Our unguarded Glucose ARIMA gets authors-style MASE = 50.0 (the clip value), compared with 1.8e5 in our window-averaged MASE.

### 1.5 WQL: assumption A-12
`metric_wql`, line 557 onward:
- Default `scale_mode=sum_y`; README command uses `--wql_reduce mean --wql_scale_mode sum_y`.
- It is scaled per horizon by the summed |y|, which differs from our literal Eq. (6).
- WQL is outside the tested claim and we did not reconcile it.

## 2. Dataset construction

These checks compared the authors' `data/*/y_*.csv` against our `data/processed/*.csv`, using a short script run interactively.

| Dataset | Authors' file | Ours | Finding |
|---|---|---|---|
| Crime | 5 series × 6574 days, 2006-01-01..2023-12-31 | 5 × 6574, same dates | Same shape and borough order. Values differ: correlation 0.72–0.85, exactly equal on only 1–4% of days. The authors' first value for Bronx is 421 vs our 228. A large Jan-1 count is typical of **occurrence-date** (CMPLNT_FR_DT) counting, so the authors appear to count by occurrence date where we count by report date (A-02). |
| Glucose | 16 × 1673, all relabelled to start at 2020-02-13 17:23 | 16 × 1686 on a 5-min grid, split relative to each subject's start | **Confirms our A-03**: relative-time alignment, with all subjects mapped onto subject 001's clock. Values are close but not identical (mean abs diff 0.05–19 mg/dL by subject), which points to different gridding and gap filling (A-05). Length is 1673 vs 1686 (Table 1 says 1,686). |
| Heart-Rate | 6 × 598 one-second steps, starting 00:00:01 | 6 × 662 beat-indexed steps | **Same 6 subjects** (M1, M2, M8, M10, M11, M12), confirming our choice in A-06. The authors **resample to a 1-second grid**: some beats are dropped (their 2nd value equals our 3rd), giving 598 steps instead of our 662 beats. Their file gives 6 × 236 = 1,416 forecasts at stride 1, not Table 1's 1,800. Our 1,794 is closer to Table 1, but the released data and the table disagree. |

### Rerunning our own pipeline on the authors' data
Diagnosis only:
- Data built by `diagnosis/build_authors_data.py`; runs launched by `diagnosis/run_authors_data.sh`.
- Models: ARIMA (64), ARIMA (T), Chronos-Bolt-S and TimesFM-2.5.
- Source: `results/summary.csv` rows `crime_authorsdata` and `heartrate_authorsdata`; authors-style columns from `results/diagnosis_authors_style_metrics.csv`.

| Dataset | Model | PCE (ours, primary) | PCE authors-style | MASE authors-style |
|---|---|---|---|---|
| Crime (authors' data) | Chronos-Bolt-S | 0.0138 | 0.0131 | 0.745 |
| | TimesFM-2.5 | 0.0189 | 0.0185 | 0.732 |
| | ARIMA (64) | 0.0655 | 0.0647 | 0.908 |
| | ARIMA (T) | 0.0281 | 0.0271 | 0.833 |
| Heart-Rate (authors' data) | Chronos-Bolt-S | 0.0602 | 0.0441 | 1.121 |
| | TimesFM-2.5 | 0.0529 | 0.0181 | 1.086 |
| | ARIMA (64) | 0.0664 | 0.0589 | 1.371 |
| | ARIMA (T) | 0.0635 | 0.0553 | 1.179 |

What this shows:
- The data differences do **not** change the qualitative picture. TSFMs keep PCE well below 0.05 on Crime. On Heart-Rate they stay below ARIMA (64), and ARIMA (T) comes close.
- ARIMA PCE stays near 0.06. The data construction therefore does **not** explain the gap to the paper's ARIMA PCE (0.168 Crime, 0.204 Heart-Rate; Table 4).

## 3. TSFM inference

| Item | Authors' code | Ours | Comment |
|---|---|---|---|
| Chronos-Bolt | `amazon/chronos-bolt-base` (`src/forecaster.py:477-478`), chronos-forecasting 1.5.2 | bolt-small (instructed) + bolt-base | Our bolt-base matches the paper's model. |
| TimesFM | `google/timesfm-2.0-500m-pytorch`, context_len = T (`src/forecaster.py:437-452`), forked timesfm `Coaster41/timesfm@e3752c2` | `timesfm-2.5-200m` (instructed substitute) | Model substitution, D-deviation; not comparable 1:1. |
| TiRex | `NX-AI/TiRex` (`src/forecaster.py:485-486`), forked tirex (`latent-extraction` branch), GPU | same weights, CPU `torch` backend | |
| Moirai | `Salesforce/moirai-2.0-R-small` (`src/forecaster.py:460`) | not run | The paper text says "Moirai 2.0" without naming the size. |
| Windows | GluonTS `split(date=forecast_date)` + `generate_instances(windows=(total−64)//stride, distance=stride, max_history=T)` (`src/run_forecaster.py:77-101`) | origins from the split to n−L, step = stride, context = last T | Equivalent up to an off-by-one in the window count: the authors use `(total−L)//stride`, which drops the last window. Negligible. |

## 4. Baselines (not explained by the released material)

Sign convention: CCE = s − coverage ([Paper §3.1], Eq. (3)).
- Positive CCE means intervals that are too narrow (overconfident).
- Negative CCE means intervals that are too wide (under-confident).
- The paper reports both baselines as consistently under-confident (§4.1; Table 4 CCE −0.17 to −0.41 on our datasets).

- **ARIMA.** Our ARIMA is consistently *better calibrated* than the paper's: PCE 0.055–0.066 vs 0.141–0.204 (Table 4).
  - This holds on the authors' data and under the authors' aggregation.
  - It errs in the **same direction** as the paper's ARIMA: under-confident, CCE −0.02 to −0.14. The magnitude is smaller.
  - The ARIMA code is not in the repository. Plausible but untestable explanations:
    1. A different predictive variance. For example, a Gaussian fitted to the forecasts in a way that inflates the spread (App. A.4: "fitting a normal distribution to the forecasts").
    2. A different refit context. App. A.2 says "max lag (context) 64"; App. A.4 says "refit on all earlier data".
    3. Different AutoARIMA settings or version: they used statsforecast 2.0.1, we use 2.1.1.
  - Our ARIMA (T) sensitivity run shows how much the refit context alone matters. With context T it nearly ties the TSFMs on Crime and Heart-Rate.
- **N-BEATS.** Neither our selection rule reproduces the paper's under-confident N-BEATS (Table 4 CCE −0.21 to −0.24).
  - Our literal "in-sample" selection picks the most flexible configuration. It is strongly **overconfident**: CCE +0.43 (HR) and +0.44 (Crime).
  - Our hold-out selection is closer to calibrated but still overconfident (CCE HR +0.016, Crime +0.123).
  - The N-BEATS code, grid and selection protocol are not released, so this cannot be diagnosed further.

## 5. Summary of diagnosis

| Question | Resolved? | Answer |
|---|---|---|
| A1 PCE aggregation | yes | The authors pool over the whole dataset. Our primary reading (pool within each series) is numerically close; the per-window reading is not used by the authors. |
| MASE gap | yes | Global in-sample naive denominator plus clip at 50, against our per-window denominator. Authors-style MASE matches Table 4 closely. |
| SIW level | yes | Global train+test denominator. Level only; not in the claim. |
| Crime data | partly | The authors count by occurrence date, we count by report date. Same qualitative result. |
| HR data | yes | Same 6 subjects. The authors use a 1-second grid (598 steps), we use the beat index (662). Same qualitative result. |
| Glucose data | yes | Relative-time alignment confirmed (our A-03). Small value differences from gridding. |
| ARIMA calibration gap | **no** | Baseline code not released. |
| N-BEATS sign of CCE | **no** | Baseline code not released. |
| Strides | no | Not in repo; our A-01 stands. |

No change was made to the primary pipeline as a result of this diagnosis. All findings are reported as differences in the article (§ Diagnosis).


## 6. Round 2 update (2026-10-24): N-BEATS CCE sign across seeds
- The opposite CCE sign is unchanged. All 18 reduced-grid runs (3 seeds × 2 selection readings × 3 datasets) have
  positive CCE, i.e. they are overconfident; see `results/seed_verdict.csv`. In-sample runs sit at +0.43 to +0.46,
  hold-out runs at +0.01 to +0.12. The paper reports −0.17 to −0.24 (Table 4).
- Likely mechanism: in-sample selection always picks 1000 steps, lr 1e-3 and blocks (3,3,3), which nearly memorises
  the training windows (selection WQL 0.003–0.063, `results/runlogs/*__nbeats*.json`). The resulting intervals are
  too narrow on the test windows. This does not explain how the paper got under-confident N-BEATS, and the authors'
  baseline code is still not released.
- Full grid: the run on Nittin's M2 Max (MPS) has replaced the pending note above.

## 7. Round 3 update (2026-10-25): full 64-config grid, CCE sign
- The full Table 3 grid, which includes Normal loss and early stopping, does **not** remove the sign discrepancy.
- The 9 full-grid in-sample runs have CCE between +0.43 and +0.46: mean ± SD Crime .444±.001, Glucose .461±.001,
  HR .432±.002. They are overconfident, as on the reduced grid.
- In-sample selection never picks a Normal-loss or early-stopped configuration. It always picks
  1000/1e-3/pat −1/(3,3,3)/Quantile, as config 37 or 39, which differ only in the validation-check setting.
- Full-grid hold-out CCE means: Crime .006±.011, Glucose .123±.046, HR .018±.005. Only one of the 18 full-grid
  runs is negative: hold-out Crime seed 2, at −0.006. That is close to calibrated, not the paper's −0.17 to −0.24.
- Hold-out Crime now beats the worst TSFM in PCE for all 3 seeds (.0256/.0343/.0167 vs .0347), so C1a fails there.
- The mechanism is unchanged and the cause of the paper's under-confident N-BEATS is unresolved, because the
  baseline code is unreleased.
- Source files: `results/seed_summary.csv`, `results/seed_verdict.csv` and `results/runlogs/*full-seed*.json`.


## 8. Round 4 (2026-10-25): ARIMA context sweep and the ARIMA gap
- The refit context moves ARIMA PCE: Crime .112 → .024 for W = 32 → 512; HR .086 → .055 for W = 32 → 128;
  Glucose .055 → .067.
- The CCE becomes more negative (more under-confident) at short W on Crime and HR (−.249 and −.176 at W = 32).
  This is the same direction as the paper's ARIMA.
- No W reaches the paper's ARIMA PCE (.141–.204, Table 4), so the refit context alone does not explain the gap.
- Source: `results/arima_ctx_sweep.csv`.
