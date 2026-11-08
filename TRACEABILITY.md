# TRACEABILITY — paper element → code, and every assumption

Paper: Adler et al., arXiv:2510.16060**v2** (27 May 2026), `paper/adler2025_v2.pdf`. Every source file states the
version in its module docstring. Tag conventions in code:
`# [Paper §x, Eq. (n)]` = taken from the paper text; `# [ASSUMPTION: not specified; chose X because Y]` = gap we
filled; `# [FROM AUTHORS' CODE: file:line, commit]` = diagnosis-only borrowing; `[FROM AUTHORS' DATA: file, commit]`
= the authors' released *data* used as input (round 5 only). The primary implementation (`src/`) contains **no**
authors' code. The authors'-code tags are in `diagnosis/` (authors' aggregation definitions re-implemented, and
loaders for the authors' released CSVs, commit `b60bfd9`); they never feed the primary results or the verdict.
See DIAGNOSIS.md. Round 5 (6-dataset extension) reads two of the authors' processed datasets as data
(`src/data.py` `build_reviews`, `build_patents`; provenance in `data/authors/README.md`). M5 is rebuilt by us from
the public raw file.

## 1. Curated map of the paper's elements (claim C1 only)

| Paper element | What it specifies | Implementation |
|---|---|---|
| §3.1 Eq. (1), Q = {0.1..0.9} | PCE | `src/metrics.py` `pce`, `pce_from_freq`, `QUANTILES` |
| §3.1 "PCE in [0, 0.5]" | bounds | `tests/test_metrics.py::test_pce_bounds` |
| §3.1 Eq. (3), S = {0.2,0.4,0.6,0.8} | CCE (signed) | `src/metrics.py` `cce`, `LEVELS`, `interval_bounds` |
| §3.1 Eq. (2) | SIW | `src/metrics.py` `siw` |
| §3.1 Eq. (4) | MASE of median, in-window naive denominator | `src/metrics.py` `mase` |
| App. B.2 Eqs. (5)-(6) | WQL | `src/metrics.py` `wql` |
| §2.1 context y_{1:T}, horizon L | window construction | `src/forecast_io.py` `build_windows` |
| App. A.2 rolling origins, stride d ∈ {1,4,8} | evaluation loop | `src/data.py` `origins`, `PROTOCOL` |
| App. A.2 Table 2 (L, T, season, split) | per-dataset protocol | `src/data.py` `PROTOCOL`, `build_*` |
| App. A.6 + Table 1 | dataset construction | `src/data.py` `build_crime`, `build_glucose`, `build_heartrate` |
| App. A.6 + Tables 1-2 (round 5) | Reviews, M5, Patents | `src/data.py` `build_reviews` (authors' UCI deposit), `build_m5` (ours, from raw M5), `build_patents` (authors' CSV), `PROTOCOL`, `SOURCES` |
| §3.2 / App. A.4 quantile heads, zero-shot | TSFM inference | `src/run_tsfm.py` `make_predictor` |
| App. A.4 checkpoints | model ids | `src/run_tsfm.py` `HF_IDS` |
| App. A.4 ARIMA: StatsForecast AutoARIMA, refit, Gaussian PPF quantiles | ARIMA baseline | `src/run_arima.py` `select_order`, `forecast_one` |
| App. A.2 "max lag (context) 64" | ARIMA refit window | `src/run_arima.py` `main` (`--refit-context`) |
| App. A.2 Table 3 N-BEATS grid, WQL selection, single global model | N-BEATS | `src/run_nbeats.py` `GRID`, `fit`, `main` |
| Fig. 2/3 captions: mean and SEM across series | aggregation | `src/evaluate.py` `main` |
| §4.1 "close to or below 5%" | threshold for C1b | `src/make_figures.py` `CLOSE_TO_5` |
| Table 4 (PCE, CCE, MASE rows of all six datasets) | paper numbers quoted verbatim | `src/make_figures.py` `PAPER_T4` → `results/paper_table4_subset.csv` |
| §4.1 "except for the relatively difficult Patents dataset" | C1b not claimed on Patents | `src/make_figures.py` `C1B_NOT_CLAIMED` → `results/verdict.csv` column `paper_claims` |
| Tables 1/4 dataset order | Reviews, M5, Crime, Heart-Rate, Glucose, Patents | `src/make_figures.py` `DATASETS`, `scripts/make_tables.py` `DSN` |
| Article tables / prose numbers | generated, never hand-typed | `scripts/make_tables.py` → `article/tables/{main,main_new,verdict,seeds,diag,validity,numbers}.tex` |
| (round 6) returned Reviews metrics | merged without forecasts | `src/evaluate.py --include-returned results/stage2` (column `source`) |
| (round 6) validity of returned forecasts | windows, T/L, quantiles, N-BEATS selection, TSFM spot re-forecasts, ARIMA reselection | `scripts/verify_returned_forecasts.py`, `scripts/verify_returned_stage2.py` → `results/stage1_validity/` |
| (diagnosis) authors' metric aggregation | dataset-pooled PCE/CCE, global SIW/MASE denominators | `diagnosis/authors_style_metrics.py` → `results/diagnosis_authors_style_metrics.csv` |
| (diagnosis) authors' released data | Crime / HR CSVs rerun through our pipeline | `diagnosis/build_authors_data.py`, `diagnosis/run_authors_data.sh` → `results/summary.csv` rows `*_authorsdata` |

## 2. Deviations by design (not gaps in the paper; instructed scope)
- D1 TimesFM: `google/timesfm-2.5-200m-pytorch` instead of the paper's `timesfm-2.0-500m-pytorch` (App. A.4) — instructed small model.
- D2 Chronos-Bolt: we run both `small` (instructed) and `base` (paper's checkpoint, App. A.4).
- D3 Moirai 2.0 and YingLong not run (scope/budget). TiRex is the paper's checkpoint, run with the CPU `torch` backend.
- D4 Datasets: Crime, Glucose, Heart-Rate in rounds 1-4 (no synthetic data). Round 5 adds Reviews, M5, Patents (D9). See PLAN.md §2.
- D5 N-BEATS: the paper's full 64-config grid (Table 3) is the primary baseline (`nbeats[-holdout]-full-seed{0,1,2}`),
  run on an M2 Max with MPS (outputs accepted unchanged; codebase identical to commit 3e54d0c). The reduced
  8-config grid (`nbeats`, `nbeats-seed*`) is kept as a sensitivity check.
- D6 Sensitivity variants not in the paper, reported alongside the primary runs: `arima-ctxT` (refit on T points),
  `arima-guarded` (post-hoc explosive-forecast guard, added after 22/13,040 Glucose windows diverged),
  `nbeats-holdout` (hold-out selection). Primary verdict rows use `arima` and `nbeats-full-seed1` (3-seed summary in `results/seed_verdict.csv`); all variants are in `results/verdict.csv`.

- D7 ARIMA refit-context sweep: W ∈ {32, 64, 128, 256, T} (ASSUMPTION in `src/run_arima.py` and
  `scripts/arima_ctx_sweep.py`), with tags `arima-ctx{W}`. Not in the paper; it is a sensitivity analysis on the
  App. A.2 vs A.4 ambiguity.
- D8 Datasets beyond the current three: feasibility study in `FEASIBILITY_6DATASETS.md`; implemented in round 5 (D9).
- D9 Round 5, 6-dataset extension (runs on the owner's M2 Max via `run_stage1.sh` / `run_stage2.sh`):
  - Reviews: `[FROM AUTHORS' DATA]`, the authors' UCI deposit `y_amazon-google-large.csv` (sha256 in `SOURCES`).
    It has **235** series vs **239** in Table 1; we use all 235 (ASSUMPTION in `build_reviews`). Stride d=8 gives
    358,610 windows (Table 1: 360,490).
  - M5: built by us from the raw `sales_train_validation.csv` (dept × store sums, d_1..d_1912; ASSUMPTION on the
    1,912 days). It equals the authors' `y_m5.csv` exactly (diagnosis cross-check). Window count: 21,280 full-target
    windows (d=1) vs 62,160 in Table 1. No stride in {1,4,8} reproduces Table 1 under the Table 2 split. This is a
    documented discrepancy (ASSUMPTION in `PROTOCOL`).
  - Patents: `[FROM AUTHORS' DATA]`, the authors' `y_patents.csv` @ b60bfd9. Table 1's 10,956 = 83 × 132 counts
    origins with partial targets; we keep full 64-step targets as for all datasets, giving 5,727 windows
    (ASSUMPTION in `PROTOCOL`).
  - Device: TSFMs may run on MPS/CUDA. Each model's first 16 windows are checked against the CPU (relative error
    ≤ 1e-2), else that model runs on the CPU (logged). Optional `FC_FLOAT16=1` storage is used only when every
    coverage indicator is unchanged (ASSUMPTION in `src/forecast_io.py`). N-BEATS configs that cannot be trained
    are scored +inf (ASSUMPTION in `src/run_nbeats.py`).
  - Round 6 integration: stage 1 (Patents, M5; run Oct 25, 2026 16:31-19:23 PDT) and stage 2 (Reviews; Oct 26 15:21
    to Oct 27 10:17 PDT) ran on an M2 Max with code byte-identical to commit 3dddf5f. TimesFM-2.5 fell back to the
    CPU (MPS has no float64); Chronos-Bolt, TiRex and N-BEATS ran on MPS; ARIMA on the CPU. Returned metrics are kept
    in `results/stage1/` and `results/stage2/`, logs in `results/runlogs/`, `results/timing/`, `results/runlogs_stage{1,2}.txt`.
  - Patents/M5 forecasts were returned and are re-evaluated here (`results/forecasts/`, gitignored); the re-evaluation
    equals the returned `results/stage1/summary.csv` exactly. Reviews forecasts were not returned, so Reviews rows in
    `results/*.csv` are the returned per-series metrics (`source = returned:results/stage2`).
- D10 Cross-platform ARIMA sensitivity (round 6): AutoARIMA on M5 re-selects different orders for 2/70 series on x86,
  and short re-forecasts differ (platform floating point on ill-conditioned high-order fits). All four Patents/M5
  ARIMA runs were re-run in full on the x86 box with the same code and data (tags `arima-box`, `arima-ctxT-box`);
  PCE differs by at most 0.0007 and no verdict changes (`results/stage1_validity/arima_crossplatform.csv`). The
  M2 Max runs stay primary.

## 3. Auto-extracted tag index (file:line)
<!-- AUTO:BEGIN (generated by scripts/make_traceability.py; do not edit by hand) -->

### A. Every paper reference in code (83 tags)

| Paper element | Location |
|---|---|
| Paper Table 2 | `diagnosis/build_authors_data.py:42` |
| Paper App. A.2, Table 2 | `scripts/arima_ctx_sweep.py:3` |
| Paper App. A.2 | `scripts/arima_ctx_sweep.py:3` |
| Paper §4.1 | `scripts/arima_ctx_sweep.py:6` |
| Paper §3.1, Eq. (1) | `scripts/check_determinism.py:23` |
| Paper App. B.1, Table 4 | `scripts/make_tables.py:14` |
| Paper Table 1 / Table 4 | `scripts/make_tables.py:18` |
| Paper §4.1 | `scripts/make_tables.py:89` |
| Paper §4, claim C1a | `scripts/seed_summary.py:31` |
| Paper App. A.2, Table 2 | `scripts/verify_returned_forecasts.py:38` |
| Paper App. A.2, Table 2 | `src/data.py:21` |
| Paper App. A.2 | `src/data.py:22` |
| Paper App. A.2, Table 2 | `src/data.py:31` |
| Paper App. A.6 | `src/data.py:57` |
| Paper Table 1 | `src/data.py:59` |
| Paper Table 2 | `src/data.py:59` |
| Paper App. A.6 | `src/data.py:65` |
| Paper Table 2 | `src/data.py:72` |
| Paper App. A.6 | `src/data.py:77` |
| Paper Table 1 | `src/data.py:78` |
| Paper Table 2 | `src/data.py:78` |
| Paper App. A.6 | `src/data.py:108` |
| Paper Table 1 | `src/data.py:109` |
| Paper Table 2 | `src/data.py:110` |
| Paper Table 2 | `src/data.py:124` |
| Paper App. A.6 | `src/data.py:126` |
| Paper App. A.2 | `src/data.py:136` |
| Paper App. A.6 | `src/data.py:150` |
| Paper Table 1 | `src/data.py:152` |
| Paper Table 2 | `src/data.py:152` |
| Paper App. A.6 | `src/data.py:163` |
| Paper Table 1 | `src/data.py:164` |
| Paper Table 2 | `src/data.py:164` |
| Paper Table 2 | `src/data.py:176` |
| Paper App. A.6 | `src/data.py:181` |
| Paper Table 1 | `src/data.py:182` |
| Paper Table 2 | `src/data.py:183` |
| Paper App. A.2 | `src/data.py:207` |
| Paper Fig. 2/3 captions | `src/evaluate.py:7` |
| Paper Fig. 2/3 captions | `src/evaluate.py:81` |
| Paper §2.1 | `src/forecast_io.py:15` |
| Paper App. A.2 | `src/forecast_io.py:15` |
| Paper Table 2 | `src/forecast_io.py:23` |
| Paper App. A.2 | `src/forecast_io.py:24` |
| Paper Table 1 / Table 4 | `src/make_figures.py:26` |
| Paper §4.1 | `src/make_figures.py:30` |
| Paper §4.1 | `src/make_figures.py:33` |
| Paper App. B.1, Table 4 | `src/make_figures.py:37` |
| Paper §4.1, Fig. 3 top | `src/make_figures.py:69` |
| Paper App. B.1, Table 4 | `src/make_figures.py:103` |
| Paper §3.1, below Eq. (1) | `src/metrics.py:12` |
| Paper §3.1, below Eq. (3) | `src/metrics.py:14` |
| Paper §3.1, before Eq. (2) | `src/metrics.py:23` |
| Paper §3.1, Eq. (1) | `src/metrics.py:29` |
| Paper §3.1, Eq. (1) | `src/metrics.py:34` |
| Paper §3.1, Eq. (1) | `src/metrics.py:39` |
| Paper §3.1, Eq. (3) | `src/metrics.py:59` |
| Paper §3.1, text after Eq. (3) | `src/metrics.py:63` |
| Paper Eq. (3) | `src/metrics.py:67` |
| Paper §3.1, Eq. (2) | `src/metrics.py:73` |
| Paper §3.1, Eq. (4) | `src/metrics.py:89` |
| Paper App. B.2, Eqs. (5)-(6) | `src/metrics.py:100` |
| Paper App. A.4 | `src/run_arima.py:29` |
| Paper App. A.2, Table 2 | `src/run_arima.py:30` |
| Paper App. A.2 | `src/run_arima.py:42` |
| Paper App. A.4 | `src/run_arima.py:44` |
| Paper App. A.2 | `src/run_arima.py:97` |
| Paper App. A.2, Table 3 | `src/run_nbeats.py:37` |
| Paper App. A.2, Table 3 | `src/run_nbeats.py:45` |
| Paper App. A.2 | `src/run_nbeats.py:91` |
| Paper App. A.2 | `src/run_nbeats.py:102` |
| Paper App. A.2 | `src/run_nbeats.py:145` |
| Paper App. A.2 | `src/run_nbeats.py:146` |
| Paper App. A.2 | `src/run_nbeats.py:194` |
| Paper App. A.4, fn. 4 | `src/run_tsfm.py:29` |
| Paper App. A.4, fn. 5 | `src/run_tsfm.py:31` |
| Paper §3.2 | `src/run_tsfm.py:61` |
| Paper §2.1 | `src/run_tsfm.py:61` |
| Paper App. A.4 | `src/run_tsfm.py:76` |
| Paper §3.1 | `tests/test_metrics.py:24` |
| Paper §3.1 | `tests/test_metrics.py:32` |
| Paper §3.1 | `tests/test_metrics.py:39` |
| Paper Eq. (4) | `tests/test_metrics.py:51` |

### B. Every assumption in code (31 tags)

| # | Location | Assumption |
|---|---|---|
| A-01 | `scripts/arima_ctx_sweep.py:2` | ASSUMPTION: sweep W in {32, 64, 128, 256, T}; T = 512 (Crime, Glucose) or 128 (Heart-Rate), the TSFM context of [Paper App. A.2, Table 2 |
| A-02 | `src/data.py:23` | ASSUMPTION: not specified per dataset; chose d=4 for Crime (longest test span, 3287 days) and d=1 for Glucose and Heart-Rate, because these choices give per-series forecast counts closest to Table 1 (# Total Forecasts / # Time Series): Crime 4110/5=822 vs ours 806; Glucose 13840/16=865 vs ours 815; Heart-Rate 1800/6=300 vs ours 299. |
| A-03 | `src/data.py:32` | ASSUMPTION: stride (App. A.2: 'd in {1, 4, 8} dependent on time series length') chosen as for the first three datasets, by closeness to Table 1 (# Total Forecasts / # Time Series): Reviews d=8 -> 235 x 1,526 = 358,610 full-target windows (Table 1: 360,490); M5 d=1 -> 70 x 304 = 21,280 (Table 1: 62,160, which no d in {1,4,8} reproduces under the Table 2 split; documented discrepancy); Patents d=1 -> 83 x 69 = 5,727 (Table 1: 10,956 = 83 x 132 matches only if origins with partial targets are counted; we keep full L-step targets as for every other dataset). |
| A-04 | `src/data.py:61` | ASSUMPTION: 'daily reports' -> we count complaints by report date (RPT_DT), not occurrence date (CMPLNT_FR_DT), because the paper says 'number of daily reports' and RPT_DT is complete from 2006-01-01. |
| A-05 | `src/data.py:80` | ASSUMPTION: subjects start on different dates (2020-02-13 .. 2020-07-19), so one absolute split timestamp cannot apply to all. We interpret the split relative to each subject's start: 2020-02-16 12:43 minus the earliest recording start (subject 001, 2020-02-13 17:23:32) = 67.3 h = step 808 on a 5-min grid. |
| A-06 | `src/data.py:84` | ASSUMPTION: Table 1 reports 1,686 steps per series; we truncate every subject to its first 1,686 grid steps so all series have the paper's length. |
| A-07 | `src/data.py:95` | ASSUMPTION: not specified; readings are snapped to a regular 5-min grid anchored at the first reading and gaps (sensor dropouts, up to ~22 h in subject 015) are linearly interpolated, because all TSFMs and ARIMA assume regular sampling. |
| A-08 | `src/data.py:117` | ASSUMPTION: the paper uses 6 of the 14 'metron' subjects without naming them, and 744 steps (= length of M1 in beats). We treat each beat as one time step (beat k -> second k, matching 'Second' granularity and 744 = len(M1)), and keep the 6 longest records (M8 770, M1 744, M12 700, M11 665, M2 663, M10 662). |
| A-09 | `src/data.py:121` | ASSUMPTION: truncate the 6 records to the shortest kept length (662) so each yields 662-300-64+1 = 299 forecasts (Table 1: 1,800/6 = 300). |
| A-10 | `src/data.py:157` | ASSUMPTION: the deposit has 235 series, not 239 (Table 1); we use all 235 as released. |
| A-11 | `src/data.py:167` | ASSUMPTION: M5 has 1,913 training days; Table 1 says 1,912 steps, so we keep d_1..d_1912 (to 2016-04-23). |
| A-12 | `src/forecast_io.py:35` | ASSUMPTION: storage precision only. It is used only when it is lossless for PCE/CCE: every coverage indicator 1{y <= q} must be identical after rounding, and all values must be inside the float16 range. Otherwise this file stays float32. MASE/WQL/SIW change by <= 5e-4 relative. Default off. |
| A-13 | `src/make_figures.py:34` | ASSUMPTION: 'close to' operationalised a priori (PLAN.md) as PCE <= 0.06. |
| A-14 | `src/metrics.py:43` | ASSUMPTION: the paper writes Eq. (1) for a single forecast (t = T+1..T+L) but reports results 'aggregated over ... all possible context-prediction combinations' (§3.3) without saying whether the average is taken inside or outside /./. 'pooled' is our primary choice because with L=64 strongly autocorrelated steps the per-window estimator has a large positive finite-sample bias (a perfectly calibrated forecaster cannot reach the PCE values ~0.01 reported in Table 4). Both are reported. |
| A-15 | `src/metrics.py:75` | ASSUMPTION: the denominator 'y^{q_high} - y^{q_low}' is not defined further; we use the empirical q_high / q_low quantiles of the observed test values of the same series (`y_ref`), because it makes SIW a unit-free width relative to the spread of the series being forecast. |
| A-16 | `src/metrics.py:91` | ASSUMPTION: windows with zero naive MAE (constant targets) are skipped, because Eq. (4) is undefined there. |
| A-17 | `src/metrics.py:102` | ASSUMPTION: Eq. (6) has no 1//Q/ factor and no x100; we follow the equation literally, so absolute values are not comparable with Table 4 WQL (which look like percentages). Reported for context only. |
| A-18 | `src/run_arima.py:31` | ASSUMPTION: AutoARIMA search settings (max orders, information criterion, stepwise) not given; library defaults are used. |
| A-19 | `src/run_arima.py:51` | ASSUMPTION: not specified; statsforecast's default CSS-ML estimator occasionally fails on high-order fits to short contexts. We retry with pure ML and then CSS estimation (same orders); only if all fail do we fall back to a Gaussian random walk with the context's lag-1 innovation s.d. Counts per stage are in the run log. |
| A-20 | `src/run_arima.py:62` | ASSUMPTION: post-hoc sensitivity ('--guard', tag suffix '-guarded'), added after the unguarded Glucose run produced 22/13,040 windows with explosive forecasts (/q/ up to 1e11 on a 40-400 mg/dL series), which dominate window-averaged SIW/MASE. A fit is treated as failed if its 10%/90% quantiles leave [min(ctx) - 10*range(ctx), max(ctx) + 10*range(ctx) |
| A-21 | `src/run_arima.py:98` | ASSUMPTION: App. A.4 instead says the model is 'refit on all earlier data before each forecast'. We read A.2 literally: each refit sees the last 64 observations (primary, model tag 'arima'); as a sensitivity run the refit sees the same T-length context as the TSFMs (model tag 'arima-ctxT'). |
| A-22 | `src/run_arima.py:105` | ASSUMPTION: round-4 context-length sensitivity sweep (not in the paper, whose ARIMA context is stated as 64 in App. A.2 and 'all earlier data' in App. A.4). W in {32, 64, 128, 256, T}; W cannot exceed the TSFM context T because the evaluation windows store T points. Orders are still selected once on the train split; only the refit window changes. Tag 'arima-ctx{W}'; W=64 and W=T reuse 'arima'/'arima-ctxT'. |
| A-23 | `src/run_nbeats.py:39` | ASSUMPTION: reduced grid for CPU budget: 'epochs' -> neuralforecast max_steps {100,1000}, LR {1e-3,1e-4}, blocks {(1,1,1),(3,3,3)}; patience=-1 (no early stopping, so validation-check steps are irrelevant); loss = Quantile only, because the paper reports 'Quantile loss outperforms normal distribution loss on all datasets' (App. A.2). 8 configurations instead of 64. |
| A-24 | `src/run_nbeats.py:46` | ASSUMPTION: 'Epochs' -> neuralforecast max_steps (as in the reduced grid). Early stopping needs a validation split, which the paper does not describe: when patience = 2 we hold out the last L steps of each fitting series (neuralforecast val_size = L) and stop on validation loss; when patience = -1, val_size = 0 and val_check_steps has no effect on training (the two such configs differ only by their random stream). 'Normal Distribution Loss' -> DistributionLoss('Normal') with the paper's 'default 100 samples' (App. A.2) and our 9 quantiles. |
| A-25 | `src/run_nbeats.py:94` | ASSUMPTION: not specified; per-window standard scaling ('scaler_type=standard') so one global model can be trained across series of different scale; batch size, MLP width and stacks are neuralforecast defaults. |
| A-26 | `src/run_nbeats.py:147` | ASSUMPTION: 'same data' read two ways. insample (primary, literal): configs are fitted on the whole training split and scored on rolling windows inside it (stride 8). holdout (sensitivity): configs are fitted on the training split minus its last L steps and scored on the one window per series whose targets are that tail; the selected config is then refitted on the whole training split. The paper's grid contains early-stopping/validation parameters, which suggests some held-out validation, but it is not described. |
| A-27 | `src/run_nbeats.py:173` | ASSUMPTION: round 5. A configuration that cannot be trained on a dataset (e.g. too short for an early-stopping validation split on Patents with the hold-out reading) is scored +inf, so it cannot be selected. The error is logged in the record; the paper does not discuss this case. |
| A-28 | `src/run_tsfm.py:75` | ASSUMPTION: CPU-only box -> TiRex 'torch' backend instead of the default CUDA kernels; numerically the same architecture/weights. |
| A-29 | `src/run_tsfm.py:77` | ASSUMPTION: with L=64 (2 output patches) we expect this setting to have little or no effect, so we keep the library default (not verified against the authors' setting). |
| A-30 | `src/run_tsfm.py:93` | ASSUMPTION: the paper used TimesFM 2.0 (no config given). For 2.5 we use the model-card recommended inference flags; max_context = T (Table 2), max_horizon = L. use_continuous_quantile_head=True and fix_quantile_crossing=True are the card defaults. |
| A-31 | `src/run_tsfm.py:184` | ASSUMPTION: heads can return crossing quantiles; we sort along the quantile axis (monotone rearrangement) so every interval in Eq. (2)/(3) is well defined. Fraction of affected values is logged. |

### C. Uses of the authors' code (diagnosis only) and data (data sourcing only) (7 tags)

- `diagnosis/authors_style_metrics.py:27` # [FROM AUTHORS' CODE: src/metrics.py:302-318, b60bfd9] reported PCE = |q - mean 1[y<=yhat_q]| with the mean
- `diagnosis/authors_style_metrics.py:31` # [FROM AUTHORS' CODE: src/metrics.py:401-427, b60bfd9] CCE: coverage pooled over series/windows per horizon h,
- `diagnosis/authors_style_metrics.py:39` # [FROM AUTHORS' CODE: src/metrics.py:481-519, b60bfd9] SIW denominator = empirical (q_high - q_low)
- `diagnosis/authors_style_metrics.py:43` # [FROM AUTHORS' CODE: src/metrics.py:155-173 and 260-281, b60bfd9] MASE: naive MAE = in-sample one-step
- `diagnosis/build_authors_data.py:21` # [FROM AUTHORS' CODE: data/police/y_police.csv, data/meditation/y_meditation.csv, commit b60bfd9] data only
- `src/data.py:154` [FROM AUTHORS' DATA: y_amazon-google-large.csv, the authors' UCI deposit (doi:10.24432/C5SP7K), referenced by
- `src/data.py:184` [FROM AUTHORS' DATA: data/patents/y_patents.csv, commit b60bfd9 (sha256 in SOURCES). The field/category

<!-- AUTO:END -->
