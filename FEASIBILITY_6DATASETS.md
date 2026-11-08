# Feasibility: extending to all 6 real datasets (Reviews, Shopping (M5), Patents)

Round 4, 2026-10-25. This is a planning document. **No runs were started.** The authors were not contacted.

Sources:
- Paper v2: Table 1 (§3.3), Table 2 (App. A.2), App. A.6, Table 4.
- Authors' repository at commit b60bfd9 (MIT licence), local clone at `/workspace/_authors_ref/repo`, used only
  for data sourcing.
- The authors' UCI deposit (DOI 10.24432/C5SP7K).

Any use of these files would be tagged `[FROM AUTHORS' DATA]`.

## What the paper specifies

| | Reviews | Shopping (M5) | Patents |
|---|---|---|---|
| Series × steps (Table 1) | 239 × 13,000, hourly | 70 × 1,912, daily | 83 × 408, monthly |
| Total forecasts (Table 1) | 360,490 | 62,160 | 10,956 |
| L / T / ARIMA season (Table 2) | 64 / 512 / 24 | 64 / 512 / 7 | 64 / 128 / 1 |
| Train/test split (Table 2) | 2020-01-31 23:00 | 2015-04-23 | 2004-01-01 |
| Construction (App. A.6) | Amazon 2023 + Google Local reviews, aggregated by product/location category; the 239 most abundant categories sampled; hourly bins; trimmed to 2020-01-01 08:00 – 2021-06-25 23:00 | daily M5 sales binned by product department × store ID, giving 70 series | USPTO historical patent files (Marco et al. 2015), monthly filings 1981–2014; sparse series removed; aggregated by patent field and category; series with minimum value < 100 removed |
| Paper Table 4 PCE: TSFMs / N-BEATS / ARIMA | .012–.015 / .061 / .176 | .031–.052 / .108 / .131 | .154–.259 / .242 / .247 |

## What is obtainable

### Reviews
- **Data:** the authors' preprocessed file `y_amazon-google-large.csv` from UCI (`fetch_ucirepo(id=1276)` or a
  direct zip of 18 MB; the CSV is 110 MB).
  - Checked here: **235** series × 13,000 hourly steps, 2020-01-01 08:00 → 2021-06-25 23:00, no missing values.
  - This matches Table 1 and App. A.6 except for the series count: **235 vs 239**.
  - sha256 `dfc5e616…bae81`.
- **Rebuilding from raw** (Amazon Reviews 2023 plus Google Local, hundreds of GB of JSON) is infeasible. The
  category sampling is also under-specified.
- **Verdict:** obtainable only as the authors' processed data, which is the same position as the authors' data
  we already used for Crime and Heart-Rate. The 239 → 235 gap must be documented.
- **Windows:** with stride d = 8, our rule gives 235 × 1,526 = **358,610** windows. The paper reports 360,490,
  which fits d = 8.

### Shopping (M5)
- **Data:** the authors' `data/m5/y_m5.csv`: 70 series × 1,912 days, 2011-01-29 → 2016-04-23. This matches
  Table 1 (sha256 `24b7acc8…3150f`).
- **Rebuilding from raw:** independent reconstruction is possible from public M5 `sales_train_*`. Kaggle needs
  an account, but Nixtla's public mirror `github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip`
  (50 MB, reachable 2026-10-25) does not. Department (7) × store (10) gives 70 series, which
  matches. We could reconstruct it and cross-check against the authors' file.
- **Open issue:** under the Table 2 split (2015-04-23), the test span is 367 days, which gives 70 × 304 =
  **21,280** windows (d = 1). Table 1 says 62,160 (888 per series). No stride/split combination we tried
  reproduces 888 (d = 1 counting every origin gives 25,690). Either Table 1 or Table 2 is inconsistent for M5, or
  "total forecasts" counts something else.
- **Verdict:** faithful, using our own reconstruction plus the authors' file for cross-check. The window count
  has to be marked as an assumption (Table 2 split).

### Patents
- **Data:** the authors' `data/patents/y_patents.csv`: 83 series × 408 months, 1981-01 → 2014-12, minimum value
  105. This matches Table 1 and the "min ≥ 100" filter (sha256 `760dfad2…e0a42a`).
- **Rebuilding from raw:** USPTO HPDF is public, but "aggregated by patent field and category" (which taxonomy?
  which sparse-series filter?) does not identify the 83 series. An independent rebuild would be a substitution.
- **Windows:** the Table 2 split gives 132 test months. Counting every origin up to the end gives 83 × 132 =
  10,956, which **exactly** matches Table 1. Counting only origins with a full 64-step target gives 83 × 69 =
  5,727. We would keep our full-target rule, as for the other datasets, and document the difference.
- **Verdict:** obtainable only as the authors' processed data.
- **Caveat:** in the paper's own Table 4, YingLong (.259) is worse than both baselines (.242/.247), so C1a does
  not hold on Patents even in the paper.

## Compute estimate for Nittin's M2 Max (32 GB)

This is an estimate, not a measurement. It is based on our measured runs:
- Box CPU (shared, 4 threads), seconds per 1,000 windows at T = 512: Chronos-Bolt-S 4–55, Chronos-Bolt-B 36–99,
  TimesFM-2.5 33–111, TiRex 203–323.
- At T = 128 (Heart-Rate), seconds per 1,000 windows: 1.5 / 4.3 / 8.1 / 182.
- ARIMA (4 processes): 22–36 ms of wall time per window.
- M2 Max MPS, full-grid N-BEATS: 533–1,313 s per run on datasets with ≤ 13k windows. N-BEATS training cost is
  set by the step count, not dataset size; test prediction is batched (2,048 windows per call).

| Dataset | Windows (ours) | TSFMs (4) | ARIMA (64) | N-BEATS full grid, 3 seeds × 2 readings | Total |
|---|---|---|---|---|---|
| Reviews | 358,610 | box-CPU equivalent 27–59 h; M2 ≈ 6–15 h (assumes 4× speedup; TiRex on MPS is untested) | 2–6 h (season 24) | ≈ 1.5–3.5 h | **≈ 10–25 h** |
| M5 | 21,280 | box 1.6–3.5 h; M2 ≈ 0.5–1 h | ≈ 10–15 min | ≈ 1–1.3 h | **≈ 2–3 h** |
| Patents | 5,727 | ≈ 0.3 h box; M2 < 0.2 h | < 2 min | ≈ 1 h | **≈ 1–1.5 h** |

- **Disk:** the Reviews forecasts are 358,610 × 64 × 9 float32 ≈ 0.83 GB per model, so about 9 GB for 11 model
  runs. Reviews contexts are about 0.73 GB in RAM, which fits in 32 GB but not alongside other jobs on the shared
  15 GB box.

## Recommendation

1. **Package a one-command runner**, like `run_gpu.sh`, covering all models for M5 and Patents (stage 1,
   ≈ 3–5 h, one evening). Make Reviews a separate stage 2 (≈ 10–25 h, overnight or a weekend) with resume,
   because it costs about 20× the three current datasets combined.
2. **Required code changes before packaging, all method-preserving:**
   - `src/data.py` loaders for the three datasets: M5 rebuilt from raw and cross-checked; Reviews and Patents
     from the authors' processed files, tagged `[FROM AUTHORS' DATA]`.
   - Stride for Reviews d = 8, M5 and Patents d = 1 (ASSUMPTION).
   - A `--device` option in `src/run_tsfm.py` (currently hard-coded to CPU), with a timing trial on the first
     ~2,000 windows that extrapolates and prints an ETA, as the N-BEATS package did.
   - A multi-process ARIMA run that uses the M2's cores.
   - Optionally float16 forecast storage for Reviews.
3. **Alternative without the M2:** M5 and Patents TSFMs + ARIMA fit on the box (≈ 2–4 h CPU), but full-grid
   N-BEATS still needs the M2. Reviews should not run on the shared box.
