# PLAN — Partial replication of Adler et al., "Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated?"

Paper: arXiv:2510.16060**v2** (27 May 2026; ICLR 2026 camera-ready). PDF: `paper/adler2025_v2.pdf`, text dump `paper/paper_v2.txt`.
Replication owner: Nittin Murthi Dhekshinamoorthy (UIUC). Implementation drafted with AI assistance (see article disclosure).

## 1. Claim under test (one claim only)
**C1 (Paper §4.1, Fig. 3 top, Table 4 PCE block; Abstract; §5):** zero-shot TSFMs are better calibrated than the
baselines (N-BEATS, ARIMA) as measured by Probabilistic Calibration Error (PCE, Eq. 1), and TSFM PCE is
"in general close to or below 5%" (except Patents).

Operationalised as two falsifiable sub-claims, checked per dataset:
- **C1a (ranking):** every tested TSFM has lower dataset-level PCE than ARIMA (and N-BEATS if run).
- **C1b (level):** every tested TSFM has PCE ≤ ~0.05 ("close to or below 5%"; we treat ≤ 0.06 as "close to", stated a priori).

Supporting (not separately verdicted): CCE sign of baselines (paper: ARIMA under-confident), MASE (accuracy competitive).

Out of scope: §4.2 as a separate claim, §4.3 prediction heads, §4.4 AR long-horizon, synthetic data, GIFT-Eval, WQL/MSIS analysis.

## 2. Datasets (3 of the paper's 6; all public, no credentials)
| Dataset | Source | Paper setup (Table 1/2) | Why included |
|---|---|---|---|
| Crime (NYC) | NYC Open Data NYPD Complaint Data Historic (qgea-i56i), daily counts per borough | 5 series, daily, 2006–2023, split 2015-01-01, T=512, L=64, ARIMA season 7 | human-behaviour count data, seasonal |
| Glucose | PhysioNet BIG IDEAs Lab Glycemic Variability v1.1.2, Dexcom G6 | 16 series, 5-min, split 2020-02-16 12:43, T=512, L=64, season 1 | out-of-pretraining-domain physiology; weak PACF (hard) |
| Heart-Rate | PhysioNet Meditation DB (Peng et al. 1999), `metron` group | 6 series, "second", split 00:04:59, T=128, L=64, season 1 | tiny, short-context regime (T=128) |

Representativeness: covers both of the paper's dataset families (human-behaviour: Crime; physiological / not
like pretraining data: Glucose, Heart-Rate), three granularities (daily, 5-min, beat/second), both context sizes
used in the paper (512 and 128), and spans the paper's TSFM PCE range on non-Patents data (0.014–0.077, Table 4).
Excluded: M5 (instructed; also in Moirai/YingLong pretraining), Reviews (239 series x 13k steps: too expensive on
shared CPU, and requires re-aggregating two large review corpora), Patents (USPTO aggregation underspecified; paper
already reports all methods fail there).

## 3. Models
| Role | Paper | This replication | Note |
|---|---|---|---|
| TSFM | chronos-bolt-base | **amazon/chronos-bolt-small** + **amazon/chronos-bolt-base** | base is exactly the paper's checkpoint and is cheap on CPU |
| TSFM | timesfm-2.0-500m-pytorch | **google/timesfm-2.5-200m-pytorch** | instructed substitute (smaller, newer); deviation |
| TSFM | TiRex (base) | **NX-AI/TiRex** | same checkpoint as paper; CPU torch backend |
| Baseline | ARIMA (StatsForecast AutoARIMA) | StatsForecast AutoARIMA | same library |
| Baseline | N-BEATS (grid search) | optional, only if cheap | |
Not run: Moirai 2.0, YingLong (scope/budget).

## 4. Metrics (from paper text, §3.1)
- PCE, Eq. (1), Q = {0.1,…,0.9}.
- CCE, Eq. (3), S = {0.2,0.4,0.6,0.8} central intervals (q_low = 0.5 − s/2, q_high = 0.5 + s/2).
- SIW, Eq. (2), same S (denominator ambiguous → assumption).
- MASE, Eq. (4), median forecast, naive denominator inside the window.
- WQL, Eq. (5)-(6) (Appendix B.2), reported for context only.

## 5. Evaluation protocol (Appendix A.2)
- L = 64; context T per Table 2; rolling origins starting at the train/test split, shifted by stride d ∈ {1,4,8}
  "dependent on time series length"; forecast until the end of the series.
- TSFMs zero-shot; quantile heads read directly at 0.1..0.9.
- ARIMA: AutoARIMA order selection on the train split (season per Table 2), then the selected order is refit before
  every forecast; quantiles from the Gaussian predictive (inverse normal CDF).
- Aggregation: metric per series, mean across series; SEM across series (Fig. 2/3 captions).

## 6. Known gaps → assumptions (full list with code locations in TRACEABILITY.md)
A1 PCE aggregation level (per window vs pooled); A2 SIW denominator; A3 stride per dataset; A4 Heart-Rate: which 6
of 14 subjects / time axis; A5 Glucose: timeline alignment and length; A6 Crime: date field; A7 ARIMA refit
window ("max lag (context) 64"); A8 TimesFM 2.5 config flags; A9 N-BEATS.

## 7. Order of work
(a) plan → (b) metrics + loop + ARIMA on one dataset → (c) TSFMs → (d) full subset → (e) only then open authors' repo,
record in DIAGNOSIS.md → (f) article + TRACEABILITY.md → (g) README + env files.

## 8. Compute constraints
Shared 8-CPU / 15 GB box: ≤ 4 torch threads, batches of ≤ 64 windows, one model process at a time, peak RSS target < 3 GB.
