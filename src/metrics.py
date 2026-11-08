"""Calibration and accuracy metrics, implemented from the paper text only.

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026). Section/equation numbers refer to v2.
Pure numpy (importable from every venv).

Array conventions (all functions):
  y   : observed targets, shape (W, L)          W forecast windows, L = prediction length
  qf  : predicted quantiles, shape (W, L, |Q|)   aligned with QUANTILES
"""
import numpy as np

# [Paper §3.1, below Eq. (1)] "q in Q = {0.1, 0.2, ..., 0.9} is the set of quantiles that we average over"
QUANTILES = np.round(np.arange(1, 10) / 10.0, 1)
# [Paper §3.1, below Eq. (3)] "For both CCE and SIW, we average over s in {0.2, 0.4, 0.6, 0.8}"
LEVELS = np.array([0.2, 0.4, 0.6, 0.8])


def _qidx(q):
    return int(np.argmin(np.abs(QUANTILES - q)))


def interval_bounds(s):
    """[Paper §3.1, before Eq. (2)] central interval with symmetric q_low/q_high, s = q_high - q_low
    (e.g. s=0.8 -> 0.1/0.9). Returns indices into QUANTILES."""
    return _qidx(0.5 - s / 2), _qidx(0.5 + s / 2)


def coverage_indicators(y, qf):
    """[Paper §3.1, Eq. (1)] 1[y_t <= yhat_t^q] for every window, step and quantile -> (W, L, |Q|) bool."""
    return y[..., None] <= qf


def pce_from_freq(freq):
    """[Paper §3.1, Eq. (1)] PCE = 1/|Q| sum_q | q - (empirical frequency of y <= yhat^q) |."""
    return float(np.mean(np.abs(QUANTILES - freq)))


def pce(y, qf, mode="pooled"):
    """[Paper §3.1, Eq. (1)] Probabilistic Calibration Error for one time series.

    mode="pooled":    empirical frequency pooled over all windows and all L steps of the series, then Eq. (1).
    mode="per_window": Eq. (1) applied literally to each L-step forecast window, then averaged over windows.
    [ASSUMPTION: the paper writes Eq. (1) for a single forecast (t = T+1..T+L) but reports results
     'aggregated over ... all possible context-prediction combinations' (§3.3) without saying whether the
     average is taken inside or outside |.|. 'pooled' is our primary choice because with L=64 strongly
     autocorrelated steps the per-window estimator has a large positive finite-sample bias (a perfectly
     calibrated forecaster cannot reach the PCE values ~0.01 reported in Table 4). Both are reported.]
    """
    ind = coverage_indicators(y, qf)
    if mode == "pooled":
        return pce_from_freq(ind.reshape(-1, ind.shape[-1]).mean(axis=0))
    if mode == "per_window":
        freq_w = ind.mean(axis=1)  # (W, |Q|) mean over t = T+1..T+L
        return float(np.mean(np.abs(QUANTILES[None, :] - freq_w)))  # mean over q, then over windows
    raise ValueError(mode)


def cce(y, qf, mode="pooled"):
    """[Paper §3.1, Eq. (3)] Centered Calibration Error:
    CCE = 1/|S| sum_s ( s - 1/L sum_t 1[yhat^{q_low} <= y <= yhat^{q_high}] ).  No absolute value: positive =
    more observations outside the interval than expected (overconfident), negative = under-confident.
    Signed, so pooling vs per-window averaging gives identical results when all windows have length L;
    `mode` is accepted for API symmetry. [Paper §3.1, text after Eq. (3)]"""
    vals = []
    for s in LEVELS:
        lo, hi = interval_bounds(s)
        inside = (qf[..., lo] <= y) & (y <= qf[..., hi])  # [Paper Eq. (3)] inclusive bounds
        vals.append(s - inside.mean())
    return float(np.mean(vals))


def siw(y, qf, y_ref):
    """[Paper §3.1, Eq. (2)] Scaled Interval Width: 1/L sum_t (yhat^{q_high} - yhat^{q_low}) / (y^{q_high} - y^{q_low}),
    averaged over s in LEVELS.
    [ASSUMPTION: the denominator 'y^{q_high} - y^{q_low}' is not defined further; we use the empirical
     q_high / q_low quantiles of the observed test values of the same series (`y_ref`), because it makes SIW
     a unit-free width relative to the spread of the series being forecast.]"""
    vals = []
    for s in LEVELS:
        lo, hi = interval_bounds(s)
        den = np.quantile(y_ref, QUANTILES[hi]) - np.quantile(y_ref, QUANTILES[lo])
        if den <= 0:
            continue
        vals.append(np.mean(qf[..., hi] - qf[..., lo]) / den)
    return float(np.mean(vals)) if vals else float("nan")


def mase(y, qf):
    """[Paper §3.1, Eq. (4)] MASE of the median forecast per window, naive (lag-1) MAE inside the window as the
    denominator ('using information from the future', text after Eq. (4)); averaged over windows.
    [ASSUMPTION: windows with zero naive MAE (constant targets) are skipped, because Eq. (4) is undefined there.]"""
    med = qf[..., _qidx(0.5)]
    num = np.mean(np.abs(med - y), axis=1)
    den = np.mean(np.abs(np.diff(y, axis=1)), axis=1)  # 1/(L-1) sum_{t=T+2}^{T+L} |y_t - y_{t-1}|
    ok = den > 0
    return float(np.mean(num[ok] / den[ok]))


def wql(y, qf):
    """[Paper App. B.2, Eqs. (5)-(6)] WQL = sum_t sum_q p_q(y_t, yhat_t^q) / sum_t |y_t|, with
    p_q = 2(1-q)(yhat-y) if yhat >= y else 2q(y-yhat). Computed per window and averaged over windows.
    [ASSUMPTION: Eq. (6) has no 1/|Q| factor and no x100; we follow the equation literally, so absolute values
     are not comparable with Table 4 WQL (which look like percentages). Reported for context only.]"""
    diff = qf - y[..., None]
    loss = np.where(diff >= 0, 2 * (1 - QUANTILES) * diff, 2 * QUANTILES * (-diff))
    num = loss.sum(axis=(1, 2))
    den = np.abs(y).sum(axis=1)
    ok = den > 0
    return float(np.mean(num[ok] / den[ok]))
