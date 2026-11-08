"""DIAGNOSIS ONLY (step e): recompute dataset-level metrics on OUR forecasts using the aggregation choices found
in the authors' evaluation code, to test whether aggregation explains differences from Table 4.
Our own re-implementation of the definitions (no code copied). Not used for primary results.

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026).
Usage: venvs/stats/bin/python diagnosis/authors_style_metrics.py  -> results/diagnosis_authors_style_metrics.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load  # noqa: E402
from metrics import QUANTILES  # noqa: E402

FC = os.path.join(ROOT, "results", "forecasts")
S = [0.2, 0.4, 0.6, 0.8]
QI = {round(q, 2): i for i, q in enumerate(QUANTILES)}


def authors_style(q, y, series):
    """q: (W, L, 9) sorted quantiles, y: (W, L) targets, series: dict of full series."""
    # [FROM AUTHORS' CODE: src/metrics.py:302-318, b60bfd9] reported PCE = |q - mean 1[y<=yhat_q]| with the mean
    #  pooled over ALL series, windows and horizons, then averaged over q (dataset-level pooling).
    ind = (y[..., None] <= q)
    pce = float(np.mean(np.abs(QUANTILES - ind.reshape(-1, 9).mean(0))))
    # [FROM AUTHORS' CODE: src/metrics.py:401-427, b60bfd9] CCE: coverage pooled over series/windows per horizon h,
    #  s - cov_h, averaged over S, then over h.
    cce_s, siw_s = [], []
    yall = np.concatenate(list(series.values()))
    for s in S:
        lo, hi = QI[round((1 - s) / 2, 2)], QI[round(1 - (1 - s) / 2, 2)]
        inside = (q[..., lo] <= y) & (y <= q[..., hi])
        cce_s.append(np.mean(s - inside.mean(0)))
        # [FROM AUTHORS' CODE: src/metrics.py:481-519, b60bfd9] SIW denominator = empirical (q_high - q_low)
        #  quantile range of ALL observations in the obs file (all series, train+test), one global value.
        den = np.quantile(yall, 1 - (1 - s) / 2) - np.quantile(yall, (1 - s) / 2)
        siw_s.append(np.mean((q[..., hi] - q[..., lo]).mean(0) / den))
    # [FROM AUTHORS' CODE: src/metrics.py:155-173 and 260-281, b60bfd9] MASE: naive MAE = in-sample one-step
    #  persistence over the whole obs file (all series pooled); MASE_h = clip(MAE_h / naive, 0, 50); mean over h.
    naive = np.mean(np.concatenate([np.abs(np.diff(v)) for v in series.values()]))
    mae_h = np.abs(y - q[..., QI[0.5]]).mean(0)
    mase = float(np.mean(np.clip(mae_h / naive, 0, 50)))
    return dict(pce_authors_style=pce, cce_authors_style=float(np.mean(cce_s)),
                siw_authors_style=float(np.mean(siw_s)), mase_authors_style=mase, naive_mae_global=float(naive))


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(FC, "*.npz"))):
        ds, model = os.path.basename(p)[:-4].split("__")
        z = np.load(p)
        series, _ = load(ds)
        q = np.sort(z["quantiles"].astype(float), axis=-1)
        rows.append(dict(dataset=ds, model=model, **authors_style(q, z["target"].astype(float), series)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "diagnosis_authors_style_metrics.csv"), index=False, float_format="%.5f")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
