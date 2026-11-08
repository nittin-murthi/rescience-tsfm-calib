"""Compute metrics from stored forecasts and write results/*.csv.

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026). Section numbers refer to v2.
Usage (any venv with pandas): python src/evaluate.py
Outputs:
  results/metrics_per_series.csv   one row per (dataset, model, series)
  results/summary.csv              per (dataset, model): mean over series and SEM  [Paper Fig. 2/3 captions]
  results/pce_variants.csv         PCE under the three aggregation readings (assumption A1)
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import metrics as M  # noqa: E402
from data import ROOT, load  # noqa: E402
from forecast_io import FC_DIR  # noqa: E402

TSFMS = ["chronos-bolt-small", "chronos-bolt-base", "timesfm-2.5-200m", "tirex"]


def evaluate_file(path):
    ds, model = os.path.basename(path)[:-4].split("__")
    z = np.load(path, allow_pickle=False)
    q, y, sid = z["quantiles"].astype(float), z["target"].astype(float), z["series_id"]
    series, meta = load(ds)
    rows = []
    for k in sorted(set(sid.tolist())):
        m = sid == k
        y_test = series[k][meta["split"]:]  # reference spread for SIW (assumption in metrics.siw)
        rows.append(dict(dataset=ds, model=model, series_id=k, n_windows=int(m.sum()),
                         pce=M.pce(y[m], q[m], "pooled"),            # [Eq. (1)] primary reading (A1)
                         pce_per_window=M.pce(y[m], q[m], "per_window"),
                         cce=M.cce(y[m], q[m]),                       # [Eq. (3)]
                         siw=M.siw(y[m], q[m], y_test),               # [Eq. (2)]
                         mase=M.mase(y[m], q[m]),                     # [Eq. (4)]
                         wql=M.wql(y[m], q[m])))                      # [App. B.2 Eq. (6)]
    # dataset-level pooled PCE (third reading of A1): pool indicators over all series of the dataset
    pooled_ds = M.pce(y, q, "pooled")
    return rows, dict(dataset=ds, model=model, pce_pooled_dataset=pooled_ds)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, help="only these datasets (default: all forecasts)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"), help="where the three CSVs go")
    ap.add_argument("--include-returned", nargs="*", default=[],
                    help="dirs holding metrics_per_series.csv + pce_variants.csv returned from another machine, "
                         "for datasets whose forecasts are not on this machine (e.g. results/stage2 = Reviews)")
    a = ap.parse_args()
    out = a.out_dir
    per, dsp = [], []
    for p in sorted(glob.glob(os.path.join(FC_DIR, "*.npz"))):
        if a.datasets and os.path.basename(p).split("__")[0] not in a.datasets:
            continue
        r, d = evaluate_file(p)
        per += r
        dsp.append(d)
    df = pd.DataFrame(per)
    df["source"] = "evaluated-here"
    dsp = pd.DataFrame(dsp)
    for d in a.include_returned:
        # Rows evaluated by this same evaluate.py on the machine that holds the forecasts (see STAGES.md);
        # only datasets with no local forecasts are taken, so local evaluation always wins.
        rp = pd.read_csv(os.path.join(d, "metrics_per_series.csv"), dtype={"series_id": str})
        rv = pd.read_csv(os.path.join(d, "pce_variants.csv"))
        new = sorted(set(rp["dataset"]) - set(df["dataset"]))
        rp = rp[rp["dataset"].isin(new)].copy()
        rp["source"] = "returned:" + os.path.relpath(os.path.abspath(d), ROOT)
        df = pd.concat([df, rp], ignore_index=True)
        dsp = pd.concat([dsp, rv[rv["dataset"].isin(new)][["dataset", "model", "pce_pooled_dataset"]]],
                        ignore_index=True)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "metrics_per_series.csv"), index=False, float_format="%.6f")
    cols = ["pce", "pce_per_window", "cce", "siw", "mase", "wql"]
    g = df.groupby(["dataset", "model"])
    # [Paper Fig. 2/3 captions] dataset value = mean across time series; error bars = SEM across time series
    mean = g[cols].mean()
    sem = g[cols].sem(ddof=1).add_suffix("_sem")
    summ = pd.concat([mean, sem, g["series_id"].count().rename("n_series"), g["n_windows"].sum(),
                      g["source"].first()], axis=1).reset_index()
    summ["family"] = np.where(summ["model"].isin(TSFMS), "TSFM", "baseline")
    summ.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.6f")
    var = summ[["dataset", "model", "pce", "pce_per_window"]].merge(dsp, on=["dataset", "model"])
    var.to_csv(os.path.join(out, "pce_variants.csv"), index=False, float_format="%.6f")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ[["dataset", "model", "n_series", "n_windows", "pce", "pce_sem", "pce_per_window", "cce",
                    "siw", "mase", "wql"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
