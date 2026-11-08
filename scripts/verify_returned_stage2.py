"""Round 6: checks on the Mac-returned stage-2 (Reviews) outputs that can be made WITHOUT the forecast files.

Replicates: Adler et al., arXiv:2510.16060v2. The stage-2 return holds metrics, run logs and code but no .npz
forecasts, so Reviews cannot be re-evaluated or re-forecast on the box. What is checked here, from files only:
  - every run log reports the full window count of our protocol (src/data.py PROTOCOL, 235 x 1,526 = 358,610)
  - TSFM run logs name the expected HF model id (revisions are pinned in envs/MODEL_REVISIONS.md)
  - N-BEATS: 64 configurations scored, selected config = argmin train_wql (infeasible configs scored +inf)
  - the returned summary.csv equals mean/SEM recomputed from the returned metrics_per_series.csv
  - the stage-2 log reports the data checksum check passed (DATA_OK)
Usage: venvs/stats/bin/python scripts/verify_returned_stage2.py   -> results/stage1_validity/stage2_reviews.csv
"""
import glob
import json
import math
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
HF = {"chronos-bolt-small": "amazon/chronos-bolt-small", "chronos-bolt-base": "amazon/chronos-bolt-base",
      "timesfm-2.5-200m": "google/timesfm-2.5-200m-pytorch", "tirex": "NX-AI/TiRex"}
N_WIN = 358610  # [ASSUMPTION A-01 / src/data.py PROTOCOL] Reviews d=8 -> 235 x 1,526
rows = []
for p in sorted(glob.glob(os.path.join(RES, "runlogs", "reviews__*.json"))):
    d = json.load(open(p))
    m = os.path.basename(p)[len("reviews__"):-5]
    if "grid" in d:
        g = d["grid"]
        sc = [x["train_wql"] if x.get("error") is None and x.get("train_wql") is not None else math.inf for x in g]
        am = g[sc.index(min(sc))]["config_id"]
        sel = d["selected"]["config_id"] if isinstance(d.get("selected"), dict) else d.get("selected")
        rows.append(dict(dataset="reviews", model=m, check="N_grid_complete", ok=len(g) == 64, value=len(g), detail=""))
        rows.append(dict(dataset="reviews", model=m, check="N_selected_is_argmin", ok=am == sel, value=sel,
                         detail=f"argmin train_wql = {am}; device={d.get('device')}"))
    else:
        rows.append(dict(dataset="reviews", model=m, check="n_windows_full", ok=d["n_windows"] == N_WIN,
                         value=d["n_windows"], detail=f"device={d.get('device', 'cpu (ARIMA)')}"))
        if m in HF:
            rows.append(dict(dataset="reviews", model=m, check="hf_id", ok=d.get("hf_id") == HF[m], value=d.get("hf_id"),
                             detail="revision pinned in envs/MODEL_REVISIONS.md"))
st = os.path.join(RES, "stage2")
per = pd.read_csv(os.path.join(st, "metrics_per_series.csv"))
summ = pd.read_csv(os.path.join(st, "summary.csv")).set_index(["dataset", "model"])
cols = ["pce", "pce_per_window", "cce", "mase"]
g = per.groupby(["dataset", "model"])[cols]
re = ((g.mean() - summ[cols]).abs() / summ[cols].abs()).max().max()
rows.append(dict(dataset="reviews", model="all", check="E_summary_equals_per_series_mean", ok=bool(re < 1e-4),
                 value=f"{re:.1e}", detail="max rel diff; per-series CSV has 6 d.p."))
ns = per.groupby("model").series_id.nunique()
rows.append(dict(dataset="reviews", model="all", check="n_series", ok=bool((ns == 235).all()), value=int(ns.min()),
                 detail="235 in the UCI deposit (Table 1: 239)"))
log = open(os.path.join(RES, "runlogs_stage2.txt")).read()
rows.append(dict(dataset="reviews", model="", check="data_sha256_on_mac", ok="DATA_OK" in log, value="DATA_OK" in log,
                 detail="runlogs_stage2.txt"))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "stage1_validity", "stage2_reviews.csv"), index=False)
print(out.to_string(index=False))
print("ALL OK" if out.ok.all() else "SOME CHECKS FAILED")
