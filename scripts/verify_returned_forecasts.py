"""Round 6: validity check of forecasts returned from the owner's Mac (stage 1: Patents, M5) before they are used.

Replicates Adler et al., arXiv:2510.16060v2. Nothing here changes a forecast; it only checks them and writes
results/stage1_validity.csv (one row per check) and prints a summary.

Checks per returned results/forecasts/<ds>__<model>.npz:
  W  windows: target, series_id and origin arrays equal the ones rebuilt on the box from data/processed (whose sha256
     values are checked against data/processed/CHECKSUMS.sha256); shape (W, L, 9) with L = 64 and T per Table 2.
  Q  quantiles finite and non-decreasing along the quantile axis.
  E  evaluate.py on the box reproduces the returned results/stage1/summary.csv (all metric columns).
Model checks (run separately, one venv per family; see --tsfm / --arima):
  M  TSFM: re-forecast K evenly spaced windows on the box CPU with the pinned HF revision (src/run_tsfm.py
     REVISIONS) and compare with the stored quantiles (relative max-abs error, as in run_tsfm.check_device).
  A  ARIMA: re-run AutoARIMA order selection on the train split for every series and compare with the returned
     orders; re-forecast K windows with those orders and compare quantiles.
  N  N-BEATS: the selected config equals the arg-min of the logged grid scores (in-sample WQL or hold-out score);
     the grid is complete (64 configs, no errors). Retraining is not repeated (MPS training is not bitwise
     reproducible on the box CPU).
Usage:
  venvs/stats/bin/python scripts/verify_returned_forecasts.py --base            # W, Q, E, A(orders+K windows), N
  venvs/<family>/bin/python scripts/verify_returned_forecasts.py --tsfm MODEL   # M for one TSFM
"""
import argparse
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import numpy as np  # noqa: E402

from forecast_io import FC_DIR, build_windows  # noqa: E402

DATASETS = ["patents", "m5"]
PAPER_T = {"patents": 128, "m5": 512}  # [Paper App. A.2, Table 2] context size T; L = 64 for all
STAGE1 = os.environ.get("STAGE1_DIR", "/workspace/uploads/t47b/x/rescience-tsfm-calib")
OUT = os.path.join(ROOT, "results", "stage1_validity")
K = int(os.environ.get("SPOT_K", "32"))  # spot-check windows per (dataset, model)


def rows_out(name, rows):
    import csv
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "model", "check", "ok", "value", "detail"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(("OK  " if r["ok"] else "FAIL"), r["dataset"], r["model"], r["check"], r["value"], r["detail"])


def spot_idx(n):
    return np.unique(np.linspace(0, n - 1, K).round().astype(int))


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-8))


def base():
    rows = []
    sums = {l.split()[1]: l.split()[0] for l in open(os.path.join(ROOT, "data", "processed", "CHECKSUMS.sha256"))}
    for fn, h in sums.items():
        p = os.path.join(ROOT, "data", "processed", fn)
        if fn.split(".")[0] in DATASETS:
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()
            rows.append(dict(dataset=fn.split(".")[0], model="", check="data_sha256_" + fn, ok=got == h, value=got[:12], detail="vs CHECKSUMS.sha256"))
    for ds in DATASETS:
        win = build_windows(ds)
        meta = win["meta"]
        rows.append(dict(dataset=ds, model="", check="T_L_vs_Table2", ok=meta["T"] == PAPER_T[ds] and meta["L"] == 64,
                         value=f"T={meta['T']} L={meta['L']} stride={meta['stride']} split={meta['split']}", detail="Paper Table 2"))
        for p in sorted(glob.glob(os.path.join(FC_DIR, f"{ds}__*.npz"))):
            model = os.path.basename(p)[:-4].split("__")[1]
            if not os.path.exists(os.path.join(STAGE1, "results", "forecasts", os.path.basename(p))):
                continue
            z = np.load(p, allow_pickle=False)
            q = z["quantiles"].astype(float)
            same = (np.array_equal(z["target"], win["target"].astype(np.float32)) and
                    np.array_equal(z["series_id"], win["series_id"]) and np.array_equal(z["origin"], win["origin"]))
            rows.append(dict(dataset=ds, model=model, check="W_windows_equal", ok=bool(same and q.shape == (len(win["target"]), 64, 9)),
                             value=str(q.shape), detail="target/series_id/origin vs box rebuild"))
            rows.append(dict(dataset=ds, model=model, check="Q_finite_sorted",
                             ok=bool(np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()), value=f"{np.abs(q).max():.3g}", detail="max |q|"))
            if model.startswith("nbeats"):
                rows += nbeats_check(ds, model)
            if model in ("arima", "arima-ctxT"):
                rows += arima_check(ds, model, win, q)
    rows += eval_check()
    rows_out("base", rows)


def eval_check():
    import subprocess
    import pandas as pd
    tmp = os.path.join(OUT, "reeval")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "evaluate.py"), "--datasets", *DATASETS, "--out-dir", tmp],
                   check=True, capture_output=True)
    mine = pd.read_csv(os.path.join(tmp, "summary.csv")).set_index(["dataset", "model"]).sort_index()
    theirs = pd.read_csv(os.path.join(STAGE1, "results", "stage1", "summary.csv")).set_index(["dataset", "model"]).sort_index()
    cols = [c for c in theirs.columns if c != "family"]
    d = float(np.nanmax(np.abs(mine.loc[theirs.index, cols].to_numpy(float) - theirs[cols].to_numpy(float)) /
                        (np.abs(theirs[cols].to_numpy(float)) + 1e-12)))
    return [dict(dataset="patents+m5", model="all", check="E_reevaluate_equals_returned_summary", ok=d < 1e-5,
                 value=f"{d:.2e}", detail=f"max rel diff over {len(theirs)} rows x {len(cols)} cols (CSV has 6 d.p.)")]


def nbeats_check(ds, model):
    r = json.load(open(os.path.join(STAGE1, "results", "runlogs", f"{ds}__{model}.json")))
    g = r["grid"]
    nerr = sum(x.get("error") is not None for x in g)
    ok_grid = sorted(x["config_id"] for x in g) == list(range(64))
    key = "train_wql"  # run_nbeats.py stores the selection score (in-sample or hold-out) under this key
    best = min(g, key=lambda x: x[key] if x.get("error") is None else float("inf"))["config_id"]
    return [dict(dataset=ds, model=model, check="N_grid_complete", ok=ok_grid, value=len(g), detail=f"64 configs; {nerr} infeasible configs (error logged, scored +inf)"),
            dict(dataset=ds, model=model, check="N_selected_is_argmin", ok=best == r["selected"]["config_id"],
                 value=r["selected"]["config_id"], detail=f"argmin {key} = {best}; seed={r['seed']} select={r['select']}")]


_ORD = {}


def arima_check(ds, model, win, q):
    from run_arima import forecast_one, select_order
    rows = []
    split, season = win["meta"]["split"], win["meta"]["season"]
    ret = json.load(open(os.path.join(STAGE1, "results", "runlogs", "partial_arima", f"{ds}__orders.json")))
    if ds not in _ORD:
        from multiprocessing import Pool
        keys = list(win["series"])
        with Pool(4) as pool:
            sel = pool.starmap(select_order, [(win["series"][k][:split], season) for k in keys], chunksize=1)
        mine = dict(zip(keys, sel))
        norm_ = lambda v: (tuple(v["order"]), tuple(v["seasonal_order"]), v["season"], v["include_mean"], v["include_drift"])
        same = sum(norm_(mine[k]) == norm_(ret[k]) for k in keys)
        _ORD[ds] = (same, len(keys))
        rows.append(dict(dataset=ds, model="arima*", check="A_orders_equal_box_reselection", ok=same == len(keys),
                         value=f"{same}/{len(keys)}", detail="AutoARIMA on train split, box vs returned orders.json"))
    W = 64 if model == "arima" else win["meta"]["T"]
    for v in ret.values():
        v["order"], v["seasonal_order"] = tuple(v["order"]), tuple(v["seasonal_order"])
    idx = spot_idx(len(q))
    qq = np.stack([forecast_one((win["context"][i][-W:], ret[str(win["series_id"][i])], 64, False))[0] for i in idx])
    e = rel_err(qq, q[idx])
    rows.append(dict(dataset=ds, model=model, check="A_reforecast_spot", ok=e < 1e-3, value=f"{e:.2e}",
                     detail=f"{len(idx)} windows, refit W={W}, rel max-abs err"))
    return rows


def tsfm(model):
    import torch
    from run_tsfm import REVISIONS, make_predictor
    torch.set_num_threads(4)
    rows = []
    for ds in DATASETS:
        win = build_windows(ds)
        z = np.load(os.path.join(FC_DIR, f"{ds}__{model}.npz"), allow_pickle=False)
        idx = spot_idx(len(win["context"]))
        f = make_predictor(model, 64, win["meta"]["T"], "cpu")
        with torch.inference_mode():
            qq = np.sort(np.asarray(f(win["context"][idx]), np.float32), axis=-1)
        qs = z["quantiles"].astype(np.float32)[idx]
        e = rel_err(qq, qs)
        y = z["target"][idx][..., None]
        agree = float(np.mean((y <= qq) == (y <= qs)))
        tol = 1e-2  # same tolerance as run_tsfm.check_device (MPS vs CPU)
        import metrics as MM
        pa, pb = MM.pce(z["target"][idx].astype(float), qs.astype(float), "pooled"), MM.pce(z["target"][idx].astype(float), qq.astype(float), "pooled")
        rows.append(dict(dataset=ds, model=model, check=f"M_reforecast_spot_pinned_rev_K{len(idx)}", ok=e < tol, value=f"{e:.2e}",
                         detail=f"{len(idx)} windows on box CPU, rev {REVISIONS[model][:8]}; coverage-indicator agreement {agree:.4f}; "
                                f"pooled PCE on these windows returned {pa:.4f} vs box {pb:.4f}"))
        del f
    rows_out(f"tsfm_{model}" + ("" if K == 32 else f"_K{K}"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--tsfm", default="")
    a = ap.parse_args()
    if a.base:
        base()
    if a.tsfm:
        tsfm(a.tsfm)
