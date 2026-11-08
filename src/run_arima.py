"""ARIMA baseline (StatsForecast AutoARIMA order selection + refit before every forecast).

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026). Section numbers refer to v2.
Usage (stats venv): python src/run_arima.py --dataset crime --refit-context 64
"""
import argparse
import json
import os
import resource
import sys
import time
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from forecast_io import build_windows, save_forecast  # noqa: E402
from metrics import QUANTILES  # noqa: E402

warnings.filterwarnings("ignore")


def select_order(y_train, season):
    """[Paper App. A.4] 'Nixtla's StatsForecast implementation of AutoARIMA ... to automatically select the optimal
    ARIMA parameters for each time series on the training set.' [Paper App. A.2, Table 2] seasonality per dataset.
    [ASSUMPTION: AutoARIMA search settings (max orders, information criterion, stepwise) not given; library
     defaults are used.]"""
    from statsforecast.models import AutoARIMA
    m = AutoARIMA(season_length=season).fit(y_train)
    p, q, P, Q, mm, d, D = m.model_["arma"]
    coef = m.model_["coef"]
    return dict(order=(int(p), int(d), int(q)), seasonal_order=(int(P), int(D), int(Q)), season=int(mm),
                include_mean="intercept" in coef, include_drift="drift" in coef)


def forecast_one(args):
    """[Paper App. A.2] 'We train an ARIMA model with AutoARIMA on the train set and then use the selected
    hyperparameters to train the models for evaluation. We train a new ARIMA model each time we shift the context.'
    [Paper App. A.4] 'Quantiles are computed by fitting a normal distribution to the forecasts and using the
    inverse CDF (PPF) with the appropriate z-scores.'"""
    ctx, spec, L, guard = args
    from statsforecast.models import ARIMA
    z90 = norm.ppf(0.9)
    mu = sd = None
    ok = 0
    # [ASSUMPTION: not specified; statsforecast's default CSS-ML estimator occasionally fails on high-order fits to
    #  short contexts. We retry with pure ML and then CSS estimation (same orders); only if all fail do we fall back
    #  to a Gaussian random walk with the context's lag-1 innovation s.d. Counts per stage are in the run log.]
    for stage, method in enumerate(["CSS-ML", "ML", "CSS"], start=1):
        try:
            m = ARIMA(order=spec["order"], seasonal_order=spec["seasonal_order"], season_length=spec["season"],
                      include_mean=spec["include_mean"], include_drift=spec["include_drift"], method=method).fit(ctx)
            p = m.predict(h=L, level=[80])
            mu = np.asarray(p["mean"], float)
            sd = (np.asarray(p["hi-80"], float) - mu) / z90       # Kalman predictive s.d. recovered from 80% PI
            if np.all(np.isfinite(mu)) and np.all(np.isfinite(sd)):
                # [ASSUMPTION: post-hoc sensitivity ('--guard', tag suffix '-guarded'), added after the unguarded
                #  Glucose run produced 22/13,040 windows with explosive forecasts (|q| up to 1e11 on a 40-400 mg/dL
                #  series), which dominate window-averaged SIW/MASE. A fit is treated as failed if its 10%/90%
                #  quantiles leave [min(ctx) - 10*range(ctx), max(ctx) + 10*range(ctx)]; the next estimator in the
                #  chain is then tried. The paper does not describe any such guard; primary results stay unguarded.]
                if guard:
                    lo, hi, rng = ctx.min(), ctx.max(), np.ptp(ctx) + 1e-9
                    b = np.concatenate([mu - z90 * sd, mu + z90 * sd])
                    if b.min() < lo - 10 * rng or b.max() > hi + 10 * rng:
                        continue
                ok = stage
                break
        except Exception:
            continue
    if ok == 0:
        mu = np.full(L, ctx[-1])
        sd = np.std(np.diff(ctx)) * np.sqrt(np.arange(1, L + 1))
    q = mu[:, None] + sd[:, None] * norm.ppf(QUANTILES)[None, :]  # Gaussian quantiles via PPF
    return q.astype(np.float32), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--refit-context", default="64",
                    help="'64' (primary), 'T' (same context as TSFMs), or an integer W <= T (sensitivity sweep)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("ARIMA_PROCS", "4")),
                    help="worker processes (round 5: the stage runners pass cores-2)")
    ap.add_argument("--trial", type=int, default=0, help="timing trial: first N windows only, writes results/timing/")
    ap.add_argument("--chunk", type=int, default=20000, help="windows per resumable chunk")
    ap.add_argument("--guard", action="store_true", help="post-hoc explosive-forecast guard (sensitivity)")
    a = ap.parse_args()
    win = build_windows(a.dataset)
    meta = win["meta"]
    L, season, split = meta["L"], meta["season"], meta["split"]
    # [Paper App. A.2] 'We limit the max lag (context) for the ARIMA model to 64 for all datasets.'
    # [ASSUMPTION: App. A.4 instead says the model is 'refit on all earlier data before each forecast'. We read
    #  A.2 literally: each refit sees the last 64 observations (primary, model tag 'arima'); as a sensitivity run
    #  the refit sees the same T-length context as the TSFMs (model tag 'arima-ctxT').]
    if a.refit_context in ("64", "T"):
        W = 64 if a.refit_context == "64" else meta["T"]
        tag = ("arima" if a.refit_context == "64" else "arima-ctxT") + ("-guarded" if a.guard else "")
    else:
        # [ASSUMPTION: round-4 context-length sensitivity sweep (not in the paper, whose ARIMA context is stated
        #  as 64 in App. A.2 and 'all earlier data' in App. A.4). W in {32, 64, 128, 256, T}; W cannot exceed the
        #  TSFM context T because the evaluation windows store T points. Orders are still selected once on the
        #  train split; only the refit window changes. Tag 'arima-ctx{W}'; W=64 and W=T reuse 'arima'/'arima-ctxT'.]
        W = int(a.refit_context)
        assert 1 < W <= meta["T"], "refit context must be in (1, T]"
        tag = f"arima-ctx{W}" + ("-guarded" if a.guard else "")
    t0 = time.time()
    # resumable state (round 5): orders + per-chunk results under results/runlogs/partial_arima/<ds>__<tag>/
    pdir = f"results/runlogs/partial_arima/{a.dataset}__{tag}" + ("__trial" if a.trial else "")
    os.makedirs(pdir, exist_ok=True)
    # orders depend only on the dataset's training split, so they are cached once per dataset and shared by all
    # refit-context variants and the timing trial
    op = f"results/runlogs/partial_arima/{a.dataset}__orders.json"
    keys = list(win["series"])
    with Pool(a.procs) as pool:
        if os.path.exists(op):
            specs = json.load(open(op))
            for v in specs.values():
                v["order"], v["seasonal_order"] = tuple(v["order"]), tuple(v["seasonal_order"])
        else:  # train split only; one AutoARIMA per series, in parallel (same result as the serial loop)
            sel = pool.starmap(select_order, [(win["series"][k][:split], season) for k in keys], chunksize=1)
            specs = dict(zip(keys, sel))
            json.dump(specs, open(op + ".tmp", "w"))
            os.replace(op + ".tmp", op)
        t_sel = time.time() - t0
        n = a.trial if a.trial else len(win["context"])
        for c0 in range(0, n, a.chunk):
            cp = os.path.join(pdir, f"{c0:09d}.npz")
            if os.path.exists(cp) and not a.trial:
                continue
            jobs = [(win["context"][i][-W:], specs[str(win["series_id"][i])], L, a.guard)
                    for i in range(c0, min(c0 + a.chunk, n))]
            res = pool.map(forecast_one, jobs, chunksize=16)
            np.savez(cp + ".tmp.npz", q=np.stack([r[0] for r in res]), st=np.array([r[1] for r in res]))
            os.replace(cp + ".tmp.npz", cp)
            print(f"  chunk {c0}/{n} done, {time.time() - t0:.0f} s", flush=True)
    parts = [np.load(os.path.join(pdir, f"{c0:09d}.npz")) for c0 in range(0, n, a.chunk)]
    q = np.concatenate([z["q"] for z in parts])
    stages = np.concatenate([z["st"] for z in parts])
    if a.trial:
        secs = time.time() - t0
        os.makedirs("results/timing", exist_ok=True)
        info = dict(model=tag, dataset=a.dataset, trial_windows=n, total_windows=int(len(win["context"])),
                    procs=a.procs, order_selection_seconds=round(t_sel, 1), seconds=round(secs - t_sel, 2),
                    sec_per_window=(secs - t_sel) / max(n, 1))
        json.dump(info, open(f"results/timing/{a.dataset}__{tag}.json", "w"), indent=2)
        print(info)
        return
    jobs = range(len(q))
    n_fail = {"css_ml_ok": int((stages == 1).sum()), "ml_retry": int((stages == 2).sum()),
              "css_retry": int((stages == 3).sum()), "random_walk_fallback": int((stages == 0).sum())}
    path = save_forecast(tag, a.dataset, win, q)
    info = dict(model=tag, dataset=a.dataset, refit_context=W, n_windows=len(jobs), n_refit_failures=n_fail, guard=a.guard,
                orders={k: v for k, v in specs.items()}, seconds=round(time.time() - t0, 1),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1), path=path)
    os.makedirs("results/runlogs", exist_ok=True)
    with open(f"results/runlogs/{a.dataset}__{tag}.json", "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    for c0 in range(0, len(q), a.chunk):  # forecast saved; drop the chunks (the orders are also in the run log)
        os.remove(os.path.join(pdir, f"{c0:09d}.npz"))
    print({k: v for k, v in info.items() if k != "orders"})


if __name__ == "__main__":
    main()
