"""N-BEATS baseline (neuralforecast): the paper's grid search (full) or our reduced 8-config grid.

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026). Section numbers refer to v2.
Usage (nbeats venv):
  python src/run_nbeats.py --dataset crime                         # reduced grid, in-sample selection, seed 1
  python src/run_nbeats.py --dataset crime --grid full --seed 0 --select holdout --device auto
Resumable: each scored grid configuration is appended to results/runlogs/partial/<dataset>__<tag>.jsonl and is
skipped on restart; the selected configuration is refitted (same seed) if it was scored in an earlier session.
"""
import argparse
import itertools
import json
import logging
import os
import resource
import sys
import time
import warnings

NTHREADS = int(os.environ.get("NBEATS_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(NTHREADS))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple-silicon: fall back to CPU for missing ops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from data import origins  # noqa: E402
from forecast_io import build_windows, save_forecast  # noqa: E402
from metrics import QUANTILES, wql  # noqa: E402

torch.set_num_threads(NTHREADS)
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

# [Paper App. A.2, Table 3] grid: Epochs {100,1000}, LR {1e-3,1e-4}, early-stop patience {-1,2},
# blocks {(1,1,1),(3,3,3)}, validation check steps {10,100}, loss {Normal, Quantile}.
# [ASSUMPTION: reduced grid for CPU budget: 'epochs' -> neuralforecast max_steps {100,1000}, LR {1e-3,1e-4},
#  blocks {(1,1,1),(3,3,3)}; patience=-1 (no early stopping, so validation-check steps are irrelevant);
#  loss = Quantile only, because the paper reports 'Quantile loss outperforms normal distribution loss on all
#  datasets' (App. A.2). 8 configurations instead of 64.]
GRID = list(itertools.product([100, 1000], [1e-3, 1e-4], [(1, 1, 1), (3, 3, 3)]))
REDUCED = [dict(steps=st, lr=lr, blocks=b, patience=-1, val_check=100, loss="quantile") for st, lr, b in GRID]
# [Paper App. A.2, Table 3] full grid, 2^6 = 64 configurations, all six hyperparameters as listed.
# [ASSUMPTION: 'Epochs' -> neuralforecast max_steps (as in the reduced grid). Early stopping needs a validation
#  split, which the paper does not describe: when patience = 2 we hold out the last L steps of each fitting series
#  (neuralforecast val_size = L) and stop on validation loss; when patience = -1, val_size = 0 and val_check_steps
#  has no effect on training (the two such configs differ only by their random stream). 'Normal Distribution Loss'
#  -> DistributionLoss('Normal') with the paper's 'default 100 samples' (App. A.2) and our 9 quantiles.]
FULL = [dict(steps=st, lr=lr, patience=pa, blocks=b, val_check=vc, loss=lo)
        for st, lr, pa, b, vc, lo in itertools.product([100, 1000], [1e-3, 1e-4], [-1, 2], [(1, 1, 1), (3, 3, 3)],
                                                       [10, 100], ["normal", "quantile"])]


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_long(arrs, start=0):
    rows = []
    for i, a in enumerate(arrs):
        rows.append(pd.DataFrame({"unique_id": str(i), "ds": np.arange(start, start + len(a)), "y": a}))
    return pd.concat(rows, ignore_index=True)


def quantile_cols(fc):
    cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    # MQLoss output names: <model>-lo-X, <model>-median, <model>-hi-X ; map to q = 0.1..0.9
    def qof(c):
        if c.endswith("-median"):
            return 0.5
        lvl = float(c.split("-")[-1]) / 100.0
        return 0.5 - lvl / 2 if "-lo-" in c else 0.5 + lvl / 2
    qs = {round(qof(c), 2): c for c in cols if ("-lo-" in c or "-hi-" in c or c.endswith("-median"))}
    return [qs[round(q, 2)] for q in QUANTILES]


def fit(train_series, T, L, cfg, seed=1, device="cpu"):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
    from neuralforecast.models import NBEATS
    if cfg["loss"] == "quantile":
        loss = MQLoss(quantiles=QUANTILES.tolist())
    else:  # [Paper App. A.2] Normal distribution loss, 'default 100 samples'
        loss = DistributionLoss(distribution="Normal", quantiles=QUANTILES.tolist(), num_samples=100)
    val_size = L if cfg["patience"] > 0 else 0
    # [ASSUMPTION: not specified; per-window standard scaling ('scaler_type=standard') so one global model can be
    #  trained across series of different scale; batch size, MLP width and stacks are neuralforecast defaults.]
    m = NBEATS(h=L, input_size=T, loss=loss, max_steps=cfg["steps"], learning_rate=cfg["lr"],
               n_blocks=list(cfg["blocks"]), scaler_type="standard", random_seed=seed,
               early_stop_patience_steps=cfg["patience"], val_check_steps=cfg["val_check"],
               enable_progress_bar=False, enable_model_summary=False, logger=False, accelerator=device, devices=1,
               enable_checkpointing=False)
    nf = NeuralForecast(models=[m], freq=1)
    nf.fit(df=to_long(train_series), val_size=val_size)  # [Paper App. A.2] one model per dataset, across series
    return nf


def predict(nf, contexts, L, batch=2048):
    out = []
    for i in range(0, len(contexts), batch):
        c = contexts[i:i + batch]
        fc = nf.predict(df=to_long(c))
        cols = quantile_cols(fc)
        fc["k"] = fc["unique_id"].astype(int)
        fc = fc.sort_values(["k", "ds"])
        out.append(fc[cols].to_numpy(float).reshape(len(c), L, 9))
    return np.sort(np.concatenate(out), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--select", default="insample", choices=["insample", "holdout"])
    ap.add_argument("--grid", default="reduced", choices=["reduced", "full", "smoke"],
                    help="smoke = config 0 of the full grid only, for SMOKE=1 package tests (tag ...-smoke)")
    ap.add_argument("--seed", type=int, default=1)  # round 1 used neuralforecast's default random_seed=1
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--config-ids", default="", help="timing trial: comma-separated grid ids to score, then stop")
    a = ap.parse_args()
    device = pick_device(a.device)
    # output tag: round-1 names are kept for the reduced grid with seed 1 (nbeats, nbeats-holdout)
    tag = "nbeats" + ("-holdout" if a.select == "holdout" else "") + {"full": "-full", "smoke": "-smoke"}.get(a.grid, "")
    if a.grid != "reduced" or a.seed != 1:
        tag += f"-seed{a.seed}"
    grid = {"full": FULL, "smoke": FULL[:1], "reduced": REDUCED}[a.grid]
    os.makedirs("results/runlogs/partial", exist_ok=True)
    part = f"results/runlogs/partial/{a.dataset}__{tag}.jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            done[r["config_id"]] = r
    print(f"tag={tag} device={device} configs={len(grid)} resumed={len(done)}", flush=True)
    win = build_windows(a.dataset)
    meta = win["meta"]
    T, L, split = meta["T"], meta["L"], meta["split"]
    train = [y[:split] for _, y in sorted(win["series"].items())]  # [Paper App. A.2] dates before the split only
    # [Paper App. A.2] 'select the model with the lowest WQL on the same data' (training data).
    # [ASSUMPTION: 'same data' read two ways. insample (primary, literal): configs are fitted on the whole training
    #  split and scored on rolling windows inside it (stride 8). holdout (sensitivity): configs are fitted on the
    #  training split minus its last L steps and scored on the one window per series whose targets are that tail; the selected
    #  config is then refitted on the whole training split. The paper's grid contains early-stopping/validation
    #  parameters, which suggests some held-out validation, but it is not described.]
    fit_series = train if a.select == "insample" else [y[:-L] for y in train]
    sel_ctx, sel_tgt = [], []
    for y in train:
        first = T if a.select == "insample" else len(y) - L  # holdout: the single window whose targets are unseen
        for o in origins(len(y), max(first, T), L, 8):
            sel_ctx.append(y[o - T:o])
            sel_tgt.append(y[o:o + L])
    sel_ctx, sel_tgt = np.asarray(sel_ctx), np.asarray(sel_tgt)
    t0 = time.time()
    best_nf = {}  # config_id -> fitted model, only for the current best of this session
    for cid, cfg in enumerate(grid):
        if cid in done:
            continue
        if a.config_ids and cid not in {int(x) for x in a.config_ids.split(",")}:
            continue
        tc = time.time()
        err = None
        try:
            nf = fit(fit_series, T, L, cfg, a.seed, device)
            s = float(wql(sel_tgt, predict(nf, sel_ctx, L)))
        except Exception as e:
            # [ASSUMPTION: round 5. A configuration that cannot be trained on a dataset (e.g. too short for an
            #  early-stopping validation split on Patents with the hold-out reading) is scored +inf, so it cannot
            #  be selected. The error is logged in the record; the paper does not discuss this case.]
            nf, s, err = None, float("inf"), f"{type(e).__name__}: {e}"[:300]
        rec = dict(config_id=cid, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                   train_wql=s, seconds=round(time.time() - tc, 1), device=device, error=err)
        done[cid] = rec
        with open(part, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        cur = min(done.values(), key=lambda r: r["train_wql"])
        best_nf = {cid: nf} if cur["config_id"] == cid else best_nf
    if len(done) < len(grid):
        print(f"stopped after {len(done)}/{len(grid)} configs (timing trial); rerun without --config-ids to resume")
        return
    best = min(done.values(), key=lambda r: r["train_wql"])
    bcfg = grid[best["config_id"]]
    if a.select == "insample" and best["config_id"] in best_nf:
        final = best_nf[best["config_id"]]
    else:  # holdout: refit on the whole training split; insample: refit after a resumed session (same seed)
        final = fit(train, T, L, bcfg, a.seed, device)
    q = predict(final, win["context"], L)  # [Paper App. A.2] reuse the same parameters at every shift
    path = save_forecast(tag, a.dataset, win, q)
    scores = [done[k] for k in sorted(done)]
    info = dict(model=tag, select=a.select, grid_name=a.grid, seed=a.seed, device=device, dataset=a.dataset,
                grid=scores, selected=best, n_windows=int(len(q)),
                seconds=round(time.time() - t0, 1), grid_seconds_total=round(sum(r["seconds"] for r in scores), 1),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1), path=path)
    os.makedirs("results/runlogs", exist_ok=True)
    with open(f"results/runlogs/{a.dataset}__{tag}.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print({k: v for k, v in info.items() if k != "grid"})


if __name__ == "__main__":
    main()
