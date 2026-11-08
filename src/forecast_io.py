"""Shared window construction and forecast storage.

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026). Section numbers refer to v2.
"""
import os

import numpy as np

from data import ROOT, load, origins

FC_DIR = os.path.join(ROOT, "results", "forecasts")


def build_windows(name):
    """[Paper §2.1] context y_{1:T}, target y_{T+1:T+L}; [Paper App. A.2] rolling origins from the split with the
    dataset's stride. Returns dict with contexts (W,T), targets (W,L), series ids, origins, and meta."""
    series, meta = load(name)
    T, L, stride, split = meta["T"], meta["L"], meta["stride"], meta["split"]
    ctx, tgt, sid, org = [], [], [], []
    for k in sorted(series):
        y = series[k]
        for o in origins(len(y), split, L, stride):
            ctx.append(y[o - T:o])        # [Paper Table 2] fixed context size T (512 or 128)
            tgt.append(y[o:o + L])        # [Paper App. A.2] L = 64
            sid.append(k)
            org.append(o)
    return dict(context=np.asarray(ctx, float), target=np.asarray(tgt, float),
                series_id=np.asarray(sid), origin=np.asarray(org), meta=meta, series=series)


def save_forecast(model, name, win, q, extra=None):
    os.makedirs(FC_DIR, exist_ok=True)
    path = os.path.join(FC_DIR, f"{name}__{model}.npz")
    # Optional half-precision storage (round 5, for the large Reviews runs): FC_FLOAT16=1 stores the quantiles as
    # float16 (relative rounding <= 4.9e-4), and evaluation casts back to float. [ASSUMPTION: storage precision only.
    # It is used only when it is lossless for PCE/CCE: every coverage indicator 1{y <= q} must be identical after
    # rounding, and all values must be inside the float16 range. Otherwise this file stays float32. MASE/WQL/SIW change
    # by <= 5e-4 relative. Default off.]
    q = np.asarray(q, np.float32)
    qdt = np.float32
    if os.environ.get("FC_FLOAT16", "0") == "1" and np.nanmax(np.abs(q)) < 6.0e4:
        y = win["target"].astype(np.float32)[..., None]
        if np.array_equal(y <= q, y <= q.astype(np.float16).astype(np.float32)):
            qdt = np.float16
    np.savez_compressed(path, quantiles=q.astype(qdt), target=win["target"].astype(np.float32),
                        series_id=win["series_id"], origin=win["origin"], **(extra or {}))
    return path


def load_forecast(model, name):
    return dict(np.load(os.path.join(FC_DIR, f"{name}__{model}.npz"), allow_pickle=False))
