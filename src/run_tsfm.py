"""Zero-shot TSFM forecasting (quantile heads read directly).

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026). Section numbers refer to v2.
Usage (inside the model's venv):  python src/run_tsfm.py --model chronos-bolt-small --dataset crime
"""
import argparse
import json
import os
import resource
import sys
import time

NTHREADS = int(os.environ.get("TSFM_THREADS", "4"))  # shared box: 4 (PLAN.md §8); the stage runners raise it
os.environ.setdefault("OMP_NUM_THREADS", str(NTHREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NTHREADS))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple silicon: run unsupported ops on the CPU
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from forecast_io import build_windows, save_forecast  # noqa: E402
from metrics import QUANTILES  # noqa: E402

torch.set_num_threads(NTHREADS)

HF_IDS = {
    "chronos-bolt-small": "amazon/chronos-bolt-small",          # instructed small variant
    "chronos-bolt-base": "amazon/chronos-bolt-base",            # [Paper App. A.4, fn. 4] checkpoint used in the paper
    "timesfm-2.5-200m": "google/timesfm-2.5-200m-pytorch",      # instructed substitute for [App. A.4] timesfm-2.0-500m
    "tirex": "NX-AI/TiRex",                                     # [Paper App. A.4, fn. 5] same checkpoint as the paper
}
# Pinned checkpoint revisions (envs/MODEL_REVISIONS.md; refs/main when first downloaded, 2026-10-23). Round 5: passed
# to every loader so a fresh machine downloads exactly the weights used on the box.
REVISIONS = {
    "chronos-bolt-small": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    "chronos-bolt-base": "5d9f166d69f47aef3401367a7b842e78fe97b121",
    "timesfm-2.5-200m": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    "tirex": "63c740922493f5fbe60b277609ec62babfba2762",
}


def pick_device(req):
    """auto -> cuda, else mps (Apple silicon), else cpu. Device choice does not change the method; outputs differ
    from the CPU at float level only, which check_device() verifies on the first windows."""
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_np(q):
    return q.detach().float().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)


def make_predictor(model, L, T, device="cpu"):
    """Return f(contexts (B,T) float array) -> quantiles (B, L, 9) for q = 0.1..0.9.
    [Paper §3.2] 'All selected TSFMs ... use a quantile prediction head'; [Paper §2.1] quantiles are 'produced
    directly by a multi-quantile prediction head'. We read q in Q = {0.1..0.9} (Eq. (1)) straight from each head."""
    if model.startswith("chronos"):
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(HF_IDS[model], device_map=device, torch_dtype=torch.float32,
                                                   revision=REVISIONS[model])

        def f(x):
            q, _ = pipe.predict_quantiles(torch.tensor(x, dtype=torch.float32), prediction_length=L,
                                          quantile_levels=QUANTILES.tolist())
            return to_np(q)
        return f
    if model == "tirex":
        from tirex import load_model
        # [ASSUMPTION: CPU-only box -> TiRex 'torch' backend instead of the default CUDA kernels; numerically the
        #  same architecture/weights.] [Paper App. A.4] sets accelerated roll-out to 4 patches of size 32 (=128 steps);
        #  [ASSUMPTION: with L=64 (2 output patches) we expect this setting to have little or no
        #  effect, so we keep the library default (not verified against the authors' setting).]
        m = load_model(HF_IDS[model], device=device, backend="torch", hf_kwargs={"revision": REVISIONS[model]})

        def f(x):
            q, _ = m.forecast(context=torch.tensor(x, dtype=torch.float32), prediction_length=L)
            q = to_np(q)
            assert q.shape[-1] == 9  # TiRex head = quantiles 0.1..0.9
            return q
        return f
    if model == "timesfm-2.5-200m":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(HF_IDS[model], revision=REVISIONS[model])
        if device != "cpu" and str(m.model.device) != device:  # the library only auto-selects CUDA or CPU
            m.model.device = torch.device(device)
            m.model.to(m.model.device)
        # [ASSUMPTION: the paper used TimesFM 2.0 (no config given). For 2.5 we use the model-card recommended
        #  inference flags; max_context = T (Table 2), max_horizon = L. use_continuous_quantile_head=True and
        #  fix_quantile_crossing=True are the card defaults.]
        m.compile(timesfm.ForecastConfig(max_context=max(T, 32), max_horizon=L, normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=True, fix_quantile_crossing=True,
                                         per_core_batch_size=64))

        def f(x):
            _, qt = m.forecast(horizon=L, inputs=[r for r in x])
            return np.asarray(qt)[:, :L, 1:10]  # index 0 = mean, 1..9 = quantiles 0.1..0.9
        return f
    raise ValueError(model)


def check_device(model, L, T, device, X, tol=1e-2):
    """Build the predictor on `device`; if it is not the CPU, compare the first windows with a CPU run.
    Falls back to the CPU (and says why) if the device raises or the outputs differ by more than `tol`, relative to
    the mean absolute CPU quantile. Returns (predictor, device_used, note)."""
    if device == "cpu":
        return make_predictor(model, L, T, "cpu"), "cpu", "cpu requested"
    probe = X[: min(16, len(X))]
    try:
        f = make_predictor(model, L, T, device)
        with torch.inference_mode():
            qd = np.sort(np.asarray(f(probe), np.float32), axis=-1)
        fc = make_predictor(model, L, T, "cpu")
        with torch.inference_mode():
            qc = np.sort(np.asarray(fc(probe), np.float32), axis=-1)
        del fc
        err = float(np.max(np.abs(qd - qc)) / (np.mean(np.abs(qc)) + 1e-8))
        if not np.isfinite(qd).all() or err > tol:
            return make_predictor(model, L, T, "cpu"), "cpu", f"fallback: {device} differs from cpu (rel err {err:.2e})"
        return f, device, f"{device} ok (rel err vs cpu {err:.2e} on {len(probe)} windows)"
    except Exception as e:  # e.g. an op missing on MPS
        return make_predictor(model, L, T, "cpu"), "cpu", f"fallback: {device} failed ({type(e).__name__}: {e})"[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF_IDS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--trial", type=int, default=0, help="timing trial: forecast only the first N windows, write "
                                                         "results/timing/<ds>__<model>.json, keep no forecast")
    ap.add_argument("--chunk", type=int, default=8192, help="windows per resumable chunk")
    # Determinism check (round 2): write a second forecast to results/determinism/ instead of results/forecasts/,
    # so evaluate.py does not pick it up; scripts/check_determinism.py compares the two files.
    ap.add_argument("--repeat", action="store_true")
    a = ap.parse_args()
    win = build_windows(a.dataset)
    L, T = win["meta"]["L"], win["meta"]["T"]
    X = win["context"]
    if a.trial:
        X = X[: a.trial]
    t_load = time.time()
    f, dev, note = check_device(a.model, L, T, pick_device(a.device), X)
    print(f"model={a.model} dataset={a.dataset} windows={len(X)} device={dev} [{note}]", flush=True)
    t_load = time.time() - t_load
    # resumable chunks (round 5): results/runlogs/partial_tsfm/<ds>__<model>/<start>.npy
    pdir = f"results/runlogs/partial_tsfm/{a.dataset}__{a.model}" + ("__trial" if a.trial else "")
    os.makedirs(pdir, exist_ok=True)
    t0 = time.time()
    n_new = 0
    for c0 in range(0, len(X), a.chunk):
        cp = os.path.join(pdir, f"{c0:09d}.npy")
        if os.path.exists(cp) and not a.repeat and not a.trial:
            continue
        out = []
        with torch.inference_mode():
            end = min(c0 + a.chunk, len(X))
            for i in range(c0, end, a.batch):
                out.append(np.asarray(f(X[i:min(i + a.batch, end)]), np.float32))
        np.save(cp + ".tmp.npy", np.concatenate(out))
        os.replace(cp + ".tmp.npy", cp)
        n_new += min(c0 + a.chunk, len(X)) - c0
        print(f"  chunk {c0}/{len(X)} done, {time.time() - t0:.0f} s", flush=True)
    secs = time.time() - t0
    q = np.concatenate([np.load(os.path.join(pdir, f"{c0:09d}.npy")) for c0 in range(0, len(X), a.chunk)])
    assert q.shape == (len(X), L, 9), q.shape
    if a.trial:
        os.makedirs("results/timing", exist_ok=True)
        info = dict(model=a.model, dataset=a.dataset, trial_windows=len(X), total_windows=int(len(win["context"])),
                    device=dev, device_note=note, seconds=round(secs, 2), load_seconds=round(t_load, 1),
                    sec_per_window=secs / max(len(X), 1))
        json.dump(info, open(f"results/timing/{a.dataset}__{a.model}.json", "w"), indent=2)
        for c0 in range(0, len(X), a.chunk):
            os.remove(os.path.join(pdir, f"{c0:09d}.npy"))
        print(info)
        return
    # [ASSUMPTION: heads can return crossing quantiles; we sort along the quantile axis (monotone rearrangement)
    #  so every interval in Eq. (2)/(3) is well defined. Fraction of affected values is logged.]
    crossing = float(np.mean(np.diff(q, axis=-1) < 0))
    q = np.sort(q, axis=-1)
    if a.repeat:
        os.makedirs("results/determinism", exist_ok=True)
        path = f"results/determinism/{a.dataset}__{a.model}__repeat.npz"
        np.savez_compressed(path, quantiles=q)
    else:
        path = save_forecast(a.model, a.dataset, win, q)
    info = dict(model=a.model, hf_id=HF_IDS[a.model], dataset=a.dataset, n_windows=int(len(X)),
                seconds=round(secs, 1), windows_this_session=n_new, crossing_frac=crossing, device=dev,
                device_note=note, threads=NTHREADS, batch=a.batch,
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1),
                path=path)
    logp = (f"results/determinism/{a.dataset}__{a.model}__repeat.json" if a.repeat
            else f"results/runlogs/{a.dataset}__{a.model}.json")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    with open(logp, "w") as fh:
        json.dump(info, fh, indent=2)
    if not a.repeat:  # forecast saved; drop the chunks
        for c0 in range(0, len(X), a.chunk):
            os.remove(os.path.join(pdir, f"{c0:09d}.npy"))
    print(info)


if __name__ == "__main__":
    main()
