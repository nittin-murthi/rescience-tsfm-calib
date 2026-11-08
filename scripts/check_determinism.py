"""Round 2: compare a repeated TSFM forecast with the round-1 forecast (Heart-Rate).

Replicates Adler et al., arXiv:2510.16060v2. Writes results/determinism.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "results", "determinism", "*__repeat.npz"))):
    ds, model = os.path.basename(p).split("__")[:2]
    a = np.load(os.path.join(ROOT, "results", "forecasts", f"{ds}__{model}.npz"))
    b = np.load(p)["quantiles"].astype(np.float32)
    qa, y = a["quantiles"], a["target"]
    d = np.abs(qa - b)
    # [Paper §3.1, Eq. (1)] PCE of each run, pooled over the whole dataset (only for a difference check)
    rows.append(dict(dataset=ds, model=model, identical=bool(np.array_equal(qa, b)), max_abs_diff=float(d.max()),
                     max_rel_diff=float((d / (np.abs(qa) + 1e-6)).max()),
                     pce_round1=M.pce(y, qa, "pooled"), pce_repeat=M.pce(y, b, "pooled")))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "determinism.csv"), index=False)
print(df.to_string(index=False))
