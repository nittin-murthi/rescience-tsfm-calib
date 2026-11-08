"""Round 2: N-BEATS seed variability (mean ± SD across seeds) and per-seed C1a verdicts.

Replicates Adler et al., arXiv:2510.16060v2. Reads results/summary.csv, writes results/seed_summary.csv and
results/seed_verdict.csv. TSFMs and ARIMA are deterministic (results/determinism.csv; ARIMA has no random
component), so only N-BEATS variants have a seed dimension.
"""
import os
import re

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
s = pd.read_csv(os.path.join(ROOT, "results", "summary.csv"))
s = s[~s.dataset.str.endswith("_authorsdata")]
nb = s[s.model.str.startswith("nbeats")].copy()


def split(m):
    g = re.match(r"^(nbeats(?:-holdout)?(?:-full)?)(?:-seed(\d+))?$", m)
    base, seed = g.group(1), g.group(2)
    # round-1 reduced-grid runs (nbeats, nbeats-holdout) used neuralforecast's default random_seed = 1
    return base, int(seed) if seed is not None else 1


nb[["variant", "seed"]] = nb.model.apply(lambda m: pd.Series(split(m)))
cols = ["pce", "cce", "siw", "mase", "wql"]
g = nb.groupby(["dataset", "variant"])
out = pd.concat([g[cols].mean(), g[cols].std(ddof=1).add_suffix("_sd"), g["seed"].apply(lambda x: ",".join(map(str, sorted(x)))).rename("seeds"),
                 g["seed"].count().rename("n_seeds")], axis=1).reset_index()
out.to_csv(os.path.join(ROOT, "results", "seed_summary.csv"), index=False, float_format="%.6f")
# [Paper §4, claim C1a] per seed: is the worst TSFM's PCE below this N-BEATS run's PCE?
tsfm = s[s.family == "TSFM"].groupby("dataset").pce.max().rename("max_tsfm_pce")
v = nb.merge(tsfm, on="dataset")
v["C1a_holds"] = v.max_tsfm_pce < v.pce
v[["dataset", "variant", "seed", "model", "pce", "cce", "max_tsfm_pce", "C1a_holds"]].sort_values(
    ["dataset", "variant", "seed"]).to_csv(os.path.join(ROOT, "results", "seed_verdict.csv"), index=False, float_format="%.6f")
print(out.round(4).to_string(index=False))
print(v.groupby(["dataset", "variant"]).C1a_holds.agg(["sum", "count"]))
