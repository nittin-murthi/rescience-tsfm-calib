"""SMOKE=1 package tests only (not used for any reported number): truncate processed datasets in place to
1 series and 3 forecast windows. The stage runners call this inside a throw-away copy (.smoke/), never on data/."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data import OUT, PROTOCOL  # noqa: E402

assert os.path.basename(os.path.dirname(os.path.dirname(OUT))) == ".smoke", "refusing to truncate real data"
for name in sys.argv[1:]:
    p = os.path.join(OUT, f"{name}.csv")
    gz = not os.path.exists(p)
    df = pd.read_csv(p + ".gz" if gz else p)
    meta = json.load(open(os.path.join(OUT, f"{name}.json")))
    first = sorted(df["series_id"].unique())[0]
    n = meta["split"] + PROTOCOL[name]["L"] + 2 * PROTOCOL[name]["stride"]
    df = df[(df["series_id"] == first) & (df["step"] < n)]
    df.to_csv(p + (".gz" if gz else ""), index=False)
    meta.update(n_series=1, steps_per_series=int(n), n_forecasts=3)
    json.dump(meta, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    print("smoke data:", name, len(df), "rows")
