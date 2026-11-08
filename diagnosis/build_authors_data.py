"""DIAGNOSIS ONLY (step e): convert the authors' released Crime and Heart-Rate CSVs into our processed format so the
same pipeline can be run on their exact data. Not used for any primary result.

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026).
Usage: venvs/stats/bin/python diagnosis/build_authors_data.py /path/to/authors/repo
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import OUT, PROTOCOL, origins  # noqa: E402

COMMIT = "b60bfd92ff836525773c71039a83ba2fe3d123bf"


def convert(repo, rel, name, base, split_ts, freq):
    # [FROM AUTHORS' CODE: data/police/y_police.csv, data/meditation/y_meditation.csv, commit b60bfd9] data only
    df = pd.read_csv(os.path.join(repo, rel))
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])
    df["step"] = df.groupby("unique_id").cumcount()
    first = df[df.unique_id == df.unique_id.min()]
    split = int((first["ds"] < pd.Timestamp(split_ts)).sum())  # index of first test step
    out = pd.DataFrame({"series_id": df["unique_id"].astype(str), "step": df["step"],
                        "timestamp": df["ds"], "value": df["y"].astype(float)})
    out.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    p = PROTOCOL[base]
    lens = out.groupby("series_id").size()
    meta = dict(name=name, split=split, freq=freq, n_series=int(len(lens)), steps_per_series=int(lens.max()),
                n_forecasts=int(sum(len(origins(int(n), split, p["L"], p["stride"])) for n in lens)),
                source=f"authors' repo {rel} @ {COMMIT} (diagnosis only)", **p)
    json.dump(meta, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    print(meta)


if __name__ == "__main__":
    repo = sys.argv[1]
    # same split timestamps as our builders ([Paper Table 2]); first test step = first ds >= split
    convert(repo, "data/police/y_police.csv", "crime_authorsdata", "crime", "2015-01-01", "D")
    convert(repo, "data/meditation/y_meditation.csv", "heartrate_authorsdata", "heartrate", "2000-01-01 00:05:00", "s")
