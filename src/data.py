"""Dataset preparation for the partial replication.

Replicates: Adler et al., "Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated?",
arXiv:2510.16060v2 (27 May 2026). All paper references below are to v2.

Only numpy/pandas are used so this module can be imported from every model venv.
Output: data/processed/<name>.csv (long format: series_id, step, timestamp, value) and
        data/processed/<name>.json (protocol metadata used by every runner).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")

# [Paper App. A.2, Table 2] prediction length L, context size T, ARIMA seasonality per dataset.
# [Paper App. A.2] "For a stride d in {1, 4, 8} dependent on time series length" -> stride per dataset is
# [ASSUMPTION: not specified per dataset; chose d=4 for Crime (longest test span, 3287 days) and d=1 for
#  Glucose and Heart-Rate, because these choices give per-series forecast counts closest to Table 1
#  (# Total Forecasts / # Time Series): Crime 4110/5=822 vs ours 806; Glucose 13840/16=865 vs ours 815;
#  Heart-Rate 1800/6=300 vs ours 299.]
PROTOCOL = {
    "crime":     {"L": 64, "T": 512, "season": 7, "stride": 4},
    "glucose":   {"L": 64, "T": 512, "season": 1, "stride": 1},
    "heartrate": {"L": 64, "T": 128, "season": 1, "stride": 1},
    # Round 5: the paper's other three real datasets. [Paper App. A.2, Table 2] L, T, season.
    # [ASSUMPTION: stride (App. A.2: 'd in {1, 4, 8} dependent on time series length') chosen as for the first
    #  three datasets, by closeness to Table 1 (# Total Forecasts / # Time Series): Reviews d=8 -> 235 x 1,526 =
    #  358,610 full-target windows (Table 1: 360,490); M5 d=1 -> 70 x 304 = 21,280 (Table 1: 62,160, which no
    #  d in {1,4,8} reproduces under the Table 2 split; documented discrepancy); Patents d=1 -> 83 x 69 = 5,727
    #  (Table 1: 10,956 = 83 x 132 matches only if origins with partial targets are counted; we keep full
    #  L-step targets as for every other dataset).]
    "reviews":   {"L": 64, "T": 512, "season": 24, "stride": 8},
    "m5":        {"L": 64, "T": 512, "season": 7, "stride": 1},
    "patents":   {"L": 64, "T": 128, "season": 1, "stride": 1},
}
AUTH = os.path.join(ROOT, "data", "authors")
AUTHORS_COMMIT = "b60bfd9"  # https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration
# Pinned public sources (used by the stage runners only if a processed file is missing)
SOURCES = {
    "reviews": dict(url="https://archive.ics.uci.edu/static/public/1276/amazon+product+and+google+locations+reviews.zip",
                    file="y_amazon-google-large.csv",
                    sha256="dfc5e6169c25c5d838281973db722ae468a0d8822ab2edc656115ba08b9bae81"),
    "patents": dict(url=f"https://raw.githubusercontent.com/Coaster41/Beyond-Accuracy-TSFM-Calibration/{AUTHORS_COMMIT}"
                        "/data/patents/y_patents.csv", file="y_patents.csv",
                    sha256="760dfad2176a2ee45b827ed6b5b7503bbcab296c4585e91e2063998c45e0a42a"),
    "m5": dict(url="https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip", file="sales_train_validation.csv"),
}


def build_crime():
    """[Paper App. A.6] NYC Crime: 'aggregated ... by number of daily reports and cutoff the dataset to only
    include reports between the years 2006 and 2023. We split the dataset into time series based on borough.'
    [Paper Table 1] 5 series, daily, 6574 steps. [Paper Table 2] split 2015-01-01."""
    df = pd.read_csv(os.path.join(RAW, "crime", "nypd_daily_by_boro_rptdt.csv"))
    # [ASSUMPTION: 'daily reports' -> we count complaints by report date (RPT_DT), not occurrence date
    #  (CMPLNT_FR_DT), because the paper says 'number of daily reports' and RPT_DT is complete from 2006-01-01.]
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    df = df[df["boro_nm"].isin(["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"])]  # drop '(null)' borough
    df = df[(df["day"] >= "2006-01-01") & (df["day"] <= "2023-12-31")]  # [Paper App. A.6] years 2006-2023
    full = pd.date_range("2006-01-01", "2023-12-31", freq="D")
    rows = []
    for b, g in df.groupby("boro_nm"):
        s = g.set_index("day")["n"].astype(float).reindex(full).fillna(0.0)  # missing day -> 0 reports
        rows.append(pd.DataFrame({"series_id": b, "step": np.arange(len(full)), "timestamp": full, "value": s.values}))
    out = pd.concat(rows, ignore_index=True)
    split_step = int((pd.Timestamp("2015-01-01") - full[0]).days)  # [Paper Table 2] train/test split 2015-01-01
    return out, split_step, "D"


def build_glucose():
    """[Paper App. A.6] Glucose: Dexcom G6 interstitial glucose (mg/dL) every 5 minutes, 16 subjects, ~10 days.
    [Paper Table 1] 16 series, 5-min, 1,686 steps/series. [Paper Table 2] split 2020-02-16 12:43."""
    rows = []
    # [ASSUMPTION: subjects start on different dates (2020-02-13 .. 2020-07-19), so one absolute split timestamp
    #  cannot apply to all. We interpret the split relative to each subject's start: 2020-02-16 12:43 minus the
    #  earliest recording start (subject 001, 2020-02-13 17:23:32) = 67.3 h = step 808 on a 5-min grid.]
    split_step = 808
    # [ASSUMPTION: Table 1 reports 1,686 steps per series; we truncate every subject to its first 1,686 grid
    #  steps so all series have the paper's length.]
    n_keep = 1686
    for sid in range(1, 17):
        f = os.path.join(RAW, "glucose", f"Dexcom_{sid:03d}.csv")
        d = pd.read_csv(f)
        d = d[d["Event Type"] == "EGV"]  # estimated glucose values only (drop alerts/calibrations)
        ts = pd.to_datetime(d["Timestamp (YYYY-MM-DDThh:mm:ss)"])
        v = pd.to_numeric(d["Glucose Value (mg/dL)"], errors="coerce").values
        s = pd.Series(v, index=ts).sort_index()
        s = s[~s.index.duplicated()]
        # [ASSUMPTION: not specified; readings are snapped to a regular 5-min grid anchored at the first reading
        #  and gaps (sensor dropouts, up to ~22 h in subject 015) are linearly interpolated, because all TSFMs
        #  and ARIMA assume regular sampling.]
        grid = pd.date_range(s.index[0], s.index[-1], freq="5min")
        s = s.reindex(s.index.union(grid)).interpolate("time").reindex(grid)
        s = s.iloc[:n_keep]
        assert len(s) == n_keep, (sid, len(s))
        rows.append(pd.DataFrame({"series_id": f"{sid:03d}", "step": np.arange(n_keep),
                                  "timestamp": s.index, "value": s.values.astype(float)}))
    return pd.concat(rows, ignore_index=True), split_step, "5min"


def build_heartrate():
    """[Paper App. A.6] Meditative Heart-Rate (Peng et al. 1999): 14 volunteers, 10-minute metronomic breathing;
    mapped to start at 2000-01-01 00:00:00. [Paper Table 1] 6 series, 'Second' granularity, 744 steps/series.
    [Paper Table 2] split 2000-01-01 00:04:59, T=128."""
    lens = {}
    data = {}
    for i in range(1, 15):
        arr = np.loadtxt(os.path.join(RAW, "heartrate", f"M{i}.txt"))  # columns: beat time (s), instantaneous HR (bpm)
        data[f"M{i}"] = arr[:, 1]
        lens[f"M{i}"] = len(arr)
    # [ASSUMPTION: the paper uses 6 of the 14 'metron' subjects without naming them, and 744 steps (= length of
    #  M1 in beats). We treat each beat as one time step (beat k -> second k, matching 'Second' granularity and
    #  744 = len(M1)), and keep the 6 longest records (M8 770, M1 744, M12 700, M11 665, M2 663, M10 662).]
    keep = sorted(lens, key=lambda k: -lens[k])[:6]
    # [ASSUMPTION: truncate the 6 records to the shortest kept length (662) so each yields
    #  662-300-64+1 = 299 forecasts (Table 1: 1,800/6 = 300).]
    n_keep = min(lens[k] for k in keep)
    split_step = 300  # [Paper Table 2] split 00:04:59 -> first 300 one-second steps (00:00:00..00:04:59) are train
    rows = []
    t0 = pd.Timestamp("2000-01-01 00:00:00")  # [Paper App. A.6] mapped to start at 2000-01-01 00:00:00
    for k in sorted(keep, key=lambda s: int(s[1:])):
        v = data[k][:n_keep]
        rows.append(pd.DataFrame({"series_id": k, "step": np.arange(n_keep),
                                  "timestamp": t0 + pd.to_timedelta(np.arange(n_keep), unit="s"), "value": v}))
    return pd.concat(rows, ignore_index=True), split_step, "s"


def _long_from_authors(df, split_ts, freq):
    """Authors' long format (ds, y, unique_id) -> ours (series_id, step, timestamp, value); split = # ds < split_ts
    ([Paper App. A.2] 'dates before train/test split column' are training data)."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    rows, splits = [], set()
    for k, g in df.groupby("unique_id", sort=True):
        g = g.sort_values("ds")
        rows.append(pd.DataFrame({"series_id": f"{int(k):03d}", "step": np.arange(len(g)), "timestamp": g["ds"].values,
                                  "value": g["y"].astype(float).values}))
        splits.add(int((g["ds"] < pd.Timestamp(split_ts)).sum()))
    assert len(splits) == 1, splits  # one common calendar split for all series
    return pd.concat(rows, ignore_index=True), splits.pop(), freq


def build_reviews():
    """[Paper App. A.6] Reviews: hourly counts of Amazon (Hou et al. 2024) and Google Places (Li et al. 2022)
    reviews, aggregated by product/location category, '239 of the most abundant categories', trimmed to
    2020-01-01 08:00 .. 2021-06-25 23:00. [Paper Table 1] 239 series x 13,000 hourly steps. [Paper Table 2] split
    2020-01-31 23:00, T=512, season 24.
    [FROM AUTHORS' DATA: y_amazon-google-large.csv, the authors' UCI deposit (doi:10.24432/C5SP7K), referenced by
     data/README.md at commit b60bfd9; sha256 in SOURCES. Rebuilding from the raw review dumps is infeasible and
     the category sampling is under-specified.]
    [ASSUMPTION: the deposit has 235 series, not 239 (Table 1); we use all 235 as released.]"""
    df = pd.read_csv(os.path.join(AUTH, "reviews", "y_amazon-google-large.csv"), index_col=0)
    return _long_from_authors(df, "2020-01-31 23:00", "h")


def build_m5():
    """[Paper App. A.6] Shopping (M5): 'binning each product by their product department and store ID, totaling
    70 time series'. [Paper Table 1] 70 x 1,912 daily steps. [Paper Table 2] split 2015-04-23, T=512, season 7.
    Built by us from the public M5 file sales_train_validation.csv (d_1 = 2011-01-29), summing unit sales over
    items per (dept_id, store_id).
    [ASSUMPTION: M5 has 1,913 training days; Table 1 says 1,912 steps, so we keep d_1..d_1912 (to 2016-04-23).]
    Cross-check (round 5, diagnosis only): the result equals the authors' data/m5/y_m5.csv (commit b60bfd9)
    exactly for all 70 series."""
    s = pd.read_csv(os.path.join(RAW, "m5", "sales_train_validation.csv"))
    days = [f"d_{i}" for i in range(1, 1913)]
    g = s.groupby(["dept_id", "store_id"])[days].sum()
    dates = pd.Timestamp("2011-01-29") + pd.to_timedelta(np.arange(len(days)), unit="D")
    rows = [pd.DataFrame({"series_id": f"{d}_{st}", "step": np.arange(len(days)), "timestamp": dates,
                          "value": g.loc[(d, st)].to_numpy(float)}) for d, st in g.index]
    split = int((dates < pd.Timestamp("2015-04-23")).sum())  # [Paper Table 2]
    return pd.concat(rows, ignore_index=True), split, "D"


def build_patents():
    """[Paper App. A.6] Patents (Marco et al. 2015): US patents filed per month 1981-2014, sparse series removed,
    'aggregated by patent field and category', series with minimum < 100 removed. [Paper Table 1] 83 x 408 monthly
    steps. [Paper Table 2] split 2004-01-01, T=128, season 1.
    [FROM AUTHORS' DATA: data/patents/y_patents.csv, commit b60bfd9 (sha256 in SOURCES). The field/category
     aggregation does not identify the 83 series, so an independent rebuild from USPTO HPDF would be a
     substitution.]"""
    df = pd.read_csv(os.path.join(AUTH, "patents", "y_patents.csv"), index_col=0)
    return _long_from_authors(df, "2004-01-01", "MS")


BUILDERS = {"crime": build_crime, "glucose": build_glucose, "heartrate": build_heartrate,
            "reviews": build_reviews, "m5": build_m5, "patents": build_patents}
ORIGINAL = ["crime", "glucose", "heartrate"]


def load(name):
    """Load a processed dataset: returns (dict series_id -> np.ndarray, meta dict)."""
    p = os.path.join(OUT, f"{name}.csv")
    df = pd.read_csv(p if os.path.exists(p) else p + ".gz")  # Reviews ships gzip-compressed
    with open(os.path.join(OUT, f"{name}.json")) as f:
        meta = json.load(f)
    series = {str(k): g.sort_values("step")["value"].to_numpy(float) for k, g in df.groupby("series_id", sort=True)}
    return series, meta


def origins(n, split, L, stride):
    """[Paper App. A.2] 'Models forecast the first 64 time steps of the test set using the end of the training
    set as the context. The forecast date and non-AR context are then shifted by the stride ... until the end of
    the time series.' Returns forecast-origin indices o (first predicted index) with o+L <= n."""
    return list(range(split, n - L + 1, stride))


def main(names=None):
    """python src/data.py                 -> the original three datasets
       python src/data.py m5 patents      -> only the named datasets
       SMOKE=1 python src/data.py ...     -> tiny test version (1 series, 3 windows), for runner smoke tests only"""
    os.makedirs(OUT, exist_ok=True)
    names = names or (sys.argv[1:] or ORIGINAL)
    for name in names:
        df, split, freq = BUILDERS[name]()
        if os.environ.get("SMOKE") == "1":  # not used for any reported number
            first = sorted(df["series_id"].unique())[0]
            n = split + PROTOCOL[name]["L"] + 2 * PROTOCOL[name]["stride"]
            df = df[(df["series_id"] == first) & (df["step"] < n)]
        p = PROTOCOL[name]
        gz = name == "reviews"
        df.to_csv(os.path.join(OUT, f"{name}.csv" + (".gz" if gz else "")), index=False)
        n_series = df["series_id"].nunique()
        lens = df.groupby("series_id").size()
        n_fc = int(sum(len(origins(int(n), split, p["L"], p["stride"])) for n in lens))
        meta = dict(name=name, split=split, freq=freq, n_series=int(n_series),
                    steps_per_series=int(lens.max()), n_forecasts=n_fc, **p)
        assert split >= p["T"] or name == "heartrate" and split >= p["T"], "context must fit in train part"
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(meta)


if __name__ == "__main__":
    main()
