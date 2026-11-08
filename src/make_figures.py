"""Figures and per-claim verdict table from results/summary.csv.

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026). Section numbers refer to v2.
Outputs: article/figures/pce_cce.pdf, results/verdict.csv, results/verdict.json, results/paper_table4_subset.csv
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
TSFMS = ["chronos-bolt-small", "chronos-bolt-base", "timesfm-2.5-200m", "tirex"]
BASES = ["arima", "arima-ctxT", "arima-guarded", "nbeats-full-seed1", "nbeats-holdout-full-seed1", "nbeats", "nbeats-holdout"]
FIG_BASES = ["arima", "arima-ctxT", "arima-guarded", "nbeats-full-seed1", "nbeats-holdout-full-seed1"]  # round 3: full grid in figure
# round 4: cross-platform sensitivity for the Mac-returned stage-1 ARIMA (same code/data/orders, re-run on x86 box)
SENS_BASES = ["arima-box", "arima-ctxT-box"]
# [Paper Table 1 / Table 4] dataset order of the paper; all six datasets of our scope (PLAN.md)
DATASETS = ["reviews", "m5", "crime", "heartrate", "glucose", "patents"]
DSLABEL = {"reviews": "Reviews", "m5": "Shopping (M5)", "crime": "Crime", "heartrate": "Heart-Rate",
           "glucose": "Glucose", "patents": "Patents"}
# [Paper §4.1] 'all TSFMs have PCE values close to or below 5% across all datasets, except for the relatively
# difficult Patents dataset' -> the paper does not claim C1b on Patents; we still report the number.
C1B_NOT_CLAIMED = {"patents"}
# [Paper §4.1] 'PCE values below 0.05 ... much better calibrated'; 'close to or below 5%'.
# [ASSUMPTION: 'close to' operationalised a priori (PLAN.md) as PCE <= 0.06.]
CLOSE_TO_5 = 0.06

# [Paper App. B.1, Table 4] values quoted verbatim for the six datasets we test.
# Column order in the paper: Chronos-Bolt, Moirai 2.0, TimesFM, TiRex, YingLong, N-BEATS, ARIMA.
PAPER_T4 = {
    ("PCE", "reviews"): dict(chronos_bolt=0.015, moirai2=0.013, timesfm=0.012, tirex=0.012, yinglong=0.014, nbeats=0.061, arima=0.176),
    ("PCE", "m5"): dict(chronos_bolt=0.033, moirai2=0.045, timesfm=0.036, tirex=0.031, yinglong=0.052, nbeats=0.108, arima=0.131),
    ("PCE", "patents"): dict(chronos_bolt=0.176, moirai2=0.154, timesfm=0.206, tirex=0.192, yinglong=0.259, nbeats=0.242, arima=0.247),
    ("CCE", "reviews"): dict(chronos_bolt=-0.012, moirai2=0.007, timesfm=0.004, tirex=-0.013, yinglong=-0.024, nbeats=-0.040, arima=-0.351),
    ("CCE", "m5"): dict(chronos_bolt=0.002, moirai2=0.000, timesfm=0.005, tirex=-0.026, yinglong=0.051, nbeats=-0.203, arima=-0.259),
    ("CCE", "patents"): dict(chronos_bolt=0.028, moirai2=0.046, timesfm=0.089, tirex=0.069, yinglong=0.201, nbeats=0.322, arima=0.031),
    ("MASE", "reviews"): dict(chronos_bolt=0.717, moirai2=0.748, timesfm=0.720, tirex=0.710, yinglong=0.823, nbeats=1.416, arima=1.211),
    ("MASE", "m5"): dict(chronos_bolt=0.785, moirai2=0.727, timesfm=0.753, tirex=0.743, yinglong=0.835, nbeats=0.791, arima=0.854),
    ("MASE", "patents"): dict(chronos_bolt=11.896, moirai2=10.639, timesfm=12.193, tirex=12.510, yinglong=21.297, nbeats=22.814, arima=14.534),
    ("PCE", "crime"): dict(chronos_bolt=0.023, moirai2=0.035, timesfm=0.023, tirex=0.021, yinglong=0.014, nbeats=0.103, arima=0.168),
    ("PCE", "glucose"): dict(chronos_bolt=0.036, moirai2=0.033, timesfm=0.050, tirex=0.031, yinglong=0.042, nbeats=0.092, arima=0.141),
    ("PCE", "heartrate"): dict(chronos_bolt=0.063, moirai2=0.051, timesfm=0.052, tirex=0.048, yinglong=0.077, nbeats=0.121, arima=0.204),
    ("CCE", "crime"): dict(chronos_bolt=-0.030, moirai2=-0.020, timesfm=-0.001, tirex=-0.041, yinglong=0.015, nbeats=-0.207, arima=-0.337),
    ("CCE", "glucose"): dict(chronos_bolt=-0.011, moirai2=-0.050, timesfm=-0.038, tirex=-0.027, yinglong=0.034, nbeats=-0.172, arima=-0.280),
    ("CCE", "heartrate"): dict(chronos_bolt=-0.061, moirai2=-0.008, timesfm=0.001, tirex=-0.037, yinglong=0.105, nbeats=-0.243, arima=-0.411),
    ("MASE", "crime"): dict(chronos_bolt=0.741, moirai2=0.759, timesfm=0.759, tirex=0.740, yinglong=0.761, nbeats=0.767, arima=0.829),
    ("MASE", "glucose"): dict(chronos_bolt=6.362, moirai2=5.986, timesfm=6.051, tirex=5.841, yinglong=6.041, nbeats=7.326, arima=5.958),
    ("MASE", "heartrate"): dict(chronos_bolt=1.119, moirai2=1.118, timesfm=1.104, tirex=1.096, yinglong=1.257, nbeats=1.335, arima=1.120),
}


def verdicts(s, per):
    rows = []
    for ds in DATASETS:
        d = s[s.dataset == ds].set_index("model")
        t = [m for m in TSFMS if m in d.index]
        # round 2: also every N-BEATS seed / full-grid run present (e.g. nbeats-seed0, nbeats-holdout-full-seed2)
        extra = sorted(m for m in d.index if (m.startswith("nbeats") and m not in BASES) or m in SENS_BASES)
        for base in [b for b in BASES if b in d.index] + extra:
            # C1a: every TSFM has lower dataset PCE than the baseline  [Paper §4.1, Fig. 3 top]
            c1a = bool(all(d.loc[m, "pce"] < d.loc[base, "pce"] for m in t))
            # per-series win count (how many (TSFM, series) pairs beat the baseline on the same series)
            p = per[per.dataset == ds].pivot(index="series_id", columns="model", values="pce")
            wins = int(sum((p[m] < p[base]).sum() for m in t))
            pairs = int(len(p) * len(t))
            rows.append(dict(dataset=ds, baseline=base, claim="C1a_tsfm_pce_below_baseline", holds=c1a, paper_claims=True,
                             max_tsfm_pce=float(max(d.loc[m, "pce"] for m in t)), baseline_pce=float(d.loc[base, "pce"]),
                             series_pair_wins=wins, series_pairs=pairs))
        c1b = bool(all(d.loc[m, "pce"] <= CLOSE_TO_5 for m in t))
        rows.append(dict(dataset=ds, baseline="", claim="C1b_tsfm_pce_close_to_or_below_0.05", holds=c1b,
                         paper_claims=ds not in C1B_NOT_CLAIMED,
                         max_tsfm_pce=float(max(d.loc[m, "pce"] for m in t)), baseline_pce=np.nan,
                         series_pair_wins=np.nan, series_pairs=np.nan))
    return pd.DataFrame(rows)


PAPER_KEY = {"chronos-bolt-base": "chronos_bolt", "timesfm-2.5-200m": "timesfm", "tirex": "tirex",
             "arima": "arima", "nbeats-full-seed1": "nbeats"}


def figure(s):
    os.makedirs(os.path.join(ROOT, "article", "figures"), exist_ok=True)
    models = [m for m in TSFMS + FIG_BASES if m in set(s.model)]
    colors = dict(zip(models, plt.cm.tab10(np.arange(len(models)))))
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    w = 0.8 / len(models)
    for j, (metric, lab) in enumerate([("pce", "PCE (lower is better)"), ("cce", "CCE (+ over / − under-conf.)")]):
        for i, m in enumerate(models):
            vals = [s[(s.dataset == ds) & (s.model == m)][metric].values for ds in DATASETS]
            errs = [s[(s.dataset == ds) & (s.model == m)][metric + "_sem"].values for ds in DATASETS]
            x = np.arange(len(DATASETS)) + (i - (len(models) - 1) / 2) * w
            ax[j].bar(x, [v[0] if len(v) else np.nan for v in vals], w, yerr=[e[0] if len(e) else 0 for e in errs],
                      color=colors[m], label=m, capsize=1.5)
            # [Paper App. B.1, Table 4] paper value for the matching model, drawn as a black marker on our bar
            pk = PAPER_KEY.get(m)
            if pk:
                pv = [PAPER_T4[(metric.upper(), ds)].get(pk, np.nan) for ds in DATASETS]
                ax[j].scatter(x, pv, marker="x", color="k", s=14, zorder=5,
                              label="paper (Table 4)" if (i == 0 or m == "chronos-bolt-base") and j == 0 else None)
        ax[j].set_ylabel(lab, fontsize=8)
        ax[j].axhline(0.05 if metric == "pce" else 0.0, color="k", lw=0.6, ls="--")
    ax[1].set_xticks(np.arange(len(DATASETS)))
    ax[1].set_xticklabels([DSLABEL[d] for d in DATASETS])
    ax[0].legend(fontsize=6.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "article", "figures", "pce_cce.pdf"))


def main():
    s = pd.read_csv(os.path.join(RES, "summary.csv"))
    per = pd.read_csv(os.path.join(RES, "metrics_per_series.csv"))
    per = per.astype({"series_id": str})
    v = verdicts(s, per)
    v.to_csv(os.path.join(RES, "verdict.csv"), index=False, float_format="%.4f")
    pt = [dict(metric=k[0], dataset=k[1], **vals) for k, vals in PAPER_T4.items()]
    pd.DataFrame(pt).to_csv(os.path.join(RES, "paper_table4_subset.csv"), index=False)
    figure(s)
    print(v.to_string(index=False))


if __name__ == "__main__":
    main()
