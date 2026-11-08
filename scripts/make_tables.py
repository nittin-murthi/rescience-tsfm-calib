"""Generate LaTeX tables for the article directly from results/*.csv (no hand-typed numbers).

Replicates: Adler et al., arXiv:2510.16060v2 (27 May 2026).
Usage: venvs/stats/bin/python scripts/make_tables.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from make_figures import PAPER_T4  # noqa: E402  [Paper App. B.1, Table 4]

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "article", "tables")
# [Paper Table 1 / Table 4] dataset order of the paper
DSN = {"reviews": "Reviews", "m5": "M5", "crime": "Crime", "heartrate": "Heart-Rate", "glucose": "Glucose",
       "patents": "Patents"}
# main table split in two so each fits a page: datasets run on the Mac (stage 1/2) and on the CPU box
MAIN_TABLES = {"main.tex": ["crime", "heartrate", "glucose"], "main_new.tex": ["reviews", "m5", "patents"]}
ORDER = ["chronos-bolt-small", "chronos-bolt-base", "timesfm-2.5-200m", "tirex",
         "arima", "arima-guarded", "arima-ctxT", "nbeats-full-seed1", "nbeats-holdout-full-seed1", "nbeats", "nbeats-holdout"]
NAME = {"chronos-bolt-small": "Chronos-Bolt-S", "chronos-bolt-base": "Chronos-Bolt-B",
        "timesfm-2.5-200m": "TimesFM-2.5$^\\dagger$", "tirex": "TiRex", "arima": "ARIMA (64)",
        "arima-guarded": "ARIMA (64, guard)$^\\ast$", "arima-ctxT": "ARIMA ($T$)$^\\ast$",
        "nbeats-full-seed1": "N-BEATS (full, in-sample)", "nbeats-holdout-full-seed1": "N-BEATS (full, hold-out)$^\\ast$",
        "nbeats": "N-BEATS (reduced, in-s.)$^\\ast$", "nbeats-holdout": "N-BEATS (reduced, hold-out)$^\\ast$"}
PAPERKEY = {"chronos-bolt-base": "chronos_bolt", "timesfm-2.5-200m": "timesfm", "tirex": "tirex",
            "arima": "arima", "arima-guarded": "arima", "arima-ctxT": "arima",
            "nbeats": "nbeats", "nbeats-holdout": "nbeats", "nbeats-full-seed1": "nbeats", "nbeats-holdout-full-seed1": "nbeats"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if abs(x) >= 1000:
        return f"{x:.1e}".replace("e+0", "e").replace("e+", "e")
    return f"{x:.{d}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    s = pd.read_csv(os.path.join(RES, "summary.csv"))
    for fname, dsl in MAIN_TABLES.items():
        main_table(s, dsl, fname)
    rest(s)


def main_table(s, dsl, fname):
    rows = []
    for ds in dsl:
        sub = s[s.dataset == ds].set_index("model")
        first = True
        for m in ORDER:
            if m not in sub.index:
                continue
            r = sub.loc[m]
            pk = PAPERKEY.get(m)
            pp = PAPER_T4[("PCE", ds)].get(pk) if pk else None
            pc = PAPER_T4[("CCE", ds)].get(pk) if pk else None
            pm = PAPER_T4[("MASE", ds)].get(pk) if pk else None
            lab = f"\\multirow{{{sum(x in sub.index for x in ORDER)}}}{{*}}{{{DSN[ds]}}}" if first else ""
            first = False
            rows.append(f"{lab} & {NAME[m]} & {f(r.pce)}$\\pm${f(r.pce_sem)} & {f(r.pce_per_window)} & "
                        f"{f(r.cce)} & {f(r.siw, 2)} & {f(r.mase, 2)} & {f(pp)} & {f(pc)} & {f(pm, 2)} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    body = "\n".join(rows)
    tex = ("\\begin{tabular}{llcccccccc}\n\\toprule\n"
           " & & \\multicolumn{5}{c}{This replication} & \\multicolumn{3}{c}{Paper, Table 4} \\\\\n"
           "\\cmidrule(lr){3-7}\\cmidrule(lr){8-10}\n"
           "Dataset & Model & PCE$\\pm$SEM & PCE$_{\\mathrm{win}}$ & CCE & SIW & MASE & PCE & CCE & MASE \\\\\n"
           "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, fname), "w").write(tex)


def rest(s):
    v = pd.read_csv(os.path.join(RES, "verdict.csv"))
    # per-seed N-BEATS rows go to tables/seeds.tex; the verdict table keeps the round-1 (seed 1) baselines
    v = v[v.baseline.isna() | v.baseline.isin(list(NAME))]
    vr = []
    for _, r in v.iterrows():
        if r.claim.startswith("C1a"):
            vr.append(f"{DSN[r.dataset]} & C1a vs {NAME[r.baseline]} & {f(r.max_tsfm_pce)} & {f(r.baseline_pce)} & "
                      f"{int(r.series_pair_wins)}/{int(r.series_pairs)} & {'yes' if r.holds else 'no'} \\\\")
        else:
            # [Paper §4.1] C1b is not claimed for Patents ('except for the relatively difficult Patents dataset')
            hv = ('yes' if r.holds else 'no') + ('' if r.paper_claims else ' (not claimed)')
            vr.append(f"{DSN[r.dataset]} & C1b (max TSFM PCE $\\le$ 0.06) & {f(r.max_tsfm_pce)} & -- & -- & "
                      f"{hv} \\\\")
    tex = ("\\begin{tabular}{llcccc}\n\\toprule\nDataset & Test & max TSFM PCE & baseline PCE & "
           "series-pair wins & holds \\\\\n\\midrule\n" + "\n".join(vr) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, "verdict.tex"), "w").write(tex)
    # diagnosis table: our data vs authors' data, our aggregation vs authors' aggregation (DIAGNOSIS.md)
    dpath = os.path.join(RES, "diagnosis_authors_style_metrics.csv")
    if os.path.exists(dpath):
        a = pd.read_csv(dpath).merge(s[["dataset", "model", "pce", "mase"]], on=["dataset", "model"])
        dr = []
        for ds in ["crime", "crime_authorsdata", "heartrate", "heartrate_authorsdata"]:
            for m in ["chronos-bolt-small", "timesfm-2.5-200m", "arima", "arima-ctxT"]:
                r = a[(a.dataset == ds) & (a.model == m)]
                if len(r) == 0:
                    continue
                r = r.iloc[0]
                base = ds.replace("_authorsdata", "")
                src = "authors'" if "authors" in ds else "ours"
                pk = PAPERKEY.get(m)
                pp = PAPER_T4[("PCE", base)].get(pk) if pk else None
                pm = PAPER_T4[("MASE", base)].get(pk) if pk else None
                dr.append(f"{DSN[base]} & {src} & {NAME[m]} & {f(r.pce)} & {f(r.pce_authors_style)} & "
                          f"{f(r.mase, 2)} & {f(r.mase_authors_style, 2)} & {f(pp)} & {f(pm, 2)} \\\\")
        tex = ("\\begin{tabular}{lllcccccc}\n\\toprule\n & & & \\multicolumn{2}{c}{PCE} & \\multicolumn{2}{c}{MASE} & "
               "\\multicolumn{2}{c}{Paper T4} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
               "Dataset & data & Model & ours & authors' agg. & ours & authors' agg. & PCE & MASE \\\\\n\\midrule\n"
               + "\n".join(dr) + "\n\\bottomrule\n\\end{tabular}\n")
        open(os.path.join(OUT, "diag.tex"), "w").write(tex)

    # key numbers as LaTeX macros so prose numbers also come from the CSVs
    mac = []
    tag = {"crime": "Crime", "glucose": "Glucose", "heartrate": "HR", "reviews": "Rev", "m5": "Mfive", "patents": "Pat"}
    ts = ["chronos-bolt-small", "chronos-bolt-base", "timesfm-2.5-200m", "tirex"]
    for ds, t in tag.items():
        sub = s[s.dataset == ds].set_index("model")
        have = [m for m in ts if m in sub.index]
        mac.append(f"\\newcommand{{\\maxTSFMpce{t}}}{{{f(sub.loc[have, 'pce'].max())}}}")
        mac.append(f"\\newcommand{{\\minTSFMpce{t}}}{{{f(sub.loc[have, 'pce'].min())}}}")
        mac.append(f"\\newcommand{{\\nTSFM{t}}}{{{len(have)}}}")
        for m, k in [("chronos-bolt-small", "Cs"), ("chronos-bolt-base", "Cb"), ("timesfm-2.5-200m", "Tfm"),
                     ("tirex", "Trx"), ("arima", "Arima"), ("arima-ctxT", "ArimaT"), ("arima-guarded", "ArimaG"),
                     ("nbeats", "Nbeats"), ("nbeats-holdout", "NbeatsH"),
                     ("nbeats-full-seed1", "NbeatsFone"), ("nbeats-holdout-full-seed1", "NbeatsFHone")]:
            for col, c in [("pce", "pce"), ("cce", "cce"), ("mase", "mase")]:
                val = f(sub.loc[m, col], 3) if m in sub.index else "n/a"
                mac.append(f"\\newcommand{{\\{c}{k}{t}}}{{{val}}}")
        for col in ["pce_per_window"]:
            mac.append(f"\\newcommand{{\\pcewinTSFMmin{t}}}{{{f(sub.loc[have, col].min())}}}")
        mac.append(f"\\newcommand{{\\nwin{t}}}{{{int(sub['n_windows'].iloc[0]):,}}}".replace(",", "{,}"))
        mac.append(f"\\newcommand{{\\nser{t}}}{{{int(sub['n_series'].iloc[0])}}}")
    # round 2: N-BEATS seed variability (results/seed_summary.csv, results/seed_verdict.csv)
    sp = os.path.join(RES, "seed_summary.csv")
    if os.path.exists(sp):
        ss, sv = pd.read_csv(sp), pd.read_csv(os.path.join(RES, "seed_verdict.csv"))
        vname = {"nbeats": "reduced (8), in-sample", "nbeats-holdout": "reduced (8), hold-out",
                 "nbeats-full": "full (64), in-sample", "nbeats-holdout-full": "full (64), hold-out"}
        dname = DSN
        rows = []
        for ds in DSN:
            for vv in vname:
                r = ss[(ss.dataset == ds) & (ss.variant == vv)]
                if not len(r):
                    continue
                r = r.iloc[0]
                k = sv[(sv.dataset == ds) & (sv.variant == vv)]
                sd = lambda c: f"{r[c]:.3f}$\\pm${r[c + '_sd']:.3f}" if r["n_seeds"] > 1 else f"{r[c]:.3f}"
                rows.append(f"{dname[ds]} & {vname[vv]} & {r['seeds']} & {sd('pce')} & {sd('cce')} & {sd('mase')} & "
                            f"{f(k.max_tsfm_pce.iloc[0])} & {int(k.C1a_holds.sum())}/{len(k)} \\\\")
        tex = ("\\begin{tabular}{llcccccc}\n\\toprule\nDataset & N-BEATS grid, selection & seeds & PCE & CCE & MASE & "
               "worst TSFM PCE & C1a holds \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        open(os.path.join(OUT, "seeds.tex"), "w").write(tex)
        for ds, t in tag.items():
            for vv, k in [("nbeats", "Nbeats"), ("nbeats-holdout", "NbeatsH"), ("nbeats-full", "NbeatsF"),
                          ("nbeats-holdout-full", "NbeatsFH")]:
                r = ss[(ss.dataset == ds) & (ss.variant == vv)]
                for c in ["pce", "cce"]:
                    mac.append(f"\\newcommand{{\\{c}{k}{t}SD}}{{{f(r[c + '_sd'].iloc[0]) if len(r) and r['n_seeds'].iloc[0] > 1 else 'n/a'}}}")
                    mac.append(f"\\newcommand{{\\{c}{k}{t}Mean}}{{{f(r[c].iloc[0]) if len(r) else 'n/a'}}}")
        ins = ss[ss.variant == "nbeats"]
        mac.append(f"\\newcommand{{\\maxSDinsample}}{{{f(ins.pce_sd.max())}}}")
        hc = sv[(sv.dataset == "crime") & (sv.variant == "nbeats-holdout")]
        mac.append(f"\\newcommand{{\\minHoldoutCrime}}{{{f(hc.pce.min())}}}")
        mac.append(f"\\newcommand{{\\maxHoldoutCrime}}{{{f(hc.pce.max())}}}")
        mac.append(f"\\newcommand{{\\seedCOneAfails}}{{{int((~sv.C1a_holds).sum())}}}")
        for vv, k in [("nbeats", "Red"), ("nbeats-holdout", "RedH"), ("nbeats-full", "Full"), ("nbeats-holdout-full", "FullH")]:
            x = sv[sv.variant == vv]
            mac.append(f"\\newcommand{{\\cOneAholds{k}}}{{{int(x.C1a_holds.sum())}}}")
            mac.append(f"\\newcommand{{\\cOneAruns{k}}}{{{len(x)}}}")
            mac.append(f"\\newcommand{{\\cceNeg{k}}}{{{int((x.cce < 0).sum())}}}")
        # selection WQL of the configuration chosen by in-sample selection, over datasets and seeds
        import glob
        import json
        sel = []
        for pth in glob.glob(os.path.join(RES, "runlogs", "*__nbeats.json")) + glob.glob(os.path.join(RES, "runlogs", "*__nbeats-seed*.json")):
            sel.append(min(g["train_wql"] for g in json.load(open(pth))["grid"]))
        mac.append(f"\\newcommand{{\\selWQLmin}}{{{f(min(sel))}}}")
        mac.append(f"\\newcommand{{\\selWQLmax}}{{{f(max(sel))}}}")
        mac.append(f"\\newcommand{{\\seedCOneAholds}}{{{int(sv.C1a_holds.sum())}}}")
        mac.append(f"\\newcommand{{\\seedCOneAruns}}{{{len(sv)}}}")
        old = sv[sv.dataset.isin(["crime", "glucose", "heartrate"])]
        for vv, k in [("nbeats-full", "Full"), ("nbeats-holdout-full", "FullH")]:
            x = old[old.variant == vv]
            mac.append(f"\\newcommand{{\\cOneAholds{k}Old}}{{{int(x.C1a_holds.sum())}}}")
            mac.append(f"\\newcommand{{\\cOneAruns{k}Old}}{{{len(x)}}}")
            mac.append(f"\\newcommand{{\\cceNeg{k}Old}}{{{int((x.cce < 0).sum())}}}")
    # quantile-crossing fraction re-sorted (TSFM run logs, all datasets)
    import glob
    import json
    cr = []
    for pth in glob.glob(os.path.join(RES, "runlogs", "*.json")):
        d = json.load(open(pth))
        if isinstance(d, dict) and "crossing_frac" in d:
            cr.append((d["crossing_frac"], d["model"], d["dataset"]))
    c = max(cr)
    mant, ex = f"{c[0]:.1e}".split("e")
    mac.append(f"\\newcommand{{\\maxCrossFrac}}{{{mant}\\times10^{{{int(ex)}}}}}")
    mac.append(f"\\newcommand{{\\maxCrossWhere}}{{{NAME[c[1]]} on {DSN[c[2]]}}}")
    mac += validity()
    open(os.path.join(OUT, "numbers.tex"), "w").write("\n".join(mac) + "\n")
    print("wrote", os.listdir(OUT))


def validity():
    """Round 4: table of the checks on the Mac-returned stage-1 forecasts (scripts/verify_returned_forecasts.py)."""
    import glob
    import re
    vd = os.path.join(RES, "stage1_validity")
    if not os.path.isdir(vd):
        return []
    b = pd.read_csv(os.path.join(vd, "base.csv"))
    t = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(vd, "tsfm_*.csv")))], ignore_index=True)
    ac = pd.read_csv(os.path.join(vd, "arima_crossplatform.csv"))
    rows, mac = [], []
    for ds in ["patents", "m5"]:
        bd = b[b.dataset == ds]
        common = (f"{int(bd[bd.check == 'W_windows_equal'].ok.sum())}/{int((bd.check == 'W_windows_equal').sum())}")
        for m in ["chronos-bolt-small", "chronos-bolt-base", "timesfm-2.5-200m", "tirex"]:
            for _, r in t[(t.dataset == ds) & (t.model == m)].iterrows():
                k = re.search(r"_K(\d+)$", r.check).group(1)
                agr = re.search(r"agreement ([0-9.]+)", r.detail).group(1)
                pr = re.search(r"returned ([0-9.]+) vs box ([0-9.]+)", r.detail)
                rows.append(f"{DSN[ds]} & {NAME[m]} & re-forecast {k} windows (CPU) & {float(r.value):.1e} & {agr} & "
                            f"{pr.group(1)} / {pr.group(2)} \\\\")
        for m in ["arima", "arima-ctxT"]:
            sp = bd[(bd.model == m) & (bd.check == "A_reforecast_spot")].iloc[0]
            a = ac[(ac.dataset == ds) & (ac.model == m)].iloc[0]
            rows.append(f"{DSN[ds]} & {NAME[m]} & full re-run on x86 box & {float(sp.value):.1e} (32 win.)$^\\S$ & -- & "
                        f"{a.pce_mac:.4f} / {a.pce_box:.4f} \\\\")
        nb = bd[bd.model.astype(str).str.startswith("nbeats")]
        g = nb[nb.check == "N_grid_complete"]
        sel = nb[nb.check == "N_selected_is_argmin"]
        rows.append(f"{DSN[ds]} & N-BEATS (6 runs) & grid 64/64, selection = argmin WQL & "
                    f"{int(g.ok.sum())}/{len(g)}, {int(sel.ok.sum())}/{len(sel)} & -- & -- \\\\")
        rows.append("\\midrule")
        ordr = bd[bd.check == "A_orders_equal_box_reselection"].value.iloc[0]
        tag = {"patents": "Pat", "m5": "Mfive"}[ds]
        mac.append(f"\\newcommand{{\\arimaOrdersEq{tag}}}{{{ordr}}}")
        mac.append(f"\\newcommand{{\\winEq{tag}}}{{{common}}}")
        for _, a in ac[ac.dataset == ds].iterrows():
            k = "ArimaT" if a.model == "arima-ctxT" else "Arima"
            mac.append(f"\\newcommand{{\\pceBoxDiff{k}{tag}}}{{{a.pce_absdiff:.4f}}}")
    ev = b[b.check == "E_reevaluate_equals_returned_summary"].value.iloc[0]
    mac.append(f"\\newcommand{{\\reevalMaxDiff}}{{{ev}}}")
    tex = ("\\begin{tabular}{lllccc}\n\\toprule\nDataset & Model & check & max rel.\\ err. / result & "
           "coverage agr. & PCE Mac / box \\\\\n\\midrule\n" + "\n".join(rows[:-1]) +
           "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, "validity.tex"), "w").write(tex)
    return mac


if __name__ == "__main__":
    main()
