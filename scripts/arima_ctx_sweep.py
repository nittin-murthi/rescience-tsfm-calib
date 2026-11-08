"""Round 4 (A): ARIMA refit-context sensitivity. Not in the paper.
[ASSUMPTION: sweep W in {32, 64, 128, 256, T}; T = 512 (Crime, Glucose) or 128 (Heart-Rate), the TSFM context of
 [Paper App. A.2, Table 2]. W = 64 is the primary 'arima' run [Paper App. A.2]; W = T is 'arima-ctxT'.]
Inputs: results/summary.csv (from src/evaluate.py). Outputs: results/arima_ctx_sweep.csv,
article/figures/arima_ctx.pdf, article/tables/arima_ctx.tex, article/tables/arima_ctx_numbers.tex.
C1a against ARIMA(W) [Paper §4.1]: holds iff every TSFM has lower dataset PCE than ARIMA(W)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from make_figures import TSFMS, PAPER_T4  # noqa: E402

T = {"crime": 512, "glucose": 512, "heartrate": 128}
DSN = {"crime": "Crime", "glucose": "Glucose", "heartrate": "Heart-Rate"}
MAC = {"crime": "Crime", "glucose": "Glucose", "heartrate": "HR"}


def fm(x):
    """MASE; values > 100 come from a few explosive ARIMA refits (cf. 'arima-guarded') and are shown as $>$100."""
    return f"{x:.3f}" if x < 100 else "$>$100$^\\ddagger$"


def tag(w, ds):
    return "arima" if w == 64 else ("arima-ctxT" if w == T[ds] else f"arima-ctx{w}")


def main():
    s = pd.read_csv(os.path.join(ROOT, "results", "summary.csv"))
    rows = []
    for ds in T:
        d = s[s.dataset == ds].set_index("model")
        t = d.loc[[m for m in TSFMS if m in d.index]]
        for w in [32, 64, 128, 256, 512]:
            if w > T[ds] or tag(w, ds) not in d.index:
                continue
            r = d.loc[tag(w, ds)]
            rows.append(dict(dataset=ds, context=w, model=tag(w, ds), pce=r.pce, pce_sem=r.pce_sem, cce=r.cce,
                             mase=r.mase, tsfm_pce_min=t.pce.min(), tsfm_pce_max=t.pce.max(),
                             n_tsfm_beaten_by_arima=int((t.pce >= r.pce).sum()), n_tsfm=len(t),
                             c1a_holds=bool((t.pce < r.pce).all())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "arima_ctx_sweep.csv"), index=False, float_format="%.6f")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(9, 2.8))
    for a, ds in zip(ax, T):
        x = df[df.dataset == ds]
        a.axhspan(x.tsfm_pce_min.iloc[0], x.tsfm_pce_max.iloc[0], color="tab:blue", alpha=0.2, label="TSFM range (ours)")
        a.errorbar(x.context, x.pce, yerr=x.pce_sem, marker="o", color="tab:red", label="ARIMA($W$) PCE $\\pm$ SEM")
        a.axhline(PAPER_T4[("PCE", ds)]["arima"], color="tab:red", ls="--", lw=0.8, label="paper ARIMA (Table 4)")
        a.set_xscale("log", base=2)
        a.set_xticks(list(x.context))
        a.set_xticklabels([str(c) for c in x.context])
        a.set_title(DSN[ds], fontsize=9)
        a.set_xlabel("ARIMA refit context $W$", fontsize=8)
        a.tick_params(labelsize=7)
    ax[0].set_ylabel("PCE", fontsize=8)
    ax[-1].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "article", "figures", "arima_ctx.pdf"))

    lines = ["\\begin{tabular}{llrrrrl}", "\\toprule",
             "Dataset & $W$ & PCE & CCE & MASE & TSFMs beaten & C1a \\\\", "\\midrule"]
    for ds in T:
        x = df[df.dataset == ds]
        for i, r in enumerate(x.itertuples()):
            w = f"{r.context}" + (" (primary)" if r.context == 64 else "") + (" ($=T$)" if r.context == T[ds] else "")
            lines.append(f"{DSN[ds] if i == 0 else ''} & {w} & {r.pce:.3f} & {r.cce:+.3f} & {fm(r.mase)} & "
                         f"{r.n_tsfm_beaten_by_arima}/{r.n_tsfm} & {'holds' if r.c1a_holds else 'fails'} \\\\")
        lines.append(f" & \\multicolumn{{6}}{{l}}{{\\scriptsize TSFM PCE range {x.tsfm_pce_min.iloc[0]:.3f}--"
                     f"{x.tsfm_pce_max.iloc[0]:.3f}; paper ARIMA {PAPER_T4[('PCE', ds)]['arima']:.3f}}} \\\\")
        lines.append("\\midrule" if ds != "heartrate" else "\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(ROOT, "article", "tables", "arima_ctx.tex"), "w").write("\n".join(lines) + "\n")
    mac = []
    for ds in T:
        x = df[df.dataset == ds]
        fails = x[~x.c1a_holds].context.tolist()
        mac.append(f"\\newcommand{{\\ctxFails{MAC[ds]}}}{{{', '.join(map(str, fails)) if fails else 'none'}}}")
        mac.append(f"\\newcommand{{\\ctxPceMin{MAC[ds]}}}{{{x.pce.min():.3f}}}")
        mac.append(f"\\newcommand{{\\ctxPceMax{MAC[ds]}}}{{{x.pce.max():.3f}}}")
    open(os.path.join(ROOT, "article", "tables", "arima_ctx_numbers.tex"), "w").write("\n".join(mac) + "\n")


if __name__ == "__main__":
    main()
