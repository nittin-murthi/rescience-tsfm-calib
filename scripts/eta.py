"""Estimated remaining runtime for a stage, from the timing trial (results/timing/*.json and the N-BEATS trial
configs in results/runlogs/partial/). Usage: python scripts/eta.py patents m5   -> prints a table + total."""
import glob
import json
import os
import sys

TSFMS = ["chronos-bolt-small", "chronos-bolt-base", "timesfm-2.5-200m", "tirex"]
ARIMAS = ["arima", "arima-ctxT"]
NB_TAGS = [f"nbeats{s}-full-seed{k}" for s in ("", "-holdout") for k in (0, 1, 2)]


def fc_done(ds, model):
    return os.path.exists(f"results/forecasts/{ds}__{model}.npz")


rows, total = [], 0.0
for ds in sys.argv[1:]:
    for m in TSFMS + ARIMAS:
        p = f"results/timing/{ds}__{m}.json"
        if fc_done(ds, m):
            rows.append((ds, m, "done", 0.0))
            continue
        if not os.path.exists(p):
            rows.append((ds, m, "no trial", float("nan")))
            continue
        t = json.load(open(p))
        sec = t["sec_per_window"] * t["total_windows"] + t.get("order_selection_seconds", 0) + t.get("load_seconds", 0)
        rows.append((ds, m, t.get("device", f"{t.get('procs')} procs"), sec))
    # N-BEATS: 32 configs with 100 steps + 32 with 1000 steps per run (Table 3 grid); trial = configs 1 (100) and
    # 32, 63 (1000) of seed-0 in-sample. Remaining configs of partially finished runs are counted.
    tr = {}
    for line in open(f"results/runlogs/partial/{ds}__nbeats-full-seed0.jsonl") if os.path.exists(
            f"results/runlogs/partial/{ds}__nbeats-full-seed0.jsonl") else []:
        r = json.loads(line)
        tr.setdefault(r["steps"], []).append(r["seconds"])
    if 100 in tr and 1000 in tr:
        t100, t1000 = sum(tr[100]) / len(tr[100]), sum(tr[1000]) / len(tr[1000])
        for tag in NB_TAGS:
            if fc_done(ds, tag):
                rows.append((ds, tag, "done", 0.0))
                continue
            pp = f"results/runlogs/partial/{ds}__{tag}.jsonl"
            have = [json.loads(x)["steps"] for x in open(pp)] if os.path.exists(pp) else []
            sec = (32 - have.count(100)) * t100 + (32 - have.count(1000)) * t1000 + 60  # + final fit/predict
            rows.append((ds, tag, "", sec))
    else:
        rows.append((ds, "nbeats (6 runs)", "no trial", float("nan")))
for ds, m, d, s in rows:
    print(f"  {ds:10s} {m:28s} {d:10s} {s / 3600:7.2f} h")
    total += 0 if s != s else s
print(f"ESTIMATED REMAINING: {total / 3600:.1f} h  (models without a trial are not counted)")
