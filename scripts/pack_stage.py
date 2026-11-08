"""Pack a stage's outputs: python scripts/pack_stage.py <stage> <out_zip> <dataset>...
Main zip: per-dataset metrics (results/stage<N>/), run logs, timing trial, gpu_env, pip freezes, code. No venvs, no
.git. Forecast .npz files go into the same zip when their total is under 2 GB, else into <out>_forecasts.zip."""
import glob
import os
import sys
import zipfile

stage, out, dss = sys.argv[1], sys.argv[2], sys.argv[3:]
ROOT = os.getcwd()
pref = os.path.basename(ROOT.rstrip("/"))
files = []
for pat in [f"results/stage{stage}/**/*", f"results/runlogs_stage{stage}*.txt", f"results/runlogs/gpu_env_stage{stage}*",
            f"results/runlogs/env_stage{stage}/*", "src/*.py", "scripts/*", "envs/*", "run_stage*.sh", "START_HERE.txt",
            "STAGES.md", "data/processed/CHECKSUMS.sha256", "data/authors/README.md"]:
    files += glob.glob(pat, recursive=True)
for ds in dss:
    for pat in [f"results/runlogs/{ds}__*.json", f"results/runlogs/partial/{ds}__*.jsonl", f"results/timing/{ds}__*",
                f"results/runlogs/partial_arima/{ds}__orders.json", f"data/processed/{ds}.json"]:
        files += glob.glob(pat)
fc = sorted(f for ds in dss for f in glob.glob(f"results/forecasts/{ds}__*.npz"))
size = sum(os.path.getsize(f) for f in fc)
separate = size > 2e9
files = sorted({f for f in files if os.path.isfile(f) and "/." not in f and "__pycache__" not in f})
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files + ([] if separate else fc):
        z.write(f, os.path.join(pref, f), compress_type=zipfile.ZIP_STORED if f.endswith(".npz") else None)
print(f"{out}: {len(files)} files + {0 if separate else len(fc)} forecasts, {os.path.getsize(out) / 1e6:.1f} MB")
if separate:
    fo = out[:-4] + "_forecasts.zip"
    with zipfile.ZipFile(fo, "w", zipfile.ZIP_STORED) as z:
        for f in fc:
            z.write(f, os.path.join(pref, f))
    print(f"{fo}: {len(fc)} forecasts, {os.path.getsize(fo) / 1e9:.2f} GB (optional to send; metrics are in the main zip)")
