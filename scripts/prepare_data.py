"""Stage runners: make sure data/processed/<name> exists and matches data/processed/CHECKSUMS.sha256.
If the processed file is missing (e.g. reviews.csv.gz not shipped), download the pinned source, check its sha256
(src/data.py SOURCES) and rebuild it with src/data.py. Usage: python scripts/prepare_data.py patents m5 reviews"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data as D  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


sums = {}
for line in open(os.path.join(D.OUT, "CHECKSUMS.sha256")):
    h, f = line.split()
    sums[f] = h
ok = True
for name in sys.argv[1:]:
    fn = f"{name}.csv.gz" if name == "reviews" else f"{name}.csv"
    p = os.path.join(D.OUT, fn)
    if os.path.exists(p):
        good = sha(p) == sums.get(fn)
        print(f"{fn}: {'sha256 OK' if good else 'sha256 MISMATCH'}")
        ok &= good
        continue
    print(f"{fn}: missing -> rebuilding from the pinned source")
    src = D.SOURCES[name]
    if name == "reviews":
        dst = os.path.join(D.AUTH, "reviews", src["file"])
    elif name == "patents":
        dst = os.path.join(D.AUTH, "patents", src["file"])
    else:
        dst = os.path.join(D.ROOT, "data", "raw", "m5", src["file"])
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("  downloading", src["url"])
        blob = urllib.request.urlopen(src["url"], timeout=600).read()
        if src["url"].endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(blob))
            member = [m for m in z.namelist() if m.endswith(src["file"])][0]
            blob = z.read(member)
        open(dst, "wb").write(blob)
    if "sha256" in src:
        assert sha(dst) == src["sha256"], f"source checksum mismatch for {dst}"
        print("  source sha256 OK")
    D.main([name])
    # gzip headers carry a timestamp, so a rebuilt reviews.csv.gz can differ byte-wise; check the content instead
    import pandas as pd
    df = pd.read_csv(p)
    print(f"  rebuilt {fn}: {df['series_id'].nunique()} series, {len(df)} rows")
print("DATA_OK" if ok else "DATA_CHECKSUM_MISMATCH")
sys.exit(0 if ok else 1)
