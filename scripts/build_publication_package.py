"""Build the sanitized public package of this replication from the git HEAD (no history).

Copies an allow-listed subset of tracked files to OUT, replaces machine-identifying strings in run logs
(home-directory user name, host name; no numbers are changed), writes MANIFEST.sha256 and a .gitignore,
and packs OUT.tar.gz. Excluded: raw and processed data files, the authors' data file, the paper PDF/text,
the duplicate vendored template copy, forecasts, model weights and virtual environments.
Usage: python3 scripts/build_publication_package.py [/workspace/rescience-tsfm-calib-publication]
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(ROOT), "rescience-tsfm-calib-publication")

EXCLUDE = [
    r"^data/raw/(?!SHA256SUMS\.txt$)",          # raw data: downloaded by scripts/fetch_raw_data.sh
    r"^data/processed/.*\.(csv|csv\.gz)$",      # processed data: rebuilt by src/data.py / prepare_data.py
    r"^data/authors/.*\.csv$",                  # authors' data file: downloaded by prepare_data.py
    r"^paper/",                                 # copy of the original paper (cite it instead)
    r"^article/rescience-template/",            # duplicate of the template files already in article/
    r"^results/forecasts/", r"\.npz$", r"(^|/)venvs?/", r"(^|/)\.venv", r"__pycache__", r"\.pyc$",
]
SANITIZE = [(re.compile(r"/Users/(?!<user>)[A-Za-z0-9._-]+"), "/Users/<user>"),
            (re.compile(r"\b[\w-]+-MacBook-(?:Pro|Air)(?:\.local)?\b"), "<hostname>")]
VERBATIM = {"scripts/build_publication_package.py"}  # never rewrite this script itself
TEXT_EXT = (".txt", ".json", ".jsonl", ".csv", ".md", ".sh", ".py", ".tex", ".yaml", ".log")


def tracked():
    out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"], check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    files = [p for p in tracked() if not any(re.search(x, p) for x in EXCLUDE)]
    n_sanitized = 0
    for rel in files:
        src, dst = os.path.join(ROOT, rel), os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if rel.endswith(TEXT_EXT) and rel not in VERBATIM:
            with open(src, "rb") as f:
                raw = f.read()
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:
                shutil.copy2(src, dst)
                continue
            new = txt
            for pat, rep in SANITIZE:
                new = pat.sub(rep, new)
            n_sanitized += new != txt
            with open(dst, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            shutil.copystat(src, dst)
        else:
            shutil.copy2(src, dst)
    with open(os.path.join(ROOT, ".gitignore")) as f:
        gi = f.read()
    gi += ("\n# public package: data, forecasts and environments are never committed\n"
           "data/raw/*/\ndata/processed/*.csv\ndata/processed/*.csv.gz\ndata/authors/**/*.csv\n"
           ".venv-*/\n.smoke/\nresults/runlogs/partial_*/\n*.zip\n")
    with open(os.path.join(OUT, ".gitignore"), "w") as f:
        f.write(gi)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    with open(os.path.join(OUT, "PACKAGE_INFO.txt"), "w") as f:
        f.write(f"Built by scripts/build_publication_package.py from development commit {head}.\n"
                "Run-log paths and host names were redacted (/Users/<user>, <hostname>); no values changed.\n"
                "Data, forecasts, model weights and environments are not included; see REPRODUCE.md.\n")
    lines = []
    for dp, _, fs in os.walk(OUT):
        for fn in sorted(fs):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, OUT)
            if rel == "MANIFEST.sha256":
                continue
            with open(p, "rb") as f:
                lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {rel}")
    with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
        f.write("\n".join(sorted(lines, key=lambda s: s[66:])) + "\n")
    tgz = OUT + ".tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(OUT, arcname=os.path.basename(OUT))
    print(f"files={len(lines) + 1} sanitized_text_files={n_sanitized} out={OUT} tgz={tgz} "
          f"tgz_bytes={os.path.getsize(tgz)}")


if __name__ == "__main__":
    main()
