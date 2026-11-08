# Third-party material and licenses

The MIT license in `LICENSE` covers only the original code and documentation of this replication. The following
files keep their own licenses and are **not** relicensed.

| Path | Origin | License |
|---|---|---|
| `article/rescience.cls`, `article/header.tex`, `article/article.tex`, `article/Makefile`, `article/yaml-to-latex.py`, `article/article.py` | ReScience C article template, https://github.com/rescience/template @ 0978d0e (2023-02-15), (C) 2015-2020 Nicolas Rougier and co-authors. `header.tex` is modified (abstract block enabled, footer punctuation fixed); `article.tex` adds one `\usepackage` line. | GPL-3.0-or-later, see `article/COPYING` |
| `article/source-serif-pro/`, `article/source-sans-pro/`, `article/source-code-pro/` | Adobe Source Serif/Sans/Code Pro fonts | SIL Open Font License 1.1 (`SIL Open Font License.txt` in each folder) |
| `article/roboto/` | Roboto font (Google) | Apache License 2.0 (`Apache License.txt`) |
| `article/content.tex`, `article/metadata.yaml`, `article/bibliography.bib`, `article/tables/`, `article/figures/`, `article/article.pdf` | The replication article itself | The article text and figures are released under CC BY 4.0, the license ReScience C applies to published articles |

## Data (not redistributed in this repository)

Raw and processed data files are not included; `scripts/fetch_raw_data.sh` and `scripts/prepare_data.py` download
them from the original sources and verify SHA-256 checksums (`data/raw/SHA256SUMS.txt`,
`data/processed/CHECKSUMS.sha256`). Users must follow each source's terms:

- NYC Crime: NYC Open Data, *NYPD Complaint Data Historic* (`qgea-i56i`), NYC Open Data terms of use.
- Glucose: PhysioNet, *BIG IDEAs Lab Glycemic Variability and Wearable Device Data* v1.1.2, ODC-By 1.0.
- Heart-Rate: PhysioNet, *Heart Rate Oscillations during Meditation* v1.0.0, ODC-By 1.0.
- Reviews: the original authors' UCI deposit (doi:10.24432/C5SP7K), CC BY 4.0.
- Patents: `data/patents/y_patents.csv` from the original authors' repository
  https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration @ b60bfd9, MIT License (c) 2026 Coen Adler.
- M5: `sales_train_validation.csv` of the M5 competition, via Nixtla's mirror
  (https://github.com/Nixtla/m5-forecasts), under the M5 data terms.

## Original authors' code

No code from the original authors' repository (MIT) is included. The two diagnosis scripts in `diagnosis/`
re-implement aggregation definitions found there and cite them with `[FROM AUTHORS' CODE: file:line, commit]` tags.

## Model weights

Not included. Chronos-Bolt, TimesFM 2.5 and TiRex checkpoints are downloaded from Hugging Face at the revisions
pinned in `envs/MODEL_REVISIONS.md`, under their respective model licenses.
