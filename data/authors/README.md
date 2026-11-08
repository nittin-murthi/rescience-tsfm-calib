# Authors' processed data (round 5): used as data only, never for method

These files are used under the authors' licence: the repository is MIT, and the UCI deposit is the authors' own.
Every use in code is tagged `[FROM AUTHORS' DATA: file, commit]`.

| Dataset | Our file | Source | sha256 |
|---|---|---|---|
| Patents | `patents/y_patents.csv` (downloaded by `scripts/prepare_data.py patents`; not redistributed in the public repository) | github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration `data/patents/y_patents.csv` @ b60bfd9 | 760dfad2176a2ee45b827ed6b5b7503bbcab296c4585e91e2063998c45e0a42a |
| Reviews | `reviews/y_amazon-google-large.csv` (not committed; 110 MB) | UCI doi:10.24432/C5SP7K (authors' deposit, named in their `data/README.md` @ b60bfd9) | dfc5e6169c25c5d838281973db722ae468a0d8822ab2edc656115ba08b9bae81 |

**M5 is built by us from the raw public M5 file** `sales_train_validation.csv`, taken from Nixtla's mirror
`https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip`. It is not committed (119 MB); see
`src/data.py:build_m5`. As a diagnosis-only cross-check, our 70 department × store series equal the authors'
`data/m5/y_m5.csv` (@ b60bfd9) exactly. That holds for all 70 series and all 1,912 days, with a max abs diff of 0.

Processed outputs are written to `data/processed/{patents,m5}.csv` and `data/processed/reviews.csv.gz` by
`scripts/prepare_data.py`; they are not redistributed in the public repository. Their sha256 values are in
`data/processed/CHECKSUMS.sha256`.
