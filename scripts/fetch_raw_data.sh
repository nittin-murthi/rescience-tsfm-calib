#!/usr/bin/env bash
# Download the raw public data for Crime, Heart-Rate and Glucose into data/raw/ and verify SHA-256 checksums
# against data/raw/SHA256SUMS.txt (the files used for the article). Reviews, M5 and Patents are fetched by
# scripts/prepare_data.py from the pinned sources in src/data.py (SOURCES).
# Usage: bash scripts/fetch_raw_data.sh        (from the repository root; needs curl and sha256sum or shasum)
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW/crime" "$RAW/glucose" "$RAW/heartrate"

# NYC Crime: NYC Open Data "NYPD Complaint Data Historic" (qgea-i56i), daily counts per borough by report date,
# aggregated server-side. This is a LIVE dataset: a new download can differ from the 2026-10-23 copy we used,
# in which case the checksum check below reports a mismatch.
[ -s "$RAW/crime/nypd_daily_by_boro_rptdt.csv" ] || curl -fsSL -G https://data.cityofnewyork.us/resource/qgea-i56i.csv \
  --data-urlencode '$select=boro_nm,date_trunc_ymd(rpt_dt) AS day,count(*) AS n' \
  --data-urlencode '$group=boro_nm,day' --data-urlencode '$order=day,boro_nm' \
  --data-urlencode '$limit=200000' -o "$RAW/crime/nypd_daily_by_boro_rptdt.csv"

# Glucose: PhysioNet "BIG IDEAs Lab Glycemic Variability and Wearable Device Data" v1.1.2 (ODC-By 1.0).
G=https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2
for i in $(seq -w 1 16); do
  n=$(printf '%03d' "$((10#$i))")
  [ -s "$RAW/glucose/Dexcom_$n.csv" ] || curl -fsSL "$G/$n/Dexcom_$n.csv" -o "$RAW/glucose/Dexcom_$n.csv"
done
[ -s "$RAW/glucose/LICENSE.txt" ] || curl -fsSL "$G/LICENSE.txt" -o "$RAW/glucose/LICENSE.txt"

# Heart-Rate: PhysioNet "Heart Rate Oscillations during Meditation" v1.0.0, data/metron/M1..M14 (ODC-By 1.0).
H=https://physionet.org/files/meditation/1.0.0/data/metron
for i in $(seq 1 14); do
  [ -s "$RAW/heartrate/M$i.txt" ] || curl -fsSL "$H/M$i" -o "$RAW/heartrate/M$i.txt"
done

cd "$RAW"
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS.txt; else shasum -a 256 -c SHA256SUMS.txt; fi
echo "RAW_DATA_OK"
