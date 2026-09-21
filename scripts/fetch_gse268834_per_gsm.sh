#!/usr/bin/env bash
# Fetch the eight GSE268834 10x matrices without relying on the series tar.
#
# The experiment is used only for frozen external state projection and
# diabetic-vs-non-diabetic composition description.  This fetcher does not
# imply that the series has patient-level healing metadata.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/GSE268834/per_gsm"
LIST="$ROOT/data/raw/GSE268834/filelist.txt"
MATRIX="$ROOT/data/raw/GSE268834/GSE268834_series_matrix.txt.gz"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE268nnn/GSE268834"
MAX_TRIES=6

mkdir -p "$DEST"
curl -fsSL --max-time 120 -o "$LIST" "$BASE/suppl/filelist.txt"
curl -fsSL --max-time 120 -o "$MATRIX" "$BASE/matrix/GSE268834_series_matrix.txt.gz"
gzip -t "$MATRIX"

manifest() { awk -F '\t' '$1 == "File" { print $2 "\t" $4 }' "$LIST"; }

while IFS=$'\t' read -r name expected; do
  gsm="${name%%_*}"
  prefix="$(printf '%s' "$gsm" | sed -E 's/GSM([0-9]+)[0-9]{3}$/GSM\1nnn/')"
  url="https://ftp.ncbi.nlm.nih.gov/geo/samples/$prefix/$gsm/suppl/$name"
  dst="$DEST/$name"
  if [ -f "$dst" ] && [ "$(stat -c%s "$dst")" -eq "$expected" ]; then
    echo "[$(date +%T)] have $name"
    continue
  fi
  for attempt in $(seq 1 "$MAX_TRIES"); do
    rm -f "$dst.part"
    curl -fsSL --max-time 1800 --speed-limit 20000 --speed-time 120 \
      -o "$dst.part" "$url" || true
    actual=0
    [ -f "$dst.part" ] && actual="$(stat -c%s "$dst.part")"
    if [ "$actual" -eq "$expected" ] && gzip -t "$dst.part" 2>/dev/null; then
      mv "$dst.part" "$dst"
      echo "[$(date +%T)] OK $name ($actual bytes, try $attempt)"
      break
    fi
    echo "[$(date +%T)] short/corrupt $name $actual/$expected (try $attempt)"
    rm -f "$dst.part"
    sleep $((attempt * 5))
  done
  if [ ! -f "$dst" ] || [ "$(stat -c%s "$dst")" -ne "$expected" ]; then
    echo "FAILED $name" >&2
    exit 1
  fi
done < <(manifest)

ok=0
while IFS=$'\t' read -r name expected; do
  actual="$(stat -c%s "$DEST/$name" 2>/dev/null || echo 0)"
  if [ "$actual" -eq "$expected" ] && gzip -t "$DEST/$name" 2>/dev/null; then
    ok=$((ok + 1))
  else
    echo "BAD $name $actual/$expected" >&2
    exit 1
  fi
done < <(manifest)
echo "[$(date +%T)] GSE268834 complete=$ok"
