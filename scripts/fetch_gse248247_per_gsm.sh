#!/usr/bin/env bash
# Retrieve the two GSE248247 10x MTX bundles with per-file checks.
#
# The series contains one diabetic and one non-diabetic plantar wound sample.
# It has no healing endpoint or authoritative patient metadata, so this data
# is for a frozen projection / low-yield stress test only.  Do not use curl -C -.
set -euo pipefail

ROOT="."
OUT="$ROOT/data/raw/GSE248247/per_gsm"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7908nnn"
FILELIST_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE248nnn/GSE248247/suppl/filelist.txt"
FILELIST="$ROOT/data/raw/GSE248247/filelist.txt"

mkdir -p "$OUT"

if [[ ! -s "$FILELIST" ]]; then
  curl -fsSL --retry 8 --retry-all-errors --connect-timeout 30 --max-time 600 \
    -o "$FILELIST.part" "$FILELIST_URL"
  test -s "$FILELIST.part"
  mv "$FILELIST.part" "$FILELIST"
fi

while IFS=$'\t' read -r kind name _time expected _type; do
  [[ "$kind" == "File" ]] || continue
  [[ "$name" == GSM7908977_* || "$name" == GSM7908978_* ]] || continue
  case "$name" in
    GSM7908977_*) gsm=GSM7908977 ;;
    GSM7908978_*) gsm=GSM7908978 ;;
    *) continue ;;
  esac
  dest="$OUT/$name"
  if [[ -s "$dest" ]] && [[ "$(stat -c%s "$dest")" == "$expected" ]] && gzip -t "$dest" 2>/dev/null; then
    echo "already verified: $name"
    continue
  fi
  url="$BASE/${gsm}/suppl/${name}"
  echo "fetching $name ($expected bytes)"
  curl -fsSL --retry 8 --retry-all-errors --connect-timeout 30 --max-time 1800 \
    -o "$dest.part" "$url"
  test "$(stat -c%s "$dest.part")" = "$expected"
  gzip -t "$dest.part"
  mv "$dest.part" "$dest"
done < "$FILELIST"

test "$(find "$OUT" -maxdepth 1 -type f -name '*.gz' | wc -l)" -eq 6
echo "GSE248247 per-GSM bundles verified under $OUT"
