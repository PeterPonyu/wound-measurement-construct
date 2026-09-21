#!/usr/bin/env bash
# Download GSE268834_RAW.tar through verified HTTP ranges.
# NCBI closes long single-body transfers for this archive; each part is
# independently checked before concatenation, and the tar is listed before
# extraction.  No curl resume/append is used.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/GSE268834"
URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE268nnn/GSE268834/suppl/GSE268834_RAW.tar"
TOTAL=605153280
CHUNK=50000000
PARTS="$DEST/.raw_parts"
TAR="$DEST/GSE268834_RAW.tar"
MAX_TRIES=6
mkdir -p "$PARTS"

download_part() {
  local start="$1" end="$2" idx="$3"
  local expected=$((end - start + 1))
  local part="$PARTS/part-$(printf '%03d' "$idx")"
  if [ -f "$part" ] && [ "$(stat -c%s "$part")" -eq "$expected" ]; then
    echo "[$(date +%T)] have part $idx ($expected bytes)"
    return 0
  fi
  for attempt in $(seq 1 "$MAX_TRIES"); do
    rm -f "$part.part"
    curl -fsSL --max-time 1800 --speed-limit 20000 --speed-time 120 \
      --retry 3 --retry-delay 5 -H "Range: bytes=$start-$end" \
      -o "$part.part" "$URL" || true
    local actual=0
    [ -f "$part.part" ] && actual="$(stat -c%s "$part.part")"
    if [ "$actual" -eq "$expected" ]; then
      mv "$part.part" "$part"
      echo "[$(date +%T)] OK part $idx ($actual bytes, try $attempt)"
      return 0
    fi
    echo "[$(date +%T)] short part $idx $actual/$expected (try $attempt)" >&2
    rm -f "$part.part"
    sleep $((attempt * 3))
  done
  echo "FAILED part $idx" >&2
  return 1
}

export URL PARTS MAX_TRIES
export -f download_part

idx=0
while [ $((idx * CHUNK)) -lt "$TOTAL" ]; do
  start=$((idx * CHUNK))
  end=$((start + CHUNK - 1))
  [ "$end" -ge "$TOTAL" ] && end=$((TOTAL - 1))
  printf '%s\t%s\t%s\n' "$start" "$end" "$idx"
  idx=$((idx + 1))
done | xargs -P 4 -n 3 bash -c 'download_part "$1" "$2" "$3"' _

for part in "$PARTS"/part-*; do
  [ -f "$part" ] || { echo "missing part: $part" >&2; exit 1; }
done
rm -f "$TAR.part"
cat "$PARTS"/part-* > "$TAR.part"
[ "$(stat -c%s "$TAR.part")" -eq "$TOTAL" ] || {
  echo "assembled tar has wrong size" >&2; exit 1;
}
tar -tf "$TAR.part" >/dev/null
mv "$TAR.part" "$TAR"
mkdir -p "$DEST/per_gsm"
tar -xf "$TAR" -C "$DEST/per_gsm"
echo "[$(date +%T)] GSE268834 archive verified and extracted"
