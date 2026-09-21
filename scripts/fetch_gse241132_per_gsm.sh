#!/usr/bin/env bash
# GSE241132 per-GSM fetcher (human acute wound, 3 donors x 4 timepoints).
#
# Goes straight to per-sample files rather than the 775,905,280-byte series
# tar. GSE326622's series tar truncated server-side twice at ~780 MB, and this
# archive is the same size class; the per-GSM zips top out at ~83 MB, well
# under that cliff.
set -u
ROOT=data/raw/GSE241132
DEST="$ROOT/per_gsm"
LIST="$ROOT/filelist.txt"
MAX_TRIES=8
mkdir -p "$DEST"

curl -sS -o "$LIST" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE241nnn/GSE241132/suppl/filelist.txt"
curl -sS -o "$ROOT/GSE241132_cell_metadata.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE241nnn/GSE241132/suppl/GSE241132_cell_metadata.txt.gz"
curl -sS -o "$ROOT/GSE241132_series_matrix.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE241nnn/GSE241132/matrix/GSE241132_series_matrix.txt.gz"

manifest () { awk -F'\t' '$1=="File"{print $2"\t"$4}' "$LIST"; }

manifest | while IFS=$'\t' read -r name want; do
  gsm=${name%%_*}
  pre=$(echo "$gsm" | sed -E 's/GSM([0-9]+)[0-9]{3}$/GSM\1nnn/')
  url="https://ftp.ncbi.nlm.nih.gov/geo/samples/$pre/$gsm/suppl/$name"
  dst="$DEST/$name"
  if [ -f "$dst" ] && [ "$(stat -c%s "$dst")" -eq "$want" ]; then
    echo "[$(date +%T)] have  $name"; continue
  fi
  for try in $(seq 1 $MAX_TRIES); do
    rm -f "$dst.part"
    curl -sS --max-time 1800 --speed-limit 20000 --speed-time 120 \
         -o "$dst.part" "$url" || true
    have=0; [ -f "$dst.part" ] && have=$(stat -c%s "$dst.part")
    if [ "$have" -eq "$want" ]; then
      mv "$dst.part" "$dst"
      echo "[$(date +%T)] OK    $name ($have bytes, try $try)"
      break
    fi
    echo "[$(date +%T)] short $name $have/$want (try $try)"
    sleep 8
  done
  rm -f "$dst.part"
done

echo "[$(date +%T)] --- verify ---"
ok=0; bad=0
while IFS=$'\t' read -r name want; do
  have=0; [ -f "$DEST/$name" ] && have=$(stat -c%s "$DEST/$name")
  if [ "$have" -eq "$want" ] && unzip -tqq "$DEST/$name" >/dev/null 2>&1; then
    ok=$((ok + 1))
  else
    bad=$((bad + 1)); echo "BAD $name $have/$want"
  fi
done < <(manifest)
echo "[$(date +%T)] complete=$ok bad=$bad"
