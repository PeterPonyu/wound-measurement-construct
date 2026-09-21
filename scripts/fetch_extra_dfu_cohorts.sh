#!/usr/bin/env bash
# Two further human DFU scRNA-seq cohorts, fetched when NCBI comes back.
#
# GSE223964 ships an annotated .h5ad alongside the raw tar, which lets our
# marker-based lineage call be checked against the submitters' own labels.
# GSE245703 is the companion 12-sample series from the same group.
#
# NCBI was returning SSL_ERROR_SYSCALL on every host (ftp and eutils alike)
# when this was written, so the script loops with backoff rather than failing.
set -u
ROOT=data/raw
OUTER_TRIES=60
SLEEP_BETWEEN=120

want_bytes () { awk -F'\t' -v n="$2" '$2==n{print $4; exit}' "$1"; }

grab () {  # url dest
  for i in 1 2 3 4 5; do
    rm -f "$2.part"
    if curl -sS --max-time 1800 --speed-limit 20000 --speed-time 120 \
            -o "$2.part" "$1" 2>/dev/null && [ -s "$2.part" ]; then
      mv "$2.part" "$2"; return 0
    fi
    sleep $((i * 5))
  done
  rm -f "$2.part"; return 1
}

for attempt in $(seq 1 $OUTER_TRIES); do
  if curl -sS -o /dev/null --max-time 30 https://ftp.ncbi.nlm.nih.gov/geo/ 2>/dev/null; then
    echo "[$(date +%T)] NCBI reachable on outer attempt $attempt"
    break
  fi
  echo "[$(date +%T)] NCBI unreachable (outer attempt $attempt), waiting"
  sleep $SLEEP_BETWEEN
done

for g in GSE223964 GSE245703; do
  pre=$(echo "$g" | sed -E 's/GSE([0-9]+)[0-9]{3}$/GSE\1nnn/')
  base="https://ftp.ncbi.nlm.nih.gov/geo/series/$pre/$g"
  dir="$ROOT/$g"
  mkdir -p "$dir"
  echo "[$(date +%T)] === $g ==="
  grab "$base/matrix/${g}_series_matrix.txt.gz" "$dir/${g}_series_matrix.txt.gz" \
    && echo "[$(date +%T)] series matrix ok" || echo "[$(date +%T)] series matrix FAILED"
  grab "$base/suppl/filelist.txt" "$dir/filelist.txt" || { echo "no filelist"; continue; }

  # Prefer the annotated h5ad when the series offers one; fall back to the tar.
  for name in $(awk -F'\t' '$1=="File" || $1=="Archive"{print $2}' "$dir/filelist.txt"); do
    case "$name" in
      *Integrated_all_cells.h5ad.gz|*_RAW.tar)
        want=$(want_bytes "$dir/filelist.txt" "$name")
        [ -f "$dir/$name" ] && [ "$(stat -c%s "$dir/$name")" = "$want" ] && \
          { echo "[$(date +%T)] have $name"; continue; }
        if grab "$base/suppl/$name" "$dir/$name"; then
          got=$(stat -c%s "$dir/$name")
          [ "$got" = "$want" ] && echo "[$(date +%T)] OK $name ($got)" \
                               || echo "[$(date +%T)] SIZE MISMATCH $name $got/$want"
        else
          echo "[$(date +%T)] FAILED $name"
        fi
        ;;
    esac
  done
done
echo "[$(date +%T)] done"
