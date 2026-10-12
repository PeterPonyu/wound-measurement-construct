"""Bounded external-sort duplicate rejection, exact QC and portable disk CSR.

The artifact is an explicitly versioned binary CSR contract, NOT H5AD. Only
one sort run, one merge record per input and one cell's CSR buffers are resident.
No global coordinate set, SciPy matrix loader, dense matrix, network or CUDA.
"""
from __future__ import annotations

from array import array
from collections import Counter
from contextlib import contextmanager
import csv
import gzip
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import zlib

from . import gse175634_preparation as identity


RECORD = struct.Struct("<QQQ")  # original zero-based cell, gene, exact count
VALUE = struct.Struct("<q")
INDEX = struct.Struct("<I")
POINTER = struct.Struct("<Q")
MAX_COUNT = 2**63 - 1
CSR_SCHEMA = "ortho.csr.binary.v1"
QC_EXTRA = ["qc_library_count_exact", "qc_detected_genes", "qc_mito_count_exact", "qc_mito_fraction_exact"]
QC_COLUMNS = ["cell_index_one_based", "cell", "library_count_exact", "detected_genes", "mito_count_exact",
              "mito_fraction_exact", "primary_keep"]


class QCError(identity.PreparationError):
    """Failed duplicate, count, axis, QC, resource or roundtrip contract."""


def policy(spec):
    cfg = spec.get("qc_runtime", {})
    for key in ("memory_bytes", "cpu_seconds", "sort_memory_bytes", "python_coordinate_bytes_budget",
                "merge_fan_in", "io_chunk_bytes", "disk_margin_bytes", "max_artifact_metadata_bytes"):
        identity._positive(cfg.get(key), key)
    if not 1 << 20 <= cfg["memory_bytes"] <= 4 << 30 or cfg["sort_memory_bytes"] > 256 << 20 \
            or cfg["python_coordinate_bytes_budget"] < 192 \
            or cfg["sort_memory_bytes"] < cfg["python_coordinate_bytes_budget"] \
            or not 2 <= cfg["merge_fan_in"] <= 16 \
            or not RECORD.size <= cfg["io_chunk_bytes"] <= 1 << 20 \
            or cfg["io_chunk_bytes"] % RECORD.size \
            or cfg["sort_memory_bytes"] * 2 >= cfg["memory_bytes"]:
        raise QCError("external-sort implementation resource ceiling/record budget exceeded")
    return cfg


def estimate_resources(spec, receipt):
    cfg, expected = policy(spec), spec["expected"]
    if expected["genes"] > 2**32 or expected["cells"] > MAX_COUNT or expected["entries"] > MAX_COUNT:
        raise QCError("binary CSR index/offset representation budget exceeded")
    run_bytes = expected["entries"] * RECORD.size
    run_capacity = cfg["sort_memory_bytes"] // cfg["python_coordinate_bytes_budget"]
    maximum_initial_runs = (expected["entries"] + run_capacity - 1) // run_capacity
    if maximum_initial_runs > 10000:
        raise QCError("external-sort file/inode/manifest budget exceeded before ingestion")
    csr_bytes = expected["entries"] * (VALUE.size + INDEX.size) + (2 * expected["cells"] + 1) * POINTER.size
    metadata_bytes = cfg["max_artifact_metadata_bytes"]
    planned = max(spec["budgets"]["matrix_followup_disk_bytes"], 2 * run_bytes + csr_bytes + metadata_bytes + cfg["disk_margin_bytes"])
    memory = (cfg["sort_memory_bytes"] + expected["cells"] * 192 + expected["genes"] * 16
              + cfg["merge_fan_in"] * cfg["io_chunk_bytes"] + (256 << 20))
    if memory > cfg["memory_bytes"]:
        raise QCError("conservative sort/axis/merge memory estimate exceeds hard memory budget")
    if sum(row["bytes"] for row in receipt["files"] if row["file"] != identity.MATRIX) > metadata_bytes:
        raise QCError("raw identity closure exceeds artifact metadata budget")
    return dict(coordinate_record_bytes=RECORD.size, one_sort_generation_bytes=run_bytes,
        two_generation_sort_bytes=2 * run_bytes, maximum_CSR_and_row_map_bytes=csr_bytes,
        artifact_metadata_reserved_bytes=metadata_bytes, planned_increment_bytes=planned,
        conservative_memory_bytes=memory, hard_memory_budget_bytes=cfg["memory_bytes"],
        maximum_initial_run_files=maximum_initial_runs,
        post_write_free_fraction_required=max(.3, spec["budgets"]["matrix_post_write_free_fraction"]))


def resource_preflight(parent, spec, estimate):
    usage = shutil.disk_usage(identity.safe_path(parent, directory=True))
    snapshot = identity.resource_snapshot(parent, spec)
    remaining = usage.free - estimate["planned_increment_bytes"]
    checks = dict(disk_bytes=remaining >= spec["budgets"]["minimum_free_bytes"],
                  disk_fraction=remaining / usage.total >= estimate["post_write_free_fraction_required"],
                  memory=snapshot["memory_available_bytes"] is not None
                         and snapshot["memory_available_bytes"] >= policy(spec)["memory_bytes"])
    return dict(status="passed" if all(checks.values()) else "blocked", checks=checks,
        disk_total_bytes=usage.total, disk_free_bytes=usage.free, post_write_free_bytes=remaining,
        memory_available_bytes=snapshot["memory_available_bytes"], **estimate)


class StorageLedger:
    """Count only newly written lane-owned bytes, including gzip headers."""

    def __init__(self, limit, root):
        self.limit, self.live, self.peak = limit, 0, 0
        self.removed_run_bytes = 0
        self.root, self.created = identity.safe_path(root, directory=True), set()

    def add(self, size):
        if self.live + size > self.limit:
            raise QCError("aggregate new-output/scratch disk byte budget exhausted")
        self.live += size
        self.peak = max(self.peak, self.live)

    def remove_run(self, run):
        path = identity.safe_path(run["path"], file=True)
        if path not in self.created or not path.is_relative_to(self.root / "scratch"):
            raise QCError("only registered lane-owned scratch runs may be removed")
        if path.stat().st_size != run["bytes"] or identity.sha256_file(path) != run["sha256"]:
            raise QCError("owned scratch run changed; do not delete it")
        path.unlink()
        self.created.remove(path)
        self.live -= run["bytes"]
        self.removed_run_bytes += run["bytes"]


class CountedWriter:
    def __init__(self, stream, ledger):
        self.stream, self.ledger = stream, ledger

    def write(self, body):
        self.ledger.add(len(body))
        return self.stream.write(body)

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.stream.tell()


@contextmanager
def fresh_binary(path, ledger):
    path = identity.safe_path(path, fresh=True)
    if not path.is_relative_to(ledger.root):
        raise QCError("output writer escaped its lane-owned root")
    with path.open("xb") as stream:
        ledger.created.add(path)
        yield CountedWriter(stream, ledger)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def fresh_gzip_text(path, ledger):
    with fresh_binary(path, ledger) as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                yield text


def write_json(path, value, ledger):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with fresh_binary(path, ledger) as stream:
        stream.write(body)


def copy_exact(source, target, expected_sha, ledger):
    digest = hashlib.sha256()
    with identity.regular_input(source) as raw, fresh_binary(target, ledger) as sink:
        for block in iter(lambda: raw.read(1 << 20), b""):
            digest.update(block)
            sink.write(block)
    if digest.hexdigest() != expected_sha:
        raise QCError("identity copy lost original hash binding")


def iter_matrix(path, row, spec, stats):
    try:
        yield from _iter_matrix(path, row, spec, stats)
    except (OSError, EOFError, zlib.error) as error:
        raise QCError(f"matrix gzip/file integrity failure: {error}") from error


def _iter_matrix(path, row, spec, stats):
    """Two passes on one stable descriptor: compressed SHA then exact gzip parse."""
    scanner = Path(__file__).parents[2] / "remote/autodl/matrix_scan.py"
    if identity.sha256_file(scanner) != spec["scanner_sha256"]:
        raise QCError("exact count parser source hash binding mismatch")
    from remote.autodl.matrix_scan import MatrixScanError, _count, _integer

    expected, bounds = spec["expected"], spec["budgets"]
    with identity.regular_input(path) as raw:
        if os.fstat(raw.fileno()).st_size != row["bytes"]:
            raise QCError("matrix compressed size differs from acceptance")
        digest = hashlib.sha256()
        for block in iter(lambda: raw.read(1 << 20), b""):
            digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise QCError("matrix compressed SHA256 differs from acceptance")
        raw.seek(0)
        field, dimensions, decoded, observed = None, False, 0, 0
        with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            line_number = 0
            while line := gz.readline(bounds["matrix_max_line_bytes"] + 1):
                line_number += 1
                decoded += len(line)
                if len(line) > bounds["matrix_max_line_bytes"] or decoded > row["decoded_bytes"] \
                        or decoded > bounds["matrix_max_decoded_bytes"]:
                    raise QCError("matrix line/decoded byte budget exceeded")
                tokens = line.split()
                if line_number == 1:
                    header = [t.lower() for t in tokens]
                    if len(header) != 5 or header[:3] != [b"%%matrixmarket", b"matrix", b"coordinate"] \
                            or header[3] not in {b"integer", b"real"} or header[4] != b"general":
                        raise QCError("unsupported MatrixMarket coordinate integer/real general header")
                    field = header[3].decode()
                    continue
                if not tokens or line.lstrip().startswith(b"%"):
                    continue
                if len(tokens) != 3:
                    raise QCError("matrix dimension/record requires exactly three columns")
                if not dimensions:
                    try:
                        shape = tuple(_integer(token, "matrix dimension") for token in tokens)
                    except MatrixScanError as error:
                        raise QCError(str(error)) from error
                    if shape != (expected["genes"], expected["cells"], expected["entries"]):
                        raise QCError("matrix axes/stored-entry count differ from bound author specification")
                    dimensions = True
                    continue
                try:
                    gene, cell = (_integer(t, "coordinate") for t in tokens[:2])
                    value = _count(tokens[2], field, bounds["matrix_max_count_digits"])
                except MatrixScanError as error:
                    raise QCError(str(error)) from error
                if not 1 <= gene <= expected["genes"] or not 1 <= cell <= expected["cells"]:
                    raise QCError("out-of-bounds original 1-based coordinate")
                if value > MAX_COUNT:
                    raise QCError("exact count exceeds signed-int64 CSR representation; no lossy cast")
                observed += 1
                if observed > expected["entries"]:
                    raise QCError("trailing entries exceed declared count")
                stats["stored_count_sum_exact"] += value
                stats["explicit_zero_records"] += value == 0
                stats["positive_records"] += value > 0
                stats["max_stored_count_exact"] = max(stats["max_stored_count_exact"], value)
                yield cell - 1, gene - 1, value
        if not dimensions or observed != expected["entries"] or decoded != row["decoded_bytes"]:
            raise QCError("premature EOF or exact decoded/stored-entry count mismatch")
        stats.update(compressed_sha256=digest.hexdigest(), compressed_bytes=row["bytes"], decoded_bytes=decoded,
                     stored_entries=observed, gzip_crc_and_eof_verified=True, source_unchanged=True, field=field)
    if identity.sha256_file(scanner) != spec["scanner_sha256"]:
        raise QCError("exact count parser source changed during ingestion")


def write_run(records, path, ledger, cfg):
    digest, count, previous, buffer = hashlib.sha256(), 0, None, bytearray()
    with fresh_binary(path, ledger) as stream:
        for record in records:
            key = record[:2]
            if key == previous:
                raise QCError(f"duplicate coordinate rejected, including explicit zeros: cell={key[0]+1}, gene={key[1]+1}")
            if previous is not None and key < previous:
                raise QCError("scratch run is not sorted by cell/gene")
            previous = key
            count += 1
            buffer.extend(RECORD.pack(*record))
            if len(buffer) >= cfg["io_chunk_bytes"]:
                stream.write(buffer)
                digest.update(buffer)
                buffer.clear()
        if buffer:
            stream.write(buffer)
            digest.update(buffer)
    return dict(path=str(Path(path).absolute()), records=count, bytes=count*RECORD.size, sha256=digest.hexdigest())


def iter_run(run, cfg):
    digest, count, previous = hashlib.sha256(), 0, None
    with identity.regular_input(run["path"]) as raw:
        if os.fstat(raw.fileno()).st_size != run["bytes"] or run["bytes"] != run["records"] * RECORD.size:
            raise QCError("scratch run size/record count changed")
        while block := raw.read(cfg["io_chunk_bytes"]):
            if len(block) % RECORD.size:
                raise QCError("partial external-sort binary record")
            digest.update(block)
            for record in RECORD.iter_unpack(block):
                if previous is not None and record[:2] <= previous:
                    raise QCError("scratch run order or duplicate contract changed")
                previous = record[:2]
                count += 1
                yield record
    if count != run["records"] or digest.hexdigest() != run["sha256"]:
        raise QCError("scratch run SHA256/record binding changed")


def merged_records(runs, cfg):
    previous = None
    for record in heapq.merge(*(iter_run(run, cfg) for run in runs)):
        if record[:2] == previous:
            raise QCError(f"duplicate coordinate rejected across runs: cell={record[0]+1}, gene={record[1]+1}")
        previous = record[:2]
        yield record


def external_runs(path, row, spec, out, ledger, stats):
    cfg = policy(spec)
    scratch = out / "scratch"
    scratch.mkdir()
    generation = scratch / "generation-000"
    generation.mkdir()
    limit = cfg["sort_memory_bytes"] // cfg["python_coordinate_bytes_budget"]
    chunk, runs = [], []
    for record in iter_matrix(path, row, spec, stats):
        chunk.append(record)
        if len(chunk) == limit:
            chunk.sort()
            runs.append(write_run(chunk, generation / f"run-{len(runs):06d}.bin", ledger, cfg))
            chunk.clear()
    if chunk:
        chunk.sort()
        runs.append(write_run(chunk, generation / f"run-{len(runs):06d}.bin", ledger, cfg))
        chunk.clear()
    write_json(scratch / "generation-000.json", dict(runs=runs, stored_entries=stats["stored_entries"]), ledger)
    initial_runs, pass_number = len(runs), 0
    while len(runs) > cfg["merge_fan_in"]:
        pass_number += 1
        generation = scratch / f"generation-{pass_number:03d}"
        generation.mkdir()
        following = []
        for start in range(0, len(runs), cfg["merge_fan_in"]):
            selected = runs[start:start+cfg["merge_fan_in"]]
            merged = write_run(merged_records(selected, cfg), generation / f"run-{len(following):06d}.bin", ledger, cfg)
            if merged["records"] != sum(run["records"] for run in selected):
                raise QCError("external merge lost stored coordinate records")
            following.append(merged)
        write_json(scratch / f"generation-{pass_number:03d}.json", dict(runs=following, previous_runs_verified=True), ledger)
        for run in runs:
            ledger.remove_run(run)
        runs = following
    return runs, dict(initial_runs=initial_runs, intermediate_merge_generations=pass_number,
                      maximum_merge_fan_in=cfg["merge_fan_in"], maximum_resident_coordinate_records=limit)


def load_axes_and_identity(metadata_path, report, spec):
    root, expected = Path(metadata_path).parent, spec["expected"]
    cells = []
    for name, target in (("cell_names_no_header.tsv.gz", cells), ("gene_names_no_header.tsv.gz", None)):
        values = []
        with identity.regular_input(root / name) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            for line in gz:
                values.append(line.decode().rstrip("\n"))
        if target is not None:
            cells = values
        else:
            genes = values
    if len(cells) != expected["cells"] or len(genes) != expected["genes"] \
            or len(set(cells)) != len(cells) or len(set(genes)) != len(genes):
        raise QCError("original identifier axes differ from hash-bound metadata")
    mito = bytearray(g.upper().startswith("MT-") for g in genes)
    if not any(mito):
        raise QCError("missing mitochondrial gene axis; cannot substitute zero MT fractions")
    author, _, _ = identity.read_bound_json(root / "identity.json", cap=8 << 20)
    compact = []
    with identity.regular_input(root / "author_cell_metadata_prepared.tsv.gz") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
        for position, (columns, row) in enumerate(identity._rows(gz,
                ["cell", "cell_index_one_based", "line", "day", "physical_pool", "specimen", "batch"],
                max_row_bytes=spec["budgets"]["max_row_bytes"])):
            if set(columns) & set(QC_EXTRA) or position >= len(cells) or row["cell"] != cells[position] \
                    or row["cell_index_one_based"] != str(position + 1):
                raise QCError("prepared metadata column collision or original cell index order mismatch")
            compact.append((sys.intern(row["line"]), int(row["day"]), sys.intern(row["physical_pool"]),
                            sys.intern(row["specimen"]), sys.intern(row["batch"])))
    if len(compact) != len(cells):
        raise QCError("metadata does not cover complete original cell axis")
    return cells, genes, mito, compact, author


def exact_qc(total, detected, mitochondrial, min_genes, mito_denominator):
    # Only exact 1/10 and 1/5 comparisons; no binary float cut-point ambiguity.
    return total > 0 and detected >= min_genes and mitochondrial * mito_denominator <= total


def groups_from_counts(counter):
    return [dict(line=k[0], day=k[1], physical_pool=k[2], specimen=k[3], batch=k[4], cells=v)
            for k, v in sorted(counter.items())]


def post_qc_plan(author, groups):
    split = identity.plan_pool_purged(dict(author, groups=groups))
    for fold in split["line_holdout_folds"]:
        for pair in fold["adjacent_pairs"]:
            pair["post_QC_pool_purged_feasible"] = pair["metadata_only_candidate"]
            pair["after_QC_eligible"] = "pool_purged_feasible" if pair["metadata_only_candidate"] else "ineligible_min50_or_training_lines"
    split["kind"] = "post_QC_pool_purged_feasibility_not_training_approval"
    split["expression_QC_performed"] = True
    return split


def build_csr(records, out, cells, genes, mito, compact, author, spec, ledger):
    n = len(cells)
    totals, detected, mt_counts = array("Q", [0]) * n, array("I", [0]) * n, array("Q", [0]) * n
    keep, variants = bytearray(n), {(g, d): Counter() for g in (200, 500) for d in (10, 5)}
    row_data, row_indices = bytearray(), bytearray()
    current, entries, retained_rows, nnz, total_kept = 0, 0, 0, 0, 0
    all_groups = Counter()
    with fresh_binary(out / "counts.data.i64le.bin", ledger) as data, \
            fresh_binary(out / "counts.indices.u32le.bin", ledger) as indices, \
            fresh_binary(out / "counts.indptr.u64le.bin", ledger) as indptr, \
            fresh_binary(out / "source_cell_indices.u64le.bin", ledger) as source_rows:
        indptr.write(POINTER.pack(0))

        def finish(cell):
            nonlocal retained_rows, nnz, total_kept
            group = compact[cell]
            for (minimum, denominator), counter in variants.items():
                if exact_qc(totals[cell], detected[cell], mt_counts[cell], minimum, denominator):
                    counter[group] += 1
            if exact_qc(totals[cell], detected[cell], mt_counts[cell], 200, 5):
                keep[cell] = 1
                retained_rows += 1
                nnz += detected[cell]
                total_kept += totals[cell]
                all_groups[group] += 1
                data.write(row_data)
                indices.write(row_indices)
                indptr.write(POINTER.pack(nnz))
                source_rows.write(POINTER.pack(cell))
            row_data.clear()
            row_indices.clear()

        for cell, gene, value in records:
            if cell < current or not 0 <= cell < n or not 0 <= gene < len(genes):
                raise QCError("merged coordinates are out of original axes/order")
            while current < cell:
                finish(current)
                current += 1
            entries += 1
            if value:
                total = totals[cell] + value
                if total > MAX_COUNT:
                    raise QCError("exact per-cell library exceeds signed-int64 artifact contract")
                totals[cell] = total
                detected[cell] += 1
                mt_counts[cell] += value if mito[gene] else 0
                row_data.extend(VALUE.pack(value))
                row_indices.extend(INDEX.pack(gene))
        while current < n:
            finish(current)
            current += 1
    if entries != spec["expected"]["entries"]:
        raise QCError("final merge record count differs from accepted MatrixMarket inventory")
    with fresh_binary(out / "keep_primary.u8.bin", ledger) as stream:
        stream.write(keep)
    with fresh_gzip_text(out / "cell-QC-all-original-rows.tsv.gz", ledger) as sink:
        writer = csv.writer(sink, delimiter="\t", lineterminator="\n")
        writer.writerow(QC_COLUMNS)
        for i, name in enumerate(cells):
            writer.writerow([i+1, name, totals[i], detected[i], mt_counts[i],
                             f"{mt_counts[i]}/{totals[i]}" if totals[i] else "undefined_zero_library", bool(keep[i])])
    split = post_qc_plan(author, groups_from_counts(all_groups))
    write_json(out / "post-QC-pool-purged-plan.json", split, ledger)
    sensitivity = []
    for (minimum, denominator), counter in variants.items():
        sensitivity.append(dict(min_detected_genes=minimum, max_mito_fraction_exact=f"1/{denominator}",
            retained_cells=sum(counter.values()), line_day_min50_grid=identity.min50_grid(groups_from_counts(counter),
                author["individuals"], author["physical_days"])))
    write_json(out / "QC-sensitivity-min50.json", dict(grids=sensitivity, biological_units="source_lines_not_donors"), ledger)
    return dict(retained_rows=retained_rows, nonzero_entries=nnz, retained_count_sum_exact=total_kept,
        raw_library_count_sum_exact=sum(totals), zero_library_cells=totals.count(0), sensitivity_grids=sensitivity,
        feasible_pool_purged_pairs=sum(p["post_QC_pool_purged_feasible"] for f in split["line_holdout_folds"] for p in f["adjacent_pairs"])), \
        (keep, totals, detected, mt_counts)


def write_retained_identity(metadata_path, out, cells, genes, metrics, spec, ledger):
    keep, totals, detected, mt_counts = metrics
    root = Path(metadata_path).parent
    with fresh_gzip_text(out / "cell_names_no_header.tsv.gz", ledger) as sink:
        for i, cell in enumerate(cells):
            if keep[i]:
                sink.write(cell + "\n")
    with fresh_gzip_text(out / "gene_names_no_header.tsv.gz", ledger) as sink:
        for gene in genes:
            sink.write(gene + "\n")
    with identity.regular_input(root / "author_cell_metadata_prepared.tsv.gz") as raw, \
            gzip.GzipFile(fileobj=raw, mode="rb") as source, \
            fresh_gzip_text(out / "author_cell_metadata_QC.tsv.gz", ledger) as sink:
        writer, header, seen = csv.writer(sink, delimiter="\t", lineterminator="\n"), None, 0
        for i, (columns, row) in enumerate(identity._rows(source, ["cell"], max_row_bytes=spec["budgets"]["max_row_bytes"])):
            if header is None:
                header = columns
                writer.writerow(header + QC_EXTRA)
            if i >= len(cells) or row["cell"] != cells[i]:
                raise QCError("retained metadata source order changed")
            seen += 1
            if keep[i]:
                writer.writerow([row[h] for h in header] + [totals[i], detected[i], mt_counts[i], f"{mt_counts[i]}/{totals[i]}"])
        if seen != len(cells):
            raise QCError("retained metadata source axis truncated")


def iter_values(path, layout, chunk_records=8192):
    with identity.regular_input(path) as raw:
        while block := raw.read(layout.size * chunk_records):
            if len(block) % layout.size:
                raise QCError("partial CSR binary value")
            for value, in layout.iter_unpack(block):
                yield value


def verify_csr(out, csr, cells, genes, mito, metrics, qc):
    """Read the complete written artifact with bounded buffers; no matrix load."""
    keep, totals, detected, mt_counts = metrics
    expected_files = {"counts.data.i64le.bin", "counts.indices.u32le.bin", "counts.indptr.u64le.bin", "source_cell_indices.u64le.bin"}
    if len(keep) != len(cells) or any(len(v) != len(cells) for v in (totals, detected, mt_counts)) \
            or sum(keep) != qc["retained_rows"]:
        raise QCError("CSR QC selector/metric original axis mismatch")
    if csr.get("schema") != CSR_SCHEMA or csr.get("shape") != [qc["retained_rows"], len(genes)] \
            or csr.get("original_shape") != [len(cells), len(genes)] or csr.get("byte_order") != "little" \
            or csr.get("orientation") != "cells_by_genes" or csr.get("index_base") != 0 \
            or csr.get("explicit_zero_records_removed") is not True \
            or csr.get("duplicate_coordinates") != "rejected_not_summed" \
            or csr.get("H5AD_schema_claimed") is not False \
            or csr.get("data_dtype") != "int64" or csr.get("indices_dtype") != "uint32" or csr.get("indptr_dtype") != "uint64" \
            or len(csr.get("files", [])) != 4 or {item["file"] for item in csr["files"]} != expected_files:
        raise QCError("unsupported or incomplete exact binary CSR schema")
    for item in csr["files"]:
        path = identity.safe_path(out / item["file"], file=True)
        if path.stat().st_size != item["bytes"] or identity.sha256_file(path) != item["sha256"]:
            raise QCError("CSR artifact hash/size binding mismatch")
    pointers = iter_values(out / "counts.indptr.u64le.bin", POINTER)
    values = iter_values(out / "counts.data.i64le.bin", VALUE)
    indices = iter_values(out / "counts.indices.u32le.bin", INDEX)
    source_rows = iter_values(out / "source_cell_indices.u64le.bin", POINTER)
    if next(pointers, None) != 0:
        raise QCError("CSR indptr must start at zero")
    previous_pointer, previous_source, rows, count_sum = 0, -1, 0, 0
    for source in source_rows:
        top = next(pointers, None)
        if top is None or top < previous_pointer or not previous_source < source < len(cells) or not keep[source]:
            raise QCError("CSR offsets/original cell selection order mismatch")
        total, mitochondrial, previous_gene, detected_genes = 0, 0, -1, 0
        for _ in range(top-previous_pointer):
            value, gene = next(values, None), next(indices, None)
            if value is None or gene is None or not 0 < value <= MAX_COUNT or not previous_gene < gene < len(genes):
                raise QCError("CSR exact values, gene axis order or duplicate roundtrip failed")
            total += value
            mitochondrial += value if mito[gene] else 0
            detected_genes += 1
            previous_gene = gene
        if (total, detected_genes, mitochondrial) != (totals[source], detected[source], mt_counts[source]):
            raise QCError("CSR per-cell count/detection/MT conservation failed")
        count_sum += total
        previous_pointer, previous_source = top, source
        rows += 1
    if next(pointers, None) is not None or next(values, None) is not None or next(indices, None) is not None \
            or (rows, previous_pointer, count_sum) != (qc["retained_rows"], qc["nonzero_entries"], qc["retained_count_sum_exact"]):
        raise QCError("CSR total offsets/nnz/count conservation failed")
    with identity.regular_input(out / "cell_names_no_header.tsv.gz") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
        for i, line in zip((i for i, bit in enumerate(keep) if bit), gz, strict=True):
            if line.decode().rstrip("\n") != cells[i]:
                raise QCError("retained cell axis roundtrip order failed")
    with identity.regular_input(out / "gene_names_no_header.tsv.gz") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
        for gene, line in zip(genes, gz, strict=True):
            if line.decode().rstrip("\n") != gene:
                raise QCError("original gene axis roundtrip failed")
    with identity.regular_input(out / "author_cell_metadata_QC.tsv.gz") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz, \
            identity.regular_input(out / "author_cell_metadata_original.tsv.gz") as original, \
            gzip.GzipFile(fileobj=original, mode="rb") as original_gz:
        reader = csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8"), delimiter="\t")
        originals = csv.DictReader(io.TextIOWrapper(original_gz, encoding="utf-8"), delimiter="\t")
        if reader.fieldnames != originals.fieldnames + QC_EXTRA:
            raise QCError("retained metadata original column-axis roundtrip failed")
        original_count = 0
        for i, raw_row in enumerate(originals):
            if i >= len(cells) or raw_row["cell"] != cells[i]:
                raise QCError("original raw metadata row axis changed")
            original_count += 1
            if not keep[i]:
                continue
            row = next(reader, None)
            if row is None or any(row[name] != raw_row[name] for name in originals.fieldnames) \
                    or row["cell_index_one_based"] != str(i+1) \
                    or (int(row["qc_library_count_exact"]), int(row["qc_detected_genes"]), int(row["qc_mito_count_exact"])) \
                    != (totals[i], detected[i], mt_counts[i]) \
                    or row["qc_mito_fraction_exact"] != f"{mt_counts[i]}/{totals[i]}":
                raise QCError("retained raw identity/all author columns/index/QC roundtrip failed")
        if original_count != len(cells) or next(reader, None) is not None:
            raise QCError("unexpected trailing retained metadata row")
    return dict(passed=True, rows=rows, nonzero_entries=previous_pointer, retained_count_sum_exact=count_sum,
        exact_count_precision_preserved=True, full_artifact_values_reread=True, original_axes_preserved=True,
        full_matrix_loaded=False, h5ad_written=False)


def verify_artifact(out, expected_report_sha256):
    """Standalone portable replay of hashes and streamed CSR/identity roundtrip."""
    out = identity.safe_path(out, directory=True)
    report, _, _ = identity.read_bound_json(out / "report.json", expected_report_sha256, cap=16 << 20)
    if report.get("schema") != "ortho.gse175634.duplicate-qc-csr.v1" \
            or report.get("status") != "duplicate_QC_CSR_prepared_not_scientific_admission" \
            or any(report.get(flag) is not False for flag in identity.NONAPPROVALS):
        raise QCError("successful nonapproval QC report required")
    required = {"DOWNLOAD-ACCEPTANCE.json", "preparation-spec.json", "metadata-preparation-report.json", "identity-original.json",
        "author_cell_metadata_original.tsv.gz", "cell_names_original.tsv.gz", "counts.data.i64le.bin",
        "counts.indices.u32le.bin", "counts.indptr.u64le.bin", "source_cell_indices.u64le.bin", "keep_primary.u8.bin",
        "cell-QC-all-original-rows.tsv.gz", "post-QC-pool-purged-plan.json", "QC-sensitivity-min50.json",
        "cell_names_no_header.tsv.gz", "gene_names_no_header.tsv.gz", "author_cell_metadata_QC.tsv.gz", "csr.json", "roundtrip.json"}
    required |= {"raw/" + name for name in identity.INVENTORY[:-1]}
    manifest = report.get("output_files", [])
    paths = [item["file"] for item in manifest]
    if len(paths) != len(set(paths)) or not required <= set(paths) \
            or any(not re.fullmatch(r"scratch/generation-[0-9]{3}\.json", name) for name in set(paths) - required):
        raise QCError("portable artifact manifest closure incomplete or conflicting")
    actual = set()
    for path in out.rglob("*"):
        identity.safe_path(path)
        if path.is_file():
            actual.add(str(path.relative_to(out)))
    if actual != set(paths) | {"report.json"}:
        raise QCError("portable artifact inventory contains unmanifested or missing files")
    for item in manifest:
        name = Path(item["file"])
        if name.is_absolute() or ".." in name.parts:
            raise QCError("invalid artifact manifest path")
        path = identity.safe_path(out / name, file=True)
        if path.stat().st_size != item["bytes"] or identity.sha256_file(path) != item["sha256"]:
            raise QCError("portable artifact manifest hash/size binding mismatch")
    receipt, _, _ = identity.load_acceptance(out / "DOWNLOAD-ACCEPTANCE.json", report["acceptance_sha256"])
    spec, _, _ = identity.load_spec(out / "preparation-spec.json", report["spec_sha256"])
    metadata, _, _ = identity.read_bound_json(out / "metadata-preparation-report.json", report["metadata_preparation_sha256"], cap=8 << 20)
    author, _, _ = identity.read_bound_json(out / "identity-original.json", cap=8 << 20)
    original_entries = {item["file"]: item for item in metadata.get("output_files", [])}
    if metadata.get("acceptance_sha256") != report["acceptance_sha256"] or metadata.get("spec_sha256") != report["spec_sha256"] \
            or identity.sha256_file(out / "identity-original.json") != original_entries["identity.json"]["sha256"] \
            or identity.sha256_file(out / "author_cell_metadata_original.tsv.gz") != original_entries["author_cell_metadata_prepared.tsv.gz"]["sha256"] \
            or identity.sha256_file(out / "cell_names_original.tsv.gz") != original_entries["cell_names_no_header.tsv.gz"]["sha256"] \
            or identity.sha256_file(out / "gene_names_no_header.tsv.gz") != original_entries["gene_names_no_header.tsv.gz"]["sha256"]:
        raise QCError("portable original metadata/axes lost preparation hash binding")
    for item in receipt["files"]:
        if item["file"] != identity.MATRIX and identity.sha256_file(out / "raw" / item["file"]) != item["sha256"]:
            raise QCError("portable raw metadata lost acceptance hash binding")
    csr, _, _ = identity.read_bound_json(out / "csr.json")
    def names(path):
        with identity.regular_input(path) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            return [line.decode().rstrip("\n") for line in gz]
    cells, genes = names(out / "cell_names_original.tsv.gz"), names(out / "gene_names_no_header.tsv.gz")
    mito = bytearray(g.upper().startswith("MT-") for g in genes)
    with identity.regular_input(out / "keep_primary.u8.bin") as raw:
        keep = raw.read(spec["expected"]["cells"] + 1)
    if len(cells) != spec["expected"]["cells"] or len(genes) != spec["expected"]["genes"] or not any(mito) \
            or len(keep) != len(cells) or any(bit not in (0, 1) for bit in keep):
        raise QCError("original-axis primary QC selector invalid")
    totals, detected, mt_counts = array("Q"), array("I"), array("Q")
    with identity.regular_input(out / "cell-QC-all-original-rows.tsv.gz") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
        for i, (columns, row) in enumerate(identity._rows(gz, QC_COLUMNS, max_row_bytes=spec["budgets"]["max_row_bytes"])):
            if columns != QC_COLUMNS:
                raise QCError("portable QC table column schema changed")
            if i >= len(cells) or row["cell"] != cells[i] or row["cell_index_one_based"] != str(i+1):
                raise QCError("portable QC table original-axis binding mismatch")
            total, detection, mt = (int(row[key]) for key in ("library_count_exact", "detected_genes", "mito_count_exact"))
            fraction = f"{mt}/{total}" if total else "undefined_zero_library"
            bit = exact_qc(total, detection, mt, 200, 5)
            if not 0 <= mt <= total <= MAX_COUNT or not 0 <= detection <= len(genes) \
                    or bool(keep[i]) != bit or row["primary_keep"] != str(bit) or row["mito_fraction_exact"] != fraction:
                raise QCError("portable exact primary QC/MT fraction/selector semantics changed")
            totals.append(total)
            detected.append(detection)
            mt_counts.append(mt)
    if len(totals) != len(cells):
        raise QCError("portable QC table incomplete")
    if sum(totals) != report["matrix_integrity"]["stored_count_sum_exact"] \
            or sum(detected) != report["matrix_integrity"]["positive_records"] \
            or totals.count(0) != report["QC"]["zero_library_cells"]:
        raise QCError("portable all-original-cell count/detection conservation failed")
    groups, variants = Counter(), {(g, d): Counter() for g in (200, 500) for d in (10, 5)}
    with identity.regular_input(out / "author_cell_metadata_original.tsv.gz") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
        observed = 0
        for i, (_, row) in enumerate(identity._rows(gz, ["cell", "line", "day", "physical_pool", "specimen", "batch"],
                max_row_bytes=spec["budgets"]["max_row_bytes"])):
            if i >= len(cells) or row["cell"] != cells[i]:
                raise QCError("portable original identity order changed")
            observed += 1
            group = (row["line"], int(row["day"]), row["physical_pool"], row["specimen"], row["batch"])
            if keep[i]:
                groups[group] += 1
            for (minimum, denominator), counter in variants.items():
                if exact_qc(totals[i], detected[i], mt_counts[i], minimum, denominator):
                    counter[group] += 1
        if observed != len(cells):
            raise QCError("portable original identity incomplete")
    plan, _, _ = identity.read_bound_json(out / "post-QC-pool-purged-plan.json", cap=16 << 20)
    if plan != post_qc_plan(author, groups_from_counts(groups)):
        raise QCError("portable post-QC physical-pool purge/min50 replay changed")
    sensitivity, _, _ = identity.read_bound_json(out / "QC-sensitivity-min50.json", cap=8 << 20)
    expected_grids = [dict(min_detected_genes=g, max_mito_fraction_exact=f"1/{d}", retained_cells=sum(counter.values()),
        line_day_min50_grid=identity.min50_grid(groups_from_counts(counter), author["individuals"], author["physical_days"]))
        for (g, d), counter in variants.items()]
    if sensitivity.get("grids") != expected_grids or report["QC"]["sensitivity_grids"] != expected_grids \
            or report["QC"]["feasible_pool_purged_pairs"] != sum(p["post_QC_pool_purged_feasible"]
                for f in plan["line_holdout_folds"] for p in f["adjacent_pairs"]):
        raise QCError("portable post-QC sensitivity/min50 feasibility replay changed")
    return verify_csr(out, csr, cells, genes, mito, (keep, totals, detected, mt_counts), report["QC"])


def prepare_qc(*, data_root, download_acceptance, expected_acceptance_sha256, out, spec_path,
               metadata_preparation, expected_metadata_sha256, expected_spec_sha256=None):
    receipt, receipt_sha, receipt_body = identity.load_acceptance(download_acceptance, expected_acceptance_sha256)
    spec, spec_sha, spec_body = identity.load_spec(spec_path, expected_spec_sha256)
    metadata, metadata_sha = identity.validate_metadata_report(metadata_preparation, expected_metadata_sha256, receipt_sha, spec_sha)
    cfg, estimate = policy(spec), estimate_resources(spec, receipt)
    root, out = identity.safe_path(data_root, directory=True), identity.safe_path(out, fresh=True)
    md_root = identity.safe_path(Path(metadata_preparation).parent, directory=True)
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in (root, md_root)):
        raise QCError("fresh QC outputs must be disjoint from payload and prepared identity inputs")
    preflight = resource_preflight(out.parent, spec, estimate)
    if preflight["status"] != "passed":
        raise QCError(f"QC resource preflight blocked: {preflight['checks']}")
    _, controls = identity.validate_inventory(root, matrix_required=True, receipt=receipt,
        acceptance_path=download_acceptance, acceptance_sha=receipt_sha)
    cells, genes, mito, compact, author = load_axes_and_identity(metadata_preparation, metadata, spec)
    selected = {row["file"]: row for row in receipt["files"]}
    # Recompute the five small files on the rebound root; hashes alone from a
    # historical prepared directory never validate a new payload location.
    for name in identity.INVENTORY[:-1]:
        identity.verify_metadata(root / name, selected[name], spec["budgets"]["metadata_decoded_caps"][name])
    out.mkdir(exist_ok=False)
    ledger, started = StorageLedger(estimate["planned_increment_bytes"], out), time.monotonic()
    source_hash, cli_hash = identity.sha256_file(Path(__file__)), identity.sha256_file(Path(__file__).parents[2] / "scripts/prepare_gse175634_qc.py")
    try:
        for name, body in (("DOWNLOAD-ACCEPTANCE.json", receipt_body), ("preparation-spec.json", spec_body)):
            with fresh_binary(out / name, ledger) as sink:
                sink.write(body)
        (out / "raw").mkdir()
        for name in identity.INVENTORY[:-1]:
            copy_exact(md_root / "raw" / name, out / "raw" / name, selected[name]["sha256"], ledger)
        entries = {item["file"]: item for item in metadata["output_files"]}
        copy_exact(metadata_preparation, out / "metadata-preparation-report.json", metadata_sha, ledger)
        copy_exact(md_root / "identity.json", out / "identity-original.json", entries["identity.json"]["sha256"], ledger)
        for source, destination in (("author_cell_metadata_prepared.tsv.gz", "author_cell_metadata_original.tsv.gz"),
                                    ("cell_names_no_header.tsv.gz", "cell_names_original.tsv.gz")):
            copy_exact(md_root / source, out / destination, entries[source]["sha256"], ledger)
        stats = dict(stored_count_sum_exact=0, explicit_zero_records=0, positive_records=0, max_stored_count_exact=0)
        runs, sorting = external_runs(root / identity.MATRIX, selected[identity.MATRIX], spec, out, ledger, stats)
        qc, metrics = build_csr(merged_records(runs, cfg), out, cells, genes, mito, compact, author, spec, ledger)
        if qc["raw_library_count_sum_exact"] != stats["stored_count_sum_exact"]:
            raise QCError("full stored-count/library conservation failed after duplicate rejection")
        write_retained_identity(metadata_preparation, out, cells, genes, metrics, spec, ledger)
        csr_files = [dict(file=name, bytes=(out/name).stat().st_size, sha256=identity.sha256_file(out/name)) for name in
                     ("counts.data.i64le.bin", "counts.indices.u32le.bin", "counts.indptr.u64le.bin", "source_cell_indices.u64le.bin")]
        csr = dict(schema=CSR_SCHEMA, shape=[qc["retained_rows"], len(genes)], original_shape=[len(cells), len(genes)],
            orientation="cells_by_genes", byte_order="little", data_dtype="int64", indices_dtype="uint32", indptr_dtype="uint64",
            index_base=0, explicit_zero_records_removed=True, duplicate_coordinates="rejected_not_summed", files=csr_files,
            cell_names="cell_names_no_header.tsv.gz", gene_names="gene_names_no_header.tsv.gz",
            original_cell_selection="source_cell_indices.u64le.bin", original_cell_axis="cell_names_original.tsv.gz",
            metadata="author_cell_metadata_QC.tsv.gz", H5AD_schema_claimed=False)
        roundtrip = verify_csr(out, csr, cells, genes, mito, metrics, qc)
        write_json(out / "csr.json", csr, ledger)
        write_json(out / "roundtrip.json", roundtrip, ledger)
        for run in runs:
            ledger.remove_run(run)
        if source_hash != identity.sha256_file(Path(__file__)) or cli_hash != identity.sha256_file(Path(__file__).parents[2] / "scripts/prepare_gse175634_qc.py"):
            raise QCError("QC implementation changed during processing")
        report = dict(schema="ortho.gse175634.duplicate-qc-csr.v1", status="duplicate_QC_CSR_prepared_not_scientific_admission",
            acceptance_sha256=receipt_sha, spec_sha256=spec_sha, metadata_preparation_sha256=metadata_sha,
            source_sha256=source_hash, cli_sha256=cli_hash, data_root=str(root), resource_preflight=preflight,
            source_axis_and_raw_identity_hash_bound=True, validated_acquisition_controls=controls,
            matrix_integrity=stats, external_sort=sorting, QC=qc, roundtrip=roundtrip,
            duplicate_coordinate_validation=dict(performed=True, method="bounded_external_sort_cell_gene_adjacent_rejection",
                stored_records_checked=stats["stored_entries"], duplicate_records=0, explicit_zero_duplicates_also_rejected=True),
            biological_count_semantics_confirmed=False, umi_semantics_confirmed=False, publisher_checksum_verified=False,
            standardized_AnnData_H5AD_written=False, binary_CSR_artifact_written=True,
            measured_scratch_output_peak_bytes=ledger.peak, verified_owned_scratch_bytes_removed=ledger.removed_run_bytes,
            runtime_resources=identity.runtime_snapshot(stage="duplicate_QC_binary_CSR"), elapsed_seconds=time.monotonic()-started,
            target_real_matrix_processing_performed=receipt_sha == spec.get("current_acceptance_sha256_reference"),
            remaining_admission_gates=["independent_author_count_semantics_review", "orig_ident_conflict_review",
                "target_scale_runtime_resource_and_roundtrip_acceptance", "consumer_binary_CSR_integration_or_official_H5AD_conversion",
                "explicit_Stage_A_and_Stage_B_authority"], **identity.NONAPPROVALS)
        report["output_files"] = identity._manifest(out)
        write_json(out / "report.json", report, ledger)
        return report
    except Exception as error:
        identity.write_json_fresh(out / "FAILED.json", dict(status="failed", error=str(error),
            acceptance_sha256=receipt_sha, spec_sha256=spec_sha, metadata_preparation_sha256=metadata_sha,
            generated_scratch_and_partial_outputs_retained=True, **identity.NONAPPROVALS))
        raise
