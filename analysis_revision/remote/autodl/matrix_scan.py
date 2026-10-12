"""Bounded, CPU-only technical scan of a local MatrixMarket gzip, not Stage A.

Only coordinate integer/real general matrices are supported. Stored values are
parsed exactly, never through binary floating point. Neither coordinates nor the
matrix are retained; duplicate-coordinate and biological semantics are untested.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import zlib


_INTEGER = re.compile(rb"[+-]?[0-9]+")
_REAL = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eEdD][+-]?[0-9]+)?")
# The fast lane admits only three nonnegative integer tokens per line, each
# <=18 digits: np.int64 parsing cannot overflow or lose >2**53 precision here.
# Longer tokens, negatives, comments and unusual syntax take the exact fallback.
_INTEGER_BATCH = re.compile(
    rb"(?:[ \t\r\v\f]*\+?[0-9]{1,18}[ \t\r\v\f]+\+?[0-9]{1,18}"
    rb"[ \t\r\v\f]+\+?[0-9]{1,18}[ \t\r\v\f]*\n)+")


class MatrixScanError(ValueError):
    """Rejected input or exhausted budget, with partial evidence when available."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


def _validate_options(options, expected):
    for name, value in options.items():
        floor = 0 if name in {"max_genes", "max_cells", "max_entries", "axis_support_max_bytes"} else 1
        if type(value) is not int or value < floor:
            raise MatrixScanError(f"{name} must be an integer >= {floor}")
    # Keep configurable working buffers/precision bounded, including hostile
    # exponents such as 1e1000000000 and JSON's integer-to-string conversion.
    for name, ceiling in (("chunk_bytes", 16 << 20), ("max_line_bytes", 4096),
                          ("max_count_digits", 1024), ("axis_support_max_bytes", 64 << 20)):
        if options[name] > ceiling:
            raise MatrixScanError(f"{name} exceeds implementation budget ceiling {ceiling}")
    for name, value in expected.items():
        if value is not None and (type(value) is not int or value < 0):
            raise MatrixScanError(f"expected {name} must be a nonnegative integer")


def _integer(token, label):
    if not _INTEGER.fullmatch(token):
        raise MatrixScanError(f"unparseable {label}: expected an integer token")
    return int(token)


def _count(token, field, max_digits):
    if field == "integer":
        if not _INTEGER.fullmatch(token):
            raise MatrixScanError("unparseable integer count")
        significant = token.lstrip(b"+-").lstrip(b"0")
        if token.startswith(b"-") and significant:
            raise MatrixScanError("negative count")
        if len(significant) > max_digits:
            raise MatrixScanError("count precision budget exceeded")
        return int(token)
    if not _REAL.fullmatch(token):
        raise MatrixScanError("unparseable or non-finite real count")
    try:
        number = Decimal(token.decode("ascii").replace("d", "e").replace("D", "E"))
    except InvalidOperation as error:
        raise MatrixScanError("unparseable real count") from error
    if not number.is_finite():
        raise MatrixScanError("non-finite count")
    if number.is_zero():
        return 0
    if number.is_signed():
        raise MatrixScanError("negative count")
    if number.adjusted() >= max_digits:
        raise MatrixScanError("count precision budget exceeded")
    if number != number.to_integral_value():
        raise MatrixScanError("non-integral real count (exact integrality required)")
    return int(number)


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _line_batches(decoded, report, options, progress):
    """Read fixed-size blocks, retaining at most one bounded incomplete line."""
    pending = b""
    line_number = 0
    while True:
        remaining = options["max_decoded_bytes"] - report["decompressed_bytes"]
        block = decoded.read(min(options["chunk_bytes"], remaining + 1))
        if not block:
            report["gzip_crc32_isize_validated"] = True
            break
        report["decompressed_bytes"] += len(block)
        if report["decompressed_bytes"] > options["max_decoded_bytes"]:
            raise MatrixScanError("decoded byte budget exceeded")
        # The list contains only this block, never the complete gzip/matrix.
        lines = (pending + block).split(b"\n")
        pending = lines.pop()
        first_line = line_number + 1
        for offset, line in enumerate(lines):
            if len(line) > options["max_line_bytes"]:
                raise MatrixScanError(f"line {first_line + offset}: line byte budget exceeded")
        line_number += len(lines)
        if len(pending) > options["max_line_bytes"]:
            raise MatrixScanError(f"line {line_number + 1}: line byte budget exceeded")
        yield first_line, lines
        if progress is not None:
            progress(report)
    if pending:
        yield line_number + 1, [pending]


def _scan_entries(lines, first_line, report, max_count_digits, genes_seen, cells_seen):
    if report["field"] == "integer" and len(lines) >= 64:
        text = b"\n".join(lines) + b"\n"
        if _INTEGER_BATCH.fullmatch(text):
            import numpy as np

            count = len(lines)
            if count > report["declared_entries"] - report["observed_entries"]:
                raise MatrixScanError("trailing entries exceed declared count")
            data = np.fromstring(text, dtype=np.int64, sep=" ").reshape(count, 3)
            invalid = (data[:, 0] == 0) | (data[:, 1] == 0)
            # Every fast-lane coordinate is <10**18; a larger dimension cannot
            # be exceeded. Avoid coercing arbitrary Python dimensions to float.
            if report["genes"] < 10**18:
                invalid |= data[:, 0] > report["genes"]
            if report["cells"] < 10**18:
                invalid |= data[:, 1] > report["cells"]
            if invalid.any():
                offset = int(np.flatnonzero(invalid)[0])
                raise MatrixScanError(f"line {first_line + offset}: out-of-bounds 1-based coordinate")
            values = data[:, 2]
            maximum = int(values.max())
            if max_count_digits < 18 and maximum >= 10**max_count_digits:
                raise MatrixScanError("count precision budget exceeded")
            # Never silently overflow a NumPy reduction. Common small counts
            # have an exact uint64 reduction; large values use Python integers.
            subtotal = (int(values.sum(dtype=np.uint64)) if maximum * count < 2**64
                        else sum(map(int, values)))
            positive = values > 0
            nonzero = int(np.count_nonzero(positive))
            if genes_seen is not None:
                np.frombuffer(genes_seen, dtype=np.uint8)[data[positive, 0] - 1] = 1
                np.frombuffer(cells_seen, dtype=np.uint8)[data[positive, 1] - 1] = 1
            report["observed_entries"] += count
            report["integer_batched_entries"] += count
            report["max_count"] = max(report["max_count"], maximum)
            report["count_sum"] += subtotal
            report["nonzero_entries"] += nonzero
            report["explicit_zero_entries"] += count - nonzero
            return
    for offset, line in enumerate(lines):
        line_number = first_line + offset
        tokens = line.split()
        if not tokens or line.lstrip().startswith(b"%"):
            continue
        if report["observed_entries"] >= report["declared_entries"]:
            raise MatrixScanError(f"line {line_number}: trailing entries exceed declared count")
        if len(tokens) != 3:
            raise MatrixScanError(f"line {line_number}: entry requires exactly three columns")
        try:
            gene, cell = _integer(tokens[0], "gene coordinate"), _integer(tokens[1], "cell coordinate")
            if not 1 <= gene <= report["genes"] or not 1 <= cell <= report["cells"]:
                raise MatrixScanError("out-of-bounds 1-based coordinate")
            value = _count(tokens[2], report["field"], max_count_digits)
        except MatrixScanError as error:
            raise MatrixScanError(f"line {line_number}: {error}") from error
        report["observed_entries"] += 1
        report["scalar_exact_entries"] += 1
        report["max_count"] = max(report["max_count"], value)
        report["count_sum"] += value
        if value:
            report["nonzero_entries"] += 1
            if genes_seen is not None:
                genes_seen[gene - 1] = cells_seen[cell - 1] = 1
        else:
            report["explicit_zero_entries"] += 1


def scan_matrix(source, *, max_decoded_bytes, max_genes, max_cells, max_entries,
                expected_genes=None, expected_cells=None, expected_entries=None,
                chunk_bytes=1 << 20, max_line_bytes=4096, max_count_digits=128,
                axis_support_max_bytes=8 << 20, progress=None):
    """Return exact technical evidence, or raise MatrixScanError with a report.

    Full compressed SHA256 is streamed in a first pass on the same descriptor;
    the second bounded pass verifies every gzip member's CRC/ISIZE through EOF.
    On failure, byte/count evidence is partial and never approval. Positive axis
    supports use one byte per gene/cell only if their combined size fits budget.
    Max/sum cover stored entries, not duplicate-collapsed biological counts.
    """
    options = dict(max_decoded_bytes=max_decoded_bytes, max_genes=max_genes,
                   max_cells=max_cells, max_entries=max_entries, chunk_bytes=chunk_bytes,
                   max_line_bytes=max_line_bytes, max_count_digits=max_count_digits,
                   axis_support_max_bytes=axis_support_max_bytes)
    expected = dict(genes=expected_genes, cells=expected_cells, entries=expected_entries)
    _validate_options(options, expected)
    source = Path(source)
    report = dict(schema_version=1, kind="autodl_matrixmarket_gzip_integrity",
                  source=str(source.absolute()), status="scanning", cpu_only=True,
                  network_started=False, raw_sequencing_supported=False,
                  budgets=options, expected=expected, compressed_bytes=0,
                  compressed_sha256=None, compressed_hash_complete=False,
                  publisher_checksum_verified=False, umi_semantics_confirmed=False,
                  decompressed_bytes=0, gzip_crc32_isize_validated=False,
                  source_unchanged=False, field=None, genes=None, cells=None,
                  declared_entries=None, observed_entries=0, nonzero_entries=0,
                  explicit_zero_entries=0, max_count=0, count_sum=0,
                  integer_batched_entries=0, scalar_exact_entries=0,
                  axis_support={"performed": False, "reason": "header not validated"},
                  duplicate_coordinate_validation={"performed": False,
                      "reason": "bounded scan does not retain coordinate pairs; duplicates are not checked"},
                  technical_count_like_integrity=False,
                  biological_count_semantics_confirmed=False, scientific_approved=False,
                  stage_a_approved=False, training_approved=False,
                  count_semantics="Exact nonnegative integral stored values only; sums include any duplicate records. "
                                  "No UMI/read provenance, biological interpretation, or scientific approval.")
    genes_seen = cells_seen = None
    try:
        if not source.name.lower().endswith(".mtx.gz"):
            raise MatrixScanError("input must be a local .mtx.gz; raw sequencing is unsupported")
        if not source.is_file():
            raise MatrixScanError("source must be a regular local file")
        with source.open("rb") as raw:
            before = os.fstat(raw.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise MatrixScanError("source must be a regular local file")
            if progress is not None:
                progress(report)
            digest = hashlib.sha256()
            while True:
                block = raw.read(chunk_bytes)
                if not block:
                    break
                digest.update(block)
                report["compressed_bytes"] += len(block)
                if report["compressed_bytes"] > before.st_size:
                    raise MatrixScanError("source changed during compressed hash pass")
                if progress is not None:
                    progress(report)
            if report["compressed_bytes"] != before.st_size or _fingerprint(os.fstat(raw.fileno())) != _fingerprint(before):
                raise MatrixScanError("source changed during compressed hash pass")
            report["compressed_sha256"] = digest.hexdigest()
            report["compressed_hash_complete"] = True
            raw.seek(0)
            with gzip.GzipFile(fileobj=raw, mode="rb") as decoded:
                for first_line, lines in _line_batches(decoded, report, options, progress):
                    offset = 0
                    while offset < len(lines) and report["declared_entries"] is None:
                        line_number = first_line + offset
                        line = lines[offset]
                        offset += 1
                        tokens = line.split()
                        if line_number == 1:
                            banner = [token.lower() for token in tokens]
                            if len(banner) != 5 or banner[:3] != [b"%%matrixmarket", b"matrix", b"coordinate"] \
                                    or banner[3] not in {b"integer", b"real"} or banner[4] != b"general":
                                raise MatrixScanError("unsupported MatrixMarket header; require coordinate integer/real general")
                            report["field"] = banner[3].decode("ascii")
                            continue
                        if not tokens or line.lstrip().startswith(b"%"):
                            continue
                        if len(tokens) != 3:
                            raise MatrixScanError(f"line {line_number}: dimension header requires exactly three columns")
                        genes, cells, entries = (_integer(token, "dimension/entry header") for token in tokens)
                        for name, value in (("genes", genes), ("cells", cells), ("entries", entries)):
                            if value < 0:
                                raise MatrixScanError(f"negative header {name}")
                            if value > options[f"max_{name}"]:
                                raise MatrixScanError(f"{name} budget exceeded by header")
                            if expected[name] is not None and value != expected[name]:
                                raise MatrixScanError(f"expected {name} header mismatch")
                        report.update(genes=genes, cells=cells, declared_entries=entries)
                        if genes + cells <= axis_support_max_bytes:
                            genes_seen, cells_seen = bytearray(genes), bytearray(cells)
                            report["axis_support"] = {"performed": False, "reason": "scan not complete",
                                                      "allocated_bytes": genes + cells}
                        else:
                            report["axis_support"] = {"performed": False, "reason": "axis support byte budget exceeded",
                                                      "allocated_bytes": 0}
                    if report["declared_entries"] is not None:
                        _scan_entries(lines[offset:], first_line + offset, report, max_count_digits,
                                      genes_seen, cells_seen)
            if report["field"] is None or report["declared_entries"] is None:
                raise MatrixScanError("premature EOF before complete MatrixMarket header")
            if report["observed_entries"] != report["declared_entries"]:
                raise MatrixScanError("premature EOF: observed entry count differs from header")
            if _fingerprint(os.fstat(raw.fileno())) != _fingerprint(before) \
                    or _fingerprint(source.stat()) != _fingerprint(before):
                raise MatrixScanError("source changed during scan")
            report["source_unchanged"] = True
        if genes_seen is not None:
            gene_support, cell_support = genes_seen.count(1), cells_seen.count(1)
            report["axis_support"] = {"performed": True, "allocated_bytes": len(genes_seen) + len(cells_seen),
                                      "genes_with_positive_counts": gene_support,
                                      "cells_with_positive_counts": cell_support,
                                      "genes_without_positive_counts": report["genes"] - gene_support,
                                      "cells_without_positive_counts": report["cells"] - cell_support}
        report.update(status="passed", technical_count_like_integrity=True)
    except (MatrixScanError, OSError, EOFError, zlib.error) as error:
        report.update(status="failed", error=str(error))
        if isinstance(error, MatrixScanError):
            error.report = report
            raise
        raise MatrixScanError(f"gzip/file integrity failure: {error}", report) from error
    if progress is not None:
        progress(report)
    return report


def _write_json(stream, payload):
    json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write("\n")
    stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local .mtx.gz only (no raw sequencing)")
    parser.add_argument("--output", "--out", type=Path, required=True, help="fresh JSON path; parent must exist")
    for name in ("decoded-bytes", "genes", "cells", "entries"):
        parser.add_argument(f"--max-{name}", type=int, required=True)
    for name in ("genes", "cells", "entries"):
        parser.add_argument(f"--expected-{name}", type=int)
    parser.add_argument("--chunk-bytes", type=int, default=1 << 20)
    parser.add_argument("--max-line-bytes", type=int, default=4096)
    parser.add_argument("--max-count-digits", type=int, default=128)
    parser.add_argument("--axis-support-max-bytes", type=int, default=8 << 20)
    parser.add_argument("--progress-dir", type=Path, help="exclusive fresh directory containing immutable progress JSON files")
    parser.add_argument("--progress-every-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    names = ("max_decoded_bytes", "max_genes", "max_cells", "max_entries", "chunk_bytes",
             "max_line_bytes", "max_count_digits", "axis_support_max_bytes")
    options = {name: getattr(args, name) for name in names}
    expected = {name: getattr(args, f"expected_{name}") for name in ("genes", "cells", "entries")}
    try:
        _validate_options(options, expected)
        if not math.isfinite(args.progress_every_seconds) or args.progress_every_seconds <= 0:
            raise MatrixScanError("progress interval must be finite and positive")
    except MatrixScanError as error:
        parser.error(str(error))
    # lexists catches dangling symlinks as well. Exclusive creation below is the
    # authoritative race-safe guard; no source/output is replaced or truncated.
    if os.path.lexists(args.output) or (args.progress_dir is not None and os.path.lexists(args.progress_dir)):
        print("output and progress paths must be fresh; nothing overwritten", file=sys.stderr)
        return 2
    if args.progress_dir is not None and (args.output.resolve() == args.progress_dir.resolve()
                                         or args.progress_dir.resolve() in args.output.resolve().parents):
        print("output must be outside the fresh progress directory", file=sys.stderr)
        return 2
    sequence, last_progress = 0, None

    def publish_progress(report):
        nonlocal sequence, last_progress
        now = time.monotonic()
        if args.progress_dir is None or (report["status"] == "scanning" and last_progress is not None
                                        and now - last_progress < args.progress_every_seconds):
            return
        path = args.progress_dir / f"progress-{sequence:06d}.json"
        with path.open("x", encoding="utf-8") as stream:
            _write_json(stream, report)
        sequence += 1
        last_progress = now

    try:
        with args.output.open("x", encoding="utf-8") as stream:
            try:
                if args.progress_dir is not None:
                    args.progress_dir.mkdir()
                result = scan_matrix(args.source, **options,
                                     **{f"expected_{name}": value for name, value in expected.items()},
                                     progress=publish_progress)
            except MatrixScanError as error:
                result = error.report
                publish_progress(result)
            _write_json(stream, result)
    except OSError as error:
        print(f"fresh output/progress publication failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output.absolute()),
                      "stage_a_approved": False, "scientific_approved": False}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
