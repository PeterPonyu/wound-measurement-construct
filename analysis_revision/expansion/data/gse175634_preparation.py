"""Offline, receipt-bound GSE175634 identity preparation and technical gates.

This module uses the standard library only. It never imports historical output
directories, downloads data, approves Stage A/B, or loads a matrix with SciPy.
The optional explicit ``scan`` stage reuses remote.autodl.matrix_scan.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time
import zlib


MATRIX = "GSE175634_cell_counts.mtx.gz"
CELL_METADATA = "GSE175634_cell_metadata.tsv.gz"
CELL_INDEX = "GSE175634_cell_indices.tsv.gz"
GENE_INDEX = "GSE175634_gene_indices_counts.tsv.gz"
COLLECTION = "GSE175634_collection_metadata.txt.gz"
EXPERIMENT = "GSE175634_experimental_design.txt.gz"
INVENTORY = (CELL_METADATA, CELL_INDEX, GENE_INDEX, COLLECTION, EXPERIMENT, MATRIX)
EMPTY_TOKENS = {"", "na", "nan", "none", "null", "<na>"}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
NONAPPROVALS = dict(stage_a_approved=False, stage_b_approved=False,
                    training_approved=False, GPU_started=False, network_started=False)
HELPER_PROVENANCE = {
    "source": "output/sdd-data-preparation-20261001-r1/audit_identity_metadata.py",
    "sha256": "ee93a258a4aaa91837e5f1519b99c640048e36721470a252d2d7ba60987d8f4c",
    "functions": ["require_columns", "design_lookup", "graph_components", "audit_cells", "audit_index"],
    "crosswalk_source": "output/sdd-small-download-20260930-r1/probe_downloaded_content.py",
    "crosswalk_sha256": "85d29d4f23a49bfa039f334c7be8ff3f64f3977f4024bc22a77e54925c5e0581",
    "crosswalk_function": "design_crosswalk",
    "adaptation": "Vendored exact composite-key/prefix/index rules; stricter width, token, row and "
                  "resource guards; preserve every raw column; full author-pool purge planner added.",
    "runtime_dependency_on_historical_output": False,
}


class PreparationError(ValueError):
    """A failed admission, binding, identity or resource contract."""


def safe_path(value, *, file=False, directory=False, fresh=False):
    path = Path(value).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise PreparationError(f"traversal or symlink in path: {path}")
    if fresh and os.path.lexists(path):
        raise PreparationError(f"fresh output required; nothing overwritten: {path}")
    if file and not path.is_file():
        raise PreparationError(f"regular local file required: {path}")
    if directory and not path.is_dir():
        raise PreparationError(f"local directory required: {path}")
    return path


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def regular_input(path):
    """Reject symlink ancestry and path/descriptor replacement during each pass."""
    path = safe_path(path, file=True)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PreparationError("regular input required")
        yield stream
        safe_path(path, file=True)
        if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(before) \
                or _fingerprint(path.stat()) != _fingerprint(before):
            raise PreparationError(f"input changed during read: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with regular_input(path) as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PreparationError(f"exact lowercase SHA256 required: {label}")
    return value


def _positive(value, label, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise PreparationError(f"invalid integer budget/count: {label}")
    return value


def _unique_json(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise PreparationError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read_bound_json(path, expected_sha256=None, *, cap=2 << 20):
    with regular_input(path) as stream:
        body = stream.read(cap + 1)
    if len(body) > cap:
        raise PreparationError("JSON byte budget exceeded")
    digest = hashlib.sha256(body).hexdigest()
    if expected_sha256 is not None and digest != _sha(expected_sha256, "JSON binding"):
        raise PreparationError("JSON SHA256 binding mismatch")
    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_json,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               PreparationError(f"nonfinite JSON token: {token}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PreparationError("JSON object required")
    return value, digest, body


def load_acceptance(path, expected_sha256):
    receipt, digest, body = read_bound_json(path, expected_sha256)
    if receipt.get("schema") != "ortho.ncbi-download-local-acceptance.v1" or receipt.get("passed") is not True:
        raise PreparationError("passed exact download acceptance required")
    if receipt.get("stage_a_approved") is not False or receipt.get("training_approved") is not False \
            or receipt.get("GPU_started") is not False or receipt.get("stage_b_approved", False) is not False:
        raise PreparationError("download receipt is not a scientific/training approval")
    rows = receipt.get("files")
    if not isinstance(rows, list) or len(rows) != 6 or any(not isinstance(r, dict) for r in rows):
        raise PreparationError("exact six-file inventory required")
    if Counter(r.get("file") for r in rows) != Counter(INVENTORY):
        raise PreparationError("exact six-file inventory mismatch")
    for row in rows:
        _positive(row.get("bytes"), "compressed bytes")
        _positive(row.get("decoded_bytes"), "decoded bytes")
        _sha(row.get("sha256"), row["file"])
        if row.get("gzip_crc_and_eof_verified") is not True:
            raise PreparationError("complete gzip CRC and EOF acceptance required for all six files")
        if not isinstance(row.get("hash_basis"), str) or not row["hash_basis"]:
            raise PreparationError("file checksum provenance must be preserved")
    if receipt.get("total_compressed_bytes") != sum(r["bytes"] for r in rows):
        raise PreparationError("receipt total compressed bytes mismatch")
    if not isinstance(receipt.get("remote_destination"), str) or not receipt["remote_destination"].startswith("/"):
        raise PreparationError("receipt original remote destination required")
    return receipt, digest, body


def load_spec(path, expected_sha256=None):
    spec, digest, body = read_bound_json(path, expected_sha256)
    if spec.get("schema") != "ortho.gse175634.preparation.v1" or spec.get("inventory") != list(INVENTORY):
        raise PreparationError("exact GSE175634 preparation specification required")
    for name in ("genes", "cells", "entries", "lines", "pools", "design_records"):
        _positive(spec["expected"].get(name), name)
    days = spec["expected"].get("days")
    if not isinstance(days, list) or len(days) < 2 or any(type(d) is not int or d < 0 for d in days) \
            or days != sorted(set(days)):
        raise PreparationError("explicit ordered physical-day grid required")
    if spec.get("min_cells_per_line_day") != 50:
        raise PreparationError("missing min50 preparation grid contract")
    _sha(spec.get("scanner_sha256"), "bounded scanner source")
    if any(spec.get(flag) is not False for flag in ("stage_a_approved", "stage_b_approved", "training_approved",
                                                   "network_allowed", "cuda_allowed")):
        raise PreparationError("preparation specification must forbid network/CUDA and automatic approvals")
    if not isinstance(spec.get("followup_gates"), list) or not spec["followup_gates"]:
        raise PreparationError("explicit real-matrix followup gates required")
    if spec.get("count_policy") != "reject_duplicate_coordinates_no_silent_sum" \
            or spec.get("doublet_policy") != "descriptive_only_no_new_singlet_threshold":
        raise PreparationError("unknown duplicate/count or doublet policy")
    qc = spec.get("qc")
    if qc != {"positive_library_required": True, "min_detected_genes": 200,
              "max_mito_fraction": 0.2, "mito_gene_rule": "gene_name_uppercase_startswith_MT-",
              "missing_mito_gene_axis": "reject", "detected_gene_rule": "unique_coordinates_with_count_gt_zero",
              "sensitivity_min_detected_genes": [200, 500], "sensitivity_max_mito_fraction": [0.1, 0.2]}:
        raise PreparationError("unknown or incomplete exact QC specification")
    bounds = spec["budgets"]
    for name in ("metadata_memory_bytes", "scan_memory_bytes", "metadata_cpu_seconds", "scan_cpu_seconds",
                 "max_output_bytes", "minimum_free_bytes", "metadata_total_decoded_bytes", "metadata_total_compressed_bytes", "max_row_bytes",
                 "matrix_max_decoded_bytes", "matrix_chunk_bytes", "matrix_max_line_bytes", "matrix_max_count_digits",
                 "matrix_axis_support_max_bytes", "matrix_followup_disk_bytes", "matrix_followup_memory_bytes"):
        _positive(bounds.get(name), name)
    fraction = bounds.get("matrix_post_write_free_fraction")
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise PreparationError("invalid disk reserve fraction")
    if bounds["metadata_memory_bytes"] > 2 << 30 or bounds["scan_memory_bytes"] > 4 << 30 \
            or bounds["max_output_bytes"] > 256 << 20 or bounds["max_row_bytes"] > 65536 \
            or bounds["matrix_chunk_bytes"] > 16 << 20 or bounds["matrix_max_line_bytes"] > 4096 \
            or bounds["matrix_max_count_digits"] > 128 or bounds["matrix_axis_support_max_bytes"] > 64 << 20:
        raise PreparationError("implementation resource ceiling exceeded")
    caps = bounds.get("metadata_decoded_caps", {})
    if set(caps) != set(INVENTORY) - {MATRIX}:
        raise PreparationError("exact five metadata decoded caps required")
    for name, cap in caps.items():
        _positive(cap, name)
        if cap > 128 << 20:
            raise PreparationError("metadata decoded cap exceeds implementation ceiling")
    return spec, digest, body


def require_columns(headers, required):
    # Vendored from the frozen identity helper, with a blank-header guard added.
    if not headers or any(not h for h in headers) or len(headers) != len(set(headers)) \
            or not set(required) <= set(headers):
        raise PreparationError("missing or duplicate exact author columns")


def _rows(source, required, *, max_row_bytes=65536):
    def lines():
        first = True
        while line := source.readline(max_row_bytes + 1):
            if len(line) > max_row_bytes:
                raise PreparationError("metadata row byte budget exceeded")
            yield line.decode("utf-8-sig" if first else "utf-8")
            first = False
    reader = csv.DictReader(lines(), delimiter="\t")
    require_columns(reader.fieldnames, required)
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise PreparationError("bad author table row width")
        yield reader.fieldnames, row


def _canonical_number(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise PreparationError(f"unknown explicit author {label} scheme")
    return value


def design_lookup(collection_text):
    """Vendored full (batch, CD, colony, NA-stripped line, day) author join."""
    keys, literal = {}, defaultdict(list)
    for _, row in _rows(io.BytesIO(collection_text.encode("utf-8")),
                        ["Experiment Batch", "CD", "col", "Line", "Differentiation Day", "orig.ident"]):
        if not re.fullmatch(r"NA(?:0|[1-9][0-9]*)", row["Line"]):
            raise PreparationError("explicit NA-prefixed author line scheme changed")
        key = tuple(_canonical_number(row[field], field) for field in ("Experiment Batch", "CD", "col")) \
            + (row["Line"][2:], _canonical_number(row["Differentiation Day"], "day"))
        if not row["orig.ident"] or key in keys:
            raise PreparationError("missing orig.ident or duplicate full composite design key")
        keys[key] = row
        literal[(row["orig.ident"], key[3], key[4])].append(row)
    if not keys:
        raise PreparationError("empty design")
    return keys, literal


def design_crosswalk(experiment_text, design):
    observed = Counter()
    for _, row in _rows(io.BytesIO(experiment_text.encode("utf-8")), ["Exp", "CD", "col", "Line", "Day"]):
        if not re.fullmatch(r"NA(?:0|[1-9][0-9]*)", row["Line"]) or not re.fullmatch(r"Day (?:0|[1-9][0-9]*)", row["Day"]):
            raise PreparationError("unknown experimental line/day scheme")
        key = tuple(_canonical_number(row[f], f) for f in ("Exp", "CD", "col")) \
            + (row["Line"][2:], row["Day"].removeprefix("Day "))
        observed[key] += 1
    if observed != Counter({key: 1 for key in design}):
        raise PreparationError("collection/design key mismatch or duplicate composite identity")
    return {"records": len(design), "exact_composite_key_crosswalk": True}


def graph_components(pool_lines):
    # Vendored verbatim algorithm; components are source lines, not donors.
    neighbors = defaultdict(set)
    for pool, lines in pool_lines.items():
        for line in lines:
            neighbors[line].update(lines)
    components = []
    unseen = set(neighbors)
    while unseen:
        start = min(unseen)
        todo, component = [start], set()
        while todo:
            line = todo.pop()
            if line in component:
                continue
            component.add(line)
            todo.extend(neighbors[line] - component)
        unseen -= component
        components.append(sorted(component))
    return components


def author_pools(design):
    pools = defaultdict(set)
    for batch, cd, colony, line, day in design:
        pools[f"E{batch}CD{cd}col{colony}"].add(line)
    return pools


def audit_cells(path, design, literal, *, sink=None, max_cells=230786, max_row_bytes=65536):
    """Adapted frozen identity audit; metadata order and all author fields survive."""
    derived = ["author_sample", "specimen", "batch", "pool", "CD", "col", "physical_pool", "line", "day",
               "author_collection_line", "original_collection_orig_ident", "orig_ident_conflict", "cell_index_one_based"]
    cells, unique, observed = [], set(), Counter()
    groups, conflicts, celltypes, probabilities, literal_joins = Counter(), Counter(), Counter(), Counter(), Counter()
    headers = None
    with regular_input(path) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as source:
        for position, (columns, row) in enumerate(_rows(source,
                ["cell", "exp.grp", "sample", "diffday", "individual", "demux.dbl.prb", "type"],
                max_row_bytes=max_row_bytes), 1):
            if position > max_cells:
                raise PreparationError("metadata cell count budget exceeded")
            if headers is None:
                headers = columns
                if set(headers) & set(derived):
                    raise PreparationError("conflicting raw and prepared metadata keys")
                if sink is not None:
                    sink.writerow(headers + derived)
            if any(row[f].strip().lower() in EMPTY_TOKENS for f in
                   ("cell", "exp.grp", "sample", "diffday", "individual", "type")):
                raise PreparationError("missing required author identity")
            if row["cell"] in unique:
                raise PreparationError("duplicate author cell key")
            pool = re.fullmatch(r"E(0|[1-9][0-9]*)CD(0|[1-9][0-9]*)col(0|[1-9][0-9]*)", row["exp.grp"])
            day = re.fullmatch(r"day(0|[1-9][0-9]*)", row["diffday"])
            if not pool or not day:
                raise PreparationError("unknown explicit pool/day scheme")
            line = _canonical_number(row["individual"], "line")
            if not row["cell"].startswith(f"E{pool[1]}_{row['exp.grp']}_") \
                    or not row["cell"].removeprefix(f"E{pool[1]}_{row['exp.grp']}_"):
                raise PreparationError("cell barcode prefix and exact experimental group/colony conflict")
            if row["sample"] != line + "_" + day[1]:
                raise PreparationError("author sample and individual/day conflict")
            key = (*pool.groups(), line, day[1])
            if key not in design:
                raise PreparationError("cell has no full author design match; unknown pool/line mapping")
            author = design[key]
            try:
                prob = float(row["demux.dbl.prb"])
            except ValueError as error:
                raise PreparationError("invalid demux doublet probability") from error
            if not math.isfinite(prob) or not 0 <= prob <= 1:
                raise PreparationError("demux doublet probability outside finite [0,1]")
            conflict = author["orig.ident"] != row["exp.grp"]
            specimen = f"{row['exp.grp']}__NA{line}__{row['diffday']}"
            cells.append(row["cell"])
            unique.add(row["cell"])
            observed[key] += 1
            conflicts[key] += conflict
            groups[(line, int(day[1]), row["exp.grp"], specimen, pool[1])] += 1
            celltypes[row["type"]] += 1
            for threshold in (.05, .1, .2):
                probabilities[str(threshold)] += prob > threshold
            matches = literal.get((row["exp.grp"], line, day[1]), [])
            literal_joins["unique" if len(matches) == 1 else "unmatched" if not matches else "ambiguous"] += 1
            if sink is not None:
                sink.writerow([row[h] for h in headers] + [row["sample"], specimen, pool[1], row["exp.grp"],
                    pool[2], pool[3], row["exp.grp"], line, day[1], author["Line"], author["orig.ident"],
                    "true" if conflict else "false", position])
    if not cells:
        raise PreparationError("empty metadata")
    pool_lines = author_pools(design)
    group_rows = [dict(line=k[0], day=k[1], physical_pool=k[2], specimen=k[3], batch=k[4], cells=n)
                  for k, n in sorted(groups.items())]
    return cells, {"author_columns": headers, "prepared_columns": headers + derived, "cells": len(cells),
        "cell_keys_unique": True, "exact_composite_design_coverage": 1.0,
        "literal_orig_ident_join_cell_counts": dict(literal_joins),
        "literal_orig_ident_disagreement_cells": sum(conflicts.values()),
        "literal_orig_ident_disagreements": [dict(composite_key=list(k), cells=n,
            original_collection_orig_ident=design[k]["orig.ident"], author_cell_pool=f"E{k[0]}CD{k[1]}col{k[2]}")
            for k, n in sorted(conflicts.items()) if n],
        "missing_design_groups": [list(k) for k in sorted(set(design) - set(observed))],
        "individuals": sorted({key[3] for key in design}),
        "individual_unit": "human_author_iPSC_source_cell_line_not_wound_donor_or_DFU_patient",
        "physical_days": sorted({int(key[4]) for key in design}), "physical_pools": len(pool_lines),
        "author_pool_lines": {k: sorted(v) for k, v in sorted(pool_lines.items())},
        "cell_type_counts": dict(celltypes), "doublet_probability_above_threshold_counts": dict(probabilities),
        "doublet_probability_policy": "descriptive_only_no_new_singlet_threshold",
        "specimen_unit": "physical_pool_plus_source_line_plus_physical_day", "groups": group_rows,
        "NA_prefix_transform": "exact_author_NA_digits_to_individual_digits_not_fuzzy_join",
        "expression_after_QC_task_eligibility": "unknown_not_performed"}


def audit_index(path, *, names_column, index_column, expected_names=None, max_names=230786, max_row_bytes=65536):
    names, seen = [], set()
    with regular_input(path) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as source:
        for position, (_, row) in enumerate(_rows(source, [index_column, names_column], max_row_bytes=max_row_bytes), 1):
            if position > max_names:
                raise PreparationError("identifier axis count budget exceeded")
            if row[index_column] != str(position) or row[names_column] in seen \
                    or row[names_column].strip().lower() in EMPTY_TOKENS \
                    or any(char in row[names_column] for char in "\t\r\n"):
                raise PreparationError("index is not one-based consecutive or identifiers duplicate/missing")
            names.append(row[names_column])
            seen.add(row[names_column])
    if not names:
        raise PreparationError("empty identifier index")
    if expected_names is not None and names != expected_names:
        raise PreparationError("matrix cell index order differs from author metadata")
    return names


def assert_pool_disjoint(training, evaluation, *, line_disjoint=True):
    train_pools = {g["physical_pool"] for g in training}
    eval_pools = {g["physical_pool"] for g in evaluation}
    if train_pools & eval_pools:
        raise PreparationError("physical-pool leakage between fitting and evaluation")
    if line_disjoint and {g["line"] for g in training} & {g["line"] for g in evaluation}:
        raise PreparationError("source-line leakage between fitting and evaluation")
    return True


def min50_grid(groups, lines, days):
    counts = Counter()
    for group in groups:
        if group["line"] not in lines or group["day"] not in days:
            raise PreparationError("unknown line/day in min50 grid")
        _positive(group["cells"], "group cells")
        counts[group["line"], group["day"]] += group["cells"]
    return [dict(line=line, day=day, cells=counts[line, day], min50=counts[line, day] >= 50)
            for line in sorted(lines) for day in days]


def plan_pool_purged(author, *, min_cells=50, minimum_training_lines=3):
    if min_cells != 50:
        raise PreparationError("missing min50 split contract")
    if type(minimum_training_lines) is not int or minimum_training_lines < 3:
        raise PreparationError("at least three retained training source lines required")
    lines, days, pools, groups = author["individuals"], author["physical_days"], author["author_pool_lines"], author["groups"]
    if len(lines) != len(set(lines)) or days != sorted(set(days)) or any(not members for members in pools.values()) \
            or set(lines) != set().union(*(set(members) for members in pools.values())):
        raise PreparationError("incomplete or conflicting authoritative pool/line/day inventory")
    seen = set()
    for group in groups:
        if group["physical_pool"] not in pools or group["line"] not in pools[group["physical_pool"]]:
            raise PreparationError("unknown pool/line mapping in split input")
        key = (group["physical_pool"], group["line"], group["day"])
        expected_specimen = f"{key[0]}__NA{key[1]}__day{key[2]}"
        if key in seen or group["specimen"] != expected_specimen \
                or not group["physical_pool"].startswith(f"E{group['batch']}CD"):
            raise PreparationError("conflicting specimen/batch/full composite split keys")
        seen.add(key)
    grid = min50_grid(groups, lines, days)
    counts = {(r["line"], r["day"]): r["cells"] for r in grid}
    folds = []
    for held in lines:
        # Use ALL authoritative pools of the held line, including currently empty
        # specimens. Never condition the purge on a selected evaluation day.
        held_pools = sorted(pool for pool, members in pools.items() if held in members)
        evaluation = [g for g in groups if g["line"] == held]
        training = [g for g in groups if g["line"] != held and g["physical_pool"] not in held_pools]
        assert_pool_disjoint(training, evaluation)
        after = min50_grid(training, [line for line in lines if line != held], days)
        retained = {(r["line"], r["day"]): r["cells"] for r in after}
        pairs = []
        for source, target in zip(days[:-1], days[1:]):
            eligible_train = [line for line in lines if line != held and
                              all(retained.get((line, day), 0) >= 50 for day in (source, target))]
            missing = [day for day in (source, target) if counts[held, day] < 50]
            pairs.append(dict(source_day=source, target_day=target, training_lines=eligible_train,
                missing_held_min50_days=missing,
                metadata_only_candidate=not missing and len(eligible_train) >= minimum_training_lines,
                after_QC_eligible="unknown"))
        folds.append(dict(held_line=held, excluded_author_pools=held_pools,
            training_selector={"line_not_equal": held, "physical_pool_not_in": held_pools},
            evaluation_selector={"line_equal": held}, physical_pool_disjoint=True, source_line_disjoint=True,
            evaluation_cells=sum(g["cells"] for g in evaluation), training_cells=sum(g["cells"] for g in training),
            purged_other_line_cells=sum(g["cells"] for g in groups if g["line"] != held and g["physical_pool"] in held_pools),
            post_purge_training_min50_grid=after, adjacent_pairs=pairs))
    batches = defaultdict(set)
    for group in groups:
        batches[group["batch"]].add(group["line"])
    return dict(kind="metadata_only_pool_purged_split_plan", min_cells_per_line_day=50,
        line_day_min50_grid=grid, line_holdout_folds=folds, author_pool_connected_components=graph_components(pools),
        batch_to_lines={k: sorted(v) for k, v in sorted(batches.items())},
        component_holdout_warning="Line-and-pool disjointness is not donor independence; author components may be batch-confounded.",
        physical_day_grid_is_fixed=True, missing_min50_days_never_bridged=True,
        representation_and_scaling_must_use_training_selector_only=True,
        interior_time_inventory=[dict(source_day=a, held_day=h, anchor_day=b,
            lines_with_raw_metadata_min50=[line for line in lines if all(counts[line, d] >= 50 for d in (a, h, b))],
            executable_time_holdout_tasks_generated=False)
            for a, h, b in zip(days[:-2], days[1:-1], days[2:])],
        jobs_generated=False, **NONAPPROVALS)


def verify_metadata(path, row, cap, *, copy_to=None):
    """Stream exact compressed SHA/size and all gzip members through CRC/EOF."""
    digest, count, decoded = hashlib.sha256(), 0, 0
    try:
        with regular_input(path) as stream:
            if os.fstat(stream.fileno()).st_size != row["bytes"]:
                raise PreparationError(f"compressed size mismatch: {row['file']}")
            sink = None if copy_to is None else safe_path(copy_to, fresh=True).open("xb")
            try:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    count += len(block)
                    if count > row["bytes"]:
                        raise PreparationError("compressed size budget exceeded")
                    digest.update(block)
                    if sink is not None:
                        sink.write(block)
            finally:
                if sink is not None:
                    sink.close()
            if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise PreparationError(f"compressed SHA256 mismatch: {row['file']}")
            stream.seek(0)
            with gzip.GzipFile(fileobj=stream, mode="rb") as gz:
                while block := gz.read(min(65536, cap - decoded + 1)):
                    decoded += len(block)
                    if decoded > cap:
                        raise PreparationError("metadata decoded byte budget exceeded")
    except (OSError, EOFError, zlib.error) as error:
        raise PreparationError(f"gzip/file integrity failure: {error}") from error
    if decoded != row["decoded_bytes"]:
        raise PreparationError(f"decoded size mismatch: {row['file']}")
    return dict(row, locally_recomputed_sha_size_gzip=True, observed_decoded_bytes=decoded)


def validate_inventory(data_root, *, matrix_required, receipt=None, acceptance_path=None, acceptance_sha=None):
    """Exactly six payloads; only fingerprint-bound acquisition controls coexist.

    The existing downloader retains acquisition.json and per-payload journals.
    They are NOT extra expression inputs and must not be ignored by a glob.
    """
    root = safe_path(data_root, directory=True)
    names, controls = set(), []
    accepted = {row["file"]: row for row in receipt["files"]} if receipt is not None else {}
    for path in root.iterdir():
        safe_path(path, file=True)
        if path.name in INVENTORY:
            names.add(path.name)
            continue
        if acceptance_path is not None and path == safe_path(acceptance_path):
            _, digest, _ = load_acceptance(path, acceptance_sha)
        elif path.name == "acquisition.json" and receipt is not None and "acquisition_receipt_sha256" in receipt:
            acquired, digest, _ = read_bound_json(path, _sha(receipt["acquisition_receipt_sha256"], "acquisition receipt"))
            rows = acquired.get("items", [])
            if acquired.get("authority_sha256") != receipt.get("authority_sha256") \
                    or acquired.get("stage_a_approved") is not False or acquired.get("training_approved") is not False \
                    or acquired.get("status") != "acquired_unverified_pending_full_content_and_identity_review" \
                    or len(rows) != 6 or Counter(r.get("destination") for r in rows) != Counter(INVENTORY) \
                    or any(row.get("observed_sha256") != accepted[row["destination"]]["sha256"]
                           or row.get("expected_bytes") != accepted[row["destination"]]["bytes"] for row in rows):
                raise PreparationError("acquisition control differs from bound six-file acceptance")
        elif path.name.endswith(".journal.json") and path.name.removesuffix(".journal.json") in accepted:
            item = accepted[path.name.removesuffix(".journal.json")]
            journal, digest, _ = read_bound_json(path)
            contract = journal.get("contract", {})
            if set(journal) != {"contract", "bytes", "sha256"} or journal["bytes"] != item["bytes"] \
                    or journal["sha256"] != item["sha256"] or contract.get("dataset_id") != "GSE175634" \
                    or contract.get("destination") != item["file"] or contract.get("expected_bytes") != item["bytes"] \
                    or contract.get("stage_a_approved") is not False or contract.get("training_approved") is not False:
                raise PreparationError("acquisition journal is partial or differs from accepted payload fingerprint")
        else:
            raise PreparationError("local exact payload inventory mismatch; unbound extra control/file/subdirectory")
        controls.append(dict(file=path.name, sha256=digest, bytes=path.stat().st_size,
                             scope="acquisition_control_not_expression_payload"))
    required = set(INVENTORY) if matrix_required else set(INVENTORY) - {MATRIX}
    if not required <= names:
        raise PreparationError("local exact payload inventory mismatch; required payload absent")
    return root, sorted(controls, key=lambda row: row["file"])


def resource_snapshot(parent, spec, *, full_matrix=False):
    disk = shutil.disk_usage(safe_path(parent, directory=True))
    bounds = spec["budgets"]
    planned = bounds["matrix_followup_disk_bytes"] if full_matrix else bounds["max_output_bytes"]
    fraction = bounds["matrix_post_write_free_fraction"] if full_matrix else 0.0
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except OSError:
        pass
    required_memory = bounds["scan_memory_bytes"] if full_matrix else bounds["metadata_memory_bytes"]
    checks = dict(disk_bytes=disk.free - planned >= bounds["minimum_free_bytes"],
                  disk_fraction=(disk.free - planned) / disk.total >= fraction,
                  available_memory=available is not None and available >= required_memory)
    return dict(checks=checks, passed=all(checks.values()), disk_total_bytes=disk.total,
        disk_free_bytes=disk.free, planned_increment_bytes=planned, post_write_free_fraction_required=fraction,
        memory_available_bytes=available, required_memory_bytes=required_memory,
        conservative_downstream_followup_required_memory_bytes=bounds["matrix_followup_memory_bytes"],
        conservative_downstream_followup_memory_gate=available is not None and available >= bounds["matrix_followup_memory_bytes"],
        peak_memory_measured=False, reserve_scope="full_followup" if full_matrix else "bounded_metadata_evidence")


def runtime_snapshot(stage="metadata_or_technical_scan"):
    import resource

    return dict(cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        numeric_threads={key: os.environ.get(key) for key in THREAD_ENV},
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        address_space_limit_bytes=list(resource.getrlimit(resource.RLIMIT_AS)),
        cpu_time_limit_seconds=list(resource.getrlimit(resource.RLIMIT_CPU)),
        file_size_limit_bytes=list(resource.getrlimit(resource.RLIMIT_FSIZE)),
        measured_stage_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        reporting_stage=stage,
        measured_peak_scope="process_lifetime_high_watermark_not_stage_delta_or_target_scale_extrapolation")


def write_json_fresh(path, value):
    with safe_path(path, fresh=True).open("x", encoding="utf-8") as sink:
        json.dump(value, sink, sort_keys=True, indent=2, allow_nan=False)
        sink.write("\n")


def _write_bytes_fresh(path, body):
    with safe_path(path, fresh=True).open("xb") as sink:
        sink.write(body)


@contextmanager
def _gzip_text_fresh(path):
    # Deterministic headers; preserve compressed raw sources separately.
    with safe_path(path, fresh=True).open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                yield text


def _small_text(path, cap):
    with regular_input(path) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as stream:
        body = stream.read(cap + 1)
    if len(body) > cap:
        raise PreparationError("decoded small author table budget exceeded")
    return body.decode("utf-8-sig")


def _manifest(out):
    return [dict(file=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha256_file(path))
            for path in sorted(out.rglob("*")) if path.is_file()]


def validate_metadata_report(path, expected_sha256, receipt_sha, spec_sha):
    report, digest, _ = read_bound_json(path, expected_sha256, cap=8 << 20)
    if report.get("mode") != "metadata" or report.get("status") != "metadata_prepared_not_scientific_admission" \
            or report.get("acceptance_sha256") != receipt_sha or report.get("spec_sha256") != spec_sha \
            or report.get("source_sha256") != sha256_file(Path(__file__)) \
            or report.get("cli_sha256") != sha256_file(Path(__file__).parents[2] / "scripts/prepare_gse175634_qualified.py"):
        raise PreparationError("metadata preparation lost acceptance/spec/source hash binding")
    if any(report.get(flag) is not False for flag in NONAPPROVALS):
        raise PreparationError("metadata report unexpectedly claims approval")
    root = safe_path(Path(path).parent, directory=True)
    files = report.get("output_files", [])
    required = {f"raw/{name}" for name in INVENTORY if name != MATRIX} | {
        "author_cell_metadata_prepared.tsv.gz", "cell_names_no_header.tsv.gz", "gene_names_no_header.tsv.gz",
        "identity.json", "pool-purged-plan.json", "DOWNLOAD-ACCEPTANCE.json", "preparation-spec.json"}
    if {r.get("file") for r in files} != required or len(files) != len(required):
        raise PreparationError("metadata output manifest closure mismatch")
    for item in files:
        name = Path(item["file"])
        if name.is_absolute() or ".." in name.parts:
            raise PreparationError("invalid metadata manifest path")
        target = safe_path(root / name, file=True)
        if target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]:
            raise PreparationError("metadata output hash binding mismatch")
    copied_receipt, _, _ = load_acceptance(root / "DOWNLOAD-ACCEPTANCE.json", receipt_sha)
    copied_spec, _, _ = load_spec(root / "preparation-spec.json", spec_sha)
    raw_entries = {item["file"]: item for item in files}
    for item in copied_receipt["files"]:
        if item["file"] != MATRIX and any(raw_entries[f"raw/{item['file']}"][field] != item[field]
                                          for field in ("sha256", "bytes")):
            raise PreparationError("raw metadata manifest lost original acceptance binding")
    if report.get("actual") != {key: copied_spec["expected"][key] for key in
                               ("genes", "cells", "lines", "pools", "design_records", "days")}:
        raise PreparationError("metadata actual-axis inventory differs from bound specification")
    return report, digest


def assess_admission(metadata, scan=None):
    """Executable refusal gate; this entrypoint does not perform duplicate/QC/CSR."""
    technical = scan is not None and scan.get("status") == "passed" and scan.get("technical_count_like_integrity") is True
    gates = dict(receipt_bound_metadata_prepared=metadata is not None, full_stored_count_scan=technical,
        duplicate_coordinate_validation=False, biological_count_semantics_review=False,
        real_expression_QC=False, post_QC_min50_and_pool_purge_recomputed=False,
        scalable_standardization_and_roundtrip=False, measured_peak_resources=False,
        explicit_scientific_approval=False)
    return dict(status="blocked", gates=gates, blockers=[key for key, ok in gates.items() if not ok],
        duplicate_validation="not_performed; scanner does not retain coordinate pairs",
        count_semantics="Nonnegative integral stored values are not a UMI/read or biological provenance attestation.",
        implemented_real_matrix_stages=["bounded_technical_scan"],
        unperformed_by_this_stage=["external_sort_duplicate_check", "expression_QC", "out_of_core_binary_CSR_writer"],
        unimplemented_real_matrix_stages=["AnnData_H5AD_writer_or_consumer_integration"],
        additional_executable_preparation_entrypoint="scripts/prepare_gse175634_qc.py",
        scope="Metadata/technical-scan evidence only; actual duplicate/QC/CSR preparation is a separate hash-bound stage. No automatic admission.",
        **NONAPPROVALS)


def prepare(*, mode, data_root, download_acceptance, expected_acceptance_sha256, out, spec_path,
            expected_spec_sha256=None, metadata_preparation=None, expected_metadata_sha256=None,
            matrix_scan_report=None, expected_matrix_scan_sha256=None):
    """Execute exactly one stage into an exclusive fresh directory."""
    if mode not in {"plan", "metadata", "scan", "admission"}:
        raise PreparationError("unknown preparation mode")
    if ((mode not in {"scan", "admission"} and (metadata_preparation is not None or expected_metadata_sha256 is not None))
            or (mode != "admission" and (matrix_scan_report is not None or expected_matrix_scan_sha256 is not None))
            or ((matrix_scan_report is None) != (expected_matrix_scan_sha256 is None))):
        raise PreparationError("stage-specific evidence arguments must not be silently ignored")
    receipt, receipt_sha, receipt_body = load_acceptance(download_acceptance, expected_acceptance_sha256)
    spec, spec_sha, spec_body = load_spec(spec_path, expected_spec_sha256)
    bounds, expected = spec["budgets"], spec["expected"]
    selected = {row["file"]: row for row in receipt["files"]}
    if sum(selected[n]["bytes"] for n in INVENTORY if n != MATRIX) > min(bounds["metadata_total_compressed_bytes"], bounds["max_output_bytes"]) \
            or sum(selected[n]["decoded_bytes"] for n in INVENTORY if n != MATRIX) > bounds["metadata_total_decoded_bytes"] \
            or any(selected[n]["decoded_bytes"] > bounds["metadata_decoded_caps"][n] for n in INVENTORY if n != MATRIX) \
            or selected[MATRIX]["decoded_bytes"] > bounds["matrix_max_decoded_bytes"]:
        raise PreparationError("accepted payload exceeds specification decoded budgets")
    root, out = safe_path(data_root), safe_path(out, fresh=True)
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise PreparationError("outputs must be disjoint from input data root")
    for source in (download_acceptance, spec_path, metadata_preparation, matrix_scan_report):
        if source is not None and safe_path(source).is_relative_to(out):
            raise PreparationError("outputs must not contain input receipts or preparation evidence")
    gate = resource_snapshot(out.parent, spec, full_matrix=mode == "scan")
    if not gate["passed"]:
        raise PreparationError(f"resource gate blocked: {gate['checks']}")
    metadata = metadata_sha = None
    if mode in {"scan", "admission"}:
        if metadata_preparation is None or expected_metadata_sha256 is None:
            raise PreparationError("explicit hash-bound metadata preparation required")
        metadata, metadata_sha = validate_metadata_report(metadata_preparation, expected_metadata_sha256, receipt_sha, spec_sha)
    controls = []
    if mode in {"metadata", "scan"}:
        _, controls = validate_inventory(root, matrix_required=mode == "scan", receipt=receipt,
                                          acceptance_path=download_acceptance, acceptance_sha=receipt_sha)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        _write_bytes_fresh(out / "DOWNLOAD-ACCEPTANCE.json", receipt_body)
        _write_bytes_fresh(out / "preparation-spec.json", spec_body)
        report = dict(dataset_id="GSE175634", mode=mode, acceptance_sha256=receipt_sha, spec_sha256=spec_sha,
            source_sha256=sha256_file(Path(__file__)), original_remote_destination=receipt["remote_destination"],
            cli_sha256=sha256_file(Path(__file__).parents[2] / "scripts/prepare_gse175634_qualified.py"),
            data_root=str(root), rebind_is_checksum_based_not_path_based=True, resource_gate=gate,
            validated_acquisition_controls=controls,
            checksum_scope="observed_TOFU_not_publisher_checksum", original_files_modified=False,
            cpu_only=True, matrix_opened=False, full_matrix_entries_scanned=False,
            helper_provenance=HELPER_PROVENANCE, **NONAPPROVALS)
        if mode == "plan":
            report.update(status="receipt_bound_plan_only", local_payload_verification_performed=False,
                expected=expected, exact_six_file_inventory=receipt["files"], qc_specification=spec["qc"],
                followup_resource_gate=resource_snapshot(out.parent, spec, full_matrix=True),
                followup_stages=[dict(stage="metadata", implemented=True, requires="five SHA/size/gzip-verified author files"),
                    dict(stage="scan", implemented=True, requires="six files + hash-bound metadata preparation + resource gate"),
                    dict(stage="duplicate_QC_binary_CSR", implemented=True,
                         entrypoint="scripts/prepare_gse175634_qc.py", target_matrix_executed=False,
                         requires=spec["followup_gates"]),
                    dict(stage="H5AD_or_consumer_integration", implemented=False,
                         requires="Separate official-schema conversion or consumer implementation and roundtrip")],
                scientific_admission=assess_admission(None))
        elif mode == "metadata":
            (out / "raw").mkdir()
            verified = [verify_metadata(root / name, selected[name], bounds["metadata_decoded_caps"][name],
                                       copy_to=out / "raw" / name) for name in INVENTORY if name != MATRIX]
            design, literal = design_lookup(_small_text(out / "raw" / COLLECTION, bounds["metadata_decoded_caps"][COLLECTION]))
            crosswalk = design_crosswalk(_small_text(out / "raw" / EXPERIMENT, bounds["metadata_decoded_caps"][EXPERIMENT]), design)
            with _gzip_text_fresh(out / "author_cell_metadata_prepared.tsv.gz") as sink:
                cells, author = audit_cells(out / "raw" / CELL_METADATA, design, literal,
                    sink=csv.writer(sink, delimiter="\t", lineterminator="\n"),
                    max_cells=expected["cells"], max_row_bytes=bounds["max_row_bytes"])
            cell_names = audit_index(out / "raw" / CELL_INDEX, names_column="cell_name", index_column="cell_index",
                                    expected_names=cells, max_names=expected["cells"])
            genes = audit_index(out / "raw" / GENE_INDEX, names_column="gene_name", index_column="gene_index", max_names=expected["genes"])
            actual = dict(cells=len(cells), genes=len(genes), lines=len(author["individuals"]),
                          pools=author["physical_pools"], design_records=len(design), days=author["physical_days"])
            if any(actual[k] != expected[k] for k in actual):
                raise PreparationError("actual author axes/line/day/pool inventory differs from specification")
            for name, values in (("cell_names_no_header.tsv.gz", cell_names), ("gene_names_no_header.tsv.gz", genes)):
                with _gzip_text_fresh(out / name) as sink:
                    for value in values:
                        sink.write(value + "\n")
            write_json_fresh(out / "identity.json", dict(author, genes=len(genes), crosswalk=crosswalk,
                cell_index_exact_metadata_order=True, gene_axis_original_order=True, **NONAPPROVALS))
            split = plan_pool_purged(author)
            write_json_fresh(out / "pool-purged-plan.json", split)
            report.update(status="metadata_prepared_not_scientific_admission", actual=actual,
                verified_metadata=verified, matrix_locally_verified=False,
                matrix_validation_scope="acceptance_receipt_only; matrix not opened in metadata mode",
                cell_index_exact_metadata_order=True, gene_axis_original_order=True,
                biological_unit="human_iPSC_cell_line_not_wound_donor", raw_metadata_preserved_byte_for_byte=True,
                orig_ident_conflict_cells=author["literal_orig_ident_disagreement_cells"],
                pool_purged_plan_generated=True, scientific_admission=assess_admission(author))
        elif mode == "scan":
            scanner_path = safe_path(Path(__file__).parents[2] / "remote/autodl/matrix_scan.py", file=True)
            if sha256_file(scanner_path) != spec["scanner_sha256"]:
                raise PreparationError("bounded scanner source SHA256 differs from specification")
            if any(os.environ.get(key) != "1" for key in THREAD_ENV) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise PreparationError("scan must use the single-thread CPU-only CLI environment")
            if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) != 1:
                raise PreparationError("scan requires exactly one CPU affinity")
            from remote.autodl.matrix_scan import scan_matrix

            for name in INVENTORY:
                path = safe_path(root / name, file=True)
                if path.stat().st_size != selected[name]["bytes"]:
                    raise PreparationError("scan payload size mismatch")
                if name != MATRIX:
                    verify_metadata(path, selected[name], bounds["metadata_decoded_caps"][name])
            scan = scan_matrix(root / MATRIX, max_decoded_bytes=bounds["matrix_max_decoded_bytes"],
                max_genes=expected["genes"], max_cells=expected["cells"], max_entries=expected["entries"],
                expected_genes=expected["genes"], expected_cells=expected["cells"], expected_entries=expected["entries"],
                chunk_bytes=bounds["matrix_chunk_bytes"], max_line_bytes=bounds["matrix_max_line_bytes"],
                max_count_digits=bounds["matrix_max_count_digits"], axis_support_max_bytes=bounds["matrix_axis_support_max_bytes"])
            write_json_fresh(out / "matrix-scan.json", scan)
            row = selected[MATRIX]
            if scan["compressed_sha256"] != row["sha256"] or scan["compressed_bytes"] != row["bytes"] \
                    or scan["decompressed_bytes"] != row["decoded_bytes"] or not scan["gzip_crc32_isize_validated"]:
                raise PreparationError("matrix scan differs from acceptance SHA/size/gzip binding")
            report.update(status="technical_scan_passed_not_scientific_admission", matrix_opened=True,
                full_matrix_entries_scanned=True, scanner_sha256=sha256_file(scanner_path),
                metadata_preparation_sha256=metadata_sha, scientific_admission=assess_admission(metadata, scan))
            if report["scanner_sha256"] != spec["scanner_sha256"]:
                raise PreparationError("bounded scanner source changed during scan")
        else:
            scan = None
            if matrix_scan_report is not None:
                if expected_matrix_scan_sha256 is None:
                    raise PreparationError("matrix scan report requires explicit SHA256 binding")
                outer, scan_sha, _ = read_bound_json(matrix_scan_report, expected_matrix_scan_sha256, cap=8 << 20)
                if outer.get("status") != "technical_scan_passed_not_scientific_admission" \
                        or outer.get("mode") != "scan" \
                        or outer.get("acceptance_sha256") != receipt_sha or outer.get("spec_sha256") != spec_sha \
                        or outer.get("metadata_preparation_sha256") != metadata_sha \
                        or outer.get("source_sha256") != report["source_sha256"] \
                        or outer.get("cli_sha256") != report["cli_sha256"] \
                        or outer.get("scanner_sha256") != spec["scanner_sha256"] \
                        or any(outer.get(flag) is not False for flag in NONAPPROVALS):
                    raise PreparationError("matrix scan report acceptance/spec/metadata/source binding mismatch")
                items = outer.get("output_files", [])
                rows = [r for r in items if r["file"] == "matrix-scan.json"]
                if len(rows) != 1:
                    raise PreparationError("matrix scan output manifest binding missing")
                scan_path = Path(matrix_scan_report).parent / "matrix-scan.json"
                scan, _, _ = read_bound_json(scan_path, rows[0]["sha256"])
                row = selected[MATRIX]
                if scan.get("compressed_sha256") != row["sha256"] or scan.get("compressed_bytes") != row["bytes"] \
                        or scan.get("decompressed_bytes") != row["decoded_bytes"] \
                        or scan.get("genes") != expected["genes"] or scan.get("cells") != expected["cells"] \
                        or scan.get("observed_entries") != expected["entries"] \
                        or scan.get("declared_entries") != expected["entries"] \
                        or scan.get("source_unchanged") is not True or scan.get("compressed_hash_complete") is not True \
                        or scan.get("gzip_crc32_isize_validated") is not True:
                    raise PreparationError("matrix scan evidence payload binding mismatch")
                report["matrix_scan_report_sha256"] = scan_sha
            report.update(status="blocked", metadata_preparation_sha256=metadata_sha,
                          scientific_admission=assess_admission(metadata, scan))
        report["elapsed_seconds"] = time.monotonic() - started
        report["runtime_resources"] = runtime_snapshot()
        if report["source_sha256"] != sha256_file(Path(__file__)) \
                or report["cli_sha256"] != sha256_file(Path(__file__).parents[2] / "scripts/prepare_gse175634_qualified.py"):
            raise PreparationError("preparation implementation changed during execution")
        report["output_files"] = _manifest(out)
        final_body = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if sum(item["bytes"] for item in report["output_files"]) + len(final_body) > bounds["max_output_bytes"]:
            raise PreparationError("total prepared output byte budget exceeded")
        _write_bytes_fresh(out / "report.json", final_body)
        return report
    except Exception as error:
        partial = getattr(error, "report", None)
        write_json_fresh(out / "FAILED.json", dict(status="failed", mode=mode, error=str(error),
            acceptance_sha256=receipt_sha, spec_sha256=spec_sha, partial_matrix_scan=partial, **NONAPPROVALS))
        raise
