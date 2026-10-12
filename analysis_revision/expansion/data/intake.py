"""Approval-bound Stage-A input closure, separate from scientific acceptance.

No candidate is qualified by this module. The operator must review author
identity/modality/count semantics first and bind an exact adapter and file set.
Only those verified files enter the new adapter mirror; archived triplets are
selectively extracted before the adapter runs. Generic fallback field matching
is not permitted in this launch path.
"""
from __future__ import annotations

from dataclasses import replace
import gzip
import os
from pathlib import Path
import shutil
import time
import zipfile

from ..common.io import atomic_write_json, content_hash, sha256_file
from .quarantine import ArchiveLimits, extract_quarantine, inspect_archive
from .registry import load_registry
from .secure_transfer import _safe_destination, _strict_json, verify_transfer_receipt


class IntakeContractError(ValueError):
    """Dataset qualification or consumed input closure is incomplete."""


ADAPTERS = {"tar_triplets", "tar_dense_csv", "dir_triplets", "zip_triplets", "dir_h5",
            "mtx_with_cell_metadata", "dense_table_with_cell_metadata", "h5ad"}
PATH_KEYS = {"archive", "series_matrix", "matrix", "genes", "cells", "cell_metadata", "table", "file"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def _bounded_gzip(path, limits):
    """Check decoded gzip input before generic matrix/table readers see it."""
    total, started = 0, time.monotonic()
    with gzip.open(path, "rb") as stream:
        while block := stream.read(limits.chunk_bytes):
            total += len(block)
            if total > limits.max_member_bytes or time.monotonic() - started > 120:
                raise IntakeContractError(f"gzip decoded byte/time cap exceeded: {path}")
    return total


def _check_rna_features(path, limits):
    """Generic triplet adapters may not silently mix RNA with HTO/ADT/ATAC."""
    opener = gzip.open if path.name.endswith(".gz") else open
    total = 0
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            total += len(line.encode("utf-8"))
            if total > limits.max_member_bytes:
                raise IntakeContractError("RNA feature metadata exceeds decoded byte cap")
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 3 and fields[2] != "Gene Expression":
                raise IntakeContractError("mixed RNA/HTO/ADT features need a separately qualified RNA filtering/demultiplexing adapter")


def qualified_intake(root, receipt_path, approval_path, qualification_path, ids, *, limits=None):
    """Read-only verification before any Stage-A output is created."""
    limits = limits or ArchiveLimits()
    ids = list(ids or [])
    if not ids or len(set(ids)) != len(ids):
        raise IntakeContractError("Stage A requires explicit unique dataset ids")
    problems = verify_transfer_receipt(root, receipt_path, approval_path=approval_path, dataset_ids=ids)
    if problems:
        raise IntakeContractError(f"transfer receipt is incomplete: {problems}")
    receipt = _strict_json(Path(receipt_path))
    qualification = _strict_json(Path(qualification_path))
    allowed = {"schema_version", "kind", "approved", "reviewed_by", "review_notes",
               "transfer_receipt_sha256", "datasets"}
    if set(qualification) != allowed:
        raise IntakeContractError("qualification keys must match the exact contract")
    if qualification.get("schema_version") != 1 or qualification.get("kind") != "sdd_stage_a_qualification" \
            or qualification.get("approved") is not True:
        raise IntakeContractError("Stage-A qualification is missing or an unapproved draft")
    if qualification.get("transfer_receipt_sha256") != sha256_file(receipt_path):
        raise IntakeContractError("qualification is not bound to the current transfer receipt")
    for key in ("reviewed_by", "review_notes"):
        if not isinstance(qualification.get(key), str) or not qualification[key].strip():
            raise IntakeContractError(f"qualification lacks {key}")
    records = qualification.get("datasets")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records) \
            or {row.get("dataset_id") for row in records} != set(ids) or len(records) != len(ids):
        raise IntakeContractError("qualification must cover exactly the selected dataset ids")
    entries = {entry.id: entry for entry in load_registry()}
    root = Path(root).resolve()
    plans = []
    for row in records:
        if set(row) != {"dataset_id", "registry_entry_sha256", "author_identity_reviewed", "modality_reviewed",
                        "count_semantics_reviewed", "adapter", "adapter_args", "required_files", "archive_members",
                        "reserved_peak_increment_bytes", "identity_evidence"}:
            raise IntakeContractError("dataset qualification keys must match the exact contract")
        if type(row["reserved_peak_increment_bytes"]) is not int or row["reserved_peak_increment_bytes"] <= 0:
            raise IntakeContractError("Stage-A peak disk reserve must be a known positive integer")
        evidence = row["identity_evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
            raise IntakeContractError("author identity evidence must have a path and SHA-256")
        if not Path(evidence["path"]).is_file() or sha256_file(evidence["path"]) != evidence["sha256"]:
            raise IntakeContractError("author identity evidence is missing or changed")
        dataset_id = row["dataset_id"]
        if dataset_id not in entries:
            raise IntakeContractError(f"unknown dataset id: {dataset_id}")
        entry = entries[dataset_id]
        signature = content_hash({"id": entry.id, "local_path": entry.local_path,
                                  "adapter": entry.adapter, "adapter_args": entry.adapter_args}, length=64)
        if row.get("registry_entry_sha256") != signature:
            raise IntakeContractError(f"registry qualification drift: {dataset_id}")
        for key in ("author_identity_reviewed", "modality_reviewed", "count_semantics_reviewed"):
            if row.get(key) is not True:
                raise IntakeContractError(f"{dataset_id}: {key} must be explicitly reviewed")
        adapter, args = row.get("adapter"), row.get("adapter_args")
        if adapter not in ADAPTERS or not isinstance(args, dict) or not isinstance(args.get("fields", {}), dict):
            raise IntakeContractError(f"unsupported qualified adapter: {dataset_id}")
        if any(not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str)
               for values in args.get("fields", {}).values()):
            raise IntakeContractError("qualified field bindings must each have one exact author column")
        if adapter not in {"mtx_with_cell_metadata", "dense_table_with_cell_metadata"} \
                and (args.get("cell_metadata") or args.get("cell_key")):
            raise IntakeContractError("selected adapter does not consume cell_metadata/cell_key; a dedicated join adapter is required")
        if args.get("count_type", "raw") != "raw":
            raise IntakeContractError("qualified launch supports validated raw counts only; transformed/estimated counts need a separate protocol")
        for key in PATH_KEYS:
            if args.get(key):
                _safe_destination(args[key])
        if args.get("directory", ".") != ".":
            _safe_destination(args["directory"])
        local = _safe_destination(entry.local_path)
        args = {**args, "strict_fields": True, "strict_metadata": True, "strict_counts": True, "min_join_fraction": 1.0}
        if "constant_time" not in args and "time" not in args.get("fields", {}):
            raise IntakeContractError("qualified adapter needs exact time field or explicit constant_time")
        if not args.get("fields", {}).get("individual") and not args.get("individual_from"):
            raise IntakeContractError("qualified adapter requires an explicit author individual field")
        if args.get("cell_metadata") and (not isinstance(args.get("cell_key"), list) or len(args["cell_key"]) != 1):
            raise IntakeContractError("qualified metadata adapter requires one exact cell_key")
        required = row.get("required_files")
        if not isinstance(required, list) or not required or len(set(required)) != len(required):
            raise IntakeContractError("required_files must be an explicit nonempty unique list")
        archives = row.get("archive_members", {})
        if not isinstance(archives, dict):
            raise IntakeContractError("archive_members must be an exact member allowlist mapping")
        inventory = {}
        minimum_bytes = 0
        for relative in required:
            _safe_destination(relative)
            if not relative.startswith(local + "/"):
                raise IntakeContractError(f"input is outside its dataset directory: {relative}")
            item = receipt["items"].get(f"{dataset_id}::{relative}")
            if item is None:
                raise IntakeContractError(f"missing required transferred file: {relative}")
            minimum_bytes += item["expected_bytes"]
            if relative.lower().endswith(ARCHIVE_SUFFIXES):
                info = inspect_archive(root / relative, limits=limits)
                selected = archives.get(relative)
                if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
                    raise IntakeContractError(f"archive needs an explicit nonempty unique member allowlist: {relative}")
                names = {member["name"] for member in info["members"] if not member["is_dir"]}
                if not set(selected) <= names:
                    raise IntakeContractError(f"selected archive member is missing: {relative}")
                if adapter == "tar_dense_csv" and set(selected) != names:
                    raise IntakeContractError("dense TAR adapter cannot safely consume an unselected subset")
                inventory[relative] = {"inventory": info, "selected": selected}
                selected_bytes = sum(member["size"] for member in info["members"] if member["name"] in selected)
                # ZIP quarantine retains selected members and a ZIP_STORED
                # mirror at once. Do not reserve only one copy of decoded data.
                minimum_bytes += selected_bytes * (2 if relative.lower().endswith(".zip") else 1)
                if relative.lower().endswith(".zip"):
                    minimum_bytes += sum(2 * len(name.encode("utf-8")) + 256 for name in selected) + 1024
            elif relative.lower().endswith(".gz"):
                minimum_bytes += _bounded_gzip(root / relative, limits)
        if set(archives) != set(inventory):
            raise IntakeContractError("archive allowlists must cover exactly the required archives")
        archive_name = f"{local}/{args.get('archive', '')}"
        if adapter in {"tar_triplets", "tar_dense_csv"} and archive_name not in inventory:
            raise IntakeContractError("TAR adapter requires a verified archive inventory")
        for key in PATH_KEYS:
            if args.get(key) and f"{local}/{args[key]}" not in required:
                raise IntakeContractError(f"adapter input {key} is not in required_files")
        if row["reserved_peak_increment_bytes"] < minimum_bytes:
            raise IntakeContractError("Stage-A peak reserve is smaller than the verified input closure")
        hashes = {relative: receipt["items"][f"{dataset_id}::{relative}"]["sha256"] for relative in required}
        plans.append({"entry": replace(entry, adapter=adapter, adapter_args=args, adapter_status="pending_probe"),
                      "source_sha256": hashes,
                      "reserved_peak_increment_bytes": row["reserved_peak_increment_bytes"],
                      "required_files": required, "archives": inventory})
    return plans


def materialize_intake(plans, source_root, destination, *, limits: ArchiveLimits):
    """Fresh verified mirror; original transfer files and receipt remain unchanged."""
    destination, source_root = Path(destination), Path(source_root)
    for ancestor in (destination, *destination.parents):
        if ancestor.is_symlink():
            raise IntakeContractError(f"symlink in adapter mirror ancestry: {ancestor}")
    if destination.exists() or destination.is_symlink():
        raise IntakeContractError("adapter mirror must be new")
    destination.mkdir(parents=True)
    entries = []
    for plan in plans:
        entry = plan["entry"]
        args = dict(entry.adapter_args)
        original_archive = args.get("archive")
        for relative in plan["required_files"]:
            source, target = source_root / relative, destination / relative
            if source.is_symlink() or sha256_file(source) != plan["source_sha256"][relative]:
                raise IntakeContractError(f"verified adapter input changed before copy: {relative}")
            if entry.adapter == "tar_triplets" and relative == f"{entry.local_path}/{original_archive}":
                extracted = target.parent / "selected-rna"
                result = extract_quarantine(source, extracted, allowlist=plan["archives"][relative]["selected"], limits=limits)
                # Legacy directory adapter scans one directory. Flatten only
                # explicitly selected files, rejecting basename collisions.
                flat = target.parent / "selected-rna-flat"
                flat.mkdir()
                for member in result["selected"]:
                    original = extracted / member["name"]
                    if member["name"].lower().endswith(".gz"):
                        _bounded_gzip(original, limits)
                    if "features" in original.name.lower():
                        _check_rna_features(original, limits)
                    try:
                        os.link(original, flat / original.name)
                    except FileExistsError as error:
                        raise IntakeContractError("selected archive basenames collide for the directory adapter") from error
                args.pop("archive")
                args["directory"] = "selected-rna-flat"
            elif relative.lower().endswith(".zip"):
                extracted = destination / "_archive_quarantine" / entry.id / str(len(entries)) / target.name
                extract_quarantine(source, extracted, allowlist=plan["archives"][relative]["selected"], limits=limits)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED) as archive:
                    for name in plan["archives"][relative]["selected"]:
                        if name.lower().endswith(".gz"):
                            _bounded_gzip(extracted / name, limits)
                        if entry.adapter == "zip_triplets" and "features" in Path(name).name.lower():
                            _check_rna_features(extracted / name, limits)
                        archive.write(extracted / name, arcname=name)
            elif entry.adapter == "tar_dense_csv" and relative == f"{entry.local_path}/{original_archive}":
                extracted = target.parent / "validated-dense-members"
                result = extract_quarantine(source, extracted, allowlist=plan["archives"][relative]["selected"], limits=limits)
                decoded = 0
                for member in result["selected"]:
                    original = extracted / member["name"]
                    decoded += _bounded_gzip(original, limits) if original.name.endswith(".gz") else member["bytes"]
                    if decoded > limits.max_total_bytes:
                        raise IntakeContractError("dense TAR inner decoded total exceeds cap")
                if decoded > plan["reserved_peak_increment_bytes"]:
                    raise IntakeContractError("dense TAR inner decoded bytes exceed approved peak reserve")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if sha256_file(target) != plan["source_sha256"][relative]:
                    raise IntakeContractError("verified dense TAR changed during copy")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if sha256_file(target) != plan["source_sha256"][relative]:
                    raise IntakeContractError(f"adapter copy hash mismatch: {relative}")
                if entry.adapter == "dir_triplets" and "features" in target.name.lower():
                    _check_rna_features(target, limits)
        entries.append(replace(entry, adapter="dir_triplets" if entry.adapter == "tar_triplets" else entry.adapter,
                               adapter_args=args))
    atomic_write_json(destination / "intake_inventory.json", {
        "datasets": [{"dataset_id": p["entry"].id, "required_files": p["required_files"],
                      "archives": p["archives"]} for p in plans], "scientific_acceptance": False})
    return entries
