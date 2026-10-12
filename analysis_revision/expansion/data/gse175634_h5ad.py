"""Hash-bound disk CSR to standard H5AD, without loading or casting full X.

AnnData 0.12.10's official on-disk specification defines the root and CSR
groups. Public anndata.io.write_elem writes obs/var/uns; h5py hyperslabs write
the three CSR arrays. This module never reads a MatrixMarket/source data root.
"""
from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import shutil
import time

from . import gse175634_preparation as identity
from . import gse175634_qc as qc


SCHEMA = "ortho.gse175634.h5ad-bridge.v1"
STATUS = "standard_H5AD_prepared_not_scientific_admission"
DOCS = {
    "anndata_format": "https://github.com/scverse/anndata/blob/0.12.10/docs/fileformat-prose.md",
    "anndata_public_element_writer": "https://anndata.readthedocs.io/en/latest/generated/anndata.io.write_elem.html",
    "h5py_datasets": "https://docs.h5py.org/en/3.16.0/high/dataset.html",
}
RESERVED = {"author_sample_raw", "time", "time_value", "cell_type", "_ortho_cell_axis", "_ortho_gene_axis", "_index"}
ARRAYS = {
    "data": ("counts.data.i64le.bin", "<i8", "<i8"),
    "indices": ("counts.indices.u32le.bin", "<u4", "<i8"),
    "indptr": ("counts.indptr.u64le.bin", "<u8", "<i8"),
}


class H5ADBridgeError(identity.PreparationError):
    """Binding, metadata, schema, precision, budget or roundtrip refusal."""


def policy(spec):
    cfg = spec.get("h5ad_runtime", {})
    for key in ("memory_bytes", "cpu_seconds", "io_chunk_bytes", "hdf5_cache_bytes",
                "metadata_decoded_bytes", "axis_decoded_bytes", "max_columns", "disk_margin_bytes"):
        identity._positive(cfg.get(key), key)
    if cfg["memory_bytes"] != 2 << 30 or not 8 <= cfg["io_chunk_bytes"] <= 1 << 20 \
            or cfg["io_chunk_bytes"] % 8 or cfg["hdf5_cache_bytes"] > 8 << 20 \
            or cfg["metadata_decoded_bytes"] > 128 << 20 or cfg["axis_decoded_bytes"] > 16 << 20 \
            or cfg["max_columns"] > 128 or cfg.get("post_write_free_fraction") != .3 \
            or cfg.get("compression") != "none_no_lossy_filters" \
            or cfg.get("obs_sample_rule") != "physical_specimen_preserve_raw_sample_as_author_sample_raw":
        raise H5ADBridgeError("unknown or unsafe H5AD resource/identity contract")
    return cfg


def dependency_versions(cfg):
    versions = {}
    for package in ("anndata", "h5py", "numpy", "scipy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise H5ADBridgeError(f"optional existing dependency missing: {package}; no installation attempted") from error
    for package in ("anndata", "h5py"):
        if versions[package] not in cfg.get("tested_dependency_versions", {}).get(package, []):
            raise H5ADBridgeError(f"unverified optional dependency version: {package}={versions[package]}")
    return versions


def rss_check(cfg):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > cfg["memory_bytes"]:
        raise H5ADBridgeError("process lifetime peak RSS exceeds 2GiB H5AD budget")
    return peak


def _header(path, cfg):
    with identity.regular_input(path) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
        line = gz.readline(65537)
    if len(line) > 65536:
        raise H5ADBridgeError("metadata header row budget exceeded")
    columns = next(csv.reader([line.decode("utf-8-sig")], delimiter="\t"), [])
    identity.require_columns(columns, [])
    if len(columns) > cfg["max_columns"] or any("/" in name or "\x00" in name for name in columns):
        raise H5ADBridgeError("HDF5 metadata column count/key budget or path collision")
    return columns


def load_context(qc_preparation, expected_qc_sha256, spec_path, expected_spec_sha256=None):
    path = identity.safe_path(qc_preparation, file=True)
    report, digest, body = identity.read_bound_json(path, expected_qc_sha256, cap=16 << 20)
    if report.get("schema") != "ortho.gse175634.duplicate-qc-csr.v1" or report.get("status") != "duplicate_QC_CSR_prepared_not_scientific_admission" \
            or any(report.get(flag) is not False for flag in identity.NONAPPROVALS) \
            or report.get("duplicate_coordinate_validation", {}).get("performed") is not True \
            or report.get("binary_CSR_artifact_written") is not True:
        raise H5ADBridgeError("successful hash-bound nonapproval QC/CSR report required")
    spec, spec_sha, spec_body = identity.load_spec(spec_path, expected_spec_sha256)
    cfg = policy(spec)
    root = identity.safe_path(path.parent, directory=True)
    source_spec, _, _ = identity.load_spec(root / "preparation-spec.json", report["spec_sha256"])
    rows, entries = report["QC"].get("retained_rows"), report["QC"].get("nonzero_entries")
    identity._positive(rows, "retained rows", zero=True)
    identity._positive(entries, "positive CSR records", zero=True)
    if rows > source_spec["expected"]["cells"] or entries > source_spec["expected"]["entries"] \
            or source_spec["expected"]["genes"] > qc.MAX_COUNT \
            or entries > qc.MAX_COUNT or any(day > 2**53 for day in source_spec["expected"]["days"]):
        raise H5ADBridgeError("signed index/exact physical-day representation exceeded")
    return dict(root=root, report=report, report_sha=digest, report_body=body, spec=spec,
                spec_sha=spec_sha, spec_body=spec_body, cfg=cfg, source_spec=source_spec)


def estimate_resources(context):
    cfg, report, expected = context["cfg"], context["report"], context["source_spec"]["expected"]
    obs_columns = _header(context["root"] / "author_cell_metadata_QC.tsv.gz", cfg)
    gene_columns = _header(context["root"] / "raw" / identity.GENE_INDEX, cfg)
    rows, entries = report["QC"]["retained_rows"], report["QC"]["nonzero_entries"]
    csr = entries * 16 + (rows + 1) * 8  # signed-int64 values AND indices
    maximum_file = csr + cfg["metadata_decoded_bytes"] * 2 + (16 << 20)
    # Two output generations reserved, even though the writer only retains one.
    planned = maximum_file * 2 + cfg["disk_margin_bytes"]
    memory = ((512 << 20) + cfg["metadata_decoded_bytes"] * 4
              + expected["cells"] * 64 * (len(obs_columns) + 5)
              + expected["genes"] * 64 * (len(gene_columns) + 2)
              + cfg["io_chunk_bytes"] * 8 + cfg["hdf5_cache_bytes"] * 4)
    if memory > cfg["memory_bytes"]:
        raise H5ADBridgeError("conservative obs/var/dependency memory estimate exceeds 2GiB bridge budget")
    return dict(CSR_int64_data_and_index_bytes=csr, maximum_H5AD_file_bytes=maximum_file,
        reserved_two_H5AD_generation_bytes=maximum_file * 2, planned_increment_bytes=planned,
        conservative_memory_bytes=memory, hard_memory_and_RSS_budget_bytes=cfg["memory_bytes"],
        post_write_free_fraction_required=.3, obs_columns=len(obs_columns), var_columns=len(gene_columns))


def resource_preflight(parent, context, estimate):
    usage = shutil.disk_usage(identity.safe_path(parent, directory=True))
    snapshot = identity.resource_snapshot(parent, context["spec"])
    remaining = usage.free - estimate["planned_increment_bytes"]
    checks = dict(disk_bytes=remaining >= context["spec"]["budgets"]["minimum_free_bytes"],
                  disk_fraction=remaining / usage.total >= .3,
                  available_memory=snapshot["memory_available_bytes"] is not None
                  and snapshot["memory_available_bytes"] >= context["cfg"]["memory_bytes"])
    return dict(status="passed" if all(checks.values()) else "blocked", checks=checks,
        disk_total_bytes=usage.total, disk_free_bytes=usage.free, post_write_free_bytes=remaining,
        memory_available_bytes=snapshot["memory_available_bytes"], **estimate)


def _table(path, required, expected_rows, cfg, cap):
    columns = _header(path, cfg)
    identity.require_columns(columns, required)
    values = {key: [] for key in columns}
    decoded = 0
    with identity.regular_input(path) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
        def bounded_lines():
            nonlocal decoded
            while line := gz.readline(65537):
                decoded += len(line)
                if len(line) > 65536 or decoded > cap:
                    raise H5ADBridgeError("obs/var decoded metadata/row budget exceeded")
                yield line
        reader = csv.DictReader((line.decode("utf-8-sig" if i == 0 else "utf-8")
            for i, line in enumerate(bounded_lines())), delimiter="\t")
        for position, row in enumerate(reader):
            if position >= expected_rows or None in row or any(v is None or "\x00" in v for v in row.values()):
                raise H5ADBridgeError("obs/var original table width/NUL/count conflict")
            for key in columns:
                values[key].append(row[key])
    if len(values[columns[0]]) != expected_rows:
        raise H5ADBridgeError("obs/var axis table incomplete")
    return columns, values


def load_frames(context):
    import numpy as np
    import pandas as pd

    cfg, root, report = context["cfg"], context["root"], context["report"]
    rows, genes = report["QC"]["retained_rows"], context["source_spec"]["expected"]["genes"]
    columns, obs_data = _table(root / "author_cell_metadata_QC.tsv.gz",
        ["cell", "individual", "line", "day", "diffday", "sample", "specimen", "physical_pool", "batch"],
        rows, cfg, cfg["metadata_decoded_bytes"])
    if set(columns) & RESERVED:
        raise H5ADBridgeError("raw obs column collides with standard/HDF5 reserved bridge keys")
    if any(obs_data["individual"][i] != obs_data["line"][i]
           or obs_data["diffday"][i] != "day" + obs_data["day"][i]
           or obs_data["sample"][i] != obs_data["line"][i] + "_" + obs_data["day"][i]
           for i in range(rows)):
        raise H5ADBridgeError("standard obs line/day/raw sample conflict")
    obs = pd.DataFrame(obs_data, dtype=object)
    obs.rename(columns={"sample": "author_sample_raw"}, inplace=True)
    obs["sample"] = obs_data["specimen"]
    obs["time"] = obs_data["diffday"]
    obs["time_value"] = np.array([int(day) for day in obs_data["day"]], dtype=np.int64)
    if "type" in obs_data:
        obs["cell_type"] = obs_data["type"]
    obs.index = pd.Index(obs_data["cell"], name="_ortho_cell_axis")
    gene_columns, gene_data = _table(root / "raw" / identity.GENE_INDEX,
        ["gene_index", "gene_name"], genes, cfg, cfg["axis_decoded_bytes"])
    if set(gene_columns) & RESERVED or gene_data["gene_index"] != [str(i+1) for i in range(genes)]:
        raise H5ADBridgeError("raw var column collision or original gene index order conflict")
    var = pd.DataFrame(gene_data, dtype=object)
    var.index = pd.Index(gene_data["gene_name"], name="_ortho_gene_axis")
    from .standard import _obs_problems
    problems = _obs_problems(obs)
    if problems:
        raise H5ADBridgeError("standard obs contract: " + "; ".join(problems))
    mapping = {key: "author_sample_raw" if key == "sample" else key for key in columns}
    del obs_data, gene_data
    gc.collect()
    if (obs.memory_usage(index=True, deep=True).sum() + var.memory_usage(index=True, deep=True).sum()) * 3 + (512 << 20) > cfg["memory_bytes"]:
        raise H5ADBridgeError("measured obs/var memory estimate exceeds bridge budget")
    rss_check(cfg)
    return obs, var, mapping


def provenance(context, mapping):
    report = context["report"]
    return dict(dataset_id="GSE175634", qc_report_sha256=context["report_sha"],
        acceptance_sha256=report["acceptance_sha256"], source_preparation_spec_sha256=report["spec_sha256"],
        bridge_spec_sha256=context["spec_sha"], raw_obs_column_mapping=mapping,
        obs_sample_rule="physical_pool_line_day_specimen; raw author sample preserved as author_sample_raw",
        individual_unit="human_iPSC_source_line_not_wound_donor_or_DFU_patient",
        count_precision="signed_int64_exact_no_float_cast", numeric_count_like_integrity=True,
        count_provenance_reviewed=False, biological_count_semantics_confirmed=False,
        publisher_checksum_verified=False, full_matrix_loaded=False, **identity.NONAPPROVALS)


def _size_check(path, estimate, cfg):
    if path.stat().st_size > estimate["maximum_H5AD_file_bytes"]:
        raise H5ADBridgeError("H5AD file exceeded planned disk byte budget")
    rss_check(cfg)


def write_h5ad(path, context, obs, var, mapping, estimate):
    import anndata
    import h5py
    import numpy as np

    cfg, root = context["cfg"], context["root"]
    path = identity.safe_path(path, fresh=True)
    expected_files = {item["file"]: item for item in context["report"]["output_files"]}
    uns = dict(standard=dict(dataset_id="GSE175634", count_type="raw", time_unit="day",
        provenance_json=json.dumps(provenance(context, mapping), sort_keys=True)),
        preparation=dict(**identity.NONAPPROVALS, count_provenance_reviewed=False,
            scientific_admission=False, physical_pool_purged_training_required=True,
            float32_adapter_conversion_approved=False))
    transfers = {}
    with h5py.File(path, "x", rdcc_nbytes=cfg["hdf5_cache_bytes"]) as store:
        store.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        for key, value in (("obs", obs), ("var", var), ("uns", uns),
                           ("layers", {}), ("obsm", {}), ("varm", {}), ("obsp", {}), ("varp", {})):
            anndata.io.write_elem(store, key, value)
        group = store.create_group("X")
        group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                            "shape": np.array([len(obs), len(var)], dtype=np.int64)})
        for key, (name, source_dtype, target_dtype) in ARRAYS.items():
            source_dtype, target_dtype = np.dtype(source_dtype), np.dtype(target_dtype)
            item = expected_files[name]
            length = item["bytes"] // source_dtype.itemsize
            if item["bytes"] % source_dtype.itemsize:
                raise H5ADBridgeError("partial bound source CSR array")
            target = group.create_dataset(key, shape=(length,), dtype=target_dtype)
            digest, offset = hashlib.sha256(), 0
            with identity.regular_input(root / name) as raw:
                while block := raw.read(cfg["io_chunk_bytes"]):
                    if len(block) % source_dtype.itemsize:
                        raise H5ADBridgeError("partial input CSR record while bridging")
                    digest.update(block)
                    values = np.frombuffer(block, dtype=source_dtype)
                    if source_dtype.kind == "u" and values.size and int(values.max()) > qc.MAX_COUNT:
                        raise H5ADBridgeError("signed H5AD CSR index precision overflow")
                    target[offset:offset+len(values)] = values.astype(target_dtype, copy=False)
                    offset += len(values)
                    store.flush()
                    _size_check(path, estimate, cfg)
            if offset != length or digest.hexdigest() != item["sha256"]:
                raise H5ADBridgeError("CSR source hash/length changed during H5AD conversion")
            transfers[key] = dict(elements=length, source_sha256=digest.hexdigest(),
                source_dtype=source_dtype.str, H5AD_dtype=target_dtype.str, full_array_loaded=False)
        store.flush()
        _size_check(path, estimate, cfg)
    return transfers


def verify_h5ad(path, context, obs, var, mapping, estimate):
    """All X bytes compared to source in bounded hyperslabs; AnnData backed replay."""
    import anndata
    import h5py
    import numpy as np
    import pandas as pd

    cfg, root = context["cfg"], context["root"]
    with identity.regular_input(path) as raw, h5py.File(raw, "r", rdcc_nbytes=cfg["hdf5_cache_bytes"]) as store:
        if dict(store.attrs) != {"encoding-type": "anndata", "encoding-version": "0.1.0"} \
                or store["X"].attrs["encoding-type"] != "csr_matrix" \
                or store["X"].attrs["encoding-version"] != "0.1.0" \
                or list(store["X"].attrs["shape"]) != [len(obs), len(var)]:
            raise H5ADBridgeError("official H5AD root/CSR encoding or axes mismatch")
        pd.testing.assert_frame_equal(anndata.io.read_elem(store["obs"]), obs, check_dtype=True)
        pd.testing.assert_frame_equal(anndata.io.read_elem(store["var"]), var, check_dtype=True)
        standard = anndata.io.read_elem(store["uns"]["standard"])
        if standard["dataset_id"] != "GSE175634" or standard["count_type"] != "raw" or standard["time_unit"] != "day" \
                or json.loads(standard["provenance_json"]) != provenance(context, mapping):
            raise H5ADBridgeError("standard H5AD provenance/identity lost binding")
        prep = anndata.io.read_elem(store["uns"]["preparation"])
        if any(bool(prep[flag]) for flag in identity.NONAPPROVALS) or bool(prep["scientific_admission"]) \
                or bool(prep["float32_adapter_conversion_approved"]):
            raise H5ADBridgeError("H5AD falsely promoted to training/scientific approval")
        for key, (name, source_dtype, target_dtype) in ARRAYS.items():
            source_dtype, target_dtype = np.dtype(source_dtype), np.dtype(target_dtype)
            target = store["X"][key]
            if target.dtype != target_dtype or target.ndim != 1:
                raise H5ADBridgeError("H5AD exact CSR dtype or sparse representation mismatch")
            offset = 0
            with identity.regular_input(root / name) as source:
                while block := source.read(cfg["io_chunk_bytes"]):
                    values = np.frombuffer(block, dtype=source_dtype)
                    if not np.array_equal(target[offset:offset+len(values)], values.astype(target_dtype, copy=False)):
                        raise H5ADBridgeError("H5AD streamed CSR value/index/offset conservation failed")
                    offset += len(values)
            if target.shape != (offset,):
                raise H5ADBridgeError("H5AD CSR array length mismatch")
            rss_check(cfg)
    # Never call standard.load_standard on a target-scale artifact: it is an
    # in-memory consumer. Backed mode is the bounded production roundtrip.
    backed = anndata.read_h5ad(path, backed="r")
    try:
        if backed.shape != (len(obs), len(var)) or backed.X.dtype != np.dtype("int64"):
            raise H5ADBridgeError("AnnData backed CSR shape/count dtype mismatch")
        pd.testing.assert_frame_equal(backed.obs, obs, check_dtype=True)
        pd.testing.assert_frame_equal(backed.var, var, check_dtype=True)
        if len(obs):
            row = backed.X[0:1]
            if row.dtype != np.dtype("int64") or row.shape != (1, len(var)):
                raise H5ADBridgeError("AnnData backed sparse row API mismatch")
    finally:
        backed.file.close()
    _size_check(Path(path), estimate, cfg)
    return dict(passed=True, official_schema=True, all_CSR_arrays_compared_to_bound_source=True,
        raw_obs_and_var_preserved=True, exact_count_dtype="int64", backed_anndata_read_verified=True,
        full_matrix_loaded=False, standard_load_target_executed=False,
        consumer_boundary="standard.load_standard is compatible but unbounded at target scale; use backed access or a separately resource-qualified load")


def prepare_h5ad(*, qc_preparation, expected_qc_sha256, out, spec_path, expected_spec_sha256=None):
    context = load_context(qc_preparation, expected_qc_sha256, spec_path, expected_spec_sha256)
    out = identity.safe_path(out, fresh=True)
    root = context["root"]
    if out == root or out.is_relative_to(root) or root.is_relative_to(out) \
            or identity.safe_path(spec_path).is_relative_to(out):
        raise H5ADBridgeError("fresh H5AD outputs must be disjoint from bound QC/CSR and spec inputs")
    cfg, estimate = context["cfg"], estimate_resources(context)
    preflight = resource_preflight(out.parent, context, estimate)
    if preflight["status"] != "passed":
        raise H5ADBridgeError(f"H5AD resource preflight blocked: {preflight['checks']}")
    versions = dependency_versions(cfg)
    rss_check(cfg)
    source_replay = qc.verify_artifact(root, context["report_sha"])
    obs, var, mapping = load_frames(context)
    implementation = Path(__file__)
    cli = implementation.parents[2] / "scripts/prepare_gse175634_h5ad.py"
    consumer = implementation.parent / "standard.py"
    hashes = {"source_sha256": identity.sha256_file(implementation), "cli_sha256": identity.sha256_file(cli),
              "existing_standard_consumer_sha256": identity.sha256_file(consumer),
              "QC_replayer_source_sha256": identity.sha256_file(Path(qc.__file__)),
              "identity_replayer_source_sha256": identity.sha256_file(Path(identity.__file__))}
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        identity._write_bytes_fresh(out / "QC-report.json", context["report_body"])
        identity._write_bytes_fresh(out / "bridge-spec.json", context["spec_body"])
        path = out / "GSE175634.standard.h5ad"
        transfers = write_h5ad(path, context, obs, var, mapping, estimate)
        roundtrip = verify_h5ad(path, context, obs, var, mapping, estimate)
        # A changing QC input must never be converted into a successful output.
        qc.verify_artifact(root, context["report_sha"])
        for key, source in (("source_sha256", implementation), ("cli_sha256", cli), ("existing_standard_consumer_sha256", consumer),
                            ("QC_replayer_source_sha256", Path(qc.__file__)), ("identity_replayer_source_sha256", Path(identity.__file__))):
            if identity.sha256_file(source) != hashes[key]:
                raise H5ADBridgeError("bridge/consumer source changed during processing")
        report = dict(schema=SCHEMA, status=STATUS, **hashes, qc_preparation_sha256=context["report_sha"],
            acceptance_sha256=context["report"]["acceptance_sha256"], bridge_spec_sha256=context["spec_sha"],
            source_preparation_spec_sha256=context["report"]["spec_sha256"], qc_root=str(root),
            dependency_versions=versions, official_schema_documentation=DOCS, resource_preflight=preflight,
            source_QC_artifact_replay=source_replay, CSR_transfers=transfers, roundtrip=roundtrip,
            raw_obs_column_mapping=mapping, standard_sample="physical_specimen", individual_unit="source_cell_line_not_wound_donor",
            count_semantics_reviewed=False, target_matrix_read_by_bridge=False,
            input_target_matrix_processing_performed=context["report"]["target_real_matrix_processing_performed"],
            runtime_resources=identity.runtime_snapshot(stage="binary_CSR_to_standard_H5AD"),
            elapsed_seconds=time.monotonic()-started, observed_H5AD_file_bytes=path.stat().st_size,
            consumer_precision_boundary="make_standard and generic adapters cast float32; those paths are NOT approved by this bridge",
            remaining_gates=["target_scale_execution_and_resources", "author_count_semantics_and_orig_ident_review",
                "separate_resource_and_precision_qualification_for_full_in_memory_or_float32_consumers",
                "explicit_scientific_Stage_A_Stage_B_authority"], **identity.NONAPPROVALS)
        rss_check(cfg)
        report["output_files"] = identity._manifest(out)
        body = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if sum(item["bytes"] for item in report["output_files"]) + len(body) > estimate["planned_increment_bytes"]:
            raise H5ADBridgeError("aggregate H5AD/report output budget exceeded")
        identity._write_bytes_fresh(out / "report.json", body)
        return report
    except Exception as error:
        identity.write_json_fresh(out / "FAILED.json", dict(status="failed", error=str(error),
            qc_preparation_sha256=context["report_sha"], partial_outputs_retained=True, **identity.NONAPPROVALS))
        raise
