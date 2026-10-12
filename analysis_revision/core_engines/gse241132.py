#!/usr/bin/env python3
"""GSE241132 join contract, fold-internal scaling, and historical-output lock.

Scripts 18-22 used to share a copy-pasted join that reset the pandas index
after merging author metadata, then treated that index as a row into coh.X.
This module is the single implementation those scripts must call.

It also owns two protocol rules the 2026-09-17 review made mandatory:

  * mu / theta standardisation is estimated on training cells only
  * historical report.json files must not be overwritten by a repaired rerun
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_engines.data_adapter import library_normalize, load_10x_zip_dir
from core_engines.topic_simplex_engine import TopicSimplexModel

SAMPLE_RE = re.compile(r"(GSM\d+)_(PWH\d+)(D\d+)\.zip$")
D2COND = {"D0": "Skin", "D1": "Wound1", "D7": "Wound7", "D30": "Wound30"}
COND_ORDER = ["Skin", "Wound1", "Wound7", "Wound30"]
COND_TIME = {"Skin": 0.0, "Wound1": 1 / 3, "Wound7": 2 / 3, "Wound30": 1.0}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPAIRED_ROOT = "outputs/repaired_20260918"
PROVENANCE_REPAIR_ROOT = "outputs/scientific_revision_20260926/provenance_repair"

HISTORICAL_OUTPUT_DIRS = (
    "outputs/human_temporal_flow",
    "outputs/time_parameterisation",
    "outputs/donor_transfer_flow",
    "outputs/donor_conditioned_benchmark",
    "outputs/negative_result_strength",
)

# Frozen hashes of the unrepaired reports. A repaired rerun that writes on
# top of these files is a protocol violation, not a refresh.
HISTORICAL_REPORT_SHA256 = {
    "outputs/human_temporal_flow/report.json":
        "218710984b0843bba8911b8d50d7a05cb7534c93d78537fc3e64acb1f1f10d51",
    "outputs/time_parameterisation/report.json":
        "829fb242a7f7541c5753d2d30e7299bd289d4cd6793a489d05cc66d367d074d8",
    "outputs/donor_transfer_flow/report.json":
        "0791a754bab2d8b814d75087f3f99191a82aa3ea9922adc510ab4d5061ca211d",
    "outputs/donor_conditioned_benchmark/report.json":
        "b3d408934ba491fbd76294e3d19b8e0473a0772f43152c095513095ffedb0987",
    "outputs/negative_result_strength/report.json":
        "37fb7bc2fa1e2308a534e394f5ca54b2679c60ef928bfc49001054b8f8a52c03",
    "outputs/direction1_dfu_mainline/report.json":
        "cede71a4ec29d5625eb88e675f8e2a22dc361e9fc37313dc8699cf21d4f0a123",
}

REQUIRED_PROTOCOL_KEYS = (
    "fold_internal_standardization",
    "held_out_target_not_in_scaler",
    "code_sha256",
    "panel_sha256",
    "model_sha256",
    "seed",
    "folds",
)

# Reports that analyse a cohort with no patient identifier or no healing
# endpoint. Each must carry its own scope flag, because the failure mode these
# guard against is a sample-level number being quoted as patient-level
# evidence once it leaves the report and lands in prose.
SCOPE_DECLARING_REPORTS = {
    "outputs/representation_benchmark/report.json": (
        "patient_level_inference_available", False),
    f"{REPAIRED_ROOT}/representation_inference/report.json": (
        "patient_level_inference_available", False),
    f"{REPAIRED_ROOT}/patient_mapping_equivalence/report.json": (
        "patient_level_inference_available", False),
    "outputs/metadata_intake/gse165816_patient_unit_remap/report.json": (
        "not_an_independent_cohort", True),
    "outputs/metadata_intake/next_public_cohort_contract/report.json": (
        "experiment_a_admitted", False),
    f"{REPAIRED_ROOT}/external_dfu_projection/report.json": (
        "status", "completed"),
    f"{REPAIRED_ROOT}/external_immune_axis/report.json": (
        "patient_level_inference_available", False),
    f"{REPAIRED_ROOT}/external_gse268834_projection/report.json": (
        "patient_level_inference_available", False),
    f"{REPAIRED_ROOT}/external_gse248247_projection/report.json": (
        "patient_level_inference_available", False),
    f"{REPAIRED_ROOT}/mouse_temporal_flow/report.json": (
        "status", "completed"),
    f"{PROVENANCE_REPAIR_ROOT}/donor_count_curve/report.json": (
        "status", "completed"),
    f"{REPAIRED_ROOT}/donor_curve_seed_stability/report.json": (
        "status", "completed"),
    f"{REPAIRED_ROOT}/design_power_simulation/report.json": (
        "patient_level_inference_available", False),
    f"{REPAIRED_ROOT}/patient_unit_representation_benchmark/report.json": (
        "not_an_independent_cohort", True),
    f"{REPAIRED_ROOT}/paired_anatomical_control/report.json": (
        "not_an_independent_cohort", True),
    f"{REPAIRED_ROOT}/mouse_all_arms_pod7_audit/report.json": (
        "not_statistical_inference", True),
    f"{REPAIRED_ROOT}/donor_curve_inference/report.json": (
        "plateau_claim_supported", False),
    f"{REPAIRED_ROOT}/mixture_semantic_inference/report.json": (
        "patient_level_inference_available", False),
    f"{REPAIRED_ROOT}/donor_geometry_inference/report.json": (
        "patient_level_inference_available", False),
}


class AlignmentError(RuntimeError):
    """Raised when a barcode does not select the matrix row it claims."""


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sample_map(per_gsm_dir: str) -> Dict[str, Dict[str, str]]:
    sample_of: Dict[str, Dict[str, str]] = {}
    for name in sorted(os.listdir(per_gsm_dir)):
        match = SAMPLE_RE.match(name)
        if match is None:
            continue
        gsm, donor, day = match.groups()
        cond = D2COND[day]
        sample_of[gsm] = {"donor": donor, "cond": cond, "label": donor + day}
    return sample_of


def barcodes_from_cells(cells: np.ndarray) -> np.ndarray:
    return np.array([str(cell).split(":", 1)[1] for cell in cells], dtype=object)


def join_from_zip_dir(coh, per_gsm_dir: str, meta_path: str,
                      extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    sample_of = parse_sample_map(per_gsm_dir)
    obs = coh.obs.copy()
    for key in ("donor", "cond", "label"):
        obs[key] = obs["gsm"].map(lambda gsm, key=key: sample_of.get(gsm, {}).get(key))
    if obs["label"].isna().any():
        missing = sorted(set(obs.loc[obs["label"].isna(), "gsm"].astype(str)))
        raise AlignmentError(f"GSM ids have no zip-name map: {missing[:8]}")
    raw_bc = barcodes_from_cells(coh.cells)
    obs["key"] = obs["label"].astype(str) + "_" + pd.Series(raw_bc).astype(str)
    obs["_raw_row"] = np.arange(len(obs), dtype=np.int64)
    meta = pd.read_csv(meta_path, sep="\t")
    wanted = [str(meta.columns[0]), "newMainCellTypes"]
    for col in extra_cols:
        if col in meta.columns and col not in wanted:
            wanted.append(col)
    meta = meta.loc[:, wanted]
    meta.columns = pd.Index(["key", "celltype"] + list(wanted[2:]))
    joined = obs.merge(meta, on="key", how="inner", sort=False,
                       validate="one_to_one")
    assert_row_contract(coh, joined)
    return joined


def assert_row_contract(coh, obs: pd.DataFrame) -> np.ndarray:
    if "_raw_row" not in obs.columns:
        raise AlignmentError("obs is missing _raw_row; refuse to guess matrix rows")
    row_idx = obs["_raw_row"].to_numpy(dtype=np.int64)
    raw_bc = barcodes_from_cells(coh.cells)
    expected = (obs["key"].astype(str).str.rsplit("_", n=1).str[-1]
                .to_numpy(dtype=object))
    if not np.array_equal(raw_bc[row_idx], expected):
        n_bad = int(np.sum(raw_bc[row_idx] != expected))
        raise AlignmentError(
            f"GSE241132 barcode-to-matrix row contract failed ({n_bad} rows)")
    return row_idx


def aligned_X(coh, obs: pd.DataFrame):
    return coh.X[assert_row_contract(coh, obs)]


def fold_standardize(mu: np.ndarray, train_mask: np.ndarray
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Centre/scale using training cells only; apply to the full matrix."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape[0] != mu.shape[0]:
        raise ValueError("train_mask length must match mu rows")
    n_train = int(train_mask.sum())
    if n_train < 2:
        raise ValueError(f"scaler needs >=2 training cells, got {n_train}")
    mean = mu[train_mask].mean(axis=0, keepdims=True)
    std = mu[train_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled = ((mu - mean) / std).astype(np.float32)
    return scaled, {
        "n_train_cells": n_train,
        "n_applied_cells": int(mu.shape[0]),
        "n_held_out_of_scaler": int((~train_mask).sum()),
        "train_mean_abs": float(np.abs(scaled[train_mask].mean())),
    }


def paired_endpoint_pairs(donors: np.ndarray, conds: np.ndarray,
                          donor_list: Sequence[str],
                          segments: Sequence[Tuple[str, str]],
                          min_cells: int = 20) -> List[Tuple[str, str, str]]:
    """Same-donor (donor, t0, t1) triples used by shared and conditioned fields."""
    pairs = []
    for donor in donor_list:
        for src, tgt in segments:
            n_src = int(np.sum((donors == donor) & (conds == src)))
            n_tgt = int(np.sum((donors == donor) & (conds == tgt)))
            if n_src >= min_cells and n_tgt >= min_cells:
                pairs.append((donor, src, tgt))
    if not pairs:
        raise ValueError("no donor-paired endpoints with enough cells")
    return pairs


def project_latent(X, genes, panel, model, dev, n_panel: int,
                   key: str = "mu") -> Tuple[np.ndarray, int]:
    if key not in {"mu", "theta"}:
        raise ValueError(f"key must be mu or theta, got {key!r}")
    pos = pd.Index(genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    n_cells = int(X.shape[0])
    mapped = sp.csr_matrix((sub.data, (sub.row, present[sub.col])),
                           shape=(n_cells, n_panel), dtype=np.int32)
    normalised = library_normalize(mapped)
    chunks = []
    with torch.no_grad():
        for start in range(0, n_cells, 4096):
            batch = torch.from_numpy(
                np.asarray(normalised[start:start + 4096].todense(),
                           dtype=np.float32)).to(dev)
            chunks.append(model(batch)[key].cpu().numpy())
    return np.vstack(chunks), int(present.size)


def load_discovery(discovery_dir: str):
    panel = np.load(os.path.join(discovery_dir, "panel.npy"), allow_pickle=True)
    beta = np.load(os.path.join(discovery_dir, "beta.npy"))
    state = torch.load(os.path.join(discovery_dir, "topic_model.pt"),
                       map_location="cpu")
    n_topics, n_panel = beta.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TopicSimplexModel(input_dim=n_panel, num_topics=n_topics).to(device)
    model.load_state_dict(state)
    model.eval()
    return panel, beta, model, n_topics, n_panel, device


def load_annotated_counts(per_gsm_dir: str, meta_path: str,
                          extra_cols: Sequence[str] = ()):
    coh = load_10x_zip_dir(per_gsm_dir)
    obs = join_from_zip_dir(coh, per_gsm_dir, meta_path, extra_cols=extra_cols)
    matrix = aligned_X(coh, obs)
    return coh, obs, matrix


def load_lineage_latent(per_gsm_dir: str, meta_path: str, discovery_dir: str,
                        lineage: str = "Fibroblast",
                        extra_cols: Sequence[str] = (),
                        key: str = "mu"):
    coh, obs, matrix = load_annotated_counts(per_gsm_dir, meta_path,
                                             extra_cols=extra_cols)
    panel, beta, model, n_topics, n_panel, device = load_discovery(discovery_dir)
    keep = np.flatnonzero((obs["celltype"] == lineage).to_numpy())
    obs_kept = obs.iloc[keep].reset_index(drop=True)
    latent, covered = project_latent(matrix[keep], coh.genes, panel, model,
                                     device, n_panel, key=key)
    return {
        "latent": latent,
        "obs": obs_kept,
        "n_topics": n_topics,
        "n_panel": n_panel,
        "covered": covered,
        "device": device,
        "panel": panel,
        "n_cells_joined": int(len(obs)),
        "n_cells_loaded": int(len(coh.obs)),
    }


def protocol_block(script_path: str, discovery_dir: str, seed: int,
                   folds: Iterable[str], extra: Optional[Dict[str, Any]] = None,
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
    # The output root is part of the run identity.  The earlier fixed value
    # pointed every fresh provenance rerun back to the historical repair root,
    # which made an otherwise valid report misleading to downstream readers.
    # Store a project-relative path so the report remains portable after the
    # repository is moved.
    actual_output_dir = output_dir or REPAIRED_ROOT
    if os.path.isabs(actual_output_dir):
        repaired_output_root = os.path.relpath(actual_output_dir, PROJECT_ROOT)
    else:
        repaired_output_root = actual_output_dir
    repaired_output_root = repaired_output_root.replace(os.sep, "/")
    block = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fold_internal_standardization": True,
        "held_out_target_not_in_scaler": True,
        "code_sha256": file_sha256(script_path),
        "helper_sha256": file_sha256(__file__),
        "panel_sha256": file_sha256(os.path.join(discovery_dir, "panel.npy")),
        "model_sha256": file_sha256(os.path.join(discovery_dir, "topic_model.pt")),
        "seed": int(seed),
        "folds": list(folds),
        "repaired_output_root": repaired_output_root,
    }
    if extra:
        block.update(extra)
    return block


def assert_not_historical(output_dir: str) -> None:
    abs_out = os.path.abspath(output_dir)
    for rel in HISTORICAL_OUTPUT_DIRS:
        if abs_out == os.path.abspath(os.path.join(PROJECT_ROOT, rel)):
            raise SystemExit(
                f"refusing to overwrite historical {rel}; "
                f"write under {REPAIRED_ROOT}/ instead")


def historical_reports_intact(root: str = PROJECT_ROOT) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "files": {}}
    for rel, expected in HISTORICAL_REPORT_SHA256.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            out["ok"] = False
            out["files"][rel] = {"status": "missing"}
            continue
        got = file_sha256(path)
        match = got == expected
        out["files"][rel] = {"status": "ok" if match else "hash_changed",
                             "sha256": got}
        if not match:
            out["ok"] = False
    return out


def repaired_report_ready(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"path": path, "ready": False, "reason": "missing"}
    try:
        report = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": path, "ready": False, "reason": str(exc)}
    proto = report.get("protocol") or {}
    missing = [k for k in REQUIRED_PROTOCOL_KEYS if k not in proto]
    if missing:
        return {"path": path, "ready": False,
                "reason": f"protocol missing {missing}"}
    if proto.get("fold_internal_standardization") is not True:
        return {"path": path, "ready": False,
                "reason": "fold_internal_standardization is not true"}
    return {"path": path, "ready": True, "reason": "ok"}


# The next-cohort pre-registration is pinned here rather than inside its own
# file, because a self-recorded hash cannot detect its own edit. Decisions that
# only mean something before the data exist - the clinical margin, the fold
# structure, the plateau criterion - stop meaning anything if they can be
# quietly adjusted later, so a silent edit has to break the release gate.
PREREGISTRATION_PATH = "config/next_cohort_preregistration.json"
PREREGISTRATION_SHA256 = (
    "84c4138dfa1c796ff54875bf589a380a9e2e016e14f09beb345fe81e830931d0")
PREREGISTRATION_REQUIRED_SECTIONS = (
    "experiment_a_patient_level_dfu",
    "experiment_b_representation_comparison",
    "experiment_c_donor_learning_curve",
    "experiment_d_causal",
    "shared_stopping_rules",
)


def preregistration_intact(root: str = PROJECT_ROOT) -> Dict[str, Any]:
    """The frozen analysis decisions must still be the frozen ones.

    Also refuses a pre-registration that has quietly acquired a clinical
    margin: the margin has to arrive with a documented clinical source, and
    filling it in from this repository would defeat the freeze.
    """
    path = os.path.join(root, PREREGISTRATION_PATH)
    if not os.path.exists(path):
        return {"ok": False, "reason": "missing"}
    digest = file_sha256(path)
    if digest != PREREGISTRATION_SHA256:
        return {"ok": False, "reason": "edited since it was frozen",
                "sha256": digest, "expected": PREREGISTRATION_SHA256}
    try:
        spec = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": str(exc)}
    missing = [s for s in PREREGISTRATION_REQUIRED_SECTIONS if s not in spec]
    if missing:
        return {"ok": False, "reason": f"missing sections {missing}"}
    mcid = (spec["experiment_a_patient_level_dfu"]
            .get("minimal_important_difference", {}))
    if mcid.get("value") is not None:
        return {"ok": False,
                "reason": "a clinical margin was filled in without an external "
                          "source; record the source and re-freeze"}
    curve = spec["experiment_c_donor_learning_curve"]
    if not curve.get("frozen_before_seeing_k_ge_3"):
        return {"ok": False,
                "reason": "the plateau criterion no longer claims to predate k>=3"}
    return {"ok": True, "reason": "ok", "sha256": digest,
            "preregistration_id": spec.get("preregistration_id"),
            "mcid_still_open": True,
            "minimum_points_before_plateau": curve.get(
                "minimum_points_before_any_plateau_statement")}


# Each report records the fingerprint of the script that produced it. Keeping
# the key present is decoration; comparing it with the file on disk is what
# makes a report traceable to code. An edited analysis that was never rerun is
# the failure this catches.
REPORT_CODE_LINKS = {
    f"{REPAIRED_ROOT}/human_temporal_flow/report.json":
        ("scripts/18_human_temporal_flow.py", "code_sha256"),
    f"{REPAIRED_ROOT}/donor_conditioned_benchmark/report.json":
        ("scripts/21_donor_conditioned_benchmark.py", "code_sha256"),
    f"{REPAIRED_ROOT}/negative_result_strength/report.json":
        ("scripts/22_strengthen_negative_result.py", "code_sha256"),
    "outputs/representation_benchmark/report.json":
        ("scripts/24_representation_benchmark.py", "script_sha256"),
    f"{REPAIRED_ROOT}/patient_mapping_equivalence/report.json":
        ("scripts/25_patient_mapping_equivalence.py", "script_sha256"),
    "outputs/metadata_intake/gse165816_patient_unit_remap/report.json":
        ("scripts/35_gse165816_patient_unit_remap.py", "script_sha256"),
    "outputs/metadata_intake/next_public_cohort_contract/report.json":
        ("scripts/36_next_public_cohort_contract.py", "script_sha256"),
    f"{REPAIRED_ROOT}/mouse_temporal_flow/report.json":
        ("scripts/26_mouse_temporal_flow_repaired.py", "script_sha256"),
    f"{REPAIRED_ROOT}/representation_inference/report.json":
        ("scripts/28_representation_inference.py", "script_sha256"),
    f"{REPAIRED_ROOT}/external_immune_axis/report.json":
        ("scripts/29_external_immune_axis_repair.py", "script_sha256"),
    f"{REPAIRED_ROOT}/external_gse268834_projection/report.json":
        ("scripts/40_gse268834_external_projection.py", "script_sha256"),
    f"{REPAIRED_ROOT}/external_gse248247_projection/report.json":
        ("scripts/41_gse248247_low_yield_projection.py", "script_sha256"),
    f"{PROVENANCE_REPAIR_ROOT}/time_parameterisation/report.json":
        ("scripts/19_time_parameterisation_fix.py", "code_sha256"),
    f"{PROVENANCE_REPAIR_ROOT}/donor_transfer_flow/report.json":
        ("scripts/20_donor_transfer_flow.py", "code_sha256"),
    f"{PROVENANCE_REPAIR_ROOT}/donor_count_curve/report.json":
        ("scripts/30_donor_count_curve.py", "code_sha256"),
    f"{REPAIRED_ROOT}/design_power_simulation/report.json":
        ("scripts/32_design_power_simulation.py", "script_sha256"),
    f"{REPAIRED_ROOT}/donor_curve_seed_stability/report.json":
        ("scripts/33_donor_curve_seed_stability.py", "code_sha256"),
    f"{REPAIRED_ROOT}/patient_unit_representation_benchmark/report.json":
        ("scripts/42_patient_unit_representation_benchmark.py", "script_sha256"),
    f"{REPAIRED_ROOT}/paired_anatomical_control/report.json":
        ("scripts/43_paired_anatomical_control.py", "script_sha256"),
    f"{REPAIRED_ROOT}/mouse_all_arms_pod7_audit/report.json":
        ("scripts/44_mouse_all_arms_pod7_audit.py", "script_sha256"),
    f"{REPAIRED_ROOT}/donor_curve_inference/report.json":
        ("scripts/45_donor_curve_inference.py", "script_sha256"),
    f"{REPAIRED_ROOT}/mixture_semantic_inference/report.json":
        ("scripts/46_mixture_semantic_inference.py", "script_sha256"),
    f"{REPAIRED_ROOT}/donor_geometry_inference/report.json":
        ("scripts/47_donor_geometry_inference.py", "script_sha256"),
}


def report_code_current(root: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Verify each report was produced by the script version now on disk."""
    files: Dict[str, Any] = {}
    for rel, (script_rel, key) in REPORT_CODE_LINKS.items():
        report_path = os.path.join(root, rel)
        script_path = os.path.join(root, script_rel)
        if not os.path.exists(report_path):
            files[rel] = {"current": False, "reason": "report missing"}
            continue
        if not os.path.exists(script_path):
            files[rel] = {"current": False, "reason": f"{script_rel} missing"}
            continue
        try:
            proto = (json.loads(open(report_path, encoding="utf-8").read())
                     .get("protocol") or {})
        except (OSError, json.JSONDecodeError) as exc:
            files[rel] = {"current": False, "reason": str(exc)}
            continue
        recorded = proto.get(key)
        on_disk = file_sha256(script_path)
        if recorded is None:
            files[rel] = {"current": False,
                          "reason": f"protocol has no {key}"}
        elif recorded != on_disk:
            files[rel] = {"current": False, "script": script_rel,
                          "reason": "script changed since this report was written; rerun it"}
        else:
            files[rel] = {"current": True, "script": script_rel, "reason": "ok"}
    return {"ok": all(item["current"] for item in files.values()), "files": files}


def scope_declaration_ready(root: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Every scope-limited report must state its own limit and its protocol.

    A missing flag is treated as a failure rather than a default, so a new
    analysis cannot enter the evidence set by staying silent about whether it
    supports patient-level inference.
    """
    files: Dict[str, Any] = {}
    for rel, (key, expected) in SCOPE_DECLARING_REPORTS.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            files[rel] = {"ready": False, "reason": "missing"}
            continue
        try:
            report = json.loads(open(path, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError) as exc:
            files[rel] = {"ready": False, "reason": str(exc)}
            continue
        if key not in report:
            files[rel] = {"ready": False, "reason": f"missing scope key {key!r}"}
            continue
        if report[key] != expected:
            files[rel] = {"ready": False,
                          "reason": f"{key}={report[key]!r}, expected {expected!r}"}
            continue
        if not isinstance(report.get("protocol"), dict) or not report["protocol"]:
            files[rel] = {"ready": False, "reason": "no protocol block"}
            continue
        files[rel] = {"ready": True, "reason": "ok",
                      "sha256": file_sha256(path)}
    return {"ok": all(item["ready"] for item in files.values()), "files": files}


def _scope_declaration_self_test() -> Dict[str, Any]:
    """Accept a compliant report; reject a silent one and a mislabelled one."""
    with tempfile.TemporaryDirectory() as tmp:
        rel = next(iter(SCOPE_DECLARING_REPORTS))
        key, expected = SCOPE_DECLARING_REPORTS[rel]
        path = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def write(payload):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

        others = {r: {k: v, "protocol": {"seed": 0}}
                  for r, (k, v) in SCOPE_DECLARING_REPORTS.items() if r != rel}
        for other_rel, payload in others.items():
            other_path = os.path.join(tmp, other_rel)
            os.makedirs(os.path.dirname(other_path), exist_ok=True)
            with open(other_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

        write({key: expected, "protocol": {"seed": 0}})
        compliant = scope_declaration_ready(tmp)["ok"]
        write({"protocol": {"seed": 0}})
        silent_rejected = not scope_declaration_ready(tmp)["ok"]
        write({key: "patient-level validated", "protocol": {"seed": 0}})
        mislabelled_rejected = not scope_declaration_ready(tmp)["ok"]
        write({key: expected})
        no_protocol_rejected = not scope_declaration_ready(tmp)["ok"]
    return {
        "compliant_report_accepted": bool(compliant),
        "silent_report_rejected": bool(silent_rejected),
        "mislabelled_scope_rejected": bool(mislabelled_rejected),
        "report_without_protocol_rejected": bool(no_protocol_rejected),
        "passes": bool(compliant and silent_rejected and mislabelled_rejected
                       and no_protocol_rejected),
    }


def _preregistration_self_test() -> Dict[str, Any]:
    """Accept the frozen file; reject an edit, a filled-in margin, a deletion."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, PREREGISTRATION_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        real = os.path.join(PROJECT_ROOT, PREREGISTRATION_PATH)
        spec = json.loads(open(real, encoding="utf-8").read())

        missing_rejected = not preregistration_intact(tmp)["ok"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(open(real, encoding="utf-8").read())
        frozen_accepted = preregistration_intact(tmp)["ok"]

        tampered = json.loads(json.dumps(spec))
        tampered["experiment_a_patient_level_dfu"][
            "minimal_important_difference"]["value"] = 0.0753
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(tampered, fh, indent=2)
        filled_margin_rejected = not preregistration_intact(tmp)["ok"]

        reworded = json.loads(json.dumps(spec))
        reworded["experiment_c_donor_learning_curve"][
            "plateau_criterion"] = "whatever the curve looks like"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(reworded, fh, indent=2)
        edit_rejected = not preregistration_intact(tmp)["ok"]

    return {
        "frozen_file_accepted": bool(frozen_accepted),
        "missing_file_rejected": bool(missing_rejected),
        "filled_in_margin_rejected": bool(filled_margin_rejected),
        "silent_edit_rejected": bool(edit_rejected),
        "passes": bool(frozen_accepted and missing_rejected
                       and filled_margin_rejected and edit_rejected),
    }


def run_contract_self_test() -> Dict[str, Any]:
    """Synthetic tests. Does not read GEO and does not print."""
    # Non-contiguous keep-set: the historical bug used the post-merge RangeIndex.
    n_raw = 8
    labels = np.array(["PWH26D0"] * 4 + ["PWH27D1"] * 4)
    barcodes = np.array([f"BC{i}-1" for i in range(n_raw)], dtype=object)
    keys = np.array([f"{lab}_{bc}" for lab, bc in zip(labels, barcodes)],
                    dtype=object)
    keep = np.array([0, 2, 5, 7], dtype=np.int64)

    class _Coh:
        cells = np.array([f"GSM{i}:{bc}" for i, bc in enumerate(barcodes)],
                         dtype=object)

    obs = pd.DataFrame({
        "key": keys[keep],
        "label": labels[keep],
        "_raw_row": keep,
    })
    meta = pd.DataFrame({"key": keys[keep][::-1],
                         "celltype": ["Fibroblast"] * len(keep)})
    joined = obs.merge(meta, on="key", how="inner", sort=False,
                       validate="one_to_one")
    raw_bc = barcodes
    expected = joined["key"].str.rsplit("_", n=1).str[-1].to_numpy(dtype=object)
    correct = np.array_equal(raw_bc[joined["_raw_row"].to_numpy(dtype=np.int64)],
                             expected)
    wrong_index = joined.index.to_numpy()
    reset_index_fails = not np.array_equal(raw_bc[wrong_index], expected)
    assert_row_contract(_Coh(), joined)

    rng = np.random.default_rng(0)
    mu = rng.normal(size=(120, 4))
    mu[80:] += 8.0
    train = np.zeros(120, dtype=bool)
    train[:80] = True
    scaled, stats = fold_standardize(mu, train)
    train_mean = float(np.abs(scaled[train].mean()))
    held_mean = float(np.abs(scaled[~train].mean()))
    scaler_ignores_holdout = train_mean < 1e-6 and held_mean > 1.0

    shuffled = joined.sample(frac=1.0, random_state=1).reset_index(drop=True)
    shuffled_ok = np.array_equal(
        raw_bc[shuffled["_raw_row"].to_numpy(dtype=np.int64)],
        shuffled["key"].str.rsplit("_", n=1).str[-1].to_numpy(dtype=object))

    scope = _scope_declaration_self_test()
    prereg = _preregistration_self_test()
    result = {
        "correct_raw_row_selects_barcode": bool(correct),
        "reset_index_is_unsafe": bool(reset_index_fails),
        "shuffled_meta_join_still_aligned": bool(shuffled_ok),
        "scaler_ignores_holdout": bool(scaler_ignores_holdout),
        "scope_declaration_gate": scope,
        "preregistration_gate": prereg,
        "scaler_stats": stats,
        "passes": bool(correct and reset_index_fails and shuffled_ok
                       and scaler_ignores_holdout and scope["passes"]
                       and prereg["passes"]),
    }
    if not result["passes"]:
        raise AssertionError(f"GSE241132 contract self-test failed: {result}")
    return result


def repaired_bundle_ready(root: str = PROJECT_ROOT) -> Dict[str, Any]:
    rels = [
        f"{REPAIRED_ROOT}/human_temporal_flow/report.json",
        f"{PROVENANCE_REPAIR_ROOT}/time_parameterisation/report.json",
        f"{PROVENANCE_REPAIR_ROOT}/donor_transfer_flow/report.json",
        f"{PROVENANCE_REPAIR_ROOT}/donor_count_curve/report.json",
        f"{REPAIRED_ROOT}/donor_conditioned_benchmark/report.json",
        f"{REPAIRED_ROOT}/negative_result_strength/report.json",
    ]
    files = {rel: repaired_report_ready(os.path.join(root, rel)) for rel in rels}
    return {"ok": all(item["ready"] for item in files.values()), "files": files}


if __name__ == "__main__":
    result = run_contract_self_test()
    print("GSE241132 contract tests:")
    for key, value in result.items():
        if key == "scaler_stats":
            continue
        if key in ("scope_declaration_gate", "preregistration_gate"):
            print(f"  {key}: passes={value['passes']}")
            continue
        print(f"  {key}: {value}")
    print("GSE241132 protocol helper PASSED!")
