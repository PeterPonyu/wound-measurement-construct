#!/usr/bin/env python3
"""Output-only precision correction: match the frozen worker's exact arithmetic.

The original source/config/test manifest is never changed. This runs the original
nine-job audit, state/projection/match replay and table generator, replacing only
the mismatched float64 tabular aggregation with the existing worker's float32
specimen and patient aggregation. No tolerances or scientific rules are relaxed.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import run_quality_topic_stability as sidecar


def patient_rows_exact_dtype(job, result, cells, theta):
    primary, cut = result.get("fibroblast_topic"), result.get("cut")
    available = primary is not None and cut is not None
    if available:
        sidecar.require(0 <= primary < theta.shape[1] and math.isfinite(cut), "invalid primary readout topic/cut")
        sidecar.require(theta.dtype == np.float32, "exact worker arithmetic requires float32 checkpoint theta")
    inventory, rows = sidecar.cohort_inventory(cells), []
    for scope in ("fibroblast_only", "all_cells"):
        mask = (cells.celltype == "fibroblast").to_numpy() if scope == "fibroblast_only" else np.ones(len(cells), bool)
        frame = cells.loc[mask, ["patient_id", "gsm"]].rename(columns={"patient_id": "unit", "gsm": "specimen"}).copy()
        patients = None
        if available and len(frame):
            frame["value"] = theta[mask, primary]  # Keep float32, exactly as existing worker.
            specimens = frame.groupby(["unit", "specimen"])["value"].agg(
                mean="mean", n_cells="size", fraction=lambda values: float(np.mean(values > cut)))
            patients = specimens.groupby("unit").agg(mean=("mean", "mean"), fraction=("fraction", "mean"),
                                                       n_cells=("n_cells", "sum"), n_specimens=("mean", "size"))
        for person in inventory["patients"]:
            block = frame[frame.unit == person["patient_id"]]
            status = ("available" if available and len(block) else "no_cells_in_scope" if not len(block)
                      else "primary_readout_unavailable_no_rescue")
            value = patients.loc[person["patient_id"]] if status == "available" else None
            rows.append({"job_id": job.job_id, "K": job.params["K"], "seed": job.seed,
                "patient_id": person["patient_id"], "group": person["group"], "is_DFU": person["is_DFU"],
                "readout_scope": scope, "n_cells": len(block), "n_specimens": block.specimen.nunique(),
                "status": status, "mean_loading": None if value is None else float(value["mean"]),
                "fraction_above_fibroblast_descriptive_cut": None if value is None else float(value["fraction"]),
                "cut_full_sample_fibroblasts": cut,
                "threshold_scope": "full_discovery_fibroblasts_descriptive_not_held_unit_density_cut",
                "patient_aggregation": "equal_specimen_mean_not_cell_pooled"})
    if available:
        existing = {str(row["unit"]): row for row in result["patient_readouts"]}
        observed = {row["patient_id"]: row for row in rows if row["readout_scope"] == "fibroblast_only"
                    and row["status"] == "available"}
        sidecar.require(set(existing) == set(observed), "worker/table patient coverage disagreement")
        for patient, row in observed.items():
            sidecar.require(np.allclose([row["mean_loading"], row["fraction_above_fibroblast_descriptive_cut"]],
                [existing[patient]["mean"], existing[patient]["fraction"]], rtol=0, atol=1e-12) and
                (row["n_cells"], row["n_specimens"]) == (existing[patient]["n_cells"], existing[patient]["n_specimens"]),
                "worker/table equal-specimen readout disagreement")
    return rows


def synthetic_check():
    from types import SimpleNamespace
    cells = pd.DataFrame({"gsm": ["s0", "s0", "s1"], "patient_id": ["p0"] * 3,
        "celltype": ["fibroblast"] * 3, "group": ["DFU-H"] * 3,
        "reconstructed_raw_cell_id": [f"synthetic-{i}" for i in range(3)]})
    values = np.array([.011234567, .09876543, .87654321], dtype=np.float32)
    theta = np.stack([values, 1 - values], axis=1)
    frame = pd.DataFrame({"unit": ["p0"] * 3, "specimen": ["s0", "s0", "s1"], "value": values})
    specimens = frame.groupby(["unit", "specimen"])["value"].agg(mean="mean", n_cells="size",
        fraction=lambda v: float(np.mean(v > .5)))
    patients = specimens.groupby("unit").agg(mean=("mean", "mean"), fraction=("fraction", "mean"),
        n_cells=("n_cells", "sum"), n_specimens=("mean", "size"))
    result = {"fibroblast_topic": 0, "cut": .5, "patient_readouts": patients.reset_index().to_dict(orient="records")}
    job = SimpleNamespace(job_id="synthetic-only", params={"K": 2}, seed=0)
    rows = patient_rows_exact_dtype(job, result, cells, theta)
    assert all(row["mean_loading"] == float(patients.loc["p0", "mean"]) for row in rows)
    result.update(fibroblast_topic=None, cut=None)
    missing = patient_rows_exact_dtype(job, result, cells, theta)
    assert len(missing) == 2 and all(row["mean_loading"] is None for row in missing)
    return {"synthetic_only": True, "exact_float32_equal_specimen_pooling": True,
            "strict_1e_12_tolerance_unchanged": True, "unavailable_primary_retained": True}


def main():
    if sys.argv[1:] == ["--synthetic-check"]:
        print(json.dumps(synthetic_check(), indent=2))
        return
    sidecar.require(not sys.argv[1:], "only --synthetic-check or approved replay is supported")
    from expansion.common.io import atomic_write_json, read_json, sha256_file
    import torch
    addendum_path = sidecar.OUTPUT / "precision_addendum.json"
    addendum = read_json(addendum_path)
    sidecar.require(addendum["status"] == "leader_reviewed_output_only_exact_dtype_correction",
                    "output-only precision correction still awaits leader review")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = sidecar.patient_rows
    try:
        sidecar.patient_rows = patient_rows_exact_dtype
        report = sidecar.verify()
    finally:
        sidecar.patient_rows = original
    report["verification_addendum"] = {"kind": "exact_worker_dtype_no_scientific_rule_or_tolerance_change",
        "script_sha256": sha256_file(__file__), "addendum_sha256": sha256_file(addendum_path),
        "frozen_source_files_unchanged": True, "synthetic_check": synthetic_check()}
    atomic_write_json(sidecar.OUTPUT / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
