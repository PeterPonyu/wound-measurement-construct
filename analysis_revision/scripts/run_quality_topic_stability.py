#!/usr/bin/env python3
"""Freeze, execute after review, and replay a bounded nine-fit P1 sidecar.

Only this sidecar's canonical output tree is written. Preparation never fits or
approves execution. Existing measurement/queue/checkpoint APIs remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "config/quality_topic_stability_20261009.json"
OWNED = ("config/quality_topic_stability_20261009.json", "scripts/run_quality_topic_stability.py",
         "tests/test_quality_topic_stability.py")
OUTPUT = ROOT / "output/publication-quality-20261009/p1"
EXPECTED_MODEL, EXPECTED_EFFORT = "gpt-6.1-sol", "xhigh"
GRID = {(k, s, 40000) for k in (10, 15, 20) for s in (0, 1, 2)}
COMPARISONS = ("M1_vs_C1", "M2_vs_C2", "M1_vs_C2", "H1_vs_C1", "M1_vs_H1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def validate_runtime(metadata, context):
    reported = {"thread_id": metadata.get("id"), "session_id": metadata.get("session_id"),
                "parent_thread_id": metadata.get("parent_thread_id"), "turn_id": context.get("turn_id"),
                "model": context.get("model"), "reasoning_effort": context.get("effort")}
    require(reported["thread_id"] and reported["session_id"] and reported["turn_id"],
            "runtime session/thread/turn metadata absent; do not edit or execute")
    require((reported["model"], reported["reasoning_effort"]) == (EXPECTED_MODEL, EXPECTED_EFFORT),
            "runtime model/effort mismatch; do not edit or execute")
    return {"requested": {"model": EXPECTED_MODEL, "reasoning_effort": EXPECTED_EFFORT},
            "runtime_reported": reported}


def runtime_record():
    thread = os.environ.get("CODEX_THREAD_ID")
    require(thread, "CODEX_THREAD_ID missing; runtime verification unavailable")
    paths = list((Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "sessions").glob(
        f"*/*/*/rollout-*-{thread}.jsonl"))
    require(len(paths) == 1, "own runtime session file missing or ambiguous")
    meta, context = None, None
    with paths[0].open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            # Full-history forks can include a later replay of the parent's
            # session_meta; only this file's actual thread identity is ours.
            if row.get("type") == "session_meta" and row["payload"].get("id") == thread:
                meta = row["payload"]
            elif row.get("type") == "turn_context":
                context = row["payload"]
    require(meta is not None and context is not None and meta.get("id") == thread,
            "own runtime turn_context absent or belongs to a different thread")
    return {**validate_runtime(meta, context), "runtime_file": str(paths[0])}


def validate_protocol(p):
    s, r, a = p["stability"], p["resources"], p["analysis"]
    require(p["kind"] == "measurement" and p["cohorts"] == {"discovery": "GSE165816", "external": []},
            "only the frozen existing discovery cohort is allowed")
    require(s["K"] == [10, 15, 20] and s["seeds"] == [0, 1, 2] and
            all(type(v) is int for v in s["K"] + s["seeds"]), "capacity/seed grid changed")
    require(s["steps"] == 40000 and type(s["steps"]) is int and s["step_budgets"] == [40000],
            "every fit requires exactly 40000 optimizer steps")
    require(s["batch_size"] == 256 and s["lr"] == .001 and s["kl_weight"] == .01,
            "frozen optimization settings changed")
    require(s.get("retain_undefined_reference_match") is True and
            "patient_partition" not in s and "panel_selection" not in s,
            "retain undefined matches; this grid uses the full frozen discovery panel")
    require(not p["semantics"]["families"] and not p["semantics"]["discovery_only_families"] and
            p["design"]["enabled"] is False, "extra jobs are outside this nine-fit sidecar")
    require(a["report_mode"] == "diagnostic_only" and a["prognostic_tests"] is False and
            a["readout_scopes"] == ["fibroblast_only", "all_cells"], "descriptive readout contract changed")
    require(r["device"] == "cuda" and r["gpu_workers"] == r["numeric_threads"] == 1 and r["max_jobs"] == 9,
            "exactly one CUDA worker and one numeric thread are required")
    require(type(r["max_incremental_artifact_bytes"]) is int and
            0 < r["max_incremental_artifact_bytes"] <= 512 * 2**20 and
            r["min_remaining_disk_bytes"] >= 8 * 2**30 and
            r["per_fit_reserve_bytes"] == 32 * 2**20 and r["summary_reserve_bytes"] == 8 * 2**20,
            "storage bounds changed or weakened")
    require(r["output_root"] == str(OUTPUT.relative_to(ROOT)) and
            all(r[name] is False for name in ("new_downloads", "raw_matrix_copies", "archives", "optimizer_resume")),
            "only the canonical output tree; no downloads, archives, copies or optimizer resume")
    require(p["stage_gate"]["prepare_stops_before_real_fits"] is True, "review hold must remain explicit")


def validate_grid(jobs):
    require(len(jobs) == 9 and len({j.job_id for j in jobs}) == 9 and
            all(j.job_type == "stability" and j.dataset_id == "GSE165816" and
                j.method == "topic_simplex" and j.representation == "panel" and
                type(j.seed) is int and type(j.params.get("K")) is int and
                type(j.params.get("steps")) is int for j in jobs) and
            {(j.params["K"], j.seed, j.params["steps"]) for j in jobs} == GRID,
            "sealed plan is not the exact nine-fit grid")


def output_bytes(folder):
    require(not folder.is_symlink() and not any(p.is_symlink() for p in folder.parents),
            "symlink output ancestry is not allowed")
    total = 0
    for path in folder.rglob("*"):
        require(not path.is_symlink(), "symlink inside canonical output tree")
        if path.is_file():
            total += path.stat().st_size
    return total


def resource_check(folder, policy, reserve=0):
    used, free = output_bytes(folder), shutil.disk_usage(folder).free
    require(used + reserve <= policy["max_incremental_artifact_bytes"], "incremental artifact budget exhausted")
    require(free - reserve >= policy["min_remaining_disk_bytes"], "remaining disk reserve below 8 GiB")
    return {"artifact_bytes": used, "free_disk_bytes": free, "reserved_next_write_bytes": reserve}


def source_manifest():
    from expansion.common.io import sha256_file
    from expansion.common.provenance import code_files
    paths = set(code_files(ROOT)) | {ROOT / name for name in OWNED}
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def cohort_inventory(cells):
    from expansion.common.io import content_hash
    from expansion.measurement.stability import _identifiers
    for name in ("gsm", "patient_id", "celltype", "group"):
        require(name in cells and cells[name].notna().all() and
                not cells[name].astype(str).str.strip().eq("").any(), f"missing {name} identity")
    require(not (cells.groupby("gsm").patient_id.nunique() > 1).any(), "specimen maps to multiple people")
    require(not (cells.groupby("patient_id").group.nunique() > 1).any(), "conflicting patient group labels")
    require(set(cells.group.astype(str)) <= {"DFU-H", "DFU-NH", "H"}, "unknown existing cohort group")
    ids = _identifiers(cells.reconstructed_raw_cell_id.to_numpy(), "cell_ids")
    people = []
    for patient, block in cells.groupby("patient_id", sort=True):
        fib = block[block.celltype == "fibroblast"]
        people.append({"patient_id": str(patient), "group": str(block.group.iloc[0]),
                       "is_DFU": bool(str(block.group.iloc[0]).startswith("DFU-")),
                       "all_cells": len(block), "all_specimens": block.gsm.nunique(),
                       "fibroblast_cells": len(fib), "fibroblast_specimens": fib.gsm.nunique()})
    return {"people": len(people), "DFU_patients": sum(x["is_DFU"] for x in people),
            "specimens": cells.gsm.nunique(), "cells": len(cells),
            "fibroblast_cells": int((cells.celltype == "fibroblast").sum()), "patients": people,
            "cell_order_identity": content_hash(ids, 64),
            "mapping_scope": "supplied_previously_audited_mapping_not_new_independent_raw_join"}


def prepare():
    from expansion.common.io import atomic_write_json, read_json
    from expansion.runner.cli import plan_run
    from expansion.runner.jobs import read_jobs
    import numpy as np
    import pandas as pd
    from expansion.measurement import stability
    from core_engines.topic_simplex_engine import TopicSimplexModel

    runtime = runtime_record()
    protocol = read_json(CONFIG)
    validate_protocol(protocol)
    require(not OUTPUT.exists(), "canonical P1 output tree already exists; preserve it, do not overwrite")
    cells = pd.read_parquet(ROOT / stability.CELLS)
    inventory = cohort_inventory(cells)
    for name in ("people", "DFU_patients", "specimens", "cells", "fibroblast_cells"):
        require(inventory[name] == protocol["expected_cohort"][name], f"frozen cohort {name} mismatch")
    panel = stability._identifiers(np.load(ROOT / stability.DISCOVERY / "panel.npy", allow_pickle=True), "panel")
    reference = np.load(ROOT / stability.DISCOVERY / "beta.npy", allow_pickle=False)
    require(reference.shape == (15, len(panel)), "reference topic/panel dimensions changed")
    # Worst-case uncompressed inference payload, plus 2 MiB of metadata per fit.
    estimates = []
    for k in protocol["stability"]["K"]:
        model = TopicSimplexModel(input_dim=len(panel), num_topics=k)
        state_size = sum(v.numel() * v.element_size() for v in model.state_dict().values())
        arrays = 4 * (len(cells) * k + k * len(panel) +
                      len(cells) * cells.reconstructed_raw_cell_id.str.len().max() + len(panel) * max(map(len, panel)))
        upper = int(state_size + arrays + 2 * 2**20)
        require(upper < protocol["resources"]["per_fit_reserve_bytes"], "checkpoint exceeds per-fit reservation")
        estimates.append({"K": k, "uncompressed_artifact_upper_bytes_per_fit": upper})
    OUTPUT.mkdir(parents=True)
    reserved = 9 * protocol["resources"]["per_fit_reserve_bytes"] + protocol["resources"]["summary_reserve_bytes"]
    budget = resource_check(OUTPUT, protocol["resources"], reserved)
    atomic_write_json(OUTPUT / "runtime_prepare.json", runtime)
    atomic_write_json(OUTPUT / "cohort_inventory.json", inventory)
    atomic_write_json(OUTPUT / "source_sha256.json", source_manifest())
    plan = plan_run(protocol, {}, OUTPUT / "run", root=ROOT)
    jobs = read_jobs(OUTPUT / "run/jobs.jsonl")
    validate_grid(jobs)
    draft = read_json(OUTPUT / "run/approval_request.json")
    draft["review_notes"] = ("Current explicit user authorization: " +
        protocol["stage_gate"]["user_authorization_verbatim"] +
        " Preparation only: leader protocol/test review has not yet authorized the real fits. "
        "This is an unapproved draft, not independent reviewer or author signoff.")
    atomic_write_json(OUTPUT / "run/execution_approval.json", draft)
    report = {"prepared_utc": stamp(), "status": "ready_for_review_no_real_fits_started", "plan": plan,
              "runtime": runtime, "resource_preflight": budget, "artifact_upper_bounds": estimates,
              "execution_approval": "unapproved_draft", "source_files": list(OWNED),
              "run_command": "python scripts/run_quality_topic_stability.py run --approval "
                             "output/publication-quality-20261009/p1/run/execution_approval.json",
              "timing_note": "4000-step pilot wall time is not a guaranteed duration for nine 40000-step fits"}
    atomic_write_json(OUTPUT / "readiness.json", report)
    return report


def validate_frozen():
    from expansion.common.io import read_json, sha256_file
    from expansion.common.provenance import code_digest, input_digests
    from expansion.runner.jobs import read_jobs
    protocol = read_json(CONFIG)
    validate_protocol(protocol)
    require(read_json(OUTPUT / "source_sha256.json") == source_manifest(), "scientific or sidecar source changed since freeze")
    require(read_json(OUTPUT / "run/protocol.json") == protocol, "frozen protocol changed")
    jobs = read_jobs(OUTPUT / "run/jobs.jsonl")
    validate_grid(jobs)
    provenance = read_json(OUTPUT / "run/provenance.json")
    require(provenance["code_digest"] == code_digest() and
            provenance["jobs_sha256"] == sha256_file(OUTPUT / "run/jobs.jsonl"), "code or sealed jobs changed")
    require(input_digests(ROOT, provenance["inputs"]) == provenance["inputs"], "frozen source inputs changed")
    require(read_json(OUTPUT / "run/datasets.json") == {}, "unexpected additional data")
    return protocol, jobs


def trace_metadata_adapter(trainer):
    """The core's retained-match writer requires keys absent in unpartitioned fits."""
    def adapted(*args, **kwargs):
        trained = trainer(*args, **kwargs)
        trained.setdefault("training_trace", [])
        trained.setdefault("evaluation_trace", [])
        return trained
    return adapted


@contextmanager
def bounded_executor(policy):
    from expansion.common.io import atomic_write_json
    from expansion.measurement import stability
    from expansion.runner import cli
    original_factory, original_trainer = cli.make_executor, stability.train_topic_model
    stability.train_topic_model = trace_metadata_adapter(original_trainer)
    def factory(*args, **kwargs):
        executor = original_factory(*args, **kwargs)
        def checked(job, directory):
            resource_check(OUTPUT, policy, policy["per_fit_reserve_bytes"] + policy["summary_reserve_bytes"])
            try:
                result = executor(job, directory)
                validate_checkpoint_result(result, job)
                resource_check(OUTPUT, policy, policy["summary_reserve_bytes"])
                return result
            except Exception as error:
                atomic_write_json(directory / "failed.json", {"job_id": job.job_id, "utc": stamp(),
                    "error": f"{type(error).__name__}: {error}", "policy": "abort_without_retry_or_threshold_rescue"})
                raise SystemExit(f"bounded grid stopped: {error}") from error
        return checked
    cli.make_executor = factory
    try:
        yield
    finally:
        cli.make_executor, stability.train_topic_model = original_factory, original_trainer


def validate_checkpoint_result(result, job):
    m = result.get("inference_checkpoint", {})
    require(m.get("reload_verified") is True and m.get("optimizer_resume_supported") is False,
            "checkpoint must be complete inference state with verified reload, not optimizer resume")
    error = m.get("reload_max_abs_error")
    require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1e-6,
            "invalid checkpoint reload discrepancy")
    require(m.get("training_config", {}).get("steps") == job.params["steps"] and
            m["training_config"].get("seed") == job.seed and
            m.get("model_config", {}).get("num_topics") == job.params["K"] and
            result.get("K") == job.params["K"] and result.get("seed") == job.seed,
            "checkpoint/result does not bind the frozen capacity, seed and 40000 steps")
    require(m["training_config"].get("device") == "cuda", "real fits require CUDA")
    losses = m.get("losses", [])
    require(len(losses) == 41 and all(type(v) in (int, float) and math.isfinite(v) for v in losses),
            "fixed-step fit must retain all 41 scheduled finite loss records")
    require(result.get("primary_match_rule") == "hungarian_one_to_one" and
            result.get("rowwise_best_sensitivity", {}).get("used_for_primary") is False,
            "rowwise rescue or changed matching rule is not permitted")


def run(approval):
    from expansion.common.io import atomic_write_json, read_json
    from expansion.runner.approval import require_execution_approval
    from expansion.runner.cli import execute_run
    from expansion.runner.queue import job_state
    import torch

    runtime = runtime_record()
    protocol, jobs = validate_frozen()
    require(Path(approval).resolve() == OUTPUT / "run/execution_approval.json", "approval must be the canonical plan-bound receipt")
    require_execution_approval(OUTPUT / "run", approval, protocol=protocol, files={},
                               provenance=read_json(OUTPUT / "run/provenance.json"))
    require(torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(all(job_state(OUTPUT / "run", job) in ("pending", "done") for job in jobs),
            "failed/running fits require inspection; do not retry or overwrite")
    lock = OUTPUT / "worker.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    started = time.perf_counter()
    try:
        atomic_write_json(OUTPUT / "runtime_run.json", runtime)
        with bounded_executor(protocol["resources"]):
            counts = execute_run(OUTPUT / "run", root=ROOT, device="cuda", max_jobs=9,
                                 approval_path=approval, retry_failed=False, save_predictions=False)
        require(counts["failed"] == counts["skipped_failed"] == 0, "failed fits remain; no aggregate")
        atomic_write_json(OUTPUT / "execution.json", {"finished_utc": stamp(), "counts": counts,
                         "wall_seconds": time.perf_counter() - started,
                         "resource_final": resource_check(OUTPUT, protocol["resources"]), "runtime": runtime})
        return verify()
    finally:
        lock.unlink(missing_ok=True)


def _get(mapping, key):
    return mapping.get(str(key), mapping.get(key))


def match_rows(job, result, n_reference=15):
    match = result["match_to_reference"]
    base = {"job_id": job.job_id, "K": job.params["K"], "seed": job.seed}
    rows = []
    for topic in range(n_reference):
        assigned = _get(match["assignment"], topic)
        rows.append({**base, "row_kind": "reference_topic", "reference_topic": topic, "new_topic": assigned,
            "status": "assigned" if assigned is not None else match.get("status", "unmatched"),
            "reason": match.get("reason", ""), "assigned_correlation": _get(match["assigned_correlation"], topic),
            "rowwise_best_topic": _get(match.get("best_match", {}), topic),
            "rowwise_best_correlation": _get(match.get("best_correlation", {}), topic),
            "best_minus_second": _get(match.get("best_minus_second", {}), topic),
            "rowwise_used_for_primary": False})
    for topic in match.get("unmatched_new", []):
        rows.append({**base, "row_kind": "unmatched_new_topic", "reference_topic": None, "new_topic": topic,
                     "status": match.get("status", "unmatched"), "reason": match.get("reason", ""),
                     "assigned_correlation": None, "rowwise_best_topic": None, "rowwise_best_correlation": None,
                     "best_minus_second": None, "rowwise_used_for_primary": False})
    return rows


def patient_rows(job, result, cells, theta):
    import numpy as np
    import pandas as pd
    primary = result.get("fibroblast_topic")
    cut = result.get("cut")
    available = primary is not None and cut is not None
    if available:
        require(0 <= primary < theta.shape[1] and math.isfinite(cut), "invalid primary readout topic/cut")
    inventory = cohort_inventory(cells)
    rows = []
    for scope in ("fibroblast_only", "all_cells"):
        mask = (cells.celltype == "fibroblast").to_numpy() if scope == "fibroblast_only" else np.ones(len(cells), bool)
        frame = cells.loc[mask, ["patient_id", "gsm"]].copy()
        if available:
            frame["mean"] = theta[mask, primary].astype(float)
            frame["fraction"] = (theta[mask, primary] > cut).astype(float)
        for person in inventory["patients"]:
            block = frame[frame.patient_id == person["patient_id"]]
            status = ("available" if available and len(block) else
                      "no_cells_in_scope" if not len(block) else "primary_readout_unavailable_no_rescue")
            values = block.groupby("gsm")[["mean", "fraction"]].mean().mean() if status == "available" else None
            rows.append({"job_id": job.job_id, "K": job.params["K"], "seed": job.seed,
                "patient_id": person["patient_id"], "group": person["group"], "is_DFU": person["is_DFU"],
                "readout_scope": scope, "n_cells": len(block), "n_specimens": block.gsm.nunique(),
                "status": status, "mean_loading": None if values is None else float(values["mean"]),
                "fraction_above_fibroblast_descriptive_cut": None if values is None else float(values["fraction"]),
                "cut_full_sample_fibroblasts": cut,
                "threshold_scope": "full_discovery_fibroblasts_descriptive_not_held_unit_density_cut",
                "patient_aggregation": "equal_specimen_mean_not_cell_pooled"})
    # Check the new tabular extraction against the existing worker's own pooling.
    if available:
        existing = {str(r["unit"]): r for r in result["patient_readouts"]}
        observed = {r["patient_id"]: r for r in rows if r["readout_scope"] == "fibroblast_only" and r["status"] == "available"}
        require(set(existing) == set(observed), "worker/table patient coverage disagreement")
        for patient, row in observed.items():
            require(np.allclose([row["mean_loading"], row["fraction_above_fibroblast_descriptive_cut"]],
                                [existing[patient]["mean"], existing[patient]["fraction"]], rtol=0, atol=1e-12) and
                    (row["n_cells"], row["n_specimens"]) == (existing[patient]["n_cells"], existing[patient]["n_specimens"]),
                    "worker/table equal-specimen readout disagreement")
    return rows


def seed_ranges(rows):
    groups = {}
    metrics = ("mean_loading", "fraction_above_fibroblast_descriptive_cut")
    for row in rows:
        key = (row["K"], row["patient_id"], row["readout_scope"])
        groups.setdefault(key, []).append(row)
    output = []
    for (k, patient, scope), block in sorted(groups.items()):
        require(len(block) == 3 and {r["seed"] for r in block} == {0, 1, 2}, "incomplete or duplicated computational seed block")
        for metric in metrics:
            values = [r[metric] for r in block if r[metric] is not None]
            require(all(math.isfinite(v) for v in values), "nonfinite readout cannot enter seed ranges")
            output.append({"K": k, "patient_id": patient, "readout_scope": scope, "metric": metric,
                "planned_seeds": 3, "available_seeds": len(values), "missing_seeds": 3 - len(values),
                "min_over_available_seeds": min(values) if values else None,
                "max_over_available_seeds": max(values) if values else None,
                "range_over_available_seeds": max(values) - min(values) if values else None,
                "complete_three_seed_readout": len(values) == 3,
                "interpretation": "computational_seed_range_not_patient_confidence_interval"})
    return output


def tables_for_fit(job, result, cells, theta):
    from expansion.common.io import canonical_json
    base = {"job_id": job.job_id, "K": job.params["K"], "seed": job.seed}
    sem = result["semantics"]
    sem_rows, patient_sem_rows, specimen_rows = [], [], []
    inventory = cohort_inventory(cells)
    for comparison in COMPARISONS:
        values = sem.get(comparison, {})
        sem_rows.append({**base, "comparison": comparison, "available": bool(sem.get("available") and values),
            "reason": sem.get("reason", ""), "specimens": values.get("specimens", 0), "units": values.get("units", 0),
            "specimens_favour_first": values.get("specimens_favour_first"), "units_favour_first": values.get("units_favour_first"),
            "median_difference_nats_per_cell": values.get("median_difference_nats_per_cell"),
            "threshold_scope": "H1_cut_fitted_excluding_scored_patient_not_full_sample_readout_cut",
            "validation_scope": "conditional_on_full_discovery_scores_not_end_to_end_cross_fitted",
            "exclusions_json": canonical_json(sem.get("exclusions", []))})
        differences = values.get("unit_mean_differences", {})
        for person in inventory["patients"]:
            value = differences.get(person["patient_id"])
            require(value is None or math.isfinite(value), "nonfinite patient density comparison")
            patient_sem_rows.append({**base, "patient_id": person["patient_id"], "group": person["group"],
                "comparison": comparison, "difference_nats_per_cell": value,
                "status": "available" if value is not None else "unavailable",
                "reason": "" if value is not None else sem.get("reason", "no evaluable specimen comparison"),
                "aggregation": "equal_specimen_mean_within_patient", "prognostic_test": False})
    existing_rows = result.get("_semantics_rows", [])
    by_specimen = {str(r["specimen"]): r for r in existing_rows}
    require(len(by_specimen) == len(existing_rows), "duplicated specimen semantics")
    for specimen, block in cells[cells.celltype == "fibroblast"].groupby("gsm", sort=True):
        original = by_specimen.get(str(specimen), {})
        specimen_rows.append({**base, "specimen": str(specimen), "patient_id": str(block.patient_id.iloc[0]),
            "status": "available" if original else "unavailable",
            "exclusions_json": canonical_json([x for x in sem.get("exclusions", []) if x.get("specimen") == specimen]),
            "n_cells_scored": original.get("n_cells"), "n_splits": original.get("n_splits"),
            "cut_training_only_logit_scale": original.get("cut"), "hard_available": original.get("hard_available"),
            "threshold_scope": "other_patients_only_in_transformed_score_space",
            **{name: original.get(name) for name in ("C1", "C2", "M1", "M2", "H1")}})
    fit = {**base, "optimizer_steps": result["inference_checkpoint"]["training_config"]["steps"],
        "reload_verified": result["inference_checkpoint"]["reload_verified"],
        "reload_max_abs_error": result["inference_checkpoint"]["reload_max_abs_error"],
        "primary_match_status": result["primary_match_status"], "primary_new_topic": result.get("fibroblast_topic"),
        "primary_correlation": result.get("fibroblast_topic_correlation"), "semantics_available": sem.get("available", False),
        "cut_full_sample_fibroblasts": result.get("cut"),
        "specimen_mean_fraction_spearman": result.get("mean_fraction_spearman"),
        "mean_fraction_correlation_unit": "specimen_descriptive_not_prognostic",
        "unmatched_reference_topics": len(result["match_to_reference"]["unmatched_reference"]),
        "unmatched_new_topics": len(result["match_to_reference"]["unmatched_new"])}
    return {"fits": [fit], "topic_matches": match_rows(job, result),
            "patient_readouts": patient_rows(job, result, cells, theta), "semantics": sem_rows,
            "patient_semantics": patient_sem_rows, "semantics_specimens": specimen_rows}


def write_csv(path, rows):
    from expansion.common.io import atomic_write_text
    require(rows, "empty output table is not a valid summary")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def verify():
    from expansion.analysis.aggregate import audit_run
    from expansion.common.io import atomic_write_json, read_json, sha256_file, sanitize
    from expansion.measurement import stability
    from expansion.runner.queue import job_dir
    import numpy as np

    protocol, jobs = validate_frozen()
    integrity = audit_run(OUTPUT / "run")
    resource_check(OUTPUT, protocol["resources"], protocol["resources"]["summary_reserve_bytes"])
    proportions, cells = stability.load_discovery_panel_counts(ROOT)
    require(cohort_inventory(cells) == read_json(OUTPUT / "cohort_inventory.json"), "frozen patient/cell inventory changed")
    panel = np.load(ROOT / stability.DISCOVERY / "panel.npy", allow_pickle=True)
    reference = np.load(ROOT / stability.DISCOVERY / "beta.npy", allow_pickle=False)
    tables, replay = {}, []
    for job in sorted(jobs, key=lambda j: (j.params["K"], j.seed)):
        folder = job_dir(OUTPUT / "run", job.job_id)
        result = read_json(folder / "result.json")
        validate_checkpoint_result(result, job)
        loaded = stability.load_inference_checkpoint(folder, panel=panel, cell_ids=cells.reconstructed_raw_cell_id.to_numpy())
        require(loaded["metadata"] == result["inference_checkpoint"], "result/checkpoint metadata disagreement")
        require(loaded["metadata"]["proportions_sha256"] == stability._matrix_hash(proportions), "checkpoint input matrix mismatch")
        theta = stability.project_theta(loaded["state"], proportions, job.params["K"], device="cpu")
        error = float(np.max(np.abs(theta - loaded["theta"])))
        require(error <= 1e-6 and math.isfinite(error), "fresh full CPU projection replay failed")
        try:
            matched = stability.match_topics(reference, loaded["beta"])
        except ValueError as error_match:
            require("constant topic rows" in str(error_match) and
                    result["match_to_reference"].get("status") == "undefined_no_rescue" and
                    result["match_to_reference"].get("reason") == str(error_match) and
                    result.get("fibroblast_topic") is None, "undefined match replay disagreement")
        else:
            require(sanitize(matched) == result["match_to_reference"], "saved Hungarian match differs from replay")
            require(result.get("fibroblast_topic") == matched["assignment"].get(0), "primary was rescued or changed")
        readout_path = folder / "readouts.json"
        result["_semantics_rows"] = read_json(readout_path).get("semantics_rows", []) if readout_path.exists() else []
        for name, rows in tables_for_fit(job, result, cells, loaded["theta"]).items():
            tables.setdefault(name, []).extend(rows)
        replay.append({"job_id": job.job_id, "K": job.params["K"], "seed": job.seed,
                       "fresh_cpu_projection_max_abs_error": error, "complete_reload_and_match_replay": True})
    tables["seed_ranges"] = seed_ranges(tables["patient_readouts"])
    destination = OUTPUT / "tables"
    destination.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = destination / f"{name}.csv"
        write_csv(path, rows)
        hashes[name] = {"rows": len(rows), "sha256": sha256_file(path)}
    report = {"verified_utc": stamp(), "integrity": integrity, "full_checkpoint_replays": replay,
        "tables": hashes, "cohort": {k: read_json(OUTPUT / "cohort_inventory.json")[k]
            for k in ("people", "DFU_patients", "specimens", "cells", "fibroblast_cells")},
        "not_claims": protocol["not_claims"], "resource_final": resource_check(OUTPUT, protocol["resources"])}
    atomic_write_json(OUTPUT / "verification.json", report)
    return report


def main():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    require(args.command == "run" or args.approval is None, "approval is accepted only by run")
    require(args.command != "run" or args.approval is not None, "reviewed plan-bound --approval is required")
    answer = prepare() if args.command == "prepare" else run(args.approval) if args.command == "run" else verify()
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
