"""Command-line interface: ``python -m expansion [--root DIR] <command> ...``.

Nothing here downloads data unless ``fetch --execute`` (or ``stage-a
--execute-fetch``) receives an approval file, and nothing trains on real data
unless ``run --approval FILE`` receives a reviewed, plan-bound receipt. Only
regenerated synthetic diagnostic inputs are exempt from execution approval.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from ..common.io import atomic_write_json, atomic_write_text, read_json, sanitize, sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOLS = {"population": PACKAGE_ROOT / "config" / "population_benchmark.json",
                     "measurement": PACKAGE_ROOT / "config" / "measurement_scaleup.json"}
FLOWS = ("cfm_independent", "cfm_ot", "cfm_sb", "cfm_conditioned")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _entries(ids=None):
    from ..data.registry import load_registry

    entries = load_registry()
    if ids:
        unknown = set(ids) - {entry.id for entry in entries}
        if unknown:
            raise SystemExit(f"unknown dataset ids: {sorted(unknown)}")
        entries = [entry for entry in entries if entry.id in set(ids)]
    return entries


# ---------------------------------------------------------------- registry, fetch, probe, stage A

def cmd_registry(args) -> int:
    from ..data.registry import validate_registry

    for entry in _entries(args.ids):
        sizes = ", ".join(f.get("size_label", "?") for f in entry.source.get("files", [])) or "local"
        print(f"{entry.id:26s} tier={entry.tier} {entry.status:8s} {entry.adapter:30s} {entry.adapter_status:15s} {sizes}")
    problems = validate_registry(_entries())
    for problem in problems:
        print("PROBLEM:", problem)
    return 1 if problems else 0


def cmd_fetch(args) -> int:
    from ..data.fetch import execute_fetch, plan_fetch

    items = plan_fetch(_entries(), ids=args.ids or None)
    if not args.execute:
        for item in items:
            print(f"{item.dataset_id:26s} {item.size_label:>7s}  {item.url}")
        print(f"{len(items)} files planned; nothing downloaded. Use --execute --approval FILE after approval.")
        return 0
    print(json.dumps(execute_fetch(items, root=args.root, approval_path=args.approval), indent=1))
    return 0


def cmd_probe(args) -> int:
    from ..data.adapters import probe_dataset

    for entry in _entries(args.ids):
        report = probe_dataset(Path(args.root) / entry.local_path)
        atomic_write_json(Path(args.standard_dir) / entry.id / "probe.json", report)
        print(f"{entry.id}: {len(report['files'])} files probed")
    return 0


def _stage_a_dataset(entry, root: Path, out: Path, *, min_cells: int, force: bool) -> dict:
    from ..data.adapters import ingest_dataset, probe_dataset
    from ..data.standard import load_standard, validate_measurement, validate_standard, write_standard

    base = root / entry.local_path
    record = {"dataset_id": entry.id, "adapter": entry.adapter, "adapter_status": entry.adapter_status,
              "stage_a_utc": utc_now()}
    atomic_write_json(out / "probe.json", probe_dataset(base))
    if not base.exists():
        return {**record, "status": "missing_data"}
    if entry.adapter_status == "probe_only":
        return {**record, "status": "probe_only"}
    target = out / "standard.h5ad"
    try:
        if target.exists() and not force:
            adata = load_standard(target)
        else:
            adata = ingest_dataset(entry, root)
            write_standard(adata, target)
        contract = validate_standard(adata, min_cells=min_cells)
        measurement = validate_measurement(adata)
        payload = {**record, "status": "ingested", "dataset_file": str(target), "dataset_sha256": sha256_file(target),
                   "contract": contract, "measurement": measurement}
        atomic_write_json(out / "contract.json", sanitize(payload))
        (out / "ingest_error.json").unlink(missing_ok=True)
        return {"dataset_id": entry.id, "status": "ingested", "population_status": contract["status"],
                "measurement_status": measurement["status"], "eligible": contract.get("eligible"),
                "n_cells": int(adata.n_obs)}
    except Exception as error:  # recorded for review; Stage A continues with the next dataset
        atomic_write_json(out / "ingest_error.json", {**record, "error": f"{type(error).__name__}: {error}",
                                                      "traceback": traceback.format_exc()})
        return {**record, "status": "ingest_failed", "error": f"{type(error).__name__}: {error}"}


def cmd_stage_a(args) -> int:
    root, standard_dir = Path(args.root), Path(args.standard_dir)
    entries = _entries(args.ids)
    if args.execute_fetch:
        from ..data.fetch import execute_fetch, plan_fetch

        items = plan_fetch(entries)
        if items:
            print(json.dumps(execute_fetch(items, root=root, approval_path=args.approval), indent=1))
    summary = [_stage_a_dataset(e, root, standard_dir / e.id, min_cells=args.min_cells, force=args.force) for e in entries]
    atomic_write_json(standard_dir / "stage_a_summary.json", {"completed_utc": utc_now(), "datasets": summary})
    for item in summary:
        print(json.dumps(sanitize(item)))
    print("Stage A stops here by design: review probe.json and contract.json before planning Stage B.")
    # Probe-only entries are an intentional diagnostic outcome, not an ingestion
    # success. Missing/failed inputs or a dataset rejected by both contracts fail.
    failed = any(item["status"] in {"missing_data", "ingest_failed"} or
                 (item["status"] == "ingested" and item.get("population_status") != "accepted"
                  and item.get("measurement_status") != "accepted") for item in summary)
    return 1 if failed else 0


# ---------------------------------------------------------------- planning and execution

def _load_contracts(standard_dir: Path, ids=None) -> dict:
    contracts = {}
    for path in sorted(Path(standard_dir).glob("*/contract.json")):
        payload = read_json(path)
        # Resolve the standardized file next to its contract, so contracts stay valid after copying.
        payload["dataset_file"] = str(path.parent / "standard.h5ad")
        probe = path.parent / "probe.json"
        if probe.exists():
            payload["probe"] = read_json(probe)
        if not ids or payload["dataset_id"] in ids:
            contracts[payload["dataset_id"]] = payload
    return contracts


def plan_run(protocol: dict, contracts: dict, run_dir, *, root, ids=None) -> dict:
    from .approval import approval_request, discovery_input_paths, generated_synthetic_only
    from ..common.provenance import code_digest, input_digests
    from .jobs import plan_measurement_jobs, plan_population_jobs, protocol_hash, write_jobs

    run_dir = Path(run_dir)
    if (run_dir / "jobs.jsonl").exists():
        raise SystemExit(f"{run_dir} already holds a plan; choose a new run directory")
    code = code_digest()
    inputs = {}
    if protocol["kind"] == "population":
        wanted = set(ids or []) or set(protocol["datasets"]["tier_W"]) | set(protocol["datasets"]["tier_G"])
        datasets = {k: {"contract": v["contract"], "dataset_sha256": v["dataset_sha256"], "dataset_file": v["dataset_file"]}
                    for k, v in contracts.items() if k in wanted}
        planned = plan_population_jobs(protocol, datasets, code_digest=code)
        files = {k: {"dataset_file": v["dataset_file"], "dataset_sha256": v["dataset_sha256"]} for k, v in datasets.items()}
    else:
        wanted = set(ids or []) or set(protocol["cohorts"]["external"])
        cohorts = {protocol["cohorts"]["discovery"]: {"source": "discovery"}}
        for k, v in contracts.items():
            if k in wanted and v.get("measurement", {}).get("status") == "accepted":
                cohorts[k] = {"source": "standard", "dataset_file": v["dataset_file"], "dataset_sha256": v["dataset_sha256"]}
        planned = plan_measurement_jobs(protocol, cohorts, code_digest=code)
        files = {k: {"dataset_file": v["dataset_file"], "dataset_sha256": v["dataset_sha256"]}
                 for k, v in cohorts.items() if v["source"] == "standard"}
        inputs = input_digests(root, discovery_input_paths())
    if not planned:
        raise SystemExit("no eligible jobs; inspect Stage-A contracts before planning")
    active_ids = {job.dataset_id for job in planned}
    if ids and set(ids) - active_ids:
        raise SystemExit(f"requested datasets have no eligible jobs: {sorted(set(ids) - active_ids)}")
    files = {k: v for k, v in files.items() if k in active_ids}
    synthetic = generated_synthetic_only(protocol, files, inputs, planned)
    material = {}
    for dataset_id in sorted(active_ids):
        if dataset_id in files:
            record = contracts[dataset_id]
            contract = record["contract"] if protocol["kind"] == "population" else record["measurement"]
            material[dataset_id] = {"basis": "stage_a", "status": contract["status"],
                                    "contract": contract, "probe": record.get("probe"),
                                    "dataset_sha256": files[dataset_id]["dataset_sha256"]}
        else:
            material[dataset_id] = {"basis": "frozen_discovery_inputs", "input_sha256": inputs}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jobs(run_dir / "jobs.jsonl", planned)
    atomic_write_json(run_dir / "protocol.json", protocol)
    atomic_write_json(run_dir / "datasets.json", files)
    atomic_write_json(run_dir / "review_material.json", material)
    atomic_write_json(run_dir / "provenance.json", {"code_digest": code, "inputs": inputs, "planned_utc": utc_now(),
                                                    "jobs_sha256": sha256_file(run_dir / "jobs.jsonl"),
                                                    "execution_scope": "verified_synthetic" if synthetic else "approval_required",
                                                    "review_material_sha256": sha256_file(run_dir / "review_material.json")})
    atomic_write_json(run_dir / "approval_request.json", approval_request(run_dir))
    counts = {}
    for job in planned:
        key = "|".join([job.job_type, job.dataset_id, str(job.task.get("kind", "-")), str(job.task.get("family", "-")), job.method])
        counts[key] = counts.get(key, 0) + 1
    summary = {"planned_utc": utc_now(), "protocol_hash": protocol_hash(protocol), "code_digest": code, "jobs": len(planned),
               "datasets": sorted(files) if files else [], "counts": counts}
    atomic_write_json(run_dir / "plan_summary.json", summary)
    return summary


def cmd_plan(args) -> int:
    protocol = read_json(args.protocol)
    summary = plan_run(protocol, _load_contracts(Path(args.standard_dir), args.ids), Path(args.run_dir),
                       root=Path(args.root), ids=args.ids)
    print(json.dumps({k: summary[k] for k in ("jobs", "protocol_hash", "datasets")}, indent=1))
    return 0


def _device(args) -> str:
    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but not available")
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


def make_executor(run_dir: Path, protocol: dict, *, root: Path, device: str, save_predictions: bool):
    from ..measurement.worker import run_design_job, run_semantics_job, run_stability_job
    from ..population.worker import reset_process_caches, run_population_job

    files = read_json(run_dir / "datasets.json")
    reset_process_caches()

    def execute(job, out_dir):
        data = job.to_dict()
        if job.job_type == "population_fit":
            return run_population_job(data, dataset_file=files[job.dataset_id]["dataset_file"], cache_dir=run_dir / "cache",
                                      protocol=protocol, device=device, out_dir=out_dir, save_predictions=save_predictions)
        if job.job_type == "semantics":
            if data["params"]["source"] == "standard":
                # The verified manifest is the sole runtime path authority; the
                # serialized job path is a historical planning location only.
                data["params"]["dataset_file"] = files[job.dataset_id]["dataset_file"]
            return run_semantics_job(data, protocol=protocol, root=root)
        if job.job_type == "stability":
            return run_stability_job(data, protocol=protocol, root=root, device=device, out_dir=out_dir)
        if job.job_type == "design":
            return run_design_job(data, protocol=protocol, root=root)
        raise ValueError(f"unknown job type {job.job_type}")

    return execute


def execute_run(run_dir, *, root, device: str, shard=(0, 1), max_jobs=None, retry_failed: bool = False,
                save_predictions: bool = True, allow_code_change: bool = False, approval_path=None) -> dict:
    from .approval import require_execution_approval
    from ..common.provenance import code_digest, input_digests
    from .jobs import protocol_hash, read_jobs
    from .queue import run_queue

    run_dir = Path(run_dir)
    if shard[1] < 1 or not 0 <= shard[0] < shard[1]:
        raise ValueError("shard must satisfy 0 <= index < total")
    if max_jobs is not None and max_jobs < 1:
        raise ValueError("max_jobs must be positive")
    provenance = read_json(run_dir / "provenance.json")
    if provenance.get("jobs_sha256") != sha256_file(run_dir / "jobs.jsonl"):
        raise SystemExit("plan changed or is not sealed; re-plan into a new run directory")
    current = code_digest()
    if current != provenance["code_digest"]:
        raise SystemExit(f"code changed since planning ({provenance['code_digest']} -> {current}); "
                         "re-plan into a new run directory; --allow-code-change cannot bypass scientific provenance")
    if provenance.get("inputs") and input_digests(root, provenance["inputs"]) != provenance["inputs"]:
        raise SystemExit("measurement inputs changed since planning; re-plan into a new run directory")
    protocol = read_json(run_dir / "protocol.json")
    planned = read_jobs(run_dir / "jobs.jsonl")
    digest = protocol_hash(protocol)
    stale = [job.job_id for job in planned if job.protocol_hash != digest or job.params.get("code_digest") != current]
    if stale:
        raise SystemExit(f"{len(stale)} jobs were planned under different code/protocol; re-plan into a new run directory")
    files = read_json(run_dir / "datasets.json")
    for dataset_id, info in files.items():
        if sha256_file(info["dataset_file"]) != info["dataset_sha256"]:
            raise SystemExit(f"{dataset_id}: standardized dataset changed after planning")
    authorization = require_execution_approval(run_dir, approval_path, protocol=protocol, files=files, provenance=provenance)
    invocation_id = uuid.uuid4().hex
    event = {"invocation_id": invocation_id, "started_utc": utc_now(), "device": device,
             "code_digest": current, "jobs_sha256": sha256_file(run_dir / "jobs.jsonl"),
             "protocol_hash": digest, "shard": list(shard), "authorization": authorization}
    event_path = run_dir / "logs" / f"invocation_{invocation_id}.json"
    atomic_write_json(event_path, event)
    invocation = {"id": invocation_id, "event_sha256": sha256_file(event_path)}
    try:
        executor = make_executor(run_dir, protocol, root=Path(root), device=device, save_predictions=save_predictions)
        counts = run_queue(planned, run_dir, executor, shard=shard, max_jobs=max_jobs,
                           retry_failed=retry_failed, invocation=invocation)
    except BaseException as error:
        atomic_write_json(run_dir / "logs" / f"run_{invocation_id}.json",
                          {**event, "status": "interrupted", "error": f"{type(error).__name__}: {error}"})
        raise
    atomic_write_json(run_dir / "logs" / f"run_{invocation_id}.json",
                      {**event, "counts": counts, "code_changed_since_planning": False,
                       "status": "failed" if counts["failed"] or counts["skipped_failed"] else "finished"})
    return counts


def cmd_run(args) -> int:
    index, total = (int(value) for value in args.shard.split("/"))
    counts = execute_run(Path(args.run_dir), root=Path(args.root), device=_device(args), shard=(index, total),
                         max_jobs=args.max_jobs, retry_failed=args.retry_failed, save_predictions=not args.no_predictions,
                         allow_code_change=args.allow_code_change, approval_path=args.approval)
    print(json.dumps(counts, indent=1))
    return 1 if counts["failed"] or counts["skipped_failed"] else 0


def cmd_status(args) -> int:
    from .jobs import read_jobs
    from .queue import status

    print(json.dumps(status(Path(args.run_dir), read_jobs(Path(args.run_dir) / "jobs.jsonl")), indent=1))
    return 0


def verify_population_sample(run_dir, sample: int, *, seed: int = 0):
    from ..common.rng import derive_seed
    from ..data.standard import load_standard
    from ..population.metrics import evaluate
    from ..population.tasks import TaskSpec, build_task_data, make_clock, representation_mask
    from ..population.worker import cached_coordinates, reset_process_caches
    from .jobs import read_jobs
    from .queue import job_dir, verify_done

    run_dir = Path(run_dir)
    if sample < 1:
        raise ValueError("verification sample must be positive")
    reset_process_caches()
    protocol = read_json(run_dir / "protocol.json")
    files = read_json(run_dir / "datasets.json")
    finished = [job for job in read_jobs(run_dir / "jobs.jsonl") if job.job_type == "population_fit"
                and verify_done(job_dir(run_dir, job.job_id), job)
                and (job_dir(run_dir, job.job_id) / "predictions.npz").exists()]
    rng = np.random.default_rng(seed)
    chosen = [finished[i] for i in sorted(rng.choice(len(finished), size=min(sample, len(finished)), replace=False))] if finished else []
    problems = []
    for job in chosen:
        directory = job_dir(run_dir, job.job_id)
        result = read_json(directory / "result.json")
        adata = load_standard(files[job.dataset_id]["dataset_file"])
        task = TaskSpec.from_dict(job.task)
        coords, _ = cached_coordinates(adata, representation_mask(adata.obs, task), protocol["representations"][job.representation],
                                       run_dir / "cache", dataset_sha256=job.params["dataset_sha256"],
                                       seed=int(protocol.get("representation_seed", 0)),
                                       code_digest=job.params.get("code_digest", ""))
        clock = make_clock(adata.obs["time_value"].unique(), job.params.get("clock", protocol["clock"]["primary"]))
        data = build_task_data(coords, adata.obs, task, clock, min_cells=int(protocol.get("min_cells_per_group", 1)))
        with np.load(directory / "predictions.npz") as saved:
            for record in (r for r in result["records"] if r["source_variant"] == "own"):
                pair = task.pairs[record["pair_index"]]
                eval_seed = derive_seed(task.dataset_id, task.kind, task.family, pair.individual, pair.source_time, pair.target_time)
                again = evaluate(saved[f"pair{record['pair_index']}_own"], data.targets[(pair.individual, pair.target_time)],
                                 seed=eval_seed, caps=protocol["evaluation"]["caps"],
                                 n_projections=int(protocol["evaluation"].get("sliced_projections", 128)))
                stored = record["energy"]
                if stored is None or again["energy"] is None or abs(again["energy"] - stored) > 1e-9 * max(1.0, abs(stored)):
                    problems.append(f"{job.job_id} pair {record['pair_index']}: stored {stored} recomputed {again['energy']}")
    return problems, len(chosen)


def cmd_verify(args) -> int:
    if args.sample < 1:
        print(json.dumps({"status": "not_verified", "checked_jobs": 0, "problems": ["sample must be positive"]}))
        return 1
    problems, checked = verify_population_sample(Path(args.run_dir), args.sample)
    status = "failed" if problems else ("passed" if checked else "not_verified")
    print(json.dumps({"status": status, "checked_jobs": checked, "problems": problems}, indent=1))
    return 0 if status == "passed" else 1


# ---------------------------------------------------------------- aggregation and power

def _geometric_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if np.any(values == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(values)))) if values.size else float("nan")


def _analysis_policy(protocol: dict) -> dict:
    """Restrictions take precedence over requests for formal inference, including legacy pilots."""
    analysis = protocol.get("analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be an object")
    mode = analysis.get("report_mode", "formal")
    if mode not in {"formal", "diagnostic_only"}:
        raise ValueError("analysis.report_mode must be formal or diagnostic_only")
    reasons = []
    for name in ("smoke_only", "pilot_only"):
        if name in protocol and not isinstance(protocol[name], bool):
            raise ValueError(f"{name} must be boolean")
        if protocol.get(name):
            reasons.append(name)
    if mode == "diagnostic_only":
        reasons.append("explicit diagnostic report mode")
    unit = str(analysis.get("inferential_unit", "")).lower()
    if unit == "none" or unit.startswith("none_"):
        reasons.append("no inferential unit declared")
    legacy = " ".join(str(protocol.get(key, "")) for key in ("protocol_id", "status")).lower()
    if any(marker in legacy for marker in ("smoke", "pilot", "diagnostic")):
        reasons.append("diagnostic protocol identity/status")
    diagnostic = bool(reasons)
    return {"mode": "diagnostic_only" if diagnostic else "formal", "inference_enabled": not diagnostic,
            "reasons": reasons, "interpretation": (
                "Diagnostics do not establish method selection, biological replication or publication claims."
                if diagnostic else "Formal analysis path; eligibility and statistical assumptions still apply.")}


def _population_diagnostics(df, protocol: dict) -> dict:
    """No tests, intervals, rankings or pooled inference; keep each target and source variant distinct."""
    from ..analysis.inference import unit_keys
    from .queue import IntegrityError

    metrics = list(dict.fromkeys([protocol["evaluation"]["primary_metric"]]
                                 + protocol["evaluation"].get("secondary_metrics", [])))
    keys = unit_keys(df) + ["method", "source_variant"]
    keys += [name for name in ("task_hash", "training_individuals", "budget_hash", "evaluation_hash") if name in df.columns]
    if df.duplicated(keys + ["seed"]).any():
        raise IntegrityError("duplicate diagnostic target/seed rows")
    if "fit_seconds" in df and (df.groupby("job_id")["fit_seconds"].nunique(dropna=False) > 1).any():
        raise IntegrityError("inconsistent fit time within a diagnostic job")
    target_summaries = []
    for identity, block in df.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, identity))
        row.update(n_seeds=int(block["seed"].nunique()), evaluation_rows=int(len(block)), metrics={})
        for metric in metrics:
            if metric not in block:
                row["metrics"][metric] = {"available": 0, "unavailable": len(block)}
                continue
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row["metrics"][metric] = {
                "available": int(finite.size), "unavailable": int(len(values) - finite.size),
                "mean_over_seeds": float(finite.mean()) if finite.size else None,
                "min_over_seeds": float(finite.min()) if finite.size else None,
                "max_over_seeds": float(finite.max()) if finite.size else None}
        target_summaries.append(row)
    fit = df.drop_duplicates("job_id")
    timing_keys = ["dataset_id", "representation", "clock", "kind", "family", "method"]
    timings = []
    for identity, block in fit.groupby(timing_keys, dropna=False, sort=True):
        values = block["fit_seconds"].to_numpy(dtype=float)
        valid = values[np.isfinite(values) & (values >= 0)]
        timings.append({**dict(zip(timing_keys, identity)), "jobs": int(len(block)),
                        "fit_timing_available": int(valid.size), "fit_timing_unavailable": int(len(block)-len(valid)),
                        "fit_seconds_sum": float(valid.sum()) if valid.size else None,
                        "fit_seconds_mean_per_job": float(valid.mean()) if valid.size else None})
    return {"jobs": int(len(fit)), "evaluation_rows": int(len(df)),
            "target_seed_summaries": target_summaries, "timings": timings,
            "notes": ["Ranges describe computational seeds, not patient confidence intervals.",
                      "Individual identifiers retain source semantics; an arm is not a biological individual.",
                      "Timing counts each job once and excludes input/representation, prediction, scoring and IO.",
                      "Unavailable metrics remain explicit; diagnostics do not qualify a method for publication."]}


def aggregate_run(run_dir, out) -> dict:
    from ..analysis.aggregate import audit_run, collect_population, iter_completed
    from ..analysis.inference import (friedman_nemenyi, log_ratio_table, metric_agreement, primary_analysis,
                                      source_information, unit_errors, clock_sensitivity, count_curve_analysis, PRIMARY_FAMILIES)
    from ..analysis.moderators import meta_regression
    from ..analysis.power import power_report
    from ..common.provenance import code_digest

    run_dir, out = Path(run_dir), Path(out)
    protocol = read_json(run_dir / "protocol.json")
    policy = _analysis_policy(protocol)
    integrity = audit_run(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"aggregated_utc": utc_now(), "protocol_id": protocol.get("protocol_id"), "kind": protocol["kind"],
              "integrity": integrity, "analysis_code_digest": code_digest(), "analysis_policy": policy}
    if protocol["kind"] == "population":
        df = collect_population(run_dir)
        if df.empty:
            raise SystemExit("no completed population jobs to aggregate")
        df.to_csv(out / "records.csv.gz", index=False)
        metric = protocol["evaluation"]["primary_metric"]
        if not policy["inference_enabled"]:
            report["diagnostics"] = _population_diagnostics(df, protocol)
            report = sanitize(report)
            atomic_write_json(out / "report.json", report)
            atomic_write_text(out / "summary.md", render_summary(report))
            return report
        report["primary"] = primary_analysis(df, protocol)
        report["source_information"] = source_information(df, metric)
        report["clock_sensitivity"] = clock_sensitivity(df, protocol["clock"]["primary"], metric)
        ranks, agreement, moderators = {}, {}, {}
        base = df[df["family"].isin(PRIMARY_FAMILIES)
                  & (df["representation"] == protocol["representation"]["primary"])
                  & (df["clock"] == protocol["clock"]["primary"])]
        for kind, comparator in protocol["analysis"]["primary_comparators"].items():
            subset = base[base["kind"] == kind]
            if subset.empty or comparator not in set(subset["method"]):
                continue
            errors = unit_errors(subset, metric)
            table = errors.groupby(["dataset_id", "method"])[metric].apply(_geometric_mean).unstack()
            ranks[kind] = friedman_nemenyi(table)
            secondary = [m for m in protocol["evaluation"].get("secondary_metrics", []) if m in subset.columns and m != "c2st_auc"]
            agreement[kind] = metric_agreement(subset, comparator, [metric] + secondary)
            lr = log_ratio_table(subset, comparator, metric)
            features = [f for f in protocol["analysis"]["moderators"].get(kind, []) if f in lr.columns and lr[f].notna().any()]
            for method in sorted(set(lr["method"])):
                rows = lr[lr["method"] == method]
                if features and len(rows) > len(features) + 2:
                    try:
                        moderators[f"{kind}:{method}"] = meta_regression(rows, features)
                    except Exception as error:  # recorded, not fatal
                        moderators[f"{kind}:{method}"] = {"error": f"{type(error).__name__}: {error}"}
            pilot_method = protocol["analysis"]["primary_family"]["methods"][0]
            if kind == "T-MT" and pilot_method in set(lr["method"]):
                report["power"] = power_report(pilot=lr[lr["method"] == pilot_method])
        report.update(ranks=ranks, metric_agreement=agreement, moderators=moderators)
        report["count_curve"] = count_curve_analysis(df, metric)
    elif not policy["inference_enabled"]:
        # Old measurement result files may contain inferential summaries. Do not re-export
        # those as a pilot report; retain their audited locations for inspection instead.
        items = [{"job_id": done["job_id"], "job_type": done["job"]["job_type"],
                  "dataset_id": done["job"]["dataset_id"],
                  "result_path": str(run_dir / "jobs" / done["job_id"] / "result.json")}
                 for done, _ in iter_completed(run_dir)]
        report["diagnostics"] = {"jobs": len(items), "artifacts": items,
                                 "notes": ["Measurement diagnostic report omits nested inferential results."]}
    else:
        results = {}
        for done, result in iter_completed(run_dir):
            results.setdefault(done["job"]["job_type"], []).append({"job": done["job"], "result": result})
        report["measurement"] = results
    report = sanitize(report)
    atomic_write_json(out / "report.json", report)
    atomic_write_text(out / "summary.md", render_summary(report))
    return report


def render_summary(report: dict) -> str:
    lines = [f"# Aggregate report ({report.get('protocol_id')})", "", f"Aggregated: {report.get('aggregated_utc')}", ""]
    if report.get("analysis_policy", {}).get("mode") == "diagnostic_only":
        diagnostic = report["diagnostics"]
        lines += ["Mode: diagnostic_only. Inference is disabled.", "",
                  f"Audited jobs: {diagnostic['jobs']}.",
                  f"Evaluation rows: {diagnostic.get('evaluation_rows', 'not summarized')}.", "",
                  "No confidence intervals, hypothesis tests, method ranks, power or plateau claims are generated.",
                  "Per-target seed summaries and deduplicated job timings are in report.json; raw records remain separate.",
                  "These results do not justify deleting prespecified comparators or making publication claims."]
        lines += ["", *diagnostic.get("notes", [])]
        return "\n".join(lines) + "\n"
    for key, entry in sorted(report.get("primary", {}).items()):
        if key.startswith("_"):
            continue
        meta = entry.get("meta", {})
        interval = meta.get("ci") or [None, None]
        lines.append(f"- {key} vs {entry['comparator']}: datasets={entry['datasets']}, mean log ratio={meta.get('mu')}, "
                     f"95% CI={interval}, method better in {entry['datasets_method_better']}/{entry['datasets']} datasets, "
                     f"classification={entry.get('classification')}, Holm p={entry.get('p_holm')}")
    family = report.get("primary", {}).get("_holm_family")
    if family:
        lines.append(f"- Holm family: {len(family['planned'])} planned, {len(family['tested'])} tested; "
                     f"untested={family['untested']} (planned multiplicity retained).")
    for key, entry in sorted(report.get("source_information", {}).items()):
        if key.startswith("pooled:"):
            lines.append(f"- source information {key}: k={entry.get('k')}, mean log ratio={entry.get('mu')}, CI={entry.get('ci')}")
    if "measurement" in report:
        for kind, items in sorted(report["measurement"].items()):
            lines.append(f"- {kind}: {len(items)} completed jobs")
    for key, entry in sorted(report.get("count_curve", {}).items()):
        lines.append(f"- count curve {key}: matched contract={entry['matched_contract_passed']}; "
                     f"diagnostic plateau k={entry['plateau_k']}; issues={entry['contract_issues']}")
    lines += ["", "Negative log ratios favour the method; the dataset (or individual) is the inferential unit.",
              "Intervals/classifications are pointwise, not multiplicity-adjusted superiority or noninferiority claims.",
              "Count curves are conditional on a representation fitted on all non-held individuals; k is not total donor requirement."]
    return "\n".join(lines) + "\n"


def render_power(report: dict) -> str:
    lines = ["# Statistical power by inferential unit", "", report["interpretation"], "", "## Exact sign test", "",
             "| n | min p | wins needed | power at 0.65 / 0.75 / 0.85 |", "|---|---|---|---|"]
    for row in report["sign_test"]:
        powers = " / ".join(f"{row['power'][k]:.2f}" for k in ("0.65", "0.75", "0.85"))
        lines.append(f"| {row['n']} | {row['min_p']:.4g} | {row['critical_wins']} | {powers} |")
    lines += ["", "## Two-sample t test: minimal detectable standardized effect (80% power)", "", "| n1 | n2 | d |", "|---|---|---|"]
    lines += [f"| {r['n1']} | {r['n2']} | {r['mde_d']:.2f} |" for r in report["t_test_mde"]]
    lines += ["", "## Prospective normal-approximation dataset requirements (80% power)", "",
              "| tau | independent units/dataset | delta (log ratio) | datasets |", "|---|---|---|---|"]
    lines += [f"| {r['tau']} | {r['tasks_per_dataset']} | {r['delta']} | {r['datasets']} |" for r in report["datasets_needed"]]
    return "\n".join(lines) + "\n"


def cmd_aggregate(args) -> int:
    report = aggregate_run(Path(args.run_dir), Path(args.out or Path(args.run_dir) / "aggregate"))
    print(render_summary(report))
    return 0


def cmd_power(args) -> int:
    from ..analysis.power import power_report

    report = power_report()
    atomic_write_json(Path(args.out) / "power_report.json", sanitize(report))
    atomic_write_text(Path(args.out) / "power_report.md", render_power(report))
    print(render_power(report))
    return 0


# ---------------------------------------------------------------- smoke, preflight, calibration, estimate

def smoke_protocol(protocol: dict, *, quick: bool) -> dict:
    smoke = copy.deepcopy(protocol)
    smoke["protocol_id"] += "-smoke"
    smoke["smoke_only"] = True
    smoke["analysis"]["report_mode"] = "diagnostic_only"
    smoke["min_cells_per_group"] = 20
    smoke["representation"] = {"primary": "pca_train"}
    smoke["representations"] = {"pca_train": {"kind": "pca_train", "n_hvg": 40, "n_components": 5, "max_fit_cells": 5000}}
    smoke["flow_defaults"].update(hidden=32, depth=2, time_dim=8, steps=120, batch_size=64, rk4_steps_per_interval=10)
    smoke["evaluation"]["caps"] = {"energy": 200, "exact_w2": 120, "sliced_w2": 200, "c2st": 200}
    smoke["evaluation"]["sliced_projections"] = 16
    smoke["tasks"]["T-LIO"]["max_held_individuals"] = 2
    smoke["tasks"]["T-K"].update(min_individuals=6, k_values=[1, 2, 4], repeats=1, max_held_individuals=1)
    smoke["seeds"]["stochastic"] = [0]
    if quick:
        keep = {"identity", "mean_shift", "location_scale", "target_marginal", "cfm_independent"}
        smoke["methods"] = {name: config for name, config in smoke["methods"].items() if name in keep}
        smoke["tasks"]["T-K"]["enabled"] = False
        smoke["analysis"]["primary_family"]["methods"] = ["cfm_independent"]
    return smoke


def run_smoke(workdir, *, device: str = "cpu", quick: bool = False) -> dict:
    from ..data.adapters import synthetic_dataset
    from ..data.standard import validate_measurement, validate_standard, write_standard

    workdir = Path(workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"{workdir} is not empty")
    specs = [("SYN-TRANSLATION", "translation", 11)]
    if not quick:
        specs += [("SYN-REWEIGHT", "reweighting", 12), ("SYN-ROTATION", "rotation", 13)]
    contracts = {}
    for dataset_id, mode, seed in specs:
        adata = synthetic_dataset(dataset_id=dataset_id, n_individuals=6, times=(0.0, 1.0, 2.0, 3.0), cells=40, genes=60,
                                  mode=mode, seed=seed)
        path = workdir / "standard" / dataset_id / "standard.h5ad"
        write_standard(adata, path)
        contracts[dataset_id] = {"dataset_id": dataset_id, "status": "ingested", "dataset_file": str(path),
                                 "dataset_sha256": sha256_file(path), "contract": validate_standard(adata, min_cells=20),
                                 "measurement": validate_measurement(adata, min_cells=20, min_specimens=3)}
    protocol = smoke_protocol(read_json(DEFAULT_PROTOCOLS["population"]), quick=quick)
    started = time.perf_counter()
    summary = plan_run(protocol, contracts, workdir / "run", root=workdir, ids=list(contracts))
    counts = execute_run(workdir / "run", root=workdir, device=device)
    problems, checked = verify_population_sample(workdir / "run", 5)
    report = aggregate_run(workdir / "run", workdir / "aggregate")
    ok = (counts["failed"] == 0 and not problems and checked > 0
          and report["integrity"]["complete"] and report["diagnostics"]["evaluation_rows"] > 0)
    evidence = {"completed_utc": utc_now(), "quick": quick, "device": device, "jobs": summary["jobs"], "counts": counts,
                "verify_checked": checked, "verify_problems": problems, "seconds": round(time.perf_counter() - started, 2),
                "analysis_mode": report["analysis_policy"]["mode"],
                "diagnostic_evaluation_rows": report["diagnostics"]["evaluation_rows"],
                "status": "passed" if ok else "failed",
                "note": "Synthetic data only; no real cohort is read."}
    atomic_write_json(workdir / "smoke_evidence.json", evidence)
    return evidence


def cmd_smoke(args) -> int:
    evidence = run_smoke(Path(args.workdir), device=args.device or "cpu", quick=args.quick)
    print(json.dumps(evidence, indent=1))
    return 0 if evidence["status"] == "passed" else 1


def _tiny_flow_fit(device: str) -> float:
    from ..population.methods import create
    from ..population.tasks import MethodInputs

    rng = np.random.default_rng(0)
    train = {(f"i{i}", t): rng.normal(t, 1.0, size=(500, 30)) for i in range(3) for t in (0.0, 1.0, 2.0)}
    inputs = MethodInputs("T-LIO", train, {0.0: 0.0, 1.0: 0.5, 2.0: 1.0}, None, "held")
    started = time.perf_counter()
    create("cfm_independent", hidden=64, depth=2, steps=200, batch_size=256).fit(inputs, seed=0, device=device)
    return round(time.perf_counter() - started, 3)


def run_preflight(root, *, require_cuda: bool) -> dict:
    import importlib.metadata as metadata

    import torch

    from ..data.registry import load_registry, validate_registry
    from .jobs import protocol_hash

    report = {"utc": utc_now(), "python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "anndata", "h5py", "POT", "torch", "statsmodels", "pyarrow"):
        try:
            report["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["registry_problems"] = validate_registry(load_registry())
    report["protocol_hashes"] = {kind: protocol_hash(read_json(path)) for kind, path in DEFAULT_PROTOCOLS.items()}
    report["disk_free_gb"] = round(shutil.disk_usage(root).free / 1e9, 1)
    cuda = {"available": bool(torch.cuda.is_available()), "torch_cuda": torch.version.cuda}
    if cuda["available"]:
        properties = torch.cuda.get_device_properties(0)
        cuda.update(name=properties.name, memory_gb=round(properties.total_memory / 1e9, 1), count=torch.cuda.device_count())
        matrix = torch.randn(1024, 1024, device="cuda")
        cuda["matmul_finite"] = bool(torch.isfinite(matrix @ matrix).all().item())
        cuda["tiny_flow_fit_seconds"] = _tiny_flow_fit("cuda")
    report["cuda"] = cuda
    report["cpu_tiny_flow_fit_seconds"] = _tiny_flow_fit("cpu")
    missing = [name for name, version in report["packages"].items() if version is None]
    failed = (require_cuda and not cuda["available"]) or report["registry_problems"] or missing
    report["status"] = "failed" if failed else "passed"
    return report


def cmd_preflight(args) -> int:
    report = run_preflight(Path(args.root), require_cuda=args.require_cuda)
    if args.out:
        atomic_write_json(args.out, report)
    print(json.dumps(report, indent=1))
    return 0 if report["status"] == "passed" else 1


def run_calibration(device: str, *, dims: int = 30, cells: int = 1500, individuals: int = 6, times: int = 4,
                    steps: int = 500, batch: int = 512) -> dict:
    """Time each flow variant, one prediction+evaluation and a representative baseline on synthetic data."""
    from ..population.methods import create
    from ..population.metrics import evaluate
    from ..population.tasks import MethodInputs

    rng = np.random.default_rng(0)
    clock = {float(t): t / (times - 1) for t in range(times)}
    train = {(f"i{i}", float(t)): rng.normal(t, 1.0, size=(cells, dims)) for i in range(individuals) for t in range(times)}
    inputs = MethodInputs("T-LIO", train, clock, None, "held")
    source, target = rng.normal(0.0, 1.0, size=(cells, dims)), rng.normal(1.0, 1.0, size=(cells, dims))
    timings = {}
    for name in FLOWS:
        method = create(name, steps=steps, batch_size=batch)
        started = time.perf_counter()
        method.fit(inputs, seed=0, device=device)
        timings[name] = (time.perf_counter() - started) * 1000.0 / steps
    method = create("cfm_independent", steps=10, batch_size=batch)
    method.fit(inputs, seed=0, device=device)
    started = time.perf_counter()
    prediction = method.predict(source, individual="held", source_time=0.0, target_time=1.0)
    evaluate(prediction, target, seed=0, caps={"energy": 2000, "exact_w2": 1000, "sliced_w2": 4000, "c2st": 2000})
    pair_seconds = time.perf_counter() - started
    baseline = create("population_ot")
    baseline.fit(inputs)
    started = time.perf_counter()
    baseline.predict(source, individual="held", source_time=0.0, target_time=1.0)
    return {"device": device, "measured_utc": utc_now(), "synthetic": True,
            "settings": {"dims": dims, "cells": cells, "individuals": individuals, "times": times, "batch": batch, "steps": steps},
            "seconds_per_1000_steps": {k: round(v, 3) for k, v in timings.items()},
            "seconds_per_pair_evaluation": round(pair_seconds, 3),
            "seconds_per_baseline_pair": round(time.perf_counter() - started, 3)}


def cmd_calibrate(args) -> int:
    calibration = run_calibration(args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    if args.out:
        atomic_write_json(args.out, calibration)
    print(json.dumps(calibration, indent=1))
    return 0


def synthetic_contract(individuals: int, timepoints: int) -> dict:
    """Contract with every individual observed at every time (used only for budget estimates)."""
    names = [f"ind{i}" for i in range(int(individuals))]
    times = [float(t) for t in range(int(timepoints))]
    missing = [{"held_time": times[k], "source_time": times[k - 1], "anchor_time": times[k + 1], "individuals": names}
               for k in range(1, len(times) - 1)]
    pairs = [{"source_time": times[k], "target_time": times[k + 1], "individuals": names} for k in range(len(times) - 1)]
    lio = names if len(names) >= 3 else []
    return {"status": "accepted", "times": times, "individuals": names, "missing_time_tasks": missing,
            "consecutive_pairs": pairs, "lio_individuals": lio,
            "eligible": {"T-MT": bool(missing), "T-LIO": len(lio) >= 3, "T-K": len(lio) >= 4}}


def estimate_budget(protocol: dict, calibration: dict, entries) -> dict:
    """Indicative worker time, not a GPU-utilization or rental-time measurement."""
    from ..population.methods import REGISTRY
    from ..population.worker import _method_params
    from .jobs import method_base, plan_population_jobs

    wanted = set(protocol["datasets"]["tier_W"]) | set(protocol["datasets"]["tier_G"])
    by_dataset, skipped, total_jobs = {}, {}, 0
    flow_seconds = baseline_seconds = 0.0
    for entry in entries:
        if entry.id not in wanted:
            continue
        individuals, timepoints = entry.design.get("individuals"), entry.design.get("timepoints")
        if not isinstance(individuals, int) or not isinstance(timepoints, int):
            skipped[entry.id] = "design counts not numeric in registry; resolved after Stage A"
            continue
        planned = plan_population_jobs(protocol, {entry.id: {"contract": synthetic_contract(individuals, timepoints),
                                                             "dataset_sha256": "estimate", "dataset_file": ""}})
        flow = baseline = 0.0
        flow_jobs = baseline_jobs = evaluations_total = 0
        flow_methods = {}
        for job in planned:
            pairs = len(job.task["pairs"])
            # The worker evaluates own-source once for ordinary tasks and
            # own + one held-out substitute for each separate source-swap job.
            evaluations = pairs * (2 if job.task["family"] == "source_swap" else 1)
            evaluations_total += evaluations
            base = method_base(protocol, job.method)
            if REGISTRY[base].spec.family == "flow":
                flow_jobs += 1
                flow_methods[job.method] = flow_methods.get(job.method, 0) + 1
                steps = int(_method_params(protocol, job.method)["steps"])
                flow += steps / 1000.0 * calibration["seconds_per_1000_steps"][base]
                flow += evaluations * calibration["seconds_per_pair_evaluation"]
            else:
                baseline_jobs += 1
                baseline += evaluations * (calibration["seconds_per_pair_evaluation"]
                                            + calibration["seconds_per_baseline_pair"])
        by_dataset[entry.id] = {"jobs": len(planned), "flow_jobs": flow_jobs, "baseline_jobs": baseline_jobs,
                                "prediction_evaluations": evaluations_total, "flow_methods": flow_methods,
                                "flow_worker_seconds": round(flow, 3), "baseline_worker_seconds": round(baseline, 3),
                                "flow_worker_hours": round(flow / 3600, 2),
                                "baseline_worker_hours": round(baseline / 3600, 2),
                                "individuals": individuals, "timepoints": timepoints}
        total_jobs += len(planned)
        flow_seconds += flow
        baseline_seconds += baseline
    return {"schema_version": 2, "calibration_device": calibration["device"], "jobs": total_jobs,
            "flow_worker_hours": round(flow_seconds / 3600, 2),
            "baseline_worker_hours": round(baseline_seconds / 3600, 2),
            "by_dataset": by_dataset, "skipped": skipped,
            "note": "Registry counts assume every individual at every time; Stage A replaces these hypothetical designs.",
            "limitations": [
                "Worker wall time includes CPU coupling/scoring; it is not GPU-active time or billable instance hours.",
                "Flow aliases use the base-method calibration; sigma/capacity changes need their own target-host pilot.",
                "One synthetic pair/baseline calibration approximates all predictions and baselines, not a validated bound.",
                "Loading, representation fitting, caching, features, output IO, aggregation and measurement jobs are excluded.",
                "No linear multi-GPU speedup or target-host memory sufficiency is implied.",
            ]}


def cmd_estimate(args) -> int:
    calibration = read_json(args.calibration)
    budget = estimate_budget(read_json(args.protocol), calibration, _entries())
    if args.out:
        atomic_write_json(args.out, budget)
    print(json.dumps(budget, indent=1))
    return 0


def cmd_bundle(args) -> int:
    from ..bundle.build import build_bundle

    summary = build_bundle(Path(args.output), root=Path(args.root), runtime=args.runtime,
                           include_discovery=not args.no_discovery, include_standardized=args.include_standardized,
                           archive=args.archive)
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=1))
    return 0


def cmd_verify_bundle(args) -> int:
    from ..bundle.build import verify_bundle

    problems = verify_bundle(Path(args.bundle or args.root))
    print(json.dumps({"problems": problems}, indent=1))
    return 1 if problems else 0


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m expansion", description=__doc__)
    parser.add_argument("--root", default=".", help="project or bundle root (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("registry"); p.add_argument("--ids", nargs="*"); p.set_defaults(func=cmd_registry)
    p = sub.add_parser("fetch"); p.add_argument("--ids", nargs="*"); p.add_argument("--execute", action="store_true")
    p.add_argument("--approval"); p.set_defaults(func=cmd_fetch)
    p = sub.add_parser("probe"); p.add_argument("--ids", nargs="*"); p.add_argument("--standard-dir", default="data/standardized")
    p.set_defaults(func=cmd_probe)
    p = sub.add_parser("stage-a"); p.add_argument("--ids", nargs="*"); p.add_argument("--standard-dir", default="data/standardized")
    p.add_argument("--min-cells", type=int, default=50); p.add_argument("--force", action="store_true")
    p.add_argument("--execute-fetch", action="store_true"); p.add_argument("--approval"); p.set_defaults(func=cmd_stage_a)
    p = sub.add_parser("plan"); p.add_argument("--protocol", required=True); p.add_argument("--run-dir", required=True)
    p.add_argument("--standard-dir", default="data/standardized"); p.add_argument("--ids", nargs="*"); p.set_defaults(func=cmd_plan)
    p = sub.add_parser("run"); p.add_argument("--run-dir", required=True); p.add_argument("--shard", default="0/1")
    p.add_argument("--device"); p.add_argument("--require-cuda", action="store_true"); p.add_argument("--max-jobs", type=int)
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--no-predictions", action="store_true")
    p.add_argument("--approval", help="reviewed Stage-B approval receipt bound to this sealed plan")
    p.add_argument("--allow-code-change", action="store_true", help="deprecated; changed code always requires a fresh plan"); p.set_defaults(func=cmd_run)
    p = sub.add_parser("status"); p.add_argument("--run-dir", required=True); p.set_defaults(func=cmd_status)
    p = sub.add_parser("verify"); p.add_argument("--run-dir", required=True); p.add_argument("--sample", type=int, default=20)
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("aggregate"); p.add_argument("--run-dir", required=True); p.add_argument("--out"); p.set_defaults(func=cmd_aggregate)
    p = sub.add_parser("power"); p.add_argument("--out", required=True); p.set_defaults(func=cmd_power)
    p = sub.add_parser("smoke"); p.add_argument("--workdir", required=True); p.add_argument("--device")
    p.add_argument("--quick", action="store_true"); p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("preflight"); p.add_argument("--require-cuda", action="store_true"); p.add_argument("--out")
    p.set_defaults(func=cmd_preflight)
    p = sub.add_parser("calibrate"); p.add_argument("--device"); p.add_argument("--out"); p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("estimate"); p.add_argument("--calibration", required=True)
    p.add_argument("--protocol", default=str(DEFAULT_PROTOCOLS["population"])); p.add_argument("--out")
    p.set_defaults(func=cmd_estimate)
    p = sub.add_parser("bundle"); p.add_argument("--output", required=True); p.add_argument("--runtime", default="none",
                                                                                          choices=["none", "vendored"])
    p.add_argument("--no-discovery", action="store_true"); p.add_argument("--include-standardized")
    p.add_argument("--archive"); p.set_defaults(func=cmd_bundle)
    p = sub.add_parser("verify-bundle"); p.add_argument("--bundle"); p.set_defaults(func=cmd_verify_bundle)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
