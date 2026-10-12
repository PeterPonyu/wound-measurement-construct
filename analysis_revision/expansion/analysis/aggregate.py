"""Collect completed job results into tidy tables."""
from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from ..common.io import content_hash, read_json, sha256_file
from ..runner.jobs import read_jobs, protocol_hash, method_base
from ..runner.queue import IntegrityError, job_dir, job_state, verify_done


def audit_run(run_dir, *, historical_queue_only=False) -> dict:
    """Fail closed: only an intact, complete, pinned plan can feed scientific summaries."""
    run_dir = Path(run_dir)
    planned = read_jobs(run_dir / "jobs.jsonl")
    ids = [job.job_id for job in planned]
    if not ids or len(ids) != len(set(ids)):
        raise IntegrityError("empty plan or duplicate planned job ids")
    protocol = read_json(run_dir / "protocol.json")
    provenance = read_json(run_dir / "provenance.json")
    if provenance.get("jobs_sha256") != sha256_file(run_dir / "jobs.jsonl"):
        raise IntegrityError("plan changed or is not sealed; re-plan into a fresh directory")
    digest = protocol_hash(protocol)
    code = provenance.get("code_digest")
    if not code or any(job.protocol_hash != digest or job.params.get("code_digest") != code for job in planned):
        raise IntegrityError("plan does not match pinned protocol/code provenance")
    for path in (run_dir / "logs").glob("run_*.json"):
        log = read_json(path)
        if log.get("code_changed_since_planning") or log.get("code_digest") != code:
            raise IntegrityError("mixed-code execution log; re-plan into a fresh directory")
    unknown = [p.name for p in (run_dir / "jobs").glob("*") if p.is_dir() and p.name not in set(ids)]
    if unknown:
        raise IntegrityError(f"unplanned job directories: {unknown}")
    counts = dict(done=0, failed=0, running=0, pending=0)
    for job in planned:
        directory = job_dir(run_dir, job.job_id)
        if verify_done(directory, job, require_invocation=not historical_queue_only):
            if not (directory / "result.json").is_file():
                raise IntegrityError(f"{job.job_id}: completed job has no result")
            counts["done"] += 1
        else:
            counts[job_state(run_dir, job)] += 1
    if counts["done"] != len(planned):
        raise IntegrityError(f"incomplete run: {counts}; complete all {len(planned)} planned jobs before aggregation")
    return {"complete": True, "total": len(planned), "counts": counts, "code_digest": code,
            "protocol_hash": digest, "output_hashes_verified": True}


def iter_completed(run_dir):
    audit_run(run_dir)
    for job in read_jobs(Path(run_dir) / "jobs.jsonl"):
        directory = job_dir(run_dir, job.job_id)
        yield read_json(directory / "done.json"), read_json(directory / "result.json")


def collect_population(run_dir) -> pd.DataFrame:
    from ..population.methods import REGISTRY

    protocol = read_json(Path(run_dir) / "protocol.json")
    rows = []
    for done, result in iter_completed(run_dir):
        job = done["job"]
        if job["job_type"] != "population_fit":
            continue
        task = job["task"]
        subset = task.get("training_individuals")
        expected_clock = job["params"].get("clock", protocol["clock"]["primary"])
        if result.get("clock") != expected_clock:
            raise IntegrityError(f"{done['job_id']}: missing or mismatched result clock")
        method = method_base(protocol, job["method"])
        settings = protocol["methods"][job["method"]].get("params", {})
        stochastic = REGISTRY[method].spec.stochastic
        budget = {**(protocol.get("flow_defaults", {}) if stochastic else {}), **settings}
        expected = {(i, "own") for i in range(len(task["pairs"]))}
        if task["family"] == "source_swap":
            expected |= {(i, f"substitute:{task['substitute']}") for i in range(len(task["pairs"]))}
        records = result["records"]
        from ..population.metrics import PAIRED_EVALUATION_CONTRACT, evaluation_contract_options

        contract = evaluation_contract_options(protocol.get("evaluation", {}))
        paired = contract is not None and contract["name"] == PAIRED_EVALUATION_CONTRACT
        if paired:
            expected = {(i, v, draw) for i, v in expected for draw in range(contract["draws"])}
        actual = [(r.get("pair_index"), r.get("source_variant"), r.get("draw_index")) if paired
                  else (r.get("pair_index"), r.get("source_variant")) for r in records]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise IntegrityError(f"{done['job_id']}: incomplete or duplicate planned evaluation pairs")
        for record in records:
            pair = task["pairs"][record["pair_index"]]
            if any(record.get(key) != pair.get(key) for key in ("individual", "source_time", "target_time", "anchor_time")):
                raise IntegrityError(f"{done['job_id']}: evaluation target differs from planned pair")
            if paired:
                from ..common.rng import derive_seed

                particles = record.get("matched_particles")
                expected_draw = derive_seed("paired-draw-v2", contract["draw_seed"], task["dataset_id"], task["kind"],
                    task["family"], pair["individual"], pair["source_time"], pair["target_time"], record["draw_index"])
                if (type(particles) is not int or not 2 <= particles <= contract["sample_size"]
                        or record.get("n_prediction") != particles or record.get("n_target") != particles
                        or record.get("fit_seed") != job["seed"] or record.get("draw_seed") != expected_draw
                        or record.get("score_seed") != derive_seed("paired-score-v2", expected_draw)
                        or record.get("evaluation_contract") != contract["name"] or record.get("finite") is not True):
                    raise IntegrityError(f"{done['job_id']}: invalid paired sampling/seed/count contract")
            value = record.get(protocol["evaluation"]["primary_metric"])
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise IntegrityError(f"{done['job_id']}: invalid primary error metric")
        base = {"job_id": done["job_id"], "dataset_id": job["dataset_id"], "method": job["method"], "seed": job["seed"],
                "task_hash": content_hash(task),
                "representation": job["representation"], "kind": task["kind"], "family": task["family"],
                "held_time": task.get("held_time"), "held_individual": task.get("held_individual"),
                "training_size": len(subset) if subset is not None else None, "repeat": task.get("repeat", 0),
                "fit_seconds": result.get("fit_seconds"), "variant": result.get("variant"),
                "clock": expected_clock, "stochastic": stochastic,
                "training_individuals": tuple(subset) if subset is not None else (),
                "budget_hash": content_hash(budget),
                "evaluation_hash": content_hash(protocol["evaluation"])}
        rows.extend({**record, **base} for record in records)
    return pd.DataFrame(rows)
