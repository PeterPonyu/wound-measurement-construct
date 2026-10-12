"""Sharded, resumable execution with integrity-checked completion records.

A job counts as done only when ``done.json`` exists, names the same job and
matches the hashes of every output written by the job.  Leftovers from an
interrupted attempt are removed before a retry; completed outputs are never
overwritten, and tampering raises instead of silently re-running.
"""
from __future__ import annotations

import os
import socket
import time
import traceback
from pathlib import Path
import re

from ..common.io import atomic_write_json, canonical_json, read_json, sanitize, sha256_file

CONTROL_FILES = {"done.json", "running.json", "failed.json"}


class IntegrityError(RuntimeError):
    """A completion record no longer matches its outputs or its job."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def shard_of(job_id: str, shards: int) -> int:
    return int(job_id[:8], 16) % int(shards)


def job_dir(run_dir, job_id: str) -> Path:
    return Path(run_dir) / "jobs" / job_id


def _output_hashes(directory: Path) -> dict:
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir())
            if path.is_file() and path.name not in CONTROL_FILES}


def _verify_invocation(directory: Path, invocation: dict, job) -> None:
    """Bind completed work to the immutable invocation event that authorized it.

    Older queue-only callers may omit ``invocation`` and remain intentionally
    compatible.  Runs produced by ``execute_run`` always carry the binding;
    accepting a missing or altered event there would make a completion record
    independent of the execution authorization that created it.
    """
    if not isinstance(invocation, dict) or not invocation.get("id") or not invocation.get("event_sha256"):
        raise IntegrityError("completion record has an invalid invocation binding")
    invocation_id = str(invocation["id"])
    if not re.fullmatch(r"[0-9a-f]{32}", invocation_id):
        raise IntegrityError("completion record has an invalid invocation id")
    run_dir = directory.parents[1]
    event_path = run_dir / "logs" / f"invocation_{invocation_id}.json"
    if not event_path.is_file() or sha256_file(event_path) != invocation["event_sha256"]:
        raise IntegrityError("completion record does not match its invocation event")
    event = read_json(event_path)
    if event.get("invocation_id") != invocation_id:
        raise IntegrityError("invocation event id does not match completion record")
    if event.get("jobs_sha256") != sha256_file(run_dir / "jobs.jsonl"):
        raise IntegrityError("invocation event is not bound to the planned jobs")
    if event.get("protocol_hash") != job.protocol_hash:
        raise IntegrityError("invocation event is not bound to the planned protocol")
    if event.get("code_digest") != job.to_dict().get("params", {}).get("code_digest"):
        raise IntegrityError("invocation event is not bound to the planned code")


def verify_done(directory, job, *, require_invocation: bool = False) -> bool:
    directory = Path(directory)
    for path in (directory, *directory.parents):
        if path.is_symlink():
            raise IntegrityError(f"symlink in completion directory ancestry: {path}")
    if directory.exists() and any(path.is_symlink() for path in directory.iterdir()):
        raise IntegrityError("symlink in job output/control files")
    done_path = directory / "done.json"
    if not done_path.exists():
        return False
    done = read_json(done_path)
    if require_invocation and done.get("invocation") is None:
        raise IntegrityError("formal completion record lost its invocation binding")
    if done.get("job_id") != job.job_id or canonical_json(done.get("job")) != canonical_json(job.to_dict()):
        raise IntegrityError(f"{directory}: completion record belongs to a different job")
    if done.get("invocation") is not None:
        _verify_invocation(directory, done["invocation"], job)
    if _output_hashes(directory) != done.get("outputs"):
        raise IntegrityError(f"{directory}: outputs changed after completion")
    return True


def job_state(run_dir, job) -> str:
    directory = job_dir(run_dir, job.job_id)
    for name, state in (("done.json", "done"), ("failed.json", "failed"), ("running.json", "running")):
        if (directory / name).exists():
            return state
    return "pending"


def run_queue(jobs, run_dir, executor, *, shard=(0, 1), max_jobs=None, retry_failed: bool = False, log=print, invocation=None) -> dict:
    run_dir = Path(run_dir)
    for path in (run_dir / "jobs", run_dir, *run_dir.parents):
        if path.is_symlink():
            raise IntegrityError(f"symlink in queue output ancestry: {path}")
    (run_dir / "jobs").mkdir(parents=True, exist_ok=True)
    index, total = shard
    counts = {"done": 0, "failed": 0, "skipped_done": 0, "skipped_failed": 0, "other_shard": 0}
    executed = 0
    for job in sorted(jobs, key=lambda item: item.job_id):
        if shard_of(job.job_id, total) != index:
            counts["other_shard"] += 1
            continue
        directory = job_dir(run_dir, job.job_id)
        for parent in (directory, *directory.parents):
            if parent.is_symlink():
                raise IntegrityError(f"symlink in job directory ancestry: {parent}")
        if verify_done(directory, job, require_invocation=invocation is not None):
            counts["skipped_done"] += 1
            continue
        # A hard kill can leave running.json behind.  Never delete that
        # directory and silently restart: the operator must inspect the
        # process/host and deliberately choose a fresh run root or perform a
        # documented recovery action.  This is especially important on a
        # remote host where two workers may share a filesystem.
        if (directory / "running.json").exists():
            raise IntegrityError(f"{directory}: active or stale running marker; refuse to overwrite")
        if (directory / "failed.json").exists() and not retry_failed:
            counts["skipped_failed"] += 1
            continue
        if max_jobs is not None and executed >= max_jobs:
            break
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation closes the check-then-write race between workers.
        # A hard kill deliberately leaves this marker for operator inspection.
        try:
            marker = os.open(directory / "running.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise IntegrityError(f"{directory}: concurrent worker already claimed this job") from error
        with os.fdopen(marker, "w") as stream:
            stream.write(canonical_json({"job_id": job.job_id, "host": socket.gethostname(),
                                         "pid": os.getpid(), "started_utc": utc_now(),
                                         "invocation": invocation}))
            stream.flush()
            os.fsync(stream.fileno())
        # Another worker may have completed after our first done check but
        # before this exclusive claim. Its outputs must never be cleaned up.
        try:
            completed = verify_done(directory, job, require_invocation=invocation is not None)
        except BaseException:
            (directory / "running.json").unlink()
            raise
        if completed:
            (directory / "running.json").unlink()
            counts["skipped_done"] += 1
            continue
        for leftover in directory.iterdir():
            if leftover.is_file() and leftover.name != "running.json":
                leftover.unlink()
        started = time.perf_counter()
        try:
            result = executor(job, directory)
            atomic_write_json(directory / "result.json", sanitize(result))
            atomic_write_json(directory / "done.json", {"job_id": job.job_id, "job": job.to_dict(),
                                                        "invocation": invocation,
                                                        "outputs": _output_hashes(directory),
                                                        "seconds": time.perf_counter() - started,
                                                        "finished_utc": utc_now()})
            counts["done"] += 1
        except Exception as error:  # a failing job is recorded and the queue continues
            atomic_write_json(directory / "failed.json", {"job_id": job.job_id, "error": f"{type(error).__name__}: {error}",
                                                          "invocation": invocation,
                                                          "traceback": traceback.format_exc(), "finished_utc": utc_now()})
            counts["failed"] += 1
            log(f"FAILED {job.job_id} {job.job_type} {job.dataset_id} {job.method}: {error}")
        finally:
            (directory / "running.json").unlink(missing_ok=True)
        executed += 1
    return counts


def status(run_dir, jobs) -> dict:
    counts = {"done": 0, "failed": 0, "pending": 0, "running": 0}
    failed = []
    for job in jobs:
        state = job_state(run_dir, job)
        counts[state] += 1
        if state == "failed":
            failed.append({"job_id": job.job_id, "dataset_id": job.dataset_id, "method": job.method,
                           "error": read_json(job_dir(run_dir, job.job_id) / "failed.json")["error"]})
    return {"counts": counts, "failed": failed, "total": len(jobs)}
