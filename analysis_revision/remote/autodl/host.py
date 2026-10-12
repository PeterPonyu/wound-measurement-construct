#!/usr/bin/env python3
"""Dependency-free read-only host inventory. No network or CUDA allocation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path


def safe_path(value):
    path = Path(os.path.abspath(value))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"symlink in target ancestry: {parent}")
    return path


# Local training writes the run directory only. Eight GiB left after that write
# is the admission margin. It is not a fraction of the whole laptop disk.
LOCAL_OUTPUT_BUDGET_BYTES = 2 << 30
LOCAL_FREE_MARGIN_BYTES = 8 << 30


def disk_gate(path, planned_bytes, reserve_fraction=.30, reserve_bytes=None):
    if type(planned_bytes) is not int or planned_bytes <= 0:
        raise ValueError("planned bytes must be a known positive integer")
    target = safe_path(path)
    existing = target
    while not existing.exists():
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    report = {"target": str(target), "measured_volume_path": str(existing),
              "total_bytes": usage.total, "free_bytes": usage.free,
              "planned_increment_bytes": planned_bytes,
              "post_write_free_bytes": usage.free - planned_bytes}
    if reserve_bytes is not None:
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("absolute reserve must be a non-negative integer")
        report["required_reserve_bytes"] = reserve_bytes
        report["required_free_fraction"] = None
        report["passed"] = usage.free - planned_bytes >= reserve_bytes
        return report
    if not .30 <= reserve_fraction < 1:
        raise ValueError("post-write reserve must be at least 30%")
    report["required_reserve_bytes"] = None
    report["required_free_fraction"] = reserve_fraction
    report["passed"] = usage.free - planned_bytes >= usage.total * reserve_fraction
    return report


def local_disk_gate(path, planned_bytes=LOCAL_OUTPUT_BUDGET_BYTES):
    """Admit a local run from the bytes it will write, plus a fixed free margin."""
    return disk_gate(path, planned_bytes, reserve_bytes=LOCAL_FREE_MARGIN_BYTES)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _unsigned_integer(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _proven_memory_chain():
    """Read a complete v2 chain, never infer unlimited from an unreadable file.

    A namespace/subtree mount can hide a limiting ancestor. Host-only fallback
    therefore requires a root cgroup2 mount, the kernel's non-root cgroup.type
    file to be *absent* (not unreadable), and readable max/current pairs on every
    non-root ancestor of the current membership. Unsupported layouts stay unknown.
    """
    membership = _read('/proc/self/cgroup')
    mountinfo = _read('/proc/self/mountinfo')
    if not membership or not mountinfo:
        return False, None
    entries = [line.split(':', 2) for line in membership.splitlines()]
    if len(entries) != 1 or len(entries[0]) != 3 or entries[0][:2] != ['0', '']:
        return False, None
    member = entries[0][2]
    if not member.startswith('/') or any(part in ('.', '..') for part in member.split('/')):
        return False, None
    mounts = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if ' - ' in line:
            separator = fields.index('-')
            if len(fields) > separator + 1 and fields[separator + 1] == 'cgroup2':
                mounts.append(fields)
    if len(mounts) != 1 or len(mounts[0]) < 6 or mounts[0][3:5] != ['/', '/sys/fs/cgroup']:
        return False, None
    root = Path('/sys/fs/cgroup')
    stats = [line.split() for line in (_read(root / 'cgroup.stat') or '').splitlines()]
    if not stats or any(len(row) != 2 or _unsigned_integer(row[1]) is None for row in stats) \
            or not {'nr_descendants', 'nr_dying_descendants'} <= {row[0] for row in stats}:
        return False, None
    try:
        (root / 'cgroup.type').stat()
    except FileNotFoundError:
        pass
    except OSError:
        return False, None
    else:
        return False, None
    remaining = []
    node = root / member.lstrip('/')
    while node != root:
        limit = _read(node / 'memory.max')
        current = _unsigned_integer(_read(node / 'memory.current'))
        bound = _unsigned_integer(limit)
        if current is None or (limit != 'max' and bound is None):
            return False, None
        if bound is not None:
            remaining.append(max(0, bound - current))
        node = node.parent
    return True, min(remaining) if remaining else None


def memory_snapshot():
    values = []
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            match = re.fullmatch(r'MemAvailable:\s+([0-9]+)\s+kB', line)
            value = _unsigned_integer(match[1]) if match else None
            values.append(value * 1024 if value is not None else None)
    available = values[0] if len(values) == 1 else None
    limit, current = _read("/sys/fs/cgroup/memory.max"), _read("/sys/fs/cgroup/memory.current")
    bound, used = _unsigned_integer(limit), _unsigned_integer(current)
    cgroup_free = max(0, bound - used) if bound is not None and used is not None else None
    cgroup_known = cgroup_free is not None
    chain_known = False
    if not cgroup_known and ((limit == 'max' and used is not None) or (limit is None and current is None)):
        cgroup_known, cgroup_free = _proven_memory_chain()
        chain_known = cgroup_known
    known = [value for value in (available, cgroup_free) if value is not None]
    return {"host_available_bytes": available, "cgroup_remaining_bytes": cgroup_free,
            "cgroup_memory_known": cgroup_known,
            "cgroup_chain_fully_known": chain_known,
            "host_memory_fallback": cgroup_known and cgroup_free is None and available is not None,
            "effective_available_bytes": min(known) if cgroup_known and known else None}


def _query(arguments, evidence=None):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=10, check=False)
        if evidence is not None:
            evidence.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired) as error:
        if evidence is not None:
            stdout = getattr(error, 'stdout', None)
            stderr = getattr(error, 'stderr', None)
            evidence.update(stdout=stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout,
                            stderr=stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr,
                            returncode=None, error=type(error).__name__)
        return None


def _csv_rows(text, columns, label, errors):
    for number, raw in enumerate((text or '').splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = next(csv.reader([raw], strict=True, skipinitialspace=True))
        except csv.Error:
            errors.append(f'{label} line {number}: malformed CSV')
            continue
        if len(row) != columns or any(not field.strip() for field in row):
            errors.append(f'{label} line {number}: unknown or malformed fields')
            continue
        yield number, [field.strip() for field in row]


def gpu_snapshot():
    evidence = {'gpu': {}, 'compute': {}}
    gpu = _query(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
                  "--format=csv,noheader,nounits"], evidence['gpu'])
    processes = _query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                        "--format=csv,noheader,nounits"], evidence['compute'])
    errors = []
    rows = []
    indices, uuids = set(), set()
    for number, line in _csv_rows(gpu, 7, 'gpu', errors):
        index, total, used, utilization = [_unsigned_integer(line[i]) for i in (0, 4, 5, 6)]
        if any(value is None for value in (index, total, used, utilization)) or total == 0 \
                or used > total or utilization > 100 \
                or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', line[1]) \
                or line[2].lower() in ('n/a', '[n/a]', 'unknown', '[unknown]', 'not supported', '[not supported]') \
                or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', line[3]) \
                or index in indices or line[1] in uuids:
            errors.append(f'gpu line {number}: unknown, invalid or duplicate GPU fields')
            continue
        indices.add(index); uuids.add(line[1])
        rows.append(dict(zip(("index", "uuid", "name", "driver", "memory_total_mib",
                              "memory_used_mib", "utilization_percent"), line)))
    apps = []
    for number, line in _csv_rows(processes, 3, 'compute', errors):
        pid, used = _unsigned_integer(line[1]), _unsigned_integer(line[2])
        if line[0] not in uuids or pid is None or pid == 0 or used is None:
            errors.append(f'compute line {number}: unknown or invalid active process fields')
            continue
        apps.append(dict(zip(("gpu_uuid", "pid", "memory_mib"), line)))
    completed = gpu is not None and processes is not None
    return {"query_succeeded": completed and bool(rows) and not errors, "gpus": rows,
            "queries_completed": completed, "parse_errors": errors, "query_details": evidence,
            "raw_gpu_query_stdout": evidence['gpu'].get('stdout', gpu),
            "raw_compute_query_stdout": evidence['compute'].get('stdout', processes),
            "compute_processes": apps, "allocation_tested": False,
            "idle_gpu_assignment": "requires_explicit_user_confirmation_not_inferred_from_snapshot"}


def inspect(path):
    target = safe_path(path)
    existing = target
    while not existing.exists():
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    affinity = sorted(os.sched_getaffinity(0))
    return {"schema_version": 1, "kind": "autodl_readonly_host_snapshot", "host": socket.gethostname(),
            "system": platform.system(), "machine": platform.machine(), "glibc": platform.libc_ver(),
            "bootstrap_python": platform.python_version(), "cpu_affinity_count": len(affinity),
            "cpu_affinity": affinity,
            "cpu_quota": _read("/sys/fs/cgroup/cpu.max"), "memory": memory_snapshot(),
            "target": str(target), "target_exists": target.exists(),
            "target_empty": target.is_dir() and not any(target.iterdir()),
            "disk": {"measured_path": str(existing), "total_bytes": usage.total, "free_bytes": usage.free},
            "gpu": gpu_snapshot(), "network_turbo_script_present": Path("/etc/network_turbo").is_file(),
            "network_started": False, "training_started": False, "environment_modified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--runtime-lock", help="optional read-only ABI floor check before running bundled Python")
    args = parser.parse_args(argv)
    result = inspect(args.target)
    if args.runtime_lock:
        lock = json.loads(safe_path(args.runtime_lock).read_text())
        libc, version = platform.libc_ver()
        result["runtime_floor"] = {"required_glibc": lock["glibc"], "observed_glibc": version,
            "passed": platform.system() == lock["system"] and platform.machine() == lock["machine"]
            and libc == "glibc" and tuple(map(int, version.split("."))) >= tuple(map(int, lock["glibc"].split("."))),
            "imports_or_gpu_certified": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("runtime_floor", {}).get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
