"""Launch-kit contracts for future data transfer and experiment execution.

The launch kit is intentionally conservative:

* ``validate`` and ``preflight`` are read-only;
* ``smoke`` writes only a fresh synthetic output root and performs no network I/O;
* transfer and training commands require an explicit ``--start`` in addition to
  their independent approval receipts.

This module does not grant scientific approval.  It verifies that the future
operator has supplied the separate approval and that the current code/data
closure has not drifted since the kit was prepared.
"""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from remote.autodl.host import LOCAL_FREE_MARGIN_BYTES
from .common.io import atomic_write_json, read_json, sha256_file
from .common.provenance import code_digest, code_files


class LaunchContractError(ValueError):
    """The launch manifest or a preflight gate is not satisfied."""


REQUIRED_STAGES = ("metadata_qualification", "payload_transfer", "stage_a", "training")
REQUIRED_DISTRIBUTIONS = ("numpy", "scipy", "pandas", "torch", "anndata", "h5py",
                          "pot", "scikit-learn", "statsmodels", "pyarrow", "packaging")
CURRENT_MANIFEST = "config/sdd_launch_manifest_20261006_r9.json"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _strict_json(path: Path) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise LaunchContractError(f"duplicate JSON key {key!r}: {path}")
            out[key] = value
        return out

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchContractError(f"invalid launch manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LaunchContractError("launch manifest must be a JSON object")
    return payload


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    manifest = _strict_json(path)
    required = {"schema_version", "kind", "manifest_id", "status", "auto_start", "stages",
                "source_files", "sealed_roots", "resource_policy", "commands"}
    missing = required - set(manifest)
    if missing:
        raise LaunchContractError(f"launch manifest lacks {sorted(missing)}")
    if manifest["schema_version"] != 1 or manifest["kind"] != "sdd_launch_manifest":
        raise LaunchContractError("unsupported launch manifest schema")
    if manifest["status"] != "prepared_not_started" or manifest["auto_start"] is not False:
        raise LaunchContractError("launch manifest must remain prepared_not_started with auto_start=false")
    if tuple(manifest["stages"]) != REQUIRED_STAGES:
        raise LaunchContractError("launch stages must be metadata -> transfer -> stage_a -> training")
    for stage in REQUIRED_STAGES:
        if stage not in manifest or not isinstance(manifest[stage], dict):
            raise LaunchContractError(f"missing stage contract: {stage}")
    if not isinstance(manifest["source_files"], list) or not manifest["source_files"]:
        raise LaunchContractError("source_files must be a non-empty list")
    if not isinstance(manifest["sealed_roots"], list) or not manifest["sealed_roots"]:
        raise LaunchContractError("sealed_roots must be a non-empty list")
    if not isinstance(manifest.get("source_closure_sha256"), str) or len(manifest["source_closure_sha256"]) != 64:
        raise LaunchContractError("launch manifest must pin the complete source closure SHA-256")
    return manifest


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value or ":" in value:
        raise LaunchContractError(f"unsafe relative path in launch manifest: {value!r}")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise LaunchContractError(f"unsafe relative path in launch manifest: {value!r}")
    return value


def launch_source_files(root: str | Path) -> list[Path]:
    """Complete executable/config/test closure, stable after relocation."""
    root = Path(root)
    files = code_files(root)
    files += [root / name for name in ("scripts/sdd_launch.py", "scripts/sdd_preflight.py",
              "scripts/prepare_population_gpu_bundle.py",
              "scripts/run_population_gpu.py",
              "config/sdd_transfer_approval.schema.json", "config/sdd_download_queue_20260929.json")]
    files += [root / name for name in ("config/gpu_scaleup_20261006.json",
              "config/p1_scaleup_protocol_20261006.json",
              "config/p2_primary_scaleup_protocol_20261006.json",
              "config/gpu_cost_probe_20261006.json",
              "remote/autodl/materialize_prepared.py")]
    files += [root / ".gitignore", root / "pytest.ini"]
    files += [p for p in (root / "remote" / "autodl").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    files += list((root / "tests").glob("test_expansion_*.py"))
    files += list((root / "tests").glob("test_sdd_launch*.py"))
    files += list((root / "tests").glob("test_sdd_archive*.py"))
    return sorted(set(files))


def launch_source_digest(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in launch_source_files(root):
        if not path.is_file() or path.is_symlink():
            raise LaunchContractError(f"missing or symlinked launch dependency: {path}")
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def assert_writable_target(manifest: dict, root: str | Path, target: str | Path) -> Path:
    """Read-only path check; protects seals and rejects symlink ancestry."""
    target = Path(os.path.abspath(target))
    root = Path(root).resolve()
    for parent in (target, *target.parents):
        if parent.is_symlink():
            raise LaunchContractError(f"symlink in output ancestry: {parent}")
        if (parent / "FINAL_STATUS.json").exists():
            raise LaunchContractError(f"historical sealed output is immutable: {parent}")
    for record in manifest.get("sealed_roots", []):
        sealed = (root / record["path"]).resolve()
        if target == sealed or sealed in target.parents or target in sealed.parents:
            raise LaunchContractError(f"output overlaps sealed historical root: {target}")
    for name in ("expansion", "core_engines", "config", "scripts", "tests", "manuscripts", "publication",
                 "inputs", "runtime", "docs", "remote"):
        protected = root / name
        if target == protected or protected in target.parents:
            raise LaunchContractError(f"output overlaps authored source: {target}")
    if (root / "bundle_manifest.json").is_file():
        # A sealed portable product has a finite write surface, not an open
        # choice of new directories beside immutable sources and inputs.
        from .bundle.build import _runtime_output
        try:
            relative = str(target.relative_to(root))
        except ValueError as error:
            raise LaunchContractError("bundled runtime output must stay inside its isolated bundle") from error
        if not _runtime_output(relative, allow_smoke_root=True):
            raise LaunchContractError(f"output is outside bundled runtime allowlist: {target}")
    return target


def validate_manifest(manifest: dict, root: str | Path) -> dict:
    root = Path(root).resolve()
    for record in manifest["source_files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise LaunchContractError("each source_files record must contain path and sha256")
        relative = _safe_relative(record["path"])
        actual = root / relative
        if not actual.is_file() or actual.is_symlink():
            raise LaunchContractError(f"missing or symlinked source file: {relative}")
        if sha256_file(actual) != record["sha256"]:
            raise LaunchContractError(f"source hash drift: {relative}")
    if launch_source_digest(root) != manifest.get("source_closure_sha256"):
        raise LaunchContractError("complete launch source closure drift; prepare a new kit before execution")
    for record in manifest["sealed_roots"]:
        if not isinstance(record, dict) or set(record) != {"path", "sentinel", "sha256"}:
            raise LaunchContractError("sealed_roots records must contain path, sentinel and sha256")
        base = root / _safe_relative(record["path"])
        sentinel = base / _safe_relative(record["sentinel"])
        if not sentinel.is_file() or sha256_file(sentinel) != record["sha256"]:
            raise LaunchContractError(f"sealed sentinel missing or changed: {sentinel}")
    return {"status": "passed", "source_files": len(manifest["source_files"]),
            "sealed_roots": len(manifest["sealed_roots"]), "code_digest": code_digest(root)}


def _memory_available() -> int | None:
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
        available = values.get("MemAvailable")
        limit_path, current_path = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
        if limit_path.is_file() and current_path.is_file():
            limit, current = limit_path.read_text().strip(), current_path.read_text().strip()
            if limit.isdigit() and current.isdigit():
                remaining = max(0, int(limit) - int(current))
                available = min(available, remaining) if available is not None else remaining
        return available
    except (OSError, ValueError):
        return None


def _package_versions(names=REQUIRED_DISTRIBUTIONS) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resource_snapshot(path: str | Path) -> dict:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    usage = shutil.disk_usage(path)
    available = _memory_available()
    return {"path": str(Path(path).resolve()), "disk_total_bytes": usage.total,
            "disk_used_bytes": usage.used, "disk_free_bytes": usage.free,
            "disk_free_fraction": usage.free / usage.total if usage.total else 0.0,
            "memory_available_bytes": available, "python": platform.python_version(),
            "platform": platform.platform(), "utc": utc_now()}


def _fresh_output(path: str | Path, *, create: bool = False) -> dict:
    path = Path(path)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise LaunchContractError(f"output root is not a real directory: {path}")
        if any(path.iterdir()):
            raise LaunchContractError(f"output root must be fresh and empty: {path}")
    elif create:
        path.mkdir(parents=True)
    return {"path": str(path.resolve()), "fresh": True}


def preflight(manifest: dict, *, root: str | Path, mode: str, output_root: str | Path | None = None,
              require_cuda: bool = False, resource_target: str | Path | None = None,
              planned_bytes: int | None = None) -> dict:
    """Run a read-only preflight; no network or training is started."""
    root = Path(root).resolve()
    validation = validate_manifest(manifest, root)
    if (root / "bundle_manifest.json").is_file():
        from .bundle.build import verify_bundle
        problems = verify_bundle(root)
        if problems:
            raise LaunchContractError(f"bundle integrity failed: {problems[:5]}")
    policy = manifest["resource_policy"]
    if mode not in {"offline_smoke", "future_transfer", "future_training"}:
        raise LaunchContractError(f"unknown preflight mode: {mode}")
    if planned_bytes is not None and (type(planned_bytes) is not int or planned_bytes < 0):
        raise LaunchContractError("planned_bytes must be a non-negative integer")
    target = resource_target or output_root or root
    if resource_target is not None or output_root is not None:
        assert_writable_target(manifest, root, target)
    resources = resource_snapshot(target)
    remaining = resources["disk_free_bytes"] - (planned_bytes or 0)
    min_fraction = float(policy["smoke_min_free_fraction"] if mode == "offline_smoke"
                         else policy["required_free_fraction"])
    min_memory = int(policy["smoke_min_memory_bytes"] if mode == "offline_smoke"
                     else policy["required_memory_bytes"])
    # future_training is a local run. Admit it from the bytes it will write
    # plus a fixed free margin. A new matrix transfer still uses the volume fraction.
    if mode == "future_training":
        disk_ok = remaining >= LOCAL_FREE_MARGIN_BYTES
    else:
        disk_ok = remaining >= resources["disk_total_bytes"] * min_fraction
    packages = _package_versions()
    checks = {
        "source_and_sealed_hashes": True,
        "packages_present": all(packages.values()),
        "disk_gate": disk_ok,
        "budget_known": mode == "offline_smoke" or (planned_bytes is not None and planned_bytes > 0),
        "memory_gate": resources["memory_available_bytes"] is not None
        and resources["memory_available_bytes"] >= min_memory,
        "network": "disabled" if mode == "offline_smoke" else "approval_only",
        "cuda": "not_required",
    }
    if require_cuda:
        try:
            torch = importlib.import_module("torch")
            available = bool(torch.cuda.is_available())
            checks["cuda"] = available
            cuda = {"available": available, "torch_cuda": torch.version.cuda,
                    "count": torch.cuda.device_count() if available else 0}
        except Exception as error:
            checks["cuda"] = False
            cuda = {"available": False, "error": f"{type(error).__name__}: {error}"}
    else:
        cuda = {"required": False}
    output = None
    if output_root is not None:
        assert_writable_target(manifest, root, output_root)
        if resource_target is not None and Path(resource_target).resolve() != Path(output_root).resolve():
            raise LaunchContractError("fresh output and resource target must identify the same write volume path")
        output = _fresh_output(output_root)
    passed = all(value is True for key, value in checks.items() if key not in {"network", "cuda"}) \
        and checks["cuda"] is not False
    return {"status": "passed" if passed else "blocked", "mode": mode, "checks": checks,
            "packages": packages, "resources": resources, "cuda": cuda, "output_root": output,
            "required_free_fraction": None if mode == "future_training" else min_fraction,
            "required_reserve_bytes": LOCAL_FREE_MARGIN_BYTES if mode == "future_training" else None,
            "required_memory_bytes": min_memory,
            "planned_increment_bytes": planned_bytes, "post_write_free_bytes": remaining,
            "validation": validation, "real_data_downloaded": False, "training_started": False}


def smoke(manifest_path: str | Path, *, root: str | Path, output_root: str | Path) -> dict:
    manifest = load_manifest(manifest_path)
    report = preflight(manifest, root=root, mode="offline_smoke", output_root=output_root)
    if report["status"] != "passed":
        return report
    # Import the local runner only after the read-only gates have passed.  The
    # synthetic smoke implementation does not call the network or read any
    # candidate payload.
    from .runner.cli import run_smoke

    output_root = Path(output_root)
    _fresh_output(output_root, create=True)
    evidence = run_smoke(output_root / "synthetic", device="cpu", quick=True)
    report["smoke"] = evidence
    report["status"] = "passed" if evidence.get("status") == "passed" else "blocked"
    report["real_data_downloaded"] = False
    report["training_started"] = False
    atomic_write_json(output_root / "launch_preflight.json", report)
    return report


def future_commands(manifest: dict) -> dict:
    return dict(manifest["commands"])
