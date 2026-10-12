"""Build a portable bundle: code, protocols, registry, discovery artifacts, optional
standardized inputs and an optional vendored runtime; every file is hashed.

Outputs written later under runs/, data/ (fetched raw files and standardized data)
and logs/ are outside the sealed manifest by design.
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from ..common.io import atomic_write_json, atomic_write_text, read_json, sha256_file

CODE_FILES = ["core_engines/data_adapter.py", "core_engines/gse241132.py", "core_engines/topic_simplex_engine.py",
              "core_engines/celltype_markers.py", "docs/superpowers/specs/2026-09-27-power-scaleup-design.md",
              "docs/22_POWER_SCALEUP_HANDOFF.md", "docs/23_EXPANSION_REPAIR_REVIEW_20260927.md",
              "docs/24_EXECUTION_GATE_HANDOFF_20260927.md", "pytest.ini",
              "config/next_cohort_preregistration.json"]
DISCOVERY_FILES = ["outputs/direction1_dfu_mainline/panel.npy", "outputs/direction1_dfu_mainline/beta.npy",
                   "outputs/direction1_dfu_mainline/topic_model.pt",
                   "outputs/scientific_revision_20260922/measurement_controls/discovery_raw_counts.npz",
                   "outputs/scientific_revision_20260922/measurement_controls/discovery_raw_genes.npy",
                   "outputs/scientific_revision_20260922/measurement/projection/cells.parquet",
                   "outputs/biological_expansion/measurement/specimen_readouts.csv"]
# Sealed inputs live under inputs/; data/ is where a remote Stage A writes fetched and standardized data.
UNSEALED = ("runs/", "data/", "logs/", "evidence/")
STANDARDIZED_TARGET = "inputs/standardized"
LAUNCHER = """#!/bin/sh
set -eu
BUNDLE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BUNDLE"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
if [ -x "$BUNDLE/runtime/bin/python" ]; then
  export LD_LIBRARY_PATH="$BUNDLE/runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  PYTHON="$BUNDLE/runtime/bin/python"
else
  PYTHON="${EXPANSION_PYTHON:-python3}"
fi
exec "$PYTHON" -B -m expansion --root "$BUNDLE" "$@"
"""


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _package_files(root: Path) -> list:
    return sorted(str(p.relative_to(root)) for p in (root / "expansion").rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def _test_files(root: Path) -> list:
    return sorted(str(p.relative_to(root)) for p in (root / "tests").glob("test_expansion_*.py"))


def write_manifest(bundle) -> dict:
    bundle = Path(bundle)
    files = {}
    for path in sorted(bundle.rglob("*")):
        relative = str(path.relative_to(bundle))
        if path.is_file() and relative not in {"bundle_manifest.json", "bundle_manifest.sha256"} \
                and not relative.startswith(UNSEALED):
            files[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    manifest = {"schema_version": 1, "files": files}
    atomic_write_json(bundle / "bundle_manifest.json", manifest)
    atomic_write_text(bundle / "bundle_manifest.sha256", sha256_file(bundle / "bundle_manifest.json") + "  bundle_manifest.json\n")
    return manifest


def verify_bundle(bundle) -> list:
    bundle = Path(bundle)
    problems = []
    expected = (bundle / "bundle_manifest.sha256").read_text().split()[0]
    if sha256_file(bundle / "bundle_manifest.json") != expected:
        return ["bundle_manifest.json does not match bundle_manifest.sha256"]
    listed = read_json(bundle / "bundle_manifest.json")["files"]
    for relative, record in listed.items():
        path = bundle / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
        elif path.is_symlink():
            problems.append(f"symlink not allowed: {relative}")
        elif sha256_file(path) != record["sha256"]:
            problems.append(f"modified: {relative}")
    for path in bundle.rglob("*"):
        relative = str(path.relative_to(bundle))
        if path.is_file() and relative not in listed and relative not in {"bundle_manifest.json", "bundle_manifest.sha256"} \
                and not relative.startswith(UNSEALED) and "__pycache__" not in relative:
            problems.append(f"unlisted: {relative}")
    return problems


def _vendored_runtime(bundle: Path, root: Path) -> dict:
    from .runtime import vendor_runtime

    return vendor_runtime(bundle, root)


def build_bundle(output, *, root, runtime: str = "none", include_discovery: bool = True, include_standardized=None,
                 archive=None) -> dict:
    output, root = Path(output).absolute(), Path(root).absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle {output}")
    if runtime not in {"none", "vendored"}:
        raise ValueError("runtime must be 'none' or 'vendored'")
    files = _package_files(root) + _test_files(root) + [f for f in CODE_FILES if (root / f).exists()]
    if include_discovery:
        files += DISCOVERY_FILES
    standardized_files = []
    if include_standardized:
        standardized = Path(include_standardized)
        standardized = (standardized if standardized.is_absolute() else root / standardized).resolve()
        if not standardized.is_dir():
            raise FileNotFoundError(f"standardized directory not found: {standardized}")
        standardized_files = sorted(p for p in standardized.rglob("*") if p.is_file())
    output.mkdir(parents=True)
    for relative in files:
        _copy(root / relative, output / relative)
    for path in standardized_files:
        _copy(path, output / STANDARDIZED_TARGET / path.relative_to(standardized))
    (output / "run.sh").write_text(LAUNCHER)
    (output / "run.sh").chmod(0o755)
    lock = _vendored_runtime(output, root) if runtime == "vendored" else None
    manifest = write_manifest(output)
    summary = {"bundle": str(output), "files": sorted(manifest["files"]), "runtime": runtime,
               "runtime_lock": lock is not None, "immutable_bytes": sum(r["size"] for r in manifest["files"].values()),
               "commands": {"verify": "./run.sh verify-bundle", "preflight": "./run.sh preflight --require-cuda",
                            "stage_a": "./run.sh stage-a --execute-fetch --approval APPROVAL.json",
                            "plan": ("./run.sh plan --protocol expansion/config/population_benchmark.json "
                                     f"--standard-dir {STANDARDIZED_TARGET if standardized_files else 'data/standardized'} "
                                     "--run-dir runs/population-v1"),
                            "run": "./run.sh run --run-dir runs/population-v1 --approval STAGE_B_APPROVAL.json --require-cuda --shard 0/1"}}
    if archive is not None:
        archive = Path(archive).absolute()
        if archive.exists():
            raise FileExistsError(f"refusing to overwrite archive {archive}")
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as stream:
            stream.add(output, arcname=output.name)
        digest = sha256_file(archive)
        atomic_write_text(archive.with_name(archive.name + ".sha256"), f"{digest}  {archive.name}\n")
        summary.update(archive=str(archive), archive_sha256=digest)
    return summary
