"""Build a portable bundle: code, protocols, registry, discovery artifacts, optional
standardized inputs and an optional vendored runtime; every file is hashed.

Outputs written later under runs/, data/ (fetched raw files and standardized data),
logs/, evidence/ and narrowly scoped SDD intake/smoke directories are outside the
sealed manifest by design.  Historical output/ evidence stays sealed.
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from ..common.io import atomic_write_json, atomic_write_text, read_json, sha256_file

CODE_FILES = ["core_engines/data_adapter.py", "core_engines/gse241132.py", "core_engines/topic_simplex_engine.py",
              "core_engines/celltype_markers.py", "docs/superpowers/specs/2026-09-27-power-scaleup-design.md",
              "docs/22_POWER_SCALEUP_HANDOFF.md", "docs/23_EXPANSION_REPAIR_REVIEW_20260927.md",
              "docs/24_EXECUTION_GATE_HANDOFF_20260927.md", "pytest.ini", ".gitignore",
              "docs/33_SDD_LAUNCH_INFRASTRUCTURE_20260930.md",
              "config/next_cohort_preregistration.json", "config/sdd_launch_manifest_20260930.json",
              "config/sdd_launch_manifest_20261003.json",
              "config/sdd_launch_manifest_20261003_r2.json",
              "config/sdd_launch_manifest_20261003_r3.json",
              "config/sdd_launch_manifest_20261003_r4.json",
              "config/sdd_launch_manifest_20261003_r5.json",
              "config/sdd_launch_manifest_20261003_r6.json",
              "config/sdd_launch_manifest_20261006_r1.json",
              "config/sdd_launch_manifest_20261006_r2.json",
              "config/sdd_launch_manifest_20261006_r3.json",
              "config/sdd_launch_manifest_20261006_r4.json",
              "config/sdd_launch_manifest_20261006_r5.json",
              "config/sdd_launch_manifest_20261006_r6.json",
              "config/sdd_launch_manifest_20261006_r7.json",
              "config/sdd_launch_manifest_20261006_r8.json",
              "config/sdd_launch_manifest_20261006_r9.json",
              "config/gpu_scaleup_20261006.json",
              "config/p1_scaleup_protocol_20261006.json",
              "config/p2_primary_scaleup_protocol_20261006.json",
              "config/gpu_cost_probe_20261006.json",
              "config/p2_fair_comparison_20261005.json",
              ".omx/plans/2026-10-06-gpu-scaleup-plan.md",
              "config/sdd_transfer_approval.schema.json", "config/sdd_download_queue_20260929.json",
              "scripts/sdd_launch.py", "scripts/sdd_preflight.py",
              "scripts/prepare_population_gpu_bundle.py",
              "scripts/run_population_gpu.py",
              "output/sdd-local-validation-20260929/FINAL_STATUS.json"]
DISCOVERY_FILES = ["outputs/direction1_dfu_mainline/panel.npy", "outputs/direction1_dfu_mainline/beta.npy",
                   "outputs/direction1_dfu_mainline/topic_model.pt",
                   "outputs/scientific_revision_20260922/measurement_controls/discovery_raw_counts.npz",
                   "outputs/scientific_revision_20260922/measurement_controls/discovery_raw_genes.npy",
                   "outputs/scientific_revision_20260922/measurement/projection/cells.parquet",
                   "outputs/biological_expansion/measurement/specimen_readouts.csv"]
# Sealed inputs live under inputs/; data/ and the SDD intake hold future runtime outputs.
UNSEALED = ("runs/", "data/", "logs/", "evidence/", "output/sdd-data-intake-next/")
TEST_PATTERNS = ("test_expansion_*.py", "test_sdd_launch*.py", "test_sdd_archive*.py")
STANDARDIZED_TARGET = "inputs/standardized"
LAUNCHER = """#!/bin/sh
set -eu
BUNDLE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BUNDLE"
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ -x "$BUNDLE/runtime/bin/python" ]; then
  export LD_LIBRARY_PATH="$BUNDLE/runtime/lib"
  PYTHON="$BUNDLE/runtime/bin/python"
else
  PYTHON="${EXPANSION_PYTHON:-python3}"
fi
exec "$PYTHON" -B -m expansion --root "$BUNDLE" "$@"
"""
SDD_LAUNCHER = """#!/bin/sh
set -eu
BUNDLE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BUNDLE"
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ -x "$BUNDLE/runtime/bin/python" ]; then
  export LD_LIBRARY_PATH="$BUNDLE/runtime/lib"
  PYTHON="$BUNDLE/runtime/bin/python"
else
  PYTHON="${EXPANSION_PYTHON:-python3}"
fi
exec "$PYTHON" -B "$BUNDLE/scripts/sdd_launch.py" "$@"
"""


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _package_files(root: Path) -> list:
    return sorted(str(p.relative_to(root)) for name in ("expansion", "core_engines", "remote/autodl")
                  for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix != ".pyc" and (name != "core_engines" or p.suffix == ".py"))


def _test_files(root: Path) -> list:
    files = {p for pattern in TEST_PATTERNS for p in (root / "tests").glob(pattern) if p.is_file()}
    return sorted(str(p.relative_to(root)) for p in files)


def _runtime_output(relative: str, *, allow_smoke_root: bool = False) -> bool:
    if relative.startswith(UNSEALED):
        return True
    parts = Path(relative).parts
    # Existing files in manifest write/verify must be *below* the smoke root.
    # Only the writable-directory admission check may allow the root itself;
    # otherwise a lookalike root file would be silently left unsealed.
    return (len(parts) >= 3 or (allow_smoke_root and len(parts) == 2)) and parts[0] == "output" \
        and parts[1].startswith("sdd-launch-smoke-") and parts[1] != "sdd-launch-smoke-"


def write_manifest(bundle) -> dict:
    bundle = Path(bundle)
    files = {}
    for path in sorted(bundle.rglob("*")):
        relative = str(path.relative_to(bundle))
        if path.is_symlink():
            raise ValueError(f"symlink not allowed in bundle: {relative}")
        if path.is_file() and relative not in {"bundle_manifest.json", "bundle_manifest.sha256"} \
                and not _runtime_output(relative):
            files[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    manifest = {"schema_version": 1, "files": files}
    atomic_write_json(bundle / "bundle_manifest.json", manifest)
    atomic_write_text(bundle / "bundle_manifest.sha256", sha256_file(bundle / "bundle_manifest.json") + "  bundle_manifest.json\n")
    return manifest


def verify_bundle(bundle) -> list:
    bundle = Path(bundle)
    problems = []
    for parent in (bundle.absolute(), *bundle.absolute().parents):
        if parent.is_symlink():
            return [f"symlink in bundle ancestry: {parent}"]
    for name in ("bundle_manifest.json", "bundle_manifest.sha256"):
        if not (bundle / name).is_file() or (bundle / name).is_symlink():
            return [f"missing or symlinked manifest: {name}"]
    words = (bundle / "bundle_manifest.sha256").read_text().split()
    if len(words) != 2 or len(words[0]) != 64 or words[1] != "bundle_manifest.json":
        return ["invalid bundle_manifest.sha256"]
    expected = words[0]
    if sha256_file(bundle / "bundle_manifest.json") != expected:
        return ["bundle_manifest.json does not match bundle_manifest.sha256"]
    listed = read_json(bundle / "bundle_manifest.json")["files"]
    for relative, record in listed.items():
        from ..launch import LaunchContractError, _safe_relative
        try:
            _safe_relative(relative)
        except LaunchContractError:
            problems.append(f"unsafe manifest path: {relative}")
            continue
        path = bundle / relative
        if any(p.is_symlink() for p in (path, *path.parents)):
            problems.append(f"symlink not allowed: {relative}")
        elif not path.is_file():
            problems.append(f"missing: {relative}")
        elif path.stat().st_size != record.get("size"):
            problems.append(f"modified: {relative}")
        elif sha256_file(path) != record["sha256"]:
            problems.append(f"modified: {relative}")
    for path in bundle.rglob("*"):
        relative = str(path.relative_to(bundle))
        if path.is_symlink():
            problems.append(f"symlink not allowed: {relative}")
            continue
        if path.is_file() and relative not in listed and relative not in {"bundle_manifest.json", "bundle_manifest.sha256"} \
                and not _runtime_output(relative) and "__pycache__" not in relative:
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
    if archive is not None:
        archive = Path(archive).absolute()
        if archive.exists() or archive.with_name(archive.name + ".sha256").exists():
            raise FileExistsError(f"refusing to overwrite archive or checksum {archive}")
        if archive == output or output in archive.parents or not archive.parent.is_dir():
            raise ValueError("archive must have an existing parent outside the bundle")
    if (root / "remote/autodl/handoff.json").is_file():
        from ..launch import CURRENT_MANIFEST
        if not (root / CURRENT_MANIFEST).is_file():
            raise FileNotFoundError("current AutoDL launch manifest must be frozen before building")
    files = sorted(set(_package_files(root) + _test_files(root)
                       + [f for f in CODE_FILES if (root / f).exists()]))
    if include_discovery:
        files += DISCOVERY_FILES
    standardized_files = []
    if include_standardized:
        standardized = Path(include_standardized)
        standardized = (standardized if standardized.is_absolute() else root / standardized).resolve()
        if not standardized.is_dir():
            raise FileNotFoundError(f"standardized directory not found: {standardized}")
        standardized_files = sorted(p for p in standardized.rglob("*") if p.is_file())
    # Fail before creating an incomplete bundle. Pin both the copied bytes and
    # their sources to detect an editor changing the working tree mid-build.
    sources = {relative: root / relative for relative in files}
    sources.update({str(Path(STANDARDIZED_TARGET) / p.relative_to(standardized)): p
                    for p in standardized_files})
    for target in (output, archive) if archive is not None else (output,):
        for ancestor in (target, *target.parents):
            if ancestor.is_symlink():
                raise ValueError(f"symlink in build target ancestry: {ancestor}")
    before = {}
    for relative, source in sources.items():
        if not source.is_file() or any(p.is_symlink() for p in (source, *source.parents)):
            raise ValueError(f"missing or symlinked bundle source: {relative}")
        before[relative] = sha256_file(source)
    output.mkdir(parents=True)
    for relative in files:
        _copy(root / relative, output / relative)
    for path in standardized_files:
        _copy(path, output / STANDARDIZED_TARGET / path.relative_to(standardized))
    for relative, source in sources.items():
        if sha256_file(source) != before[relative] or sha256_file(output / relative) != before[relative]:
            raise ValueError(f"source drift during bundle build: {relative}; retain failed build for inspection")
    (output / "run.sh").write_text(LAUNCHER)
    (output / "run.sh").chmod(0o755)
    (output / "sdd-launch.sh").write_text(SDD_LAUNCHER)
    (output / "sdd-launch.sh").chmod(0o755)
    for path in (output / "remote/autodl").glob("*.sh"):
        path.chmod(0o755)
    lock = _vendored_runtime(output, root) if runtime == "vendored" else None
    manifest = write_manifest(output)
    intake = "output/sdd-data-intake-next/<RUN_ID>"
    run_dir = f"{intake}/runs/<PLAN_ID>"
    standard_dir = STANDARDIZED_TARGET if standardized_files else f"{intake}/standardized"
    summary = {"bundle": str(output), "files": sorted(manifest["files"]), "runtime": runtime,
               "runtime_lock": lock is not None, "immutable_bytes": sum(r["size"] for r in manifest["files"].values()),
               "commands": {"verify": "./run.sh verify-bundle",
                            "preflight": f"./sdd-launch.sh preflight --mode future_training --resource-target {run_dir} --planned-bytes <PEAK_BYTES> --require-cuda --out evidence/sdd-launch-<RUN_ID>/training-preflight.json",
                            "sdd_validate": "./sdd-launch.sh validate",
                            "sdd_offline_smoke": "./sdd-launch.sh smoke --output-root output/sdd-launch-smoke-<RUN_ID>",
                            "sdd_preflight": f"./sdd-launch.sh preflight --mode future_transfer --resource-target {intake} --planned-bytes <PEAK_BYTES> --out evidence/sdd-launch-<RUN_ID>/transfer-preflight.json",
                            "transfer": f"./sdd-launch.sh transfer --transfer-root {intake} --approval TRANSFER_APPROVAL.json --receipt {intake}/transfer_receipt.json --planned-bytes <PEAK_BYTES> --start",
                            "stage_a": f"./sdd-launch.sh stage-a --transfer-root {intake} --approval TRANSFER_APPROVAL.json --qualification STAGE_A_QUALIFICATION.json --receipt {intake}/transfer_receipt.json --standard-dir {intake}/standardized --ids <APPROVED_DATASET_IDS> --planned-bytes <PEAK_BYTES> --start",
                            "plan": ("./sdd-launch.sh plan --protocol expansion/config/population_benchmark.json "
                                     f"--standard-dir {standard_dir} --run-dir {run_dir}"),
                            "run": f"./sdd-launch.sh train --run-dir {run_dir} --approval STAGE_B_APPROVAL.json --planned-bytes <PEAK_BYTES> --max-jobs <MAX_JOBS> --require-cuda --shard 0/1 --start"}}
    if standardized_files and not (output / STANDARDIZED_TARGET / "intake_attestation.json").is_file():
        summary["commands"]["plan"] = ("./remote/autodl/launch.sh plan-existing "
            f"--protocol expansion/config/population_benchmark.json --standard-dir {STANDARDIZED_TARGET} "
            f"--run-dir {run_dir} --ids <REVIEWED_EXISTING_DATASET_IDS>")
        summary["existing_input_route"] = "legacy_contract_requires_current_min50_review_not_sdd_attestation"
    if archive is not None:
        with tarfile.open(archive, "x", format=tarfile.PAX_FORMAT) as stream:
            stream.add(output, arcname=output.name)
        digest = sha256_file(archive)
        atomic_write_text(archive.with_name(archive.name + ".sha256"), f"{digest}  {archive.name}\n")
        summary.update(archive=str(archive), archive_sha256=digest)
    return summary
