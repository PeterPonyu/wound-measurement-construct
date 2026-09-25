#!/usr/bin/env python3
"""Verify an extracted publication archive and optional analysis smoke checks."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    manifest = json.loads((ROOT / "MANIFEST.sha256.json").read_text())
    errors = []
    for name, digest in manifest.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(name)
    if errors:
        raise RuntimeError(f"Checksum mismatch or missing files: {errors}")
    for path in ROOT.rglob("*.py"):
        if 'provenance' in path.relative_to(ROOT).parts:
            continue
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    import yaml
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    archive = json.loads((ROOT / ".zenodo.json").read_text())
    assert citation["authors"] == [{"family-names": "Fu", "given-names": "Zeyu"}]
    assert archive["creators"] == [{"name": "Fu, Zeyu"}]
    assert citation["title"] == archive["title"]
    assert citation["version"] == archive["version"]
    assert citation.get("license") == "MIT"
    assert str(archive.get("license", "")).lower() == "mit"
    assert (ROOT / "LICENSE").is_file() and (ROOT / "NOTICE").is_file()
    status=json.loads((ROOT/'RELEASE_STATUS.json').read_text())
    assert status['status']=='release' and status['doi']==citation['doi']
    assert status['version']==citation['version'] and citation.get('doi')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    commands = []
    if args.smoke:
        for path in sorted((ROOT / "scripts").glob("*.py")):
            result = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"{path.name}: {result.stderr[-2500:]}")
            commands.append(path.name)
        if (ROOT / "tests").is_dir():
            result = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "tests", "-q"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError("Mathematical verification failed: " + result.stderr[-2500:])
    print(json.dumps({"verified_files": len(manifest), "python_syntax": "passed", "citation_metadata": "consistent", "cli_smoke_checks": len(commands), "numerical_checks": "not part of software-only archive"}, indent=2))


if __name__ == "__main__":
    main()
