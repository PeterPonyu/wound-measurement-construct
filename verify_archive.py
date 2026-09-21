#!/usr/bin/env python3
"""Verify an extracted publication archive and optional analysis smoke checks."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
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
    commands = []
    if args.smoke:
        for path in sorted((ROOT / "scripts").glob("*.py")):
            result = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"{path.name}: {result.stderr[-2500:]}")
            commands.append(path.name)
        import numpy as np
        import torch
        from wound_models.human_wound_data import fold_standardize, load_discovery
        from scipy.spatial.distance import cdist
        panel, beta, model, _, _, device = load_discovery(str(ROOT / "outputs/expression_representation"))
        assert beta.shape == (15, len(panel))
        assert np.allclose(beta.sum(1), 1, atol=1e-5)
        batch = torch.ones((3, len(panel)), device=device) / len(panel)
        with torch.no_grad():
            first = model(batch)["theta"]
            second = model(batch)["theta"]
        assert torch.equal(first, second)
        assert torch.allclose(first.sum(1), torch.ones(3, device=device))
        x = np.array([[0., 2.], [2., 4.], [1000., -1000.]])
        scaled, _ = fold_standardize(x, np.array([True, True, False]))
        assert np.allclose(scaled[:2].mean(0), 0, atol=1e-6)
        assert np.abs(scaled[2]).max() > 100
    print(json.dumps({"verified_files": len(manifest), "python_syntax": "passed", "citation_metadata": "consistent", "cli_smoke_checks": len(commands), "numerical_checks": "passed" if args.smoke else "not requested"}, indent=2))


if __name__ == "__main__":
    main()
