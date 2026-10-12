#!/usr/bin/env python3
"""Aggregate the per-analysis input SHA256 manifests and hash every script and
result file in outputs/followup_20261010/paper1/ into MANIFEST_SHA256.json."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import OUT, RES, ROOT, sha256

inputs = {}
for f in sorted(RES.glob("*_input_manifest.json")):
    d = json.loads(f.read_text())
    for k, v in d["inputs_sha256"].items():
        if k.startswith("outputs/followup_20261010/"):
            continue
        assert inputs.get(k, v) == v, k
        inputs[k] = v
CITED = ["scripts/10_run_dfu_mainline.py", "scripts/11_validate_gse231643.py", "docs/05_DATASET_AUDIT.md",
         "logs/direction1_mainline.log", "outputs/direction1_dfu_mainline/report.json",
         "core_engines/celltype_markers.py", "config/pipeline_config.yaml", "config/datasets_matrix.json",
         "scripts/43_paired_anatomical_control.py", "scripts/13_bimodal_stratification.py"]
cited = {c: sha256(ROOT / c) for c in CITED}
files = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(OUT.rglob("*"))
         if p.is_file() and p.name != "MANIFEST_SHA256.json" and "__pycache__" not in p.parts}
(OUT / "MANIFEST_SHA256.json").write_text(json.dumps(
    {"repository_inputs_read_only": inputs, "provenance_documents_cited": cited,
     "followup_files": files}, indent=2))
print(len(inputs), "inputs;", len(files), "follow-up files")
