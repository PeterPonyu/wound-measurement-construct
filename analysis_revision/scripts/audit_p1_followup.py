#!/usr/bin/env python3
"""Recompute exact topic-search statistics and bind current follow-up files."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/followup_20261010/paper1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    patients = pd.read_csv(SOURCE / "results/p1_1_patient_readouts_all_topics.csv")
    labels = patients.healed.astype(str).str.lower().eq("true").to_numpy()
    if len(patients) != 11 or labels.sum() != 7:
        raise ValueError("Exact patient/allocation contract changed")
    exported = pd.read_csv(SOURCE / "results/p1_7_selection_null_draws.csv")
    original = json.loads((SOURCE / "results/p1_1_selection.json").read_text())
    result = {}
    for readout, key in (("S", "S_all_cell_mean"), ("M", "M_fibroblast_mean"),
                         ("W", "W_fibroblast_fraction_topic_cut")):
        scores = patients[[f"{readout}_topic{i}" for i in range(15)]].to_numpy(float)
        observed = scores[labels].mean(0) / scores[~labels].mean(0)
        maxima = []
        for allocation in itertools.combinations(range(11), 7):
            group = np.zeros(11, dtype=bool)
            group[list(allocation)] = True
            maxima.append(float(np.max(scores[group].mean(0) / scores[~group].mean(0))))
        maxima = np.array(maxima)
        saved = exported[exported.readout == readout].sort_values("allocation")
        if len(saved) != 330 or not np.allclose(maxima, saved.max_ratio, atol=1e-12, rtol=1e-12):
            raise ValueError("Selection-null figure export does not reproduce")
        p_global = float(np.mean(maxima >= observed.max() - 1e-12))
        if abs(p_global - original[key]["max_ratio"]["selection_adjusted_p"]) > 1e-12:
            raise ValueError("Original global maximum test changed")
        result[readout] = dict(topic0_ratio=float(observed[0]), observed_max=float(observed.max()),
                               argmax_topic=int(np.argmax(observed)), allocations=330,
                               topic0_max_calibrated_p=float(np.mean(maxima >= observed[0] - 1e-12)),
                               global_max_statistic_p=p_global)
    files = [p for p in SOURCE.rglob("*") if p.is_file() and
             not any(x in p.parts for x in (".omc", "__pycache__"))]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    report = dict(status="passed", outcome_patients=11, healed=7, nonhealed=4,
                  topic_search=result,
                  selection_sensitivity_scope="patient-mean topic-search statistics, conditional on discovery-trained upstream processing",
                  original_cell_weighted_arm_selection_is_distinct=True,
                  sources=hashes, audit_script_sha256=sha(Path(__file__)))
    (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
