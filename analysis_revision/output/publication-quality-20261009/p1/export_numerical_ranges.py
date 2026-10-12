#!/usr/bin/env python3
"""Mechanical range export from already-verified CSVs; no training or inference."""
from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from expansion.common.io import atomic_write_json, read_json, sha256_file
from scripts import run_quality_topic_stability as sidecar


def summary(values):
    values = list(values)
    valid = [float(value) for value in values if pd.notna(value)]
    assert all(math.isfinite(value) for value in valid)
    return {"min": min(valid) if valid else None, "max": max(valid) if valid else None,
            "available": len(valid), "missing": len(values) - len(valid)}


def main():
    sidecar.validate_frozen()
    report = read_json(sidecar.OUTPUT / "verification.json")
    tables, numerical, inventory = {}, {}, {}
    for name, record in report["tables"].items():
        path = sidecar.OUTPUT / "tables" / f"{name}.csv"
        assert sha256_file(path) == record["sha256"]
        frame = pd.read_csv(path)
        assert len(frame) == record["rows"]
        tables[name] = frame
        inventory[name] = record
        numerical[name] = {column: summary(frame[column]) for column in frame.select_dtypes(include="number")
                           if column != "reason"}
    grouped_readouts, grouped_widths, comparisons = [], [], []
    for (k, scope), frame in tables["patient_readouts"].groupby(["K", "readout_scope"]):
        grouped_readouts.append({"K": int(k), "readout_scope": scope,
            **{metric: summary(frame[metric]) for metric in
               ("mean_loading", "fraction_above_fibroblast_descriptive_cut")}})
    for (k, scope, metric), frame in tables["seed_ranges"].groupby(["K", "readout_scope", "metric"]):
        grouped_widths.append({"K": int(k), "readout_scope": scope, "metric": metric,
            **summary(frame.range_over_available_seeds), "median_range_width_over_20_people":
                float(frame.range_over_available_seeds.median())})
    for comparison, frame in tables["semantics"].groupby("comparison"):
        individual = tables["patient_semantics"].query("comparison == @comparison")
        comparisons.append({"comparison": comparison,
            "specimen_median_difference_nats_per_cell": summary(frame.median_difference_nats_per_cell),
            "people_favour_first_out_of_20": summary(frame.units_favour_first),
            "individual_difference_nats_per_cell": summary(individual.difference_nats_per_cell)})
    result = {"source_verification_sha256": sha256_file(sidecar.OUTPUT / "verification.json"),
        "source_tables": inventory, "all_numeric_column_ranges": numerical,
        "per_fit_primary_values": tables["fits"][["K", "seed", "primary_correlation",
            "cut_full_sample_fibroblasts", "specimen_mean_fraction_spearman"]].to_dict(orient="records"),
        "readout_value_ranges_by_K_and_scope": grouped_readouts,
        "computational_seed_widths_by_K_scope_metric": grouped_widths,
        "density_comparison_ranges": comparisons,
        "interpretation": "mechanical_descriptive_range_export_no_new_tests_selection_or_biological_units"}
    atomic_write_json(sidecar.OUTPUT / "numerical_ranges.json", result)
    print(json.dumps({"path": str(sidecar.OUTPUT / "numerical_ranges.json"),
                      "sha256": sha256_file(sidecar.OUTPUT / "numerical_ranges.json"),
                      "verification_sha256": result["source_verification_sha256"],
                      "table_hashes_and_frozen_source_input_manifest": "passed"}, indent=2))


if __name__ == "__main__":
    main()
