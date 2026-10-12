#!/usr/bin/env python3
"""P1-2 step 1: project GSE165816 specimens through the frozen pipeline.

Reuses (read-only import) the RT06 whole-pipeline functions in
scripts/55_full_pipeline_depth_audit.py: raw CSV reader, QC (>=200 genes,
mito<=0.20), frozen 76,461-cell marker reference gate (exclusive max score),
frozen 7,002-gene panel and frozen encoder softmax(mu).  Nothing is refitted.

Projected: the 25 discovery foot specimens (reproduction check against
cells.parquet), the 8 diabetic non-ulcer foot specimens and all 11 forearm
specimens.  Output: results/p1_2_projected_cells.parquet.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from p1_common import ROOT, RES, register, write_manifest, dump_json, FROZEN_CUT  # noqa: E402

sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rt06", ROOT / "scripts/55_full_pipeline_depth_audit.py")
rt06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt06)
from core_engines.topic_simplex_engine import TopicSimplexModel  # noqa: E402


def main() -> int:
    t0 = time.time()
    torch.set_num_threads(16)
    torch.use_deterministic_algorithms(True)
    register(ROOT / "scripts/55_full_pipeline_depth_audit.py")
    register(ROOT / "scripts/50_measurement_control_repair.py")
    register(ROOT / "core_engines/topic_simplex_engine.py")
    register(ROOT / "core_engines/data_adapter.py")
    mp = pd.read_csv(register(ROOT / "metadata_intake/GSE165816_ncomms_subject_sample_map.csv"))
    sel = mp[mp.tissue_site.isin(["foot", "forearm"])].copy()
    sel["role"] = np.where(sel.tissue_site.eq("forearm"), "forearm",
                           np.where(sel.group.eq("DM"), "dm_foot", "discovery_foot"))
    assert (sel.role == "discovery_foot").sum() == 25
    assert (sel.role == "dm_foot").sum() == 8
    assert (sel.role == "forearm").sum() == 11
    wanted = dict(zip(sel.sample_id, sel.role))

    panel = np.load(register(ROOT / "outputs/direction1_dfu_mainline/panel.npy"), allow_pickle=True).astype(str)
    state = torch.load(register(ROOT / "outputs/direction1_dfu_mainline/topic_model.pt"),
                       map_location="cpu", weights_only=True)
    hidden = int(state["encoder.net.0.weight"].shape[0])
    model = TopicSimplexModel(7002, num_topics=15, hidden_dim=hidden)
    model.load_state_dict(state)
    model.eval()
    reference = rt06.load_marker_reference(
        register(ROOT / "outputs/scientific_revision_20260922/measurement_controls/frozen_marker_reference.csv"))
    cut = json.loads(register(ROOT / "outputs/bimodal_stratification/report.json").read_text()
                     )["cohorts"]["GSE165816_discovery"]["gmm"]["midpoint"]
    assert abs(cut - FROZEN_CUT) < 1e-15

    frames, records = [], []
    tar_path = ROOT / "data/raw/GSE165816/GSE165816_RAW.tar"
    with tarfile.open(tar_path, "r") as archive:
        members = []
        for m in sorted(archive.getmembers(), key=lambda m: m.name):
            g = re.search(r"GSM\d+", m.name)
            if m.isfile() and m.name.endswith((".csv", ".csv.gz")) and g and g.group() in wanted:
                members.append((g.group(), m))
        assert {g for g, _ in members} == set(wanted), "missing archive members"
        for gsm, m in members:
            payload = archive.extractfile(m).read()
            counts, genes, barcodes, ncol = rt06.read_count_member(payload, m.name.endswith(".gz"), 256, None)
            frame, coverage = rt06.evaluate_counts(counts, genes, panel, reference, model, "cpu", 512)
            frame.insert(0, "gsm", gsm)
            frame["raw_barcode"] = barcodes
            frame["reconstructed_raw_cell_id"] = gsm + ":" + frame["raw_barcode"]
            row = sel.set_index("sample_id").loc[gsm]
            for col in ("sample_title", "patient_id", "group", "tissue_site", "healing_status", "role"):
                frame[col] = row[col]
            frame = frame[frame.qc_pass].reset_index(drop=True)
            frames.append(frame)
            records.append({"gsm": gsm, "role": row.role, "title": row.sample_title,
                            "member": m.name, "raw_columns": ncol, "qc_cells": len(frame),
                            "panel_coverage": coverage,
                            "fibroblasts": int((frame.celltype == "fibroblast").sum())})
            print(f"[{time.time()-t0:7.1f}s] {gsm} {row.sample_title} {row.role} qc={len(frame)} "
                  f"fib={records[-1]['fibroblasts']} cov={coverage}", flush=True)
    cells = pd.concat(frames, ignore_index=True)
    cells.to_parquet(RES / "p1_2_projected_cells.parquet", index=False)
    pd.DataFrame(records).to_csv(RES / "p1_2_projection_specimens.csv", index=False)

    # reproduction check on the 25 discovery specimens
    ref = pd.read_parquet(register(ROOT / "outputs/scientific_revision_20260922/measurement/projection/cells.parquet"))
    disc = cells[cells.role == "discovery_foot"].set_index("reconstructed_raw_cell_id")
    ref = ref.set_index("reconstructed_raw_cell_id")
    common = ref.index.intersection(disc.index)
    tcols = [f"topic_{k}" for k in range(15)]
    diff = np.abs(disc.loc[common, tcols].to_numpy(float) - ref.loc[common, tcols].to_numpy(float))
    check = {"reference_cells": len(ref), "reprojected_discovery_cells": len(disc),
             "common_cells": len(common),
             "max_abs_topic_difference": float(diff.max()),
             "celltype_agreement": float((disc.loc[common, "celltype"].astype(str).to_numpy()
                                          == ref.loc[common, "celltype"].astype(str).to_numpy()).mean()),
             "fibroblasts_reprojected": int((disc.loc[common, "celltype"] == "fibroblast").sum()),
             "elapsed_s": time.time() - t0}
    dump_json(check, RES / "p1_2_reproduction_check.json")
    print(json.dumps(check, indent=1))
    register(tar_path)
    write_manifest("p1_2_project_excluded", {"reproduction_check": check})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
