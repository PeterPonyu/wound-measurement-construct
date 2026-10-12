#!/usr/bin/env python3
"""P1-6c: all-cell readout S recomputed excluding melanocytes (and, separately,
excluding melanocytes + pericyte/SMC); 11 DFU patients, exact p over 330."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import RES, TOL, allocations, dump_json, group_means, load_cells, observed_index, write_manifest

cells = load_cells()
A = allocations(11, 7)
out = {}
for name, drop in [("all_cells", []), ("no_melanocyte", ["melanocyte"]),
                   ("no_melanocyte_no_pericyte", ["melanocyte", "pericyte_smc"]),
                   ("fibroblast_only", None)]:
    c = cells[cells.arm.isin(["DFU-healer", "DFU-nonhealer"])]
    c = c[c.celltype == "fibroblast"] if drop is None else c[~c.celltype.isin(drop)]
    pats = sorted(c.patient_id.unique())
    lab = np.array([c.loc[c.patient_id == p, "healing_status"].iloc[0] == "healed" for p in pats])
    v = np.array([c.loc[c.patient_id == p, "topic_0"].mean() for p in pats])
    mH, mN = group_means(v, A); d = mH - mN; o = observed_index(A, lab)
    out[name] = {"healed_mean": float(mH[o]), "nonhealed_mean": float(mN[o]), "diff": float(d[o]),
                 "p_exact": float(np.mean(np.abs(d) >= abs(d[o]) - TOL))}
dump_json(out, RES / "p1_6c_allcell_without_melanocytes.json")
write_manifest("p1_6c_allcell_without_melanocytes")
print(out)
