#!/usr/bin/env python3
"""P1-6a lineage specificity of topic 0 and doublet sensitivity.

1. Lineage table over the 76,461 discovery cells (frozen gate labels):
   n, mean topic-0, fraction above the frozen cut, share of total topic-0 mass.
2. Doublet scores: scanpy.pp.scrublet (Scrublet re-implementation) on the raw
   QC'd counts (outputs/scientific_revision_20260922/measurement_controls/
   discovery_raw_counts.npz, row-aligned to cells.parquet; alignment verified),
   run separately per specimen (batch_key=gsm), default parameters
   (expected_doublet_rate=0.05, automatic threshold), random_state=0.
3. Co-expression: within melanocytes and pericyte/SMC, fraction of cells with
   >0 counts of fibroblast genes (COL1A1, COL1A2, DCN, PDGFRA, LUM) and of
   lineage genes, split by topic-0 above/below the cut; doublet score by bin.
4. Readouts after removing predicted doublets: lineage table, 11-patient
   S/M/W contrasts with exact p over 330 allocations, specimen mean-fraction rho.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import (RES, ROOT, FROZEN_CUT, TOL, allocations, dump_json, group_means,  # noqa: E402
                       load_cells, observed_index, register, write_manifest)

MC = ROOT / "outputs/scientific_revision_20260922/measurement_controls"


def lineage_table(c: pd.DataFrame) -> pd.DataFrame:
    tot = c.topic_0.sum()
    g = c.groupby("celltype").agg(n=("topic_0", "size"), mean_topic0=("topic_0", "mean"),
                                  frac_above_cut=("topic_0", lambda x: (x > FROZEN_CUT).mean()),
                                  mass=("topic_0", "sum"))
    g["share_of_topic0_mass"] = g.mass / tot
    g["share_of_above_cut_cells"] = c[c.topic_0 > FROZEN_CUT].celltype.value_counts().reindex(g.index).fillna(0) / (c.topic_0 > FROZEN_CUT).sum()
    return g.drop(columns="mass").sort_values("mean_topic0", ascending=False)


def contrasts(c: pd.DataFrame) -> dict:
    d = c[c.arm.isin(["DFU-healer", "DFU-nonhealer"])]
    pats = sorted(d.patient_id.unique())
    lab = np.array([d.loc[d.patient_id == p, "healing_status"].iloc[0] == "healed" for p in pats])
    A = allocations(11, 7); obs = observed_index(A, lab)
    out = {}
    for r in ["S", "M", "W", "F"]:
        v = []
        for p in pats:
            b = d[d.patient_id == p]; f = b[b.celltype == "fibroblast"]
            v.append({"S": b.topic_0.mean(), "M": f.topic_0.mean(), "W": (f.topic_0 > FROZEN_CUT).mean(),
                      "F": len(f) / len(b)}[r])
        mH, mN = group_means(np.array(v), A); dd = mH - mN
        out[r] = {"diff": float(dd[obs]), "p_exact": float(np.mean(np.abs(dd) >= abs(dd[obs]) - TOL))}
    f = c[c.celltype == "fibroblast"]
    s = f.groupby("gsm").topic_0.agg(["mean", lambda x: (x > FROZEN_CUT).mean()])
    out["specimen_rho_MF"] = float(spearmanr(s.iloc[:, 0], s.iloc[:, 1])[0])
    return out


def main() -> int:
    import scanpy as sc
    import anndata as ad
    cells = load_cells()
    ident = pd.read_csv(register(MC / "raw_cell_identity.csv.gz"))
    assert (ident.raw_cell_id.to_numpy() == cells.reconstructed_raw_cell_id.to_numpy()).all()
    X = sp.load_npz(register(MC / "discovery_raw_counts.npz")).tocsr()
    genes = np.load(register(MC / "discovery_raw_genes.npy"), allow_pickle=True).astype(str)
    assert X.shape[0] == len(cells)

    base = lineage_table(cells)
    base_dfu = lineage_table(cells[cells.arm.isin(["DFU-healer", "DFU-nonhealer"])])

    adata = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame({"gsm": cells.gsm.to_numpy()},
                                                                 index=cells.reconstructed_raw_cell_id.to_numpy()))
    adata.var_names = genes
    sc.pp.scrublet(adata, batch_key="gsm", random_state=0)
    cells["doublet_score"] = adata.obs["doublet_score"].to_numpy()
    cells["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool).to_numpy()
    cells[["reconstructed_raw_cell_id", "gsm", "celltype", "topic_0", "doublet_score", "predicted_doublet"]].to_parquet(
        RES / "p1_6_doublet_scores.parquet", index=False)

    # co-expression
    gi = {g: i for i, g in enumerate(genes)}
    check = ["COL1A1", "COL1A2", "DCN", "PDGFRA", "LUM", "PMEL", "MLANA", "TYRP1", "DCT", "RGS5", "ACTA2",
             "PTPRC", "KRT14", "CHI3L1", "TIMP1", "MMP1"]
    co = []
    for lin in ["melanocyte", "pericyte_smc", "fibroblast", "keratinocyte", "endothelial"]:
        for hi in [False, True]:
            m = ((cells.celltype == lin) & ((cells.topic_0 > FROZEN_CUT) == hi)).to_numpy()
            if m.sum() == 0:
                continue
            sub = X[m]
            r = {"lineage": lin, "topic0_above_cut": hi, "n": int(m.sum()),
                 "mean_doublet_score": float(cells.doublet_score[m].mean()),
                 "frac_predicted_doublet": float(cells.predicted_doublet[m].mean()),
                 "median_library": float(np.median(np.asarray(sub.sum(1)).ravel())),
                 "median_genes": float(np.median(np.asarray((sub > 0).sum(1)).ravel()))}
            for g in check:
                if g in gi:
                    r[f"pct_{g}"] = float((sub[:, gi[g]].toarray().ravel() > 0).mean())
            co.append(r)
    co = pd.DataFrame(co)
    co.to_csv(RES / "p1_6_coexpression_by_lineage_bin.csv", index=False)

    kept = cells[~cells.predicted_doublet]
    after = lineage_table(kept)
    res = {"n_cells": len(cells), "n_predicted_doublets": int(cells.predicted_doublet.sum()),
           "doublet_rate_by_lineage": cells.groupby("celltype").predicted_doublet.mean().to_dict(),
           "doublet_rate_topic0_above_vs_below_cut": {
               "above": float(cells.predicted_doublet[cells.topic_0 > FROZEN_CUT].mean()),
               "below": float(cells.predicted_doublet[cells.topic_0 <= FROZEN_CUT].mean())},
           "spearman_doublet_score_vs_topic0_within_lineage": {
               lin: float(spearmanr(b.doublet_score, b.topic_0)[0]) for lin, b in cells.groupby("celltype")},
           "lineage_before": base.reset_index().to_dict(orient="records"),
           "lineage_before_dfu_only": base_dfu.reset_index().to_dict(orient="records"),
           "lineage_after_doublet_removal": after.reset_index().to_dict(orient="records"),
           "contrasts_before": contrasts(cells), "contrasts_after_doublet_removal": contrasts(kept)}
    # stricter sensitivity: remove top 10% doublet scores within each specimen
    q = cells.groupby("gsm").doublet_score.transform(lambda x: x > x.quantile(0.9))
    res["contrasts_after_removing_top10pct_scores"] = contrasts(cells[~q])
    res["lineage_after_top10pct_removal"] = lineage_table(cells[~q]).reset_index().to_dict(orient="records")
    base.to_csv(RES / "p1_6_lineage_table_before.csv"); after.to_csv(RES / "p1_6_lineage_table_after_doublets.csv")
    dump_json(res, RES / "p1_6_lineage_doublets.json")
    write_manifest("p1_6_lineage_doublets", {"scanpy": sc.__version__})
    pd.set_option("display.width", 250)
    print(base.round(4).to_string()); print(after.round(4).to_string())
    print(co.round(3).to_string())
    import json
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("lineage")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
