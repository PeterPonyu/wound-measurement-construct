#!/usr/bin/env python3
"""P1-6b bulk GSE80178 topic-0 score: DFU vs intact foot skin, random-signature control.

Data limitation (stated, not worked around): the per-sample series matrix has
53,617 HuGene-2.0-ST transcript-cluster IDs without gene symbols; no GPL16686
annotation is available offline.  Gene symbols exist only in
GSE80178_further_processed_data.txt.gz, which the submitters filtered to
DFU-vs-foot-skin DE clusters (all rows FDR < 0.05 and |FC| >= 2; checked).
The analysis therefore runs inside that DE-filtered universe; any DFU-vs-intact
contrast in it is selection-biased, and the random-signature control (drawn from
the same universe) only asks whether topic-0 genes are more DFU-up than
arbitrary DE genes of the same set size.  DM-foot vs non-DM-foot (both intact,
pooled in the submitters' contrast) is less affected but has n = 3 vs 3.

Score: per gene, z-score the log2 signal across the 12 samples (collapsing
duplicate clusters by mean); (a) TOP50 = mean z over the 50 overlapping genes
with the largest frozen decoder probability beta_0g; (b) LIFT50 = same for the
50 genes with largest beta_0g / mean_k beta_kg among genes with beta_0g above
the 90th percentile; (c) BETA = beta_0g-weighted mean z over all overlap genes.
Exact permutation (two-sided, all relabellings) and 10,000 size-matched random
signatures (seed 20261010) from the annotated universe.
"""
from __future__ import annotations

import gzip
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import RES, ROOT, TOL, dump_json, register, write_manifest  # noqa: E402

D = ROOT / "data/raw/GSE80178"


def series(path):
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    titles = [l for l in lines if l.startswith("!Sample_title")][0].split("\t")[1:]
    titles = [t.strip('"') for t in titles]
    i0 = lines.index("!series_matrix_table_begin")
    i1 = lines.index("!series_matrix_table_end")
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[i0 + 1:i1])), sep="\t", index_col=0)
    return df, titles


def exact_perm(score, g1, g2):
    idx = np.flatnonzero(g1 | g2); s = score[idx]; n1 = int(g1[idx].sum())
    obs = s[g1[idx]].mean() - s[g2[idx]].mean()
    null = np.array([s[list(c)].mean() - np.delete(s, list(c)).mean()
                     for c in itertools.combinations(range(len(s)), n1)])
    return float(obs), float(np.mean(np.abs(null) >= abs(obs) - TOL)), len(null)


def main() -> int:
    expr, titles = series(register(D / "GSE80178_series_matrix.txt.gz"))
    proc = pd.read_csv(register(D / "GSE80178_further_processed_data.txt.gz"), sep="\t")
    filt = {"n_rows": len(proc), "all_fdr_below_0.05": bool((proc["FDR p-value (DFU vs. FS)"] < 0.05).all()),
            "all_absFC_ge_2": bool((proc["Fold Change (linear) (DFU vs. FS)"].abs() >= 2).all()),
            "frac_up_in_DFU": float((proc["Fold Change (linear) (DFU vs. FS)"] > 0).mean())}
    proc = proc.dropna(subset=["Gene Symbol"])
    proc["sym"] = proc["Gene Symbol"].astype(str).str.split(" /// ").str[0].str.strip()
    sub = expr.loc[expr.index.intersection(proc["Transcript Cluster ID"])]
    sym = proc.set_index("Transcript Cluster ID").loc[sub.index, "sym"]
    G = sub.groupby(sym.values).mean()
    Z = G.sub(G.mean(1), axis=0).div(G.std(1, ddof=1).replace(0, np.nan), axis=0).dropna()
    grp = np.array(["DFU" if "Ulcer" in t else ("NDM" if t.startswith("non-") else "DM") for t in titles])
    dfu, dm, ndm = grp == "DFU", grp == "DM", grp == "NDM"

    panel = np.load(register(ROOT / "outputs/direction1_dfu_mainline/panel.npy"), allow_pickle=True).astype(str)
    beta = np.load(register(ROOT / "outputs/direction1_dfu_mainline/beta.npy"))
    b0 = pd.Series(beta[0], index=panel)
    lift = pd.Series(beta[0] / beta.mean(0), index=panel)
    ov = Z.index.intersection(panel)
    top50 = b0[ov].sort_values(ascending=False).index[:50]
    hi = b0[b0 > b0.quantile(0.9)].index.intersection(ov)
    lift50 = lift[hi].sort_values(ascending=False).index[:50]
    MEL = {"PMEL", "MLANA", "DCT", "TYRP1", "TYR"}
    lift50_nomel = [g for g in lift[hi].sort_values(ascending=False).index if g not in MEL][:50]
    sigs = {"TOP50": list(top50), "LIFT50": list(lift50), "LIFT50_no_melanocyte_genes": lift50_nomel}
    rng = np.random.default_rng(20261010)
    out = {"universe": filt, "n_annotated_genes": int(len(Z)), "n_overlap_with_panel": int(len(ov)),
           "groups": dict(zip(titles, grp)), "signatures": {}}
    universe = np.array(Z.index)
    for name, genes in sigs.items():
        sc = Z.loc[genes].mean(0).to_numpy()
        r = {"genes": genes, "n_genes": len(genes), "scores": dict(zip(titles, sc.round(4)))}
        for lab, g1, g2 in [("DFU_vs_intact", dfu, dm | ndm), ("DFU_vs_DMfoot", dfu, dm), ("DMfoot_vs_NDMfoot", dm, ndm)]:
            d, p, n = exact_perm(sc, g1, g2)
            rnd = np.array([Z.loc[rng.choice(universe, len(genes), replace=False)].mean(0).to_numpy()
                            for _ in range(10000)])
            rd = rnd[:, g1].mean(1) - rnd[:, g2].mean(1)
            r[lab] = {"diff": d, "p_exact_perm": p, "n_relabellings": n,
                      "random_signature_frac_ge_observed": float(np.mean(rd >= d - TOL)),
                      "random_signature_median": float(np.median(rd))}
        out["signatures"][name] = r
    w = b0[ov] / b0[ov].sum()
    sc = (Z.loc[ov].mul(w, axis=0)).sum(0).to_numpy()
    r = {"n_genes": int(len(ov)), "scores": dict(zip(titles, sc.round(4)))}
    for lab, g1, g2 in [("DFU_vs_intact", dfu, dm | ndm), ("DFU_vs_DMfoot", dfu, dm), ("DMfoot_vs_NDMfoot", dm, ndm)]:
        d, p, n = exact_perm(sc, g1, g2)
        r[lab] = {"diff": d, "p_exact_perm": p, "n_relabellings": n}
    out["signatures"]["BETA_weighted_all_overlap"] = r
    # topic-specificity control: the same LIFT50 construction for every topic
    tk = []
    for k in range(beta.shape[0]):
        bk = pd.Series(beta[k], index=panel); lk = pd.Series(beta[k] / beta.mean(0), index=panel)
        hk = bk[bk > bk.quantile(0.9)].index.intersection(ov)
        gk = lk[hk].sort_values(ascending=False).index[:50]
        sck = Z.loc[gk].mean(0).to_numpy()
        d, p, _ = exact_perm(sck, dfu, dm | ndm)
        tk.append({"topic": k, "n_genes": len(gk), "DFU_vs_intact_diff": d, "p_exact_perm": p})
    out["all_topics_LIFT50_DFU_vs_intact"] = tk
    out["topic0_rank_among_15_by_DFU_vs_intact_diff"] = int(1 + sum(t["DFU_vs_intact_diff"] > tk[0]["DFU_vs_intact_diff"] for t in tk))
    dump_json(out, RES / "p1_6b_bulk_gse80178.json")
    write_manifest("p1_6b_bulk_gse80178")
    import json
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1, default=float)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
