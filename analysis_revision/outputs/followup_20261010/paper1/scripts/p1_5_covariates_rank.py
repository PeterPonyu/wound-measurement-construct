#!/usr/bin/env python3
"""P1-5 covariate balance by outcome and rank / hurdle patient statistics.

Covariates (GSE165816_ncomms_subject_sample_map.csv, author clinical table):
age, sex, BMI, HbA1c, wound area, diabetes duration.  11 DFU patients
(7 healed / 4 non-healed); DM and H groups are also tabulated descriptively.

Topic-0 patient readouts (cell-pooled, frozen cut): S, M, W, F, plus hurdle
summaries: H05 = 1{W>0.05}, H02 = 1{W>0.02}, COND = mean topic-0 of above-cut
fibroblasts (patients with >= 10 such cells).
Statistics: exact two-sided Mann-Whitney over all 330 allocations (mid-ranks),
AUC = P(healed > non-healed) + 0.5 ties, Hodges-Lehmann shift with exact
Lehmann CI from the exact U distribution, exact Fisher test for hurdles, and
Spearman rho of each readout with each covariate within the 11 DFU patients
(exact permutation p by full enumeration where n<=9 non-missing, otherwise
200,000 random permutations, seed 20261010).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import (RES, FROZEN_CUT, TOL, allocations, dump_json, exact_rank_test,  # noqa: E402
                       load_cells, load_map, write_manifest)

COV = ["age", "bmi", "hba1c", "wound_area_cm2", "dm_duration_years"]


def main() -> int:
    mp = load_map()
    pat = mp.drop_duplicates("patient_id").set_index("patient_id")
    cells = load_cells()
    dfu = cells[cells.arm.isin(["DFU-healer", "DFU-nonhealer"])]
    rows = []
    for p, b in dfu.groupby("patient_id"):
        f = b[b.celltype == "fibroblast"]
        hi = f.topic_0 > FROZEN_CUT
        W = hi.mean()
        rows.append({"patient_id": p, "healed": b.healing_status.iloc[0] == "healed",
                     "S": b.topic_0.mean(), "M": f.topic_0.mean(), "W": W, "F": len(f) / len(b),
                     "H05": float(W > 0.05), "H02": float(W > 0.02),
                     "COND": f.topic_0[hi].mean() if hi.sum() >= 10 else np.nan,
                     "n_cells": len(b), "n_fib": len(f)})
    t = pd.DataFrame(rows).set_index("patient_id")
    for c in COV + ["sex"]:
        t[c] = pat.loc[t.index, c]
    t["male"] = (t.sex == "M").astype(float)
    t.to_csv(RES / "p1_5_patient_table.csv")

    # balance table (all groups, descriptive), DFU exact tests
    grp = mp.drop_duplicates("patient_id")[["patient_id", "group"] + COV + ["sex"]].copy()
    bal = []
    for c in COV + ["male"]:
        col = (grp.sex == "M").astype(float) if c == "male" else grp[c]
        r = {"covariate": c}
        for g in ["DFU-H", "DFU-NH", "DM", "H"]:
            v = col[grp.group == g].dropna()
            r[f"{g}_n"] = len(v)
            r[f"{g}_median"] = float(v.median()) if len(v) else np.nan
            r[f"{g}_IQR"] = f"{v.quantile(.25):.2f}-{v.quantile(.75):.2f}" if len(v) else ""
            r[f"{g}_mean"] = float(v.mean()) if len(v) else np.nan
        x = t[c].astype(float); ok = x.notna().to_numpy()
        h = t.healed.to_numpy()
        if c == "male":
            tab = [[int(((x == 1) & h).sum()), int(((x == 0) & h).sum())],
                   [int(((x == 1) & ~h).sum()), int(((x == 0) & ~h).sum())]]
            r["dfu_test"] = "Fisher exact"
            r["dfu_p"] = float(fisher_exact(tab)[1])
            r["healed_male_n"] = tab[0][0]; r["nonhealed_male_n"] = tab[1][0]
        else:
            hv, nv = h[ok], x.to_numpy()[ok]
            A = allocations(len(hv), int(hv.sum()))
            rt = exact_rank_test(nv, hv, A)
            r.update({"dfu_test": f"exact Mann-Whitney ({int(hv.sum())} vs {int((~hv).sum())})",
                      "dfu_p": rt["p_exact_two_sided"], "dfu_AUC_healed_gt": rt["AUC"],
                      "dfu_HL_healed_minus_nonhealed": rt["hodges_lehmann"],
                      "dfu_HL_CI": rt["hl_ci"], "dfu_HL_CI_coverage": rt["hl_ci_coverage"],
                      "dfu_missing": int((~ok).sum())})
        bal.append(r)
    bal = pd.DataFrame(bal)
    bal.to_csv(RES / "p1_5_covariate_balance.csv", index=False)

    # rank / hurdle statistics for readouts
    A = allocations(11, 7)
    h = t.healed.to_numpy()
    rank = []
    for r in ["S", "M", "W", "F"]:
        out = exact_rank_test(t[r].to_numpy(float), h, A)
        out["readout"] = r
        rank.append(out)
    for r in ["H05", "H02"]:
        x = t[r].to_numpy()
        tab = [[int((x[h] == 1).sum()), int((x[h] == 0).sum())], [int((x[~h] == 1).sum()), int((x[~h] == 0).sum())]]
        rank.append({"readout": r, "table_healed_[pos,neg]_nonhealed_[pos,neg]": tab,
                     "p_fisher_exact": float(fisher_exact(tab)[1])})
    ok = t.COND.notna().to_numpy()
    hv = h[ok]; A2 = allocations(int(ok.sum()), int(hv.sum()))
    c = exact_rank_test(t.COND.to_numpy(float)[ok], hv, A2); c["readout"] = "COND (patients with >=10 high cells)"
    rank.append(c)

    # within-DFU Spearman of readouts with covariates
    rng = np.random.default_rng(20261010)
    sp = []
    for r in ["S", "M", "W", "F"]:
        for cv in COV + ["male"]:
            x = t[cv].astype(float); y = t[r]; ok = x.notna()
            xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
            rho = spearmanr(xv, yv)[0]
            null = np.array([spearmanr(xv, rng.permutation(yv))[0] for _ in range(20000)])
            sp.append({"readout": r, "covariate": cv, "n": int(ok.sum()), "spearman_rho": float(rho),
                       "p_perm_two_sided_20000": float((1 + np.sum(np.abs(null) >= abs(rho) - TOL)) / 20001)})
    sp = pd.DataFrame(sp)
    sp.to_csv(RES / "p1_5_readout_covariate_spearman.csv", index=False)
    # covariate adjustment illustration: residualise W on age (rank-based) is not
    # attempted with n=11; instead report the stratified view
    dump_json({"rank_hurdle": rank, "balance": bal.to_dict(orient="records")}, RES / "p1_5_covariates_rank.json")
    write_manifest("p1_5_covariates_rank")
    pd.set_option("display.width", 250)
    print(t.round(4).to_string())
    print(bal.round(3).to_string())
    for x in rank:
        print(x)
    print(sp.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
