#!/usr/bin/env python3
"""P1-2 step 2: diabetes / wound / site decomposition at patient level.

Input: results/p1_2_projected_cells.parquet (frozen gate + frozen encoder,
reproduced the 76,461 discovery cells to 4.8e-7).  Topic 0, frozen cut
0.1842958.  Patient readouts pool cells across a patient's specimens of the
given site:
  W  fibroblast fraction above cut      M  fibroblast mean topic-0
  S  all-cell mean topic-0              F  fibroblast composition
  HURDLE_05 / HURDLE_02  indicator W > 0.05 / > 0.02
  COND  mean topic-0 among above-cut fibroblasts (patients with >= 10 such cells)
Groups (foot): H (9 non-diabetic), DM (6 diabetic non-ulcer), DFU-H (7), DFU-NH (4).
Exact tests enumerate every relabelling of the patients in the two compared
groups: two-sided mean difference and two-sided Mann-Whitney (mid-ranks).
Site: paired foot-minus-forearm, exact sign flip.
"""
from __future__ import annotations

import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import RES, ROOT, FROZEN_CUT, TOL, dump_json, register, write_manifest  # noqa: E402

READS = ["W", "M", "S", "F", "HURDLE_05", "HURDLE_02", "COND"]


def patient_readouts(cells: pd.DataFrame, site: str) -> pd.DataFrame:
    rows = []
    sub = cells[cells.tissue_site == site]
    for p, b in sub.groupby("patient_id"):
        fib = b[b.celltype == "fibroblast"]
        hi = fib.topic_0 > FROZEN_CUT
        W = hi.mean() if len(fib) else np.nan
        rows.append({"patient_id": p, "site": site, "group": b.group.iloc[0],
                     "specimens": "|".join(sorted(b.sample_title.unique())),
                     "n_cells": len(b), "n_fib": len(fib), "n_high": int(hi.sum()),
                     "W": W, "M": fib.topic_0.mean() if len(fib) else np.nan,
                     "S": b.topic_0.mean(), "F": len(fib) / len(b),
                     "HURDLE_05": float(W > 0.05), "HURDLE_02": float(W > 0.02),
                     "COND": fib.topic_0[hi].mean() if hi.sum() >= 10 else np.nan})
    return pd.DataFrame(rows)


def exact_two_group(x: np.ndarray, y: np.ndarray, chunk: int = 200_000) -> dict:
    """Exact two-sided permutation for mean difference and Mann-Whitney U."""
    v = np.concatenate([x, y]); n, nx = len(v), len(x)
    r = rankdata(v)
    obs_d = x.mean() - y.mean()
    obs_u = r[:nx].sum() - nx * (nx + 1) / 2
    cu = nx * (n - nx) / 2
    total = comb(n, nx); ge_d = ge_u = 0
    it = itertools.combinations(range(n), nx)
    sv, sr = v.sum(), r.sum()
    while True:
        block = list(itertools.islice(it, chunk))
        if not block:
            break
        idx = np.array(block)
        sx = v[idx].sum(1); d = sx / nx - (sv - sx) / (n - nx)
        u = r[idx].sum(1) - nx * (nx + 1) / 2
        ge_d += int(np.sum(np.abs(d) >= abs(obs_d) - TOL))
        ge_u += int(np.sum(np.abs(u - cu) >= abs(obs_u - cu) - TOL))
    return {"n1": nx, "n2": n - nx, "mean1": float(x.mean()), "mean2": float(y.mean()),
            "median1": float(np.median(x)), "median2": float(np.median(y)),
            "diff": float(obs_d), "p_perm_mean": ge_d / total,
            "AUC": float(obs_u / (nx * (n - nx))), "p_exact_mw": ge_u / total,
            "n_relabellings": total}


def sign_flip(d: np.ndarray) -> dict:
    d = np.asarray(d, float); n = len(d)
    null = np.array([np.mean(d * np.array(s)) for s in itertools.product((-1, 1), repeat=n)])
    return {"n_pairs": n, "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "p_exact_sign_flip": float(np.mean(np.abs(null) >= abs(d.mean()) - TOL)),
            "min_attainable_p": 2 / 2 ** n, "diffs": d.tolist()}


def main() -> int:
    cells = pd.read_parquet(RES / "p1_2_projected_cells.parquet")
    foot = patient_readouts(cells, "foot")
    arm = patient_readouts(cells, "forearm")
    gmap = {"H": "H", "DM": "DM", "DFU-H": "DFU_H", "DFU-NH": "DFU_NH"}
    foot["grp"] = foot.group.map(gmap)
    tab = pd.concat([foot, arm.assign(grp=arm.group.map(gmap))])
    tab.to_csv(RES / "p1_2_patient_readouts.csv", index=False)
    spec_rows = []
    for (g, site), b in cells.groupby(["gsm", "tissue_site"]):
        fib = b[b.celltype == "fibroblast"]
        spec_rows.append({"gsm": g, "title": b.sample_title.iloc[0], "patient_id": b.patient_id.iloc[0],
                          "group": b.group.iloc[0], "site": site, "n_cells": len(b), "n_fib": len(fib),
                          "W": (fib.topic_0 > FROZEN_CUT).mean() if len(fib) else np.nan,
                          "M": fib.topic_0.mean() if len(fib) else np.nan, "S": b.topic_0.mean(),
                          "F": len(fib) / len(b)})
    pd.DataFrame(spec_rows).to_csv(RES / "p1_2_specimen_readouts.csv", index=False)

    groups = {g: foot[foot.grp == g] for g in ["H", "DM", "DFU_H", "DFU_NH"]}
    groups["DFU"] = foot[foot.grp.isin(["DFU_H", "DFU_NH"])]
    groups["INTACT"] = foot[foot.grp.isin(["H", "DM"])]
    contrasts = [("DM", "H", "diabetes (intact foot skin)"), ("DFU", "DM", "wound within diabetes"),
                 ("DFU", "H", "DFU vs non-diabetic"), ("DFU", "INTACT", "wound vs all intact foot"),
                 ("DFU_H", "DM", "healed DFU vs DM"), ("DFU_NH", "DM", "non-healed DFU vs DM"),
                 ("DFU_H", "DFU_NH", "healing (check: reproduces paper)")]
    res = {"group_summary": {}, "contrasts": [], "site": {}}
    for g, b in groups.items():
        res["group_summary"][g] = {r: {"n": int(b[r].notna().sum()), "mean": float(b[r].mean()),
                                       "median": float(b[r].median()),
                                       "values": dict(zip(b.patient_id, b[r].round(5)))}
                                   for r in READS}
    rows = []
    for a, b, lab in contrasts:
        for r in READS:
            x = groups[a][r].dropna().to_numpy(float); y = groups[b][r].dropna().to_numpy(float)
            out = exact_two_group(x, y)
            out.update({"contrast": f"{a} vs {b}", "meaning": lab, "readout": r})
            rows.append(out)
    ct = pd.DataFrame(rows)
    ct.to_csv(RES / "p1_2_group_contrasts.csv", index=False)
    res["contrasts"] = rows

    # site: paired foot minus forearm
    pair = foot.merge(arm, on="patient_id", suffixes=("_foot", "_arm"))
    pair["status"] = pair.group_foot
    prow = []
    for r in ["W", "M", "S", "F"]:
        pair[f"d_{r}"] = pair[f"{r}_foot"] - pair[f"{r}_arm"]
    pair.to_csv(RES / "p1_2_paired_site.csv", index=False)
    for subset, mask in [("all_10_pairs (manuscript anatomical control)", np.ones(len(pair), bool)),
                         ("intact_skin_only (DM+H)", pair.group_foot.isin(["H", "DM"]).to_numpy()),
                         ("intact_non_diabetic_only (H)", pair.group_foot.eq("H").to_numpy()),
                         ("intact_excluding_S19", (pair.group_foot.isin(["H", "DM"]) & pair.patient_id.ne("NCOMMS2022_S19")).to_numpy()),
                         ("DFU_only", pair.group_foot.isin(["DFU-H", "DFU-NH"]).to_numpy())]:
        res["site"][subset] = {"patients": pair.patient_id[mask].tolist()}
        for r in ["W", "M", "S", "F"]:
            res["site"][subset][r] = sign_flip(pair.loc[mask, f"d_{r}"].to_numpy())
    # wound-by-site interaction: DFU pair differences vs intact pair differences
    inter = {}
    dfu_m = pair.group_foot.isin(["DFU-H", "DFU-NH"]).to_numpy()
    for r in ["W", "M", "S"]:
        inter[r] = exact_two_group(pair.loc[dfu_m, f"d_{r}"].to_numpy(float),
                                   pair.loc[~dfu_m, f"d_{r}"].to_numpy(float))
    res["site"]["DFU_minus_intact_site_difference"] = inter
    # comparison with the original anatomical-control file (re-standardised gate)
    orig = pd.read_csv(register(ROOT / "outputs/repaired_20260918/paired_anatomical_control/paired_patient_scores.csv"))
    m = orig.merge(pair[["patient_id", "S_foot", "S_arm", "M_foot", "M_arm"]], on="patient_id")
    res["site"]["agreement_with_original_control"] = {
        "max_abs_diff_all_cell_foot": float((m.topic0_all_mean_foot - m.S_foot).abs().max()),
        "max_abs_diff_all_cell_forearm": float((m.topic0_all_mean_forearm - m.S_arm).abs().max()),
        "max_abs_diff_fibro_foot": float((m.topic0_fibro_mean_foot - m.M_foot).abs().max()),
        "max_abs_diff_fibro_forearm": float((m.topic0_fibro_mean_forearm - m.M_arm).abs().max()),
        "note": "original control re-standardised the marker gate on foot+forearm cells; this run uses the frozen 76,461-cell reference"}
    dump_json(res, RES / "p1_2_decomposition.json")
    register(RES / "p1_2_projected_cells.parquet")
    write_manifest("p1_2_decomposition")
    pd.set_option("display.width", 250)
    print(foot.sort_values(["grp", "W"])[["patient_id", "grp", "specimens", "n_fib", "W", "M", "S", "F", "COND"]].to_string())
    print(arm[["patient_id", "group", "n_fib", "W", "M", "S"]].to_string())
    print(ct[["contrast", "readout", "mean1", "mean2", "diff", "p_perm_mean", "AUC", "p_exact_mw"]].to_string())
    for k, v in res["site"].items():
        if "patients" in v:
            print(k, {r: (round(v[r]["mean_diff"], 4), round(v[r]["p_exact_sign_flip"], 4)) for r in "WMSF"})
    print({r: (round(x["diff"], 4), round(x["p_perm_mean"], 4), round(x["p_exact_mw"], 4)) for r, x in inter.items()})
    print(res["site"]["agreement_with_original_control"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
