#!/usr/bin/env python3
"""P1-4 propagate seed/capacity variability to the healing contrast.

Fits: nine fixed-grid fits (K=10/15/20 x seeds 0-2, 40k steps, batch 256;
output/publication-quality-20261009/p1/run/jobs/*) and six budget/panel
diagnostic fits (K=15, batch 128; output/sdd-local-validation-20260929/runs/
topic-inherited|topic-training/jobs/*).  The frozen fit is included as reference.

For each fit: the saved Hungarian-assigned topic (result.json
fibroblast_topic) and saved descriptive cut (result.json cut) are used
unchanged.  Patient readouts (11 DFU patients, cell-pooled, as in the
manuscript's primary estimands): S all-cell mean, M fibroblast mean, W
fibroblast fraction above the fit's cut.  Exact two-sided p over 330
allocations.  Separately, the topic with the largest healed/non-healed ratio is
re-chosen per fit (patient-level S ratio and cell-pooled arm-mean ratio as in
report.json) with a selection-adjusted (max over K) p.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import (RES, ROOT, FROZEN_CUT, TOL, allocations, dfu_patient_table, dump_json,  # noqa: E402
                       group_means, load_cells, observed_index, register, write_manifest)


def fits():
    out = [("frozen", 15, None, "frozen_discovery_fit", None)]
    grid = ROOT / "output/publication-quality-20261009/p1/run"
    for line in register(grid / "jobs.jsonl").read_text().splitlines():
        j = json.loads(line)
        out.append((f"grid_K{j['params']['K']}_s{j['seed']}", j["params"]["K"], j["seed"], "grid_batch256_40k",
                    grid / "jobs" / j["job_id"]))
    for run in ["topic-inherited", "topic-training"]:
        d = ROOT / "output/sdd-local-validation-20260929/runs" / run
        for line in register(d / "jobs.jsonl").read_text().splitlines():
            j = json.loads(line)
            out.append((f"diag_{run}_{j['params']['steps']//1000}k_s{j['seed']}", j["params"]["K"], j["seed"],
                        f"diagnostic_{run}_batch128", d / "jobs" / j["job_id"]))
    return out


def main() -> int:
    cells = load_cells()
    key = cells["reconstructed_raw_cell_id"].to_numpy()
    A = allocations(11, 7)
    rows, sel_rows, patient_rows = [], [], []
    for name, K, seed, kind, path in fits():
        if path is None:
            theta = cells[[f"topic_{k}" for k in range(15)]].to_numpy(np.float64)
            topic, cut, corr = 0, FROZEN_CUT, 1.0
        else:
            res = json.loads(register(path / "result.json").read_text())
            z = np.load(register(path / "topics.npz"), allow_pickle=False)
            ids = z["cell_ids"].astype(str)
            pos = pd.Index(ids).get_indexer(key)
            assert (pos >= 0).all(), name
            theta = z["theta"][pos].astype(np.float64)
            topic, cut, corr = int(res["fibroblast_topic"]), float(res["cut"]), float(res["fibroblast_topic_correlation"])
        cutv = np.full(theta.shape[1], np.nan); cutv[topic] = cut
        tab = dfu_patient_table(cells, theta, np.where(np.isnan(cutv), 1.0, cutv))
        lab = tab["healed"]; obs = observed_index(A, lab)
        row = {"fit": name, "kind": kind, "K": K, "seed": seed, "matched_topic": topic,
               "decoder_match_r": corr, "cut": cut}
        for r in ["S", "M", "W"]:
            v = tab[r][:, topic]
            mH, mN = group_means(v, A); d = mH - mN
            row[f"{r}_healed"] = mH[obs]; row[f"{r}_nonhealed"] = mN[obs]
            row[f"{r}_diff"] = d[obs]
            row[f"{r}_p_exact"] = float(np.mean(np.abs(d) >= abs(d[obs]) - TOL))
            row[f"{r}_rank_of_obs_among_330"] = int(1 + np.sum(d > d[obs] + TOL))
            for i, p in enumerate(tab["patients"]):
                patient_rows.append({"fit": name, "patient_id": p, "healed": bool(lab[i]), "readout": r, "value": v[i]})
        # sanity: equal-specimen fibroblast mean vs saved patient_readouts
        if path is not None:
            saved = {d_["unit"]: d_["mean"] for d_ in res["patient_readouts"]}
            fibm = cells.celltype.eq("fibroblast").to_numpy()
            eq = []
            for p in tab["patients"]:
                ms = [theta[fibm & (cells.gsm == g).to_numpy(), topic].mean()
                      for g in cells.loc[cells.patient_id == p, "gsm"].unique()]
                eq.append(abs(np.mean(ms) - saved[p]))
            row["max_abs_diff_vs_saved_equal_specimen_mean"] = float(max(eq))
        rows.append(row)
        # per-fit re-selection by max ratio
        S = tab["S"]
        mH, mN = group_means(S, A); ratio = mH / mN
        dfu = cells.arm.isin(["DFU-healer", "DFU-nonhealer"]).to_numpy()
        sums = np.vstack([theta[dfu & (cells.patient_id == p).to_numpy()].sum(0) for p in tab["patients"]])
        Af = A.astype(float); cnt = tab["n_cells"].astype(float)
        cr = ((Af @ sums) / (Af @ cnt)[:, None]) / (((1 - Af) @ sums) / ((1 - Af) @ cnt)[:, None])
        for stat, R in [("patient_S_ratio", ratio), ("cell_pooled_arm_ratio", cr)]:
            k = int(np.argmax(R[obs]))
            sel_rows.append({"fit": name, "K": K, "seed": seed, "statistic": stat, "selected_topic": k,
                             "selected_equals_matched": k == topic, "selected_ratio": float(R[obs, k]),
                             "matched_topic_ratio": float(R[obs, topic]),
                             "matched_topic_ratio_rank": int(1 + np.sum(R[obs] > R[obs, topic])),
                             "selection_adjusted_p_max_ratio": float(np.mean(R.max(1) >= R[obs].max() - TOL)),
                             "marginal_p_one_sided_matched": float(np.mean(R[:, topic] >= R[obs, topic] - TOL))})
    df = pd.DataFrame(rows); sel = pd.DataFrame(sel_rows)
    df.to_csv(RES / "p1_4_fit_contrasts.csv", index=False)
    sel.to_csv(RES / "p1_4_fit_reselection.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(RES / "p1_4_fit_patient_readouts.csv", index=False)
    summ = {}
    for kind_sel, m in [("grid9", df.kind.eq("grid_batch256_40k")), ("diag6", df.kind.str.startswith("diagnostic")),
                        ("grid9_plus_diag6", df.kind.ne("frozen_discovery_fit"))]:
        s = {}
        for r in ["S", "M", "W"]:
            s[r] = {"diff_min": float(df.loc[m, f"{r}_diff"].min()), "diff_max": float(df.loc[m, f"{r}_diff"].max()),
                    "diff_median": float(df.loc[m, f"{r}_diff"].median()),
                    "n_positive": int((df.loc[m, f"{r}_diff"] > 0).sum()), "n": int(m.sum()),
                    "p_min": float(df.loc[m, f"{r}_p_exact"].min()), "p_max": float(df.loc[m, f"{r}_p_exact"].max()),
                    "n_p_below_0.05": int((df.loc[m, f"{r}_p_exact"] < 0.05).sum())}
        summ[kind_sel] = s
    dump_json({"summary": summ, "fits": rows, "reselection": sel_rows}, RES / "p1_4_fit_propagation.json")
    write_manifest("p1_4_fit_propagation")
    pd.set_option("display.width", 250)
    print(df[["fit", "matched_topic", "decoder_match_r", "cut", "S_diff", "S_p_exact", "M_diff", "M_p_exact",
              "W_diff", "W_p_exact"] + (["max_abs_diff_vs_saved_equal_specimen_mean"])].round(4).to_string())
    print(sel.round(4).to_string())
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
