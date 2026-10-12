#!/usr/bin/env python3
"""P1-1 selection-aware healing inference over the 15 frozen topics.

Unit: 11 author-mapped DFU patients (7 healed / 4 non-healed), cell-pooled
within patient.  All 330 label allocations are enumerated.

Readout families (each 11 x 15):
  S  all-cell mean loading (primary estimand, Eq. 8)
  M  fibroblast mean loading (Eq. 18)
  W  fibroblast fraction above that topic's analogous cut (2-component GMM
     midpoint on all 19,410 discovery fibroblast loadings for that topic,
     same recipe as scripts/13: n_init=10, random_state=0).  For topic 0 the
     frozen cut 0.1842958 is also reported separately.
  C  cell-pooled arm-mean ratio (the statistic printed in report.json
     mean_theta_by_arm, i.e. the statistic under which topic 0 stood out).

Selection statistics per allocation: max_k healed/non-healed ratio (one-sided,
direction of the original selection), max_k |log ratio|, max_k |difference|,
max_k |Welch t|.  Selection-adjusted p = fraction of the 330 allocations whose
maximum is >= the observed maximum (observed allocation included).

Selection loops: (i) leave-one-patient-out: choose argmax-ratio topic on 10
patients, score the held patient as a z-score against the 10 training values
of that topic, AUC over 11 out-of-fold scores; (ii) held pair: leave one healed
and one non-healed patient out (28 pairs), choose topic on the other 9, compare
the two held patients on that topic.  Both loops are rerun under all 330
allocations for an exact null.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import (RES, FROZEN_CUT, TOL, allocations, dfu_patient_table, dump_json,  # noqa: E402
                       group_means, load_cells, observed_index, write_manifest)

K = 15


def gmm_midpoint(x: np.ndarray) -> float:
    g = GaussianMixture(n_components=2, n_init=10, random_state=0).fit(x.reshape(-1, 1))
    return float(g.means_.ravel().mean())


def welch_t(values: np.ndarray, A: np.ndarray) -> np.ndarray:
    out = np.zeros((len(A), values.shape[1]))
    for r, a in enumerate(A):
        h, n = values[a], values[~a]
        se = np.sqrt(h.var(0, ddof=1) / len(h) + n.var(0, ddof=1) / len(n))
        se[se == 0] = np.inf
        out[r] = (h.mean(0) - n.mean(0)) / se
    return out


def selection_family(name: str, V: np.ndarray, A: np.ndarray, obs: int,
                     mH=None, mN=None) -> tuple[dict, pd.DataFrame]:
    if mH is None:
        mH, mN = group_means(V, A)
    ratio = mH / mN
    diff = mH - mN
    logr = np.abs(np.log(ratio))
    t = welch_t(V, A) if V is not None else None
    stats = {"max_ratio": ratio.max(1), "max_abs_log_ratio": logr.max(1),
             "max_abs_diff": np.abs(diff).max(1)}
    if t is not None:
        stats["max_abs_welch_t"] = np.abs(t).max(1)
    out = {"readout": name}
    for s, null in stats.items():
        o = null[obs]
        arg = {"max_ratio": ratio, "max_abs_log_ratio": logr, "max_abs_diff": np.abs(diff),
               "max_abs_welch_t": np.abs(t) if t is not None else None}[s][obs]
        out[s] = {"observed_max": float(o), "argmax_topic": int(np.argmax(arg)),
                  "selection_adjusted_p": float(np.mean(null >= o - TOL)),
                  "null_median": float(np.median(null)), "null_95pct": float(np.quantile(null, 0.95))}
    # marginal topic-0 inference
    out["topic0"] = {"healed_mean": float(mH[obs, 0]), "nonhealed_mean": float(mN[obs, 0]),
                     "difference": float(diff[obs, 0]), "ratio": float(ratio[obs, 0]),
                     "ratio_rank_among_15": int(1 + np.sum(ratio[obs] > ratio[obs, 0])),
                     "p_two_sided_abs_diff": float(np.mean(np.abs(diff[:, 0]) >= abs(diff[obs, 0]) - TOL)),
                     "p_one_sided_ratio": float(np.mean(ratio[:, 0] >= ratio[obs, 0] - TOL))}
    rows = pd.DataFrame({"readout": name, "topic": np.arange(K), "healed_mean": mH[obs],
                         "nonhealed_mean": mN[obs], "difference": diff[obs], "ratio": ratio[obs],
                         "p_two_sided_marginal": [np.mean(np.abs(diff[:, k]) >= abs(diff[obs, k]) - TOL) for k in range(K)],
                         "p_one_sided_ratio_marginal": [np.mean(ratio[:, k] >= ratio[obs, k] - TOL) for k in range(K)]})
    return out, rows


def choose(V: np.ndarray, train: np.ndarray, lab: np.ndarray) -> int:
    h = V[train & lab].mean(0); n = V[train & ~lab].mean(0)
    return int(np.argmax(h / n))


def loops(V: np.ndarray, lab: np.ndarray) -> dict:
    n = len(lab)
    # LOPO
    z = np.zeros(n); picks = []
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        k = choose(V, tr, lab); picks.append(k)
        mu, sd = V[tr, k].mean(), V[tr, k].std(ddof=1)
        z[i] = (V[i, k] - mu) / sd
    hz, nz = z[lab][:, None], z[~lab][None, :]
    auc = float(((hz > nz).sum() + 0.5 * (hz == nz).sum()) / (hz.size * nz.size))
    # held pair
    wins = []; ppicks = []
    for i in np.flatnonzero(lab):
        for j in np.flatnonzero(~lab):
            tr = np.ones(n, bool); tr[[i, j]] = False
            k = choose(V, tr, lab); ppicks.append(k)
            wins.append(1.0 if V[i, k] > V[j, k] else (0.5 if V[i, k] == V[j, k] else 0.0))
    return {"lopo_auc": auc, "lopo_topics": picks, "pair_auc": float(np.mean(wins)), "pair_topics": ppicks}


def main() -> int:
    cells = load_cells()
    theta = cells[[f"topic_{k}" for k in range(K)]].to_numpy(np.float64)
    fib = (cells.celltype == "fibroblast").to_numpy()
    cuts = np.array([gmm_midpoint(theta[fib, k]) for k in range(K)])
    tab = dfu_patient_table(cells, theta, cuts)
    tab_frozen = dfu_patient_table(cells, theta, np.full(K, FROZEN_CUT))
    A = __import__("p1_common").allocations(11, 7)
    lab = tab["healed"]; obs = observed_index(A, lab)

    # cell-pooled arm-mean statistic (selection statistic actually printed)
    dfu = cells.arm.isin(["DFU-healer", "DFU-nonhealer"]).to_numpy()
    sums = np.vstack([theta[dfu & (cells.patient_id == p).to_numpy()].sum(0) for p in tab["patients"]])
    cnt = tab["n_cells"].astype(float)
    Af = A.astype(float)
    cH = (Af @ sums) / (Af @ cnt)[:, None]; cN = ((1 - Af) @ sums) / ((1 - Af) @ cnt)[:, None]

    results = {"n_allocations": len(A), "per_topic_gmm_cuts": cuts.tolist(),
               "topic0_cut_refit": float(cuts[0]), "topic0_cut_frozen": FROZEN_CUT}
    allrows = []
    fams = [("S_all_cell_mean", tab["S"]), ("M_fibroblast_mean", tab["M"]),
            ("W_fibroblast_fraction_topic_cut", tab["W"])]
    for name, V in fams:
        r, rows = selection_family(name, V, A, obs)
        results[name] = r; allrows.append(rows)
    r, rows = selection_family("C_cell_pooled_arm_mean", None, A, obs, cH, cN)
    results["C_cell_pooled_arm_mean"] = r; allrows.append(rows)
    # topic 0 with the frozen cut (marginal only; other topics have no frozen cut)
    W0 = tab_frozen["W"][:, 0]
    mH, mN = group_means(W0, A)
    d = mH - mN
    results["W0_frozen_cut_marginal"] = {"difference": float(d[obs]),
                                         "p_two_sided": float(np.mean(np.abs(d) >= abs(d[obs]) - TOL))}
    # family-wise max over all three patient readouts jointly (standardised by |log ratio|)
    joint = np.maximum.reduce([np.abs(np.log(group_means(V, A)[0] / group_means(V, A)[1])).max(1)
                               for _, V in fams])
    results["joint_max_abs_log_ratio_S_M_W"] = {"observed": float(joint[obs]),
                                                "selection_adjusted_p": float(np.mean(joint >= joint[obs] - TOL))}

    # selection loops
    loop_res = {}
    for name, V in fams:
        o = loops(V, lab)
        nullp, nulll = [], []
        for a in A:
            q = loops(V, a)
            nullp.append(q["pair_auc"]); nulll.append(q["lopo_auc"])
        nullp, nulll = np.array(nullp), np.array(nulll)
        loop_res[name] = {
            "lopo_auc": o["lopo_auc"], "lopo_p_one_sided": float(np.mean(nulll >= o["lopo_auc"] - TOL)),
            "lopo_topic_counts": pd.Series(o["lopo_topics"]).value_counts().to_dict(),
            "held_pair_auc": o["pair_auc"], "held_pair_p_one_sided": float(np.mean(nullp >= o["pair_auc"] - TOL)),
            "held_pair_topic_counts": pd.Series(o["pair_topics"]).value_counts().to_dict(),
            "null_lopo_auc_median": float(np.median(nulll)), "null_pair_auc_median": float(np.median(nullp))}
    results["selection_loops"] = loop_res

    # fixed topic-0 held-pair for comparison (no selection)
    results["patients"] = tab["patients"]; results["healed"] = lab.tolist()
    pd.concat(allrows).to_csv(RES / "p1_1_topic_contrasts.csv", index=False)
    pt = pd.DataFrame({"patient_id": tab["patients"], "healed": lab, "n_cells": tab["n_cells"],
                       "n_fibroblasts": tab["n_fib"], "F_fibroblast_composition": tab["F"]})
    for k in range(K):
        pt[f"S_topic{k}"] = tab["S"][:, k]; pt[f"M_topic{k}"] = tab["M"][:, k]; pt[f"W_topic{k}"] = tab["W"][:, k]
    pt["W_topic0_frozen_cut"] = W0
    pt.to_csv(RES / "p1_1_patient_readouts_all_topics.csv", index=False)
    dump_json(results, RES / "p1_1_selection.json")
    write_manifest("p1_1_selection")
    import json
    print(json.dumps({k: v for k, v in results.items() if k not in ("patients",)}, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
