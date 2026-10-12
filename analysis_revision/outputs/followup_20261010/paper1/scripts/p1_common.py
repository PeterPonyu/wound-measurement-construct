"""Shared helpers for the paper-1 follow-up analyses (2026-10-10).

Read-only with respect to the repository: every input is opened for reading and
every output goes under outputs/followup_20261010/paper1/.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/followup_20261010/paper1"
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)

CELLS = ROOT / "outputs/scientific_revision_20260922/measurement/projection/cells.parquet"
SUBJECT_MAP = ROOT / "metadata_intake/GSE165816_ncomms_subject_sample_map.csv"
MAINLINE = ROOT / "outputs/direction1_dfu_mainline"
FROZEN_CUT = 0.18429584801197052  # outputs/bimodal_stratification/report.json GSE165816_discovery.gmm.midpoint
TOL = 1e-12

INPUTS_USED: dict[str, str] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def register(path: Path) -> Path:
    path = Path(path)
    INPUTS_USED[str(path.relative_to(ROOT))] = sha256(path)
    return path


def write_manifest(name: str, extra: dict | None = None) -> None:
    data = {"script": name, "inputs_sha256": dict(sorted(INPUTS_USED.items()))}
    if extra:
        data.update(extra)
    (RES / f"{name}_input_manifest.json").write_text(json.dumps(data, indent=2))


def dump_json(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=conv))


def load_cells() -> pd.DataFrame:
    return pd.read_parquet(register(CELLS))


def load_map() -> pd.DataFrame:
    return pd.read_csv(register(SUBJECT_MAP))


def dfu_patient_table(cells: pd.DataFrame, theta: np.ndarray | None = None,
                      cut_by_topic: np.ndarray | None = None) -> dict:
    """Cell-pooled patient summaries for every topic.

    Returns dict with arrays shaped (n_patients, K): S (all-cell mean),
    M (fibroblast mean), W (fibroblast fraction above per-topic cut), plus
    F (fibroblast composition, shape n_patients), patient ids and healed flag.
    Only the 11 DFU patients (14 DFU foot specimens) are returned.
    """
    if theta is None:
        theta = cells[[f"topic_{k}" for k in range(15)]].to_numpy(np.float64)
    K = theta.shape[1]
    dfu = cells["arm"].isin(["DFU-healer", "DFU-nonhealer"]).to_numpy()
    fib = (cells["celltype"] == "fibroblast").to_numpy()
    pats = sorted(cells.loc[dfu, "patient_id"].unique())
    healed = []
    S = np.zeros((len(pats), K)); M = np.zeros((len(pats), K)); W = np.zeros((len(pats), K))
    F = np.zeros(len(pats)); nall = np.zeros(len(pats), int); nfib = np.zeros(len(pats), int)
    for i, p in enumerate(pats):
        m = dfu & (cells["patient_id"] == p).to_numpy()
        hs = cells.loc[m, "healing_status"].unique()
        assert len(hs) == 1
        healed.append(hs[0] == "healed")
        S[i] = theta[m].mean(0)
        mf = m & fib
        M[i] = theta[mf].mean(0)
        if cut_by_topic is not None:
            W[i] = (theta[mf] > cut_by_topic[None, :]).mean(0)
        F[i] = mf.sum() / m.sum()
        nall[i] = m.sum(); nfib[i] = mf.sum()
    return {"patients": pats, "healed": np.array(healed), "S": S, "M": M, "W": W,
            "F": F, "n_cells": nall, "n_fib": nfib}


def allocations(n: int = 11, k: int = 7) -> np.ndarray:
    """Boolean matrix (C(n,k), n): every allocation of k 'healed' labels."""
    combos = list(itertools.combinations(range(n), k))
    A = np.zeros((len(combos), n), bool)
    for r, c in enumerate(combos):
        A[r, list(c)] = True
    return A


def group_diff(values: np.ndarray, A: np.ndarray) -> np.ndarray:
    """values (n,) or (n,K); returns mean(healed)-mean(non-healed) per allocation."""
    v = np.asarray(values, float)
    nh = A.sum(1)[0]; nn = A.shape[1] - nh
    Af = A.astype(float)
    if v.ndim == 1:
        return Af @ v / nh - (1 - Af) @ v / nn
    return Af @ v / nh - (1 - Af) @ v / nn


def group_means(values: np.ndarray, A: np.ndarray):
    v = np.asarray(values, float)
    nh = A.sum(1)[0]; nn = A.shape[1] - nh
    Af = A.astype(float)
    return Af @ v / nh, (1 - Af) @ v / nn


def exact_p_two_sided(null: np.ndarray, obs: float) -> float:
    return float(np.mean(np.abs(null) >= abs(obs) - TOL))


def exact_p_greater(null: np.ndarray, obs: float) -> float:
    return float(np.mean(null >= obs - TOL))


def observed_index(A: np.ndarray, healed: np.ndarray) -> int:
    idx = np.flatnonzero((A == healed[None, :]).all(1))
    assert len(idx) == 1
    return int(idx[0])


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> float:
    """U for x>y with half credit for ties."""
    x = np.asarray(x, float)[:, None]; y = np.asarray(y, float)[None, :]
    return float((x > y).sum() + 0.5 * (x == y).sum())


def exact_rank_test(values: np.ndarray, healed: np.ndarray, A: np.ndarray) -> dict:
    """Exact two-sided Mann-Whitney / AUC by enumeration of allocations A."""
    v = np.asarray(values, float)
    nh = int(healed.sum()); nn = len(v) - nh
    us = np.array([mann_whitney_u(v[a], v[~a]) for a in A])
    u = mann_whitney_u(v[healed], v[~healed])
    center = nh * nn / 2
    p = float(np.mean(np.abs(us - center) >= abs(u - center) - TOL))
    diffs = np.sort((v[healed][:, None] - v[~healed][None, :]).ravel())
    hl = float(np.median(diffs))
    # exact CI by inversion of the U distribution (Lehmann): k = number of
    # extreme pairwise differences trimmed from each side.
    sorted_u = np.sort(us)
    alpha = 0.05
    k = 0
    for kk in range(1, len(diffs) // 2 + 1):
        # P(U <= kk-1) under the null
        if np.mean(us <= kk - 1 + TOL) <= alpha / 2:
            k = kk
        else:
            break
    ci = [float(diffs[k - 1]), float(diffs[len(diffs) - k])] if k >= 1 else [None, None]
    achieved = float(1 - 2 * np.mean(us <= k - 1 + TOL)) if k >= 1 else None
    return {"U": u, "AUC": u / (nh * nn), "p_exact_two_sided": p,
            "hodges_lehmann": hl, "hl_ci": ci, "hl_ci_coverage": achieved,
            "n_healed": nh, "n_nonhealed": nn, "n_allocations": len(A)}
