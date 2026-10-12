"""Standardized dataset contract shared by every adapter and task generator."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REQUIRED_OBS = ("individual", "time", "time_value", "sample")


def _obs_problems(obs: pd.DataFrame) -> list[str]:
    problems = [f"missing obs column {c}" for c in REQUIRED_OBS if c not in obs.columns]
    if problems:
        return problems
    for column in ("individual", "time", "sample"):
        text = obs[column].astype("string").str.strip().str.lower()
        if (text.isna() | text.isin(["", "none", "nan", "<na>", "null", "nat"])).any():
            problems.append(f"obs column {column} has missing or blank identifiers")
    values = pd.to_numeric(obs["time_value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        problems.append("time_value must be finite numbers")
    if problems:
        return problems
    frame = obs.assign(time_value=values)
    if (frame.groupby("time", observed=True)["time_value"].nunique() > 1).any():
        problems.append("a time label maps to more than one time_value")
    for column in ("individual", "time_value"):
        if (frame.groupby("sample", observed=True)[column].nunique() > 1).any():
            problems.append(f"a sample maps to more than one {column}")
    return problems


def _standard_problems(adata) -> list[str]:
    """Shared admission checks; scan every stored value with bounded temporary memory."""
    problems = _obs_problems(adata.obs)
    count_type = adata.uns.get("standard", {}).get("count_type", "raw")
    if count_type not in {"raw", "normalized"}:
        problems.append("count_type must be 'raw' or 'normalized'")
    if not adata.n_obs or not adata.n_vars:
        problems.append("expression matrix must have cells and genes")
    data = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X).ravel()
    nonfinite = negative = fractional = False
    for start in range(0, data.size, 200_000):
        block = np.asarray(data[start:start + 200_000], dtype=float)
        finite = np.isfinite(block)
        nonfinite |= not bool(finite.all())
        valid = block[finite]
        negative |= bool(np.any(valid < 0))
        if count_type == "raw":
            fractional |= not np.allclose(valid, np.round(valid), rtol=0, atol=1e-6)
    if nonfinite:
        problems.append("expression values must be finite")
    if negative:
        problems.append("negative expression values")
    if fractional:
        problems.append("raw counts must be integer valued")
    return problems


def make_standard(counts, genes, obs: pd.DataFrame, *, dataset_id: str, count_type: str = "raw",
                  time_unit: str = "day", provenance: dict | None = None):
    import anndata

    counts = sp.csr_matrix(counts, dtype=np.float32)
    genes = np.asarray(genes, dtype=object).astype(str)
    if counts.shape != (len(obs), len(genes)):
        raise ValueError(f"counts {counts.shape} do not match obs={len(obs)} genes={len(genes)}")
    problems = _obs_problems(obs)
    if problems:
        raise ValueError("; ".join(problems))
    if count_type not in {"raw", "normalized"}:
        raise ValueError("count_type must be 'raw' or 'normalized'")
    frame = obs.reset_index(drop=True).copy()
    frame["time_value"] = pd.to_numeric(frame["time_value"], errors="raise").astype(float)
    for column in frame.columns:
        if column != "time_value" and not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype(str)
    for column in ("individual", "time", "sample"):
        frame[column] = frame[column].astype(str)
    frame.index = pd.Index([f"c{i}" for i in range(len(frame))])
    adata = anndata.AnnData(X=counts, obs=frame, var=pd.DataFrame(index=pd.Index(genes)))
    adata.var_names_make_unique()
    adata.uns["standard"] = {"dataset_id": dataset_id, "count_type": count_type, "time_unit": time_unit,
                             "provenance_json": json.dumps(provenance or {}, sort_keys=True, default=str)}
    return adata


def write_standard(adata, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".h5ad", dir=path.parent)
    os.close(handle)
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def load_standard(path):
    import anndata

    return anndata.read_h5ad(str(path))


def validate_standard(adata, *, min_cells: int = 20) -> dict:
    """Population-task contract: structure, counts and eligible task families."""
    obs = adata.obs
    problems = _standard_problems(adata)
    if problems:
        return {"status": "rejected", "problems": problems}
    meta = dict(adata.uns.get("standard", {}))
    count_type = meta.get("count_type", "raw")
    sizes = obs.groupby(["individual", "time_value"], observed=True).size()
    usable = {(str(i), float(t)) for (i, t), n in sizes.items() if n >= min_cells}
    times = sorted({t for _, t in usable})
    individuals = sorted({i for i, _ in usable})
    missing_time = []
    for k in range(1, len(times) - 1):
        a, h, b = times[k - 1], times[k], times[k + 1]
        members = [i for i in individuals if {(i, a), (i, h), (i, b)} <= usable]
        if members:
            missing_time.append({"held_time": h, "source_time": a, "anchor_time": b, "individuals": members})
    pairs = []
    for k in range(len(times) - 1):
        a, b = times[k], times[k + 1]
        pairs.append({"source_time": a, "target_time": b,
                      "individuals": [i for i in individuals if {(i, a), (i, b)} <= usable]})
    lio = sorted({i for p in pairs if len(p["individuals"]) >= 3 for i in p["individuals"]})
    eligible = {"T-MT": bool(missing_time), "T-LIO": len(lio) >= 3, "T-K": len(lio) >= 4}
    if not problems and not (eligible["T-MT"] or eligible["T-LIO"]):
        problems.append("no eligible population task: need an interior time with source and anchor, "
                        "or >=3 individuals sharing a consecutive pair")
    return {"status": "accepted" if not problems else "rejected", "problems": problems,
            "dataset_id": meta.get("dataset_id"), "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
            "min_cells": min_cells, "times": times, "individuals": individuals,
            "group_sizes": {f"{i}|{float(t):g}": int(n) for (i, t), n in sizes.items()},
            "missing_time_tasks": missing_time, "consecutive_pairs": pairs, "lio_individuals": lio,
            "eligible": eligible, "count_type": count_type}


def validate_measurement(adata, *, min_cells: int = 50, min_specimens: int = 5) -> dict:
    """Measurement contract: enough specimens with enough cells."""
    problems = _standard_problems(adata)
    if problems:
        return {"status": "rejected", "problems": problems}
    sizes = adata.obs.groupby("sample", observed=True).size()
    usable = sorted(str(s) for s, n in sizes.items() if n >= min_cells)
    problems = [] if len(usable) >= min_specimens else [
        f"{len(usable)} specimens with >= {min_cells} cells; need {min_specimens}"]
    return {"status": "accepted" if not problems else "rejected", "problems": problems,
            "specimens": usable, "specimen_sizes": {str(k): int(v) for k, v in sizes.items()}}
