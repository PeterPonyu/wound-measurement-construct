"""Dataset adapters producing the standardized AnnData contract, plus a read-only probe."""
from __future__ import annotations

import os
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .formats import (GSM_RE, open_binary, read_dense_table_sparse, read_h5_directory,
                      read_mtx_with_identifiers, read_table)
from .standard import make_standard

ZERO_TIME_WORDS = {"skin", "unwounded", "unwound", "intact", "uninjured", "baseline", "uw", "control", "ctrl"}
GSE241132_DAYS = {"Skin": 0.0, "Wound1": 1.0, "Wound7": 7.0, "Wound30": 30.0}


class FieldResolutionError(ValueError):
    """A required field could not be found; the message lists the available columns."""


def parse_time_value(label, mapping=None):
    if mapping is not None and str(label) in mapping:
        return float(mapping[str(label)])
    text = str(label).strip().lower()
    if text in ZERO_TIME_WORDS:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def resolve_column(frame: pd.DataFrame, candidates) -> str:
    columns = list(frame.columns)
    lowered = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if str(candidate).lower() in lowered:
            return lowered[str(candidate).lower()]
    for candidate in candidates:
        hits = [c for c in columns if str(candidate).lower() in str(c).lower()]
        if len(hits) == 1:
            return hits[0]
    raise FieldResolutionError(f"none of {list(candidates)} matched; available columns: {columns}")


def qc_mask(counts, genes, *, min_genes: int = 200, max_mito: float = 0.2) -> np.ndarray:
    counts = sp.csr_matrix(counts)
    detected = np.diff(counts.indptr)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    mito = np.array([str(g).upper().startswith("MT-") for g in genes])
    if mito.any():
        fraction = np.asarray(counts[:, np.flatnonzero(mito)].sum(axis=1)).ravel() / np.maximum(totals, 1.0)
    else:
        fraction = np.zeros(counts.shape[0])
    return (detected >= min_genes) & (fraction <= max_mito)


def synthetic_dataset(*, dataset_id: str = "SYN", n_individuals: int = 4, times=(0.0, 1.0, 2.0, 3.0),
                      cells: int = 80, genes: int = 40, mode: str = "translation", seed: int = 0):
    """Poisson counts from a 3-D latent process with known dynamics (tests and smoke runs)."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 0.8, size=(3, genes))
    base = rng.normal(0.0, 0.3, size=genes) + 0.5
    rows, obs = [], []
    for i in range(n_individuals):
        offset = rng.normal(0.0, 0.25, size=3)
        for k, t in enumerate(times):
            frac = k / max(len(times) - 1, 1)
            z = rng.normal(0.0, 0.45, size=(cells, 3)) + offset
            if mode == "translation":
                z[:, 0] += 2.0 * frac
            elif mode == "reweighting":
                z[rng.random(cells) < 0.1 + 0.8 * frac, 1] += 2.5
            elif mode == "rotation":
                z[:, 0] += 2.0 * np.cos(np.pi / 2 * frac)
                z[:, 1] += 2.0 * np.sin(np.pi / 2 * frac)
            else:
                raise ValueError(f"unknown synthetic mode {mode!r}")
            rows.append(rng.poisson(np.exp(np.clip(base + 0.5 * z @ loadings, -3.0, 4.0))))
            obs.extend({"individual": f"ind{i}", "time": f"t{float(t):g}", "time_value": float(t),
                        "sample": f"ind{i}_t{float(t):g}"} for _ in range(cells))
    names = np.array([f"G{g}" for g in range(genes)], dtype=object)
    return make_standard(np.vstack(rows).astype(np.float32), names, pd.DataFrame(obs), dataset_id=dataset_id,
                         provenance={"synthetic": True, "mode": mode, "seed": seed})


def _labels_from_names(names, pattern: str) -> pd.DataFrame:
    regex = re.compile(pattern)
    rows = {}
    for name in names:
        match = regex.search(Path(name).name)
        if match and "gsm" in match.groupdict():
            rows[match.group("gsm")] = match.groupdict()
    if not rows:
        raise FieldResolutionError(f"label_regex {pattern!r} matched no file names")
    return pd.DataFrame(list(rows.values()))


def load_cohort(kind: str, base: Path, args: dict):
    """Return (counts cells x genes, genes, obs) for a generic adapter kind."""
    from core_engines import data_adapter as da

    base = Path(base)
    series = str(base / args["series_matrix"]) if args.get("series_matrix") else None
    policy = args.get("gene_policy", "union")
    names: list[str] = []
    if kind in {"tar_triplets", "tar_dense_csv"}:
        archive = base / args["archive"]
        with tarfile.open(archive) as handle:
            names = handle.getnames()
        loader = da.load_10x_mtx_tar if kind == "tar_triplets" else da.load_dense_csv_tar
        cohort = loader(str(archive), series_matrix=series, gene_policy=policy)
    elif kind in {"dir_triplets", "zip_triplets", "dir_h5"}:
        directory = base / args.get("directory", ".")
        names = sorted(os.listdir(directory))
        if kind == "dir_triplets":
            cohort = da.load_10x_mtx_dir(str(directory), series_matrix=series, gene_policy=policy)
        elif kind == "zip_triplets":
            cohort = da.load_10x_zip_dir(str(directory), series_matrix=series, gene_policy=policy)
        else:
            cohort = read_h5_directory(directory, pattern=args.get("pattern", r"\.h5$"), series_matrix=series,
                                       gene_policy=policy)
    elif kind == "mtx_with_cell_metadata":
        counts, genes, cells = read_mtx_with_identifiers(
            base / args["matrix"], base / args["genes"], base / args["cells"], gene_column=args.get("gene_column", 0),
            genes_header=args.get("genes_header", False), cells_header=args.get("cells_header", False))
        return counts, genes, _join_cell_metadata(cells, base, args)
    elif kind == "dense_table_with_cell_metadata":
        counts, genes, cells = read_dense_table_sparse(base / args["table"], genes_in_rows=args.get("genes_in_rows", True))
        return counts, genes, _join_cell_metadata(cells, base, args)
    elif kind == "h5ad":
        import anndata

        adata = anndata.read_h5ad(str(base / args["file"]))
        if args.get("use_raw"):
            counts, genes = adata.raw.X, adata.raw.var_names.to_numpy(dtype=object)
        else:
            counts = adata.layers[args["layer"]] if args.get("layer") else adata.X
            genes = adata.var_names.to_numpy(dtype=object)
        return sp.csr_matrix(counts, dtype=np.float32), genes, adata.obs.reset_index()
    else:
        raise ValueError(f"unknown adapter kind {kind!r}")
    obs = cohort.obs.copy()
    obs["barcode"] = [str(c).split(":", 1)[1] if ":" in str(c) else str(c) for c in cohort.cells]
    if args.get("label_regex"):
        labels = _labels_from_names(names, args["label_regex"])
        obs = obs.merge(labels, on="gsm", how="left")
    return sp.csr_matrix(cohort.X, dtype=np.float32), cohort.genes, obs


def _join_cell_metadata(cells, base: Path, args: dict) -> pd.DataFrame:
    meta = read_table(base / args["cell_metadata"], dtype=str)
    key = resolve_column(meta, args.get("cell_key", ["cell", "barcode", "cell_id", "cell_name"]))
    meta = meta.drop_duplicates(key).set_index(key)
    joined = meta.reindex(pd.Index([str(c) for c in cells]))
    coverage = float(joined.notna().any(axis=1).mean()) if len(joined) else 0.0
    if coverage < float(args.get("min_join_fraction", 0.95)):
        raise FieldResolutionError(f"cell metadata covers {coverage:.1%} of matrix cells")
    joined.index.name = "cell"
    return joined.reset_index()


def _finish(counts, genes, obs: pd.DataFrame, *, entry_id: str, args: dict, provenance: dict):
    fields = args.get("fields", {})
    frame = obs.reset_index(drop=True)
    out = pd.DataFrame(index=frame.index)
    if "individual" in fields:
        out["individual"] = frame[resolve_column(frame, fields["individual"])].astype(str)
    elif args.get("individual_from") in frame.columns:
        out["individual"] = frame[args["individual_from"]].astype(str)
    else:
        raise FieldResolutionError("adapter_args needs 'fields.individual' candidates or 'individual_from'")
    if "constant_time" in args:
        out["time"] = f"t{float(args['constant_time']):g}"
        out["time_value"] = float(args["constant_time"])
    else:
        column = resolve_column(frame, fields.get("time", ["time", "timepoint", "day"]))
        out["time"] = frame[column].astype(str)
        mapping = args.get("time_mapping")
        values = out["time"].map(lambda label: parse_time_value(label, mapping))
        if values.isna().any():
            bad = sorted(out.loc[values.isna(), "time"].unique())[:10]
            raise FieldResolutionError(f"unparseable time labels {bad}; add adapter_args.time_mapping")
        out["time_value"] = values.astype(float)
    sample = args.get("sample_from", "gsm")
    out["sample"] = (frame[sample].astype(str) if sample in frame.columns
                     else out["individual"] + "_" + out["time"])
    for optional in ("condition", "cell_type", "batch"):
        if optional in fields:
            out[optional] = frame[resolve_column(frame, fields[optional])].astype(str)
    keep = np.ones(len(out), dtype=bool)
    population = args.get("population") or {}
    if population.get("cell_type_keep"):
        if "cell_type" not in out.columns:
            raise FieldResolutionError("population.cell_type_keep needs fields.cell_type")
        keep &= out["cell_type"].isin(population["cell_type_keep"]).to_numpy()
    for column, allowed in (args.get("filters") or {}).items():
        keep &= out[column].astype(str).isin([str(v) for v in allowed]).to_numpy()
    qc = args.get("qc", {"min_genes": 200, "max_mito": 0.2})
    if qc:
        keep &= qc_mask(counts, genes, min_genes=qc.get("min_genes", 200), max_mito=qc.get("max_mito", 0.2))
    index = np.flatnonzero(keep)
    provenance = dict(provenance, n_cells_loaded=int(len(out)), n_cells_kept=int(index.size), qc=qc,
                      population=population)
    return make_standard(sp.csr_matrix(counts)[index], genes, out.iloc[index].reset_index(drop=True),
                         dataset_id=entry_id, count_type=args.get("count_type", "raw"),
                         time_unit=args.get("time_unit", "day"), provenance=provenance)


def ingest_local_gse241132(root, args: dict):
    """Author-annotated GSE241132 cells through the verified barcode-row contract."""
    from core_engines.gse241132 import load_annotated_counts, load_discovery, project_latent

    raw = Path(root) / "data/raw/GSE241132"
    cohort, obs, matrix = load_annotated_counts(str(raw / "per_gsm"), str(raw / "GSE241132_cell_metadata.txt.gz"))
    frame = pd.DataFrame({"individual": obs["donor"].astype(str).to_numpy(),
                          "time": obs["cond"].astype(str).to_numpy(),
                          "time_value": obs["cond"].map(GSE241132_DAYS).to_numpy(dtype=float),
                          "sample": obs["gsm"].astype(str).to_numpy(),
                          "cell_type": obs["celltype"].astype(str).to_numpy()})
    keep_types = (args.get("population") or {}).get("cell_type_keep", ["Fibroblast"])
    keep = np.flatnonzero(frame["cell_type"].isin(keep_types).to_numpy())
    counts = sp.csr_matrix(matrix)[keep]
    adata = make_standard(counts, cohort.genes, frame.iloc[keep].reset_index(drop=True), dataset_id="GSE241132",
                          provenance={"adapter": "local_gse241132", "cell_types": keep_types,
                                      "n_cells_joined": int(len(obs))})
    if args.get("frozen_topic_mu"):
        panel, _beta, model, _k, n_panel, device = load_discovery(str(Path(root) / "outputs/direction1_dfu_mainline"))
        latent, covered = project_latent(counts, cohort.genes, panel, model, device, n_panel, key="mu")
        adata.obsm["X_frozen_mu"] = latent.astype(np.float32)
        adata.uns["standard"]["frozen_panel_genes_covered"] = int(covered)
    return adata


def ingest_dataset(entry, root):
    """Dispatch on entry.adapter and return a standardized AnnData."""
    args = dict(getattr(entry, "adapter_args", {}) or {})
    if entry.adapter == "local_gse241132":
        return ingest_local_gse241132(root, args)
    base = Path(root) / entry.local_path
    counts, genes, obs = load_cohort(entry.adapter, base, args)
    return _finish(counts, genes, obs, entry_id=entry.id, args=args,
                   provenance={"adapter": entry.adapter, "local_path": str(entry.local_path)})


def probe_dataset(base, *, max_values: int = 20, max_table_bytes: int = 200_000_000) -> dict:
    """Read-only structural report of whatever files exist under ``base``."""
    base = Path(base)
    report = {"base": str(base), "exists": base.exists(), "files": []}
    for path in sorted(base.rglob("*")) if base.exists() else []:
        if not path.is_file():
            continue
        name = path.name.lower()
        entry = {"name": str(path.relative_to(base)), "size": path.stat().st_size}
        try:
            if name.endswith(".tar"):
                with tarfile.open(path) as archive:
                    members = archive.getnames()
                entry["n_members"] = len(members)
                entry["members_head"] = members[:40]
                entry["gsm_ids"] = sorted({m.group(1) for m in map(GSM_RE.search, members) if m})[:200]
            elif "series_matrix" in name:
                from core_engines.data_adapter import parse_series_matrix

                pheno = parse_series_matrix(str(path))
                entry["n_samples"] = int(len(pheno))
                entry["sample_fields"] = {str(c): sorted(map(str, pheno[c].dropna().unique()))[:max_values]
                                          for c in pheno.columns}
            elif name.endswith(".h5ad"):
                import anndata

                adata = anndata.read_h5ad(str(path), backed="r")
                entry["h5ad"] = {"shape": list(adata.shape), "layers": list(adata.layers.keys()),
                                 "raw": adata.raw is not None,
                                 "obs": {str(c): sorted(map(str, adata.obs[c].unique()))[:max_values]
                                         for c in adata.obs.columns}}
            elif name.endswith((".mtx", ".mtx.gz")):
                with open_binary(path) as handle:
                    entry["mtx_header"] = [handle.readline().decode(errors="replace").strip() for _ in range(3)]
            elif name.endswith((".tsv", ".tsv.gz", ".csv", ".csv.gz", ".txt", ".txt.gz", ".zip")):
                if path.stat().st_size <= max_table_bytes:
                    head = read_table(path, nrows=5, dtype=str)
                    entry["columns"] = [str(c) for c in head.columns][:80]
                    entry["n_columns"] = int(head.shape[1])
                    entry["head"] = head.iloc[:, :12].values.tolist()
                else:
                    from .formats import open_text

                    with open_text(path) as handle:
                        header = handle.readline().rstrip("\n")
                    separator = "\t" if header.count("\t") >= header.count(",") else ","
                    fields = header.split(separator)
                    entry["n_columns"] = len(fields)
                    entry["columns"] = fields[:40]
        except Exception as error:  # the probe records failures instead of stopping
            entry["probe_error"] = f"{type(error).__name__}: {error}"
        report["files"].append(entry)
    return report
