"""Readers that complement core_engines.data_adapter for newly registered deposits.

core_engines.data_adapter already assembles per-GSM 10x triplets from tar files,
directories and zip files and per-GSM dense CSVs from tar files.  This module adds
per-GSM CellRanger h5 directories, a single matrix with separate identifier files,
streamed dense tables and generic delimited tables.  Every reader returns cells x
genes counts; orientation is checked against identifier lengths, never assumed.
"""
from __future__ import annotations

import gzip
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

GSM_RE = re.compile(r"(GSM\d+)")


def open_binary(path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def open_text(path):
    path = Path(path)
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        names = [n for n in archive.namelist() if not n.endswith("/") and "__MACOSX" not in n]
        if len(names) != 1:
            archive.close()
            raise ValueError(f"{path}: expected exactly one member, found {len(names)}")
        return io.TextIOWrapper(archive.open(names[0]), encoding="utf-8")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def sniff_separator(line: str) -> str:
    return "\t" if line.count("\t") >= line.count(",") else ","


def read_table(path, **kwargs) -> pd.DataFrame:
    with open_text(path) as handle:
        separator = sniff_separator(handle.readline())
    with open_text(path) as handle:
        return pd.read_csv(handle, sep=separator, **kwargs)


def read_identifiers(path, *, column: int = 0, header: bool = False) -> np.ndarray:
    frame = read_table(path, header=0 if header else None, dtype=str)
    return frame.iloc[:, column].astype(str).to_numpy(dtype=object)


def read_mtx(path, *, strict_counts: bool = False) -> sp.csr_matrix:
    with open_binary(path) as handle:
        matrix = scipy.io.mmread(handle)
        if strict_counts:
            from core_engines.data_adapter import validate_raw_source
            validate_raw_source(matrix.data if sp.issparse(matrix) else matrix)
            matrix = matrix.astype(np.int64)
        result = sp.csr_matrix(matrix)
        if strict_counts:
            validate_raw_source(result.data)
        return result


def orient_cells_by_genes(matrix, n_genes: int, n_cells: int) -> sp.csr_matrix:
    if matrix.shape == (n_genes, n_cells):
        return sp.csr_matrix(matrix.T)
    if matrix.shape == (n_cells, n_genes):
        return sp.csr_matrix(matrix)
    raise ValueError(f"matrix shape {matrix.shape} is neither genes x cells ({n_genes}, {n_cells}) "
                     "nor cells x genes")


def collapse_duplicate_genes(matrix, genes, *, strict_counts: bool = False):
    """Sum columns that share a gene symbol; returns (matrix, unique symbols)."""
    genes = np.asarray(genes, dtype=object).astype(str)
    unique, inverse = np.unique(genes, return_inverse=True)
    if len(unique) == len(genes):
        return sp.csr_matrix(matrix), genes.astype(object)
    indicator = sp.csr_matrix((np.ones(len(genes), dtype=np.int64 if strict_counts else np.float32), (np.arange(len(genes)), inverse)),
                              shape=(len(genes), len(unique)))
    return sp.csr_matrix(sp.csr_matrix(matrix) @ indicator), unique.astype(object)


def read_mtx_with_identifiers(matrix_path, genes_path, cells_path, *, gene_column: int = 0,
                              genes_header: bool = False, cells_header: bool = False, strict_counts: bool = False):
    genes = read_identifiers(genes_path, column=gene_column, header=genes_header)
    cells = read_identifiers(cells_path, column=0, header=cells_header)
    matrix = orient_cells_by_genes(read_mtx(matrix_path, strict_counts=strict_counts), len(genes), len(cells))
    if strict_counts:
        from core_engines.data_adapter import validate_raw_source
        validate_raw_source(matrix.data)
    matrix = matrix.astype(np.int64 if strict_counts else np.float32)
    matrix, genes = collapse_duplicate_genes(matrix, genes, strict_counts=strict_counts)
    if strict_counts:
        validate_raw_source(matrix.data)
    return matrix, genes, cells


def read_dense_table_sparse(path, *, genes_in_rows: bool = True, chunksize: int = 2000, strict_counts: bool = False):
    """Stream a dense expression table into sparse cells x genes counts."""
    with open_text(path) as handle:
        separator = sniff_separator(handle.readline())
    blocks, row_ids, column_ids = [], [], None
    with open_text(path) as handle:
        for chunk in pd.read_csv(handle, sep=separator, index_col=0, chunksize=chunksize):
            if column_ids is None:
                column_ids = chunk.columns.astype(str).to_numpy(dtype=object)
            if strict_counts:
                from core_engines.data_adapter import validate_raw_source
                validate_raw_source(chunk.to_numpy())
            blocks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.int64 if strict_counts else np.float32)))
            row_ids.extend(chunk.index.astype(str))
    if not blocks:
        raise ValueError(f"{path}: empty expression table")
    matrix = sp.vstack(blocks).tocsr()
    rows = np.asarray(row_ids, dtype=object)
    if genes_in_rows:
        matrix, genes = collapse_duplicate_genes(sp.csr_matrix(matrix.T), rows, strict_counts=strict_counts)
        if strict_counts:
            validate_raw_source(matrix.data)
        return matrix, genes, column_ids
    matrix, genes = collapse_duplicate_genes(matrix, column_ids, strict_counts=strict_counts)
    if strict_counts:
        validate_raw_source(matrix.data)
    return matrix, genes, rows


def _decode(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def read_10x_h5(path, *, strict_counts: bool = False):
    """CellRanger HDF5 (v3 ``/matrix`` or v2 ``/<genome>``) -> (cells x genes counts, symbols, barcodes).

    Only 'Gene Expression' features are kept, matching scanpy's default; duplicate symbols are summed.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        if "matrix" in handle:
            group = handle["matrix"]
            names = _decode(group["features"]["name"][:])
            types = _decode(group["features"]["feature_type"][:]) if "feature_type" in group["features"] else None
        else:
            group = handle[next(iter(handle.keys()))]
            names, types = _decode(group["gene_names"][:]), None
        shape = tuple(int(v) for v in group["shape"][:])
        matrix = sp.csc_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape)
        barcodes = _decode(group["barcodes"][:])
    if strict_counts:
        from core_engines.data_adapter import validate_raw_source
        validate_raw_source(matrix.data)
    matrix = sp.csr_matrix(matrix.T, dtype=np.int64 if strict_counts else np.float32)
    if types is not None:
        keep = types == "Gene Expression"
        matrix, names = matrix[:, np.flatnonzero(keep)], names[keep]
    matrix, genes = collapse_duplicate_genes(matrix, names, strict_counts=strict_counts)
    if strict_counts:
        validate_raw_source(matrix.data)
    return matrix, genes, barcodes


def read_h5_directory(directory, *, pattern: str = r"\.h5$", series_matrix=None, gene_policy: str = "union", strict_counts: bool = False):
    """Per-GSM CellRanger h5 files -> core_engines CohortMatrix."""
    from core_engines.data_adapter import _assemble

    per_sample = []
    for path in sorted(Path(directory).iterdir()):
        if not re.search(pattern, path.name):
            continue
        hit = GSM_RE.search(path.name)
        matrix, genes, barcodes = read_10x_h5(path, strict_counts=strict_counts)
        per_sample.append((hit.group(1) if hit else path.stem, matrix, genes, barcodes))
    if not per_sample:
        raise ValueError(f"no h5 files matching {pattern!r} in {directory}")
    return _assemble(per_sample, str(series_matrix) if series_matrix else None, gene_policy, 0.0, [], strict_counts=strict_counts)
