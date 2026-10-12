"""Representations fitted on permitted training cells; the geometry is shared by all methods in a task."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


def log_normalize(counts, target: float = 1e4) -> sp.csr_matrix:
    counts = sp.csr_matrix(counts, dtype=np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scaled = sp.csr_matrix(sp.diags((target / totals).astype(np.float32)) @ counts)
    scaled.data = np.log1p(scaled.data)
    return scaled


@dataclass
class Representation:
    kind: str
    params: dict = field(default_factory=dict)
    genes: np.ndarray | None = None
    gene_mean: np.ndarray | None = None
    gene_scale: np.ndarray | None = None
    components: np.ndarray | None = None
    coord_mean: np.ndarray | None = None
    coord_scale: np.ndarray | None = None

    def transform_counts(self, counts, chunk: int = 20_000) -> np.ndarray:
        counts = sp.csr_matrix(counts)
        blocks = []
        for start in range(0, counts.shape[0], chunk):
            dense = log_normalize(counts[start:start + chunk])[:, self.genes].toarray()
            blocks.append(((dense - self.gene_mean) / self.gene_scale) @ self.components.T)
        return ((np.vstack(blocks) - self.coord_mean) / self.coord_scale).astype(np.float32)

    def transform_latent(self, latent) -> np.ndarray:
        return ((np.asarray(latent, dtype=np.float64) - self.coord_mean) / self.coord_scale).astype(np.float32)


def fit_pca(counts_train, *, n_hvg: int = 2000, n_components: int = 30, max_fit_cells: int = 50_000,
            seed: int = 0) -> Representation:
    from sklearn.decomposition import PCA

    counts_train = sp.csr_matrix(counts_train)
    n_cells = counts_train.shape[0]
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n_cells, size=max_fit_cells, replace=False)) if n_cells > max_fit_cells else np.arange(n_cells)
    logged = log_normalize(counts_train[rows])
    mean = np.asarray(logged.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(logged.multiply(logged).mean(axis=0)).ravel() - mean ** 2, 0.0)
    order = np.argsort(-variance, kind="stable")
    genes = np.sort(order[: min(int(n_hvg), int((variance > 0).sum()))])
    if genes.size < 2:
        raise ValueError("fewer than two variable genes among the training cells")
    scale = np.sqrt(variance[genes])
    scale[scale == 0] = 1.0
    block = (logged[:, genes].toarray() - mean[genes]) / scale
    dimension = int(min(n_components, block.shape[0] - 1, block.shape[1]))
    solver = "randomized" if min(block.shape) > 500 else "full"
    pca = PCA(n_components=dimension, svd_solver=solver, random_state=seed).fit(block)
    rep = Representation("pca_train", {"n_hvg": int(genes.size), "n_components": dimension,
                                       "n_fit_cells": int(rows.size), "seed": int(seed)},
                         genes, mean[genes], scale, pca.components_.astype(np.float64))
    projected = block @ rep.components.T
    rep.coord_mean = projected.mean(axis=0)
    spread = projected.std(axis=0)
    spread[spread == 0] = 1.0
    rep.coord_scale = spread
    return rep


def fit_standardizer(latent_train) -> Representation:
    latent_train = np.asarray(latent_train, dtype=np.float64)
    mean = latent_train.mean(axis=0)
    scale = latent_train.std(axis=0)
    scale[scale == 0] = 1.0
    return Representation("obsm_standardized", {"n_components": int(latent_train.shape[1])},
                          coord_mean=mean, coord_scale=scale)


def representation_arrays(rep: Representation) -> dict:
    """Portable numeric state, never a pickled estimator."""
    names = ("genes", "gene_mean", "gene_scale", "components", "coord_mean", "coord_scale")
    return {name: np.asarray(getattr(rep, name)).copy() for name in names if getattr(rep, name) is not None}


def restore_representation(kind: str, params: dict, arrays: dict) -> Representation:
    required = {"coord_mean", "coord_scale"}
    if kind == "pca_train":
        required |= {"genes", "gene_mean", "gene_scale", "components"}
    elif kind != "obsm_standardized":
        raise ValueError("unknown saved representation")
    if set(arrays) != required or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("saved representation has missing, extra or nonfinite numeric state")
    mean, scale = arrays["coord_mean"], arrays["coord_scale"]
    if mean.ndim != 1 or mean.size == 0 or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("invalid saved coordinate standardizer")
    if kind == "pca_train":
        genes = arrays["genes"]
        if (genes.ndim != 1 or genes.dtype.kind not in "iu" or np.any(genes < 0)
                or len(set(genes.tolist())) != len(genes)
                or arrays["gene_mean"].shape != genes.shape or arrays["gene_scale"].shape != genes.shape
                or np.any(arrays["gene_scale"] <= 0)
                or arrays["components"].shape != (mean.size, genes.size)):
            raise ValueError("invalid saved PCA transform")
    return Representation(kind, dict(params), **{k: v.copy() for k, v in arrays.items()})


def task_coordinates(adata, fit_mask, rep_config: dict, *, seed: int = 0, return_state: bool = False):
    fit_mask = np.asarray(fit_mask, dtype=bool)
    kind = rep_config["kind"]
    if kind == "pca_train":
        rep = fit_pca(sp.csr_matrix(adata.X)[np.flatnonzero(fit_mask)], n_hvg=rep_config.get("n_hvg", 2000),
                      n_components=rep_config.get("n_components", 30),
                      max_fit_cells=rep_config.get("max_fit_cells", 50_000), seed=seed)
        coords = rep.transform_counts(adata.X)
    elif kind == "obsm_standardized":
        latent = np.asarray(adata.obsm[rep_config["key"]], dtype=np.float64)
        rep = fit_standardizer(latent[fit_mask])
        coords = rep.transform_latent(latent)
    else:
        raise ValueError(f"unknown representation kind {kind!r}")
    result = (coords, {"kind": kind, **rep.params, "n_fit_mask": int(fit_mask.sum())})
    return (*result, rep) if return_state else result
