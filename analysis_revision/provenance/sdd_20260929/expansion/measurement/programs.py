"""Program score families: frozen topics, NMF proportions and module scores, plus the lineage gate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..population.represent import log_normalize


def gate_lineages(counts, genes) -> np.ndarray:
    """Exclusive maximum-score lineage labels from the project's marker panel (log1p CP10K z-scores)."""
    from core_engines.celltype_markers import score_cell_types

    labels, _scores, _found = score_cell_types(sp.csr_matrix(counts), np.asarray(genes, dtype=object))
    return np.asarray(labels, dtype=object)


def topic_theta(counts, genes, discovery_dir):
    """Deterministic softmax(mu) projection through the frozen discovery encoder."""
    from core_engines.gse241132 import load_discovery, project_latent

    panel, _beta, model, _k, n_panel, device = load_discovery(str(Path(discovery_dir)))
    theta, covered = project_latent(sp.csr_matrix(counts), np.asarray(genes, dtype=object), panel, model, device,
                                    n_panel, key="theta")
    return theta.astype(np.float64), int(covered)


def top_genes_per_topic(beta, panel, n: int = 30) -> dict:
    beta = np.asarray(beta)
    panel = np.asarray(panel, dtype=object)
    return {k: [str(g) for g in panel[np.argsort(-beta[k])[:n]]] for k in range(beta.shape[0])}


def _aligned(counts, genes, wanted) -> sp.csr_matrix:
    index = pd.Index(np.asarray(genes, dtype=object).astype(str)).get_indexer(np.asarray(wanted).astype(str))
    present = np.flatnonzero(index >= 0)
    counts = sp.csr_matrix(counts)
    block = sp.lil_matrix((counts.shape[0], len(wanted)), dtype=np.float32)
    if present.size:
        block[:, present] = counts[:, index[present]]
    return block.tocsr()


def fit_nmf(counts, genes, *, n_components: int = 15, n_hvg: int = 2000, seed: int = 0, max_iter: int = 400) -> dict:
    from sklearn.decomposition import NMF

    logged = log_normalize(counts)
    mean = np.asarray(logged.mean(axis=0)).ravel()
    variance = np.asarray(logged.multiply(logged).mean(axis=0)).ravel() - mean ** 2
    keep = np.sort(np.argsort(-variance, kind="stable")[: min(n_hvg, int((variance > 0).sum()))])
    model = NMF(n_components=n_components, init="nndsvda", random_state=seed, max_iter=max_iter)
    model.fit(logged[:, keep])
    return {"model": model, "genes": np.asarray(genes, dtype=object)[keep].astype(str)}


def nmf_proportions(fitted: dict, counts, genes) -> np.ndarray:
    logged = log_normalize(_aligned(counts, genes, fitted["genes"]))
    weights = fitted["model"].transform(logged)
    totals = weights.sum(axis=1, keepdims=True)
    uniform = np.full_like(weights, 1.0 / weights.shape[1])
    return np.where(totals > 0, weights / np.where(totals > 0, totals, 1.0), uniform)


def module_scores(counts, genes, gene_sets: dict, *, n_control: int = 50, n_bins: int = 25, seed: int = 0) -> pd.DataFrame:
    """Mean log-normalized expression of a set minus expression-matched random control genes."""
    logged = log_normalize(counts)
    genes = np.asarray(genes, dtype=object).astype(str)
    average = np.asarray(logged.mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(average).rank(method="first"), q=min(n_bins, len(average)), labels=False).to_numpy()
    rng = np.random.default_rng(seed)
    lookup = {g: i for i, g in enumerate(genes)}
    out = {}
    for name, members in gene_sets.items():
        chosen = [lookup[g] for g in members if g in lookup]
        if not chosen:
            out[name] = np.full(logged.shape[0], np.nan)
            continue
        controls = []
        for index in chosen:
            pool = np.setdiff1d(np.flatnonzero(bins == bins[index]), chosen)
            if pool.size:
                controls.extend(rng.choice(pool, size=min(n_control, pool.size), replace=False).tolist())
        target = np.asarray(logged[:, chosen].mean(axis=1)).ravel()
        background = np.asarray(logged[:, controls].mean(axis=1)).ravel() if controls else 0.0
        out[name] = target - background
    return pd.DataFrame(out)
