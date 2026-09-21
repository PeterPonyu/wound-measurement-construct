#!/usr/bin/env python3
"""Representation checks."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph
from typing import Dict
EDGE_SURVIVAL_MAX_SOURCES = 2000

def build_knn_graph(X: np.ndarray, k: int=15) -> sp.csr_matrix:
    """
    Constructs binary symmetric k-NN adjacency as a sparse CSR matrix.

    Memory is O(N*k) rather than O(N^2): a dense bool adjacency alone is 16.5 GiB
    at N=94k, and the cdist matrix backing it another 66 GiB.
    """
    adj = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.astype(bool).tocsr()

def reachability_survival(ref_adj, cand_adj) -> float:
    """
    Computes fraction of reference-connected pairs that remain reachable in candidate.
    Pair-layer evaluation prevents edge-averaging shatter blindness.

    Evaluated exactly via a component cross-tabulation in O(N) memory: an ordered
    pair (u != v) is same-component in both graphs iff u and v share *both* labels,
    so the surviving count is sum_ij n_ij^2 - N over the ref x cand contingency table.
    """
    _, labels_ref = connected_components(ref_adj, directed=False)
    _, labels_cand = connected_components(cand_adj, directed=False)
    n = labels_ref.shape[0]
    ref_sizes = np.bincount(labels_ref).astype(np.int64)
    n_ref_pairs = int(np.sum(ref_sizes ** 2) - n)
    if n_ref_pairs == 0:
        return 1.0
    n_cand = int(labels_cand.max()) + 1
    joint = labels_ref.astype(np.int64) * n_cand + labels_cand.astype(np.int64)
    joint_sizes = np.bincount(joint).astype(np.int64)
    surviving_pairs = int(np.sum(joint_sizes ** 2) - n)
    return float(surviving_pairs / n_ref_pairs)

def edge_survival(ref_adj, cand_adj, max_hop: int=10, max_sources: int=EDGE_SURVIVAL_MAX_SOURCES, seed: int=0) -> float:
    """
    Measures local distortion: average shortest path hops in candidate graph
    for pairs that were 1-hop neighbours in reference graph.
    Returns normalized score in [0, 1] where 1.0 = perfect 1-hop preservation.

    BFS runs only from the source nodes that actually carry reference edges, in
    row batches, so peak memory is O(batch * N) instead of the O(N^2) dense
    all-pairs matrix. Above `max_sources` nodes the mean is estimated from a
    deterministic random subsample of sources; use audit_surrogate_manifold's
    returned `edge_survival_n_sources` to report that it is an estimate.
    """
    ref_adj = sp.csr_matrix(ref_adj)
    cand_adj = sp.csr_matrix(cand_adj)
    n = ref_adj.indptr.size - 1
    sources = np.flatnonzero(np.diff(ref_adj.indptr) > 0)
    if sources.size == 0:
        return 1.0
    if sources.size > max_sources:
        sources = np.sort(np.random.default_rng(seed).choice(sources, size=max_sources, replace=False))
    total, count = (0.0, 0)
    batch = max(1, min(512, int(200000000.0 // max(n, 1))))
    for start in range(0, sources.size, batch):
        idx = sources[start:start + batch]
        dist = np.asarray(shortest_path(cand_adj, directed=False, unweighted=True, indices=idx))
        for row, u in enumerate(idx):
            nbrs = ref_adj.indices[ref_adj.indptr[u]:ref_adj.indptr[u + 1]]
            hops = dist[row, nbrs]
            total += float(np.sum(np.where(np.isinf(hops), max_hop, hops)))
            count += nbrs.size
        del dist
    mean_hops = total / count
    score = float(np.clip(1.0 - (mean_hops - 1.0) / (max_hop - 1.0), 0.0, 1.0))
    return score

def compute_perplexity(probs: np.ndarray) -> float:
    """Computes mean row perplexity exp(Entropy)."""
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)
    mean_perp = float(np.mean(np.exp(entropy)))
    return mean_perp

def audit_surrogate_manifold(X_ref: np.ndarray, Z_cand: np.ndarray, theta_alloc: np.ndarray, k: int=10) -> Dict[str, float]:
    """
    Runs the complete 4-quantity topological audit battery.
    Args:
        X_ref: Original high-dimensional gene expression matrix [N, G]
        Z_cand: Surrogate low-dimensional embedding coordinates [N, D]
        theta_alloc: Allocation coordinates on simplex [N, K]
    """
    ref_adj = build_knn_graph(X_ref, k=k)
    cand_adj = build_knn_graph(Z_cand, k=k)
    reach = reachability_survival(ref_adj, cand_adj)
    edge_score = edge_survival(ref_adj, cand_adj)
    n_edge_sources = min(int(np.sum(np.diff(ref_adj.indptr) > 0)), EDGE_SURVIVAL_MAX_SOURCES)
    k_components = theta_alloc.shape[1]
    mean_perp = compute_perplexity(theta_alloc)
    near_uniform = mean_perp > 0.95 * k_components
    return {'reachability_survival': reach, 'edge_survival_score': edge_score, 'edge_survival_n_sources': n_edge_sources, 'edge_survival_exact': bool(n_edge_sources < EDGE_SURVIVAL_MAX_SOURCES), 'mean_perplexity': mean_perp, 'near_uniform_warning': bool(near_uniform), 'manifold_intact': bool(reach > 0.7 and (not near_uniform))}
if __name__ == '__main__':
    print('=' * 60)
    print('Self-Testing Manifold Topological Audit Engine...')
    np.random.seed(42)
    n_samples = 150
    t = np.linspace(0, 3 * np.pi, n_samples)
    X_ref = np.column_stack([np.sin(t), np.cos(t), t * 0.5]) + np.random.randn(n_samples, 3) * 0.05
    Z_faithful = np.column_stack([np.sin(t), np.cos(t)]) + np.random.randn(n_samples, 2) * 0.05
    Z_shattered = Z_faithful.copy()
    Z_shattered[n_samples // 2:] += 20.0
    theta_dummy = np.random.dirichlet(np.ones(5) * 0.5, size=n_samples)
    audit_faithful = audit_surrogate_manifold(X_ref, Z_faithful, theta_dummy, k=8)
    audit_shattered = audit_surrogate_manifold(X_ref, Z_shattered, theta_dummy, k=8)
    print('\nFaithful Embedding Audit:')
    for k, v in audit_faithful.items():
        print(f'  - {k}: {v}')
    print('\nShattered Adversary Embedding Audit:')
    for k, v in audit_shattered.items():
        print(f'  - {k}: {v}')
    assert audit_faithful['reachability_survival'] > 0.85, 'Faithful manifold should have high reachability!'
    assert audit_shattered['reachability_survival'] < 0.6, 'Shattered manifold must trigger drop in reachability!'
    print('\nTopological Audit Engine correctly caught the shattered adversary!')
    print('Manifold Audit Engine test PASSED!')
    print('=' * 60)
