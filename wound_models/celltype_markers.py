#!/usr/bin/env python3
"""Celltype markers."""
from typing import Dict, List, Sequence, Tuple
import numpy as np
import scipy.sparse as sp
MARKER_PANEL: Dict[str, List[str]] = {'keratinocyte': ['KRT14', 'KRT5', 'KRT1', 'KRT10', 'KRT16', 'DMKN', 'SFN', 'KRTDAP'], 'fibroblast': ['COL1A1', 'COL1A2', 'PDGFRA', 'TWIST2', 'COL6A3', 'MMP2', 'CCDC80'], 'myeloid': ['CD68', 'CD163', 'LYZ', 'AIF1', 'ITGAM', 'CD14', 'FCER1G', 'TYROBP'], 't_cell': ['CD3D', 'CD3E', 'CD2', 'IL7R', 'TRBC2', 'CCL5', 'SKAP1'], 'b_plasma': ['MS4A1', 'CD79A', 'MZB1', 'JCHAIN', 'DERL3'], 'endothelial': ['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'EGFL7', 'RAMP2'], 'lymphatic_ec': ['LYVE1', 'PROX1', 'CCL21', 'MMRN1', 'TFF3'], 'pericyte_smc': ['RGS5', 'ACTA2', 'MYH11', 'NOTCH3', 'TAGLN', 'NDUFA4L2'], 'mast': ['TPSAB1', 'TPSB2', 'CPA3', 'MS4A2', 'KIT'], 'melanocyte': ['PMEL', 'MLANA', 'DCT', 'TYRP1'], 'sweat_gland': ['DCD', 'SCGB2A2', 'MUCL1', 'SCGB1D2', 'PIP']}
MARKER_PANEL_MOUSE: Dict[str, List[str]] = {lineage: [g.capitalize() for g in genes] for lineage, genes in MARKER_PANEL.items()}

def marker_overlap_with(top_genes: Sequence[str]) -> Dict[str, List[str]]:
    """Reports which panel markers also appear in `top_genes` (circularity audit)."""
    top = {str(g) for g in top_genes}
    return {lineage: sorted(set(genes) & top) for lineage, genes in MARKER_PANEL.items() if set(genes) & top}

def score_cell_types(X: sp.csr_matrix, genes: np.ndarray, panel: Dict[str, List[str]]=MARKER_PANEL, min_markers: int=3, min_margin: float=0.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Assigns each cell a lineage by mean z-scored marker expression.

    Scores are computed on log1p(CP10K) values, z-scored per gene across cells,
    then averaged within each lineage's marker set. Cells whose best and
    second-best lineage differ by less than `min_margin` are left "ambiguous"
    rather than forced into a call.

    Returns (labels, score_matrix [n_cells, n_lineages], markers_found_per_lineage).
    """
    Xf = sp.csr_matrix(X).astype(np.float32)
    totals = np.asarray(Xf.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    Xf = sp.csr_matrix(sp.diags((10000.0 / totals).astype(np.float32)) @ Xf)
    Xf.data = np.log1p(Xf.data)
    lookup = {str(g): i for i, g in enumerate(genes)}
    lineages = [l for l in panel if sum((g in lookup for g in panel[l])) >= min_markers]
    found = {l: sum((g in lookup for g in panel[l])) for l in panel}
    if not lineages:
        raise ValueError('no lineage had enough markers present in this matrix')
    n_cells = Xf.indptr.size - 1
    scores = np.zeros((n_cells, len(lineages)), dtype=np.float32)
    for j, lineage in enumerate(lineages):
        cols = [lookup[g] for g in panel[lineage] if g in lookup]
        block = np.asarray(Xf[:, cols].todense(), dtype=np.float32)
        mu = block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        scores[:, j] = ((block - mu) / sd).mean(axis=1)
    order = np.argsort(-scores, axis=1)
    best = order[:, 0]
    labels = np.array([lineages[b] for b in best], dtype=object)
    if min_margin > 0 and scores.shape[1] > 1:
        margin = scores[np.arange(n_cells), order[:, 0]] - scores[np.arange(n_cells), order[:, 1]]
        labels[margin < min_margin] = 'ambiguous'
    return (labels, scores, found)
if __name__ == '__main__':
    print('=' * 60)
    print('Self-Testing Cell Type Marker Scorer...')
    rng = np.random.default_rng(0)
    lineages = ['keratinocyte', 'fibroblast', 'myeloid']
    genes = np.array(sum((MARKER_PANEL[l] for l in lineages), []) + ['JUNK1', 'JUNK2'], dtype=object)
    lookup = {g: i for i, g in enumerate(genes)}
    blocks, truth = ([], [])
    for l in lineages:
        n = 60
        m = rng.poisson(0.3, size=(n, len(genes))).astype(np.int32)
        for g in MARKER_PANEL[l]:
            m[:, lookup[g]] += rng.poisson(30, size=n).astype(np.int32)
        blocks.append(m)
        truth.extend([l] * n)
    X = sp.csr_matrix(np.vstack(blocks))
    truth = np.array(truth, dtype=object)
    labels, scores, found = score_cell_types(X, genes)
    acc = float((labels == truth).mean())
    print(f'  markers found per lineage: { {k: v for k, v in found.items() if v}}')
    print(f'  assignment accuracy on synthetic populations: {acc:.1%}')
    assert acc > 0.95, f'marker scorer should recover planted lineages, got {acc:.1%}'
    ov = marker_overlap_with(['TIMP1', 'COL1A1', 'CHI3L1', 'KRT14'])
    assert ov.get('fibroblast') == ['COL1A1'], ov
    assert ov.get('keratinocyte') == ['KRT14'], ov
    print(f'  marker_overlap_with flags contamination: {ov}')
    topic0 = ['TIMP1', 'COL3A1', 'DCN', 'FN1', 'IER3', 'SOD2', 'LUM', 'CHI3L1', 'MMP1', 'COL6A2']
    ov0 = marker_overlap_with(topic0)
    assert 'fibroblast' not in ov0, f'fibroblast panel overlaps fibroblast-associated topic: {ov0}'
    print(f"  fibroblast-associated topic top genes overlap no lineage panel: {ov0 or 'none'}")
    print('Cell Type Marker Scorer test PASSED!')
    print('=' * 60)
