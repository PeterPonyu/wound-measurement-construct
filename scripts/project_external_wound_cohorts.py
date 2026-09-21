#!/usr/bin/env python3
"""Project external wound cohorts."""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import library_normalize, mito_fraction, parse_series_matrix
from wound_models.topic_model import TopicModel
FIBRO_MARKERS = ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'PDGFRA', 'THY1']
IMMUNE_MARKERS = ['PTPRC', 'CD3D', 'CD3E', 'TRBC1', 'MS4A1', 'CD79A', 'CD74', 'LYZ', 'FCER1G', 'TYROBP', 'NKG7', 'MZB1', 'JCHAIN']
B_PLASMA_MARKERS = ['MS4A1', 'CD79A', 'CD74', 'CD37', 'CD79B', 'MZB1', 'JCHAIN', 'SDC1', 'CD38']

def decode(values) -> np.ndarray:
    return np.asarray([v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values], dtype=object)

def load_10x_h5(path: str):
    """Load a 10x Genomics H5 file as cells x genes CSR counts."""
    with h5py.File(path, 'r') as h5:
        g = h5['matrix']
        shape = tuple((int(x) for x in g['shape'][()]))
        data = np.asarray(g['data'][()], dtype=np.int32)
        indices = np.asarray(g['indices'][()], dtype=np.int64)
        indptr = np.asarray(g['indptr'][()], dtype=np.int64)
        genes = decode(g['features/name'][()])
        barcodes = decode(g['barcodes'][()])
        feature_type = decode(g['features/feature_type'][()])
    if feature_type.size == genes.size:
        keep = feature_type == 'Gene Expression'
        if not keep.all():
            csc = sp.csc_matrix((data, indices, indptr), shape=shape)
            csc = csc[keep, :]
            genes = genes[keep]
            return (csc.T.tocsr(), genes, barcodes)
    csc = sp.csc_matrix((data, indices, indptr), shape=shape)
    return (csc.T.tocsr(), genes, barcodes)

def collapse_gene_duplicates(X: sp.csr_matrix, genes: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Sum duplicate gene symbols so marker lookup and panel projection agree."""
    unique_genes, inverse = np.unique(genes.astype(str), return_inverse=True)
    if unique_genes.size == genes.size:
        return (X, genes.astype(object))
    coo = X.tocoo()
    out = sp.csr_matrix((coo.data, (coo.row, inverse[coo.col])), shape=(X.shape[0], unique_genes.size), dtype=np.int32)
    return (out, unique_genes.astype(object))

def panel_matrix(X: sp.csr_matrix, genes: np.ndarray, panel: np.ndarray) -> tuple[sp.csr_matrix, int]:
    """Reindex counts onto the frozen panel, zero-filling absent genes."""
    X, genes = collapse_gene_duplicates(X, genes)
    pos = pd.Index(genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    if not present.size:
        raise ValueError('external H5 file has zero overlap with frozen panel')
    sub = X[:, pos[present]].tocoo()
    out = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(X.shape[0], panel.size), dtype=np.int32)
    return (out, int(present.size))

def patient_proxy(title: str) -> str:
    """Return a display-only proxy by removing a trailing replicate letter."""
    m = re.search('(?i)(diac\\d+)', str(title))
    return m.group(1).upper() if m else str(title).upper()

def title_sample_key(title: str) -> str:
    """Map DIAC007/DIAC019B-style titles to the author's 07-d/19b-n key."""
    m = re.search('(?i)diac(\\d+)([a-z]*)', str(title))
    if not m:
        return ''
    return f'{int(m.group(1)):02d}{m.group(2).lower()}'

def load_author_sample_metadata(path: str | None) -> dict[str, dict]:
    """Read only author-provided sample metadata from an optional GSE223964 h5ad."""
    if not path or not os.path.exists(path):
        return {}
    import anndata as ad
    adata = ad.read_h5ad(path, backed='r')
    if 'sample' not in adata.obs.columns:
        return {}
    rows = {}
    grouped = adata.obs.groupby('sample', observed=True)
    for sample, frame in grouped:
        key = re.sub('[^a-z0-9]', '', str(sample).lower())
        row = {'author_sample': str(sample), 'author_cells': int(len(frame))}
        for col in ['age', 'sex', 'condition']:
            if col in frame.columns:
                row[col] = str(frame[col].iloc[0])
        rows[key] = row
    return rows

def marker_fraction(X: sp.csr_matrix, genes: np.ndarray, markers: list[str]) -> np.ndarray:
    pos = pd.Index(genes).get_indexer(markers)
    pos = pos[pos >= 0]
    if len(pos) == 0:
        return np.zeros(X.shape[0], dtype=bool)
    return np.asarray((X[:, pos] > 0).sum(axis=1)).ravel() > 0

def marker_count(X: sp.csr_matrix, genes: np.ndarray, markers: list[str]) -> np.ndarray:
    pos = pd.Index(genes).get_indexer(markers)
    pos = pos[pos >= 0]
    if len(pos) == 0:
        return np.zeros(X.shape[0], dtype=np.int16)
    return np.asarray((X[:, pos] > 0).sum(axis=1)).ravel().astype(np.int16)

def project(model, Xn: sp.csr_matrix, device: str) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, Xn.shape[0], 4096):
            xb = torch.from_numpy(np.asarray(Xn[start:start + 4096].todense(), dtype=np.float32)).to(device)
            out.append(model(xb, deterministic=True)['theta'].cpu().numpy())
    return np.vstack(out)

def run_cohort(*, name: str, directory: str, series_matrix: str, model, panel: np.ndarray, device: str, min_genes: int, max_mito: float, max_cells: int | None, author_meta: dict[str, dict] | None=None) -> dict:
    pheno = parse_series_matrix(series_matrix)
    pheno_by_gsm = pheno.set_index('gsm')
    paths = sorted(glob.glob(os.path.join(directory, '*.h5')))
    if not paths:
        raise FileNotFoundError(f'no per-sample H5 files in {directory}')
    sample_rows = []
    cell_total = 0
    for path in paths:
        basename = os.path.basename(path)
        gsm_match = re.search('(GSM\\d+)', basename)
        if not gsm_match:
            continue
        gsm = gsm_match.group(1)
        X, genes, barcodes = load_10x_h5(path)
        X, genes = collapse_gene_duplicates(X, genes)
        if max_cells is not None and X.shape[0] > max_cells:
            rng = np.random.default_rng(0)
            keep_idx = np.sort(rng.choice(X.shape[0], size=max_cells, replace=False))
            X = X[keep_idx]
            barcodes = barcodes[keep_idx]
        genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
        mito = mito_fraction(X, genes)
        keep = (genes_per_cell >= min_genes) & (mito <= max_mito)
        X = X[keep]
        if X.shape[0] == 0:
            continue
        obs_row = pheno_by_gsm.loc[gsm].to_dict() if gsm in pheno_by_gsm.index else {'gsm': gsm}
        title = str(obs_row.get('title', gsm))
        title_key = re.sub('[^a-z0-9]', '', title_sample_key(title).lower())
        author_row = (author_meta or {}).get(title_key, {})
        if not author_row and author_meta:
            candidates = [v for k, v in author_meta.items() if k.startswith(title_key)]
            if not title_key[-1:].isalpha():
                candidates = [v for k, v in author_meta.items() if k in {title_key + 'n', title_key + 'd'}]
            if len(candidates) == 1:
                author_row = candidates[0]
        panel_X, covered = panel_matrix(X, genes, panel)
        Xn = library_normalize(panel_X)
        theta = project(model, Xn, device)
        fibro_n = marker_count(X, genes, FIBRO_MARKERS)
        immune = marker_fraction(X, genes, IMMUNE_MARKERS)
        bplasma = marker_fraction(X, genes, B_PLASMA_MARKERS)
        fibro = (fibro_n >= 2) & ~marker_fraction(X, genes, ['PTPRC'])
        row = {'cohort': name, 'gsm': gsm, 'title': title, 'patient_proxy': patient_proxy(title), 'disease_or_genotype': obs_row.get('disease', obs_row.get('genotype')), 'author_sample': author_row.get('author_sample'), 'author_age': author_row.get('age'), 'author_sex': author_row.get('sex'), 'author_condition': author_row.get('condition'), 'author_cells': author_row.get('author_cells'), 'cells_raw': int(genes_per_cell.size), 'cells_qc': int(X.shape[0]), 'panel_covered': covered, 'panel_total': int(panel.size), 'panel_fraction': covered / panel.size, 'fibro_cells': int(fibro.sum()), 'immune_fraction': float(immune.mean()), 'b_plasma_fraction': float(bplasma.mean()), 'topic0_all_mean': float(theta[:, 0].mean()), 'topic0_fibro_mean': float(theta[fibro, 0].mean()) if fibro.any() else None, 'fibro_gate': '>=2_fibro_markers_and_PTPRC_negative'}
        sample_rows.append(row)
        cell_total += int(X.shape[0])
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        raise RuntimeError(f'no samples survived in {name}')
    stats = {}
    for outcome in ['immune_fraction', 'b_plasma_fraction']:
        valid = samples[['topic0_fibro_mean', outcome]].dropna()
        if len(valid) >= 3:
            rho, p = spearmanr(valid['topic0_fibro_mean'], valid[outcome])
            stats[outcome] = {'n_samples': int(len(valid)), 'rho': float(rho), 'p_descriptive': float(p), 'interpretation': 'descriptive; sample titles are not authoritative patient IDs'}
        else:
            stats[outcome] = {'n_samples': int(len(valid)), 'rho': None, 'p_descriptive': None}
    duplicate_groups = samples.groupby('patient_proxy')['gsm'].apply(list).loc[lambda x: x.map(len) > 1].to_dict()
    return {'cohort': name, 'source_files': {'series_matrix': {'name': os.path.basename(series_matrix), 'bytes': int(os.path.getsize(series_matrix))}, 'per_sample_h5': [{'name': os.path.basename(p), 'bytes': int(os.path.getsize(p))} for p in paths]}, 'samples': samples.to_dict(orient='records'), 'sample_count': int(len(samples)), 'cells_qc_total': cell_total, 'patient_id_authoritative': False, 'duplicate_title_proxies': duplicate_groups, 'sample_level_associations': stats}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--gse223-dir', default='data/raw/GSE223964')
    ap.add_argument('--gse223-series', default='data/raw/GSE223964/GSE223964_series_matrix.txt.gz')
    ap.add_argument('--gse223-h5ad', default='data/raw/GSE223964/GSE223964_Integrated_all_cells.h5ad')
    ap.add_argument('--gse245-dir', default='data/raw/GSE245703')
    ap.add_argument('--gse245-series', default='data/raw/GSE245703/GSE245703_series_matrix.txt.gz')
    ap.add_argument('--output-dir', default='outputs/analysis/external_dfu_projection')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    ap.add_argument('--max-cells-per-sample', type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    model = TopicModel(input_dim=beta.shape[1], num_topics=beta.shape[0])
    model.load_state_dict(state)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    author_meta = load_author_sample_metadata(args.gse223_h5ad)
    report = {'status': 'completed', 'protocol': {'frozen_discovery_dir': args.discovery_dir, 'frozen_panel_genes': int(panel.size), 'frozen_topics': int(beta.shape[0]), 'projection_deterministic': True, 'qc': {'min_genes_per_cell': args.min_genes_per_cell, 'max_mito': args.max_mito}, 'fibro_gate': '>=2 fibro markers and PTPRC negative', 'immune_axis': 'marker-derived fraction over QC-passing cells', 'patient_id_policy': 'title-derived proxies are descriptive only; no pooled patient inference', 'author_metadata': {'GSE223964': 'sample/age/sex/condition read from author h5ad when available; no cell-type labels in that object', 'GSE245703': 'series-matrix title and genotype only in this run'}, 'gse223964_author_h5ad_loaded': bool(author_meta), 'scope': 'external state projection and alternative immune-axis audit; no healing endpoint'}, 'cohorts': {}}
    for name, d, sm in [('GSE223964', args.gse223_dir, args.gse223_series), ('GSE245703', args.gse245_dir, args.gse245_series)]:
        report['cohorts'][name] = run_cohort(name=name, directory=d, series_matrix=sm, model=model, panel=panel, device=device, min_genes=args.min_genes_per_cell, max_mito=args.max_mito, max_cells=args.max_cells_per_sample, author_meta=author_meta if name == 'GSE223964' else None)
    report['interpretation'] = {'healing_validation': 'not performed: neither external series matrix supplies a healing outcome', 'external_reproduction': 'projection and immune-axis direction are descriptive until authoritative patient IDs and labels are verified'}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: {'sample_count': v['sample_count'], 'cells_qc_total': v['cells_qc_total'], 'associations': v['sample_level_associations']} for k, v in report['cohorts'].items()}, indent=2))
    print(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
