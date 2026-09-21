#!/usr/bin/env python3
"""Assess paired anatomy."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.celltype_markers import score_cell_types
from wound_models.data_adapter import library_normalize, load_dense_csv_tar, mito_fraction
from wound_models.topic_model import TopicModel
DEFAULT_MAP = ROOT / 'cohort_metadata' / 'GSE165816_ncomms_subject_sample_map.csv'
DEFAULT_DISCOVERY = ROOT / 'outputs' / 'expression_representation'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'analysis' / 'paired_anatomical_control'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _paired_sign_flip(diffs: np.ndarray) -> dict:
    """Exact two-sided sign-flip test with a finite-sample correction."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    observed = float(diffs.mean())
    if n == 0:
        return {'n_pairs': 0, 'observed_difference': None, 'p_two_sided_exact': None, 'bootstrap_95_ci': [None, None]}
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        null.append(float((diffs * np.asarray(signs)).mean()))
    null = np.asarray(null)
    p = float((1.0 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (len(null) + 1.0))
    rng = np.random.default_rng(0)
    boot = np.asarray([rng.choice(diffs, size=n, replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'n_pairs': int(n), 'observed_difference_foot_minus_forearm': observed, 'p_two_sided_exact_sign_flip': p, 'null_95_ci': [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))], 'bootstrap_95_ci': [float(lo), float(hi)]}

def _panel_matrix(X: sp.csr_matrix, genes: np.ndarray, panel: np.ndarray) -> tuple[sp.csr_matrix, int]:
    pos = pd.Index(genes.astype(str)).get_indexer(panel.astype(str))
    present = np.flatnonzero(pos >= 0)
    if present.size == 0:
        raise RuntimeError('frozen panel has no genes in GSE165816 matrix')
    sub = X[:, pos[present]].tocoo()
    out = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(X.shape[0], len(panel)), dtype=np.int32)
    return (out, int(present.size))

def _project(model: TopicModel, Xn: sp.csr_matrix, device: str, batch_size: int=4096) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, Xn.shape[0], batch_size):
            xb = torch.from_numpy(np.asarray(Xn[start:start + batch_size].todense(), dtype=np.float32)).to(device)
            out.append(model(xb)['theta'].cpu().numpy())
    return np.vstack(out)

def _collapse_patient_sample(frame: pd.DataFrame, tissue: str) -> pd.DataFrame:
    rows = []
    for patient, block in frame[frame['tissue_site'] == tissue].groupby('patient_id', sort=True):
        weights = block['n_cells'].to_numpy(float)
        rows.append({'patient_id': str(patient), 'tissue_site': tissue, 'n_samples': int(len(block)), 'n_cells': int(weights.sum()), 'topic_means': np.average(np.vstack(block['topic_means'].to_numpy()), axis=0, weights=weights).tolist(), 'topic0_all_mean': float(np.average(block['topic0_all_mean'], weights=weights)), 'topic0_fibro_mean': float(np.average(block.loc[block['topic0_fibro_mean'].notna(), 'topic0_fibro_mean'], weights=block.loc[block['topic0_fibro_mean'].notna(), 'n_fibroblasts'])) if block['topic0_fibro_mean'].notna().any() else None, 'n_fibroblasts': int(block['n_fibroblasts'].sum())})
    return pd.DataFrame(rows)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--map-csv', default=str(DEFAULT_MAP))
    ap.add_argument('--discovery-dir', default=str(DEFAULT_DISCOVERY))
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mapping = pd.read_csv(args.map_csv, dtype=str)
    mapping = mapping[mapping['tissue_site'].isin(['foot', 'forearm'])].copy()
    if mapping['sample_id'].duplicated().any():
        raise RuntimeError('author map has duplicate sample IDs')
    map_by_gsm = mapping.set_index('sample_id')
    cohort = load_dense_csv_tar(args.tar, series_matrix=args.series_matrix)
    cell_map = cohort.obs['gsm'].astype(str).map(map_by_gsm['patient_id'])
    keep = cell_map.notna().to_numpy()
    if not keep.any():
        raise RuntimeError('no mapped foot/forearm cells found')
    X = cohort.X[keep]
    obs = cohort.obs.loc[keep, ['gsm']].reset_index(drop=True)
    meta = mapping.set_index('sample_id').loc[obs['gsm'].to_numpy()].reset_index()
    obs = pd.concat([obs, meta[['patient_id', 'tissue_site']]], axis=1)
    mito = mito_fraction(X, cohort.genes)
    gpc = np.asarray((X > 0).sum(axis=1)).ravel()
    qc = (gpc >= args.min_genes_per_cell) & (mito <= args.max_mito)
    X = X[qc]
    obs = obs.iloc[np.flatnonzero(qc)].reset_index(drop=True)
    if X.shape[0] == 0:
        raise RuntimeError('no mapped cells survived fixed QC')
    discovery = Path(args.discovery_dir)
    panel = np.load(discovery / 'panel.npy', allow_pickle=True).astype(object)
    beta = np.load(discovery / 'beta.npy')
    state = torch.load(discovery / 'topic_model.pt', map_location='cpu')
    model = TopicModel(input_dim=beta.shape[1], num_topics=beta.shape[0])
    model.load_state_dict(state)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    Xp, covered = _panel_matrix(X, cohort.genes, panel)
    theta = _project(model, library_normalize(Xp), device)
    labels, _, found = score_cell_types(X, cohort.genes)
    obs['celltype'] = labels
    rows = []
    for gsm, block in obs.groupby('gsm', sort=True):
        idx = block.index.to_numpy()
        t = theta[idx]
        fib = block['celltype'].to_numpy() == 'fibroblast'
        rows.append({'gsm': str(gsm), 'patient_id': str(block['patient_id'].iloc[0]), 'tissue_site': str(block['tissue_site'].iloc[0]), 'n_cells': int(len(idx)), 'n_fibroblasts': int(fib.sum()), 'topic0_all_mean': float(t[:, 0].mean()), 'topic0_fibro_mean': float(t[fib, 0].mean()) if fib.any() else None, 'topic_means': t.mean(axis=0).tolist()})
    samples = pd.DataFrame(rows)
    foot = _collapse_patient_sample(samples, 'foot')
    forearm = _collapse_patient_sample(samples, 'forearm')
    paired = foot.merge(forearm, on='patient_id', suffixes=('_foot', '_forearm'))
    if len(paired) < 3:
        raise RuntimeError(f'too few paired patients: {len(paired)}')
    contrasts = {}
    for metric in ('topic0_all_mean', 'topic0_fibro_mean'):
        valid = paired[[f'{metric}_foot', f'{metric}_forearm']].notna().all(axis=1)
        diffs = paired.loc[valid, f'{metric}_foot'].to_numpy(float) - paired.loc[valid, f'{metric}_forearm'].to_numpy(float)
        contrasts[metric] = _paired_sign_flip(diffs)
    topic_results = []
    for topic in range(theta.shape[1]):
        diffs = paired['topic_means_foot'].apply(lambda x: x[topic]).to_numpy(float) - paired['topic_means_forearm'].apply(lambda x: x[topic]).to_numpy(float)
        result = _paired_sign_flip(diffs)
        result['topic'] = int(topic)
        topic_results.append(result)
    n_sig = sum((v['p_two_sided_exact_sign_flip'] < 0.05 for v in topic_results))
    pvals = np.asarray([v['p_two_sided_exact_sign_flip'] for v in topic_results], dtype=float)
    order = np.argsort(pvals)
    q_sorted = pvals[order] * len(pvals) / (np.arange(len(pvals)) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    for result, q in zip(topic_results, qvals):
        result['bh_q'] = float(min(1.0, q))
    n_bh_sig = sum((v['bh_q'] < 0.05 for v in topic_results))
    paired_out = paired.drop(columns=['topic_means_foot', 'topic_means_forearm'])
    paired_out.to_csv(out_dir / 'paired_patient_scores.csv', index=False)
    samples.drop(columns=['topic_means']).to_csv(out_dir / 'sample_scores.csv', index=False)
    report = {'experiment': 'paired_anatomical_positive_control', 'status': 'completed', 'scientific_unit': 'author-mapped patient with paired foot and forearm', 'scientific_scope': 'positive anatomical control for frozen encoder sensitivity; not a healing endpoint, not experiment A, and not an independent cohort', 'not_an_independent_cohort': True, 'healing_validation': False, 'patient_counts': {'mapped_foot_samples': int((mapping['tissue_site'] == 'foot').sum()), 'mapped_forearm_samples': int((mapping['tissue_site'] == 'forearm').sum()), 'paired_patients': int(len(paired))}, 'qc': {'cells_after_qc': int(X.shape[0]), 'min_genes_per_cell': args.min_genes_per_cell, 'max_mito': args.max_mito}, 'panel': {'size': int(len(panel)), 'covered': int(covered), 'fraction': float(covered / len(panel))}, 'celltype_markers_found': {k: int(v) for k, v in found.items()}, 'topic0_contrasts': contrasts, 'all_topic_contrasts': topic_results, 'positive_control_summary': {'n_topics_p_below_0_05': int(n_sig), 'n_topics_bh_q_below_0_05': int(n_bh_sig), 'topic0_all_p_below_0_05': bool(contrasts['topic0_all_mean']['p_two_sided_exact_sign_flip'] < 0.05), 'topic0_all_bh_q': float(topic_results[0]['bh_q']), 'topic0_fibro_p_below_0_05': bool(contrasts['topic0_fibro_mean']['p_two_sided_exact_sign_flip'] < 0.05), 'interpretation': 'paired anatomy is a sensitivity control only; a significant topic difference cannot establish healing specificity'}, 'protocol': {'script_sha256': sha256(Path(__file__)), 'author_map_sha256': sha256(Path(args.map_csv)), 'panel_sha256': sha256(discovery / 'panel.npy'), 'model_sha256': sha256(discovery / 'topic_model.pt'), 'encoder_refit': False, 'patient_pairing': 'author-mapped patient; repeated foot samples cell-count-weighted', 'test': 'exact paired sign-flip over patients; 4000 bootstrap resamples', 'seed': 0, 'elapsed_seconds': time.time() - t0}}
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'paired_patients': report['patient_counts']['paired_patients'], 'topic0_all': contrasts['topic0_all_mean'], 'topic0_fibro': contrasts['topic0_fibro_mean'], 'n_topics_p_below_0_05': n_sig, 'output': str(out_dir / 'report.json')}, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
