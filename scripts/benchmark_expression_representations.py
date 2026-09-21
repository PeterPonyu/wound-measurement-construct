#!/usr/bin/env python3
"""Benchmark expression representations."""
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
from scipy.stats import spearmanr
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import library_normalize, load_dense_csv_tar, mito_fraction, select_target_genes
MODULE_GENES = ['CHI3L1', 'TIMP1', 'COL3A1', 'DCN', 'FN1', 'LUM', 'COL6A2', 'MMP1', 'MMP3']
PINNED_REGULATORS = ['CDKN1A', 'FOS', 'JUNB', 'IL1B', 'TLR4', 'NLRP3', 'TNF', 'IFNG', 'MMP1', 'MMP3', 'MMP13', 'CHI3L1', 'TIMP1', 'COL7A1', 'NRG1', 'ASPN', 'KRT14', 'KRT5', 'CD68', 'CD163', 'PRG4', 'THY1', 'VEGFA']
KEEP_ARMS = {'DFU-healer', 'DFU-nonhealer', 'Non-diabetic'}

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va = np.var(a, ddof=1) if len(a) > 1 else 0.0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float('nan')

def _exact_two_group_p(scores: np.ndarray, y: np.ndarray) -> float:
    """Two-sided exact permutation p for the observed mean difference."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    observed = abs(float(scores[y == 1].mean() - scores[y == 0].mean()))
    values = []
    for pos in itertools.combinations(range(len(scores)), n_pos):
        mask = np.zeros(len(scores), dtype=bool)
        mask[list(pos)] = True
        values.append(abs(float(scores[mask].mean() - scores[~mask].mean())))
    values = np.asarray(values)
    return float((1 + np.sum(values >= observed - 1e-12)) / (len(values) + 1))

def _orient(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    diff = float(score[y == 1].mean() - score[y == 0].mean())
    sign = 1.0 if diff >= 0 else -1.0
    return (score * sign, sign, abs(diff))

def _sample_means(X: sp.csr_matrix, obs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    means = []
    for gsm, block in obs.groupby('gsm', sort=True):
        idx = block.index.to_numpy()
        means.append(np.asarray(X[idx].mean(axis=0)).ravel())
        rows.append({'gsm': gsm, 'disease': str(block['disease'].iloc[0]), 'n_cells': int(len(idx))})
    return (pd.DataFrame(rows), np.vstack(means).astype(np.float32))

def _module_cv(X: np.ndarray, genes: np.ndarray, y: np.ndarray, folds: list[int]) -> tuple[np.ndarray, list[dict]]:
    pos = pd.Index(genes.astype(str)).get_indexer(MODULE_GENES)
    pos = pos[pos >= 0]
    if len(pos) < 5:
        raise RuntimeError(f'only {len(pos)} module genes present; need at least 5')
    scores = np.full(len(y), np.nan, dtype=float)
    details = []
    for test_i in folds:
        train = np.array([i for i in folds if i != test_i], dtype=int)
        mu = X[train][:, pos].mean(axis=0)
        sd = X[train][:, pos].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        train_score = ((X[train][:, pos] - mu) / sd).mean(axis=1)
        test_score = ((X[test_i, pos] - mu) / sd).mean()
        _, sign, train_abs_diff = _orient(train_score, y[train])
        scores[test_i] = sign * test_score
        details.append({'test_index': int(test_i), 'sign': sign, 'train_abs_mean_difference': train_abs_diff})
    return (scores, details)

def _pca_cv(X: np.ndarray, y: np.ndarray, folds: list[int], seed: int, n_components: int) -> tuple[np.ndarray, list[dict]]:
    scores = np.full(len(y), np.nan, dtype=float)
    details = []
    for test_i in folds:
        train = np.array([i for i in folds if i != test_i], dtype=int)
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_z = scaler.fit_transform(X[train])
        test_z = scaler.transform(X[[test_i]])
        k = min(n_components, len(train) - 1, train_z.shape[1])
        model = PCA(n_components=k, svd_solver='full', random_state=seed)
        train_score = model.fit_transform(train_z)
        test_score = model.transform(test_z)[0]
        diffs = np.abs([train_score[y[train] == 1, j].mean() - train_score[y[train] == 0, j].mean() for j in range(k)])
        j = int(np.argmax(diffs))
        sign = 1.0 if train_score[y[train] == 1, j].mean() >= train_score[y[train] == 0, j].mean() else -1.0
        scores[test_i] = sign * test_score[j]
        details.append({'test_index': int(test_i), 'component': j, 'sign': sign, 'train_abs_mean_difference': float(diffs[j]), 'explained_variance_ratio': float(model.explained_variance_ratio_[j])})
    return (scores, details)

def _nmf_cv(X: np.ndarray, y: np.ndarray, folds: list[int], seed: int, n_components: int, max_iter: int) -> tuple[np.ndarray, list[dict]]:
    scores = np.full(len(y), np.nan, dtype=float)
    details = []
    for test_i in folds:
        train = np.array([i for i in folds if i != test_i], dtype=int)
        k = min(n_components, len(train) - 1, X.shape[1])
        model = NMF(n_components=k, init='nndsvda', random_state=seed, max_iter=max_iter, tol=0.0001, l1_ratio=0.0, beta_loss='frobenius')
        train_w = model.fit_transform(np.maximum(X[train], 0.0))
        test_w = model.transform(np.maximum(X[[test_i]], 0.0))[0]
        diffs = np.abs([train_w[y[train] == 1, j].mean() - train_w[y[train] == 0, j].mean() for j in range(k)])
        j = int(np.argmax(diffs))
        sign = 1.0 if train_w[y[train] == 1, j].mean() >= train_w[y[train] == 0, j].mean() else -1.0
        scores[test_i] = sign * test_w[j]
        details.append({'test_index': int(test_i), 'component': j, 'sign': sign, 'train_abs_mean_difference': float(diffs[j]), 'reconstruction_err': float(model.reconstruction_err_)})
    return (scores, details)

def _full_module(X: np.ndarray, genes: np.ndarray) -> np.ndarray:
    pos = pd.Index(genes.astype(str)).get_indexer(MODULE_GENES)
    pos = pos[pos >= 0]
    mu = X[:, pos].mean(axis=0)
    sd = X[:, pos].std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return ((X[:, pos] - mu) / sd).mean(axis=1)

def _full_pca(X: np.ndarray, y: np.ndarray, seed: int, n_components: int) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    z = scaler.fit_transform(X)
    k = min(n_components, len(y) - 1, z.shape[1])
    model = PCA(n_components=k, svd_solver='full', random_state=seed)
    w = model.fit_transform(z)
    diffs = np.asarray([w[y == 1, j].mean() - w[y == 0, j].mean() for j in range(k)])
    j = int(np.argmax(np.abs(diffs)))
    sign = 1.0 if diffs[j] >= 0 else -1.0
    return (sign * w[:, j], {'component': j, 'sign': sign, 'train_abs_mean_difference': float(abs(diffs[j])), 'explained_variance_ratio': float(model.explained_variance_ratio_[j])})

def _full_nmf(X: np.ndarray, y: np.ndarray, seed: int, n_components: int, max_iter: int) -> tuple[np.ndarray, dict]:
    k = min(n_components, len(y) - 1, X.shape[1])
    model = NMF(n_components=k, init='nndsvda', random_state=seed, max_iter=max_iter, tol=0.0001, l1_ratio=0.0, beta_loss='frobenius')
    w = model.fit_transform(np.maximum(X, 0.0))
    diffs = np.asarray([w[y == 1, j].mean() - w[y == 0, j].mean() for j in range(k)])
    j = int(np.argmax(np.abs(diffs)))
    sign = 1.0 if diffs[j] >= 0 else -1.0
    return (sign * w[:, j], {'component': j, 'sign': sign, 'train_abs_mean_difference': float(abs(diffs[j])), 'reconstruction_err': float(model.reconstruction_err_)})

def _metric(scores: np.ndarray, y: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, scores))
    return {'auc': auc, 'n': int(len(y)), 'n_healer': int(y.sum()), 'n_nonhealer': int((1 - y).sum())}

def load_dfu_sample_unit(*, tar: str, series_matrix: str, discovery_dir: str, base_panel_size: int=7000, min_genes_per_cell: int=200, max_mito: float=0.2) -> dict:
    discovery = Path(discovery_dir)
    theta = np.load(discovery / 'theta.npy')
    frozen_obs = pd.read_csv(discovery / 'obs.csv')
    panel = np.load(discovery / 'panel.npy', allow_pickle=True).astype(object)
    if len(theta) != len(frozen_obs):
        raise RuntimeError('theta/obs row count mismatch')
    cohort = load_dense_csv_tar(tar, series_matrix=series_matrix)
    keep = (cohort.obs['tissue'].str.lower() == 'foot skin') & cohort.obs['disease'].isin(KEEP_ARMS)
    idx = np.flatnonzero(keep.to_numpy())
    X = cohort.X[idx]
    obs = cohort.obs.iloc[idx].reset_index(drop=True)
    mito = mito_fraction(X, cohort.genes)
    gpc = np.asarray((X > 0).sum(axis=1)).ravel()
    qc = (gpc >= min_genes_per_cell) & (mito <= max_mito)
    X = X[qc]
    obs = obs.iloc[np.flatnonzero(qc)].reset_index(drop=True)
    gene_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    genes = cohort.genes[gene_counts >= 10]
    X = X[:, gene_counts >= 10]
    gene_idx, selected_panel = select_target_genes(X, genes, n_top=base_panel_size, pinned=PINNED_REGULATORS, exclude_technical=True, log_normalize=True)
    selected_panel = selected_panel.astype(object)
    if not np.array_equal(selected_panel.astype(str), panel.astype(str)):
        raise RuntimeError('recomputed panel differs from frozen expression representation panel')
    Xn = library_normalize(X[:, gene_idx])
    sample_table, sample_means = _sample_means(Xn, obs)
    frozen_key = frozen_obs['gsm'].astype(str)
    theta_by_gsm = {}
    for gsm, block in frozen_obs.groupby('gsm', sort=True):
        theta_by_gsm[str(gsm)] = theta[frozen_key.to_numpy() == str(gsm)].mean(axis=0)
    sample_table = sample_table.sort_values('gsm').reset_index(drop=True)
    theta_sample = np.vstack([theta_by_gsm[str(g)] for g in sample_table['gsm']])
    y_mask = sample_table['disease'].isin(['DFU-healer', 'DFU-nonhealer']).to_numpy()
    dfu = sample_table.loc[y_mask].reset_index(drop=True)
    y = (dfu['disease'] == 'DFU-healer').astype(int).to_numpy()
    if len(y) != 14 or int(y.sum()) != 9 or int((1 - y).sum()) != 5:
        raise RuntimeError(f'unexpected DFU sample unit: n={len(y)}, healer={y.sum()}')
    return {'X_dfu': sample_means[y_mask], 'theta_dfu': theta_sample[y_mask], 'y': y, 'panel': selected_panel, 'dfu': dfu, 'sample_table': sample_table}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/representation_benchmark')
    ap.add_argument('--n-components', type=int, default=5)
    ap.add_argument('--base-panel-size', type=int, default=7000, help='dispersion quota used before pinned expression representation regulators are appended')
    ap.add_argument('--nmf-max-iter', type=int, default=300)
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    unit = load_dfu_sample_unit(tar=args.tar, series_matrix=args.series_matrix, discovery_dir=args.discovery_dir, base_panel_size=args.base_panel_size, min_genes_per_cell=args.min_genes_per_cell, max_mito=args.max_mito)
    panel = np.load(Path(args.discovery_dir) / 'panel.npy', allow_pickle=True).astype(object)
    selected_panel = unit['panel']
    sample_table = unit['sample_table']
    dfu = unit['dfu']
    X_dfu = unit['X_dfu']
    theta_dfu = unit['theta_dfu']
    y = unit['y']
    folds = list(range(len(y)))
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    method_cv = {}
    details = {}
    topic_scores = np.full(len(y), np.nan)
    topic_details = []
    for test_i in folds:
        train = np.array([i for i in folds if i != test_i], dtype=int)
        train_score = theta_dfu[train, 0]
        test_score = theta_dfu[test_i, 0]
        _, sign, train_abs_diff = _orient(train_score, y[train])
        topic_scores[test_i] = sign * test_score
        topic_details.append({'test_index': test_i, 'sign': sign, 'train_abs_mean_difference': train_abs_diff})
    method_cv['topic_simplex_theta0'] = {'scores': topic_scores.tolist(), 'auc': _metric(topic_scores, y)['auc']}
    details['topic_simplex_theta0'] = topic_details
    module_scores, module_details = _module_cv(X_dfu, selected_panel, y, folds)
    method_cv['module_score'] = {'scores': module_scores.tolist(), 'auc': _metric(module_scores, y)['auc']}
    details['module_score'] = module_details
    full_scores = {'topic_simplex_theta0': theta_dfu[:, 0].copy(), 'module_score': _full_module(X_dfu, selected_panel)}
    cv_seed_metrics = {'topic_simplex_theta0': [_metric(topic_scores, y)], 'module_score': [_metric(module_scores, y)]}
    full_seed_scores = {'topic_simplex_theta0': [full_scores['topic_simplex_theta0']], 'module_score': [full_scores['module_score']]}
    full_fit_details = {'topic_simplex_theta0': [{}], 'module_score': [{}]}
    for seed in seeds:
        pca_scores, pca_details = _pca_cv(X_dfu, y, folds, seed, args.n_components)
        nmf_scores, nmf_details = _nmf_cv(X_dfu, y, folds, seed, args.n_components, args.nmf_max_iter)
        method_cv.setdefault('pca', {})[str(seed)] = {'scores': pca_scores.tolist(), **_metric(pca_scores, y)}
        method_cv.setdefault('nmf', {})[str(seed)] = {'scores': nmf_scores.tolist(), **_metric(nmf_scores, y)}
        details.setdefault('pca', {})[str(seed)] = pca_details
        details.setdefault('nmf', {})[str(seed)] = nmf_details
        pca_full, pca_meta = _full_pca(X_dfu, y, seed, args.n_components)
        nmf_full, nmf_meta = _full_nmf(X_dfu, y, seed, args.n_components, args.nmf_max_iter)
        full_seed_scores.setdefault('pca', []).append(pca_full)
        full_seed_scores.setdefault('nmf', []).append(nmf_full)
        full_fit_details.setdefault('pca', []).append(pca_meta)
        full_fit_details.setdefault('nmf', []).append(nmf_meta)
    summary = {}
    for method, scores_list in full_seed_scores.items():
        effects = []
        for scores in scores_list:
            oriented, sign, abs_diff = _orient(np.asarray(scores), y)
            effects.append({'mean_difference_healer_minus_nonhealer': float(oriented[y == 1].mean() - oriented[y == 0].mean()), 'cohens_d': _cohens_d(oriented[y == 1], oriented[y == 0]), 'exact_two_sided_permutation_p': _exact_two_group_p(oriented, y), 'orientation_sign': sign, 'absolute_mean_difference': abs_diff})
        summary[method] = {'cv': cv_seed_metrics[method] if method in cv_seed_metrics else [{'seed': int(seed), 'auc': method_cv[method][str(seed)]['auc']} for seed in seeds], 'full_sample_descriptive': effects, 'full_fit_details': full_fit_details[method]}
    stability = {}
    for method, score_list in full_seed_scores.items():
        pairs = []
        for i in range(len(score_list)):
            for j in range(i + 1, len(score_list)):
                rho = spearmanr(score_list[i], score_list[j]).statistic
                pairs.append(float(rho) if np.isfinite(rho) else float('nan'))
        stability[method] = {'n_score_vectors': len(score_list), 'pairwise_spearman_mean': float(np.nanmean(pairs)) if pairs else 1.0, 'pairwise_spearman_min': float(np.nanmin(pairs)) if pairs else 1.0}
    representative = {m: np.asarray(v[0]) for m, v in full_seed_scores.items()}
    cross_method = {}
    for a, b in itertools.combinations(representative, 2):
        rho = spearmanr(representative[a], representative[b]).statistic
        cross_method[f'{a}__{b}'] = float(rho) if np.isfinite(rho) else None
    scores_out = dfu.copy()
    scores_out['y_healer'] = y
    scores_out['topic_simplex_theta0'] = full_scores['topic_simplex_theta0']
    scores_out['module_score'] = full_scores['module_score']
    for method in ('pca', 'nmf'):
        scores_out[f'{method}_seed{seeds[0]}'] = full_seed_scores[method][0]
    scores_out.to_csv(os.path.join(args.output_dir, 'sample_scores.csv'), index=False)
    report = {'experiment': 'frozen_representation_benchmark', 'status': 'completed', 'scientific_unit': 'sample', 'patient_level_inference_available': False, 'patient_level_caveat': 'GSE165816 exposes no subject identifier linking the 14 DFU samples; results are sample-level and exploratory.', 'inputs': {'tar': args.tar, 'series_matrix': args.series_matrix, 'discovery_dir': args.discovery_dir}, 'sample_counts': {'all_foot_skin_samples': int(len(sample_table)), 'dfu_samples': int(len(y)), 'healer': int(y.sum()), 'nonhealer': int((1 - y).sum()), 'non_diabetic': int((sample_table['disease'] == 'Non-diabetic').sum())}, 'module': {'genes_requested': MODULE_GENES, 'genes_present': [str(g) for g in selected_panel if str(g) in set(MODULE_GENES)], 'source': 'pre-specified fibro-inflammatory marker module'}, 'summary': summary, 'stability': stability, 'cross_method_spearman_representative': cross_method, 'protocol': {'seeds': seeds, 'leave_one_sample_out': True, 'test_sample_not_used_for_scaling_or_fit': True, 'topic_simplex_frozen': True, 'pca_n_components': args.n_components, 'nmf_n_components': args.n_components, 'nmf_max_iter': args.nmf_max_iter, 'panel_sha256': hashlib.sha256(np.asarray(panel.astype(str)).tobytes()).hexdigest(), 'script_sha256': _sha256(__file__), 'elapsed_seconds': time.time() - t0}}
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    print(json.dumps({'output_dir': args.output_dir, 'dfu_samples': len(y), 'methods': list(summary), 'cv_auc': {m: v['cv'] for m, v in summary.items()}, 'stability': stability}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
