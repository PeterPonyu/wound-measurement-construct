#!/usr/bin/env python3
"""Benchmark patient readouts."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_BENCH_PATH = ROOT / 'scripts' / 'benchmark_expression_representations.py'
_SPEC = importlib.util.spec_from_file_location('representation_benchmark', _BENCH_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('cannot load scripts/benchmark_expression_representations.py')
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)
DEFAULT_MAP = ROOT / 'cohort_metadata' / 'GSE165816_ncomms_subject_sample_map.csv'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'analysis' / 'patient_unit_representation_benchmark'
DEFAULT_DISCOVERY = ROOT / 'outputs' / 'expression_representation'
DEFAULT_THETA_PARQUET = ROOT / 'outputs' / 'celltype_resolved' / 'GSE165816_discovery_theta.parquet'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def exact_permutations(y: np.ndarray, n_max: int | None, seed: int) -> tuple[list[np.ndarray], bool]:
    n_pos = int(y.sum())
    combos = list(itertools.combinations(range(len(y)), n_pos))
    exact = n_max is None or n_max >= len(combos)
    if not exact:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(combos), size=n_max, replace=False)
        combos = [combos[int(i)] for i in chosen]
    labels = []
    for pos in combos:
        lab = np.zeros(len(y), dtype=int)
        lab[list(pos)] = 1
        labels.append(lab)
    return (labels, exact)

def auc(scores: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, scores))

def lopo_scores(method: str, X: np.ndarray, theta: np.ndarray, y: np.ndarray, panel: np.ndarray, seed: int, n_components: int, nmf_max_iter: int) -> np.ndarray:
    folds = list(range(len(y)))
    if method == 'topic_simplex_theta0':
        out = np.full(len(y), np.nan)
        for test_i in folds:
            train = np.array([i for i in folds if i != test_i], dtype=int)
            _, sign, _ = _BENCH._orient(theta[train, 0], y[train])
            out[test_i] = sign * theta[test_i, 0]
        return out
    if method == 'module_score':
        return _module_patient_cv(X, y, panel)
    if method == 'pca':
        return _BENCH._pca_cv(X, y, folds, seed, n_components)[0]
    if method == 'nmf':
        return _BENCH._nmf_cv(X, y, folds, seed, n_components, nmf_max_iter)[0]
    raise ValueError(method)

def _module_patient_cv(X: np.ndarray, y: np.ndarray, panel: np.ndarray) -> np.ndarray:
    genes = np.asarray(panel).astype(str)
    pos = pd.Index(genes).get_indexer(_BENCH.MODULE_GENES)
    pos = pos[pos >= 0]
    if len(pos) < 5:
        raise RuntimeError('fewer than five module genes present on frozen panel')
    out = np.full(len(y), np.nan)
    for test_i in range(len(y)):
        train = np.array([i for i in range(len(y)) if i != test_i], dtype=int)
        mu = X[train][:, pos].mean(axis=0)
        sd = X[train][:, pos].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        tr = ((X[train][:, pos] - mu) / sd).mean(axis=1)
        te = ((X[test_i, pos] - mu) / sd).mean()
        _, sign, _ = _BENCH._orient(tr, y[train])
        out[test_i] = sign * te
    return out

def stratified_bootstrap(scores: dict[str, np.ndarray], y: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    draws = []
    for _ in range(n_boot):
        draws.append(np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]))
    boot = {m: np.empty(n_boot) for m in scores}
    for i, idx in enumerate(draws):
        for m, s in scores.items():
            boot[m][i] = auc(s[idx], y[idx])
    out = {}
    for m, vals in boot.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[m] = {'auc': auc(scores[m], y), 'bootstrap_95_ci': [float(lo), float(hi)], 'bootstrap_sd': float(vals.std(ddof=1))}
    pairs = {}
    for a, b in itertools.combinations(scores, 2):
        diff = boot[a] - boot[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pairs[f'{a}__minus__{b}'] = {'observed_difference': float(auc(scores[a], y) - auc(scores[b], y)), 'bootstrap_95_ci': [float(lo), float(hi)], 'ci_excludes_zero': bool(lo > 0 or hi < 0)}
    return {'auc': out, 'pairwise_auc_difference': pairs}

def patient_semantics(map_csv: Path, theta_parquet: Path, cut: float) -> dict:
    mapping = pd.read_csv(map_csv, dtype=str)
    theta = pd.read_parquet(theta_parquet)
    if 'celltype' not in theta.columns:
        raise RuntimeError('theta parquet lacks celltype')
    theta = theta[theta['celltype'].astype(str) == 'fibroblast'].copy()
    foot = mapping[mapping['tissue_site'].astype(str) == 'foot'].copy()
    theta = theta.merge(foot[['sample_id', 'patient_id', 'group']], left_on='gsm', right_on='sample_id', how='inner')
    if theta.empty:
        raise RuntimeError('no mapped foot cells in theta parquet')
    sample = theta.groupby('gsm', observed=True).agg(patient_id=('patient_id', 'first'), group=('group', 'first'), mean_loading=('topic_0', 'mean'), mixture_weight=('topic_0', lambda x: float((x.to_numpy() > cut).mean())), n_cells=('topic_0', 'size')).reset_index()
    patient = theta.groupby('patient_id', observed=True).agg(group=('group', 'first'), mean_loading=('topic_0', 'mean'), mixture_weight=('topic_0', lambda x: float((x.to_numpy() > cut).mean())), n_cells=('topic_0', 'size')).reset_index()
    r_sample = spearmanr(sample['mean_loading'], sample['mixture_weight'])
    r_patient = spearmanr(patient['mean_loading'], patient['mixture_weight'])
    return {'cut': cut, 'sample_unit': {'n_samples': int(len(sample)), 'n_patients': int(sample.patient_id.nunique()), 'spearman_rho': float(r_sample.statistic), 'p': float(r_sample.pvalue)}, 'patient_collapsed_unit': {'n_patients': int(len(patient)), 'spearman_rho': float(r_patient.statistic), 'p': float(r_patient.pvalue)}, 'sample_scores': sample.to_dict(orient='records'), 'patient_scores': patient.to_dict(orient='records')}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--discovery-dir', default=str(DEFAULT_DISCOVERY))
    ap.add_argument('--map-csv', default=str(DEFAULT_MAP))
    ap.add_argument('--theta-parquet', default=str(DEFAULT_THETA_PARQUET))
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--n-components', type=int, default=5)
    ap.add_argument('--nmf-max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-boot', type=int, default=4000)
    ap.add_argument('--perm-expensive', type=int, default=200)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    unit = _BENCH.load_dfu_sample_unit(tar=args.tar, series_matrix=args.series_matrix, discovery_dir=args.discovery_dir)
    panel = np.asarray(unit['panel']).astype(object)
    mapping = pd.read_csv(args.map_csv, dtype=str)
    dfu = unit['dfu'].copy()
    dfu['gsm'] = dfu['gsm'].astype(str)
    dfu = dfu.merge(mapping[['sample_id', 'patient_id', 'healing_status']], left_on='gsm', right_on='sample_id', how='left')
    if dfu['patient_id'].isna().any() or dfu['patient_id'].nunique() != 11:
        raise RuntimeError('DFU sample unit did not map to 11 patients')
    rows = []
    X = unit['X_dfu']
    theta = unit['theta_dfu']
    for pid, block in dfu.groupby('patient_id', sort=True):
        idx = block.index.to_numpy()
        weights = block['n_cells'].to_numpy(float)
        weights = weights / weights.sum()
        rows.append({'patient_id': str(pid), 'healing_status': str(block['healing_status'].iloc[0]), 'n_samples': int(len(idx)), 'n_cells': int(block['n_cells'].sum()), 'X_index': idx.tolist()})
    patient_table = pd.DataFrame(rows)
    if (patient_table.healing_status == 'healed').sum() != 7 or (patient_table.healing_status == 'not_healed').sum() != 4:
        raise RuntimeError('unexpected patient arms')
    Xp, Tp = ([], [])
    for row in patient_table.itertuples(index=False):
        idx = np.asarray(row.X_index, dtype=int)
        weights = dfu.loc[idx, 'n_cells'].to_numpy(float)
        weights = weights / weights.sum()
        Xp.append(np.average(X[idx], axis=0, weights=weights))
        Tp.append(np.average(theta[idx], axis=0, weights=weights))
    Xp = np.asarray(Xp, dtype=np.float32)
    Tp = np.asarray(Tp, dtype=np.float32)
    y = (patient_table.healing_status == 'healed').astype(int).to_numpy()
    methods = ['topic_simplex_theta0', 'module_score', 'pca', 'nmf']
    scores = {}
    for method in methods:
        scores[method] = lopo_scores(method, Xp, Tp, y, panel, args.seed, args.n_components, args.nmf_max_iter)
    boot = stratified_bootstrap(scores, y, args.n_boot, args.seed)
    perm_block = {}
    for method in methods:
        expensive = method in {'pca', 'nmf'}
        labs, exact = exact_permutations(y, None if not expensive else args.perm_expensive, args.seed)
        null = np.empty(len(labs))
        for i, lab in enumerate(labs):
            null[i] = auc(lopo_scores(method, Xp, Tp, lab, panel, args.seed, args.n_components, args.nmf_max_iter), lab)
        obs_auc = auc(scores[method], y)
        perm_block[method] = {'n_relabelings': int(len(labs)), 'exact_enumeration': bool(exact), 'null_auc_mean': float(null.mean()), 'null_auc_95th_percentile': float(np.percentile(null, 95)), 'p_one_sided': float((1 + np.sum(null >= obs_auc - 1e-12)) / (len(null) + 1))}
    patient_table_out = patient_table.drop(columns=['X_index'])
    patient_table_out['topic_simplex_theta0_lopo'] = scores['topic_simplex_theta0']
    patient_table_out['module_score_lopo'] = scores['module_score']
    patient_table_out['pca_lopo'] = scores['pca']
    patient_table_out['nmf_lopo'] = scores['nmf']
    patient_table_out.to_csv(out_dir / 'patient_scores.csv', index=False)
    semantics = patient_semantics(Path(args.map_csv), Path(args.theta_parquet), cut=0.18429584801197052)
    report = {'experiment': 'patient_unit_representation_benchmark', 'status': 'completed', 'scientific_unit': 'patient', 'scientific_scope': 'author-mapped GSE165816 discovery-unit correction; not an independent cohort', 'not_an_independent_cohort': True, 'patient_level_inference_available': True, 'patient_counts': {'dfu': int(len(y)), 'healed': int(y.sum()), 'not_healed': int((1 - y).sum()), 'repeated_sample_patients': int((patient_table.n_samples > 1).sum())}, 'auc': boot['auc'], 'pairwise_auc_difference': boot['pairwise_auc_difference'], 'permutation_null': perm_block, 'patient_semantics': semantics, 'protocol': {'author_map_sha256': sha256(Path(args.map_csv)), 'benchmark_script_sha256': sha256(_BENCH_PATH), 'script_sha256': sha256(Path(__file__)), 'aggregate_within_patient': 'cell-count-weighted mean of repeated sample means', 'leave_one_patient_out': True, 'orientation_inside_fold': True, 'bootstrap_stratified_shared_indices': True, 'expression_matrix_reloaded': True, 'mcid_filled': False, 'seed': args.seed, 'n_boot': args.n_boot, 'elapsed_seconds': time.time() - t0}}
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'auc': report['auc'], 'permutation': report['permutation_null'], 'patient_semantics': report['patient_semantics']['patient_collapsed_unit']}, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
