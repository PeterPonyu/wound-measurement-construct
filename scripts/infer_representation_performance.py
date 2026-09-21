#!/usr/bin/env python3
"""Infer representation performance."""
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
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
_BENCH_PATH = Path(__file__).resolve().parent / 'benchmark_expression_representations.py'
_spec = importlib.util.spec_from_file_location('representation_benchmark', _BENCH_PATH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def loso_scores(method: str, unit: dict, y: np.ndarray, seed: int, n_components: int, nmf_max_iter: int) -> np.ndarray:
    """Leave-one-sample-out scores for one method under one label vector."""
    folds = list(range(len(y)))
    if method == 'topic_simplex_theta0':
        theta = unit['theta_dfu']
        out = np.full(len(y), np.nan)
        for test_i in folds:
            train = np.array([i for i in folds if i != test_i], dtype=int)
            _, sign, _ = bench._orient(theta[train, 0], y[train])
            out[test_i] = sign * theta[test_i, 0]
        return out
    if method == 'module_score':
        return bench._module_cv(unit['X_dfu'], unit['panel'], y, folds)[0]
    if method == 'pca':
        return bench._pca_cv(unit['X_dfu'], y, folds, seed, n_components)[0]
    if method == 'nmf':
        return bench._nmf_cv(unit['X_dfu'], y, folds, seed, n_components, nmf_max_iter)[0]
    raise ValueError(method)

def bootstrap_indices(y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Stratified resample indices; both classes stay present by construction."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    draws = np.empty((n_boot, len(y)), dtype=np.int64)
    for b in range(n_boot):
        draws[b] = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
    return draws

def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, scores))

def label_permutations(y: np.ndarray, n_max: int | None, seed: int) -> tuple[list[np.ndarray], bool]:
    """Exact enumeration when affordable, otherwise a random subsample."""
    n_pos = int(y.sum())
    all_pos = list(itertools.combinations(range(len(y)), n_pos))
    if n_max is None or n_max >= len(all_pos):
        chosen, exact = (all_pos, True)
    else:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(all_pos), size=n_max, replace=False)
        chosen, exact = ([all_pos[i] for i in picks], False)
    out = []
    for pos in chosen:
        lab = np.zeros(len(y), dtype=int)
        lab[list(pos)] = 1
        out.append(lab)
    return (out, exact)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/analysis/representation_inference')
    ap.add_argument('--n-components', type=int, default=5)
    ap.add_argument('--nmf-max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-boot', type=int, default=4000)
    ap.add_argument('--perm-expensive', type=int, default=200, help='label permutations for PCA/NMF (14 refits each)')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    log('building the frozen 14-sample DFU unit')
    unit = bench.load_dfu_sample_unit(tar=args.tar, series_matrix=args.series_matrix, discovery_dir=args.discovery_dir)
    y = unit['y']
    methods = ['topic_simplex_theta0', 'module_score', 'pca', 'nmf']
    observed = {}
    for method in methods:
        scores = loso_scores(method, unit, y, args.seed, args.n_components, args.nmf_max_iter)
        observed[method] = {'scores': scores, 'auc': _auc(scores, y)}
        log(f"  {method}: LOSO AUC {observed[method]['auc']:.4f}")
    log(f'stratified bootstrap x{args.n_boot} (shared indices across methods)')
    draws = bootstrap_indices(y, args.n_boot, args.seed)
    boot = {m: np.empty(args.n_boot) for m in methods}
    for b, idx in enumerate(draws):
        yb = y[idx]
        for method in methods:
            boot[method][b] = _auc(observed[method]['scores'][idx], yb)
    auc_block = {}
    for method in methods:
        lo, hi = np.percentile(boot[method], [2.5, 97.5])
        auc_block[method] = {'loso_auc': observed[method]['auc'], 'bootstrap_95_ci': [float(lo), float(hi)], 'bootstrap_sd': float(boot[method].std(ddof=1)), 'ci_includes_0_5': bool(lo <= 0.5 <= hi)}
    pairwise = {}
    for a, b_ in itertools.combinations(methods, 2):
        diff = boot[a] - boot[b_]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pairwise[f'{a}__minus__{b_}'] = {'observed_difference': observed[a]['auc'] - observed[b_]['auc'], 'bootstrap_95_ci': [float(lo), float(hi)], 'ci_excludes_zero': bool(lo > 0 or hi < 0), 'p_two_sided_bootstrap': float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))}
    perm_block = {}
    for method in methods:
        expensive = method in ('pca', 'nmf')
        perms, exact = label_permutations(y, args.perm_expensive if expensive else None, args.seed)
        log(f"permutation null for {method}: {len(perms)} relabelings ({('exact' if exact else 'subsample')})")
        null = np.empty(len(perms))
        for i, lab in enumerate(perms):
            null[i] = _auc(loso_scores(method, unit, lab, args.seed, args.n_components, args.nmf_max_iter), lab)
        obs = observed[method]['auc']
        perm_block[method] = {'n_relabelings': len(perms), 'exact_enumeration': exact, 'null_auc_mean': float(null.mean()), 'null_auc_95th_percentile': float(np.percentile(null, 95)), 'p_one_sided': float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1)), 'significant_at_0_05': bool((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1) < 0.05)}
        log(f"  null mean {null.mean():.3f}, p={perm_block[method]['p_one_sided']:.4f}")
    ranked = sorted(methods, key=lambda m: -observed[m]['auc'])
    orderable = [k for k, v in pairwise.items() if v['ci_excludes_zero']]
    report = {'experiment': 'representation_benchmark_inference', 'status': 'completed', 'scientific_unit': 'sample', 'patient_level_inference_available': False, 'question': 'Can the four representations be ordered at this sample size?', 'sample_counts': {'dfu': int(len(y)), 'healer': int(y.sum()), 'nonhealer': int((1 - y).sum())}, 'auc': auc_block, 'pairwise_auc_difference': pairwise, 'permutation_null': perm_block, 'point_estimate_ranking': ranked, 'pairs_with_nonzero_difference_at_95': orderable, 'verdict': 'no pairwise AUC difference excludes zero; the point-estimate ranking is not a ranking' if not orderable else 'at least one pairwise difference excludes zero: ' + ', '.join(orderable), 'protocol': {'leave_one_sample_out': True, 'orientation_estimated_inside_fold': True, 'permutation_reruns_full_loso': True, 'bootstrap_stratified_shared_indices': True, 'n_boot': args.n_boot, 'seed': args.seed, 'pca_n_components': args.n_components, 'nmf_n_components': args.n_components, 'nmf_max_iter': args.nmf_max_iter, 'benchmark_script_sha256': _sha256(str(_BENCH_PATH)), 'script_sha256': _sha256(__file__), 'elapsed_seconds': time.time() - t0}}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    log(f'wrote {out}')
    print(json.dumps({'auc': {m: auc_block[m] for m in methods}, 'verdict': report['verdict']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
