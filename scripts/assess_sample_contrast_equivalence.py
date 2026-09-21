#!/usr/bin/env python3
"""Assess sample contrast equivalence."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import parse_series_matrix
PATIENT_TOKENS = ('patient', 'subject', 'donor', 'individual', 'participant')
TITLE_PREFIX_RE = re.compile('^\\s*([A-Za-z]+\\d+[A-Za-z]?)\\s*:')

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _exact_p(scores: np.ndarray, y: np.ndarray) -> float:
    observed = abs(float(scores[y == 1].mean() - scores[y == 0].mean()))
    n_pos = int(y.sum())
    hits = 0
    total = 0
    for pos in itertools.combinations(range(len(scores)), n_pos):
        mask = np.zeros(len(scores), dtype=bool)
        mask[list(pos)] = True
        delta = abs(float(scores[mask].mean() - scores[~mask].mean()))
        hits += int(delta >= observed - 1e-12)
        total += 1
    return float((1 + hits) / (total + 1))

def _bootstrap_ci(a: np.ndarray, b: np.ndarray, seed: int, n: int=30000) -> list[float]:
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n, len(a)))
    ib = rng.integers(0, len(b), size=(n, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]

def _tost(a: np.ndarray, b: np.ndarray, bound: float, alpha: float=0.05) -> dict:
    """Welch TOST for healer minus non-healer mean difference."""
    diff = float(a.mean() - b.mean())
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    se = float(np.sqrt(va / len(a) + vb / len(b)))
    if se == 0:
        return {'bound': float(bound), 'p_lower': 0.0 if diff > -bound else 1.0, 'p_upper': 0.0 if diff < bound else 1.0, 'equivalent_at_alpha_0_05': bool(abs(diff) < bound), 'welch_df': None}
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / max(len(a) - 1, 1) + (vb / len(b)) ** 2 / max(len(b) - 1, 1))
    p_lower = float(1.0 - t.cdf((diff + bound) / se, df))
    p_upper = float(t.cdf((diff - bound) / se, df))
    return {'bound': float(bound), 'p_lower': p_lower, 'p_upper': p_upper, 'welch_df': float(df), 'equivalent_at_alpha_0_05': bool(max(p_lower, p_upper) < alpha)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--output-dir', default='outputs/analysis/patient_mapping_equivalence')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--equivalence-bound', type=float, default=None, help='pre-specified absolute mean-difference margin; omit for sensitivity grid')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    discovery = Path(args.discovery_dir)
    theta = np.load(discovery / 'theta.npy')
    obs = pd.read_csv(discovery / 'obs.csv')
    if len(theta) != len(obs):
        raise RuntimeError('theta/obs row count mismatch')
    col = f'topic_{args.focus_topic}'
    if args.focus_topic < 0 or args.focus_topic >= theta.shape[1]:
        raise ValueError(f'focus topic {args.focus_topic} outside 0..{theta.shape[1] - 1}')
    cell = obs[['gsm', 'title', 'tissue', 'disease']].copy()
    cell['score'] = theta[:, args.focus_topic]
    samples = cell.groupby(['gsm', 'title', 'tissue', 'disease'], observed=True).agg(score=('score', 'mean'), n_cells=('score', 'size')).reset_index()
    dfu = samples[samples['disease'].isin(['DFU-healer', 'DFU-nonhealer'])].copy()
    a = dfu.loc[dfu['disease'] == 'DFU-healer', 'score'].to_numpy(float)
    b = dfu.loc[dfu['disease'] == 'DFU-nonhealer', 'score'].to_numpy(float)
    y = (dfu['disease'] == 'DFU-healer').to_numpy(int)
    if len(a) != 9 or len(b) != 5:
        raise RuntimeError(f'unexpected DFU sample unit: healer={len(a)} nonhealer={len(b)}')
    series = parse_series_matrix(args.series_matrix)
    id_cols = [c for c in series.columns if any((tok in c.lower() for tok in PATIENT_TOKENS))]
    title_prefix = series['title'].astype(str).str.extract(TITLE_PREFIX_RE, expand=False)
    title_prefix = title_prefix.dropna().astype(str)
    explicit_mapping = bool(id_cols)
    prefix_reuse = bool(title_prefix.duplicated().any())
    mapping = {'status': 'available' if explicit_mapping else 'unavailable', 'explicit_identifier_columns': id_cols, 'series_matrix_columns': series.columns.tolist(), 'title_prefix_observation': {'n_nonmissing': int(title_prefix.notna().sum()), 'n_unique': int(title_prefix.nunique()), 'reused_prefix_present': prefix_reuse, 'interpretation': 'title prefixes are sample labels; they are not promoted to patient IDs without an explicit author mapping'}, 'patient_level_inference_available': explicit_mapping, 'blocking_reason': None if explicit_mapping else 'public series matrix has no patient/subject/donor/individual identifier'}
    diff = float(a.mean() - b.mean())
    welch = ttest_ind(a, b, equal_var=False)
    margins = [float(args.equivalence_bound)] if args.equivalence_bound is not None else [0.05, 0.1, 0.2, 0.3, 0.5]
    if any((x <= 0 for x in margins)):
        raise ValueError('equivalence margins must be positive')
    sensitivity = [_tost(a, b, m) for m in margins]
    lo, hi = (1e-06, 5.0)
    if _tost(a, b, hi)['equivalent_at_alpha_0_05']:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _tost(a, b, mid)['equivalent_at_alpha_0_05']:
                hi = mid
            else:
                lo = mid
        smallest = {'margin': float(hi), 'multiple_of_observed_difference': float(hi / abs(diff)) if diff != 0 else None, 'note': 'smallest absolute mean-difference margin this sample can declare equivalent at alpha=0.05; a clinically meaningful margin below this value cannot be ruled in or out here'}
    else:
        smallest = {'margin': None, 'note': 'no margin up to 5.0 reaches equivalence'}
    report = {'experiment': 'patient_mapping_and_equivalence_audit', 'status': 'completed', 'scientific_unit': 'sample', 'patient_level_inference_available': explicit_mapping, 'patient_mapping': mapping, 'primary_metric': 'frozen fibroblast-associated topic mean across all retained cells per sample', 'sample_counts': {'dfu': int(len(dfu)), 'healer': int(len(a)), 'nonhealer': int(len(b))}, 'effect': {'healer_minus_nonhealer_mean_difference': diff, 'welch_t': float(welch.statistic), 'welch_two_sided_p': float(welch.pvalue), 'exact_two_sided_permutation_p': _exact_p(np.r_[a, b], np.r_[np.ones(len(a)), np.zeros(len(b))]), 'bootstrap_95_ci': _bootstrap_ci(a, b, args.seed)}, 'equivalence': {'status': 'pre_specified_margin_run' if args.equivalence_bound is not None else 'sensitivity_grid_only', 'alpha': 0.05, 'margins': sensitivity, 'smallest_equivalent_margin': smallest, 'interpretation': 'TOST is sample-level and exploratory until a clinical margin is pre-specified; no patient-level equivalence claim is made'}, 'protocol': {'seed': args.seed, 'focus_topic': args.focus_topic, 'discovery_dir': args.discovery_dir, 'series_matrix': args.series_matrix, 'theta_sha256': _sha256(str(discovery / 'theta.npy')), 'obs_sha256': _sha256(str(discovery / 'obs.csv')), 'script_sha256': _sha256(__file__), 'patient_identifier_required': True, 'cell_to_sample_aggregation': 'arithmetic mean within GSM'}}
    dfu['title_prefix'] = dfu['title'].astype(str).str.extract(TITLE_PREFIX_RE, expand=False)
    dfu.to_csv(Path(args.output_dir) / 'sample_scores.csv', index=False)
    with open(Path(args.output_dir) / 'report.json', 'w') as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    print(json.dumps({'output_dir': args.output_dir, 'patient_level_inference_available': explicit_mapping, 'difference': diff, 'exact_p': report['effect']['exact_two_sided_permutation_p'], 'equivalence': sensitivity}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
