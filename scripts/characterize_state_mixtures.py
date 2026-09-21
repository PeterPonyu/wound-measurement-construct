#!/usr/bin/env python3
"""Characterize state mixtures."""
import argparse
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import parse_series_matrix
G_RE = re.compile('^G(\\d+)')
TOPIC_COLS = [f'topic_{i}' for i in range(15)]
LOW_MAX = 0.12
HIGH_MIN = 0.3
LINEAGES = ('fibroblast', 'keratinocyte', 'myeloid', 'pericyte_smc', 'endothelial', 't_cell')

def g_index_from_title(title) -> float:
    if not isinstance(title, str):
        return float('nan')
    hit = G_RE.search(title)
    return float(hit.group(1)) if hit else float('nan')

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    return obj

def assign_mode(mean_theta: float) -> str:
    if mean_theta >= HIGH_MIN:
        return 'high'
    if mean_theta <= LOW_MAX:
        return 'low'
    return 'mid'

def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = (x[ok], y[ok])
    n = int(x.size)
    if n < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {'rho': None, 'p': None, 'n': n}
    rho, p = spearmanr(x, y)
    return {'rho': float(rho), 'p': float(p), 'n': n}

def _mw(a: np.ndarray, b: np.ndarray) -> dict:
    if a.size == 0 or b.size == 0:
        return {}
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    return {'a_mean': float(a.mean()), 'b_mean': float(b.mean()), 'p': float(p), 'n_a': int(a.size), 'n_b': int(b.size)}

def fit_fibroblast_gmm(theta0: np.ndarray, seed: int=0) -> dict:
    x = theta0.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=10, random_state=seed)
    gmm.fit(x)
    means = gmm.means_.ravel()
    high_k = int(np.argmax(means))
    low_k = 1 - high_k
    post = gmm.predict_proba(x)[:, high_k]
    return {'low_mean': float(means[low_k]), 'high_mean': float(means[high_k]), 'weights': [float(gmm.weights_[low_k]), float(gmm.weights_[high_k])], 'post_high': post, 'map_high': np.argmax(gmm.predict_proba(x), axis=1) == high_k, 'midpoint': float(0.5 * (means[low_k] + means[high_k]))}

def per_sample_table(df: pd.DataFrame, pheno: pd.DataFrame) -> pd.DataFrame:
    fib = df[df['celltype'] == 'fibroblast']
    gmm = fit_fibroblast_gmm(fib['topic_0'].to_numpy())
    mid = gmm['midpoint']
    rows = []
    for (gsm, arm), g in fib.groupby(['gsm', 'arm'], observed=True):
        x = g['topic_0'].to_numpy()
        p10 = float(np.percentile(x, 10))
        p90 = float(np.percentile(x, 90))
        if p10 >= 0.1:
            shape = 'pure_high'
        elif p90 <= 0.15:
            shape = 'pure_low'
        elif p10 <= 0.05 and p90 >= 0.4:
            shape = 'mixed'
        else:
            shape = 'other'
        rows.append({'gsm': gsm, 'arm': arm, 'n_fib': int(x.size), 'topic0_mean': float(x.mean()), 'topic0_p10': p10, 'topic0_p50': float(np.percentile(x, 50)), 'topic0_p90': p90, 'mixing_weight': float((x > mid).mean()), 'shape': shape})
    sample = pd.DataFrame(rows)
    sample['mode'] = sample['topic0_mean'].map(assign_mode)
    n_tot = df.groupby('gsm', observed=True).size().rename('n_cells')
    sample = sample.merge(n_tot, on='gsm', how='left')
    for lin in LINEAGES:
        frac = df.assign(_hit=df['celltype'] == lin).groupby('gsm', observed=True)['_hit'].mean().rename(f'frac_{lin}')
        sample = sample.merge(frac, on='gsm', how='left')
    other = fib.groupby(['gsm'], observed=True)[TOPIC_COLS].mean().reset_index()
    sample = sample.merge(other, on='gsm', how='left')
    keep = ['gsm']
    if 'title' in pheno.columns:
        keep.append('title')
    meta = pheno[keep].drop_duplicates('gsm')
    sample = sample.merge(meta, on='gsm', how='left')
    if 'title' in sample.columns:
        sample['g_index'] = sample['title'].map(g_index_from_title)
    else:
        sample['g_index'] = np.nan
    return (sample, gmm)

def evaluate_hypotheses(sample: pd.DataFrame) -> dict:
    healer = sample[sample['arm'] == 'DFU-healer']
    high_h = healer[healer['mode'] == 'high']
    low_h = healer[healer['mode'] == 'low']
    shift = False
    mixture = False
    if len(high_h) and len(low_h):
        p10_high = float(high_h['topic0_p10'].median())
        p50_low = float(low_h['topic0_p50'].median())
        p90_low = float(low_h['topic0_p90'].median())
        shift = p10_high > p90_low
        mixture = not shift
    else:
        p10_high = p50_low = p90_low = None
    mix_corr = _spearman(sample['topic0_mean'].to_numpy(), sample['mixing_weight'].to_numpy())
    high_non_healer = sample[(sample['mode'] == 'high') & (sample['arm'] != 'DFU-healer')]
    healer_specific = len(high_non_healer) == 0
    dfu = sample[sample['arm'].isin(['DFU-healer', 'DFU-nonhealer'])]
    is_healer = (dfu['arm'] == 'DFU-healer').to_numpy()
    is_high = (dfu['mode'] == 'high').to_numpy()
    table = np.array([[int(np.sum(is_healer & is_high)), int(np.sum(is_healer & ~is_high))], [int(np.sum(~is_healer & is_high)), int(np.sum(~is_healer & ~is_high))]])
    _, fisher_p = fisher_exact(table, alternative='two-sided')
    covariates = {}
    for col in ['n_fib', 'n_cells', 'frac_fibroblast', 'frac_keratinocyte', 'frac_myeloid', 'frac_pericyte_smc', 'g_index', 'topic_2', 'topic_4', 'topic_6', 'topic_11']:
        if col not in sample.columns:
            continue
        covariates[col] = _spearman(sample['topic0_mean'].to_numpy(), sample[col].to_numpy())
    high_vs_low = {}
    if len(high_h) and len(low_h):
        for col in ['n_fib', 'frac_keratinocyte', 'frac_myeloid', 'frac_pericyte_smc', 'mixing_weight']:
            high_vs_low[col] = _mw(high_h[col].to_numpy(), low_h[col].to_numpy())
    return {'n_healer_high': int(len(high_h)), 'n_healer_low': int(len(low_h)), 'n_healer_mid': int((healer['mode'] == 'mid').sum()), 'p10_high_median': p10_high, 'p50_low_median': p50_low, 'p90_low_median': p90_low, 'SHIFT': bool(shift), 'MIXTURE': bool(mixture), 'mean_vs_mixing_weight': mix_corr, 'HEALER_SPECIFIC': bool(healer_specific), 'high_mode_non_healer': high_non_healer[['gsm', 'arm', 'topic0_mean', 'mixing_weight']].to_dict(orient='records'), 'fisher_healer_vs_highmode': {'table_healer_high_healer_other_nh_high_nh_other': table.tolist(), 'p': float(fisher_p)}, 'spearman_vs_sample_mean': covariates, 'healer_high_vs_low': high_vs_low, 'shape_counts': sample['shape'].value_counts().to_dict(), 'mixed_samples': sample.loc[sample['shape'] == 'mixed', ['gsm', 'arm', 'mode', 'topic0_mean', 'mixing_weight']].to_dict(orient='records'), 'verdict': 'MIXTURE — sample means track the fraction of a high-θ fibroblast state; that state is not restricted to healers.' if mixture and (not healer_specific) else 'MIXTURE — sample means track a two-state mixing weight.' if mixture else 'SHIFT — high-mode fibroblasts moved as a population.' if shift else 'UNRESOLVED — neither SHIFT nor MIXTURE criterion fired.'}

def analyse_cohort(parquet_path: str, series_matrix: str) -> dict:
    df = pd.read_parquet(parquet_path)
    pheno = parse_series_matrix(series_matrix) if os.path.exists(series_matrix) else pd.DataFrame({'gsm': []})
    sample, gmm = per_sample_table(df, pheno)
    sample_sorted = sample.sort_values('topic0_mean', ascending=False)
    hyp = evaluate_hypotheses(sample)
    log(f"  GMM fibroblast fibroblast-associated topic: low μ={gmm['low_mean']:.3f}, high μ={gmm['high_mean']:.3f}, midpoint={gmm['midpoint']:.3f}, weights={gmm['weights'][0]:.2f}/{gmm['weights'][1]:.2f}")
    log(f"  {hyp['verdict']}")
    log(f"  healer modes: high={hyp['n_healer_high']} mid={hyp['n_healer_mid']} low={hyp['n_healer_low']}")
    if hyp['p10_high_median'] is not None:
        log(f"  median p10(high healers)={hyp['p10_high_median']:.3f} vs p50(low healers)={hyp['p50_low_median']:.3f} / p90(low healers)={hyp['p90_low_median']:.3f}")
        log(f"  SHIFT={hyp['SHIFT']}  MIXTURE={hyp['MIXTURE']}  HEALER_SPECIFIC={hyp['HEALER_SPECIFIC']}")
    rho = hyp['mean_vs_mixing_weight']
    log(f"  Spearman(sample mean, mixing weight) ρ={rho['rho']:.3f} p={rho['p']:.2e}")
    for col in ('n_fib', 'frac_keratinocyte', 'frac_myeloid', 'g_index', 'topic_4', 'topic_11'):
        hit = hyp['spearman_vs_sample_mean'].get(col) or {}
        if hit.get('rho') is not None:
            log(f"  Spearman(mean, {col}) ρ={hit['rho']:.3f} p={hit['p']:.3g}")
    log(f"  Fisher healer×high-mode p={hyp['fisher_healer_vs_highmode']['p']:.4f} table={hyp['fisher_healer_vs_highmode']['table_healer_high_healer_other_nh_high_nh_other']}")
    for rec in hyp['high_mode_non_healer']:
        log(f"  HIGH outside healer: {rec['gsm']} {rec['arm']} mean={rec['topic0_mean']:.3f} mix={rec['mixing_weight']:.3f}")
    log('  per-sample (fibroblast fibroblast-associated topic):')
    log('    gsm        arm            mode  shape      mean  p10   p90   mix   n_fib  title')
    for r in sample_sorted.itertuples():
        title = getattr(r, 'title', '') or ''
        short = title.split(':')[0] if title else ''
        log(f'    {r.gsm} {r.arm:<14} {r.mode:<5} {r.shape:<9} {r.topic0_mean:.3f} {r.topic0_p10:.3f} {r.topic0_p90:.3f} {r.mixing_weight:.3f} {r.n_fib:5d}  {short}')
    out = sample_sorted.copy()
    return {'gmm': {k: gmm[k] for k in ('low_mean', 'high_mean', 'weights', 'midpoint')}, 'hypotheses': hyp, 'per_sample': out.to_dict(orient='records')}

def self_test() -> None:
    """Planted two-state fibroblasts: sample means must recover mixing weights."""
    rng = np.random.default_rng(0)
    low, high = (0.03, 0.55)
    records = []
    planted = [('H', 'DFU-healer', 0.6)] * 4 + [('L', 'DFU-healer', 0.05)] * 4 + [('NDH', 'Non-diabetic', 0.8)] + [('NDL', 'Non-diabetic', 0.02)]
    for i, (tag, arm, pi) in enumerate(planted):
        n = 200
        n_high = int(round(pi * n))
        theta = np.concatenate([rng.normal(high, 0.04, n_high).clip(0.001, 0.95), rng.normal(low, 0.01, n - n_high).clip(0.001, 0.95)])
        gsm = f'GSM{i:07d}'
        for t in theta:
            records.append({'topic_0': float(t), **{f'topic_{k}': 0.0 for k in range(1, 15)}, 'gsm': gsm, 'arm': arm, 'celltype': 'fibroblast'})
        for _ in range(20):
            records.append({'topic_0': 0.01, **{f'topic_{k}': 0.0 for k in range(1, 15)}, 'gsm': gsm, 'arm': arm, 'celltype': 'keratinocyte'})
    df = pd.DataFrame(records)
    tmp_parq = os.path.join('/tmp', 'bimodal_selftest.parquet')
    df.to_parquet(tmp_parq, index=False)
    pheno = pd.DataFrame({'gsm': df['gsm'].unique(), 'title': [f'G{i + 1}: mock' for i in range(df['gsm'].nunique())]})
    sample, gmm = per_sample_table(df, pheno)
    hyp = evaluate_hypotheses(sample)
    assert hyp['MIXTURE'] and (not hyp['SHIFT']), hyp
    assert not hyp['HEALER_SPECIFIC'], hyp['high_mode_non_healer']
    assert hyp['mean_vs_mixing_weight']['rho'] > 0.9, hyp['mean_vs_mixing_weight']
    nd_high = sample[(sample['arm'] == 'Non-diabetic') & (sample['mode'] == 'high')]
    assert len(nd_high) == 1, sample[['gsm', 'arm', 'mode', 'topic0_mean']]
    log(f"  self-test ok (MIXTURE={hyp['MIXTURE']}, healer-specific={hyp['HEALER_SPECIFIC']}, ρ(mean,mix)={hyp['mean_vs_mixing_weight']['rho']:.3f})")
    os.remove(tmp_parq)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--celltype-dir', default='outputs/celltype_resolved')
    ap.add_argument('--output-dir', default='outputs/bimodal_stratification')
    ap.add_argument('--discovery-parquet', default='GSE165816_discovery_theta.parquet')
    ap.add_argument('--validation-parquet', default='GSE231643_validation_theta.parquet')
    ap.add_argument('--discovery-series', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--validation-series', default='data/raw/GSE231643/GSE231643_series_matrix.txt.gz')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log('self-test on planted two-state fibroblasts')
    self_test()
    report = {'mode_cuts': {'low_max': LOW_MAX, 'high_min': HIGH_MIN}, 'cohorts': {}}
    jobs = [('GSE165816_discovery', os.path.join(args.celltype_dir, args.discovery_parquet), args.discovery_series), ('GSE231643_validation', os.path.join(args.celltype_dir, args.validation_parquet), args.validation_series)]
    for name, parquet, series in jobs:
        log()
        log('=' * 68)
        log(f'COHORT {name}')
        log('=' * 68)
        if not os.path.exists(parquet):
            log(f'  SKIP: {parquet} missing (run scripts/compare_lineage_readouts.py)')
            continue
        report['cohorts'][name] = analyse_cohort(parquet, series)
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(_jsonable(report), fh, indent=2)
    log()
    log(f'wrote {out}')
    disc = report['cohorts'].get('GSE165816_discovery', {}).get('hypotheses')
    if not disc:
        return 2
    return 0 if disc['MIXTURE'] else 1
if __name__ == '__main__':
    raise SystemExit(main())
