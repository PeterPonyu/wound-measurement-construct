#!/usr/bin/env python3
"""Scan sample covariates."""
import argparse
import gzip
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import parse_series_matrix

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up q-values."""
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--theta-parquet', default='outputs/celltype_resolved/GSE165816_discovery_theta.parquet')
    ap.add_argument('--triage-report', default='outputs/artifact_triage/report.json')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--output-dir', default='outputs/stratifier_sweep')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--lineage', default='fibroblast')
    ap.add_argument('--theta-cut', type=float, default=0.184)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    col = f'topic_{args.focus_topic}'
    tdf = pd.read_parquet(args.theta_parquet)
    topic_cols = [c for c in tdf.columns if c.startswith('topic_')]
    fib = tdf[tdf['celltype'] == args.lineage].copy()
    fib['is_high'] = fib[col] >= args.theta_cut
    per = fib.groupby(['gsm', 'arm'], observed=True).agg(weight=('is_high', 'mean'), n_fib=('is_high', 'size')).reset_index()
    log(f"{len(per)} samples with {args.lineage}s; weight range {per['weight'].min():.3f}-{per['weight'].max():.3f}")
    feats = per[['gsm', 'arm', 'weight']].copy()
    feats['n_fibroblasts'] = per['n_fib'].to_numpy()
    other = fib.groupby('gsm', observed=True)[topic_cols].mean().reset_index()
    for c in topic_cols:
        if c != col:
            feats[f'fib_mean_{c}'] = feats['gsm'].map(dict(zip(other['gsm'], other[c])))
    comp = tdf.groupby(['gsm', 'celltype'], observed=True).size().unstack(fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0)
    for lin in comp.columns:
        feats[f'frac_{lin}'] = feats['gsm'].map(comp[lin].to_dict())
    feats['n_cells_total'] = feats['gsm'].map(tdf.groupby('gsm', observed=True).size().to_dict())
    if os.path.exists(args.triage_report):
        tri = json.load(open(args.triage_report))
        tmap = {r['gsm']: r for r in tri.get('per_sample', [])}
        for key in ('diss', 'umi', 'ptprc_pos'):
            feats[f'tech_{key}'] = feats['gsm'].map({g: r.get(key) for g, r in tmap.items()})
    pheno = parse_series_matrix(args.series_matrix)
    extra = {}
    with gzip.open(args.series_matrix, 'rt', errors='ignore') as fh:
        gsms, rows = ([], {})
        for line in fh:
            if line.startswith('!series_matrix_table_begin'):
                break
            parts = [p.strip().strip('"') for p in line.rstrip('\n').split('\t')]
            if parts[0] == '!Sample_geo_accession':
                gsms = parts[1:]
            elif parts[0] in ('!Sample_instrument_model', '!Sample_submission_date', '!Sample_last_update_date', '!Sample_library_selection'):
                rows[parts[0].replace('!Sample_', '')] = parts[1:]
    for key, values in rows.items():
        if len(values) == len(gsms):
            extra[key] = dict(zip(gsms, values))
    for key, mapping in extra.items():
        vals = feats['gsm'].map(mapping)
        if vals.nunique(dropna=True) > 1:
            feats[f'meta_{key}'] = pd.factorize(vals)[0].astype(float)
            log(f'  meta {key}: {vals.nunique()} distinct values, kept')
        else:
            log(f'  meta {key}: constant across samples, dropped')
    if 'title' in pheno.columns:
        tmap2 = dict(zip(pheno['gsm'], pheno['title']))
        feats['meta_g_number'] = feats['gsm'].map(lambda g: float(str(tmap2.get(g, '')).split(':')[0].lstrip('Gg').rstrip('Aa') or 'nan') if tmap2.get(g) else np.nan)
    results = []
    y = feats['weight'].to_numpy(dtype=float)
    for c in feats.columns:
        if c in ('gsm', 'arm', 'weight'):
            continue
        x = pd.to_numeric(feats[c], errors='coerce').to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6 or np.unique(x[ok]).size < 2:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        results.append({'covariate': c, 'rho': float(rho), 'p': float(p), 'n': int(ok.sum())})
    res = pd.DataFrame(results)
    res['q_bh'] = bh_qvalues(res['p'].to_numpy())
    res = res.reindex(res['rho'].abs().sort_values(ascending=False).index)
    log()
    log(f'swept {len(res)} covariates against the per-sample mixture weight')
    log(f"{'covariate':<28}{'rho':>7}{'p':>9}{'q_BH':>9}")
    for r in res.itertuples():
        flag = '  <-- survives BH' if r.q_bh < 0.05 else ''
        log(f'{r.covariate:<28}{r.rho:>+7.2f}{r.p:>9.3f}{r.q_bh:>9.3f}{flag}')
    hits = res[res['q_bh'] < 0.05]
    log()
    if len(hits) == 0:
        log(f'NO covariate survives BH correction across {len(res)} tests.')
        log('The stratifier remains unidentified after a systematic search, which is a stronger statement than it being merely untested.')
    else:
        log(f"{len(hits)} covariate(s) survive BH: {', '.join(hits['covariate'].tolist())}")
    feats.to_csv(os.path.join(args.output_dir, 'sample_features.csv'), index=False)
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump({'n_samples': int(len(per)), 'n_covariates': int(len(res)), 'sweep': res.to_dict(orient='records'), 'survives_bh': hits['covariate'].tolist()}, fh, indent=2)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
