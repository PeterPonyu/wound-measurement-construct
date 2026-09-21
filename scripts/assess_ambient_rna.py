#!/usr/bin/env python3
"""Assess ambient rna."""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.celltype_markers import MARKER_PANEL, score_cell_types
from wound_models.data_adapter import library_normalize, load_dense_csv_tar, mito_fraction
from wound_models.topic_model import TopicModel
KEEP_ARMS = {'DFU-healer', 'DFU-nonhealer', 'Non-diabetic'}
RHO_SURVIVE = 0.8
IMMUNE_STRICT = ['PTPRC', 'LYZ', 'CD68', 'CD163', 'AIF1', 'TYROBP', 'FCER1G', 'CD3D', 'CD3E', 'MS4A1', 'CD79A', 'MZB1', 'JCHAIN', 'IGKC']

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _rows(m) -> int:
    r, _ = m.shape
    return int(r)

def bimodality_bic(x: np.ndarray, seed: int=0) -> dict:
    """2-component vs 1-component GMM BIC. Positive delta favours bimodal."""
    v = np.asarray(x, dtype=float).reshape(-1, 1)
    v = v[np.isfinite(v).ravel()]
    if v.shape[0] < 8:
        return {'delta_bic': float('nan'), 'bimodal': False, 'n': int(v.shape[0])}
    b1 = GaussianMixture(1, random_state=seed).fit(v).bic(v)
    b2 = GaussianMixture(2, random_state=seed).fit(v).bic(v)
    return {'bic_1': float(b1), 'bic_2': float(b2), 'delta_bic': float(b1 - b2), 'bimodal': bool(b1 - b2 > 0), 'n': int(v.shape[0])}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/ambient_invariance')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--lineage', default='fibroblast')
    ap.add_argument('--theta-cut', type=float, default=0.184)
    ap.add_argument('--myeloid-quantile', type=float, default=0.25, help='keep fibroblasts below this myeloid-score quantile')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    report = {'theta_cut': args.theta_cut, 'rho_survive': RHO_SURVIVE}
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    log(f'loading {args.tar}')
    t0 = time.time()
    coh = load_dense_csv_tar(args.tar, series_matrix=args.series_matrix)
    log(f'loaded {coh!r} in {time.time() - t0:.1f}s')
    obs = coh.obs.copy()
    obs['arm'] = obs['disease']
    keep = obs['arm'].isin(list(KEEP_ARMS)).to_numpy() & (obs['tissue'].str.lower() == 'foot skin').to_numpy()
    mito = mito_fraction(coh.X, coh.genes)
    gpc = np.asarray((coh.X > 0).sum(axis=1)).ravel()
    keep = keep & (gpc >= args.min_genes_per_cell) & (mito <= args.max_mito)
    idx = np.flatnonzero(keep)
    X = coh.X[idx]
    obs = obs.iloc[idx].reset_index(drop=True)
    labels, _, _ = score_cell_types(X, coh.genes, panel=MARKER_PANEL)
    obs['celltype'] = labels
    lookup = {str(g): i for i, g in enumerate(coh.genes)}
    strict_cols = [lookup[g] for g in IMMUNE_STRICT if g in lookup]
    missing = [g for g in IMMUNE_STRICT if g not in lookup]
    log(f'strict immune genes used: {len(strict_cols)}/{len(IMMUNE_STRICT)}' + (f', missing {missing}' if missing else ''))
    obs['immune_counts'] = np.asarray(X[:, strict_cols].sum(axis=1)).ravel()
    Xlog = library_normalize(X).astype(np.float32).tocsr()
    Xlog.data = np.log1p(Xlog.data * 10000.0)
    mye_cols = [lookup[g] for g in MARKER_PANEL['myeloid'] if g in lookup]
    blk = np.asarray(Xlog[:, mye_cols].todense(), dtype=np.float32)
    mu, sd = (blk.mean(0, keepdims=True), blk.std(0, keepdims=True))
    sd[sd == 0] = 1.0
    obs['myeloid_score'] = ((blk - mu) / sd).mean(axis=1)
    pos = pd.Index(coh.genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    Xp = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(_rows(X), panel.size), dtype=np.int32)
    Xn = library_normalize(Xp)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopicModel(input_dim=n_panel, num_topics=n_topics).to(dev)
    model.load_state_dict(state)
    model.eval()
    th = []
    with torch.no_grad():
        for s in range(0, _rows(Xn), 4096):
            xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            th.append(model(xb)['theta'].cpu().numpy())
    theta = np.vstack(th)
    obs['theta0'] = theta[:, args.focus_topic]
    fib = obs[obs['celltype'] == args.lineage].copy()
    fib['is_high'] = fib['theta0'] >= args.theta_cut
    mye_cut = float(fib['myeloid_score'].quantile(args.myeloid_quantile))
    pure = fib[(fib['immune_counts'] == 0) & (fib['myeloid_score'] <= mye_cut)]
    log(f'fibroblasts {len(fib)} -> strictly immune-negative (zero counts on {len(strict_cols)} immune genes AND myeloid score <= q{args.myeloid_quantile:.0%}={mye_cut:.3f}): {len(pure)}')
    w_all = fib.groupby('gsm', observed=True)['is_high'].mean().rename('weight_all')
    w_pure = pure.groupby('gsm', observed=True)['is_high'].agg(['mean', 'size'])
    w_pure.columns = pd.Index(['weight_pure', 'n_pure'])
    w = pd.concat([w_all, w_pure], axis=1).dropna()
    w = w[w['n_pure'] >= 20]
    rho_p, p_p = spearmanr(w['weight_all'], w['weight_pure'])
    log(f'samples with >=20 pure fibroblasts: {len(w)}')
    log(f'weight_all vs weight_pure: rho={rho_p:+.3f} p={p_p:.2e}')
    bi_all = bimodality_bic(w['weight_all'].to_numpy())
    bi_pure = bimodality_bic(w['weight_pure'].to_numpy())
    comp = obs.groupby(['gsm', 'celltype'], observed=True).size().unstack(fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0)
    ptprc_rate = fib.assign(pp=fib['immune_counts'] > 0).groupby('gsm', observed=True)['pp'].mean()
    feats = pd.DataFrame({'weight': w_all, 'b_plasma': comp.get('b_plasma', pd.Series(dtype=float)), 't_cell': comp.get('t_cell', pd.Series(dtype=float)), 'myeloid': comp.get('myeloid', pd.Series(dtype=float)), 'immune_pos_rate': ptprc_rate}).dropna()
    placebo_cols = [c for c in ('pericyte_smc', 'endothelial', 'melanocyte', 'sweat_gland') if c in comp.columns]
    feats_pl = pd.DataFrame({'weight': w_all, **{c: comp[c] for c in placebo_cols}}).dropna()
    Xc = feats[['b_plasma', 't_cell', 'myeloid', 'immune_pos_rate']].to_numpy()
    yv = feats['weight'].to_numpy()
    lr = LinearRegression().fit(Xc, yv)
    resid = yv - lr.predict(Xc)
    r2 = float(lr.score(Xc, yv))
    bi_resid = bimodality_bic(resid)
    log(f'immune covariates explain R^2={r2:.3f} of the sample mixture weight')
    Xpl = feats_pl[placebo_cols].to_numpy()
    ypl = feats_pl['weight'].to_numpy()
    lr_pl = LinearRegression().fit(Xpl, ypl)
    resid_pl = ypl - lr_pl.predict(Xpl)
    r2_pl = float(lr_pl.score(Xpl, ypl))
    bi_resid_pl = bimodality_bic(resid_pl)
    log(f"placebo covariates ({', '.join(placebo_cols)}) explain R^2={r2_pl:.3f}")
    survives_purity = bool(rho_p > RHO_SURVIVE)
    ambient_verdict = 'INVARIANT to ambient immune RNA' if survives_purity and bi_pure['bimodal'] else 'CONTAMINATED by ambient immune RNA'
    placebo_also_flattens = not bi_resid_pl['bimodal']
    if placebo_also_flattens:
        separability_verdict = 'UNDETERMINED - the placebo covariates flatten the residual just as well, so test B is overfitting at this sample size'
    elif bi_resid['bimodal']:
        separability_verdict = 'SEPARABLE from tissue immune composition'
    else:
        separability_verdict = 'NOT SEPARABLE from tissue immune composition'
    report.update({'n_fibroblasts': int(len(fib)), 'n_pure': int(len(pure)), 'myeloid_cut': mye_cut, 'immune_genes_used': len(strict_cols), 'test_A_purity': {'rho': float(rho_p), 'p': float(p_p), 'n_samples': int(len(w)), 'survives': survives_purity, 'bimodal_all': bi_all, 'bimodal_pure': bi_pure}, 'test_B_residual': {'r2_immune_covariates': r2, 'bimodal_residual': bi_resid, 'coefficients': dict(zip(['b_plasma', 't_cell', 'myeloid', 'immune_pos_rate'], [float(c) for c in lr.coef_]))}, 'test_B_placebo': {'covariates': placebo_cols, 'r2': r2_pl, 'bimodal_residual': bi_resid_pl, 'also_flattens': placebo_also_flattens}, 'verdict_ambient_contamination': ambient_verdict, 'verdict_separability': separability_verdict})
    w.reset_index().to_csv(os.path.join(args.output_dir, 'sample_weights.csv'), index=False)
    log()
    log('=' * 70)
    log(f"TEST A purity     rho(all, pure)={rho_p:+.3f} -> {('survives' if survives_purity else 'FAILS')} (cut {RHO_SURVIVE})")
    log(f"                  bimodal all  : dBIC={bi_all['delta_bic']:+.1f} ({('yes' if bi_all['bimodal'] else 'no')})")
    log(f"                  bimodal pure : dBIC={bi_pure['delta_bic']:+.1f} ({('yes' if bi_pure['bimodal'] else 'no')})")
    log(f"TEST B residual   immune covariates R^2={r2:.3f}, bimodal resid dBIC={bi_resid['delta_bic']:+.1f} ({('yes' if bi_resid['bimodal'] else 'no')})")
    log(f"  PLACEBO         {', '.join(placebo_cols)} R^2={r2_pl:.3f}, bimodal resid dBIC={bi_resid_pl['delta_bic']:+.1f} ({('yes' if bi_resid_pl['bimodal'] else 'no')})")
    log('=' * 70)
    log(f'CONTAMINATION  : {ambient_verdict}')
    log(f'SEPARABILITY   : {separability_verdict}')
    log('=' * 70)
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
