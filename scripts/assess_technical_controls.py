#!/usr/bin/env python3
"""Assess technical controls."""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.celltype_markers import MARKER_PANEL, score_cell_types
from wound_models.data_adapter import library_normalize, load_dense_csv_tar, mito_fraction
from wound_models.topic_model import TopicModel
DISSOCIATION_GENES = ['FOS', 'FOSB', 'JUN', 'JUNB', 'JUND', 'EGR1', 'ATF3', 'HSPA1A', 'HSPA1B', 'HSP90AA1', 'HSP90AB1', 'DNAJA1', 'DNAJB1', 'DNAJB4', 'HSPH1', 'HSPB1', 'SOCS3', 'ZFP36', 'IER2', 'IER3', 'DUSP1', 'KLF6', 'NR4A1', 'PPP1R15A', 'RHOB', 'CEBPB', 'CEBPD', 'ID1', 'ID2', 'MYC', 'BTG1', 'BTG2', 'CCN1', 'HSPE1', 'HSPD1', 'UBC']
KEEP_ARMS = {'DFU-healer', 'DFU-nonhealer', 'Non-diabetic'}
SMD_REJECT = 0.8
PTPRC_RATIO_REJECT = 2.0
RHO_REJECT = 0.6

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _rows(m) -> int:
    r, _ = m.shape
    return int(r)

def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (Cohen's d, pooled SD)."""
    if a.size < 2 or b.size < 2:
        return float('nan')
    s = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float('nan')

def gene_set_score(Xlog: sp.csr_matrix, genes: np.ndarray, gene_set: list) -> tuple:
    """Mean z-scored log expression over the genes of `gene_set` that exist."""
    lookup = {str(g): i for i, g in enumerate(genes)}
    cols = [lookup[g] for g in gene_set if g in lookup]
    missing = [g for g in gene_set if g not in lookup]
    if not cols:
        return (np.zeros(_rows(Xlog)), missing)
    block = np.asarray(Xlog[:, cols].todense(), dtype=np.float32)
    mu = block.mean(axis=0, keepdims=True)
    sd = block.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (((block - mu) / sd).mean(axis=1), missing)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/artifact_triage')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--lineage', default='fibroblast')
    ap.add_argument('--theta-cut', type=float, default=0.184, help='GMM boundary between the low and high fibroblast-associated topic states, taken from scripts/characterize_state_mixtures.py')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    report = {'theta_cut': args.theta_cut, 'thresholds': {'smd_reject': SMD_REJECT, 'ptprc_ratio_reject': PTPRC_RATIO_REJECT, 'rho_reject': RHO_REJECT}}
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    top_focus = [str(panel[j]) for j in np.argsort(-beta[args.focus_topic])[:10]]
    log(f"topic {args.focus_topic} top genes: {', '.join(top_focus)}")
    shared = sorted(set(top_focus) & set(DISSOCIATION_GENES))
    log(f"topic {args.focus_topic} genes that are ALSO dissociation-artifact genes: {shared or 'none'}")
    report['topic_dissociation_gene_overlap'] = shared
    log(f'loading {args.tar}')
    t0 = time.time()
    coh = load_dense_csv_tar(args.tar, series_matrix=args.series_matrix)
    log(f'loaded {coh!r} in {time.time() - t0:.1f}s')
    obs = coh.obs.copy()
    obs['arm'] = obs['disease']
    keep = obs['arm'].isin(list(KEEP_ARMS)).to_numpy() & (obs['tissue'].str.lower() == 'foot skin').to_numpy()
    mito = mito_fraction(coh.X, coh.genes)
    gpc = np.asarray((coh.X > 0).sum(axis=1)).ravel()
    umi = np.asarray(coh.X.sum(axis=1)).ravel()
    keep = keep & (gpc >= args.min_genes_per_cell) & (mito <= args.max_mito)
    idx = np.flatnonzero(keep)
    X = coh.X[idx]
    obs = obs.iloc[idx].reset_index(drop=True)
    obs['umi'] = umi[idx]
    obs['n_genes'] = gpc[idx]
    obs['mito'] = mito[idx]
    log(f'cells after QC: {idx.size}')
    labels, _, _ = score_cell_types(X, coh.genes, panel=MARKER_PANEL)
    obs['celltype'] = labels
    Xlog = library_normalize(X).astype(np.float32).tocsr()
    Xlog.data = np.log1p(Xlog.data * 10000.0)
    diss, diss_missing = gene_set_score(Xlog, coh.genes, DISSOCIATION_GENES)
    obs['dissociation'] = diss
    log(f"dissociation genes missing from assay: {diss_missing or 'none'}")
    lookup = {str(g): i for i, g in enumerate(coh.genes)}
    if 'PTPRC' not in lookup:
        log('FATAL: PTPRC absent from the assay; the doublet test cannot run')
        return 2
    obs['ptprc'] = np.asarray(X[:, lookup['PTPRC']].todense()).ravel()
    obs['ptprc_pos'] = obs['ptprc'] > 0
    myeloid, _ = gene_set_score(Xlog, coh.genes, MARKER_PANEL['myeloid'])
    obs['myeloid_score'] = myeloid
    kera, _ = gene_set_score(Xlog, coh.genes, MARKER_PANEL['keratinocyte'])
    obs['keratinocyte_score'] = kera
    pos = pd.Index(coh.genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    Xp = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(_rows(X), panel.size), dtype=np.int32)
    Xn = library_normalize(Xp)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopicModel(input_dim=n_panel, num_topics=n_topics).to(dev)
    model.load_state_dict(state)
    model.eval()
    thetas = []
    with torch.no_grad():
        for s in range(0, _rows(Xn), 4096):
            xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            thetas.append(model(xb)['theta'].cpu().numpy())
    theta = np.vstack(thetas)
    obs['theta0'] = theta[:, args.focus_topic]
    fib = obs[obs['celltype'] == args.lineage].copy()
    fib['state'] = np.where(fib['theta0'] >= args.theta_cut, 'high', 'low')
    n_hi = int((fib['state'] == 'high').sum())
    n_lo = int((fib['state'] == 'low').sum())
    log(f'fibroblasts: {n_hi} fibroblast-associated topic-high, {n_lo} low (cut theta0 >= {args.theta_cut})')
    report['fibroblast_counts'] = {'high': n_hi, 'low': n_lo}
    hi = fib[fib['state'] == 'high']
    lo = fib[fib['state'] == 'low']
    d_smd = smd(hi['dissociation'].to_numpy(), lo['dissociation'].to_numpy())
    per_sample = fib.groupby('gsm', observed=True).agg(weight=('state', lambda s: float((s == 'high').mean())), diss=('dissociation', 'mean'), umi=('umi', 'median'), ptprc_pos=('ptprc_pos', 'mean')).reset_index()
    rho_d, p_d = spearmanr(per_sample['weight'], per_sample['diss'])
    h1 = {'smd': d_smd, 'high_mean': float(hi['dissociation'].mean()), 'low_mean': float(lo['dissociation'].mean()), 'rho_weight_vs_sample_dissociation': float(rho_d), 'p': float(p_d), 'rejects_biology': bool(abs(d_smd) > SMD_REJECT and abs(rho_d) > RHO_REJECT)}
    report['H1_dissociation'] = h1
    r_hi = float(hi['ptprc_pos'].mean())
    r_lo = float(lo['ptprc_pos'].mean())
    ratio = r_hi / r_lo if r_lo > 0 else float('inf')
    h2 = {'ptprc_pos_rate_high': r_hi, 'ptprc_pos_rate_low': r_lo, 'rate_ratio': ratio, 'myeloid_score_smd': smd(hi['myeloid_score'].to_numpy(), lo['myeloid_score'].to_numpy()), 'keratinocyte_score_smd': smd(hi['keratinocyte_score'].to_numpy(), lo['keratinocyte_score'].to_numpy()), 'rejects_biology': bool(ratio > PTPRC_RATIO_REJECT)}
    report['H2_doublet'] = h2
    u_smd = smd(np.log10(hi['umi'].to_numpy() + 1), np.log10(lo['umi'].to_numpy() + 1))
    g_smd = smd(hi['n_genes'].to_numpy().astype(float), lo['n_genes'].to_numpy().astype(float))
    h3 = {'log10_umi_smd': u_smd, 'n_genes_smd': g_smd, 'median_umi_high': float(hi['umi'].median()), 'median_umi_low': float(lo['umi'].median()), 'mito_smd': smd(hi['mito'].to_numpy(), lo['mito'].to_numpy()), 'rejects_biology': bool(abs(u_smd) > SMD_REJECT or abs(g_smd) > SMD_REJECT)}
    report['H3_depth'] = h3
    fib_idx = np.flatnonzero((obs['celltype'] == args.lineage).to_numpy())
    panel_umi = np.asarray(Xp.sum(axis=1)).ravel()
    target = int(np.percentile(panel_umi[fib_idx], 25))
    keep_ds = fib_idx[panel_umi[fib_idx] >= target]
    log()
    log(f"depth control: downsampling {keep_ds.size} fibroblasts to {target} PANEL UMI (whole-transcriptome median was {np.median(obs['umi'].to_numpy()[fib_idx]):.0f}; dropping {fib_idx.size - keep_ds.size} cells below target)")
    rng = np.random.default_rng(0)
    Xds = Xp[keep_ds].tocsr().astype(np.int64).copy()
    n_changed = 0
    for i in range(_rows(Xds)):
        lo_i, hi_i = (Xds.indptr[i], Xds.indptr[i + 1])
        row = Xds.data[lo_i:hi_i]
        tot = int(row.sum())
        if tot > target:
            Xds.data[lo_i:hi_i] = rng.multivariate_hypergeometric(row.astype(int), target)
            n_changed += 1
    Xds = sp.csr_matrix(Xds).astype(np.int32)
    Xds.eliminate_zeros()
    ds_tot = np.asarray(Xds.sum(axis=1)).ravel()
    log(f'  downsampled {n_changed}/{_rows(Xds)} cells; panel UMI after: min={ds_tot.min()} max={ds_tot.max()} (target {target})')
    assert n_changed > 0, 'depth control was a no-op; target unit is wrong'
    assert ds_tot.max() == target, f'downsampling left cells above target: max={ds_tot.max()} > {target}'
    Xnd = library_normalize(Xds)
    th_ds = []
    with torch.no_grad():
        for s in range(0, _rows(Xnd), 4096):
            xb = torch.from_numpy(np.asarray(Xnd[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            th_ds.append(model(xb)['theta'].cpu().numpy())
    theta_ds = np.vstack(th_ds)[:, args.focus_topic]
    ds = obs.iloc[keep_ds][['gsm', 'arm', 'umi']].copy().reset_index(drop=True)
    ds['panel_umi_after'] = ds_tot
    ds['theta0_before'] = obs['theta0'].to_numpy()[keep_ds]
    ds['theta0_after'] = theta_ds
    ds['state_before'] = np.where(ds['theta0_before'] >= args.theta_cut, 'high', 'low')
    ds['state_after'] = np.where(ds['theta0_after'] >= args.theta_cut, 'high', 'low')
    w = ds.groupby('gsm', observed=True).agg(before=('state_before', lambda s: float((s == 'high').mean())), after=('state_after', lambda s: float((s == 'high').mean()))).reset_index()
    rho_w, p_w = spearmanr(w['before'], w['after'])
    agree = float((ds['state_before'] == ds['state_after']).mean())
    hi_ds = ds[ds['state_after'] == 'high']
    lo_ds = ds[ds['state_after'] == 'low']
    umi_smd_after = smd(np.log10(hi_ds['umi'].to_numpy() + 1), np.log10(lo_ds['umi'].to_numpy() + 1))
    h3b = {'target_panel_umi': target, 'cells_used': int(keep_ds.size), 'cells_downsampled': int(n_changed), 'panel_umi_after_is_constant': bool(ds_tot.min() == ds_tot.max()), 'cells_dropped': int(fib_idx.size - keep_ds.size), 'high_fraction_before': float((ds['state_before'] == 'high').mean()), 'high_fraction_after': float((ds['state_after'] == 'high').mean()), 'cell_label_agreement': agree, 'rho_sample_weight_before_after': float(rho_w), 'p': float(p_w), 'log10_umi_smd_after_matching': umi_smd_after, 'survives_depth_control': bool(rho_w > 0.8)}
    report['H3b_depth_control'] = h3b
    survives = not (h1['rejects_biology'] or h2['rejects_biology'] or h3['rejects_biology']) and h3b['survives_depth_control']
    report['H4_genuine_state'] = {'survives': survives}
    report['per_sample'] = per_sample.to_dict(orient='records')
    report['depth_control_per_sample'] = w.to_dict(orient='records')
    _, p_diss = mannwhitneyu(hi['dissociation'], lo['dissociation'], alternative='two-sided')
    log()
    log('=' * 70)
    log(f"H1 DISSOCIATION  SMD={d_smd:+.2f} (high {h1['high_mean']:+.3f} vs low {h1['low_mean']:+.3f}), sample rho={rho_d:+.2f} p={p_d:.3f}")
    log(f'                 cell-level MWU p={p_diss:.2e} (n is huge; read the SMD)')
    log(f"                 -> {('REJECTS biology' if h1['rejects_biology'] else 'does not reject')}")
    log(f'H2 DOUBLET       PTPRC+ rate high={r_hi:.3%} low={r_lo:.3%}, ratio={ratio:.2f}x')
    log(f"                 myeloid SMD={h2['myeloid_score_smd']:+.2f}, keratinocyte SMD={h2['keratinocyte_score_smd']:+.2f}")
    log(f"                 -> {('REJECTS biology' if h2['rejects_biology'] else 'does not reject')}")
    log(f"H3 DEPTH         log10UMI SMD={u_smd:+.2f} (median {h3['median_umi_high']:.0f} vs {h3['median_umi_low']:.0f}), nGenes SMD={g_smd:+.2f}, mito SMD={h3['mito_smd']:+.2f}")
    log(f"                 -> {('REJECTS biology' if h3['rejects_biology'] else 'does not reject')}")
    log(f"H3b DEPTH CTRL   {n_changed} cells downsampled to exactly {target} panel UMI ({h3b['cells_used']} used, {h3b['cells_dropped']} dropped)")
    log(f'                 panel depth now identical across cells; residual whole-transcriptome log10UMI SMD={umi_smd_after:+.2f}')
    log(f"                 high fraction {h3b['high_fraction_before']:.3f} -> {h3b['high_fraction_after']:.3f}, cell-label agreement={agree:.1%}")
    log(f'                 sample mixture weight before vs after: rho={rho_w:+.3f} p={p_w:.2e}')
    log(f"                 -> {('survives depth control' if h3b['survives_depth_control'] else 'FAILS depth control')}")
    log('=' * 70)
    log(f"H4 GENUINE STATE {('SURVIVES all artifact tests incl. depth control' if survives else 'DOES NOT survive')}")
    log('=' * 70)
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0 if survives else 1
if __name__ == '__main__':
    raise SystemExit(main())
