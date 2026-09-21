#!/usr/bin/env python3
"""Compare lineage readouts."""
import argparse
import json
import os
import sys
import time
from math import comb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.celltype_markers import MARKER_PANEL, marker_overlap_with, score_cell_types
from wound_models.data_adapter import load_10x_mtx_tar, load_dense_csv_tar, library_normalize, mito_fraction
from wound_models.topic_model import TopicModel
KEEP_ARMS = {'DFU-healer', 'DFU-nonhealer', 'Non-diabetic'}
DECLARED_DFU_PATIENTS = {'GSE165816_discovery': 11, 'GSE231643_validation': 8}

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def healing_status(title: str) -> str:
    t = str(title).strip().upper()
    return 'DFU-nonhealer' if t.startswith('NH') else 'DFU-healer' if t.startswith('H') else 'unknown'

def _rows(m) -> int:
    r, _ = m.shape
    return int(r)

def bootstrap_ratio_ci(h: np.ndarray, nh: np.ndarray, n_boot: int=20000, seed: int=0) -> tuple:
    """Percentile CI for mean(h)/mean(nh), resampling samples within each arm."""
    rng = np.random.default_rng(seed)
    hb = rng.choice(h, size=(n_boot, h.size), replace=True).mean(axis=1)
    nb = rng.choice(nh, size=(n_boot, nh.size), replace=True).mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = hb / nb
    ratios = ratios[np.isfinite(ratios)]
    return (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))

def project(Xp: sp.csr_matrix, model: TopicModel, dev: str) -> np.ndarray:
    Xn = library_normalize(Xp)
    out = []
    with torch.no_grad():
        for s in range(0, _rows(Xn), 4096):
            xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            out.append(model(xb)['theta'].cpu().numpy())
    return np.vstack(out)

def reindex_to_panel(X: sp.csr_matrix, genes: np.ndarray, panel: np.ndarray) -> tuple:
    pos = pd.Index(genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    Xp = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(_rows(X), panel.size), dtype=np.int32)
    return (Xp, int(present.size))

def arm_test(sample_means: pd.DataFrame, col: str) -> dict:
    h = sample_means.loc[sample_means['arm'] == 'DFU-healer', col].to_numpy()
    nh = sample_means.loc[sample_means['arm'] == 'DFU-nonhealer', col].to_numpy()
    if h.size == 0 or nh.size == 0:
        return {}
    _, p = mannwhitneyu(h, nh, alternative='two-sided')
    lo, hi = bootstrap_ratio_ci(h, nh)
    return {'healer_mean': float(h.mean()), 'nonhealer_mean': float(nh.mean()), 'ratio': float(h.mean() / nh.mean()) if nh.mean() > 0 else None, 'ratio_ci95': [lo, hi], 'p': float(p), 'n_healer': int(h.size), 'n_nonhealer': int(nh.size), 'healer_values': [float(v) for v in h], 'nonhealer_values': [float(v) for v in nh]}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/celltype_resolved')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--lineage', default='fibroblast')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    ap.add_argument('--min-cells-per-sample', type=int, default=50, help='a sample needs this many cells of the lineage to be tested')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    report = {'focus_topic': args.focus_topic, 'lineage': args.lineage}
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    top_focus = [str(panel[j]) for j in np.argsort(-beta[args.focus_topic])[:10]]
    log(f'frozen model K={n_topics}, panel={n_panel}')
    log(f"topic {args.focus_topic} top genes: {', '.join(top_focus)}")
    overlap = marker_overlap_with(top_focus)
    log(f"marker panel overlap with topic {args.focus_topic}: {overlap or 'none'}")
    if args.lineage in overlap:
        log(f'FATAL: the {args.lineage} marker panel shares {overlap[args.lineage]} with the topic under test; the cell-type call would not be independent of the result.')
        return 2
    report['circularity_audit'] = {'topic_top_genes': top_focus, 'marker_overlap': overlap, 'lineage_is_independent': True}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopicModel(input_dim=n_panel, num_topics=n_topics).to(dev)
    model.load_state_dict(state)
    model.eval()
    cohorts = [('GSE165816_discovery', 'dense', 'data/raw/GSE165816/GSE165816_RAW.tar', 'data/raw/GSE165816/GSE165816_series_matrix.txt.gz'), ('GSE231643_validation', '10x', 'data/raw/GSE231643/GSE231643_RAW.tar', 'data/raw/GSE231643/GSE231643_series_matrix.txt.gz')]
    report['cohorts'] = {}
    for name, kind, tar, sm in cohorts:
        log()
        log('=' * 68)
        log(f'COHORT {name}')
        log('=' * 68)
        t0 = time.time()
        coh = load_dense_csv_tar(tar, series_matrix=sm) if kind == 'dense' else load_10x_mtx_tar(tar, series_matrix=sm)
        log(f'loaded {coh!r} in {time.time() - t0:.1f}s')
        obs = coh.obs.copy()
        if 'disease' in obs.columns and obs['disease'].notna().any():
            obs['arm'] = obs['disease']
            tissue_ok = obs['tissue'].str.lower() == 'foot skin'
        else:
            obs['arm'] = obs['title'].map(healing_status)
            tissue_ok = pd.Series(True, index=obs.index)
        keep_arm = obs['arm'].isin(KEEP_ARMS) & tissue_ok
        mito = mito_fraction(coh.X, coh.genes)
        gpc = np.asarray((coh.X > 0).sum(axis=1)).ravel()
        keep = keep_arm.to_numpy() & (gpc >= args.min_genes_per_cell) & (mito <= args.max_mito)
        idx = np.flatnonzero(keep)
        X = coh.X[idx]
        obs = obs.iloc[idx].reset_index(drop=True)
        log(f"after arm filter + QC: {idx.size} cells; arms {obs['arm'].value_counts().to_dict()}")
        labels, _, found = score_cell_types(X, coh.genes, panel=MARKER_PANEL)
        obs['celltype'] = labels
        log(f'markers found: { {k: v for k, v in found.items() if v}}')
        log(f'lineage composition: {pd.Series(labels).value_counts().to_dict()}')
        Xp, covered = reindex_to_panel(X, coh.genes, panel)
        log(f'panel coverage {covered}/{n_panel} ({covered / n_panel:.1%})')
        theta = project(Xp, model, dev)
        tdf = pd.DataFrame(theta, columns=pd.Index([f'topic_{i}' for i in range(n_topics)]))
        tdf['gsm'] = obs['gsm'].to_numpy()
        tdf['arm'] = obs['arm'].to_numpy()
        tdf['celltype'] = obs['celltype'].to_numpy()
        tdf.to_parquet(os.path.join(args.output_dir, f'{name}_theta.parquet'))
        col = f'topic_{args.focus_topic}'
        entry = {'cells': int(idx.size), 'composition': {k: int(v) for k, v in pd.Series(labels).value_counts().items()}}
        comp = tdf.assign(is_lin=tdf['celltype'] == args.lineage).groupby(['gsm', 'arm'], observed=True)['is_lin'].mean().reset_index()
        comp.columns = pd.Index(['gsm', 'arm', 'lineage_frac'])
        comp_test = arm_test(comp, 'lineage_frac')
        entry['composition_test'] = comp_test
        whole = tdf.groupby(['gsm', 'arm'], observed=True)[col].mean().reset_index()
        entry['whole_sample_test'] = arm_test(whole, col)
        lin = tdf[tdf['celltype'] == args.lineage]
        per = lin.groupby(['gsm', 'arm'], observed=True)[col].agg(['mean', 'size']).reset_index()
        per.columns = pd.Index(['gsm', 'arm', col, 'n_cells'])
        small = per[per['n_cells'] < args.min_cells_per_sample]
        if len(small):
            log(f"  dropping {len(small)} sample(s) with <{args.min_cells_per_sample} {args.lineage} cells: {small['gsm'].tolist()}")
        per = per[per['n_cells'] >= args.min_cells_per_sample]
        entry['within_lineage_test'] = arm_test(per, col)
        entry['within_lineage_per_sample'] = per.to_dict(orient='records')
        n_dfu_gsm = int(per['arm'].isin({'DFU-healer', 'DFU-nonhealer'}).sum())
        declared = DECLARED_DFU_PATIENTS.get(name)
        if declared is not None and n_dfu_gsm > declared:
            entry['patient_independence'] = {'dfu_gsms_used': n_dfu_gsm, 'declared_patients': declared, 'excess_samples': n_dfu_gsm - declared, 'subject_id_available': False, 'verdict': 'sample-level test is pseudoreplicated at patient level'}
            log(f'  WARNING: {n_dfu_gsm} DFU samples used but the paper declares {declared} DFU patients. At least {n_dfu_gsm - declared} are repeat draws from the same patient, and GEO exposes no subject ID to collapse them. Treat the p-value below as optimistic.')
        else:
            entry['patient_independence'] = {'dfu_gsms_used': n_dfu_gsm, 'declared_patients': declared, 'excess_samples': 0, 'subject_id_available': False, 'verdict': 'one sample per patient, as far as public metadata shows'}
        loo = {}
        for gsm in per['gsm']:
            sub = per[per['gsm'] != gsm]
            t = arm_test(sub, col)
            if t:
                loo[str(gsm)] = {'ratio': t['ratio'], 'p': t['p'], 'n': [t['n_healer'], t['n_nonhealer']]}
        entry['leave_one_out'] = loo
        report['cohorts'][name] = entry
        log()
        ct = entry['composition_test']
        if ct:
            log(f"[composition] {args.lineage} fraction: healer {ct['healer_mean']:.3f} vs non-healer {ct['nonhealer_mean']:.3f}, ratio {ct['ratio']:.2f}x, p={ct['p']:.4f}")
        ws = entry['whole_sample_test']
        if ws:
            log(f"[whole sample ] topic {args.focus_topic}: {ws['healer_mean']:.4f} vs {ws['nonhealer_mean']:.4f} = {ws['ratio']:.2f}x [{ws['ratio_ci95'][0]:.2f}, {ws['ratio_ci95'][1]:.2f}], p={ws['p']:.4f}")
        wl = entry['within_lineage_test']
        if wl:
            nh_, nnh_ = (wl['n_healer'], wl['n_nonhealer'])
            floor = 2.0 / comb(nh_ + nnh_, min(nh_, nnh_))
            log(f"[within {args.lineage}] topic {args.focus_topic}: {wl['healer_mean']:.4f} vs {wl['nonhealer_mean']:.4f} = {wl['ratio']:.2f}x [{wl['ratio_ci95'][0]:.2f}, {wl['ratio_ci95'][1]:.2f}], p={wl['p']:.4f} (n={nh_}v{nnh_}, p floor {floor:.4f})")
            log(f"  healer per-sample    : {[round(v, 4) for v in wl['healer_values']]}")
            log(f"  non-healer per-sample: {[round(v, 4) for v in wl['nonhealer_values']]}")
            if loo:
                worst = max(loo.items(), key=lambda kv: kv[1]['ratio'] or 0)
                best = min(loo.items(), key=lambda kv: kv[1]['ratio'] or 1000000000.0)
                log(f"  leave-one-out ratio range: {best[1]['ratio']:.2f}x (drop {best[0]}) to {worst[1]['ratio']:.2f}x (drop {worst[0]})")
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log()
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
