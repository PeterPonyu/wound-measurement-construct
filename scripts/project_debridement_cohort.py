#!/usr/bin/env python3
"""Project debridement cohort."""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import load_10x_mtx_tar, library_normalize, mito_fraction
from wound_models.topic_model import TopicModel

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def healing_status(title: str) -> str:
    """H1..H5 -> healer, NH1/NH2/NH4 -> nonhealer."""
    t = str(title).strip().upper()
    if t.startswith('NH'):
        return 'DFU-nonhealer'
    if t.startswith('H'):
        return 'DFU-healer'
    return 'unknown'

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE231643/GSE231643_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE231643/GSE231643_series_matrix.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/validation_gse231643')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    report = {'discovery_dir': args.discovery_dir, 'validation_tar': args.tar}
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    log(f'frozen model: K={n_topics}, panel={n_panel} genes')
    top_focus = [str(panel[j]) for j in np.argsort(-beta[args.focus_topic])[:10]]
    log(f"focus topic {args.focus_topic} top genes: {', '.join(top_focus)}")
    report['focus_topic'] = {'index': args.focus_topic, 'top_genes': top_focus}
    log(f'loading {args.tar}')
    t0 = time.time()
    coh = load_10x_mtx_tar(args.tar, series_matrix=args.series_matrix)
    log(f'loaded {coh!r} in {time.time() - t0:.1f}s')
    for n in coh.notes:
        log(f'  note: {n}')
    if 'title' not in coh.obs.columns:
        log('FATAL: series matrix carried no !Sample_title; cannot assign labels')
        return 2
    coh.obs['arm'] = coh.obs['title'].map(healing_status)
    per_sample = coh.obs.groupby(['gsm', 'title', 'arm'], observed=True).size().reset_index(name='n_cells')
    log('sample labels recovered from titles:')
    for r in per_sample.itertuples():
        log(f'  {r.gsm} {r.title:>4} -> {r.arm:<14} {r.n_cells} cells')
    if (coh.obs['arm'] == 'unknown').any():
        log('FATAL: some samples could not be labelled')
        return 2
    mito = mito_fraction(coh.X, coh.genes)
    genes_per_cell = np.asarray((coh.X > 0).sum(axis=1)).ravel()
    keep = (genes_per_cell >= args.min_genes_per_cell) & (mito <= args.max_mito)
    log(f'cell QC: {int((genes_per_cell < args.min_genes_per_cell).sum())} below {args.min_genes_per_cell} genes, {int((mito > args.max_mito).sum())} above {args.max_mito:.0%} mito (median mito {np.median(mito):.1%}); keeping {int(keep.sum())} / {keep.size}')
    X = coh.X[np.flatnonzero(keep)]
    obs = coh.obs.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    pos = pd.Index(coh.genes).get_indexer(panel)
    covered = int((pos >= 0).sum())
    log(f'panel coverage in GSE231643: {covered} / {n_panel} genes ({covered / n_panel:.1%}); {n_panel - covered} zero-filled')
    report['panel_coverage'] = {'covered': covered, 'total': int(n_panel), 'fraction': covered / n_panel}
    if covered / n_panel < 0.8:
        log('WARNING: under 80% panel coverage; the projection is extrapolating')
    n_cells = _rows(X)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    Xp = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(n_cells, n_panel), dtype=np.int32)
    del sub
    Xn = library_normalize(Xp)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopicModel(input_dim=n_panel, num_topics=n_topics).to(dev)
    model.load_state_dict(state)
    model.eval()
    thetas = []
    with torch.no_grad():
        for s in range(0, n_cells, 4096):
            xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            thetas.append(model(xb)['theta'].cpu().numpy())
    theta = np.vstack(thetas)
    np.save(os.path.join(args.output_dir, 'theta.npy'), theta)
    obs.to_csv(os.path.join(args.output_dir, 'obs.csv'), index=False)
    tdf = pd.DataFrame(theta, columns=pd.Index([f'topic_{i}' for i in range(n_topics)]))
    tdf['gsm'] = obs['gsm'].to_numpy()
    tdf['title'] = obs['title'].to_numpy()
    tdf['arm'] = obs['arm'].to_numpy()
    sample_means = tdf.groupby(['gsm', 'title', 'arm'], observed=True).mean(numeric_only=True).reset_index()
    log('')
    log('per-sample mean theta (all topics):')
    log('  sample  arm             ' + ' '.join((f't{i:<5}' for i in range(n_topics))))
    for r in sample_means.itertuples():
        vals = ' '.join((f"{getattr(r, f'topic_{i}'):.4f}" for i in range(n_topics)))
        log(f'  {r.title:<7} {r.arm:<15} {vals}')
    results = {}
    for i in range(n_topics):
        col = f'topic_{i}'
        h = sample_means.loc[sample_means['arm'] == 'DFU-healer', col].to_numpy()
        nh = sample_means.loc[sample_means['arm'] == 'DFU-nonhealer', col].to_numpy()
        u, p = mannwhitneyu(h, nh, alternative='two-sided')
        results[col] = {'healer_mean': float(h.mean()), 'nonhealer_mean': float(nh.mean()), 'ratio': float(h.mean() / nh.mean()) if nh.mean() > 0 else None, 'u': float(u), 'p_sample_level': float(p), 'n_healer': int(h.size), 'n_nonhealer': int(nh.size)}
    n_h = int((sample_means['arm'] == 'DFU-healer').sum())
    n_nh = int((sample_means['arm'] == 'DFU-nonhealer').sum())
    from math import comb
    p_floor = 2.0 / comb(n_h + n_nh, min(n_h, n_nh))
    log('')
    log(f'sample-level Mann-Whitney, n={n_h} healer vs {n_nh} non-healer.')
    log(f'  smallest attainable two-sided p at this n is {p_floor:.4f}; nothing here can be more significant than that.')
    focus = results[f'topic_{args.focus_topic}']
    log('')
    log(f"FOCUS topic {args.focus_topic} ({', '.join(top_focus[:5])}):")
    log(f"  healer mean theta     = {focus['healer_mean']:.4f} (n={focus['n_healer']})")
    log(f"  non-healer mean theta = {focus['nonhealer_mean']:.4f} (n={focus['n_nonhealer']})")
    if focus['ratio']:
        log(f"  ratio                 = {focus['ratio']:.2f}x")
    log(f"  p (sample level)      = {focus['p_sample_level']:.4f}")
    hc = tdf.loc[tdf['arm'] == 'DFU-healer', f'topic_{args.focus_topic}'].to_numpy()
    nhc = tdf.loc[tdf['arm'] == 'DFU-nonhealer', f'topic_{args.focus_topic}'].to_numpy()
    _, p_cell = mannwhitneyu(hc, nhc, alternative='two-sided')
    log(f'  p (cell level, PSEUDOREPLICATED - do not report as the result) = {p_cell:.2e}')
    direction_ok = focus['healer_mean'] > focus['nonhealer_mean']
    log('')
    log(f"VALIDATION: direction {('REPRODUCES' if direction_ok else 'DOES NOT reproduce')} the discovery finding (healer > non-healer).")
    report['sample_means'] = sample_means.to_dict(orient='records')
    report['topic_tests'] = results
    report['p_floor_at_this_n'] = p_floor
    report['focus_cell_level_p_pseudoreplicated'] = float(p_cell)
    report['direction_reproduces'] = bool(direction_ok)
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0 if direction_ok else 1

def _rows(m) -> int:
    rows, _ = m.shape
    return int(rows)
if __name__ == '__main__':
    raise SystemExit(main())
