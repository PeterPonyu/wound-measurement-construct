#!/usr/bin/env python3
"""Assess threshold and capacity."""
import argparse
import hashlib
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
from wound_models.data_adapter import library_normalize, load_dense_csv_tar, mito_fraction, select_target_genes
from wound_models.topic_model import TopicModel
PINNED = ['CDKN1A', 'FOS', 'JUNB', 'IL1B', 'TLR4', 'NLRP3', 'TNF', 'IFNG', 'MMP1', 'MMP3', 'MMP13', 'CHI3L1', 'TIMP1', 'COL7A1', 'NRG1', 'ASPN', 'KRT14', 'KRT5', 'CD68', 'CD163', 'PRG4', 'THY1', 'VEGFA']
KEEP_ARMS = {'DFU-healer', 'DFU-nonhealer', 'Non-diabetic'}

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _rows(m) -> int:
    r, _ = m.shape
    return int(r)

def three_claims(per_sample: pd.DataFrame, col: str, cut: float) -> dict:
    """Claims A, B and C at one binarisation of one theta column."""
    w = per_sample.assign(hi=per_sample[col] >= cut).groupby(['gsm', 'arm'], observed=True).agg(weight=('hi', 'mean'), mean_theta=(col, 'mean'), n_fib=('hi', 'size')).reset_index()
    w = w[w['n_fib'] >= 20]
    if len(w) < 6:
        return {'usable_samples': int(len(w))}
    rho, p_rho = spearmanr(w['mean_theta'], w['weight'])
    top = w.loc[w['weight'].idxmax()]
    healer = w.loc[w['arm'] == 'DFU-healer', 'weight'].to_numpy()
    nonheal = w.loc[w['arm'] == 'DFU-nonhealer', 'weight'].to_numpy()
    if healer.size and nonheal.size:
        _, p_arm = mannwhitneyu(healer, nonheal, alternative='two-sided')
        ratio = float(healer.mean() / nonheal.mean()) if nonheal.mean() > 0 else None
    else:
        p_arm, ratio = (float('nan'), None)
    return {'usable_samples': int(len(w)), 'A_rho_mean_vs_weight': float(rho), 'A_p': float(p_rho), 'B_purest_sample': str(top['gsm']), 'B_purest_arm': str(top['arm']), 'B_purest_weight': float(top['weight']), 'B_counterexample_holds': bool(top['arm'] == 'Non-diabetic'), 'C_healer_mean': float(healer.mean()) if healer.size else None, 'C_nonhealer_mean': float(nonheal.mean()) if nonheal.size else None, 'C_ratio': ratio, 'C_p': float(p_arm)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--theta-parquet', default='outputs/celltype_resolved/GSE165816_discovery_theta.parquet')
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--output-dir', default='outputs/method_constant_sensitivity')
    ap.add_argument('--scratch-dir', default='outputs/method_constant_sensitivity/scratch_k')
    ap.add_argument('--lineage', default='fibroblast')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--k-grid', type=int, nargs='+', default=[10, 15, 20])
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--n-genes', type=int, default=7000)
    ap.add_argument('--skip-k', action='store_true', help='cut sweep only; do not refit the topic model')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()
    report = {'focus_topic': args.focus_topic, 'lineage': args.lineage}
    tdf = pd.read_parquet(args.theta_parquet)
    col = f'topic_{args.focus_topic}'
    fib = tdf[tdf['celltype'] == args.lineage][['gsm', 'arm', col]].copy()
    log(f"cut sweep on {len(fib)} frozen {args.lineage}s ({fib['gsm'].nunique()} samples)")
    gmm_cut = 0.18429584801197052
    gmm_cut_r = round(gmm_cut, 4)
    grid = sorted({round(float(c), 4) for c in list(np.arange(0.08, 0.41, 0.02)) + [gmm_cut]})
    cut_rows = []
    for cut in grid:
        r = three_claims(fib, col, float(cut))
        r['cut'] = float(cut)
        r['is_published_cut'] = bool(abs(cut - gmm_cut_r) < 1e-09)
        cut_rows.append(r)
    assert any((r['is_published_cut'] for r in cut_rows)), 'the published cut must appear in its own sweep'
    usable = [r for r in cut_rows if 'A_rho_mean_vs_weight' in r]
    rhos = [r['A_rho_mean_vs_weight'] for r in usable]
    ce = [r['B_counterexample_holds'] for r in usable]
    ps = [r['C_p'] for r in usable if np.isfinite(r['C_p'])]
    log(f'  grid {min(grid):.3f}-{max(grid):.3f}, {len(usable)} usable cuts')
    published = [r for r in usable if r['is_published_cut']][0]
    log(f"  A rho range {min(rhos):+.3f} to {max(rhos):+.3f} (published cut {gmm_cut_r:.4f}: {published['A_rho_mean_vs_weight']:+.3f})")
    log(f'  B counter-example holds at {sum(ce)}/{len(ce)} cuts')
    log(f'  C arm p range {min(ps):.3f} to {max(ps):.3f}; significant at {sum((1 for p in ps if p < 0.05))}/{len(ps)} cuts')
    report['cut_sweep'] = {'grid': grid, 'gmm_cut': gmm_cut, 'rows': cut_rows, 'A_rho_min': float(min(rhos)), 'A_rho_max': float(max(rhos)), 'A_stable': bool(min(rhos) > 0.8), 'B_holds_fraction': float(sum(ce) / len(ce)), 'C_significant_cuts': int(sum((1 for p in ps if p < 0.05))), 'C_total_cuts': int(len(ps))}
    if args.skip_k:
        log('K sweep skipped by flag')
        report['k_sweep'] = {'status': 'skipped'}
    else:
        os.makedirs(args.scratch_dir, exist_ok=True)
        log(f'loading {args.tar} once for the K sweep')
        coh = load_dense_csv_tar(args.tar, series_matrix=args.series_matrix)
        obs = coh.obs.copy()
        obs['arm'] = obs['disease']
        keep = obs['arm'].isin(list(KEEP_ARMS)).to_numpy() & (obs['tissue'].str.lower() == 'foot skin').to_numpy()
        mito = mito_fraction(coh.X, coh.genes)
        gpc = np.asarray((coh.X > 0).sum(axis=1)).ravel()
        keep = keep & (gpc >= 200) & (mito <= 0.2)
        idx = np.flatnonzero(keep)
        X = coh.X[idx]
        obs = obs.iloc[idx].reset_index(drop=True)
        labels, _, _ = score_cell_types(X, coh.genes, panel=MARKER_PANEL)
        obs['celltype'] = labels
        cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
        gene_ok = np.flatnonzero(cells_per_gene >= 10)
        X = X[:, gene_ok]
        genes = coh.genes[gene_ok]
        gi, panel = select_target_genes(X, genes, n_top=min(args.n_genes, X.shape[1]), pinned=PINNED, exclude_technical=True, log_normalize=True)
        Xn = library_normalize(X[:, gi])
        log(f'  {_rows(Xn)} cells x {len(panel)} genes; fitting K={args.k_grid}')
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        rng = np.random.default_rng(args.seed)
        fib_mask = (obs['celltype'] == args.lineage).to_numpy()
        k_rows = []
        for K in args.k_grid:
            torch.manual_seed(args.seed)
            model = TopicModel(input_dim=len(panel), num_topics=K).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            t0 = time.time()
            for _ in range(args.steps):
                rows = rng.integers(0, _rows(Xn), args.batch_size)
                xb = torch.from_numpy(np.asarray(Xn[rows].todense(), dtype=np.float32)).to(dev)
                opt.zero_grad()
                model(xb)['loss'].backward()
                opt.step()
            model.eval()
            th = []
            with torch.no_grad():
                for s in range(0, _rows(Xn), 4096):
                    xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
                    th.append(model(xb)['theta'].cpu().numpy())
            theta = np.vstack(th)
            beta = model.decoder.beta.detach().cpu().numpy()
            markers = [i for i, g in enumerate(panel) if str(g) in ('TIMP1', 'CHI3L1', 'MMP1')]
            focus = int(np.argmax(beta[:, markers].sum(axis=1)))
            sub = pd.DataFrame({'gsm': obs.loc[fib_mask, 'gsm'].to_numpy(), 'arm': obs.loc[fib_mask, 'arm'].to_numpy(), 'topic_focus': theta[fib_mask, focus]})
            from sklearn.mixture import GaussianMixture
            v = sub['topic_focus'].to_numpy().reshape(-1, 1)
            gm = GaussianMixture(2, random_state=args.seed).fit(v)
            cut_k = float(np.mean(sorted(gm.means_.ravel())))
            r = three_claims(sub, 'topic_focus', cut_k)
            r.update({'K': K, 'focus_topic_at_this_K': focus, 'derived_cut': cut_k, 'top_genes': [str(panel[j]) for j in np.argsort(-beta[focus])[:6]], 'fit_seconds': time.time() - t0})
            k_rows.append(r)
            log(f"  K={K:>2} focus=topic_{focus} cut={cut_k:.4f} A_rho={r.get('A_rho_mean_vs_weight', float('nan')):+.3f} B={r.get('B_counterexample_holds')} C_p={r.get('C_p', float('nan')):.3f} ({r['fit_seconds']:.0f}s)")
            np.save(os.path.join(args.scratch_dir, f'beta_K{K}.npy'), beta)
        krhos = [r['A_rho_mean_vs_weight'] for r in k_rows if 'A_rho_mean_vs_weight' in r]
        report['k_sweep'] = {'status': 'run', 'rows': k_rows, 'A_rho_min': float(min(krhos)), 'A_rho_max': float(max(krhos)), 'A_stable': bool(min(krhos) > 0.8), 'B_holds_all_K': bool(all((r.get('B_counterexample_holds') for r in k_rows))), 'checkpoint_untouched': True}
    verdict = []
    if report['cut_sweep']['A_stable']:
        verdict.append('claim A survives the cut grid')
    else:
        verdict.append('claim A DEPENDS on the cut')
    if report['cut_sweep']['B_holds_fraction'] == 1.0:
        verdict.append('counter-example B holds at every cut')
    else:
        verdict.append(f"counter-example B holds at only {report['cut_sweep']['B_holds_fraction']:.0%} of cuts")
    if report['k_sweep'].get('status') == 'run':
        verdict.append('claim A survives K' if report['k_sweep']['A_stable'] else 'claim A DEPENDS on K')
        if report['k_sweep']['B_holds_all_K']:
            verdict.append('counter-example B holds at every K')
        else:
            bad = [r['K'] for r in report['k_sweep']['rows'] if not r.get('B_counterexample_holds')]
            verdict.append(f'counter-example B FAILS at K={bad}: at those sizes the TIMP1-carrying topic is a different programme, so the counter-example is capacity-dependent')
    log()
    log('=' * 74)
    for v in verdict:
        log(f'  {v}')
    log('=' * 74)
    report['verdict'] = verdict
    report['protocol'] = {'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size, 'k_grid': args.k_grid, 'n_genes': args.n_genes, 'frozen_checkpoint_modified': False, 'cut_sweep_reuses_frozen_projection': True, 'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), 'elapsed_seconds': time.time() - started}
    report['scope'] = {'patient_level_inference_available': False, 'limit': "Sensitivity of the method's own constants on one discovery cohort. It does not establish the claims externally."}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
