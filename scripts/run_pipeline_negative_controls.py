#!/usr/bin/env python3
"""Run pipeline negative controls."""
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
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.representation_checks import audit_surrogate_manifold
from wound_models.topic_model import TopicModel
THETA_CUT = 0.18429584801197052

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _rows(m) -> int:
    r, _ = m.shape
    return int(r)

def arm_statistic(per_sample: pd.DataFrame, arms) -> float:
    """Healer-minus-nonhealer difference in mixture weight; nan if unusable."""
    a = per_sample.loc[arms == 'DFU-healer', 'weight'].to_numpy()
    b = per_sample.loc[arms == 'DFU-nonhealer', 'weight'].to_numpy()
    if a.size < 2 or b.size < 2:
        return float('nan')
    return float(a.mean() - b.mean())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--theta-parquet', default='outputs/celltype_resolved/GSE165816_discovery_theta.parquet')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/pipeline_negative_control')
    ap.add_argument('--lineage', default='fibroblast')
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--n-perm', type=int, default=20000)
    ap.add_argument('--n-null-cells', type=int, default=6000)
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--load-for-n3', action='store_true', default=True)
    ap.add_argument('--no-load-for-n3', dest='load_for_n3', action='store_false')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    col = f'topic_{args.focus_topic}'
    report = {}
    tdf = pd.read_parquet(args.theta_parquet)
    fib = tdf[tdf['celltype'] == args.lineage]
    per = fib.assign(hi=fib[col] >= THETA_CUT).groupby(['gsm', 'arm'], observed=True).agg(weight=('hi', 'mean'), n=('hi', 'size')).reset_index()
    per = per[per['n'] >= 20].reset_index(drop=True)
    arms = per['arm'].to_numpy()
    observed = arm_statistic(per, arms)
    null = []
    for _ in range(args.n_perm):
        s = arm_statistic(per, rng.permutation(arms))
        if np.isfinite(s):
            null.append(s)
    null = np.asarray(null)
    p_perm = float((np.abs(null) >= abs(observed)).mean())
    inside = bool(np.percentile(null, 2.5) <= observed <= np.percentile(null, 97.5))
    log(f'N1 label permutation ({len(null)} usable draws)')
    log(f'   observed healer-nonhealer difference = {observed:+.4f}')
    log(f'   null 95% interval = [{np.percentile(null, 2.5):+.4f}, {np.percentile(null, 97.5):+.4f}]  two-sided p = {p_perm:.4f}')
    log(f"   -> observed statistic {('is inside' if inside else 'LIES OUTSIDE')} the null; pipeline {('does not' if inside else 'DOES')} manufacture an arm effect")
    report['N1_label_permutation'] = {'observed': observed, 'n_draws': int(len(null)), 'null_mean': float(null.mean()), 'null_sd': float(null.std(ddof=1)), 'null_ci95': [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))], 'p_two_sided': p_perm, 'observed_inside_null': inside, 'passes': inside}
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    theta_real = np.load(os.path.join(args.discovery_dir, 'theta.npy'))
    sub = rng.choice(theta_real.shape[0], min(args.n_null_cells, theta_real.shape[0]), replace=False)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopicModel(input_dim=n_panel, num_topics=n_topics).to(dev)
    model.load_state_dict(state)
    model.eval()
    real_theta = theta_real[sub]
    x_null = np.empty((len(sub), n_panel), dtype=np.float32)
    base = np.asarray(real_theta @ beta, dtype=np.float32)
    for j in range(n_panel):
        x_null[:, j] = base[rng.permutation(len(sub)), j]
    x_null /= np.maximum(x_null.sum(axis=1, keepdims=True), 1e-12)
    with torch.no_grad():
        out = model(torch.from_numpy(x_null).to(dev))
        theta_null = out['theta'].cpu().numpy()
        mu_null = out['mu'].cpu().numpy()
        out_r = model(torch.from_numpy(base).to(dev))
        mu_real = out_r['mu'].cpu().numpy()
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=50, random_state=args.seed)
    ref_null = svd.fit_transform(x_null)
    audit_null = audit_surrogate_manifold(ref_null, mu_null, theta_null, k=10)
    ref_real = TruncatedSVD(n_components=50, random_state=args.seed).fit_transform(base)
    audit_real = audit_surrogate_manifold(ref_real, mu_real, real_theta, k=10)
    perp_null = float(np.exp(-(theta_null * np.log(theta_null + 1e-12)).sum(axis=1)).mean())
    perp_real = float(np.exp(-(real_theta * np.log(real_theta + 1e-12)).sum(axis=1)).mean())
    log()
    log(f'N2 structureless input ({len(sub)} cells, per-gene column shuffle)')
    log(f"   real  : perplexity {perp_real:.2f}/{n_topics}, near_uniform={audit_real['near_uniform_warning']}, intact={audit_real['manifold_intact']}")
    log(f"   null  : perplexity {perp_null:.2f}/{n_topics}, near_uniform={audit_null['near_uniform_warning']}, intact={audit_null['manifold_intact']}")
    n2_pass = bool(perp_null > perp_real)
    log(f"   -> structureless input is {('less' if n2_pass else 'NOT less')} concentrated than real cells; {('gate distinguishes them' if n2_pass else 'GATE DOES NOT DISTINGUISH')}")
    report['N2_structureless_input'] = {'n_cells': int(len(sub)), 'perplexity_real': perp_real, 'perplexity_null': perp_null, 'audit_real': {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in audit_real.items()}, 'audit_null': {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in audit_null.items()}, 'passes': n2_pass, 'note': 'Column shuffle preserves every gene marginal and destroys co-expression. Perplexity is the discriminating statistic; reachability alone is not, because a shuffled cloud is still connected.'}
    tis = None
    if args.load_for_n3:
        from wound_models.celltype_markers import MARKER_PANEL, score_cell_types
        from wound_models.data_adapter import library_normalize, load_dense_csv_tar, mito_fraction
        log()
        log('N3 reloading cohort for forearm skin')
        coh = load_dense_csv_tar(args.tar, series_matrix=args.series_matrix)
        o = coh.obs.copy()
        o['tissue_l'] = o['tissue'].astype(str).str.lower()
        mito = mito_fraction(coh.X, coh.genes)
        gpc = np.asarray((coh.X > 0).sum(axis=1)).ravel()
        keep = o['tissue_l'].isin(['foot skin', 'forearm skin']).to_numpy() & (gpc >= 200) & (mito <= 0.2)
        idx = np.flatnonzero(keep)
        Xs = coh.X[idx]
        o = o.iloc[idx].reset_index(drop=True)
        lab, _, _ = score_cell_types(Xs, coh.genes, panel=MARKER_PANEL)
        o['celltype'] = lab
        pos = pd.Index(coh.genes).get_indexer(panel)
        present = np.flatnonzero(pos >= 0)
        subm = Xs[:, pos[present]].tocoo()
        Xp = sp.csr_matrix((subm.data, (subm.row, present[subm.col])), shape=(Xs.shape[0], n_panel), dtype=np.int32)
        Xn = library_normalize(Xp)
        th = []
        with torch.no_grad():
            for s0 in range(0, Xn.shape[0], 4096):
                xb = torch.from_numpy(np.asarray(Xn[s0:s0 + 4096].todense(), dtype=np.float32)).to(dev)
                th.append(model(xb)['theta'].cpu().numpy())
        o[col] = np.vstack(th)[:, args.focus_topic]
        tis = o[o['celltype'] == args.lineage].copy()
    if tis is not None and 'tissue_l' in tis.columns:
        ctrl = tis.groupby(['gsm', 'tissue_l'], observed=True)[col].agg(['mean', 'size']).reset_index()
        ctrl.columns = pd.Index(['gsm', 'tissue_l', 'mean_theta', 'n'])
        ctrl = ctrl[ctrl['n'] >= 20]
        foot = ctrl.loc[ctrl['tissue_l'] == 'foot skin', 'mean_theta'].to_numpy()
        fore = ctrl.loc[ctrl['tissue_l'] == 'forearm skin', 'mean_theta'].to_numpy()
        if foot.size >= 3 and fore.size >= 3:
            _, p_real = mannwhitneyu(foot, fore, alternative='two-sided')
            pool = np.concatenate([foot, fore])
            hits = 0
            for _ in range(args.n_perm // 10):
                q = rng.permutation(pool)
                _, pp = mannwhitneyu(q[:foot.size], q[foot.size:], alternative='two-sided')
                hits += pp < 0.05
            frac = hits / (args.n_perm // 10)
            n3_pass = bool(p_real < 0.05 and frac < 0.1)
            log()
            log(f'N3 permuted positive control (foot n={foot.size} vs forearm n={fore.size})')
            log(f'   real labels  p = {p_real:.4f}')
            log(f'   permuted labels significant in {frac:.1%} of draws')
            log(f"   -> {('control is informative' if n3_pass else 'CONTROL IS NOT INFORMATIVE')}")
            report['N3_permuted_positive_control'] = {'p_real': float(p_real), 'n_foot': int(foot.size), 'n_forearm': int(fore.size), 'fraction_significant_under_permutation': float(frac), 'passes': n3_pass}
        else:
            report['N3_permuted_positive_control'] = {'status': 'insufficient_samples'}
    else:
        report['N3_permuted_positive_control'] = {'status': 'skipped_pass_--load-for-n3_to_run'}
    checks = [v.get('passes') for v in report.values() if isinstance(v, dict) and 'passes' in v]
    all_pass = all(checks) if checks else False
    log()
    log('=' * 74)
    log(f'negative controls passed: {sum((1 for c in checks if c))}/{len(checks)}')
    log('The pipeline does not report structure that is not there.' if all_pass else 'AT LEAST ONE NEGATIVE CONTROL FAILED - read the JSON before using any positive result.')
    log('=' * 74)
    report['all_negative_controls_pass'] = all_pass
    report['protocol'] = {'seed': args.seed, 'n_perm': args.n_perm, 'theta_cut': THETA_CUT, 'retrains_model': False, 'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), 'elapsed_seconds': time.time() - started}
    report['scope'] = {'patient_level_inference_available': False, 'limit': 'Negative controls on the discovery cohort only. They show the pipeline stays quiet on null input; they do not validate any positive finding.'}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0 if all_pass else 1
if __name__ == '__main__':
    raise SystemExit(main())
