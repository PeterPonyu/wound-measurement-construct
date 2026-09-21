#!/usr/bin/env python3
"""Assess power and donor geometry."""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import mannwhitneyu, norm, spearmanr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.celltype_markers import MARKER_PANEL, score_cell_types
from wound_models.data_adapter import library_normalize, load_10x_mtx_tar, load_dense_csv_tar, mito_fraction
from wound_models.human_wound_data import ANALYSIS_ROOT, preserve_reference_outputs, load_annotated_counts, protocol_block, project_latent
from wound_models.topic_model import TopicModel
OUR_IMMUNE = ['myeloid', 't_cell', 'b_plasma', 'mast']
AUTHOR_IMMUNE = ['Myeloid', 'Lymphoid', 'Mast']

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _rows(m) -> int:
    r, _ = m.shape
    return int(r)

def project(X, genes, panel, model, dev, n_panel):
    pos = pd.Index(genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    Xp = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(_rows(X), n_panel), dtype=np.int32)
    Xn = library_normalize(Xp)
    out = []
    with torch.no_grad():
        for s in range(0, _rows(Xn), 4096):
            xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            out.append(model(xb)['theta'].cpu().numpy())
    return (np.vstack(out), int(present.size))

def min_detectable_effect(values_a, values_b, n_a, n_b, n_sim=4000, power=0.8, seed=0):
    """
    Smallest group difference this design catches `power` of the time.

    Simulates from the pooled between-sample SD actually observed, so the
    answer reflects this cohort's variability rather than a textbook one.
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([values_a, values_b])
    sd = float(np.std(pooled, ddof=1))
    if sd == 0 or n_a < 2 or n_b < 2:
        return {'sd': sd, 'mde': float('nan'), 'power_at_observed': float('nan')}

    def power_at(delta):
        hits = 0
        for _ in range(n_sim):
            a = rng.normal(0.0, sd, n_a)
            b = rng.normal(delta, sd, n_b)
            try:
                if mannwhitneyu(a, b, alternative='two-sided')[1] < 0.05:
                    hits += 1
            except ValueError:
                pass
        return hits / n_sim
    lo, hi = (0.0, 6.0 * sd)
    for _ in range(14):
        mid = (lo + hi) / 2
        if power_at(mid) < power:
            lo = mid
        else:
            hi = mid
    observed = abs(float(np.mean(values_a) - np.mean(values_b)))
    return {'sd': sd, 'mde': (lo + hi) / 2, 'observed_difference': observed, 'power_at_observed': power_at(observed)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default=os.path.join(ANALYSIS_ROOT, 'negative_result_strength'))
    ap.add_argument('--theta-cut', type=float, default=0.184)
    ap.add_argument('--focus-topic', type=int, default=0)
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    ap.add_argument('--min-fib', type=int, default=20)
    args = ap.parse_args()
    preserve_reference_outputs(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    report = {}
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopicModel(input_dim=n_panel, num_topics=n_topics).to(dev)
    model.load_state_dict(state)
    model.eval()
    col = f'topic_{args.focus_topic}'
    frames = {}
    for name, kind, tar, sm in [('GSE165816', 'dense', 'data/raw/GSE165816/GSE165816_RAW.tar', 'data/raw/GSE165816/GSE165816_series_matrix.txt.gz'), ('GSE231643', '10x', 'data/raw/GSE231643/GSE231643_RAW.tar', 'data/raw/GSE231643/GSE231643_series_matrix.txt.gz')]:
        log(f'loading {name}')
        coh = load_dense_csv_tar(tar, series_matrix=sm) if kind == 'dense' else load_10x_mtx_tar(tar, series_matrix=sm)
        obs = coh.obs.copy()
        mito = mito_fraction(coh.X, coh.genes)
        gpc = np.asarray((coh.X > 0).sum(axis=1)).ravel()
        keep = (gpc >= args.min_genes_per_cell) & (mito <= args.max_mito)
        if name == 'GSE165816':
            obs['tissue_l'] = obs['tissue'].str.lower()
            keep = keep & obs['tissue_l'].isin(['foot skin', 'forearm skin']).to_numpy()
        idx = np.flatnonzero(keep)
        X = coh.X[idx]
        obs = obs.iloc[idx].reset_index(drop=True)
        labels, _, _ = score_cell_types(X, coh.genes, panel=MARKER_PANEL)
        obs['celltype'] = labels
        theta, cov = project(X, coh.genes, panel, model, dev, n_panel)
        obs['theta0'] = theta[:, args.focus_topic]
        for t in range(n_topics):
            obs[f'topic_{t}'] = theta[:, t]
        obs['immune'] = obs['celltype'].isin(OUR_IMMUNE)
        obs['is_fib'] = obs['celltype'] == 'fibroblast'
        obs['cohort'] = name
        frames[name] = obs
        log(f'  {name}: {len(obs)} cells, panel coverage {cov}/{n_panel}')
    log('loading GSE241132')
    coh, obs, X = load_annotated_counts('data/raw/GSE241132/per_gsm', 'data/raw/GSE241132/GSE241132_cell_metadata.txt.gz', extra_cols=('newCellTypes',))
    if 'newCellTypes' in obs.columns:
        obs = obs.rename(columns={'newCellTypes': 'celltype_fine'})
    theta, cov = project_latent(X, coh.genes, panel, model, dev, n_panel, key='theta')
    obs['theta0'] = theta[:, args.focus_topic]
    for t in range(n_topics):
        obs[f'topic_{t}'] = theta[:, t]
    obs['immune'] = obs['celltype'].isin(AUTHOR_IMMUNE)
    obs['is_fib'] = obs['celltype'] == 'Fibroblast'
    obs['cohort'] = 'GSE241132'
    obs['gsm'] = obs['label']
    frames['GSE241132'] = obs
    log(f"  GSE241132: {len(obs)} cells, panel coverage {cov}/{n_panel}, annotation = submitters' newMainCellTypes")
    log()
    log('A. does the immune association replicate, and which version of it?')
    bplasma_of = {'GSE165816': ['b_plasma'], 'GSE231643': ['b_plasma'], 'GSE241132': ['Plasma_Bcell']}
    report['A_immune_replication'] = {}
    for label, getter in [('broad_immune', lambda o, n: o['immune']), ('b_plasma_like', lambda o, n: (o['celltype_fine'] if n == 'GSE241132' else o['celltype']).isin(bplasma_of[n]))]:
        log(f'  -- definition: {label}')
        per_cohort = {}
        zs, ws = ([], [])
        for name, obs in frames.items():
            sel = obs
            if name == 'GSE165816':
                sel = obs[obs['tissue_l'] == 'foot skin']
            fib = sel[sel['is_fib']]
            w = fib.assign(hi=fib['theta0'] >= args.theta_cut).groupby('gsm', observed=True)['hi'].agg(['mean', 'size'])
            w.columns = pd.Index(['weight', 'n_fib'])
            frac = sel.assign(_f=getter(sel, name)).groupby('gsm', observed=True)['_f'].mean().rename('frac')
            d = pd.concat([w, frac], axis=1).dropna()
            d = d[d['n_fib'] >= args.min_fib]
            if len(d) < 5:
                log(f'     {name}: only {len(d)} usable samples, skipped')
                continue
            rho, p = spearmanr(d['weight'], d['frac'])
            per_cohort[name] = {'n_samples': int(len(d)), 'rho': float(rho), 'p': float(p), 'annotation': 'authors' if name == 'GSE241132' else 'markers'}
            if name == 'GSE241132':
                donor_tab = d.copy()
                donor_tab['donor'] = donor_tab.index.to_series().astype(str).str.replace('D\\d+$', '', regex=True)
                agg = donor_tab.groupby('donor')[['weight', 'frac']].mean()
                per_cohort[name]['n_donors'] = int(len(agg))
                per_cohort[name]['independent_units'] = '3 paired donors, not 12 samples'
                per_cohort[name]['sample_level_spearman_is_descriptive_only'] = True
                if len(agg) >= 3:
                    rho_d, p_d = spearmanr(agg['weight'], agg['frac'])
                    per_cohort[name]['donor_level'] = {'n_donors': int(len(agg)), 'rho': float(rho_d), 'p': float(p_d), 'note': 'n=3; p-value is descriptive only'}
                log(f"     {name:<10} n_samples={len(d):>2} n_donors={len(agg)}  sample rho={rho:+.3f} (not independent)  ({per_cohort[name]['annotation']})")
                continue
            zs.append(norm.isf(p / 2) * np.sign(rho))
            ws.append(np.sqrt(len(d)))
            log(f"     {name:<10} n={len(d):>2}  rho={rho:+.3f}  p={p:.4f}  ({per_cohort[name]['annotation']})")
        if zs:
            z_meta = float(np.sum(np.array(ws) * np.array(zs)) / np.sqrt(np.sum(np.array(ws) ** 2)))
            p_meta = float(2 * norm.sf(abs(z_meta)))
            same_sign = len({np.sign(v['rho']) for v in per_cohort.values()}) == 1
            driven_by_discovery = all((v['p'] > 0.05 for k, v in per_cohort.items() if k != 'GSE165816'))
            log(f'     meta Z={z_meta:+.2f} p={p_meta:.3e}  consistent sign={same_sign}  independent cohorts all non-significant={driven_by_discovery}')
            report['A_immune_replication'][label] = {'per_cohort': per_cohort, 'stouffer_z': z_meta, 'p': p_meta, 'consistent_sign': bool(same_sign), 'carried_by_discovery_only': bool(driven_by_discovery), 'stouffer_excludes': 'GSE241132 (paired 3-donor design)'}
    log()
    log('B. minimum detectable healing effect on the mixture weight')
    disc = frames['GSE165816']
    disc_foot = disc[(disc['tissue_l'] == 'foot skin') & disc['is_fib']]
    w = disc_foot.assign(hi=disc_foot['theta0'] >= args.theta_cut).groupby(['gsm', 'disease'], observed=True)['hi'].agg(['mean', 'size']).reset_index()
    w.columns = pd.Index(['gsm', 'disease', 'weight', 'n_fib'])
    w = w[w['n_fib'] >= args.min_fib]
    h = w.loc[w['disease'] == 'DFU-healer', 'weight'].to_numpy()
    nh = w.loc[w['disease'] == 'DFU-nonhealer', 'weight'].to_numpy()
    mde = min_detectable_effect(h, nh, len(h), len(nh))
    log(f"  n={len(h)} healer vs {len(nh)} non-healer; between-sample SD={mde['sd']:.3f}")
    log(f"  observed difference = {mde['observed_difference']:.3f} (power at that size = {mde['power_at_observed']:.1%})")
    log(f"  minimum detectable difference at 80% power = {mde['mde']:.3f}, i.e. {mde['mde'] / max(np.mean(np.concatenate([h, nh])), 1e-09):.1f}x the pooled mean weight")
    log('  -> this is a model-based design-sensitivity boundary; it is not an observed exclusion bound, and smaller effects are not resolved')
    report['observed_design_power'] = {**mde, 'n_healer': int(len(h)), 'n_nonhealer': int(len(nh))}
    log()
    log('C. positive control: foot vs forearm skin, same test, matched n')
    ctrl = disc[disc['is_fib']]
    topic_cols = [f'topic_{t}' for t in range(n_topics)]
    cw = ctrl.groupby(['gsm', 'tissue_l'], observed=True)[topic_cols].mean().reset_index()
    cnt = ctrl.groupby('gsm', observed=True).size()
    cw = cw[cw['gsm'].map(cnt) >= args.min_fib]
    foot = cw[cw['tissue_l'] == 'foot skin']
    fore = cw[cw['tissue_l'] == 'forearm skin']
    rng = np.random.default_rng(0)
    n_a, n_b = (len(h), len(nh))
    hits = {}
    if len(foot) >= n_a and len(fore) >= n_b:
        fa = foot.iloc[rng.choice(len(foot), n_a, replace=False)]
        fb = fore.iloc[rng.choice(len(fore), n_b, replace=False)]
        for t in topic_cols:
            _, p = mannwhitneyu(fa[t], fb[t], alternative='two-sided')
            hits[t] = float(p)
        sig = [t for t, p in hits.items() if p < 0.05]
        log(f'  subsampled to n={n_a} foot vs {n_b} forearm (from {len(foot)} and {len(fore)})')
        log(f"  topics separating at p<0.05: {len(sig)}/{n_topics} -> {(', '.join(sig) if sig else 'none')}")
        log(f'  smallest p = {min(hits.values()):.4f}')
        report['C_positive_control'] = {'n_foot': n_a, 'n_forearm': n_b, 'p_values': hits, 'n_significant': len(sig), 'design_has_power': bool(len(sig) > 0)}
        if sig:
            log('  -> the design does detect real effects at this n, so the healing null is a finding rather than a power failure')
        else:
            log('  -> nothing separates even for an anatomically certain contrast; the healing null cannot be interpreted')
    else:
        log('  not enough forearm samples for a matched-n control')
    report['protocol'] = protocol_block(__file__, args.discovery_dir, 0, folds=['GSE165816', 'GSE231643', 'GSE241132'], extra={'paired_same_donor_endpoints': False, 'gse241132_excluded_from_stouffer': True, 'theta_unscaled': True})
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log()
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
