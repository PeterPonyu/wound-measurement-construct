#!/usr/bin/env python3
"""
expression representation mainline: human DFU single-cell topic decomposition + topological audit.

Replaces the retired Direction-1 stub. Every number this script reports comes
from GSE165816 counts on disk; nothing is simulated.

Pipeline
  1. Load GSE165816 RAW.tar (54 dense CSVs) as CSR int32 via the data adapter.
  2. Restrict to foot skin and to the labelled arms. Forearm skin and PBMCs are
     dropped: they are different tissues and would dominate the variance.
  3. Cell/gene QC, then pick the 7000-gene target line by dispersion.
  4. Library-normalize to per-cell proportions (the simplex decoder's input).
  5. Fit TopicModel; track whether beta escapes the near-uniform basin,
     because at G=7000 the default init sits at perplexity ~= G and moves slowly.
  6. connectivity audit audit of the learned latent against a PCA reference geometry.

The audit's reference graph is built on a 50-dim SVD of the normalized counts,
not on the raw 7000-dim matrix: kNN in 7000 dimensions is distance-concentrated
and the resulting "reference" topology would itself be an artifact.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import load_dense_csv_tar, library_normalize, select_target_genes, mito_fraction
from wound_models.representation_checks import audit_surrogate_manifold
from wound_models.topic_model import TopicModel
PINNED_REGULATORS = ['CDKN1A', 'FOS', 'JUNB', 'IL1B', 'TLR4', 'NLRP3', 'TNF', 'IFNG', 'MMP1', 'MMP3', 'MMP13', 'CHI3L1', 'TIMP1', 'COL7A1', 'NRG1', 'ASPN', 'KRT14', 'KRT5', 'CD68', 'CD163', 'PRG4', 'THY1', 'VEGFA']
KEEP_ARMS = {'DFU-healer', 'DFU-nonhealer', 'Non-diabetic'}

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', default='data/raw/GSE165816/GSE165816_RAW.tar')
    ap.add_argument('--series-matrix', default='data/raw/GSE165816/GSE165816_series_matrix.txt.gz')
    ap.add_argument('--output-dir', default='outputs/expression_representation')
    ap.add_argument('--max-samples', type=int, default=None, help='smoke-test on the first N GSMs')
    ap.add_argument('--n-genes', type=int, default=7000)
    ap.add_argument('--n-topics', type=int, default=15)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--audit-cells', type=int, default=20000, help='cells sampled for the connectivity audit reference graph')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--min-cells-per-gene', type=int, default=10)
    ap.add_argument('--max-mito', type=float, default=0.2, help='drop cells whose mitochondrial count fraction exceeds this')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    report = {'inputs': {'tar': args.tar, 'max_samples': args.max_samples}}
    log(f'loading {args.tar}')
    t0 = time.time()
    coh = load_dense_csv_tar(args.tar, series_matrix=args.series_matrix, max_samples=args.max_samples)
    log(f'loaded {coh!r} in {time.time() - t0:.1f}s')
    for n in coh.notes:
        log(f'  note: {n}')
    report['loaded'] = {'cells': int(coh.X.shape[0]), 'genes': int(coh.X.shape[1]), 'samples': int(coh.obs['gsm'].nunique()), 'notes': coh.notes}
    if 'disease' not in coh.obs.columns or 'tissue' not in coh.obs.columns:
        log("FATAL: series matrix did not supply 'disease'/'tissue'; cannot proceed")
        return 2
    keep = (coh.obs['tissue'].str.lower() == 'foot skin') & coh.obs['disease'].isin(KEEP_ARMS)
    keep_idx = np.flatnonzero(keep.to_numpy())
    log(f'foot-skin + labelled arms: {keep_idx.size} / {coh.X.shape[0]} cells')
    log('  arm counts: ' + json.dumps(coh.obs.loc[keep, 'disease'].value_counts().to_dict()))
    if keep_idx.size == 0:
        log('FATAL: no cells survived the tissue/arm filter')
        return 2
    X = coh.X[keep_idx]
    obs = coh.obs.iloc[keep_idx].reset_index(drop=True)
    report['filtered'] = {'cells': int(keep_idx.size), 'arms': {k: int(v) for k, v in obs['disease'].value_counts().items()}}
    mito = mito_fraction(X, coh.genes)
    genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    cell_ok = (genes_per_cell >= args.min_genes_per_cell) & (mito <= args.max_mito)
    log(f'cell QC: {int((genes_per_cell < args.min_genes_per_cell).sum())} below {args.min_genes_per_cell} genes, {int((mito > args.max_mito).sum())} above {args.max_mito:.0%} mito (median mito {np.median(mito):.1%})')
    X = X[np.flatnonzero(cell_ok)]
    obs = obs.iloc[np.flatnonzero(cell_ok)].reset_index(drop=True)
    cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    gene_ok = np.flatnonzero(cells_per_gene >= args.min_cells_per_gene)
    X = X[:, gene_ok]
    genes = coh.genes[gene_ok]
    log(f'after QC: {X.shape[0]} cells x {X.shape[1]} genes (dropped {int((~cell_ok).sum())} cells, {int(len(cells_per_gene) - gene_ok.size)} genes)')
    report['qc'] = {'cells_kept': int(X.shape[0]), 'cells_dropped': int((~cell_ok).sum()), 'median_mito': float(np.median(mito)), 'max_mito_threshold': args.max_mito, 'arms_after_qc': {k: int(v) for k, v in obs['disease'].value_counts().items()}}
    n_top = min(args.n_genes, X.shape[1])
    idx, panel = select_target_genes(X, genes, n_top=n_top, pinned=PINNED_REGULATORS, exclude_technical=True, log_normalize=True)
    found = sorted(set(panel) & set(PINNED_REGULATORS))
    missing = sorted(set(PINNED_REGULATORS) - set(genes.tolist()))
    log(f'target line: {len(panel)} genes; pinned present {len(found)}/{len(PINNED_REGULATORS)}; absent from assay: {missing}')
    X = X[:, idx]
    report['target_line'] = {'n_genes': int(len(panel)), 'pinned_present': found, 'pinned_absent_from_assay': missing}
    Xn = library_normalize(X)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = Xn.shape[1]
    model = TopicModel(input_dim=G, num_topics=args.n_topics).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    n_cells = Xn.shape[0]

    def beta_perplexity() -> float:
        b = model.decoder.beta.detach()
        ent = -(b * torch.log(b + 1e-12)).sum(-1).mean().item()
        return float(np.exp(ent))
    log(f'fitting on {dev}: {n_cells} cells, G={G}, K={args.n_topics}, {args.steps} steps; beta perplexity at init = {beta_perplexity():.0f} (uniform = {G})')
    t0 = time.time()
    trace = []
    for step in range(1, args.steps + 1):
        rows = rng.integers(0, n_cells, args.batch_size)
        xb = torch.from_numpy(np.asarray(Xn[rows].todense(), dtype=np.float32)).to(dev)
        opt.zero_grad()
        out = model(xb)
        out['loss'].backward()
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            bp = beta_perplexity()
            trace.append({'step': step, 'loss': float(out['loss'].item()), 'beta_perplexity': bp})
            log(f"  step {step}: loss={out['loss'].item():.4f} beta_perp={bp:.0f}")
    fit_s = time.time() - t0
    final_bp = beta_perplexity()
    log(f'fit done in {fit_s:.1f}s; final beta perplexity {final_bp:.0f} / {G}')
    report['fit'] = {'seconds': fit_s, 'trace': trace, 'final_beta_perplexity': final_bp, 'beta_near_uniform': bool(final_bp > 0.5 * G)}
    if final_bp > 0.5 * G:
        log('  WARNING: beta is still above half of uniform. The topic-gene matrix has not concentrated; topics are not yet interpretable.')
    model.eval()
    thetas, mus = ([], [])
    with torch.no_grad():
        for s in range(0, n_cells, 4096):
            xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            o = model(xb)
            thetas.append(o['theta'].cpu().numpy())
            mus.append(o['mu'].cpu().numpy())
    theta = np.vstack(thetas)
    mu = np.vstack(mus)
    beta = model.decoder.beta.detach().cpu().numpy()
    np.save(os.path.join(args.output_dir, 'theta.npy'), theta)
    np.save(os.path.join(args.output_dir, 'beta.npy'), beta)
    np.save(os.path.join(args.output_dir, 'panel.npy'), panel)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'topic_model.pt'))
    obs.to_csv(os.path.join(args.output_dir, 'obs.csv'), index=False)
    top_genes = {}
    for t in range(beta.shape[0]):
        order = np.argsort(-beta[t])[:12]
        top_genes[f'topic_{t}'] = [str(panel[j]) for j in order]
        log(f"  topic {t:2d}: {', '.join(top_genes[f'topic_{t}'][:8])}")
    arm_means = {}
    for arm in sorted(obs['disease'].unique()):
        rows = np.flatnonzero((obs['disease'] == arm).to_numpy())
        arm_means[arm] = [round(float(v), 4) for v in theta[rows].mean(axis=0)]
    report['topics'] = {'top_genes': top_genes, 'mean_theta_by_arm': arm_means}
    log('  mean theta by arm: ' + json.dumps(arm_means))
    n_audit = min(args.audit_cells, n_cells)
    sub = np.sort(rng.choice(n_cells, size=n_audit, replace=False))
    log(f'connectivity audit audit on {n_audit} cells (PCA-50 reference geometry)')
    t0 = time.time()
    svd = TruncatedSVD(n_components=50, random_state=args.seed)
    X_ref = svd.fit_transform(Xn[sub])
    audit = audit_surrogate_manifold(X_ref, mu[sub], theta[sub], k=10)
    log(f'audit done in {time.time() - t0:.1f}s')
    for k, v in audit.items():
        log(f'  {k}: {v}')
    report['raag_audit'] = {k: float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) else v for k, v in audit.items()}
    report['raag_audit']['n_cells_audited'] = int(n_audit)
    out_json = os.path.join(args.output_dir, 'report.json')
    with open(out_json, 'w') as fh:
        json.dump(report, fh, indent=2)
    log(f'wrote {out_json}')
    if not audit['manifold_intact']:
        log('GATE FAILED: manifold_intact is False. The latent geometry does not preserve the reference topology; do not publish trajectories from this fit.')
        return 1
    log('GATE PASSED: manifold_intact is True.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
