#!/usr/bin/env python3
"""Assess disjoint immune compartments."""
from __future__ import annotations
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import library_normalize, mito_fraction, parse_series_matrix
from wound_models.topic_model import TopicModel
_EXT_PATH = Path(__file__).resolve().parent / 'project_external_wound_cohorts.py'
_spec = importlib.util.spec_from_file_location('external_dfu_projection', _EXT_PATH)
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)
FIBRO_MARKERS = ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'PDGFRA', 'THY1']
PAN_LEUKOCYTE = ['PTPRC']
T_NK_MARKERS = ['CD3D', 'CD3E', 'CD2', 'TRBC1', 'TRBC2', 'NKG7', 'GNLY']
MYELOID_MARKERS = ['LYZ', 'CD68', 'CD14', 'ITGAX', 'FCER1G', 'TYROBP', 'AIF1']
B_MARKERS = ['MS4A1', 'CD79A', 'CD79B', 'BANK1']
PLASMA_MARKERS = ['MZB1', 'JCHAIN', 'DERL3', 'TNFRSF17']
SATURATION_CONTROL = ['CD74']
OLD_IMMUNE_MARKERS = ext.IMMUNE_MARKERS
OLD_B_PLASMA_MARKERS = ext.B_PLASMA_MARKERS

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def assign_exclusive(X, genes) -> dict[str, np.ndarray]:
    """Assign each QC cell to one compartment.

    Immune is pan-leukocyte positive or backed by two lineage-specific
    immune markers.  Fibroblast-like requires two structural markers and no
    immune assignment.  A cell can belong to at most one compartment, so the
    reported fractions are a composition and not overlapping detection rates.
    """
    ptprc = ext.marker_fraction(X, genes, PAN_LEUKOCYTE)
    t_nk = ext.marker_count(X, genes, T_NK_MARKERS)
    myeloid = ext.marker_count(X, genes, MYELOID_MARKERS)
    b_cells = ext.marker_count(X, genes, B_MARKERS)
    plasma = ext.marker_count(X, genes, PLASMA_MARKERS)
    specific_immune = (t_nk >= 2) | (myeloid >= 2) | (b_cells >= 2) | (plasma >= 2)
    immune = ptprc | specific_immune
    fibro_n = ext.marker_count(X, genes, FIBRO_MARKERS)
    fibro = (fibro_n >= 2) & ~immune
    b_plasma = immune & ((b_cells >= 1) | (plasma >= 1))
    return {'immune': immune, 'fibro': fibro, 'b_plasma': b_plasma, 'other': ~immune & ~fibro, 'cd74_positive': ext.marker_fraction(X, genes, SATURATION_CONTROL), 'old_immune': ext.marker_fraction(X, genes, OLD_IMMUNE_MARKERS), 'old_b_plasma': ext.marker_fraction(X, genes, OLD_B_PLASMA_MARKERS)}

def run_cohort(*, name: str, directory: str, series_matrix: str, model, panel: np.ndarray, device: str, min_genes: int, max_mito: float) -> dict:
    pheno = parse_series_matrix(series_matrix).set_index('gsm')
    paths = sorted(glob.glob(os.path.join(directory, '*.h5')))
    if not paths:
        raise FileNotFoundError(f'no per-sample H5 files in {directory}')
    rows = []
    for path in paths:
        match = re.search('(GSM\\d+)', os.path.basename(path))
        if not match:
            continue
        gsm = match.group(1)
        X, genes, _ = ext.load_10x_h5(path)
        X, genes = ext.collapse_gene_duplicates(X, genes)
        genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
        umis = np.asarray(X.sum(axis=1)).ravel()
        mito = mito_fraction(X, genes)
        keep = (genes_per_cell >= min_genes) & (mito <= max_mito)
        X = X[keep]
        if X.shape[0] == 0:
            continue
        genes_per_cell = genes_per_cell[keep]
        umis = umis[keep]
        panel_X, covered = ext.panel_matrix(X, genes, panel)
        theta = ext.project(model, library_normalize(panel_X), device)
        gates = assign_exclusive(X, genes)
        n = X.shape[0]
        obs_row = pheno.loc[gsm].to_dict() if gsm in pheno.index else {'gsm': gsm}
        title = str(obs_row.get('title', gsm))
        fibro = gates['fibro']
        rows.append({'gsm': gsm, 'title': title, 'patient_proxy': ext.patient_proxy(title), 'disease_or_genotype': obs_row.get('disease', obs_row.get('genotype')), 'cells_qc': int(n), 'panel_fraction': covered / panel.size, 'median_genes_per_cell': float(np.median(genes_per_cell)), 'median_umis_per_cell': float(np.median(umis)), 'immune_fraction_exclusive': float(gates['immune'].mean()), 'fibro_fraction_exclusive': float(fibro.mean()), 'other_fraction_exclusive': float(gates['other'].mean()), 'b_plasma_fraction_strict': float(gates['b_plasma'].mean()), 'b_plasma_share_of_immune': float(gates['b_plasma'].sum() / gates['immune'].sum()) if gates['immune'].any() else None, 'cd74_positive_fraction': float(gates['cd74_positive'].mean()), 'old_immune_fraction': float(gates['old_immune'].mean()), 'old_b_plasma_fraction': float(gates['old_b_plasma'].mean()), 'topic0_fibro_mean': float(theta[fibro, 0].mean()) if fibro.any() else None, 'topic0_all_mean': float(theta[:, 0].mean())})
        log(f"  {gsm} {title}: n={n} immune={gates['immune'].mean():.3f} fibro={fibro.mean():.3f} bplasma={gates['b_plasma'].mean():.4f} (old bplasma {gates['old_b_plasma'].mean():.3f}) CD74+={gates['cd74_positive'].mean():.3f}")
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError(f'no samples survived in {name}')
    axes = ['immune_fraction_exclusive', 'b_plasma_fraction_strict', 'b_plasma_share_of_immune', 'old_immune_fraction', 'old_b_plasma_fraction', 'cd74_positive_fraction', 'median_genes_per_cell', 'median_umis_per_cell']
    assoc = {}
    for axis in axes:
        valid = samples[['topic0_fibro_mean', axis]].dropna()
        if len(valid) >= 3:
            rho, p = spearmanr(valid['topic0_fibro_mean'], valid[axis])
            assoc[axis] = {'n_samples': int(len(valid)), 'rho': float(rho), 'p_descriptive': float(p)}
        else:
            assoc[axis] = {'n_samples': int(len(valid)), 'rho': None, 'p_descriptive': None}
    exclusive_sum = samples['immune_fraction_exclusive'] + samples['fibro_fraction_exclusive'] + samples['other_fraction_exclusive']
    old_sum = samples['old_immune_fraction'] + samples['fibro_fraction_exclusive']
    return {'cohort': name, 'sample_count': int(len(samples)), 'cells_qc_total': int(samples['cells_qc'].sum()), 'patient_id_authoritative': False, 'duplicate_title_proxies': samples.groupby('patient_proxy')['gsm'].apply(list).loc[lambda x: x.map(len) > 1].to_dict(), 'exclusivity_check': {'exclusive_compartment_sum_max': float(exclusive_sum.max()), 'exclusive_compartment_sum_min': float(exclusive_sum.min()), 'old_gate_overlap_max': float(old_sum.max()), 'old_gates_were_overlapping': bool(old_sum.max() > 1.0)}, 'samples': samples.to_dict(orient='records'), 'sample_level_associations': assoc}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--gse223964-dir', default='data/raw/GSE223964')
    ap.add_argument('--gse245703-dir', default='data/raw/GSE245703')
    ap.add_argument('--output-dir', default='outputs/analysis/external_immune_axis')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    discovery = Path(args.discovery_dir)
    panel = np.load(discovery / 'panel.npy', allow_pickle=True).astype(object)
    beta = np.load(discovery / 'beta.npy')
    n_topics = int(beta.shape[0])
    state = torch.load(discovery / 'topic_model.pt', map_location='cpu')
    model = TopicModel(input_dim=int(beta.shape[1]), num_topics=n_topics)
    model.load_state_dict(state)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    log(f'frozen encoder: {n_topics} topics, {panel.size} panel genes, {device}')
    cohorts = {}
    for name, directory, series in (('GSE223964', args.gse223964_dir, os.path.join(args.gse223964_dir, 'GSE223964_series_matrix.txt.gz')), ('GSE245703', args.gse245703_dir, os.path.join(args.gse245703_dir, 'GSE245703_series_matrix.txt.gz'))):
        log(f'{name}: exclusive re-gating on frozen projection')
        cohorts[name] = run_cohort(name=name, directory=directory, series_matrix=series, model=model, panel=panel, device=device, min_genes=args.min_genes_per_cell, max_mito=args.max_mito)

    def rho(cohort: str, axis: str):
        return cohorts[cohort]['sample_level_associations'][axis]['rho']
    strict_signs = {c: np.sign(rho(c, 'immune_fraction_exclusive') or 0.0) for c in cohorts}
    bp_signs = {c: np.sign(rho(c, 'b_plasma_fraction_strict') or 0.0) for c in cohorts}
    depth_link = {c: rho(c, 'median_genes_per_cell') for c in cohorts}
    report = {'experiment': 'external_immune_axis_repair', 'status': 'completed', 'scientific_unit': 'sample', 'patient_level_inference_available': False, 'defect_repaired': {'old_gates': {'immune': OLD_IMMUNE_MARKERS, 'b_plasma': OLD_B_PLASMA_MARKERS}, 'why_unusable': "CD74 (plus CD37/CD38/SDC1 in the B/plasma gate) is detected in most cells, so 'any marker detected' saturates and the fibroblast gate overlapped the immune gate", 'new_gates': {'pan_leukocyte': PAN_LEUKOCYTE, 't_nk': T_NK_MARKERS, 'myeloid': MYELOID_MARKERS, 'b': B_MARKERS, 'plasma': PLASMA_MARKERS, 'fibro': FIBRO_MARKERS}}, 'cohorts': cohorts, 'cross_cohort': {'strict_immune_sign_consistent': bool(len(set(strict_signs.values())) == 1 and 0 not in strict_signs.values()), 'strict_immune_rho': {c: rho(c, 'immune_fraction_exclusive') for c in cohorts}, 'strict_b_plasma_sign_consistent': bool(len(set(bp_signs.values())) == 1 and 0 not in bp_signs.values()), 'strict_b_plasma_rho': {c: rho(c, 'b_plasma_fraction_strict') for c in cohorts}, 'old_broad_immune_rho': {c: rho(c, 'old_immune_fraction') for c in cohorts}, 'topic0_vs_sequencing_depth_rho': depth_link, 'no_pooled_test': 'cohorts are kept separate; sample titles are not authoritative patient IDs and neither cohort has a healing endpoint'}, 'protocol': {'frozen_discovery_dir': args.discovery_dir, 'frozen_panel_genes': int(panel.size), 'frozen_topics': n_topics, 'projection_deterministic': True, 'encoder_refit_on_external': False, 'exclusive_compartment_assignment': True, 'promiscuous_markers_excluded': SATURATION_CONTROL + ['CD37', 'CD38', 'SDC1'], 'depth_confound_reported': True, 'qc': {'min_genes_per_cell': args.min_genes_per_cell, 'max_mito': args.max_mito}, 'healing_endpoint_available': False, 'source_script_sha256': _sha256(str(_EXT_PATH)), 'script_sha256': _sha256(__file__), 'elapsed_seconds': time.time() - t0}}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    log(f'wrote {out}')
    print(json.dumps(report['cross_cohort'], indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
