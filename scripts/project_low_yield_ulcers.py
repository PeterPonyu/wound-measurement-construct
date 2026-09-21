#!/usr/bin/env python3
"""Project low yield ulcers."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_SPEC = importlib.util.spec_from_file_location('gse268834_external_projection', ROOT / 'scripts' / 'project_chronic_wound_edges.py')
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('cannot import scripts/project_chronic_wound_edges.py')
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
from wound_models.data_adapter import library_normalize, mito_fraction, parse_series_matrix
from wound_models.topic_model import TopicModel

def run(args) -> dict:
    t0 = time.time()
    discovery = Path(args.discovery_dir)
    matrix_dir = Path(args.matrix_dir)
    series_path = Path(args.series_matrix)
    panel = np.load(discovery / 'panel.npy', allow_pickle=True).astype(object)
    beta = np.load(discovery / 'beta.npy')
    state = torch.load(discovery / 'topic_model.pt', map_location='cpu')
    model = TopicModel(input_dim=int(beta.shape[1]), num_topics=int(beta.shape[0]))
    model.load_state_dict(state)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    pheno = parse_series_matrix(str(series_path)).set_index('gsm')
    rows = []
    source_files = [{'name': series_path.name, 'bytes': series_path.stat().st_size, 'sha256': _BASE.sha256(series_path)}]
    for gsm in pheno.index.astype(str):
        matrix_path, features_path, barcodes_path = _BASE._files_for_gsm(matrix_dir, gsm)
        source_files.extend(({'name': path.name, 'bytes': path.stat().st_size, 'sha256': _BASE.sha256(path)} for path in (matrix_path, features_path, barcodes_path)))
        X, genes, _barcodes = _BASE.load_10x_mtx(matrix_path, features_path, barcodes_path)
        X, genes = _BASE._AXIS.ext.collapse_gene_duplicates(X, genes)
        raw_cell_count = int(X.shape[0])
        genes_per_cell = np.asarray(X.getnnz(axis=1)).ravel()
        umis = np.asarray(X.sum(axis=1)).ravel()
        mito = mito_fraction(X, genes)
        keep = (genes_per_cell >= args.min_genes_per_cell) & (mito <= args.max_mito)
        X = X[keep]
        genes_per_cell = genes_per_cell[keep]
        umis = umis[keep]
        if X.shape[0] == 0:
            raise ValueError(f'{gsm}: no cells survive fixed QC')
        panel_X, covered = _BASE._AXIS.ext.panel_matrix(X, genes, panel)
        theta = _BASE._AXIS.ext.project(model, library_normalize(panel_X), device)
        gates = _BASE._AXIS.assign_exclusive(X, genes)
        fibro = gates['fibro']
        row = pheno.loc[gsm]
        title = str(row.get('title', gsm))
        group = _BASE._group_from_row(row, title)
        if group not in {'diabetic', 'non_diabetic'}:
            raise ValueError(f'unexpected GSE248247 group for {gsm}: {group}')
        rows.append({'gsm': gsm, 'title': title, 'group': group, 'cells_raw': raw_cell_count, 'cells_qc': int(X.shape[0]), 'panel_covered': int(covered), 'panel_total': int(panel.size), 'panel_fraction': float(covered / panel.size), 'median_genes_per_cell': float(np.median(genes_per_cell)), 'median_umis_per_cell': float(np.median(umis)), 'immune_fraction_exclusive': float(gates['immune'].mean()), 'fibro_fraction_exclusive': float(fibro.mean()), 'other_fraction_exclusive': float(gates['other'].mean()), 'b_plasma_fraction_strict': float(gates['b_plasma'].mean()), 'b_plasma_share_of_immune': float(gates['b_plasma'].sum() / gates['immune'].sum()) if gates['immune'].any() else None, 'cd74_positive_fraction': float(gates['cd74_positive'].mean()), 'old_immune_fraction': float(gates['old_immune'].mean()), 'old_b_plasma_fraction': float(gates['old_b_plasma'].mean()), 'topic0_all_mean': float(theta[:, 0].mean()), 'topic0_fibro_mean': float(theta[fibro, 0].mean()) if fibro.any() else None})
    frame = pd.DataFrame(rows)
    if set(frame['group']) != {'diabetic', 'non_diabetic'}:
        raise ValueError(f"unexpected groups: {sorted(frame['group'].unique())}")
    group_stats = {}
    diabetic = frame['group'].eq('diabetic').to_numpy()
    for metric in ['topic0_all_mean', 'topic0_fibro_mean', 'immune_fraction_exclusive', 'fibro_fraction_exclusive', 'b_plasma_fraction_strict']:
        valid = frame[metric].notna().to_numpy()
        group_stats[metric] = _BASE._two_sided_label_permutation(frame.loc[valid, metric].to_numpy(dtype=float), diabetic[valid], seed=0)
    associations = {axis: _BASE._association(frame, axis) for axis in ['immune_fraction_exclusive', 'b_plasma_fraction_strict', 'b_plasma_share_of_immune', 'median_genes_per_cell', 'median_umis_per_cell', 'old_immune_fraction', 'old_b_plasma_fraction']}
    exclusive_sum = frame['immune_fraction_exclusive'] + frame['fibro_fraction_exclusive'] + frame['other_fraction_exclusive']
    old_sum = frame['old_immune_fraction'] + frame['fibro_fraction_exclusive']
    script_path = Path(__file__)
    axis_script = ROOT / 'scripts' / 'assess_disjoint_immune_compartments.py'
    helper_script = ROOT / 'scripts' / 'project_chronic_wound_edges.py'
    return {'experiment': 'gse248247_low_yield_frozen_projection', 'status': 'completed', 'scientific_unit': 'sample', 'patient_level_inference_available': False, 'independent_prognostic_validation_eligible': False, 'healing_endpoint_available': False, 'cohort': {'accession': 'GSE248247', 'sample_count': int(len(frame)), 'groups': frame['group'].value_counts().to_dict(), 'cells_raw_total': int(frame['cells_raw'].sum()), 'cells_qc_total': int(frame['cells_qc'].sum()), 'authoritative_patient_ids': False, 'metadata_limit': 'one diabetic and one non-diabetic plantar wound sample; no authoritative patient IDs, healing outcomes or follow-up'}, 'source_files': source_files, 'samples': frame.to_dict(orient='records'), 'sample_level_group_contrasts': group_stats, 'sample_level_associations': associations, 'exclusivity_check': {'exclusive_compartment_sum_min': float(exclusive_sum.min()), 'exclusive_compartment_sum_max': float(exclusive_sum.max()), 'old_gate_overlap_max': float(old_sum.max()), 'old_gates_were_overlapping': bool(old_sum.max() > 1.0)}, 'interpretation': 'This is a two-sample, low-yield feasibility projection of the frozen encoder and exclusive lineage gates. It is not a prognostic result, not patient-level replication, and not evidence that diabetic versus non-diabetic status is a healing endpoint.', 'protocol': {'frozen_discovery_dir': str(args.discovery_dir), 'frozen_panel_genes': int(panel.size), 'frozen_topics': int(beta.shape[0]), 'encoder_refit_on_external': False, 'projection_deterministic': True, 'input_format': '10x matrix-market per GSM', 'qc': {'min_genes_per_cell': args.min_genes_per_cell, 'max_mito': args.max_mito}, 'exclusive_compartment_assignment': True, 'promiscuous_markers_excluded': _BASE._AXIS.SATURATION_CONTROL + ['CD37', 'CD38', 'SDC1'], 'no_healing_endpoint': True, 'no_authoritative_patient_ids': True, 'source_axis_script_sha256': _BASE.sha256(axis_script), 'helper_script_sha256': _BASE.sha256(helper_script), 'script_sha256': _BASE.sha256(script_path), 'checked_at_utc': pd.Timestamp.utcnow().isoformat(), 'elapsed_seconds': time.time() - t0}, 'scope': {'patient_level_inference_available': False, 'limit': 'Two-sample low-yield feasibility projection only; no healing validation, no pooled patient inference, and no MCID test.'}}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--matrix-dir', default='data/raw/GSE248247/per_gsm')
    ap.add_argument('--series-matrix', default='data/raw/GSE248247/GSE248247_series_matrix.txt.gz')
    ap.add_argument('--output-dir', default='outputs/analysis/external_gse248247_projection')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    args = ap.parse_args()
    report = run(args)
    out = Path(args.output_dir) / 'report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'cohort': report['cohort'], 'group_contrasts': report['sample_level_group_contrasts'], 'associations': report['sample_level_associations'], 'output': str(out)}, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
