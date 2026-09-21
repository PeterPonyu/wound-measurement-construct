#!/usr/bin/env python3
"""Project chronic wound edges."""
from __future__ import annotations
import argparse
import gzip
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
import scipy.io as sio
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.data_adapter import library_normalize, mito_fraction, parse_series_matrix
from wound_models.topic_model import TopicModel
_SPEC = importlib.util.spec_from_file_location('external_immune_axis_repair', ROOT / 'scripts' / 'assess_disjoint_immune_compartments.py')
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('cannot import scripts/assess_disjoint_immune_compartments.py')
_AXIS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AXIS)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def _decode_line(path: Path) -> list[str]:
    with gzip.open(path, 'rt', errors='replace') as handle:
        return [line.rstrip('\n\r').split('\t')[0] for line in handle]

def load_10x_mtx(matrix_path: Path, features_path: Path, barcodes_path: Path):
    """Load a Cell Ranger MTX bundle as cells x genes CSR counts."""
    features = pd.read_csv(features_path, sep='\t', header=None, compression='gzip', dtype=str)
    if features.shape[1] < 2:
        raise ValueError(f'{features_path} has no gene-symbol column')
    genes = features.iloc[:, 1].fillna(features.iloc[:, 0]).to_numpy(dtype=object)
    if features.shape[1] >= 3:
        keep = features.iloc[:, 2].fillna('').eq('Gene Expression').to_numpy()
    else:
        keep = np.ones(len(features), dtype=bool)
    matrix = sio.mmread(gzip.open(matrix_path, 'rb'))
    matrix = sp.csr_matrix(matrix, dtype=np.int32)
    if matrix.shape[0] != len(genes):
        raise ValueError(f'feature/matrix row mismatch for {matrix_path}: {len(genes)} vs {matrix.shape[0]}')
    if not keep.all():
        matrix = matrix[keep]
        genes = genes[keep]
    barcodes = np.asarray(_decode_line(barcodes_path), dtype=object)
    if matrix.shape[1] != len(barcodes):
        raise ValueError(f'barcode/matrix column mismatch for {matrix_path}: {len(barcodes)} vs {matrix.shape[1]}')
    return (matrix.T.tocsr(), genes, barcodes)

def _files_for_gsm(matrix_dir: Path, gsm: str) -> tuple[Path, Path, Path]:
    candidates = sorted(matrix_dir.glob(f'{gsm}_*_matrix.mtx.gz'))
    if len(candidates) != 1:
        raise FileNotFoundError(f'expected one matrix for {gsm}, found {candidates}')
    matrix = candidates[0]
    stem = matrix.name[:-len('_matrix.mtx.gz')]
    features = matrix_dir / f'{stem}_features.tsv.gz'
    barcodes = matrix_dir / f'{stem}_barcodes.tsv.gz'
    if not features.exists() or not barcodes.exists():
        raise FileNotFoundError(f'incomplete 10x bundle for {gsm}')
    return (matrix, features, barcodes)

def _group_from_row(row: pd.Series, title: str) -> str:
    value = str(row.get('tissue_type', ''))
    if not value or value.lower() == 'nan':
        value = title
    value = value.lower()
    if 'non-diabetic' in value or 'non diabetic' in value:
        return 'non_diabetic'
    if 'diabetic' in value:
        return 'diabetic'
    return 'unknown'

def _two_sided_label_permutation(values: np.ndarray, labels: np.ndarray, seed: int=0):
    labels = np.asarray(labels, dtype=bool)
    observed = float(values[labels].mean() - values[~labels].mean())
    rng = np.random.default_rng(seed)
    n_true = int(labels.sum())
    null = np.empty(20000, dtype=float)
    for i in range(null.size):
        chosen = rng.choice(values.size, size=n_true, replace=False)
        mask = np.zeros(values.size, dtype=bool)
        mask[chosen] = True
        null[i] = values[mask].mean() - values[~mask].mean()
    p = float((np.abs(null) >= abs(observed)).mean())
    return {'difference_diabetic_minus_non_diabetic': observed, 'n_permutations': int(null.size), 'null_ci95': [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))], 'p_two_sided_descriptive': p, 'scope_note': 'sample-level descriptive contrast; no authoritative patient IDs'}

def _association(frame: pd.DataFrame, axis: str) -> dict:
    valid = frame[['topic0_fibro_mean', axis]].dropna()
    if len(valid) < 3:
        return {'n_samples': int(len(valid)), 'rho': None, 'p_descriptive': None}
    rho, p = spearmanr(valid['topic0_fibro_mean'], valid[axis])
    return {'n_samples': int(len(valid)), 'rho': float(rho), 'p_descriptive': float(p)}

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
    source_files = [{'name': series_path.name, 'bytes': series_path.stat().st_size, 'sha256': sha256(series_path)}]
    for gsm in pheno.index.astype(str):
        matrix_path, features_path, barcodes_path = _files_for_gsm(matrix_dir, gsm)
        source_files.extend(({'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in (matrix_path, features_path, barcodes_path)))
        X, genes, _barcodes = load_10x_mtx(matrix_path, features_path, barcodes_path)
        X, genes = _AXIS.ext.collapse_gene_duplicates(X, genes)
        genes_per_cell = np.asarray(X.getnnz(axis=1)).ravel()
        umis = np.asarray(X.sum(axis=1)).ravel()
        mito = mito_fraction(X, genes)
        keep = (genes_per_cell >= args.min_genes_per_cell) & (mito <= args.max_mito)
        X = X[keep]
        genes_per_cell = genes_per_cell[keep]
        umis = umis[keep]
        if X.shape[0] == 0:
            raise ValueError(f'{gsm}: no cells survive fixed QC')
        if args.max_cells_per_sample is not None and X.shape[0] > args.max_cells_per_sample:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(X.shape[0], args.max_cells_per_sample, replace=False))
            X = X[idx]
            genes_per_cell = genes_per_cell[idx]
            umis = umis[idx]
        panel_X, covered = _AXIS.ext.panel_matrix(X, genes, panel)
        theta = _AXIS.ext.project(model, library_normalize(panel_X), device)
        gates = _AXIS.assign_exclusive(X, genes)
        fibro = gates['fibro']
        row = pheno.loc[gsm]
        title = str(row.get('title', gsm))
        group = _group_from_row(row, title)
        rows.append({'gsm': gsm, 'title': title, 'group': group, 'sex': str(row.get('sex', '')), 'age': str(row.get('age', '')), 'cells_raw': int(genes_per_cell.size), 'cells_qc': int(X.shape[0]), 'panel_covered': int(covered), 'panel_total': int(panel.size), 'panel_fraction': float(covered / panel.size), 'median_genes_per_cell': float(np.median(genes_per_cell)), 'median_umis_per_cell': float(np.median(umis)), 'immune_fraction_exclusive': float(gates['immune'].mean()), 'fibro_fraction_exclusive': float(fibro.mean()), 'other_fraction_exclusive': float(gates['other'].mean()), 'b_plasma_fraction_strict': float(gates['b_plasma'].mean()), 'b_plasma_share_of_immune': float(gates['b_plasma'].sum() / gates['immune'].sum()) if gates['immune'].any() else None, 'cd74_positive_fraction': float(gates['cd74_positive'].mean()), 'old_immune_fraction': float(gates['old_immune'].mean()), 'old_b_plasma_fraction': float(gates['old_b_plasma'].mean()), 'topic0_all_mean': float(theta[:, 0].mean()), 'topic0_fibro_mean': float(theta[fibro, 0].mean()) if fibro.any() else None})
    frame = pd.DataFrame(rows)
    if set(frame['group']) != {'diabetic', 'non_diabetic'}:
        raise ValueError(f"unexpected GSE268834 groups: {sorted(frame['group'].unique())}")
    group_stats = {}
    diabetic = frame['group'].eq('diabetic').to_numpy()
    for metric in ['topic0_all_mean', 'topic0_fibro_mean', 'immune_fraction_exclusive', 'fibro_fraction_exclusive', 'b_plasma_fraction_strict']:
        valid = frame[metric].notna().to_numpy()
        group_stats[metric] = _two_sided_label_permutation(frame.loc[valid, metric].to_numpy(dtype=float), diabetic[valid], seed=0)
    associations = {axis: _association(frame, axis) for axis in ['immune_fraction_exclusive', 'b_plasma_fraction_strict', 'b_plasma_share_of_immune', 'median_genes_per_cell', 'median_umis_per_cell', 'old_immune_fraction', 'old_b_plasma_fraction']}
    exclusive_sum = frame['immune_fraction_exclusive'] + frame['fibro_fraction_exclusive'] + frame['other_fraction_exclusive']
    old_sum = frame['old_immune_fraction'] + frame['fibro_fraction_exclusive']
    script_path = Path(__file__)
    axis_script = ROOT / 'scripts' / 'assess_disjoint_immune_compartments.py'
    return {'experiment': 'gse268834_frozen_external_projection', 'status': 'completed', 'scientific_unit': 'sample', 'patient_level_inference_available': False, 'independent_prognostic_validation_eligible': False, 'healing_endpoint_available': False, 'cohort': {'accession': 'GSE268834', 'sample_count': int(len(frame)), 'groups': frame['group'].value_counts().to_dict(), 'cells_qc_total': int(frame['cells_qc'].sum()), 'authoritative_patient_ids': False, 'metadata_limit': 'series labels distinguish diabetic from non-diabetic wound-edge samples but do not provide patient IDs, wound IDs, follow-up, or healed/not_healed outcomes'}, 'source_files': source_files, 'samples': frame.to_dict(orient='records'), 'sample_level_group_contrasts': group_stats, 'sample_level_associations': associations, 'exclusivity_check': {'exclusive_compartment_sum_min': float(exclusive_sum.min()), 'exclusive_compartment_sum_max': float(exclusive_sum.max()), 'old_gate_overlap_max': float(old_sum.max()), 'old_gates_were_overlapping': bool(old_sum.max() > 1.0)}, 'interpretation': 'This extends frozen state and immune-axis auditing to an independent wound-edge series. Diabetic/non-diabetic contrasts are descriptive sample-level quantities; they are not healing or patient-level validation.', 'protocol': {'frozen_discovery_dir': str(args.discovery_dir), 'frozen_panel_genes': int(panel.size), 'frozen_topics': int(beta.shape[0]), 'encoder_refit_on_external': False, 'projection_deterministic': True, 'input_format': '10x matrix-market per GSM', 'qc': {'min_genes_per_cell': args.min_genes_per_cell, 'max_mito': args.max_mito}, 'exclusive_compartment_assignment': True, 'promiscuous_markers_excluded': _AXIS.SATURATION_CONTROL + ['CD37', 'CD38', 'SDC1'], 'no_healing_endpoint': True, 'no_authoritative_patient_ids': True, 'source_axis_script_sha256': sha256(axis_script), 'script_sha256': sha256(script_path), 'checked_at_utc': pd.Timestamp.utcnow().isoformat(), 'elapsed_seconds': time.time() - t0}, 'scope': {'patient_level_inference_available': False, 'limit': 'Independent external construct projection only; no healing validation, no pooled patient inference, and no MCID test.'}}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--matrix-dir', default='data/raw/GSE268834/per_gsm')
    ap.add_argument('--series-matrix', default='data/raw/GSE268834/GSE268834_series_matrix.txt.gz')
    ap.add_argument('--output-dir', default='outputs/analysis/external_gse268834_projection')
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    ap.add_argument('--max-cells-per-sample', type=int, default=None)
    args = ap.parse_args()
    report = run(args)
    out = Path(args.output_dir) / 'report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'cohort': report['cohort'], 'group_contrasts': report['sample_level_group_contrasts'], 'associations': report['sample_level_associations'], 'output': str(out)}, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
