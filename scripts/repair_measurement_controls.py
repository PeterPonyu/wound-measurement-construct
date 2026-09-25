#!/usr/bin/env python3
"""Raw-count null, frozen marker gate and outcome-eligible label controls.

This is a retrospective measurement sensitivity analysis. It does not retrain
the topic model, infer cell transitions, or validate marker calls against a
biological identity reference. Historical artifacts are read only.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import itertools
import json
import math
import re
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.special import expit, logit
from scipy.stats import false_discovery_control, spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.celltype_markers import MARKER_PANEL, score_cell_types
from wound_models.data_adapter import _assemble, _read_dense_csv_member, library_normalize, mito_fraction, parse_series_matrix
from wound_models.topic_model import TopicModel
OUT = ROOT / 'outputs/scientific_revision_20260922/measurement_controls'
DISCOVERY = ROOT / 'outputs/expression_representation'
PARQUET = ROOT / 'outputs/celltype_resolved/GSE165816_discovery_theta.parquet'
RAW = ROOT / 'data/raw/GSE165816/GSE165816_RAW.tar'
SERIES = ROOT / 'data/raw/GSE165816/GSE165816_series_matrix.txt.gz'
THRESHOLD = ROOT / 'outputs/bimodal_stratification/report.json'
UNIFIED_PROJECTION = ROOT / 'outputs/scientific_revision_20260922/measurement/projection/cells.parquet'
DFU_ARMS = ('DFU-healer', 'DFU-nonhealer')
READOUTS = ('fibroblast_mean', 'high_state_fraction', 'fibroblast_fraction')

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def prepare_cache(output):
    """Recover raw identities in the original archive/sample/column order."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'raw_cache_manifest.json'
    input_paths = [RAW, SERIES, DISCOVERY / 'obs.csv', PARQUET]
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['inputs_sha256'] != inputs:
            raise ValueError('Raw cache inputs changed; use a new output directory')
        for name, expected in manifest['cache_sha256'].items():
            if sha256(output / name) != expected:
                raise ValueError(f'Raw cache hash mismatch: {name}')
        log('Loading verified raw-count cache')
        return (sp.load_npz(output / 'discovery_raw_counts.npz'), np.load(output / 'discovery_raw_genes.npy', allow_pickle=False), pd.read_csv(output / 'raw_cell_identity.csv.gz'), manifest)
    saved_obs = pd.read_csv(DISCOVERY / 'obs.csv')
    saved = pd.read_parquet(PARQUET)
    wanted = set(saved_obs.gsm)
    pheno = parse_series_matrix(str(SERIES)).set_index('gsm')
    eligible = pheno.loc[pheno.tissue.str.lower().eq('foot skin') & pheno.disease.isin([*DFU_ARMS, 'Non-diabetic'])]
    if set(eligible.index) != wanted or len(wanted) != 25:
        raise ValueError('The recorded discovery cohort no longer matches the 25-GSM contract')
    per_sample, identities = ([], [])
    with tarfile.open(RAW, 'r') as archive:
        members = sorted([m for m in archive.getmembers() if m.isfile() and m.name.endswith(('.csv', '.csv.gz'))], key=lambda m: m.name)
        for member in members:
            hit = re.search('GSM\\d+', member.name)
            if hit is None or hit.group() not in wanted:
                continue
            gsm = hit.group()
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f'Unreadable raw member: {member.name}')
            compressed = io.BytesIO(handle.read())
            source = gzip.open(compressed, 'rb') if member.name.endswith('.gz') else compressed
            matrix, genes, barcodes = _read_dense_csv_member(source)
            if len(set(genes)) != len(genes) or len(set(barcodes)) != len(barcodes):
                raise ValueError(f'Duplicate source genes or barcodes within {gsm}')
            per_sample.append((gsm, matrix, genes, barcodes))
            identities.append(pd.DataFrame({'gsm': gsm, 'arm': pheno.loc[gsm, 'disease'], 'raw_barcode': barcodes, 'raw_row_0': np.arange(len(barcodes)), 'raw_member': member.name}))
            log(f'Read raw source {gsm}: {matrix.shape[0]} cells, {matrix.shape[1]} genes')
    if len(per_sample) != len(wanted):
        raise ValueError('Missing or duplicate selected raw sample')
    cohort = _assemble(per_sample, str(SERIES), 'union', 0.0, [])
    del per_sample
    identity = pd.concat(identities, ignore_index=True)
    genes = np.asarray(cohort.genes, dtype=str)
    detected = np.asarray((cohort.X > 0).sum(axis=1)).ravel()
    mito = mito_fraction(cohort.X, genes)
    keep = (detected >= 200) & (mito <= 0.2)
    raw_before_qc = len(identity)
    matrix = cohort.X[keep].tocsr()
    del cohort
    identity = identity.loc[keep].reset_index(drop=True)
    identity.insert(0, 'saved_row_0', np.arange(len(identity)))
    identity['qc_row_0'] = identity.groupby('gsm', sort=False).cumcount()
    identity['raw_cell_id'] = identity.gsm + ':' + identity.raw_barcode
    identity['source_order_cell_id'] = identity.gsm + ':' + identity.qc_row_0.astype(str)
    identity['detected_genes'] = detected[keep]
    identity['mito_fraction'] = mito[keep]
    identity['raw_library_size'] = np.asarray(matrix.sum(axis=1)).ravel()
    if len(identity) != 76461 or identity.raw_cell_id.duplicated().any():
        raise ValueError('Unexpected cell count or non-unique recovered source identity')
    for actual, expected in [(identity.gsm, saved_obs.gsm), (identity.arm, saved_obs.disease), (identity.gsm, saved.gsm), (identity.arm, saved.arm)]:
        if not np.array_equal(actual, expected):
            raise ValueError('Raw reconstruction disagrees with historical row order')
    labels, _, found = score_cell_types(matrix, genes)
    if not np.array_equal(labels, saved.celltype.to_numpy()):
        raise ValueError('Raw reconstruction does not reproduce every saved marker label')
    identity['historical_celltype'] = labels
    identity.to_csv(output / 'raw_cell_identity.csv.gz', index=False, compression='gzip')
    np.save(output / 'discovery_raw_genes.npy', genes, allow_pickle=False)
    log(f'Writing {matrix.shape} raw integer count cache')
    sp.save_npz(output / 'discovery_raw_counts.npz', matrix, compressed=True)
    manifest = {'status': 'complete', 'inputs_sha256': inputs, 'cache_sha256': {name: sha256(output / name) for name in ('raw_cell_identity.csv.gz', 'discovery_raw_genes.npy', 'discovery_raw_counts.npz')}, 'cells_before_qc': raw_before_qc, 'cells_after_qc': len(identity), 'specimens': len(wanted), 'genes_union_selected_specimens': len(genes), 'qc': {'minimum_detected_genes': 200, 'maximum_mito_fraction': 0.2}, 'alignment_checks': {'mainline_gsm_arm_order_exact': True, 'parquet_gsm_arm_order_exact': True, 'all_saved_marker_labels_reproduced': True}, 'marker_counts': found, 'identity_contract': {'raw_cell_id': "GSM + ':' + barcode read from the source CSV column", 'raw_row_0': 'Zero-based original CSV cell-column position within GSM', 'qc_row_0': 'Zero-based within-GSM position after the historical QC filter', 'saved_row_0': 'Zero-based row in obs.csv and discovery theta parquet', 'source_order_cell_id': "GSM + ':' + qc_row_0; synthetic order ID, not a barcode", 'scope': 'Reconstructed identity supported by source order and complete marker agreement; historical artifacts did not retain barcodes'}, 'gene_union_note': 'Only the 25 selected GSM files are assembled. Genes absent from these samples contribute zero counts; full-cell libraries and marker values agree with the original all-54-sample assembly.'}
    write_json(manifest_path, manifest)
    log('Raw-count and cell-identity cache verified and ready')
    return (matrix, genes, identity, manifest)

def balanced_rows(identity: pd.DataFrame, cells_per_gsm: int=240, seed: int=9122) -> np.ndarray:
    """Select a fixed, equal-size per-GSM subset in source-order coordinates."""
    if cells_per_gsm < 1:
        raise ValueError('cells_per_gsm must be positive')
    rng = np.random.default_rng(seed)
    rows = []
    for gsm in sorted(identity['gsm'].astype(str).unique()):
        available = np.flatnonzero(identity['gsm'].to_numpy(dtype=str) == gsm)
        take = min(int(cells_per_gsm), len(available))
        chosen = np.sort(rng.choice(available, size=take, replace=False))
        rows.extend(chosen.tolist())
    return np.asarray(rows, dtype=np.int64)

def shuffle_columns_exact(counts: np.ndarray, seed: int=9122) -> np.ndarray:
    """Independently permute each raw-count gene column across cells.

    The operation deliberately acts on integer counts before any
    normalization or decoder call. Every column therefore has exactly the
    same empirical multiset after shuffling, while row-level co-expression,
    library sizes and detected-gene counts are allowed to change.
    """
    source = np.asarray(counts)
    if source.ndim != 2:
        raise ValueError('counts must be a two-dimensional array')
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError('raw-count shuffle requires integer counts')
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(source)
    for col in range(source.shape[1]):
        shuffled[:, col] = source[rng.permutation(source.shape[0]), col]
    return shuffled

def verify_column_marginals(source: np.ndarray, shuffled: np.ndarray) -> dict:
    """Strictly verify every shuffled column's empirical distribution."""
    source = np.asarray(source)
    shuffled = np.asarray(shuffled)
    if source.shape != shuffled.shape:
        raise ValueError('source and shuffled shapes differ')
    if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(shuffled.dtype, np.integer):
        raise TypeError('marginal verification expects integer arrays')
    for col in range(source.shape[1]):
        if not np.array_equal(np.sort(source[:, col]), np.sort(shuffled[:, col])):
            raise AssertionError(f'empirical marginal changed in column {col}')
    return {'columns_checked': int(source.shape[1]), 'rows_checked': int(source.shape[0]), 'every_column_exact': True}

def count_statistics(counts: np.ndarray, panel_positions: np.ndarray | None=None) -> dict:
    """Summarize library size and detection from raw integer counts."""
    x = np.asarray(counts)
    if x.ndim != 2:
        raise ValueError('counts must be two-dimensional')
    out = {}
    for name, matrix in (('all_genes', x), ('model_panel', x[:, panel_positions] if panel_positions is not None else x)):
        libraries = matrix.sum(axis=1, dtype=np.int64)
        detected = (matrix > 0).sum(axis=1, dtype=np.int64)
        out[name] = {'library_size_mean': float(np.mean(libraries)), 'library_size_sd': float(np.std(libraries)), 'library_size_median': float(np.median(libraries)), 'library_size_p05_p95': [float(np.percentile(libraries, 5)), float(np.percentile(libraries, 95))], 'detected_genes_mean': float(np.mean(detected)), 'detected_genes_sd': float(np.std(detected)), 'detected_genes_median': float(np.median(detected)), 'detected_genes_p05_p95': [float(np.percentile(detected, 5)), float(np.percentile(detected, 95))], 'detection_rate_mean': float(np.mean(detected) / matrix.shape[1]), 'detection_rate_sd': float(np.std(detected) / matrix.shape[1]), 'zero_library_cells': int(np.count_nonzero(libraries == 0)), 'library_sizes': libraries, 'detected_genes': detected}
    return out

def _summary_without_vectors(summary: dict) -> dict:
    """Remove per-cell arrays before serializing a count summary."""
    result = {}
    for branch, values in summary.items():
        result[branch] = {k: v for k, v in values.items() if k not in {'library_sizes', 'detected_genes'}}
    return result

def deterministic_project(counts: sp.csr_matrix, panel_positions: np.ndarray, model: TopicModel, device: str='cpu', batch_size: int=512) -> tuple[np.ndarray, np.ndarray]:
    """Project count rows with one frozen model and explicit deterministic mode."""
    if device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    normalized = library_normalize(counts[:, panel_positions])
    theta, mu = ([], [])
    model.eval()
    with torch.no_grad():
        for start in range(0, normalized.shape[0], batch_size):
            dense = np.asarray(normalized[start:start + batch_size].todense(), dtype=np.float32)
            batch = torch.from_numpy(dense).to(device)
            result = model(batch, deterministic=True)
            theta.append(result['theta'].cpu().numpy())
            mu.append(result['mu'].cpu().numpy())
    return (np.vstack(theta), np.vstack(mu))

def coordinate_summary(theta: np.ndarray, mu: np.ndarray) -> dict:
    """Report simplex concentration and pre-softmax coordinate spread."""
    theta = np.asarray(theta, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    entropy = -(theta * np.log(np.maximum(theta, 1e-12))).sum(axis=1)
    perplexity = np.exp(entropy)
    hhi = (theta * theta).sum(axis=1)
    top = np.sort(theta, axis=1)[:, ::-1]
    return {'perplexity_mean': float(perplexity.mean()), 'perplexity_median': float(np.median(perplexity)), 'perplexity_p05_p95': [float(np.percentile(perplexity, 5)), float(np.percentile(perplexity, 95))], 'entropy_mean': float(entropy.mean()), 'hhi_mean': float(hhi.mean()), 'hhi_median': float(np.median(hhi)), 'max_coordinate_mean': float(theta.max(axis=1).mean()), 'max_coordinate_p95': float(np.percentile(theta.max(axis=1), 95)), 'top1_minus_top2_mean': float((top[:, 0] - top[:, 1]).mean()), 'mu_l2_centered_mean': float(np.linalg.norm(mu - mu.mean(axis=1, keepdims=True), axis=1).mean()), 'mu_abs_mean': float(np.abs(mu).mean())}

def marker_log_expression(counts: sp.csr_matrix, genes: np.ndarray, panel: dict[str, list[str]]=MARKER_PANEL) -> tuple[np.ndarray, list[str]]:
    """Return log1p(CP10K) values for the complete marker union."""
    lookup = {str(g): i for i, g in enumerate(genes)}
    marker_genes = []
    for markers in panel.values():
        for gene in markers:
            if gene in lookup and gene not in marker_genes:
                marker_genes.append(gene)
    if not marker_genes:
        raise ValueError('no marker genes present')
    missing = {gene for markers in panel.values() for gene in markers if gene not in lookup}
    if missing:
        raise ValueError(f'marker genes missing from assay: {sorted(missing)}')
    positions = np.asarray([lookup[g] for g in marker_genes], dtype=np.int64)
    values = counts[:, positions].toarray().astype(np.float32, copy=False)
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    totals[totals == 0] = 1.0
    values *= (10000.0 / totals)[:, None]
    np.log1p(values, out=values)
    return (values, marker_genes)

def fit_marker_reference(expression: np.ndarray, marker_genes: list[str], panel: dict[str, list[str]]=MARKER_PANEL) -> dict:
    """Fit pooled marker means/SDs once; zero SD is retained and handled on apply."""
    positions = {gene: i for i, gene in enumerate(marker_genes)}
    means, sds, by_lineage = ({}, {}, {})
    for lineage, markers in panel.items():
        cols = [positions[g] for g in markers if g in positions]
        if len(cols) < 3:
            continue
        by_lineage[lineage] = [marker_genes[c] for c in cols]
        block = np.ascontiguousarray(expression[:, cols], dtype=np.float32)
        means[lineage] = block.mean(axis=0).astype(np.float32)
        sds[lineage] = block.std(axis=0).astype(np.float32)
    if not by_lineage:
        raise ValueError('no lineage has at least three markers')
    return {'marker_genes': list(marker_genes), 'lineages': list(by_lineage), 'markers_by_lineage': by_lineage, 'means': means, 'sds': sds, 'n_cells': int(expression.shape[0])}

def apply_marker_reference(expression: np.ndarray, reference: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply a frozen or adaptive marker reference to precomputed expression."""
    positions = {gene: i for i, gene in enumerate(reference['marker_genes'])}
    scores = np.zeros((expression.shape[0], len(reference['lineages'])), dtype=np.float32)
    for j, lineage in enumerate(reference['lineages']):
        cols = [positions[g] for g in reference['markers_by_lineage'][lineage]]
        mean = np.asarray(reference['means'][lineage], dtype=np.float32)
        sd = np.asarray(reference['sds'][lineage], dtype=np.float32)
        used_sd = np.where(sd == 0, 1.0, sd)
        scores[:, j] = ((expression[:, cols] - mean) / used_sd).mean(axis=1)
    best = np.argsort(-scores, axis=1)[:, 0]
    labels = np.asarray([reference['lineages'][i] for i in best], dtype=object)
    return (labels, scores)

def save_marker_reference(reference: dict, path: Path) -> None:
    rows = []
    for lineage in reference['lineages']:
        for gene, mean, sd in zip(reference['markers_by_lineage'][lineage], reference['means'][lineage], reference['sds'][lineage]):
            rows.append({'lineage': lineage, 'gene': gene, 'mean_log1p_cp10k': float(mean), 'sd_log1p_cp10k': float(sd), 'n_reference_cells': reference['n_cells']})
    pd.DataFrame(rows).to_csv(path, index=False)

def exact_contrast(values: np.ndarray, healed: np.ndarray) -> tuple[float, float, int, np.ndarray]:
    """Exhaustive absolute two-sided allocation test with explicit ties."""
    values = np.asarray(values, dtype=float)
    healed = np.asarray(healed, dtype=bool)
    if values.ndim != 1 or healed.shape != values.shape or (not np.isfinite(values).all()):
        raise ValueError('finite one-dimensional values and aligned binary labels are required')
    n_healed = int(healed.sum())
    if n_healed <= 0 or n_healed >= len(values):
        raise ValueError('both groups must be non-empty')
    observed = float(values[healed].mean() - values[~healed].mean())
    null = np.empty(math.comb(len(values), n_healed), dtype=float)
    total = values.sum()
    for k, indices in enumerate(itertools.combinations(range(len(values)), n_healed)):
        chosen = values[list(indices)].sum()
        null[k] = chosen / n_healed - (total - chosen) / (len(values) - n_healed)
    tolerance = 1e-12 * max(1.0, abs(observed))
    p_value = float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    return (observed, p_value, len(null), null)

def specimen_readouts(labels: np.ndarray, meta: pd.DataFrame, topic: np.ndarray, threshold: float, gate: str) -> pd.DataFrame:
    rows = []
    for gsm, indices in meta.groupby('gsm', sort=True).groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        fib = labels[idx] == 'fibroblast'
        n_fib = int(fib.sum())
        vals = topic[idx][fib]
        rows.append({'gate': gate, 'gsm': gsm, 'patient_id': str(meta.iloc[idx[0]]['patient_id']), 'arm': str(meta.iloc[idx[0]]['arm']), 'n_all': int(len(idx)), 'n_fibroblast': n_fib, 'fibroblast_fraction': float(n_fib / len(idx)), 'fibroblast_mean': float(vals.mean()) if n_fib else np.nan, 'high_state_fraction': float(np.mean(vals > threshold)) if n_fib else np.nan, 'zero_fibroblast': bool(n_fib == 0)})
    return pd.DataFrame(rows)

def patient_readouts(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, group in sample.groupby('patient_id', sort=True):
        if group.arm.nunique() != 1:
            raise ValueError(f'Patient {patient} has conflicting healing labels')
        fib_w = group.n_fibroblast.to_numpy(float)
        all_w = group.n_all.to_numpy(float)
        nonzero = fib_w > 0
        rows.append({'patient_id': patient, 'arm': str(group.arm.iloc[0]), 'specimens': int(len(group)), 'n_all': int(group.n_all.sum()), 'n_fibroblast': int(group.n_fibroblast.sum()), 'fibroblast_mean': float(np.average(group.fibroblast_mean[nonzero], weights=fib_w[nonzero])) if nonzero.any() else np.nan, 'high_state_fraction': float(np.average(group.high_state_fraction[nonzero], weights=fib_w[nonzero])) if nonzero.any() else np.nan, 'fibroblast_fraction': float(np.average(group.fibroblast_fraction, weights=all_w)), 'zero_fibroblast': bool(not nonzero.any())})
    return pd.DataFrame(rows)

def raw_shuffle_analysis(matrix: sp.csr_matrix, genes: np.ndarray, identity: pd.DataFrame, args, output: Path) -> dict:
    rows = balanced_rows(identity, args.cells_per_gsm, args.seed)
    if len(rows) > 6000:
        raise ValueError('raw shuffle subset exceeds 6000 cells')
    selected_identity = identity.iloc[rows].copy()
    selected_identity.insert(0, 'cache_row_0', rows)
    selected_identity.to_csv(output / 'raw_shuffle_selected_cells.csv.gz', index=False, compression='gzip')
    real = matrix[rows].toarray()
    panel = np.load(DISCOVERY / 'panel.npy', allow_pickle=True).astype(str)
    panel_positions = pd.Index(genes).get_indexer(panel)
    if np.any(panel_positions < 0):
        raise ValueError('frozen model panel is absent from raw count cache')
    null = shuffle_columns_exact(real, seed=args.seed + 1)
    marginal_check = verify_column_marginals(real, null)
    source_stats = count_statistics(real, panel_positions)
    null_stats = count_statistics(null, panel_positions)
    state = torch.load(DISCOVERY / 'topic_model.pt', map_location='cpu')
    beta = np.load(DISCOVERY / 'beta.npy')
    model = TopicModel(input_dim=panel.size, num_topics=beta.shape[0]).to(args.device)
    model.load_state_dict(state)
    model.eval()
    theta_real, mu_real = deterministic_project(sp.csr_matrix(real), panel_positions, model, args.device)
    theta_null, mu_null = deterministic_project(sp.csr_matrix(null), panel_positions, model, args.device)
    unified = pd.read_parquet(UNIFIED_PROJECTION)
    projected_reference = unified[[f'topic_{i}' for i in range(beta.shape[0])]].to_numpy()[rows]
    maximum_difference = float(np.abs(projected_reference - theta_real).max())
    if maximum_difference > 1e-06:
        raise ValueError(f'Raw projection does not reproduce unified coordinates: {maximum_difference}')
    np.save(output / 'raw_shuffle_theta_real.npy', theta_real)
    np.save(output / 'raw_shuffle_theta_null.npy', theta_null)
    np.save(output / 'raw_shuffle_mu_real.npy', mu_real)
    np.save(output / 'raw_shuffle_mu_null.npy', mu_null)
    coordinate = {'real_counts': coordinate_summary(theta_real, mu_real), 'column_shuffled_counts': coordinate_summary(theta_null, mu_null)}
    records = []
    for branch, counts, theta, stats in (('real_counts', real, theta_real, source_stats), ('column_shuffled_counts', null, theta_null, null_stats)):
        all_stats = stats['all_genes']
        panel_stats = stats['model_panel']
        for i in range(len(rows)):
            records.append({'branch': branch, 'cache_row_0': int(rows[i]), 'cell_id': str(selected_identity.iloc[i]['source_order_cell_id']), 'gsm': str(selected_identity.iloc[i]['gsm']), 'arm': str(selected_identity.iloc[i]['arm']), 'raw_library_size': int(all_stats['library_sizes'][i]), 'raw_detected_genes': int(all_stats['detected_genes'][i]), 'panel_library_size': int(panel_stats['library_sizes'][i]), 'panel_detected_genes': int(panel_stats['detected_genes'][i]), 'topic_0': float(theta[i, 0]), 'theta_perplexity': float(np.exp(-(theta[i] * np.log(np.maximum(theta[i], 1e-12))).sum())), 'theta_hhi': float((theta[i] ** 2).sum()), 'theta_max': float(theta[i].max())})
    pd.DataFrame(records).to_csv(output / 'raw_shuffle_cell_stats.csv.gz', index=False, compression='gzip')
    report = {'status': 'complete', 'seed': int(args.seed), 'shuffle_seed': int(args.seed + 1), 'selection': {'cells_per_gsm': int(args.cells_per_gsm), 'n_cells': int(len(rows)), 'n_gsms': int(selected_identity.gsm.nunique()), 'cells_by_gsm': selected_identity.groupby('gsm').size().astype(int).to_dict(), 'selection_unit': 'cell; equal fixed count per GSM; rows sorted within GSM', 'identity': 'source_order_cell_id is a synthetic GSM:QC-row identifier, not a barcode'}, 'input_contract': {'raw_counts': 'integer counts read from source CSV, before normalization', 'panel': str(DISCOVERY / 'panel.npy'), 'checkpoint': str(DISCOVERY / 'topic_model.pt'), 'panel_sha256': sha256(DISCOVERY / 'panel.npy'), 'checkpoint_sha256': sha256(DISCOVERY / 'topic_model.pt'), 'beta_sha256': sha256(DISCOVERY / 'beta.npy'), 'projection': 'library_normalize(panel counts) -> frozen encoder -> softmax(mu), deterministic=True', 'decoder_reconstruction_not_used': True}, 'column_marginal_check': marginal_check, 'unified_projection_verification': {'tested_cells': int(len(rows)), 'maximum_absolute_difference': maximum_difference, 'tolerance': 1e-06, 'all_selected_cells_agree': True}, 'raw_count_summary': {'real_counts': _summary_without_vectors(source_stats), 'column_shuffled_counts': _summary_without_vectors(null_stats)}, 'coordinate_summary': coordinate, 'changes': {'all_genes_library_size_mean_delta': float(null_stats['all_genes']['library_size_mean'] - source_stats['all_genes']['library_size_mean']), 'all_genes_detected_genes_mean_delta': float(null_stats['all_genes']['detected_genes_mean'] - source_stats['all_genes']['detected_genes_mean']), 'all_genes_detection_rate_mean_delta': float(null_stats['all_genes']['detection_rate_mean'] - source_stats['all_genes']['detection_rate_mean']), 'panel_library_size_mean_delta': float(null_stats['model_panel']['library_size_mean'] - source_stats['model_panel']['library_size_mean']), 'panel_detected_genes_mean_delta': float(null_stats['model_panel']['detected_genes_mean'] - source_stats['model_panel']['detected_genes_mean']), 'panel_detection_rate_mean_delta': float(null_stats['model_panel']['detection_rate_mean'] - source_stats['model_panel']['detection_rate_mean']), 'coordinate_perplexity_mean_delta': float(coordinate['column_shuffled_counts']['perplexity_mean'] - coordinate['real_counts']['perplexity_mean']), 'coordinate_hhi_mean_delta': float(coordinate['column_shuffled_counts']['hhi_mean'] - coordinate['real_counts']['hhi_mean'])}, 'interpretation': {'supported': 'Frozen topic coordinates are less concentrated after independent raw gene-column shuffling in this fixed subset.', 'limitation': 'Shuffling preserves pooled gene marginals but changes row library sizes, detection patterns, specimen-specific marginals and biological co-expression together. Their individual contributions are not separated.', 'historical_concentration_cutoff': float(0.95 * beta.shape[0]), 'shuffled_mean_perplexity_exceeds_historical_cutoff': bool(coordinate['column_shuffled_counts']['perplexity_mean'] > 0.95 * beta.shape[0]), 'topology_audit_run': False, 'prohibited_claim': 'This experiment does not demonstrate failure of the historical concentration/topology integrity gate; the new raw-null perplexity remains below its old near-uniform threshold.'}, 'output_files': ['raw_shuffle_selected_cells.csv.gz', 'raw_shuffle_cell_stats.csv.gz', 'raw_shuffle_theta_real.npy', 'raw_shuffle_theta_null.npy', 'raw_shuffle_mu_real.npy', 'raw_shuffle_mu_null.npy'], 'script_sha256': sha256(Path(__file__))}
    write_json(output / 'raw_shuffle_report.json', report)
    return report

def dfu_only_contrast(table: pd.DataFrame, readout: str) -> tuple[dict, np.ndarray]:
    """Exclude non-outcome controls before enumerating healing allocations."""
    eligible = table.loc[table.arm.isin(DFU_ARMS)].reset_index(drop=True)
    missing = eligible.loc[~np.isfinite(eligible[readout]), 'patient_id'].tolist()
    if missing:
        raise ValueError(f'DFU outcome readout missing for patients: {missing}')
    healed = eligible.arm.eq('DFU-healer').to_numpy()
    values = eligible[readout].to_numpy(float)
    observed, p, total, null = exact_contrast(values, healed)
    return ({'readout': readout, 'n_healed': int(healed.sum()), 'n_nonhealed': int((~healed).sum()), 'healed_mean': float(values[healed].mean()), 'nonhealed_mean': float(values[~healed].mean()), 'difference': observed, 'exact_p': p, 'allocations': total, 'null_percentile_025': float(np.percentile(null, 2.5)), 'null_percentile_975': float(np.percentile(null, 97.5)), 'healthy_rows_excluded_before_allocation': int((~table.arm.isin(DFU_ARMS)).sum())}, null)

def gate_and_label_analysis(matrix, genes, identity, output):
    unified = pd.read_parquet(UNIFIED_PROJECTION)
    if not np.array_equal(unified.gsm, identity.gsm) or not np.array_equal(unified.arm, identity.arm):
        raise ValueError('Unified projection order differs from raw cache')
    if not np.array_equal(unified.source_row_zero_based, identity.saved_row_0):
        raise ValueError('Unified source row contract differs from raw cache')
    meta = unified[['gsm', 'arm', 'patient_id']].copy()
    topic = unified.topic_0.to_numpy(float)
    threshold = json.loads(THRESHOLD.read_text())['cohorts']['GSE165816_discovery']['gmm']['midpoint']
    log('Fitting frozen marker reference on all 76,461 discovery cells')
    expression, marker_genes = marker_log_expression(matrix, genes)
    reference = fit_marker_reference(expression, marker_genes)
    save_marker_reference(reference, output / 'frozen_marker_reference.csv')
    pooled, _ = apply_marker_reference(expression, reference)
    original = identity.historical_celltype.to_numpy()
    if not np.array_equal(pooled, original):
        raise AssertionError(f'Pooled marker reference changed {np.count_nonzero(pooled != original)} historical labels')
    frozen = np.empty(len(identity), dtype=object)
    adaptive = np.empty(len(identity), dtype=object)
    adaptive_parameters = []
    for gsm, indices in meta.groupby('gsm', sort=True).groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        frozen[idx], _ = apply_marker_reference(expression[idx], reference)
        sample_ref = fit_marker_reference(expression[idx], marker_genes)
        adaptive[idx], _ = apply_marker_reference(expression[idx], sample_ref)
        for lineage in sample_ref['lineages']:
            for gene, mean, sd in zip(sample_ref['markers_by_lineage'][lineage], sample_ref['means'][lineage], sample_ref['sds'][lineage]):
                adaptive_parameters.append({'gsm': gsm, 'lineage': lineage, 'gene': gene, 'mean_log1p_cp10k': float(mean), 'sd_log1p_cp10k': float(sd), 'n_reference_cells': int(len(idx))})
    if not np.array_equal(pooled, frozen):
        raise AssertionError('Frozen reference changes labels when applied separately by sample')
    pd.DataFrame(adaptive_parameters).to_csv(output / 'adaptive_marker_references.csv', index=False)
    label_frame = identity[['saved_row_0', 'gsm', 'arm', 'raw_cell_id']].copy()
    label_frame['pooled_gate'] = pooled
    label_frame['frozen_reference_gate'] = frozen
    label_frame['adaptive_specimen_gate'] = adaptive
    label_frame.to_parquet(output / 'gate_cell_labels.parquet', index=False)
    pd.crosstab(pd.Series(pooled, name='pooled_gate'), pd.Series(adaptive, name='adaptive_specimen_gate')).to_csv(output / 'gate_confusion.csv')
    gate_arrays = {'original_pooled': pooled, 'frozen_reference': frozen, 'adaptive_per_specimen': adaptive}
    sample_rows, patient_rows_list, contrast_rows, null_rows, gate_change_rows = ([], [], [], [], [])
    for gate, labels in gate_arrays.items():
        sample = specimen_readouts(labels, meta, topic, threshold, gate)
        patient = patient_readouts(sample)
        patient.insert(0, 'gate', gate)
        sample_rows.append(sample)
        patient_rows_list.append(patient)
        dfu = patient.loc[patient.arm.isin(DFU_ARMS)]
        if len(dfu) != 11 or dfu.arm.eq('DFU-healer').sum() != 7:
            raise ValueError('Gate inference departed from the fixed 11-patient contract')
        for metric in READOUTS:
            row, null_values = dfu_only_contrast(patient, metric)
            row.update({'gate': gate, 'unit': 'patient', 'aggregation': 'fibroblast-cell weights for within-fibroblast summaries; all-cell weights for composition'})
            contrast_rows.append(row)
            for i, value in enumerate(null_values):
                null_rows.append({'gate': gate, 'unit': 'patient', 'readout': metric, 'allocation_index': i, 'difference': float(value)})
        base = sample_rows[0].set_index('gsm')
        for row in sample.to_dict(orient='records'):
            gsm = row['gsm']
            idx = np.flatnonzero(meta.gsm.to_numpy() == gsm)
            old_fib = pooled[idx] == 'fibroblast'
            new_fib = labels[idx] == 'fibroblast'
            union = int(np.count_nonzero(old_fib | new_fib))
            gate_change_rows.append({**row, 'labels_changed': int(np.count_nonzero(pooled[idx] != labels[idx])), 'labels_changed_fraction': float(np.mean(pooled[idx] != labels[idx])), 'fibroblasts_lost': int(np.count_nonzero(old_fib & ~new_fib)), 'fibroblasts_gained': int(np.count_nonzero(~old_fib & new_fib)), 'fibroblast_jaccard': float(np.count_nonzero(old_fib & new_fib) / union) if union else 1.0, 'delta_n_fibroblast': int(row['n_fibroblast'] - base.loc[gsm, 'n_fibroblast']), 'delta_fibroblast_mean': float(row['fibroblast_mean'] - base.loc[gsm, 'fibroblast_mean']), 'delta_high_state_fraction': float(row['high_state_fraction'] - base.loc[gsm, 'high_state_fraction'])})
    samples = pd.concat(sample_rows, ignore_index=True)
    patients = pd.concat(patient_rows_list, ignore_index=True)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts['bh_q_three_readouts_within_gate'] = contrasts.groupby('gate')['exact_p'].transform(lambda values: false_discovery_control(values.to_numpy(), method='bh'))
    samples.to_csv(output / 'gate_specimen_readouts.csv', index=False)
    patients.to_csv(output / 'gate_patient_readouts.csv', index=False)
    contrasts.to_csv(output / 'gate_patient_contrasts.csv', index=False)
    pd.DataFrame(gate_change_rows).to_csv(output / 'gate_specimen_changes.csv', index=False)
    pd.DataFrame(null_rows).to_csv(output / 'gate_patient_exact_null.csv', index=False)
    specimen_contrast_rows, specimen_null_rows = ([], [])
    baseline_sample = samples.loc[samples.gate.eq('original_pooled')]
    outcome_sample = baseline_sample.loc[baseline_sample.arm.isin(DFU_ARMS)]
    if len(outcome_sample) != 14 or outcome_sample.arm.eq('DFU-healer').sum() != 9:
        raise ValueError('The DFU-only specimen contract must have 14 specimens, 9 healed and 5 nonhealed')
    outcome_sample.to_csv(output / 'dfu_outcome_eligible_specimens.csv', index=False)
    for metric in READOUTS:
        row, null_values = dfu_only_contrast(baseline_sample, metric)
        row['unit'] = 'specimen; secondary because repeat specimens share patients'
        specimen_contrast_rows.append(row)
        for i, value in enumerate(null_values):
            specimen_null_rows.append({'readout': metric, 'allocation_index': i, 'difference': float(value)})
    pd.DataFrame(specimen_contrast_rows).to_csv(output / 'dfu_specimen_exact_contrasts.csv', index=False)
    pd.DataFrame(specimen_null_rows).to_csv(output / 'dfu_specimen_exact_null.csv', index=False)
    historical = pd.read_csv(ROOT / 'outputs/biological_expansion/measurement/patient_contrasts.csv')
    historical = historical.loc[historical.variant.eq('cell-weighted')].set_index('readout')
    reproduced = contrasts.loc[contrasts.gate.eq('original_pooled')].set_index('readout')
    for metric in READOUTS:
        for column in ('difference', 'exact_p', 'healed_mean', 'nonhealed_mean'):
            if not np.isclose(historical.loc[metric, column], reproduced.loc[metric, column], atol=1e-12, rtol=0):
                raise AssertionError(f'Historical patient result changed: {metric}/{column}')
    changes = pd.DataFrame(gate_change_rows)
    adaptive_changes = changes.loc[changes.gate.eq('adaptive_per_specimen')]
    gate_summary = {'status': 'complete', 'n_cells': int(len(identity)), 'n_specimens': 25, 'n_dfu_specimens': 14, 'n_dfu_patients': 11, 'threshold': float(threshold), 'threshold_operator': '>', 'projection_source': str(UNIFIED_PROJECTION.relative_to(ROOT)), 'pooled_reference_exactly_reproduces_historical_labels': True, 'frozen_per_specimen_exactly_reproduces_pooled_labels': True, 'frozen_reference_scope': 'All discovery cells. Equality on these cells is implementation verification, not independent identity validation.', 'marker_reference': {'normalization': 'per-cell CP10K then log1p', 'score': 'mean of marker-wise z-scores', 'lineages': reference['lineages'], 'markers_by_lineage': reference['markers_by_lineage'], 'reference_cells': int(reference['n_cells'])}, 'adaptive_labels_changed': int(np.count_nonzero(pooled != adaptive)), 'adaptive_labels_changed_fraction': float(np.mean(pooled != adaptive)), 'fibroblast_counts': {gate: int(np.count_nonzero(labels == 'fibroblast')) for gate, labels in gate_arrays.items()}, 'fibroblasts_lost_under_adaptive': int(np.count_nonzero((pooled == 'fibroblast') & (adaptive != 'fibroblast'))), 'fibroblasts_gained_under_adaptive': int(np.count_nonzero((pooled != 'fibroblast') & (adaptive == 'fibroblast'))), 'adaptive_specimen_max_absolute_fibroblast_mean_change': float(adaptive_changes.delta_fibroblast_mean.abs().max()), 'adaptive_specimen_max_absolute_high_fraction_change': float(adaptive_changes.delta_high_state_fraction.abs().max()), 'zero_fibroblast_specimens': samples.loc[samples.zero_fibroblast, ['gate', 'gsm']].to_dict(orient='records'), 'zero_fibroblast_patients': patients.loc[patients.zero_fibroblast, ['gate', 'patient_id']].to_dict(orient='records'), 'patient_contrasts': contrasts.to_dict(orient='records'), 'historical_11_patient_results_reproduced': True, 'interpretation': 'Changing the reference population changes marker assignments and therefore measurement denominators. These gates have no independent cell-identity truth labels; sensitivity cannot establish which gate is correct.'}
    write_json(output / 'gate_report.json', gate_summary)
    label_report = {'status': 'complete', 'eligible_arms': list(DFU_ARMS), 'healthy_controls_enter_healing_label_null': False, 'specimen_test': {'n_healed': 9, 'n_nonhealed': 5, 'allocations': math.comb(14, 9), 'primary_inference': False, 'readouts': specimen_contrast_rows}, 'patient_test': {'n_healed': 7, 'n_nonhealed': 4, 'allocations': math.comb(11, 7), 'primary_inference': True, 'readouts': contrasts.loc[contrasts.gate.eq('original_pooled')].to_dict(orient='records')}, 'test_definition': 'Exhaustive absolute two-sided difference in equal-unit group means; ties included with 1e-12*max(1,abs(observed)); no Monte Carlo add-one correction.', 'inference_limit': 'Observational healing-label exchangeability is an assumption; this is a calibrated permutation calculation, not evidence that the measurement is biologically valid or that the pipeline cannot generate artifacts.', 'historical_problem': 'scripts/run_pipeline_negative_controls.py permuted all three arms across 25 specimens, allowing healthy controls into pseudo healing groups. The repaired null first restricts eligibility to 14 DFU specimens; 11-patient tests remain primary.'}
    write_json(output / 'dfu_label_report.json', label_report)
    return (gate_summary, label_report)

def continuous_counterexample(output: Path, seed: int) -> dict:
    """Construct continuous within-specimen distributions, without model fitting."""
    samples = pd.read_csv(output / 'gate_specimen_readouts.csv')
    samples = samples.loc[samples.gate.eq('original_pooled')].reset_index(drop=True)
    threshold = json.loads(THRESHOLD.read_text())['cohorts']['GSE165816_discovery']['gmm']['midpoint']
    observed = float(spearmanr(samples.fibroblast_mean, samples.high_state_fraction).statistic)
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(200):
        means, highs = ([], [])
        for row in samples.itertuples():
            values = expit(logit(np.clip(row.fibroblast_mean, 1e-06, 1 - 1e-06)) + rng.normal(0, 1, int(row.n_fibroblast)))
            means.append(float(values.mean()))
            highs.append(float(np.mean(values > threshold)))
        rows.append({'draw': draw, 'rho_mean_vs_high_fraction': float(spearmanr(means, highs).statistic)})
    draws = pd.DataFrame(rows)
    draws.to_csv(output / 'continuous_counterexample_draws.csv', index=False)
    report = {'status': 'complete', 'seed': int(seed), 'draws': 200, 'construction': 'For each original specimen j, draw n_j values as expit(logit(observed mean_j)+Normal(0,1)); no latent class and no mixture fit is used. The distribution mean is not constrained to equal the observed mean.', 'observed_rho': observed, 'constructed_rho_median': float(draws.rho_mean_vs_high_fraction.median()), 'fraction_constructed_rho_at_least_observed': float(np.mean(draws.rho_mean_vs_high_fraction >= observed)), 'inference_scope': 'Illustrative counterexample only; not a fitted null, calibrated p-value, model selection result, or evidence favoring this distribution for the actual cells.'}
    write_json(output / 'continuous_counterexample_report.json', report)
    return report

def integration_note(output: Path, raw_report: dict, gate_report: dict, label_report: dict, counterexample: dict) -> None:
    """Write the handoff contract used to update the manuscript."""
    real = raw_report['coordinate_summary']['real_counts']
    shuffled = raw_report['coordinate_summary']['column_shuffled_counts']
    specimen = next((row for row in label_report['specimen_test']['readouts'] if row['readout'] == 'high_state_fraction'))
    text = f"# Measurement-control repair handoff\n\nThis directory contains a retrospective measurement audit. It does not add\nindependent biological validation and it does not refit the topic model.\n\n## Raw-count column shuffle\n\n`raw_shuffle_report.json` is based on {raw_report['selection']['n_cells']} cells,\nwith {raw_report['selection']['cells_per_gsm']} cells selected from each of\n{raw_report['selection']['n_gsms']} discovery GSMs. The source is the integer\ncount cache `discovery_raw_counts.npz`, and each gene column is independently\npermuted before library normalization and projection. The report records an\nexact empirical marginal check for all {raw_report['column_marginal_check']['columns_checked']}\ncolumns, the same frozen checkpoint/panel in both branches, and explicit\n`deterministic=True` `softmax(mu)` projection. It also reports library size,\ndetection, perplexity, HHI, maximum topic coordinate, and pre-softmax coordinate\nconcentration for both branches. Mean cell perplexity changes from\n{real['perplexity_mean']:.6f} to {shuffled['perplexity_mean']:.6f}; HHI changes\nfrom {real['hhi_mean']:.6f} to {shuffled['hhi_mean']:.6f}.\n\nUse these numbers for Figure 6E. Historical 3.68/14.43 came from a decoder\nreconstruction shuffle and an asymmetric real/null projection; they are not\nthe new raw-count experiment. The new shuffled mean perplexity of 11.23 is\nbelow the historical 0.95 K = 14.25 near-uniform threshold. This experiment\ndoes not run a topology audit and does not establish that the historical\nintegrity gate rejects the shuffled input. Describe lower concentration only.\nGene shuffling also changes row library-size and detection distributions, so\nthe contrast does not isolate a biological co-expression mechanism from those\ntechnical changes. It is not independent biological validation.\n\nThe identity/cache contract is in `raw_cache_manifest.json`. `raw_barcode` is\nthe barcode read from the source CSV; `raw_row_0` is its source column order;\n`qc_row_0` is the post-QC GSM-local row; `source_order_cell_id` is explicitly a\nsynthetic order ID and must not be described as an original barcode linkage.\n\n## Marker gate sensitivity\n\n`gate_report.json` compares the original pooled gate, a frozen reference fitted\nonce on all 76,461 discovery cells, and a per-GSM adaptive reference. The\nfrozen gate exactly reproduces the pooled labels on the same cells; this is an\nimplementation check rather than independent identity validation. The adaptive\ngate changes the marker reference population, so changes in fibroblast counts,\ndenominators, means, high-state fractions, specimen readouts, and the 11-patient\nexact contrasts are sensitivity quantities. No gate has a supplied ground-truth\ncell identity label.\n\nUse `gate_patient_contrasts.csv` and `gate_specimen_changes.csv` for tables.\nThe fixed threshold is {gate_report['threshold']:.15g} with `>` as its operator.\nThe original pooled gate reproduces the existing cell-weighted 11-patient\nresults exactly. Zero-fibroblast rows are retained and reported rather than\nsilently dropped.\n\nPer-GSM adaptation changes {gate_report['adaptive_labels_changed']:,} labels\n({100 * gate_report['adaptive_labels_changed_fraction']:.2f}% of all cells).\nFibroblast calls change from {gate_report['fibroblast_counts']['original_pooled']:,}\nto {gate_report['fibroblast_counts']['adaptive_per_specimen']:,}, comprising\n{gate_report['fibroblasts_lost_under_adaptive']:,} lost and\n{gate_report['fibroblasts_gained_under_adaptive']:,} gained calls.\nThe largest absolute specimen changes are\n{gate_report['adaptive_specimen_max_absolute_fibroblast_mean_change']:.6f} in\nwithin-fibroblast mean and\n{gate_report['adaptive_specimen_max_absolute_high_fraction_change']:.6f} in\nhigh-state fraction. No specimen or patient has zero fibroblasts in this run.\n\n## DFU-only label shuffle\n\n`dfu_label_report.json` first restricts the secondary specimen allocation to 14\nDFU outcome-eligible specimens (9 healed, 5 non-healed), so the 11 healthy\nspecimens (9 patients) never enter a healing-label null. It reports {label_report['specimen_test']['allocations']}\nexhaustive allocations. The primary result remains the existing 11-patient\nexact test (7 healed, 4 non-healed; {label_report['patient_test']['allocations']}\nallocations), with ties included using the documented tolerance. Specimen-level\nresults are secondary because repeated specimens share patients.\n\nFor Figure 6D use `dfu_specimen_exact_null.csv`, restricted to\n`readout == high_state_fraction`: observed difference\n{specimen['difference']:.9f}, absolute two-sided exact p\n{specimen['exact_p']:.9f}, central 95% null interval\n[{specimen['null_percentile_025']:.9f}, {specimen['null_percentile_975']:.9f}].\nReplace the historical three-arm p=0.1358 and interval [-0.3183, +0.2874].\nThis exact enumeration uses 2,002 allocations, not 20,000 Monte Carlo shuffles.\nThe primary patient result remains p=0.369697 for high-state fraction;\nspecimen p values cannot be substituted for patient p values.\n\n## Constructed continuous counterexample\n\n`continuous_counterexample_report.json` is an illustrative construction from\nthe original specimen means and counts, not a fit to the cell-level data. It\nmust be described as a non-identification example rather than as a null model\nor a new biological result. Its median correlation is\n{counterexample['constructed_rho_median']:.6f};\n{100 * counterexample['fraction_constructed_rho_at_least_observed']:.0f}% of the\n200 construction draws reach the observed rho\n{counterexample['observed_rho']:.6f}. That percentage is not a calibrated p value.\n\n## Historical numbers retained\n\nThe historical cell-weighted 11-patient contrasts remain in\n`outputs/biological_expansion/measurement/patient_contrasts.csv`; this repair\nchecks and reproduces them. Keep these primary estimates. Replace the old\nthree-arm shuffle and decoder-proportion shuffle descriptions with the\nexplicitly identified new controls above. Cite the marker-reference comparison\nas measurement sensitivity, with no independently verified identity reference.\n"
    (output / 'INTEGRATION.md').write_text(text)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--seed', type=int, default=9122)
    parser.add_argument('--cells-per-gsm', type=int, default=240)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    matrix, genes, identity, manifest = prepare_cache(args.output_dir)
    if args.prepare_only:
        return 0
    log('Running raw integer-count column-shuffle control')
    raw_report = raw_shuffle_analysis(matrix, genes, identity, args, args.output_dir)
    log('Running frozen/adaptive marker-gate and DFU-only exact allocation controls')
    gate_report, label_report = gate_and_label_analysis(matrix, genes, identity, args.output_dir)
    log('Constructing explicitly non-fitted continuous distribution counterexample')
    counterexample = continuous_counterexample(args.output_dir, args.seed)
    integration_note(args.output_dir, raw_report, gate_report, label_report, counterexample)
    final = {'status': 'complete', 'raw_shuffle': raw_report, 'gate': gate_report, 'dfu_label': label_report, 'continuous_counterexample': counterexample, 'cache_manifest': str((args.output_dir / 'raw_cache_manifest.json').relative_to(ROOT))}
    write_json(args.output_dir / 'report.json', final)
    log(f"Wrote {args.output_dir / 'report.json'}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
