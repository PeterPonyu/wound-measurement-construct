#!/usr/bin/env python3
"""RT06: raw-count -> QC -> frozen marker gate -> deterministic frozen score.

Only fresh output directories are writable.  Published source matrices, the
model, references and historical results are read-only.  Use --mode smoke for
a bounded real-data executable check; --mode full includes every raw column
from all 25 specimens and all protocol-fixed rates/seeds.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.stats import spearmanr
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.celltype_markers import MARKER_PANEL
from wound_models.data_adapter import library_normalize, mito_fraction, parse_series_matrix
from wound_models.topic_model import TopicModel
_SPEC = importlib.util.spec_from_file_location('depth_existing_controls', ROOT / 'scripts/repair_measurement_controls.py')
_CONTROL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_CONTROL)
DEFAULT_PROTOCOL = ROOT / 'config/full_pipeline_depth_protocol.json'
DEFAULT_OUTPUT = ROOT / 'outputs/scientific_revision_20260925/full_pipeline_depth/full'
DFU_ARMS = ('DFU-healer', 'DFU-nonhealer')
READOUTS = ('all_cell_topic0_mean', 'fibroblast_topic0_mean', 'fibroblast_high_state_fraction', 'fibroblast_fraction')
POPULATIONS = ('full_pipeline', 'original_qc_fixed_gate', 'qc_intersection_fixed_gate', 'qc_intersection_regated')
LOG = logging.getLogger('full_pipeline_depth')

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, data):
    """Fail on nonfinite JSON rather than hiding an invalid analysis."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def text_array_hash(values):
    return hashlib.sha256(json.dumps(list(map(str, values)), ensure_ascii=False).encode()).hexdigest()

def sparse_hash(matrix):
    digest = hashlib.sha256()
    for value in (np.asarray(matrix.shape, dtype='<i8'), matrix.indptr.astype('<i8'), matrix.indices.astype('<i8'), matrix.data.astype('<i8')):
        digest.update(value.tobytes())
    return digest.hexdigest()

def fresh_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path

def validate_protocol(protocol):
    """Reject accidental refits, cohort changes and incomplete perturbation grids."""
    if protocol['protocol_id'] != 'RT06-full-raw-depth-20260925-v1':
        raise ValueError('Unknown protocol; this runner implements the fixed v1 contract')
    if protocol['perturbation']['baseline_rate'] != 1.0:
        raise ValueError('Baseline must be unthinned')
    if protocol['perturbation']['retention_rates'] != [0.75, 0.5, 0.25, 0.1]:
        raise ValueError('Do not silently change the fixed retention-rate grid')
    if protocol['perturbation']['seeds'] != [20260925, 20260926, 20260927]:
        raise ValueError('Do not silently change the fixed seeds')
    if protocol['qc'] != {'minimum_detected_genes': 200, 'maximum_mito_fraction': 0.2}:
        raise ValueError('QC must reproduce the frozen historical thresholds')
    if tuple(protocol['populations']) != POPULATIONS:
        raise ValueError('Population definitions differ from implemented endpoints')

def runtime_provenance(log_path, requested):
    """Read only model/cwd metadata, never prompts, tools, credentials or peers."""
    result = {'requested_model': requested, 'runtime_reported_model': None, 'runtime_metadata_source': None, 'runtime_timestamp': None, 'same_exact_model_verified': False}
    if log_path is None:
        result['limitation'] = 'Runtime metadata not supplied; requested model is not verification.'
        return result
    found = None
    with Path(log_path).open() as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get('type') == 'turn_context':
                payload = item.get('payload', {})
                if payload.get('cwd') == str(ROOT):
                    found = (item.get('timestamp'), payload.get('model'), payload.get('effort'))
    if found is None:
        raise ValueError('No matching workspace turn_context in runtime log')
    result.update(runtime_reported_model=found[1], runtime_metadata_source=str(log_path), runtime_timestamp=found[0], runtime_reasoning_effort=found[2], same_exact_model_verified=found[1] == requested)
    if found[1] != requested:
        raise ValueError(f'Actual runtime model {found[1]} differs from requested {requested}')
    return result

def read_count_member(payload, compressed=True, chunk_rows=256, max_cells=None):
    """Convert a dense genes x cells member to CSR using bounded gene chunks."""

    def source():
        buf = io.BytesIO(payload)
        return gzip.GzipFile(fileobj=buf, mode='rb') if compressed else buf
    with io.TextIOWrapper(source(), encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header, first_data = (next(reader), next(reader))
    if len(first_data) == len(header) + 1:
        barcodes = np.asarray(header, dtype=str)
    elif len(first_data) == len(header):
        barcodes = np.asarray(header[1:], dtype=str)
    else:
        raise ValueError('CSV header/data field counts are inconsistent')
    if not len(barcodes) or len(set(barcodes)) != len(barcodes):
        raise ValueError('Missing or duplicated original CSV barcodes')
    full_columns = len(barcodes)
    names = ['__source_gene_index__', *barcodes]
    if max_cells is not None:
        barcodes = barcodes[:max_cells]
    columns = list(range(len(barcodes) + 1))
    blocks, genes = ([], [])
    with source() as stream:
        for frame in pd.read_csv(stream, header=0, names=names, index_col=0, usecols=columns, chunksize=chunk_rows):
            if not np.array_equal(frame.columns.to_numpy(str), barcodes):
                raise ValueError('CSV reader changed barcode identity/order')
            values = frame.to_numpy()
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any() or (values > np.iinfo(np.int32).max).any():
                raise ValueError('Source contains noninteger, negative or overflowing counts')
            genes.extend(frame.index.astype(str))
            blocks.append(sp.csr_matrix(values.T.astype(np.int32)))
    if not blocks or len(set(genes)) != len(genes):
        raise ValueError('Missing or duplicated source gene symbols')
    matrix = sp.hstack(blocks, format='csr', dtype=np.int32)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    return (matrix, np.asarray(genes, dtype=str), barcodes, full_columns)

def align_panel(counts, genes, panel):
    """Zero fill sample-absent detected genes; never index absent genes with -1."""
    pos = pd.Index(genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    subset = counts[:, pos[present]].tocoo()
    aligned = sp.csr_matrix((subset.data, (subset.row, present[subset.col])), shape=(counts.shape[0], len(panel)), dtype=np.int32)
    return (aligned, len(present))

def thin_counts(counts, rate, seed, gsm):
    if not sp.isspmatrix_csr(counts) or not np.issubdtype(counts.dtype, np.integer):
        raise TypeError('Thinning requires CSR integer raw counts')
    if (counts.data < 0).any() or not 0 <= rate <= 1:
        raise ValueError('Counts must be nonnegative and rate must lie in [0, 1]')
    result = counts.copy()
    result.sum_duplicates()
    result.sort_indices()
    if rate < 1:
        match = re.fullmatch('GSM(\\d+)', gsm)
        if match is None:
            raise ValueError('Expected a GSM accession for sample-specific RNG')
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(round(rate * 1000000)), int(match.group(1))]))
        result.data = rng.binomial(result.data, rate).astype(counts.dtype)
        result.eliminate_zeros()
    return result

def qc_metrics(counts, genes, minimum_genes=200, maximum_mito=0.2):
    detected = np.asarray((counts > 0).sum(axis=1)).ravel()
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    mito = mito_fraction(counts, genes)
    retained = (detected >= minimum_genes) & (mito <= maximum_mito)
    return (retained, detected, library, mito)

def load_marker_reference(path):
    table = pd.read_csv(path)
    if table[['lineage', 'gene']].duplicated().any():
        raise ValueError('Duplicate marker reference rows')
    lineages = list(table.lineage.unique())
    if lineages != list(MARKER_PANEL):
        raise ValueError('Frozen lineage order differs from current marker contract')
    reference = {'lineages': lineages, 'marker_genes': list(dict.fromkeys(table.gene)), 'markers_by_lineage': {}, 'means': {}, 'sds': {}}
    if table.n_reference_cells.nunique() != 1 or int(table.n_reference_cells.iloc[0]) != 76461:
        raise ValueError('Frozen reference must be the 76,461-cell historical reference')
    for lineage in lineages:
        rows = table.loc[table.lineage.eq(lineage)]
        if rows.gene.tolist() != MARKER_PANEL[lineage]:
            raise ValueError(f'Frozen markers changed for {lineage}')
        reference['markers_by_lineage'][lineage] = rows.gene.tolist()
        for key, column in (('means', 'mean_log1p_cp10k'), ('sds', 'sd_log1p_cp10k')):
            value = rows[column].to_numpy(np.float32)
            if not np.isfinite(value).all() or (key == 'sds' and (value < 0).any()):
                raise ValueError('Invalid frozen marker parameters')
            reference[key][lineage] = value
    return reference

def frozen_gate(counts, genes, reference):
    panel_counts, _ = align_panel(counts, genes, reference['marker_genes'])
    values = panel_counts.toarray().astype(np.float32)
    detected_markers = (values > 0).sum(axis=1)
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    totals[totals == 0] = 1
    values *= (10000.0 / totals)[:, None]
    np.log1p(values, out=values)
    labels, scores = _CONTROL.apply_marker_reference(values, reference)
    ordered = np.sort(scores, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return (labels.astype(str), margin, detected_markers)

def project_panel(counts, model, device='cpu', batch_size=512):
    """Equivalent deterministic encoder-only projection; decoder is not needed."""
    normalized = library_normalize(counts)
    nonzero = np.asarray(counts.sum(axis=1)).ravel() > 0
    result = np.full((counts.shape[0], model.num_topics), np.nan, dtype=np.float32)
    model.eval()
    indices = np.flatnonzero(nonzero)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            batch = torch.from_numpy(normalized[rows].toarray()).to(device)
            mu, _ = model.encoder(batch)
            result[rows] = torch.softmax(mu, dim=-1).cpu().numpy()
    if len(indices) and (not np.isfinite(result[indices]).all() or not np.allclose(result[indices].sum(1), 1, atol=1e-06, rtol=0)):
        raise ValueError('Frozen projection failed simplex/finite checks')
    return (result, nonzero)

def evaluate_counts(counts, genes, panel, reference, model, device, batch_size):
    keep, detected, library, mito = qc_metrics(counts, genes)
    labels, margin, marker_count = frozen_gate(counts, genes, reference)
    aligned, coverage = align_panel(counts, genes, panel)
    theta, nonzero = project_panel(aligned, model, device, batch_size)
    frame = pd.DataFrame({'raw_row_0': np.arange(counts.shape[0]), 'qc_pass': keep, 'detected_genes': detected, 'library_size': library, 'mito_fraction': mito, 'celltype': labels, 'marker_top_two_margin': margin, 'detected_marker_genes': marker_count, 'score_available': nonzero})
    for topic in range(theta.shape[1]):
        frame[f'topic_{topic}'] = theta[:, topic]
    return (frame, coverage)

def attach_metadata(frame, gsm, arm, patient, barcodes):
    frame = frame.copy()
    frame.insert(0, 'gsm', gsm)
    frame.insert(1, 'arm', arm)
    frame.insert(2, 'patient_id', patient)
    frame.insert(3, 'raw_cell_id', [f'{gsm}:{b}' for b in barcodes])
    return frame

def verify_baseline(frame, expected, raw_identity, tolerance):
    """Align by raw source IDs, then verify every retained label/coordinate."""
    retained = frame.loc[frame.qc_pass].reset_index(drop=True)
    if not np.array_equal(retained.raw_cell_id, raw_identity.raw_cell_id):
        raise ValueError('Full-raw QC identity differs from historical raw-barcode mapping')
    if not np.array_equal(retained.raw_row_0, raw_identity.raw_row_0):
        raise ValueError('Full-raw source rows differ from historical raw identity')
    if len(retained) != len(expected):
        raise ValueError('Baseline retained cells differ from historical projection')
    for column in ('gsm', 'arm', 'celltype'):
        if not np.array_equal(retained[column].to_numpy(str), expected[column].to_numpy(str)):
            raise ValueError(f'Baseline historical {column} mismatch')
    if not np.array_equal(retained.library_size, raw_identity.raw_library_size):
        raise ValueError('Raw library sizes differ from historical reconstruction')
    if not np.array_equal(retained.detected_genes, raw_identity.detected_genes):
        raise ValueError('Raw detected-gene counts differ from historical reconstruction')
    topics = [f'topic_{i}' for i in range(15)]
    difference = float(np.max(np.abs(retained[topics].to_numpy(float) - expected[topics].to_numpy(float)))) if len(retained) else 0.0
    if not np.isfinite(difference) or difference > tolerance:
        raise ValueError(f'Baseline coordinates differ from historical projection: {difference}')
    return {'gsm': str(frame.gsm.iloc[0]), 'raw_cells': len(frame), 'qc_cells': len(retained), 'fibroblasts': int(retained.celltype.eq('fibroblast').sum()), 'raw_ids_exact': True, 'marker_labels_exact': True, 'raw_libraries_exact': True, 'raw_detected_genes_exact': True, 'coordinate_max_abs_difference': difference}

def population_masks(baseline, current):
    original = baseline.qc_pass.to_numpy(bool)
    keep = current.qc_pass.to_numpy(bool)
    original_gate = baseline.celltype.to_numpy(str)
    current_gate = current.celltype.to_numpy(str)
    return {'full_pipeline': (keep, current_gate), 'original_qc_fixed_gate': (original, original_gate), 'qc_intersection_fixed_gate': (original & keep, original_gate), 'qc_intersection_regated': (original & keep, current_gate)}

def readout_row(frame, mask, labels, threshold):
    values = frame.topic_0.to_numpy(np.float64)
    fib = mask & (labels == 'fibroblast')
    finite = np.isfinite(values)
    n_all, n_fib = (int(mask.sum()), int(fib.sum()))
    missing, missing_fib = (int((mask & ~finite).sum()), int((fib & ~finite).sum()))
    total = float(values[mask & finite].sum())
    fib_total = float(values[fib & finite].sum())
    high = int((fib & finite & (values > threshold)).sum())
    return {'n_all': n_all, 'n_fibroblast': n_fib, 'n_missing_score': missing, 'n_missing_fibroblast_score': missing_fib, 'topic0_sum': total, 'fibroblast_topic0_sum': fib_total, 'fibroblast_high_count': high, 'all_cell_topic0_mean': total / n_all if n_all and (not missing) else np.nan, 'fibroblast_topic0_mean': fib_total / n_fib if n_fib and (not missing_fib) else np.nan, 'fibroblast_high_state_fraction': high / n_fib if n_fib and (not missing_fib) else np.nan, 'fibroblast_fraction': n_fib / n_all if n_all else np.nan}

def specimen_readouts(baseline, current, tag, rate, seed, threshold):
    meta = {key: str(current[key].iloc[0]) for key in ('gsm', 'arm', 'patient_id')}
    return [{'run': tag, 'rate': rate, 'seed': seed, 'population': population, **meta, **readout_row(current, mask, labels, threshold)} for population, (mask, labels) in population_masks(baseline, current).items()]

def aggregate_patients(samples):
    """Never weight fibroblast means by all cells or omit empty specimens."""
    records = []
    keys = ['run', 'rate', 'seed', 'population', 'patient_id']
    for name, block in samples.groupby(keys, dropna=False, sort=True):
        if block.arm.nunique() != 1:
            raise ValueError('Patient outcome labels disagree across specimens')
        n_all, n_fib = (int(block.n_all.sum()), int(block.n_fibroblast.sum()))
        missing, missing_fib = (int(block.n_missing_score.sum()), int(block.n_missing_fibroblast_score.sum()))
        row = dict(zip(keys, name))
        row.update(arm=str(block.arm.iloc[0]), n_specimens=len(block), n_empty_specimens=int(block.n_all.eq(0).sum()), n_all=n_all, n_fibroblast=n_fib, n_missing_score=missing, n_missing_fibroblast_score=missing_fib, all_cell_topic0_mean=float(block.topic0_sum.sum() / n_all) if n_all and (not missing) else np.nan, fibroblast_topic0_mean=float(block.fibroblast_topic0_sum.sum() / n_fib) if n_fib and (not missing_fib) else np.nan, fibroblast_high_state_fraction=float(block.fibroblast_high_count.sum() / n_fib) if n_fib and (not missing_fib) else np.nan, fibroblast_fraction=n_fib / n_all if n_all else np.nan)
        records.append(row)
    return pd.DataFrame(records)

def outcome_contrasts(patients, require_full=True):
    results = []
    for name, block in patients.groupby(['run', 'rate', 'seed', 'population'], dropna=False, sort=True):
        eligible = block.loc[block.arm.isin(DFU_ARMS)].sort_values('patient_id')
        healed = eligible.arm.eq('DFU-healer').to_numpy()
        if require_full and (len(eligible), int(healed.sum())) != (11, 7):
            raise ValueError('Outcome contrast no longer has the fixed 11 patients (7/4)')
        for readout in READOUTS:
            row = dict(zip(('run', 'rate', 'seed', 'population'), name))
            row.update(readout=readout, n_healed=int(healed.sum()), n_nonhealed=int((~healed).sum()), healthy_patients_excluded=int((~block.arm.isin(DFU_ARMS)).sum()))
            values = eligible[readout].to_numpy(float)
            missing = eligible.loc[~np.isfinite(values), 'patient_id'].tolist()
            row['missing_patients'] = '|'.join(missing)
            if missing or not healed.any() or healed.all():
                row.update(status='not_estimable', difference=np.nan, exact_p=np.nan, extreme_allocations=0, allocations=0)
            else:
                difference, p_value, total, null = _CONTROL.exact_contrast(values, healed)
                extreme = int(round(p_value * total))
                row.update(status='descriptive_sensitivity', difference=difference, healer_mean=float(values[healed].mean()), nonhealer_mean=float(values[~healed].mean()), exact_p=p_value, extreme_allocations=extreme, allocations=total)
            results.append(row)
    return pd.DataFrame(results)

def paired_stability(baseline, current, mask, threshold):
    before = baseline.topic_0.to_numpy(float)[mask]
    after = current.topic_0.to_numpy(float)[mask]
    finite = np.isfinite(before) & np.isfinite(after)
    result = {'n_selected': int(mask.sum()), 'n_paired': int(finite.sum()), 'n_missing_pairs': int((~finite).sum())}
    if not finite.any():
        return result
    before, after = (before[finite], after[finite])
    delta = after - before
    rho = float(spearmanr(before, after).statistic) if len(before) > 1 and np.ptp(before) > 0 and (np.ptp(after) > 0) else np.nan
    result.update(spearman=rho, mean_signed_change=float(delta.mean()), mean_absolute_change=float(np.abs(delta).mean()), rmse=float(np.sqrt(np.mean(delta ** 2))), absolute_change_p50=float(np.median(np.abs(delta))), absolute_change_p95=float(np.quantile(np.abs(delta), 0.95)), absolute_change_max=float(np.max(np.abs(delta))), historical_threshold_crossing_fraction=float(np.mean((before > threshold) != (after > threshold))))
    return result

def stability_records(baseline, current, tag, rate, seed, threshold, gsm=None):
    original = baseline.qc_pass.to_numpy(bool)
    current_keep = current.qc_pass.to_numpy(bool)
    fib = baseline.celltype.eq('fibroblast').to_numpy()
    masks = {'original_qc': original, 'qc_intersection': original & current_keep, 'original_fibroblast': original & fib, 'original_fibroblast_qc_intersection': original & current_keep & fib}
    return [{'run': tag, 'rate': rate, 'seed': seed, 'gsm': gsm, 'conditional_population': key, **paired_stability(baseline, current, mask, threshold)} for key, mask in masks.items()]

def transition_record(baseline, current, tag, rate, seed):
    original = baseline.qc_pass.to_numpy(bool)
    keep = current.qc_pass.to_numpy(bool)
    both = original & keep
    old_gate = baseline.celltype.to_numpy(str)
    new_gate = current.celltype.to_numpy(str)
    old_fib, new_fib = (old_gate == 'fibroblast', new_gate == 'fibroblast')
    n_raw, n_orig = (len(baseline), int(original.sum()))
    n_both = int(both.sum())
    margins = current.loc[keep, 'marker_top_two_margin'].to_numpy()
    record = {'run': tag, 'rate': rate, 'seed': seed, **{key: str(current[key].iloc[0]) for key in ('gsm', 'arm', 'patient_id')}, 'n_raw': n_raw, 'original_qc': n_orig, 'current_qc': int(keep.sum()), 'common_qc': n_both, 'lost_qc': int((original & ~keep).sum()), 'gained_qc': int((~original & keep).sum()), 'current_qc_fraction_of_raw': float(keep.mean()), 'original_qc_retention_fraction': n_both / n_orig if n_orig else np.nan, 'original_fibroblasts': int((original & old_fib).sum()), 'original_fibroblasts_lost_qc': int((original & old_fib & ~keep).sum()), 'current_fibroblasts': int((keep & new_fib).sum()), 'common_qc_lineage_changed': int((both & (old_gate != new_gate)).sum()), 'common_qc_fibroblast_exit': int((both & old_fib & ~new_fib).sum()), 'common_qc_fibroblast_entry': int((both & ~old_fib & new_fib).sum()), 'current_qc_zero_marker_cells': int((keep & current.detected_marker_genes.eq(0).to_numpy()).sum()), 'current_qc_exact_marker_ties': int((keep & current.marker_top_two_margin.eq(0).to_numpy()).sum()), 'current_qc_missing_score': int((keep & ~current.score_available.to_numpy(bool)).sum()), 'marker_margin_p05': float(np.quantile(margins, 0.05)) if len(margins) else np.nan, 'marker_margin_median': float(np.median(margins)) if len(margins) else np.nan, 'raw_count_total': int(baseline.library_size.sum()), 'thinned_count_total': int(current.library_size.sum())}
    confusion = pd.crosstab(pd.Series(old_gate[both], name='original_lineage'), pd.Series(new_gate[both], name='current_lineage')).stack()
    rows = [{'run': tag, 'rate': rate, 'seed': seed, 'gsm': record['gsm'], 'original_lineage': a, 'current_lineage': b, 'cells': int(count)} for (a, b), count in confusion.items() if count]
    return (record, rows)

def baseline_patient_verification(patients, expected_path, tolerance):
    actual = patients.loc[patients.population.eq('full_pipeline')].set_index('patient_id')
    expected = pd.read_csv(expected_path).set_index('patient_id')
    if set(actual.index) != set(expected.index):
        raise ValueError('Baseline patient membership differs from unified historical projection')
    delta = float(np.max(np.abs(actual.loc[expected.index, 'all_cell_topic0_mean'].to_numpy() - expected.score_cell_weighted.to_numpy())))
    if delta > tolerance:
        raise ValueError(f'Baseline patient scores differ: {delta}')
    return {'patients_checked': len(actual), 'all_cell_patient_max_abs_difference': delta, 'all_cell_patient_aggregation_reproduced': True}

def run_tag(rate, seed):
    return 'baseline' if rate == 1 else f'p{int(round(rate * 100)):03d}_s{seed}'

def build_summary(transitions, contrasts, stability, baseline_check, mode):
    count_fields = ['n_raw', 'original_qc', 'current_qc', 'common_qc', 'lost_qc', 'gained_qc', 'original_fibroblasts', 'original_fibroblasts_lost_qc', 'current_fibroblasts', 'common_qc_lineage_changed', 'common_qc_fibroblast_exit', 'common_qc_fibroblast_entry', 'current_qc_zero_marker_cells', 'current_qc_exact_marker_ties', 'current_qc_missing_score', 'raw_count_total', 'thinned_count_total']
    pooled = transitions.groupby(['run', 'rate', 'seed'], dropna=False, sort=True)[count_fields].sum().reset_index()
    pooled['original_qc_retention_fraction'] = pooled.common_qc / pooled.original_qc
    pooled['common_qc_lineage_change_fraction'] = pooled.common_qc_lineage_changed / pooled.common_qc
    pooled['realized_count_fraction'] = pooled.thinned_count_total / pooled.raw_count_total
    chosen = contrasts.loc[contrasts.population.eq('full_pipeline')].copy()
    ranges = []
    for (rate, readout), block in chosen.groupby(['rate', 'readout'], sort=False):
        valid = block.loc[block.status.eq('descriptive_sensitivity')]
        ranges.append({'rate': float(rate), 'readout': readout, 'n_runs': len(block), 'n_estimable': len(valid), 'difference_min': float(valid.difference.min()) if len(valid) else None, 'difference_max': float(valid.difference.max()) if len(valid) else None, 'exact_p_min': float(valid.exact_p.min()) if len(valid) else None, 'exact_p_max': float(valid.exact_p.max()) if len(valid) else None})
    return {'status': 'completed', 'mode': mode, 'scope': 'Full released pre-analysis-QC count cohort' if mode == 'full' else 'REAL-DATA SUBSET SMOKE ONLY', 'baseline_reproduction': baseline_check, 'pooled_qc_gate_by_run': json.loads(pooled.to_json(orient='records', double_precision=15)), 'full_pipeline_outcome_seed_ranges': ranges, 'conditional_stability': json.loads(stability.to_json(orient='records', double_precision=15)), 'concern_status': 'Computational whole-count-pipeline test executed; broader RT06 robustness/validation only partially addressed, not closed.', 'limitations': ['A thinning stress test does not establish marker truth or portability.', 'Conditional stability excludes or freezes selection effects; inspect the full-pipeline branch separately.', 'The reference, checkpoint, cohort, and high-state cutoff are discovery-derived, not external validation.', 'Three seeds add Monte Carlo checks, not patients or biological replicates.', 'No clinical acceptance or equivalence margin is specified or tested.', 'Uniform count thinning omits upstream cell calling, ambient/doublet uncertainty, cell-type-specific capture and biological repeat sampling.']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='full')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--runtime-log', type=Path)
    args = parser.parse_args()
    started = time.time()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs = {key: ROOT / value for key, value in protocol['inputs'].items()}
    for value in inputs.values():
        if not value.is_file():
            raise FileNotFoundError(f'Missing required input: {value}')
    provenance = runtime_provenance(args.runtime_log, protocol['requested_agent_model'])
    output = fresh_directory(args.output_dir)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'run.log')])
    source_files = [Path(__file__), args.protocol, ROOT / 'scripts/repair_measurement_controls.py', ROOT / 'scripts/repair_measurement_projection.py', ROOT / 'wound_models/data_adapter.py', ROOT / 'wound_models/celltype_markers.py', ROOT / 'wound_models/topic_model.py']
    input_hashes = {key: {'path': str(value.relative_to(ROOT)), 'sha256': sha256(value), 'bytes': value.stat().st_size} for key, value in inputs.items()}
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    frozen = {'frozen_at_utc': utc_now(), 'protocol': protocol, 'protocol_source_sha256': sha256(args.protocol), 'mode': args.mode, 'code_sha256': code_hashes, 'inputs': input_hashes}
    write_json(output / 'protocol_frozen.json', frozen)
    contract = {'started_at_utc': utc_now(), 'status': 'started', 'mode': args.mode, 'argv': sys.argv, 'protocol_frozen_sha256': sha256(output / 'protocol_frozen.json'), 'agent_model': provenance, 'device': args.device, 'torch_threads': args.threads, 'packages': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__}, 'new_dependencies_installed': False, 'model_training': False, 'historical_outputs_writable': False}
    write_json(output / 'run_contract.json', contract)
    LOG.info('Protocol frozen before raw reconstruction or perturbation: %s', frozen['protocol_source_sha256'])
    obs = pd.read_csv(inputs['obs'])
    expected = pd.read_parquet(inputs['projection'])
    historical_raw = pd.read_csv(inputs['historical_raw_identity'])
    if len(obs) != len(expected) or len(obs) != len(historical_raw):
        raise ValueError('Historical projection and raw identity row counts disagree')
    for left, right in ((obs.gsm, expected.gsm), (obs.disease, expected.arm), (historical_raw.gsm, expected.gsm), (historical_raw.arm, expected.arm)):
        if not np.array_equal(left.to_numpy(str), right.to_numpy(str)):
            raise ValueError('Historical source orders disagree')
    wanted = set(obs.gsm)
    phenotype = parse_series_matrix(str(inputs['series'])).set_index('gsm')
    eligible = phenotype.loc[phenotype.tissue.str.lower().eq('foot skin') & phenotype.disease.isin(protocol['cohort']['arms'])]
    if set(eligible.index) != wanted or len(wanted) != protocol['cohort']['specimens']:
        raise ValueError('The frozen 25-specimen discovery cohort is not reproduced')
    mapping = pd.read_csv(inputs['patient_map'])
    if mapping.sample_id.duplicated().any():
        raise ValueError('Duplicate author-mapped sample IDs')
    mapping = mapping.set_index('sample_id')
    for gsm in wanted:
        if gsm not in mapping.index or pd.isna(mapping.loc[gsm, 'patient_id']):
            raise ValueError(f'Missing patient identity: {gsm}')
        if mapping.loc[gsm, 'geo_disease'] != phenotype.loc[gsm, 'disease']:
            raise ValueError(f'Patient-map phenotype disagrees with raw metadata: {gsm}')
    panel = np.load(inputs['panel'], allow_pickle=True).astype(str)
    beta = np.load(inputs['beta'], allow_pickle=False)
    if len(panel) != 7002 or len(set(panel)) != len(panel) or beta.shape != (15, 7002):
        raise ValueError('Frozen panel/model dimensions differ from protocol')
    state = torch.load(inputs['model'], map_location='cpu', weights_only=True)
    hidden_dim = int(state['encoder.net.0.weight'].shape[0])
    model = TopicModel(7002, num_topics=15, hidden_dim=hidden_dim).to(args.device)
    model.load_state_dict(state)
    model.eval()
    checkpoint_beta64 = torch.softmax(state['decoder.beta_raw'].double(), dim=-1).numpy()
    if not np.allclose(checkpoint_beta64, beta, rtol=0, atol=1e-07):
        raise ValueError('Frozen checkpoint decoder differs from saved beta')
    reference = load_marker_reference(inputs['marker_reference'])
    threshold = json.loads(inputs['mixture_threshold'].read_text())['cohorts']['GSE165816_discovery']['gmm']['midpoint']
    cache = output / 'raw_counts'
    cache.mkdir()
    baseline_dir = output / 'cells/baseline'
    baseline_dir.mkdir(parents=True)
    sample_records, identities, baseline_checks = ([], [], [])
    specimen_records, transition_records, confusions = ([], [], [])
    stability_by_sample, stability_by_run = ([], [])
    tolerance = protocol['score']['baseline_coordinate_absolute_tolerance']
    batch_size = protocol['compute']['encoder_batch_size']
    with tarfile.open(inputs['raw_tar'], 'r') as archive:
        selected = []
        for member in sorted(archive.getmembers(), key=lambda member: member.name):
            match = re.search('GSM\\d+', member.name)
            if member.isfile() and member.name.endswith(('.csv', '.csv.gz')) and match and (match.group() in wanted):
                selected.append((match.group(), member))
        if len(selected) != len(wanted) or {gsm for gsm, _ in selected} != wanted:
            raise ValueError('Missing/duplicated selected raw specimen members')
        if args.mode == 'smoke':
            selected = selected[:protocol['smoke']['first_sorted_specimens']]
        for gsm, member in selected:
            LOG.info('Raw reconstruction and baseline: %s', gsm)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'Unreadable archive member {member.name}')
            payload = stream.read()
            counts, genes, barcodes, full_columns = read_count_member(payload, member.name.endswith('.gz'), protocol['compute']['raw_csv_gene_chunk_rows'], protocol['smoke']['first_raw_cells_per_specimen'] if args.mode == 'smoke' else None)
            frame, coverage = evaluate_counts(counts, genes, panel, reference, model, args.device, batch_size)
            frame = attach_metadata(frame, gsm, str(phenotype.loc[gsm, 'disease']), str(mapping.loc[gsm, 'patient_id']), barcodes)
            hist_mask = historical_raw.gsm.eq(gsm).to_numpy()
            if args.mode == 'smoke':
                hist_mask &= historical_raw.raw_row_0.to_numpy() < len(barcodes)
            check = verify_baseline(frame, expected.loc[hist_mask], historical_raw.loc[hist_mask], tolerance)
            baseline_checks.append(check)
            sp.save_npz(cache / f'{gsm}.npz', counts, compressed=True)
            np.save(cache / f'{gsm}_genes.npy', genes, allow_pickle=False)
            frame.to_parquet(baseline_dir / f'{gsm}.parquet', index=False)
            identity = frame[['gsm', 'arm', 'patient_id', 'raw_cell_id', 'raw_row_0', 'qc_pass', 'detected_genes', 'library_size', 'mito_fraction', 'celltype']].copy()
            identity['raw_member'] = member.name
            identity['raw_barcode'] = barcodes
            identities.append(identity)
            record = {'gsm': gsm, 'raw_member': member.name, 'raw_member_compressed_sha256': hashlib.sha256(payload).hexdigest(), 'source_raw_cell_columns': full_columns, 'evaluated_raw_cells': len(frame), 'source_genes': len(genes), 'raw_nnz': int(counts.nnz), 'source_genes_sha256': text_array_hash(genes), 'evaluated_barcode_order_sha256': text_array_hash(barcodes), 'canonical_sparse_counts_sha256': sparse_hash(counts), 'cached_counts_sha256': sha256(cache / f'{gsm}.npz'), 'cached_genes_sha256': sha256(cache / f'{gsm}_genes.npy'), 'baseline_cells_sha256': sha256(baseline_dir / f'{gsm}.parquet'), 'panel_genes_present_in_source': coverage, 'panel_genes_zero_filled': len(panel) - coverage}
            sample_records.append(record)
            specimen_records.extend(specimen_readouts(frame, frame, 'baseline', 1.0, -1, threshold))
            transition, confusion = transition_record(frame, frame, 'baseline', 1.0, -1)
            transition_records.append(transition)
            confusions.extend(confusion)
            stability_by_sample.extend(stability_records(frame, frame, 'baseline', 1.0, -1, threshold, gsm))
            LOG.info('Baseline reproduced %s: raw=%d QC=%d fibroblast=%d max|theta error|=%.3g', gsm, check['raw_cells'], check['qc_cells'], check['fibroblasts'], check['coordinate_max_abs_difference'])
    identity = pd.concat(identities, ignore_index=True)
    identity.insert(0, 'cohort_raw_row_0', np.arange(len(identity)))
    if identity.raw_cell_id.duplicated().any():
        raise ValueError('Global raw GSM:barcode identity is not unique')
    identity.to_csv(output / 'raw_cell_identity.csv.gz', index=False, compression='gzip')
    check = {'all_selected_members_read': True, 'raw_cells': len(identity), 'qc_cells': int(identity.qc_pass.sum()), 'fibroblasts': int((identity.qc_pass & identity.celltype.eq('fibroblast')).sum()), 'specimens': len(sample_records), 'per_specimen': baseline_checks, 'coordinate_tolerance': tolerance, 'maximum_coordinate_absolute_difference': max((row['coordinate_max_abs_difference'] for row in baseline_checks))}
    if args.mode == 'full':
        cohort = protocol['cohort']
        if (check['raw_cells'], check['qc_cells'], check['fibroblasts'], check['specimens']) != (cohort['expected_pre_qc_cells'], cohort['expected_baseline_qc_cells'], cohort['expected_baseline_fibroblasts'], cohort['specimens']):
            raise ValueError(f'Full baseline cohort count contract failed: {check}')
        check.update(baseline_patient_verification(aggregate_patients(pd.DataFrame(specimen_records)), inputs['historical_patient_scores'], tolerance))
    write_json(output / 'baseline_reproduction.json', check)
    write_json(output / 'raw_input_manifest.json', {'mode': args.mode, 'inputs': input_hashes, 'samples': sample_records, 'raw_cell_identity_sha256': sha256(output / 'raw_cell_identity.csv.gz'), 'identity': 'GSM + original CSV barcode; every evaluated raw cell including original QC failures is recorded.', 'not_the_old_qc_cache': True, 'raw_scope': protocol['cohort']['raw_scope']})
    LOG.info('All baseline checks passed before thinning: %s raw cells / %s QC survivors', check['raw_cells'], check['qc_cells'])
    gsm_order = [row['gsm'] for row in sample_records]
    baseline_all = pd.concat([pd.read_parquet(baseline_dir / f'{gsm}.parquet') for gsm in gsm_order], ignore_index=True)
    stability_by_run.extend(stability_records(baseline_all, baseline_all, 'baseline', 1.0, -1, threshold))
    grid = [(rate, seed) for rate in protocol['perturbation']['retention_rates'] for seed in protocol['perturbation']['seeds']]
    if args.mode == 'smoke':
        grid = [(protocol['smoke']['rate'], protocol['smoke']['seed'])]
    for rate, seed in grid:
        tag = run_tag(rate, seed)
        run_dir = output / 'cells' / tag
        run_dir.mkdir()
        current_blocks = []
        LOG.info('Starting %s (%d complete raw specimens)', tag, len(gsm_order))
        for record in sample_records:
            gsm = record['gsm']
            if sha256(cache / f'{gsm}.npz') != record['cached_counts_sha256']:
                raise ValueError(f'Raw count cache changed: {gsm}')
            counts = sp.load_npz(cache / f'{gsm}.npz')
            genes = np.load(cache / f'{gsm}_genes.npy', allow_pickle=False)
            original = pd.read_parquet(baseline_dir / f'{gsm}.parquet')
            thinned = thin_counts(counts, rate, seed, gsm)
            current, _ = evaluate_counts(thinned, genes, panel, reference, model, args.device, batch_size)
            for key in ('gsm', 'arm', 'patient_id', 'raw_cell_id'):
                current[key] = original[key].to_numpy()
            current.to_parquet(run_dir / f'{gsm}.parquet', index=False)
            current_blocks.append(current)
            specimen_records.extend(specimen_readouts(original, current, tag, rate, seed, threshold))
            transition, confusion = transition_record(original, current, tag, rate, seed)
            transition_records.append(transition)
            confusions.extend(confusion)
            stability_by_sample.extend(stability_records(original, current, tag, rate, seed, threshold, gsm))
        current_all = pd.concat(current_blocks, ignore_index=True)
        if not np.array_equal(baseline_all.raw_cell_id, current_all.raw_cell_id):
            raise ValueError('Perturbed raw-cell identity/order changed')
        stability_by_run.extend(stability_records(baseline_all, current_all, tag, rate, seed, threshold))
        LOG.info('Finished %s: QC=%d/%d; raw cells evaluated=%d', tag, int(current_all.qc_pass.sum()), check['qc_cells'], len(current_all))
    specimens = pd.DataFrame(specimen_records)
    patients = aggregate_patients(specimens)
    contrasts = outcome_contrasts(patients, require_full=args.mode == 'full')
    transitions = pd.DataFrame(transition_records)
    stability = pd.DataFrame(stability_by_run)
    outputs = {'specimen_readouts.csv': specimens, 'patient_readouts.csv': patients, 'patient_exact_contrasts.csv': contrasts, 'qc_gate_transitions_by_specimen.csv': transitions, 'gate_confusions.csv': pd.DataFrame(confusions), 'conditional_stability_by_specimen.csv': pd.DataFrame(stability_by_sample), 'conditional_stability_by_run.csv': stability}
    for name, table in outputs.items():
        table.to_csv(output / name, index=False)
    if len(transitions) != len(gsm_order) * (len(grid) + 1):
        raise ValueError('Incomplete specimen x perturbation grid')
    if transitions.groupby('run').n_raw.sum().nunique() != 1:
        raise ValueError('Different perturbations evaluated different raw cohorts')
    for key, info in input_hashes.items():
        if sha256(inputs[key]) != info['sha256']:
            raise ValueError(f'An input changed during the experiment: {key}')
    for relative, before in code_hashes.items():
        if sha256(ROOT / relative) != before:
            raise ValueError(f'Analysis source changed during the experiment: {relative}')
    summary = build_summary(transitions, contrasts, stability, check, args.mode)
    summary.update(completed_at_utc=utc_now(), elapsed_seconds=time.time() - started, all_input_and_code_hashes_unchanged=True, protocol_frozen_sha256=contract['protocol_frozen_sha256'], frozen_high_state_threshold=threshold, agent_model=provenance)
    write_json(output / 'report.json', summary)
    hashes = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*')) if path.is_file() and path.name not in {'run.log', 'run_contract.json', 'artifact_manifest.json'}}
    write_json(output / 'artifact_manifest.json', {'files_sha256': hashes, 'created_at_utc': utc_now()})
    contract.update(status='completed', completed_at_utc=utc_now(), elapsed_seconds=time.time() - started, artifacts_manifest_sha256=sha256(output / 'artifact_manifest.json'))
    write_json(output / 'run_contract.json', contract)
    LOG.info('Completed %s in %.1f seconds; report: %s', args.mode, time.time() - started, output / 'report.json')
if __name__ == '__main__':
    main()
