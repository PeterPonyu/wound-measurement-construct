#!/usr/bin/env python3
"""Verify saved-input mathematical estimands."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT = Path(__file__).resolve().parents[1]
CURRENT_REVISION = ROOT / 'outputs/scientific_revision_20260922/measurement'

def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def producing_fingerprint(path, root=None):
    """Return a validated native producing hash, without relabelling reruns."""
    root = ROOT if root is None else Path(root)
    path = Path(path)
    if not path.is_file() or any((p.is_symlink() for p in (path, *path.parents))):
        raise ValueError(f'Invalid revision input: {path}')
    actual = fingerprint(path)
    producing = actual
    provenance = root / 'EXPORT_PROVENANCE.json'
    if provenance.is_file():
        records = json.loads(provenance.read_text())['files']
        record = next((row for row in records if row['exported'] == path.relative_to(root).as_posix()), None)
        if record is None or actual != record['exported_sha256']:
            raise ValueError(f'Export provenance mismatch: {path}')
        if record.get('original'):
            original = root / record['original']
            if not original.resolve().is_relative_to(root.resolve()):
                raise ValueError('Original provenance path escapes the product')
            producing = fingerprint(original)
            if producing != record['source_sha256']:
                raise ValueError(f'Producing snapshot mismatch: {path}')
    return producing

def verified_revision_file(root, revision, relative, report):
    """Bind current files to their producing hashes, including standalone exports."""
    path = revision / relative
    producing = producing_fingerprint(path, root)
    if producing != report.get('output_sha256', {}).get(relative):
        raise ValueError(f'Current revision checksum mismatch: {relative}')
    return (path, fingerprint(path))

def load_current_measurement(revision=None, root=None):
    """Validate deterministic cells and recompute patient aggregation before audit."""
    root = ROOT if root is None else Path(root)
    revision = CURRENT_REVISION if revision is None else Path(revision)
    report_path = revision / 'report.json'
    report = json.loads(report_path.read_text())
    identity = report.get('source_identity', {})
    if report.get('status') != 'completed' or identity.get('status') != 'accepted' or 'softmax(mu)' not in identity.get('coordinate_definition', '') or (not all((identity.get('order_contract', {}).get(key) is True for key in ('obs_gsm_equal', 'obs_arm_equal', 'unique_cell_ids')))):
        raise ValueError('Current deterministic projection contract is absent or invalid')
    sources = {str(report_path.relative_to(root)): fingerprint(report_path)}
    paths = {}
    for relative in ('projection/cells.parquet', 'patient_unit_remap/patient_scores.csv', 'patient_unit_remap/report.json'):
        paths[relative], digest = verified_revision_file(root, revision, relative, report)
        sources[str(paths[relative].relative_to(root))] = digest
    cells = pd.read_parquet(paths['projection/cells.parquet'])
    patients = pd.read_csv(paths['patient_unit_remap/patient_scores.csv'])
    patient_report = json.loads(paths['patient_unit_remap/report.json'].read_text())
    required = {'patient_id', 'healing_status', 'n_samples', 'n_cells', 'sample_ids', 'score_cell_weighted', 'score_sample_mean'}
    if not required <= set(patients) or not patients.patient_id.is_unique or patients[list(required)].isna().any().any():
        raise ValueError('Missing or duplicate current patient data')
    if len(patients) != 11 or patients.healing_status.value_counts().to_dict() != {'healed': 7, 'not_healed': 4} or patient_report.get('patient_counts', {}).get('dfu') != 11 or (patient_report.get('protocol', {}).get('deterministic_projection') is not True):
        raise ValueError('Current patient label/count contract mismatch')
    values = patients[['score_cell_weighted', 'score_sample_mean', 'n_cells', 'n_samples']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, :2] > 1).any():
        raise ValueError('Nonfinite or out-of-domain patient scores')
    if np.any(values[:, 2:] < 1) or not np.equal(values[:, 2:], np.floor(values[:, 2:])).all() or int(patients.n_samples.sum()) != 14:
        raise ValueError('Invalid patient specimen/cell counts')
    if len(cells) != 76461 or cells.gsm.nunique() != 25 or cells.patient_id.nunique() != 20 or (not cells.cell_id.is_unique) or (not np.array_equal(cells.source_row_zero_based, np.arange(len(cells)))):
        raise ValueError('Current cell identity/order/count mismatch')
    theta = cells[[f'topic_{i}' for i in range(15)]].to_numpy(float)
    if not np.isfinite(theta).all() or (theta < 0).any() or (not np.allclose(theta.sum(1), 1, rtol=0, atol=2e-06)):
        raise ValueError('Invalid current simplex coordinates')
    if (cells.groupby('patient_id').arm.nunique() > 1).any():
        raise ValueError('Incompatible patient arms')
    dfu = cells[cells.healing_status.isin(['healed', 'not_healed'])].copy()
    dfu['score'] = dfu.topic_0.to_numpy(np.float64)
    if set(dfu.patient_id) != set(patients.patient_id):
        raise ValueError('Current patient coverage mismatch')
    for row in patients.itertuples():
        block = dfu[dfu.patient_id.eq(row.patient_id)]
        specimen_means = block.groupby('gsm').score.mean()
        if set(block.healing_status) != {row.healing_status} or len(block) != row.n_cells or len(specimen_means) != row.n_samples or (set(str(row.sample_ids).split('|')) != set(block.gsm)) or (not np.isclose(block.topic_0.to_numpy(np.float64).mean(), row.score_cell_weighted, rtol=0, atol=1e-14)) or (not np.isclose(specimen_means.mean(), row.score_sample_mean, rtol=0, atol=1e-14)):
            raise ValueError(f'Patient aggregation mismatch: {row.patient_id}')
    return (cells, patients, {'projection_version': 'deterministic softmax(mu)', 'source_fingerprints': sources, 'aggregation_recomputed_from_cells': True, 'n_patients': 11, 'n_healed': 7, 'n_not_healed': 4, 'n_specimens': 14, 'no_model_training': True, 'no_new_biological_units': True})

def state_summary(values, cut, low_reference, high_reference):
    values = np.asarray(values, dtype=np.float64)
    high = values > cut
    weight = float(high.mean())
    low_mean = float(values[~high].mean()) if (~high).any() else None
    high_mean = float(values[high].mean()) if high.any() else None
    low_term = (1 - weight) * low_mean if low_mean is not None else 0.0
    high_term = weight * high_mean if high_mean is not None else 0.0
    mean = float(values.mean())
    fixed = (1 - weight) * low_reference + weight * high_reference
    residual = low_term - (1 - weight) * low_reference + (high_term - weight * high_reference)
    return dict(n_cells=len(values), n_high=int(high.sum()), mean_loading=mean, high_state_fraction=weight, low_state_mean=low_mean, high_state_mean=high_mean, fixed_state_prediction=fixed, within_state_residual=residual, identity_error=abs(mean - low_term - high_term), residual_identity_error=abs(mean - fixed - residual))

def exhaustive_contrast(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    observed = float(values[labels].mean() - values[~labels].mean())
    contrasts = []
    for chosen in itertools.combinations(range(len(values)), int(labels.sum())):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(chosen)] = True
        contrasts.append(float(values[mask].mean() - values[~mask].mean()))
    extreme = int(np.count_nonzero(np.abs(contrasts) >= abs(observed) - 1e-12))
    return dict(observed_difference=observed, allocations=len(contrasts), extreme_allocations=extreme, exact_probability=extreme / len(contrasts), recorded_add_one_probability=(extreme + 1) / (len(contrasts) + 1))

def equivalence_boundary(a, b, alpha=0.05):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    va, vb = (a.var(ddof=1) / len(a), b.var(ddof=1) / len(b))
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    delta = float(a.mean() - b.mean())
    half_width = float(t.ppf(1 - alpha, df) * se)
    return dict(standard_error=float(se), welch_df=float(df), two_sided_90_interval=[delta - half_width, delta + half_width], equivalence_margin_infimum=abs(delta) + half_width)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/mathematical_audit/measurement')
    ap.add_argument('--revision-dir', type=Path, default=CURRENT_REVISION)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory; reference reports are read only')
    inputs = {'cell_projection': ROOT / 'outputs/celltype_resolved/GSE165816_discovery_theta.parquet', 'mixture_fit': ROOT / 'outputs/bimodal_stratification/report.json'}
    cells, patients, binding = load_current_measurement(args.revision_dir)
    revision = json.loads((args.revision_dir / 'report.json').read_text())
    for path in inputs.values():
        if producing_fingerprint(path) != revision['source_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError(f'Projection/threshold source checksum mismatch: {path}')
    cells = cells[cells.celltype.eq('fibroblast')]
    mixture = json.loads(inputs['mixture_fit'].read_text())['cohorts']['GSE165816_discovery']
    cut = mixture['gmm']['midpoint']
    loading = cells.topic_0.to_numpy(dtype=np.float64)
    low_reference = float(loading[loading <= cut].mean())
    high_reference = float(loading[loading > cut].mean())
    rows = [dict(specimen=str(gsm), **state_summary(block.topic_0, cut, low_reference, high_reference)) for gsm, block in cells.groupby('gsm', observed=True, sort=True)]
    assert len(rows) == 25 and len(cells) == 19410
    assert max((row['identity_error'] for row in rows)) < 1e-12
    assert max((row['residual_identity_error'] for row in rows)) < 1e-12
    labels = patients.healing_status.eq('healed').to_numpy()
    scores = patients.score_cell_weighted.to_numpy()
    assert len(scores) == 11 and labels.sum() == 7
    ranges = {key: [min((row[key] for row in rows if row[key] is not None)), max((row[key] for row in rows if row[key] is not None))] for key in ('low_state_mean', 'high_state_mean', 'within_state_residual')}
    report = {'status': 'completed', 'scope': 'Frozen-score algebra and conditional patient-label inference; no new biological replication or causal identification.', 'state_decomposition': {'n_specimens': len(rows), 'n_fibroblasts': len(cells), 'cut': cut, 'reference': 'Cell-pooled means of the two threshold bins; not fitted Gaussian means', 'low_reference': low_reference, 'high_reference': high_reference, 'empty_bin_specimens': sum((row['low_state_mean'] is None or row['high_state_mean'] is None for row in rows)), 'ranges': ranges, 'specimen_weighted_residual_rms': float(np.sqrt(np.mean([row['within_state_residual'] ** 2 for row in rows]))), 'max_identity_error': max((row['identity_error'] for row in rows)), 'max_residual_identity_error': max((row['residual_identity_error'] for row in rows)), 'specimens': rows}, 'patient_permutation': exhaustive_contrast(scores, labels), 'patient_equivalence': equivalence_boundary(scores[labels], scores[~labels]), 'current_source_binding': binding, 'protocol': {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source_sha256': {str(path.relative_to(ROOT)): fingerprint(path) for path in inputs.values()}, 'computation_dtype': 'float64', 'label_exchangeability': 'assumed, not supplied by randomization'}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output_dir / 'state_decomposition.csv', index=False)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'state_decomposition'}, indent=2))
    print(json.dumps({key: value for key, value in report['state_decomposition'].items() if key != 'specimens'}, indent=2))
if __name__ == '__main__':
    main()
