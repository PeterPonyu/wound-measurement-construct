#!/usr/bin/env python3
"""Infer patient outcome contrast."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
_EQ_PATH = os.path.join(ROOT, 'scripts', 'assess_sample_contrast_equivalence.py')
_SPEC = importlib.util.spec_from_file_location('patient_mapping_equivalence', _EQ_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('cannot load scripts/assess_sample_contrast_equivalence.py')
_EQ = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EQ)
_exact_p = _EQ._exact_p
_bootstrap_ci = _EQ._bootstrap_ci
_tost = _EQ._tost
MAP_CSV = os.path.join(ROOT, 'cohort_metadata', 'GSE165816_ncomms_subject_sample_map.csv')
CONTRACT_REPORT = os.path.join(ROOT, 'outputs', 'cohort_metadata', 'gse165816_author_map_contract.json')
DISCOVERY_DIR = os.path.join(ROOT, 'outputs', 'expression_representation')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'cohort_metadata', 'gse165816_patient_unit_remap')
SAMPLE_LEVEL_DIFF = 0.04346680645313528

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _smallest_equivalent_margin(a: np.ndarray, b: np.ndarray, diff: float) -> dict:
    lo, hi = (1e-06, 5.0)
    if not _tost(a, b, hi)['equivalent_at_alpha_0_05']:
        return {'margin': None, 'note': 'no margin up to 5.0 reaches equivalence'}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _tost(a, b, mid)['equivalent_at_alpha_0_05']:
            hi = mid
        else:
            lo = mid
    return {'margin': float(hi), 'multiple_of_observed_difference': float(hi / abs(diff)) if diff != 0 else None, 'note': 'smallest absolute mean-difference margin this patient-level sample can declare equivalent at alpha=0.05; a clinically meaningful margin below this value cannot be ruled in or out here'}

def _load_accepted_contract(path: str) -> dict:
    """Require the author-map contract before restating patient-level scores.

    The patient remap is downstream of metadata intake.  It must not silently
    turn a partially populated map into a patient-level result, so the
    collapsed contract's accepted status is checked from the machine-readable
    report produced by script 34.
    """
    try:
        with open(path, encoding='utf-8') as handle:
            contract = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'cannot read author-map contract: {exc}') from exc
    sample = contract.get('sample_level') or {}
    collapsed = contract.get('patient_collapsed') or {}
    if contract.get('status') != 'accepted':
        raise RuntimeError(f"author-map contract is not accepted: {contract.get('status')!r}")
    if sample.get('status') != 'rejected':
        raise RuntimeError('sample-level contract unexpectedly passed')
    if not any(('duplicate patient_id + wound_id + timepoint_days' in error for error in sample.get('errors', []))):
        raise RuntimeError('sample-level contract no longer records replicate rows')
    if collapsed.get('status') != 'accepted' or collapsed.get('errors'):
        raise RuntimeError(f"patient-collapsed contract is not clean: status={collapsed.get('status')!r}, errors={collapsed.get('errors')!r}")
    for name, block in (('sample_level', sample), ('patient_collapsed', collapsed)):
        protocol = block.get('protocol') or {}
        if protocol.get('does_not_open_expression_matrix') is not True:
            raise RuntimeError(f'{name} contract opened an expression matrix')
        if protocol.get('not_an_independent_cohort') is not True:
            raise RuntimeError(f'{name} contract lost its discovery-cohort scope')
    return contract

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-csv', default=MAP_CSV)
    parser.add_argument('--contract-report', default=CONTRACT_REPORT)
    parser.add_argument('--discovery-dir', default=DISCOVERY_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--focus-topic', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    contract = _load_accepted_contract(args.contract_report)
    mapped = pd.read_csv(args.map_csv, dtype=str)
    foot = mapped[(mapped['tissue_site'] == 'foot') & mapped['healing_status'].isin(['healed', 'not_healed'])].copy()
    if len(foot) != 14:
        raise RuntimeError(f'expected 14 DFU foot samples in the map, found {len(foot)}')
    if foot['patient_id'].nunique() != 11:
        raise RuntimeError('author map no longer has 11 DFU foot patients')
    if foot['batch'].map(lambda x: str(x).strip()).eq('').any():
        raise RuntimeError('map still has blank batch; rebuild with scripts/build_discovery_patient_map.py')
    if foot['sequencing_run'].map(lambda x: str(x).strip()).eq('').any():
        raise RuntimeError('map still has blank sequencing_run; rebuild with scripts/build_discovery_patient_map.py')
    discovery = Path(args.discovery_dir)
    theta = np.load(discovery / 'theta.npy')
    obs = pd.read_csv(discovery / 'obs.csv')
    if len(theta) != len(obs):
        raise RuntimeError('theta/obs row count mismatch')
    if args.focus_topic < 0 or args.focus_topic >= theta.shape[1]:
        raise ValueError(f'focus topic {args.focus_topic} outside 0..{theta.shape[1] - 1}')
    cell = obs[['gsm']].copy()
    cell['score'] = theta[:, args.focus_topic]
    gsm_to_patient = foot.set_index('sample_id')
    missing = sorted(set(foot['sample_id']) - set(cell['gsm'].astype(str)))
    if missing:
        raise RuntimeError('discovery projection missing mapped GSMs: ' + ', '.join(missing))
    cell = cell[cell['gsm'].astype(str).isin(set(foot['sample_id']))].copy()
    cell['patient_id'] = cell['gsm'].map(gsm_to_patient['patient_id'])
    cell['healing_status'] = cell['gsm'].map(gsm_to_patient['healing_status'])
    if cell['patient_id'].isna().any():
        raise RuntimeError('unmapped GSM reached the patient join')
    rows: list[dict[str, object]] = []
    for patient_id, block in cell.groupby('patient_id', sort=True):
        samples = foot[foot['patient_id'] == patient_id].sort_values('sample_title')
        sample_means = block.groupby('gsm', observed=True)['score'].mean().to_numpy(float)
        rows.append({'patient_id': patient_id, 'healing_status': block['healing_status'].iloc[0], 'n_samples': int(samples['sample_id'].nunique()), 'n_cells': int(len(block)), 'sample_ids': '|'.join(samples['sample_id'].tolist()), 'sample_titles': '|'.join(samples['sample_title'].tolist()), 'score_cell_weighted': float(block['score'].mean()), 'score_sample_mean': float(sample_means.mean())})
    patients = pd.DataFrame(rows)
    if len(patients) != 11:
        raise RuntimeError(f'expected 11 patient scores, found {len(patients)}')
    healed = patients.loc[patients['healing_status'] == 'healed', 'score_cell_weighted'].to_numpy(float)
    not_healed = patients.loc[patients['healing_status'] == 'not_healed', 'score_cell_weighted'].to_numpy(float)
    if len(healed) != 7 or len(not_healed) != 4:
        raise RuntimeError(f'unexpected patient arms: healer={len(healed)} nonhealer={len(not_healed)}')
    y = (patients['healing_status'] == 'healed').to_numpy(int)
    scores = patients['score_cell_weighted'].to_numpy(float)
    diff = float(healed.mean() - not_healed.mean())
    welch = ttest_ind(healed, not_healed, equal_var=False)
    margins = [0.05, 0.1, 0.2, 0.3, 0.5]
    sensitivity = [_tost(healed, not_healed, m) for m in margins]
    sample_mean_diff = float(patients.loc[patients['healing_status'] == 'healed', 'score_sample_mean'].mean() - patients.loc[patients['healing_status'] == 'not_healed', 'score_sample_mean'].mean())
    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, 'patient_scores.csv')
    report_path = os.path.join(args.output_dir, 'report.json')
    patients.to_csv(scores_path, index=False)
    report = {'experiment': 'gse165816_discovery_patient_unit_remap', 'status': 'completed', 'scientific_unit': 'patient', 'scientific_scope': 'unit correction of the already-unblinded discovery cohort; not experiment A and not a confirmatory result', 'not_an_independent_cohort': True, 'mcid_filled': False, 'does_not_open_expression_matrix': True, 'does_not_rewrite_scripts_25': True, 'primary_metric': "frozen fibroblast-associated topic mean across all retained cells from that patient's DFU foot samples", 'patient_counts': {'dfu': int(len(patients)), 'healed': int(len(healed)), 'not_healed': int(len(not_healed)), 'collapsed_replicate_patients': int((patients['n_samples'] > 1).sum())}, 'effect_cell_weighted': {'healer_minus_nonhealer_mean_difference': diff, 'welch_t': float(welch.statistic), 'welch_two_sided_p': float(welch.pvalue), 'exact_two_sided_permutation_p': _exact_p(scores, y), 'bootstrap_95_ci': _bootstrap_ci(healed, not_healed, args.seed)}, 'effect_unweighted_sample_means': {'healer_minus_nonhealer_mean_difference': sample_mean_diff, 'note': 'mean of per-GSM means, then mean across patients; not the primary metric'}, 'sample_level_reference_not_rewritten': {'healer_minus_nonhealer_mean_difference': SAMPLE_LEVEL_DIFF, 'source': 'outputs/analysis/patient_mapping_equivalence/report.json'}, 'equivalence': {'status': 'sensitivity_grid_only', 'alpha': 0.05, 'margins': sensitivity, 'smallest_equivalent_margin': _smallest_equivalent_margin(healed, not_healed, diff), 'interpretation': 'TOST is a patient-level restatement of the discovery set and exploratory until a clinical margin is pre-specified from an external source; this is not experiment A'}, 'collapsed_pairs': [{'patient_id': row.patient_id, 'sample_titles': row.sample_titles, 'n_cells': int(row.n_cells)} for row in patients.loc[patients['n_samples'] > 1].itertuples(index=False)], 'protocol': {'seed': args.seed, 'focus_topic': args.focus_topic, 'map_csv': args.map_csv, 'contract_report': args.contract_report, 'discovery_dir': args.discovery_dir, 'map_sha256': sha256(args.map_csv), 'contract_sha256': sha256(args.contract_report), 'author_map_contract_status': contract['status'], 'theta_sha256': sha256(str(discovery / 'theta.npy')), 'obs_sha256': sha256(str(discovery / 'obs.csv')), 'script_sha256': sha256(os.path.abspath(__file__)), 'checked_at_utc': datetime.now(timezone.utc).isoformat()}}
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({'output_dir': args.output_dir, 'patients': int(len(patients)), 'healed': int(len(healed)), 'not_healed': int(len(not_healed)), 'cell_weighted_difference': diff, 'sample_mean_difference': sample_mean_diff, 'exact_p': report['effect_cell_weighted']['exact_two_sided_permutation_p'], 'welch_p': report['effect_cell_weighted']['welch_two_sided_p'], 'smallest_equivalent_margin': report['equivalence']['smallest_equivalent_margin']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
