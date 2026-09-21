"""Assess robustness."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def association(x, y):
    return float(spearmanr(x, y).statistic)

def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()

def patient_analysis(n_resamples):
    source = ROOT / 'outputs/bimodal_stratification/report.json'
    mapping = ROOT / 'cohort_metadata/GSE165816_ncomms_subject_sample_map.csv'
    scores = ROOT / 'outputs/cohort_metadata/gse165816_patient_unit_remap/patient_scores.csv'
    rows = json.loads(source.read_text())['cohorts']['GSE165816_discovery']['per_sample']
    sample = pd.DataFrame(rows).merge(pd.read_csv(mapping)[['sample_id', 'patient_id']], left_on='gsm', right_on='sample_id', validate='one_to_one')
    assert len(sample) == 25 and sample.patient_id.nunique() == 20
    patients = []
    for pid, block in sample.groupby('patient_id', sort=True):
        patients.append({'patient': pid.rsplit('_', 1)[-1], 'fibroblast_mean': float(np.average(block.topic0_mean, weights=block.n_fib)), 'high_state_fraction': float(np.average(block.mixing_weight, weights=block.n_fib)), 'specimens': len(block)})
    patients = pd.DataFrame(patients)
    x = patients.fibroblast_mean.to_numpy()
    y = patients.high_state_fraction.to_numpy()
    rng = np.random.default_rng(217)
    indices = rng.integers(0, len(x), size=(n_resamples, len(x)))
    boot = np.asarray([association(x[i], y[i]) for i in indices])
    if not np.isfinite(boot).all():
        raise ValueError('Nonfinite patient bootstrap draw')
    omission = [{'omitted_patient': patients.patient[i], 'rho': association(np.delete(x, i), np.delete(y, i))} for i in range(len(x))]
    outcome = pd.read_csv(scores)
    values = outcome.score_cell_weighted.to_numpy()
    healed = outcome.healing_status.eq('healed').to_numpy()
    contrast = float(values[healed].mean() - values[~healed].mean())
    outcome_omission = []
    for i in range(len(outcome)):
        keep = np.arange(len(outcome)) != i
        delta = float(values[healed & keep].mean() - values[~healed & keep].mean())
        outcome_omission.append({'omitted_patient': outcome.patient_id[i].rsplit('_', 1)[-1], 'healing_minus_nonhealing': delta})
    return {'patient_semantics': {'n_patients': len(patients), 'n_specimens': len(sample), 'rho': association(x, y), 'bootstrap_95_interval': interval(boot), 'n_resamples': n_resamples, 'seed': 217, 'resampling_unit': 'patient after fibroblast-count-weighted collapse', 'leave_one_patient_out': omission, 'omission_range': [min((r['rho'] for r in omission)), max((r['rho'] for r in omission))], 'patients': patients.to_dict('records')}, 'patient_outcome_influence': {'n_patients': len(outcome), 'observed_difference': contrast, 'leave_one_patient_out': outcome_omission, 'omission_range': [min((r['healing_minus_nonhealing'] for r in outcome_omission)), max((r['healing_minus_nonhealing'] for r in outcome_omission))], 'interpretation': 'Influence diagnostic; no interval, significance test or independent validation.'}, 'source_fingerprints': {str(p.relative_to(ROOT)): fingerprint(p) for p in (source, mapping, scores)}}

def main():
    ap = argparse.ArgumentParser(description="Recompute the study robustness analysis")
    ap.add_argument('--output-dir', default=str(ROOT/'outputs/reruns/robustness'))
    ap.add_argument('--bootstrap-draws', type=int, default=10000)
    args=ap.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    started=time.time()
    result={'status':'completed','scope':'Retrospective sensitivity within the existing cohort; no additional biological replicates.', 'patient':patient_analysis(args.bootstrap_draws)}
    result['protocol']={'script_sha256':fingerprint(Path(__file__)), 'elapsed_seconds':time.time()-started}
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed: '+str(output/'report.json'), flush=True)
if __name__ == '__main__':
    main()
