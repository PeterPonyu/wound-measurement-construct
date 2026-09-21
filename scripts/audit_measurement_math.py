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
    args = ap.parse_args()
    if (args.output_dir / 'report.json').exists():
        raise FileExistsError('Choose a fresh output directory; reference reports are read only')
    inputs = {'cell_projection': ROOT / 'outputs/celltype_resolved/GSE165816_discovery_theta.parquet', 'mixture_fit': ROOT / 'outputs/bimodal_stratification/report.json', 'patient_scores': ROOT / 'outputs/cohort_metadata/gse165816_patient_unit_remap/patient_scores.csv'}
    cells = pd.read_parquet(inputs['cell_projection'])
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
    patients = pd.read_csv(inputs['patient_scores'])
    labels = patients.healing_status.eq('healed').to_numpy()
    scores = patients.score_cell_weighted.to_numpy()
    assert len(scores) == 11 and labels.sum() == 7
    ranges = {key: [min((row[key] for row in rows if row[key] is not None)), max((row[key] for row in rows if row[key] is not None))] for key in ('low_state_mean', 'high_state_mean', 'within_state_residual')}
    report = {'status': 'completed', 'scope': 'Frozen-score algebra and conditional patient-label inference; no new biological replication or causal identification.', 'state_decomposition': {'n_specimens': len(rows), 'n_fibroblasts': len(cells), 'cut': cut, 'reference': 'Cell-pooled means of the two threshold bins; not fitted Gaussian means', 'low_reference': low_reference, 'high_reference': high_reference, 'empty_bin_specimens': sum((row['low_state_mean'] is None or row['high_state_mean'] is None for row in rows)), 'ranges': ranges, 'specimen_weighted_residual_rms': float(np.sqrt(np.mean([row['within_state_residual'] ** 2 for row in rows]))), 'max_identity_error': max((row['identity_error'] for row in rows)), 'max_residual_identity_error': max((row['residual_identity_error'] for row in rows)), 'specimens': rows}, 'patient_permutation': exhaustive_contrast(scores, labels), 'patient_equivalence': equivalence_boundary(scores[labels], scores[~labels]), 'protocol': {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()}, 'computation_dtype': 'float64', 'label_exchangeability': 'assumed, not supplied by randomization'}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / 'state_decomposition.csv', index=False)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'state_decomposition'}, indent=2))
    print(json.dumps({key: value for key, value in report['state_decomposition'].items() if key != 'specimens'}, indent=2))
if __name__ == '__main__':
    main()
