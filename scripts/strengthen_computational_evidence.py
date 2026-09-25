#!/usr/bin/env python3
"""Saved-input computational sensitivity."""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.expand_biological_evidence import READOUTS, exact_contrast, patient_rows, sha
from wound_models.human_wound_data import COND_ORDER, COND_TIME, fold_standardize
OUT = ROOT / 'outputs/computational_extension'
PROTOCOL = ROOT / 'config/computational_extension_protocol.json'

def save(folder, report, inputs):
    report['provenance'] = {'script_sha256': sha(__file__), 'protocol_sha256': sha(PROTOCOL), 'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs}, 'output_sha256': {p.name: sha(p) for p in sorted(folder.glob('*.csv'))}}
    (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'provenance'}, indent=2), flush=True)

def allocation_weights(n, n_healed):
    if not 0 < n_healed < n:
        raise ValueError('Both patient groups must be nonempty')
    weights = np.full((len(list(itertools.combinations(range(n), n_healed))), n), -1 / (n - n_healed))
    for i, selected in enumerate(itertools.combinations(range(n), n_healed)):
        weights[i, list(selected)] = 1 / n_healed
    return weights

def ecdf_contrast(patient_values, healed):
    """One normalized empirical measure per patient; cells never get labels."""
    support = np.unique(np.concatenate(patient_values))
    cdf = np.array([np.searchsorted(np.sort(v), support, side='right') / len(v) for v in patient_values])
    healed = np.asarray(healed, bool)
    observed = cdf[healed].mean(0) - cdf[~healed].mean(0)
    null = allocation_weights(len(cdf), int(healed.sum())) @ cdf
    maxima = np.max(np.abs(null), axis=1)
    statistic = float(np.max(np.abs(observed)))
    return (support, observed, maxima, statistic, float(np.mean(maxima >= statistic - 1e-12)))

def measurement():
    folder = OUT / 'measurement'
    folder.mkdir(parents=True, exist_ok=True)
    bio = ROOT / 'outputs/biological_expansion/measurement'
    par = ROOT / 'outputs/celltype_resolved/GSE165816_discovery_theta.parquet'
    sample_path = bio / 'specimen_readouts.csv'
    samples = pd.read_csv(sample_path)
    samples = samples[samples.arm.isin(['DFU-healer', 'DFU-nonhealer'])].copy()
    frame = pd.read_parquet(par)
    values = {s: g.topic_0.to_numpy(float) for s, g in frame[frame.celltype.eq('fibroblast')].groupby('gsm') if s in set(samples['sample'])}
    patients = patient_rows(samples)
    healed = patients.arm.eq('DFU-healer').to_numpy()
    assert len(patients) == 11 and healed.sum() == 7
    arrays = [np.concatenate([values[s] for s in samples.loc[samples.patient.eq(p), 'sample']]) for p in patients.patient]
    support, observed, maxima, stat, p = ecdf_contrast(arrays, healed)
    threshold = json.loads((bio / 'report.json').read_text())['threshold']
    cuts = sorted(set(json.loads(PROTOCOL.read_text())['measurement']['threshold_grid'] + [threshold]))
    rows = []
    for cut in cuts:
        fractions = np.array([np.mean(v > cut) for v in arrays])
        delta, prob, n = exact_contrast(fractions, healed)
        rows.append(dict(threshold=cut, is_original=bool(cut == threshold), difference=delta, exact_p=prob, support_max_adjusted_p=float(np.mean(maxima >= abs(delta) - 1e-12))))
    pd.DataFrame(rows).to_csv(folder / 'threshold_sensitivity.csv', index=False)
    pd.DataFrame({'loading': support, 'healed_minus_nonhealed_cdf': observed}).to_csv(folder / 'patient_ecdf_contrast.csv', index=False)
    pd.DataFrame({'allocation': np.arange(len(maxima)), 'supremum': maxima}).to_csv(folder / 'ecdf_permutation_null.csv', index=False)
    rng = np.random.default_rng(29)
    rare = []
    assert all((len(v) >= 100 for v in values.values()))
    for draw in range(250):
        sparse = samples.copy()
        for idx, row in sparse.iterrows():
            x = rng.choice(values[row['sample']], 100, replace=False)
            sparse.loc[idx, 'fibroblast_mean'] = x.mean()
            sparse.loc[idx, 'high_state_fraction'] = np.mean(x > threshold)
        table = patient_rows(sparse, equal=True)
        for readout in READOUTS[:2]:
            delta, prob, _ = exact_contrast(table[readout], table.arm.eq('DFU-healer'))
            rare.append(dict(draw=draw, readout=readout, difference=delta, exact_p=prob))
    pd.DataFrame(rare).to_csv(folder / 'equal_cell_draws.csv', index=False)
    choices = [g.index.tolist() for _, g in samples.groupby('patient', sort=True)]
    selected_rows = []
    for choice, idx in enumerate(itertools.product(*choices)):
        table = samples.loc[list(idx)]
        for readout in READOUTS:
            delta, prob, _ = exact_contrast(table[readout], table.arm.eq('DFU-healer'))
            selected_rows.append(dict(choice=choice, readout=readout, difference=delta, exact_p=prob, specimens='|'.join(table['sample'])))
    pd.DataFrame(selected_rows).to_csv(folder / 'one_specimen_choices.csv', index=False)
    summaries = []
    for name, table in [('Equal 100 cells per specimen', pd.DataFrame(rare)), ('One specimen per patient', pd.DataFrame(selected_rows))]:
        for readout, g in table.groupby('readout', sort=False):
            summaries.append(dict(sensitivity=name, readout=readout, comparisons=len(g), difference_mean=float(g.difference.mean()), difference_min=float(g.difference.min()), difference_max=float(g.difference.max()), p_min=float(g.exact_p.min()), p_max=float(g.exact_p.max()), positive=int((g.difference > 0).sum())))
    pd.DataFrame(summaries).to_csv(folder / 'sampling_summary.csv', index=False)
    save(folder, dict(status='complete', patients=11, healed=7, nonhealed=4, threshold_free_statistic=stat, threshold_free_exact_p=p, allocations=len(maxima), threshold_rows=rows, sampling=summaries, limits='Retrospective patient-label exchangeability assumption; sampling ranges are not confidence intervals or independent replications'), [par, sample_path, bio / 'report.json', ROOT / 'scripts/expand_biological_evidence.py'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recompute saved-input computational sensitivity')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reruns/computational_extension')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if (OUT / 'measurement' / 'report.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    measurement()
