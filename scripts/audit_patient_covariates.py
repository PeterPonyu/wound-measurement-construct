#!/usr/bin/env python3
"""Patient-level descriptive re-analysis of the historical 31-covariate scan.

No p values, discovery labels or model selection: this remains an unblinded
construct diagnostic in 20 existing patients. Original files are read only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_measurement_math import fingerprint, load_current_measurement

def collapse_covariates(cells, technical, cut):
    """Use biological denominators, not an average of incompatible fractions."""
    if not technical.gsm.is_unique or set(technical.gsm) != set(cells.gsm):
        raise ValueError('Historical technical table coverage/uniqueness mismatch')
    needed = ['tech_diss', 'tech_umi', 'tech_ptprc_pos']
    if not np.isfinite(technical[needed].to_numpy(float)).all():
        raise ValueError('Nonfinite technical values')
    technical = technical.set_index('gsm')
    fibro = cells[cells.celltype.eq('fibroblast')].copy()
    fibro['high'] = fibro.topic_0 > cut
    rows = []
    for pid, all_cells in cells.groupby('patient_id', sort=True):
        block = fibro[fibro.patient_id.eq(pid)]
        if block.empty:
            raise ValueError('Patient has no fibroblast support')
        counts = block.groupby('gsm').size()
        proportions = all_cells.celltype.value_counts() / len(all_cells)
        row = dict(patient_id=pid, arm=all_cells.arm.iloc[0], n_specimens=all_cells.gsm.nunique(), weight=float(block.high.mean()), n_fibroblasts=len(block), n_cells_total=len(all_cells))
        for i in range(1, 15):
            row[f'fib_mean_topic_{i}'] = float(block[f'topic_{i}'].to_numpy(np.float64).mean())
        for lineage in sorted(cells.celltype.unique()):
            row[f'frac_{lineage}'] = float(proportions.get(lineage, 0))
        for key in needed:
            row[key] = float(np.average(technical.loc[counts.index, key], weights=counts))
        specimen_numbers = all_cells.groupby('gsm').sample_title.first().str.extract('^G(\\d+)', expand=False)
        if specimen_numbers.isna().any():
            raise ValueError('Unexpected specimen title for order diagnostic')
        row['meta_g_number'] = float(specimen_numbers.astype(float).mean())
        rows.append(row)
    patients = pd.DataFrame(rows)
    if not np.allclose(patients.filter(regex='^frac_').sum(axis=1), 1, rtol=0, atol=1e-14):
        raise ValueError('Patient composition denominator does not close')
    return patients

def describe_associations(patients):
    feature_names = [key for key in patients if key not in {'patient_id', 'arm', 'n_specimens', 'weight'}]
    if len(feature_names) != 31:
        raise ValueError('Expected the same 31-feature scope, without selecting winners')
    y = patients.weight.to_numpy(float)
    rows = []
    for key in feature_names:
        x = patients[key].to_numpy(float)
        if not np.isfinite(x).all() or len(np.unique(x)) < 2:
            raise ValueError(f'Non-estimable patient covariate: {key}')
        rho = float(spearmanr(x, y).statistic)
        omissions = [float(spearmanr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(len(x))]
        if not np.isfinite(omissions).all():
            raise ValueError(f'Non-estimable omission diagnostic: {key}')
        rows.append(dict(covariate=key, n_patients=len(x), rho=rho, omission_min=min(omissions), omission_max=max(omissions), omission_sign_changes=sum((value * rho < 0 for value in omissions))))
    return pd.DataFrame(rows).sort_values('covariate').reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory')
    cells, _, binding = load_current_measurement()
    technical_path = ROOT / 'outputs/stratifier_sweep/sample_features.csv'
    mixture_path = ROOT / 'outputs/bimodal_stratification/report.json'
    technical = pd.read_csv(technical_path)
    cut = json.loads(mixture_path.read_text())['cohorts']['GSE165816_discovery']['gmm']['midpoint']
    counts = cells.groupby('gsm').size()
    fibro_counts = cells[cells.celltype.eq('fibroblast')].groupby('gsm').size()
    historical = technical.set_index('gsm')
    if not technical.gsm.is_unique or set(historical.index) != set(counts.index) or (not np.array_equal(historical.loc[counts.index, 'n_cells_total'], counts)) or (not np.array_equal(historical.loc[fibro_counts.index, 'n_fibroblasts'], fibro_counts)):
        raise ValueError('Historical technical summaries are not aligned to current cells')
    patients = collapse_covariates(cells, technical, cut)
    associations = describe_associations(patients)
    sources = {**binding['source_fingerprints'], str(technical_path.relative_to(ROOT)): fingerprint(technical_path), str(mixture_path.relative_to(ROOT)): fingerprint(mixture_path)}
    report = dict(status='completed', n_patients=len(patients), n_specimens=cells.gsm.nunique(), n_covariates=len(associations), new_model_fits=0, new_biological_units=0, sources_sha256=sources, script_sha256=fingerprint(Path(__file__)), protocol=dict(threshold=cut, outcome='Patient fibroblast high-fraction at the current midpoint', topic_means='Pool current deterministic fibroblast coordinates across patient specimens', composition='Pool each lineage count across specimens, divide by all patient cells', count_covariates='Sum counts across patient specimens', technical='Fibroblast-count-weighted mean of three saved specimen summaries; tech_umi is a mean of specimen medians, NOT a pooled-patient median', specimen_order='Unweighted average of specimen G-number within patient; not a biological batch identifier', uncertainty='Leave-one-patient-out influence range, NOT confidence bounds', inference='No p/BH values, no causal or adjusted effect, no selection or independent validation; mixed clinical arms remain confounded'), associations=associations.to_dict('records'))
    if any((fingerprint(ROOT / path) != digest for path, digest in sources.items())):
        raise ValueError('An input changed during patient-covariate computation')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    patients.to_csv(args.output_dir / 'patient_covariates.csv', index=False)
    associations.to_csv(args.output_dir / 'associations.csv', index=False)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(associations.to_json(orient='records', indent=2))
if __name__ == '__main__':
    main()
