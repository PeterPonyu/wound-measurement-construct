#!/usr/bin/env python3
"""Observed cells and biological-unit sensitivity."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import false_discovery_control
from sklearn.decomposition import PCA
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.data_adapter import load_dense_csv_tar, mito_fraction
from wound_models.celltype_markers import score_cell_types
from wound_models.human_wound_data import COND_ORDER, COND_TIME, fold_standardize, load_annotated_counts
OUT = ROOT / 'outputs/biological_expansion'
READOUTS = ('fibroblast_mean', 'high_state_fraction', 'fibroblast_fraction')
MARKERS = ['COL1A1', 'PDGFRA', 'KRT14', 'KRT10', 'LYZ', 'CD3D', 'MS4A1', 'PECAM1', 'CCL21', 'RGS5', 'TPSAB1', 'PMEL', 'DCD', 'TIMP1', 'CHI3L1', 'FN1', 'DCN', 'LUM']
TEMP_GENES = ['COL1A1', 'COL3A1', 'DCN', 'LUM', 'TIMP1', 'CHI3L1', 'FN1', 'MMP1', 'MMP3', 'IL6', 'CXCL12', 'ACTA2']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def provenance(paths):
    return {'protocol_sha256': sha(ROOT / 'config/biological_expansion_protocol.json'), 'script_sha256': sha(__file__), 'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths}}

def save_report(folder, data, paths):
    data['provenance'] = provenance(paths)
    (folder / 'report.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def patient_rows(sample, equal=False):
    rows = []
    for patient, d in sample.groupby('patient', sort=True):
        w_fib = np.ones(len(d)) if equal else d.n_fib.to_numpy(float)
        w_all = np.ones(len(d)) if equal else d.n_all.to_numpy(float)
        rows.append({'patient': patient, 'arm': d.arm.iloc[0], 'specimens': len(d), 'n_all': int(d.n_all.sum()), 'n_fib': int(d.n_fib.sum()), 'fibroblast_mean': float(np.average(d.fibroblast_mean, weights=w_fib)), 'high_state_fraction': float(np.average(d.high_state_fraction, weights=w_fib)), 'fibroblast_fraction': float(np.average(d.fibroblast_fraction, weights=w_all))})
    return pd.DataFrame(rows)

def exact_contrast(values, is_healed):
    values = np.asarray(values, dtype=float)
    is_healed = np.asarray(is_healed, dtype=bool)
    n_h = int(is_healed.sum())
    n = len(values)
    observed = float(values[is_healed].mean() - values[~is_healed].mean())
    null = []
    for indices in itertools.combinations(range(n), n_h):
        total_h = values[list(indices)].sum()
        null.append(total_h / n_h - (values.sum() - total_h) / (n - n_h))
    tol = 1e-12 * max(1.0, abs(observed))
    p = float(np.mean(np.abs(null) >= abs(observed) - tol))
    return (observed, p, len(null))

def measurement():
    folder = OUT / 'measurement'
    folder.mkdir(parents=True, exist_ok=True)
    par = ROOT / 'outputs/celltype_resolved/GSE165816_discovery_theta.parquet'
    mapping = ROOT / 'cohort_metadata/GSE165816_ncomms_subject_sample_map.csv'
    threshold_path = ROOT / 'outputs/bimodal_stratification/report.json'
    frame = pd.read_parquet(par)
    midpoint = json.loads(threshold_path.read_text())['cohorts']['GSE165816_discovery']['gmm']['midpoint']
    map_df = pd.read_csv(mapping).set_index('sample_id')
    ids = sorted(map_df.loc[sorted(frame.gsm.unique()), 'patient_id'].unique())
    labels = {key: f'Patient {i + 1:02d}' for i, key in enumerate(ids)}
    samples = []
    for gsm, d in frame.groupby('gsm', sort=True):
        fib = d.loc[d.celltype.eq('fibroblast'), 'topic_0'].to_numpy(float)
        assert len(fib) >= 50
        samples.append({'sample': gsm, 'patient': labels[map_df.loc[gsm, 'patient_id']], 'arm': d.arm.iloc[0], 'n_all': len(d), 'n_fib': len(fib), 'fibroblast_mean': float(fib.mean()), 'high_state_fraction': float(np.mean(fib > midpoint)), 'fibroblast_fraction': float(len(fib) / len(d))})
    sample = pd.DataFrame(samples)
    patient = patient_rows(sample)
    outcome = patient.loc[patient.arm.isin(['DFU-healer', 'DFU-nonhealer'])].reset_index(drop=True)
    assert len(outcome) == 11 and outcome.arm.eq('DFU-healer').sum() == 7
    sample.to_csv(folder / 'specimen_readouts.csv', index=False)
    patient.to_csv(folder / 'patient_readouts.csv', index=False)
    variants = {'cell-weighted': outcome, 'equal-specimen': patient_rows(sample, equal=True), 'minimum-100-fibroblasts': patient_rows(sample.loc[sample.n_fib.ge(100)])}
    rows = []
    omissions = []
    rng = np.random.default_rng(17)
    for variant, table in variants.items():
        rng = np.random.default_rng(17)
        table = table.loc[table.arm.isin(['DFU-healer', 'DFU-nonhealer'])]
        healed = table.arm.eq('DFU-healer').to_numpy()
        for readout in READOUTS:
            x = table[readout].to_numpy()
            delta, p, nperm = exact_contrast(x, healed)
            h, nh = (x[healed], x[~healed])
            draws = rng.choice(h, (10000, len(h))).mean(axis=1) - rng.choice(nh, (10000, len(nh))).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append({'variant': variant, 'readout': readout, 'n_healed': len(h), 'n_nonhealed': len(nh), 'healed_mean': float(h.mean()), 'nonhealed_mean': float(nh.mean()), 'difference': delta, 'ci_low': float(low), 'ci_high': float(high), 'exact_p': p, 'permutations': nperm})
            if variant == 'cell-weighted':
                for i, row in table.iterrows():
                    retained = table.drop(index=i)
                    dd, pp, _ = exact_contrast(retained[readout], retained.arm.eq('DFU-healer'))
                    omissions.append({'omitted': row.patient, 'readout': readout, 'difference': dd, 'exact_p': pp})
    for variant in variants:
        current = [r for r in rows if r['variant'] == variant]
        for r, q in zip(current, false_discovery_control([r['exact_p'] for r in current])):
            r['bh_q'] = float(q)
    pd.DataFrame(rows).to_csv(folder / 'patient_contrasts.csv', index=False)
    pd.DataFrame(omissions).to_csv(folder / 'patient_omissions.csv', index=False)
    composition = frame.groupby(['gsm', 'arm', 'celltype'], observed=True).size().rename('cells').reset_index()
    composition['fraction'] = composition.cells / composition.groupby('gsm').cells.transform('sum')
    composition.to_csv(folder / 'cell_composition.csv', index=False)
    histogram = []
    for arm, group in frame.loc[frame.celltype.eq('fibroblast')].groupby('arm', sort=True):
        counts, edges = np.histogram(group.topic_0.to_numpy(), bins=np.linspace(0, 1, 51))
        for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
            histogram.append({'arm': arm, 'loading': float((lo + hi) / 2), 'density': float(count / len(group) / (hi - lo)), 'cells': int(count)})
    pd.DataFrame(histogram).to_csv(folder / 'loading_distribution.csv', index=False)
    embedding_file = folder / 'embedding_all.npy'
    if not embedding_file.exists():
        from umap import UMAP
        theta = frame[[f'topic_{i}' for i in range(15)]].to_numpy(float)
        assert np.all(theta > 0)
        clr = np.log(theta)
        clr -= clr.mean(axis=1, keepdims=True)
        coords = UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=17, n_jobs=1).fit_transform(clr)
        np.save(embedding_file, coords)
    coords = np.load(embedding_file)
    chosen = np.random.default_rng(17).choice(len(frame), 2000, replace=False)
    display = frame.iloc[chosen][['gsm', 'arm', 'celltype', 'topic_0']].copy()
    display['x'], display['y'] = coords[chosen].T
    display.to_csv(folder / 'atlas_display.csv', index=False)
    save_report(folder, {'status': 'complete', 'threshold': midpoint, 'cells': len(frame), 'fibroblasts': int(frame.celltype.eq('fibroblast').sum()), 'specimens': len(sample), 'patients': len(patient), 'lineage_counts': frame.celltype.value_counts().to_dict(), 'contrasts': rows, 'omissions': omissions, 'display_cells': len(chosen), 'umap': {'neighbors': 30, 'minimum_distance': 0.3, 'seed': 17, 'input': 'centered log ratio of 15 frozen topic proportions'}}, [par, mapping, threshold_path])
    print('Measurement patient tests and atlas complete', flush=True)

def expression_summary(matrix, genes, obs, grouping, selected_genes):
    pos = pd.Index(genes).get_indexer(selected_genes)
    assert np.all(pos >= 0), 'A requested display gene is absent'
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    counts = matrix[:, pos].toarray().astype(float)
    logs = np.log1p(counts / np.maximum(totals[:, None], 1) * 10000.0)
    records = []
    for key, group in obs.groupby(grouping, observed=True, sort=True):
        keys = (key,) if not isinstance(key, tuple) else key
        ix = group.index.to_numpy()
        for j, gene in enumerate(selected_genes):
            records.append({**dict(zip(grouping, keys)), 'gene': gene, 'cells': len(ix), 'mean_log1p_cp10k': float(logs[ix, j].mean()), 'detected_fraction': float((counts[ix, j] > 0).mean()), 'pseudobulk_log1p_cp10k': float(np.log1p(counts[ix, j].sum() / max(totals[ix].sum(), 1) * 10000.0))})
    return pd.DataFrame(records)

def raw_expression():
    folder = OUT / 'measurement'
    raw = ROOT / 'data/raw/GSE165816/GSE165816_RAW.tar'
    matrix_path = ROOT / 'data/raw/GSE165816/GSE165816_series_matrix.txt.gz'
    if not (folder / 'marker_expression.csv').exists():
        print('Loading discovery raw counts for aligned marker display', flush=True)
        coh = load_dense_csv_tar(str(raw), series_matrix=str(matrix_path))
        obs = coh.obs.copy()
        obs['arm'] = obs.disease
        keep = (obs.tissue.str.lower().eq('foot skin') & obs.arm.isin(['DFU-healer', 'DFU-nonhealer', 'Non-diabetic'])).to_numpy()
        keep &= (np.asarray((coh.X > 0).sum(axis=1)).ravel() >= 200) & (mito_fraction(coh.X, coh.genes) <= 0.2)
        obs = obs.loc[keep].reset_index(drop=True)
        x = coh.X[keep]
        saved = pd.read_parquet(ROOT / 'outputs/celltype_resolved/GSE165816_discovery_theta.parquet')
        assert len(obs) == len(saved) and np.array_equal(obs.gsm, saved.gsm) and np.array_equal(obs.arm, saved.arm)
        assigned, _, _ = score_cell_types(x, coh.genes)
        assert np.array_equal(assigned, saved.celltype), 'Raw-row reconstruction changed lineage labels'
        obs['celltype'] = assigned
        expression_summary(x, coh.genes, obs, ['celltype'], MARKERS).to_csv(folder / 'marker_expression.csv', index=False)
        expression_summary(x, coh.genes, obs, ['gsm', 'celltype'], MARKERS).to_csv(folder / 'specimen_marker_expression.csv', index=False)
        (folder / 'raw_alignment.json').write_text(json.dumps({'cells': len(obs), 'gsm_order_matches': True, 'arm_order_matches': True, 'all_lineage_assignments_match': True, **provenance([raw, matrix_path])}, indent=2) + '\n')
        del coh, x, obs
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Recompute the observed-cell and biological-unit extension')
    ap.add_argument('part', choices=['measurement', 'expression'])
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reruns/biological_expansion')
    args = ap.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'measurement').mkdir(parents=True, exist_ok=True)
    if args.part == 'measurement' and (OUT / 'measurement' / 'report.json').exists():
        raise FileExistsError('Choose a fresh output directory to preserve completed results')
    {'measurement': measurement, 'expression': raw_expression}[args.part]()
