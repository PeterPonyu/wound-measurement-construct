#!/usr/bin/env python3
"""Read-only probes of measurement artifacts; writes only its companion JSON.

No model is trained and no production analysis artifact is changed. Run from any
directory with the repository's existing Python environment.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, t
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core_engines.data_adapter import _read_dense_csv_member, library_normalize, mito_fraction
from core_engines.topic_simplex_engine import TopicSimplexModel


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def difference(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return {'maximum_absolute': float(np.max(np.abs(d))),
            'rms': float(np.sqrt(np.mean(d * d)))}


def auc(scores, healed):
    a, b = np.asarray(scores)[healed], np.asarray(scores)[~healed]
    return float(((a[:, None] > b[None, :]) + .5 * (a[:, None] == b[None, :])).mean())


def oriented_loo(x, healed):
    out = np.empty(len(x))
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        sign = 1 if x[keep & healed].mean() >= x[keep & ~healed].mean() else -1
        out[i] = sign * x[i]
    return out


def contrast(x, healed):
    x, healed = np.asarray(x, float), np.asarray(healed, bool)
    n_h, n_n = int(healed.sum()), int((~healed).sum())
    delta = float(x[healed].mean() - x[~healed].mean())
    null, auc_null = [], []
    for inds in itertools.combinations(range(len(x)), n_h):
        h = np.isin(np.arange(len(x)), inds)
        null.append(x[h].mean() - x[~h].mean())
        auc_null.append(auc(oriented_loo(x, h), h))
    null = np.asarray(null)
    n_exceed = int(np.sum(np.abs(null) >= abs(delta) - 1e-12))
    rng = np.random.default_rng(0)
    draws = rng.choice(x[healed], (30000, n_h)).mean(1) - rng.choice(x[~healed], (30000, n_n)).mean(1)
    vh, vn = x[healed].var(ddof=1)/n_h, x[~healed].var(ddof=1)/n_n
    se = float(np.sqrt(vh + vn))
    df = float((vh + vn)**2/(vh**2/(n_h-1) + vn**2/(n_n-1)))
    observed_auc = auc(oriented_loo(x, healed), healed)
    auc_exceed = int(np.sum(np.asarray(auc_null) >= observed_auc - 1e-12))
    return {'n_healed': n_h, 'n_nonhealed': n_n, 'difference': delta,
            'all_allocations': len(null), 'absolute_exceed_count': n_exceed,
            'exact_p': n_exceed/len(null), 'add_one_p': (1+n_exceed)/(1+len(null)),
            'bootstrap_seed': 0, 'bootstrap_draws': 30000,
            'bootstrap_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'welch_se': se, 'welch_df': df,
            'equivalence_margin_infimum': abs(delta) + float(t.ppf(.95, df))*se,
            'oriented_loo_auc': observed_auc,
            'auc_null_mean': float(np.mean(auc_null)),
            'auc_permutation_exact_p': auc_exceed/len(null),
            'auc_permutation_add_one_p': (1+auc_exceed)/(1+len(null))}


def main():
    torch.set_num_threads(2)
    paths = {
        'raw_tar': ROOT/'data/raw/GSE165816/GSE165816_RAW.tar',
        'series_matrix': ROOT/'data/raw/GSE165816/GSE165816_series_matrix.txt.gz',
        'main_theta': ROOT/'outputs/direction1_dfu_mainline/theta.npy',
        'main_obs': ROOT/'outputs/direction1_dfu_mainline/obs.csv',
        'model': ROOT/'outputs/direction1_dfu_mainline/topic_model.pt',
        'panel': ROOT/'outputs/direction1_dfu_mainline/panel.npy',
        'beta': ROOT/'outputs/direction1_dfu_mainline/beta.npy',
        'lineage_theta': ROOT/'outputs/celltype_resolved/GSE165816_discovery_theta.parquet',
        'patient_map': ROOT/'metadata_intake/GSE165816_ncomms_subject_sample_map.csv',
        'anatomy_samples': ROOT/'outputs/repaired_20260918/paired_anatomical_control/sample_scores.csv',
        'specimen_readouts': ROOT/'outputs/biological_expansion/measurement/specimen_readouts.csv',
        'model_source': ROOT/'core_engines/topic_simplex_engine.py',
        'mainline_source': ROOT/'scripts/10_run_dfu_mainline.py',
        'lineage_source': ROOT/'scripts/12_celltype_resolved_test.py',
        'patient_source': ROOT/'scripts/35_gse165816_patient_unit_remap.py',
        'shuffle_source': ROOT/'scripts/38_pipeline_negative_control.py',
        'external_report': ROOT/'outputs/repaired_20260918/external_immune_axis/report.json',
    }
    main_theta = np.load(paths['main_theta'])
    obs = pd.read_csv(paths['main_obs'])
    frame = pd.read_parquet(paths['lineage_theta'])
    cols = [f'topic_{i}' for i in range(15)]
    assert len(obs) == len(frame) == len(main_theta) == 76461
    assert np.array_equal(obs.gsm, frame.gsm)
    assert np.array_equal(obs.disease, frame.arm)
    current = frame[cols].to_numpy()
    result = {
        'scope': 'Read-only artifact and single-specimen raw-count reprojection probes; no model fitting, no claim that a stored run has a proven historical code version.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        'whole_artifact_comparison': {
            'rows': len(frame), 'columns': 15, 'gsm_row_order_equal': True,
            'arm_row_order_equal': True,
            'cell_barcodes_present_in_saved_obs': False,
            'cell_barcodes_present_in_saved_parquet': False,
            'main_vs_lineage_theta': difference(main_theta, current),
            'topic0_pearson': float(np.corrcoef(main_theta[:, 0], current[:, 0])[0, 1]),
        },
    }
    # Read one original sample in its CSV column (barcode) order; apply exactly
    # the original QC and remap genes to the saved panel. Nothing is refitted.
    gsm = 'GSM5050530'
    with tarfile.open(paths['raw_tar']) as archive:
        member = next(m for m in archive.getmembers() if m.name.startswith(gsm+'_'))
        payload = archive.extractfile(member).read()
        X, genes, barcodes = _read_dense_csv_member(gzip.open(io.BytesIO(payload), 'rb'))
    qc = (np.asarray((X > 0).sum(1)).ravel() >= 200) & (mito_fraction(X, genes) <= .2)
    raw_indices = np.flatnonzero(qc)
    X = X[qc]
    panel = np.load(paths['panel'], allow_pickle=True)
    pos = pd.Index(genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    Xp = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(len(raw_indices), len(panel)))
    model = TopicSimplexModel(len(panel), 15)
    model.load_state_dict(torch.load(paths['model'], map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        deterministic = model(torch.tensor(library_normalize(Xp).toarray(), dtype=torch.float32), deterministic=True)['theta'].numpy()
    global_indices = np.flatnonzero(obs.gsm.eq(gsm).to_numpy())
    assert len(global_indices) == len(raw_indices) == 2279
    lineage_rows = frame.loc[frame.gsm.eq(gsm), cols].to_numpy()
    main_rows = main_theta[global_indices]
    assert np.max(np.abs(deterministic-lineage_rows)) < 1e-6
    lineage_mask = frame.loc[frame.gsm.eq(gsm), 'celltype'].eq('fibroblast').to_numpy()
    result['single_specimen_raw_reprojection'] = {
        'gsm': gsm, 'raw_archive_member': member.name,
        'archive_member_sha256': hashlib.sha256(payload).hexdigest(),
        'raw_cells': len(barcodes), 'qc_cells': len(raw_indices),
        'raw_qc_rows_zero_based': raw_indices.tolist(),
        'saved_global_rows_zero_based': global_indices.tolist(),
        'reconstructed_cell_ids_in_order': [f'{gsm}:{b}' for b in barcodes[qc]],
        'cell_id_limitation': 'Original saved artifacts omit barcode IDs. Linkage uses documented stable CSV column order after identical QC, the identical GSM row block length, and a cellwise match to all 15 deterministic coordinates within 1e-6. Raw IDs are reconstructed, not claimed to have been stored with theta.npy.',
        'deterministic_vs_lineage_parquet': difference(deterministic, lineage_rows),
        'deterministic_vs_main_theta': difference(deterministic, main_rows),
        'deterministic_all_cell_topic0_mean': float(deterministic[:, 0].mean()),
        'lineage_all_cell_topic0_mean': float(lineage_rows[:, 0].mean()),
        'main_all_cell_topic0_mean': float(main_rows[:, 0].mean()),
        'deterministic_fibroblast_mean': float(deterministic[lineage_mask, 0].mean()),
        'main_fibroblast_mean_using_same_gate': float(main_rows[lineage_mask, 0].mean()),
    }
    historical = subprocess.run(['git', 'show', 'd9ca82d:core_engines/topic_simplex_engine.py'], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    result['historical_source_check'] = {
        'commit': 'd9ca82d', 'source': 'core_engines/topic_simplex_engine.py',
        'source_sha256': hashlib.sha256(historical.encode()).hexdigest(),
        'forward_unconditionally_samples': 'z = self.reparameterize(mu, logvar)' in historical,
        'evidence_lines': '89-92: forward computes mu/logvar, unconditionally reparameterizes, then softmaxes; no eval-mode deterministic branch.',
        'origin_inference': 'Historical stochastic projection is a supported candidate explanation, but no artifact-generating code hash or RNG state proves this origin. The confirmed defect is incompatible saved projection artifacts under the stated deterministic definition.',
    }
    mapping = pd.read_csv(paths['patient_map']).set_index('sample_id')
    frame['patient_id'] = frame.gsm.map(mapping.patient_id)
    frame['main_topic0'] = main_theta[:, 0]
    outcome = frame.loc[frame.arm.isin(['DFU-healer', 'DFU-nonhealer'])]
    patients = outcome.groupby('patient_id', sort=True).agg(
        main=('main_topic0', 'mean'), deterministic=('topic_0', 'mean'), arm=('arm', 'first'))
    healed = patients.arm.eq('DFU-healer').to_numpy()
    result['patient_all_cell_comparison'] = {
        'grouping': 'All retained cells pooled within each author-mapped patient; patients weighted equally in outcome contrast.',
        'main_theta': contrast(patients.main, healed),
        'deterministic_parquet': contrast(patients.deterministic, healed),
        'rows': patients.reset_index().to_dict('records'),
    }
    samples = outcome.groupby('gsm', sort=True).agg(main=('main_topic0', 'mean'), deterministic=('topic_0', 'mean'), arm=('arm', 'first'))
    healed_s = samples.arm.eq('DFU-healer').to_numpy()
    result['specimen_all_cell_comparison'] = {
        'main_theta': contrast(samples.main, healed_s),
        'deterministic_parquet': contrast(samples.deterministic, healed_s),
    }
    # Independent implementation of the threshold-free patient statistic.
    fibro = outcome[outcome.celltype.eq('fibroblast')]
    arrays = [g.topic_0.to_numpy(float) for _, g in fibro.groupby('patient_id', sort=True)]
    support = np.unique(np.concatenate(arrays))
    cdf = np.array([np.searchsorted(np.sort(a), support, side='right')/len(a) for a in arrays])
    observed = float(np.max(np.abs(cdf[healed].mean(0)-cdf[~healed].mean(0))))
    maxima = []
    for inds in itertools.combinations(range(11), 7):
        h = np.isin(np.arange(11), inds)
        maxima.append(float(np.max(np.abs(cdf[h].mean(0)-cdf[~h].mean(0)))))
    result['independent_patient_ecdf_check'] = {'statistic': observed, 'exact_p': float(np.mean(np.array(maxima)>=observed-1e-12)), 'allocations': len(maxima), 'distinct_loading_values': len(support)}
    fractions = fibro.groupby(['gsm', 'arm'], sort=True).topic_0.apply(lambda x: float((x >= .18429584801197052).mean())).reset_index(name='fraction')
    hf = fractions.arm.eq('DFU-healer').to_numpy()
    xf = fractions.fraction.to_numpy()
    df = float(xf[hf].mean()-xf[~hf].mean())
    null_f = []
    for inds in itertools.combinations(range(14), 9):
        h = np.isin(np.arange(14), inds)
        null_f.append(float(xf[h].mean()-xf[~h].mean()))
    result['dfu_only_specimen_permutation'] = {
        'scope': 'Historical sample-unit diagnostic only; repeated patients prevent treating this as primary inference.',
        'observed_high_fraction_difference': df, 'allocations': len(null_f),
        'extreme': int(np.sum(np.abs(null_f)>=abs(df)-1e-12)),
        'exact_p': float(np.mean(np.abs(null_f)>=abs(df)-1e-12)),
        'existing_fig6D_control': 'Its 20,000 permutations shuffle all 25 three-arm labels, including healthy specimens; p=0.13575 refers to a different null.',
    }
    # Saved specimen outputs expose gate dependence without reloading the atlas.
    anatomy = pd.read_csv(paths['anatomy_samples'])
    lineage = frame.groupby('gsm').agg(n_all=('topic_0', 'size'))
    fib_counts = frame[frame.celltype.eq('fibroblast')].groupby('gsm').topic_0.agg(['size', 'mean'])
    gate = anatomy.merge(lineage, on='gsm').merge(fib_counts, on='gsm')
    assert np.array_equal(gate.n_cells, gate.n_all)
    gate['relative_fibro_count_change'] = gate.n_fibroblasts/gate['size']-1
    result['gate_context_comparison'] = {
        'same_total_cells_per_specimen': True,
        'discovery_context': '25 selected foot specimens',
        'anatomy_context': 'all mapped foot and forearm specimens; gene standardization recalculated',
        'maximum_absolute_relative_fibro_count_change': float(np.max(abs(gate.relative_fibro_count_change))),
        'rows': gate[['gsm','n_cells','size','n_fibroblasts','mean','topic0_fibro_mean','relative_fibro_count_change']].to_dict('records'),
    }
    specimens = pd.read_csv(paths['specimen_readouts']).merge(mapping.reset_index()[['sample_id','patient_id','sample_title']], left_on='sample', right_on='sample_id')
    rep = specimens[specimens.arm.isin(['DFU-healer','DFU-nonhealer']) & specimens.patient_id.duplicated(keep=False)]
    result['same_patient_repeated_specimens'] = rep[['patient_id','sample_title','n_fib','fibroblast_mean','high_state_fraction','fibroblast_fraction']].to_dict('records')
    external = json.loads(paths['external_report'].read_text())['cohorts']
    result['external_sensitivity_checks'] = {}
    for name, values in external.items():
        d = pd.DataFrame(values['samples'])
        x, y = 'topic0_fibro_mean', 'immune_fraction_exclusive'
        adequate = d[d.panel_fraction > .9]
        proxy = d.groupby('patient_proxy')[[x,y]].mean()
        result['external_sensitivity_checks'][name] = {
            'n_specimens': len(d), 'panel_fraction_range': [float(d.panel_fraction.min()),float(d.panel_fraction.max())],
            'all_specimens_rho': float(spearmanr(d[x],d[y]).statistic),
            'panel_fraction_above_09_n':len(adequate), 'panel_fraction_above_09_rho':float(spearmanr(adequate[x],adequate[y]).statistic),
            'title_proxy_n':len(proxy), 'title_proxy_mean_rho':float(spearmanr(proxy[x],proxy[y]).statistic),
            'proxy_warning':'Title-proxy collapse is only a sensitivity; patient identity is not verified.',
            'leave_one_specimen_rho_range':[float(min(spearmanr(d.loc[d.gsm.ne(g),x],d.loc[d.gsm.ne(g),y]).statistic for g in d.gsm)),float(max(spearmanr(d.loc[d.gsm.ne(g),x],d.loc[d.gsm.ne(g),y]).statistic for g in d.gsm))],
        }
    # A deliberately constructed counterexample, not a fitted alternative model
    # or a formal test: every beta below has one boundary mode at zero because
    # alpha < 1 and beta > 1. There are no latent high/low populations.
    ordered = specimens.sort_values('sample')
    means = ordered.fibroblast_mean.to_numpy(float)
    ns = ordered.n_fib.to_numpy(int)
    rng = np.random.default_rng(20260922)
    correlations = []
    for _ in range(1000):
        simulated = [rng.beta(2*m, 2*(1-m), size=n) for m, n in zip(means, ns)]
        correlations.append(float(spearmanr([a.mean() for a in simulated], [(a>.18429584801197052).mean() for a in simulated]).statistic))
    result['exploratory_nonmixture_counterexample'] = {
        'purpose': 'Existence counterexample to using mean/fraction correlation as evidence for a discrete mixture; not a fit to the observed distributions, not a p value, not confirmatory model comparison.',
        'distribution': 'One continuous Beta(alpha=2*observed_specimen_mean, beta=2*(1-observed_specimen_mean)) distribution per specimen, using its observed fibroblast count.',
        'alpha_range': [float(min(2*means)), float(max(2*means))],
        'beta_range': [float(min(2*(1-means))), float(max(2*(1-means)))],
        'all_distributions_monotone_with_single_boundary_mode_zero': bool(max(2*means)<1 and min(2*(1-means))>1),
        'simulations': 1000, 'seed': 20260922,
        'rho_quantiles_025_50_975': np.quantile(correlations, [.025,.5,.975]).tolist(),
    }
    result['affected_outputs'] = [
        'Figure 4B; patient-score, exact arithmetic, and equivalence tables; Results all-cell contrast and uncertainty',
        'Figures 7A/7B and 8A topic-readout branch, permutation probabilities and comparisons; ranking AUC may remain unchanged while permutation distributions change',
        'Figure 6E structureless-control real_theta branch and topic-concentration statistic',
        'Conditional design simulations read legacy sample-score variance; require regeneration if deterministic target is chosen',
        'New Figure 5 patient-fibroblast contrasts and ECDF use deterministic parquet and are not directly invalidated',
    ]
    result['probe_sha256'] = sha(__file__)
    destination = Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(destination), 'artifact_difference': result['whole_artifact_comparison']['main_vs_lineage_theta'], 'raw_reprojection': {k:v for k,v in result['single_specimen_raw_reprojection'].items() if k.startswith('deterministic_vs')}, 'patient_contrasts': {k:v for k,v in result['patient_all_cell_comparison'].items() if k in ['main_theta','deterministic_parquet']}, 'specimen_contrasts': result['specimen_all_cell_comparison'], 'ecdf':result['independent_patient_ecdf_check']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
