#!/usr/bin/env python3
"""Rebuild lightweight measurement statistics from one deterministic projection.

Historical artifacts are read only. The encoder is never trained or run here.
The cell input is the existing deterministic lineage parquet, checked against
the saved observation order and the independently recorded raw-count probe.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t, ttest_ind, wasserstein_distance
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_measurement_math import equivalence_boundary, exhaustive_contrast, state_summary
DEFAULT_OUTPUT = ROOT / 'outputs/scientific_revision_20260922/measurement'
TOPICS = [f'topic_{i}' for i in range(15)]
METHODS = ['topic_simplex_theta0', 'module_score', 'pca', 'nmf']
LABELS = dict(zip(METHODS, ['Fibroblast topic', 'Module', 'PCA', 'NMF']))
INPUTS = {'projection': ROOT / 'outputs/celltype_resolved/GSE165816_discovery_theta.parquet', 'obs': ROOT / 'outputs/expression_representation/obs.csv', 'legacy_theta': ROOT / 'outputs/expression_representation/theta.npy', 'model': ROOT / 'outputs/expression_representation/topic_model.pt', 'panel': ROOT / 'outputs/expression_representation/panel.npy', 'beta': ROOT / 'outputs/expression_representation/beta.npy', 'map': ROOT / 'cohort_metadata/GSE165816_ncomms_subject_sample_map.csv', 'map_contract': ROOT / 'outputs/cohort_metadata/gse165816_author_map_contract.json', 'probe': ROOT / 'outputs/scientific_review_20260922/measurement_projection_probe.json', 'probe_script': ROOT / 'outputs/scientific_review_20260922/measurement_projection_probe.py', 'mixture': ROOT / 'outputs/bimodal_stratification/report.json', 'sample_effect': ROOT / 'outputs/analysis/patient_mapping_equivalence/report.json', 'patient_effect': ROOT / 'outputs/cohort_metadata/gse165816_patient_unit_remap/report.json', 'sample_benchmark': ROOT / 'outputs/representation_benchmark/report.json', 'sample_scores': ROOT / 'outputs/representation_benchmark/sample_scores.csv', 'sample_inference': ROOT / 'outputs/analysis/representation_inference/report.json', 'patient_benchmark': ROOT / 'outputs/analysis/patient_unit_representation_benchmark/report.json', 'patient_scores': ROOT / 'outputs/analysis/patient_unit_representation_benchmark/patient_scores.csv', 'math': ROOT / 'outputs/mathematical_audit/measurement/report.json', 'robustness': ROOT / 'outputs/robustness_extensions/report.json', 'design': ROOT / 'outputs/analysis/design_power_simulation/report.json', 'design_script': ROOT / 'scripts/simulate_future_cohort_designs.py', 'math_script': ROOT / 'scripts/audit_measurement_math.py', 'inference_table': ROOT / 'manuscripts/tables/representation_inference.csv', 'patient_table': ROOT / 'manuscripts/tables/patient_unit_and_anatomy.csv', 'cross_table': ROOT / 'manuscripts/tables/representation_cross_method.csv'}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def validated_cells(projection, obs, mapping, raw_identity=None):
    """Reject incompatible inputs before constructing source-order identities.

    A QC row identifies a position in the hashed input; it is not a barcode.
    Raw IDs, when supplied, are reconstructed by identical QC and source order.
    """
    if len(projection) != len(obs):
        raise ValueError('projection/observation row count mismatch')
    for a, b in ((projection.gsm, obs.gsm), (projection.arm, obs.disease)):
        if not np.array_equal(a.to_numpy(str), b.to_numpy(str)):
            raise ValueError('projection/observation GSM or arm row order mismatch')
    theta = projection[TOPICS].to_numpy(float)
    if not np.isfinite(theta).all() or (theta < 0).any():
        raise ValueError('nonfinite or negative simplex coordinate')
    if not np.allclose(theta.sum(1), 1.0, rtol=0, atol=2e-06):
        raise ValueError('projection does not lie on the simplex')
    if mapping.sample_id.duplicated().any():
        raise ValueError('author mapping contains duplicate sample IDs')
    lookup = mapping.set_index('sample_id')
    cells = projection.copy().reset_index(drop=True)
    cells['source_row_zero_based'] = np.arange(len(cells))
    cells['qc_row_within_gsm_zero_based'] = cells.groupby('gsm', sort=False).cumcount()
    cells['cell_id'] = cells.gsm.astype(str) + ':qc_row_' + cells.qc_row_within_gsm_zero_based.astype(str)
    for key in ('patient_id', 'sample_title', 'healing_status', 'group', 'tissue_site'):
        cells[key] = cells.gsm.map(lookup[key])
        required = ~cells.arm.eq('Non-diabetic') if key == 'healing_status' else np.ones(len(cells), bool)
        if cells.loc[required, key].isna().any():
            raise ValueError(f'unmapped {key}')
    if (cells.groupby('patient_id').arm.nunique() > 1).any():
        raise ValueError('one patient has incompatible outcome arms')
    if not cells.cell_id.is_unique:
        raise ValueError('cell identity is not unique')
    if raw_identity is not None:
        raw = raw_identity.reset_index(drop=True)
        aliases = {'barcode': 'raw_barcode', 'sample_raw_row': 'raw_row_0', 'qc_row_in_sample': 'qc_row_0'}
        for target, source in aliases.items():
            if target not in raw and source in raw:
                raw[target] = raw[source]
        if len(raw) != len(cells):
            raise ValueError('raw identity row count mismatch')
        for key in ('gsm', 'arm'):
            if not np.array_equal(raw[key].to_numpy(str), cells[key].to_numpy(str)):
                raise ValueError(f'raw identity {key} order mismatch')
        if not np.array_equal(raw.qc_row_in_sample, cells.qc_row_within_gsm_zero_based):
            raise ValueError('raw identity QC row order mismatch')
        if 'saved_row_0' in raw and (not np.array_equal(raw.saved_row_0, cells.source_row_zero_based)):
            raise ValueError('raw identity saved row order mismatch')
        if 'historical_celltype' in raw and (not np.array_equal(raw.historical_celltype, cells.celltype)):
            raise ValueError('raw reconstructed marker labels differ')
        for _, group in raw.groupby('gsm', sort=False):
            if np.any(np.diff(group.sample_raw_row.to_numpy()) <= 0):
                raise ValueError('raw pre-QC rows must retain increasing source order within GSM')
        cells['reconstructed_barcode'] = raw.barcode.astype(str)
        cells['sample_raw_row_zero_based'] = raw.sample_raw_row.to_numpy()
        cells['raw_archive_member'] = raw.raw_member.to_numpy()
        cells['reconstructed_raw_cell_id'] = cells.gsm.astype(str) + ':' + cells.reconstructed_barcode
        if not cells.reconstructed_raw_cell_id.is_unique:
            raise ValueError('reconstructed GSM/barcode IDs are not unique')
    return cells

def aggregate_scores(cells):
    frame = cells.copy()
    frame['score'] = frame.topic_0.to_numpy(np.float64)
    samples = frame.groupby('gsm', sort=True).agg(title=('sample_title', 'first'), disease=('arm', 'first'), patient_id=('patient_id', 'first'), healing_status=('healing_status', 'first'), score=('score', 'mean'), n_cells=('score', 'size')).reset_index()
    patients = []
    for pid, block in frame.groupby('patient_id', sort=True):
        specimens = samples[samples.patient_id.eq(pid)].sort_values('title')
        patients.append(dict(patient_id=pid, healing_status=block.healing_status.iloc[0], n_samples=len(specimens), n_cells=len(block), sample_ids='|'.join(specimens.gsm), sample_titles='|'.join(specimens.title), score_cell_weighted=float(block.score.mean()), score_sample_mean=float(specimens.score.mean())))
    return (samples, pd.DataFrame(patients))

def auc(scores, healed):
    return float(pair_credit(scores, healed).mean())

def pair_credit(scores, healed):
    scores, healed = (np.asarray(scores, float), np.asarray(healed, bool))
    a, b = (scores[healed, None], scores[None, ~healed])
    return (a > b).astype(float) + 0.5 * (a == b)

def oriented_loo(values, healed):
    values, healed = (np.asarray(values, float), np.asarray(healed, bool))
    if min(healed.sum(), (~healed).sum()) < 2:
        raise ValueError('LOO orientation requires two units in each arm')
    out = np.empty(len(values))
    for i in range(len(values)):
        keep = np.arange(len(values)) != i
        sign = 1 if values[keep & healed].mean() >= values[keep & ~healed].mean() else -1
        out[i] = sign * values[i]
    return out

def auc_permutation(values, healed):
    values, healed = (np.asarray(values, float), np.asarray(healed, bool))
    observed = auc(oriented_loo(values, healed), healed)
    null = []
    for indices in itertools.combinations(range(len(values)), int(healed.sum())):
        lab = np.zeros(len(values), bool)
        lab[list(indices)] = True
        null.append(auc(oriented_loo(values, lab), lab))
    null = np.asarray(null)
    extreme = int(np.sum(null >= observed - 1e-12))
    return (dict(n_relabelings=len(null), exact_enumeration=True, null_auc_mean=float(null.mean()), null_auc_95th_percentile=float(np.percentile(null, 95)), extreme_allocations=extreme, exact_probability=extreme / len(null), add_one_probability=(extreme + 1) / (len(null) + 1), p_one_sided=(extreme + 1) / (len(null) + 1), p_one_sided_convention='add-one comparison; exact_probability is exhaustive k/N', significant_at_0_05=bool((extreme + 1) / (len(null) + 1) < 0.05)), null)

def bootstrap_auc(scores, healed, n_boot=4000, seed=0):
    healed = np.asarray(healed, bool)
    rng = np.random.default_rng(seed)
    positive, negative = (np.flatnonzero(healed), np.flatnonzero(~healed))
    draws = np.array([np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))] for _ in range(n_boot)])
    boot = {m: np.array([auc(np.asarray(x)[d], healed[d]) for d in draws]) for m, x in scores.items()}
    report, pairs = ({}, {})
    for m, samples in boot.items():
        report[m] = dict(auc=auc(scores[m], healed), bootstrap_95_ci=np.quantile(samples, [0.025, 0.975]).tolist(), bootstrap_sd=float(samples.std(ddof=1)))
    for a, b in itertools.combinations(scores, 2):
        diff = boot[a] - boot[b]
        lo, hi = np.quantile(diff, [0.025, 0.975])
        pairs[f'{a}__minus__{b}'] = dict(observed_difference=report[a]['auc'] - report[b]['auc'], bootstrap_95_ci=[float(lo), float(hi)], ci_excludes_zero=bool(lo > 0 or hi < 0))
    return (report, pairs, draws)

def effect(values, healed, seed=0):
    values, healed = (np.asarray(values, float), np.asarray(healed, bool))
    a, b = (values[healed], values[~healed])
    permutation = exhaustive_contrast(values, healed)
    rng = np.random.default_rng(seed)
    draws = rng.choice(a, (30000, len(a))).mean(1) - rng.choice(b, (30000, len(b))).mean(1)
    welch = ttest_ind(a, b, equal_var=False)
    return dict(healer_minus_nonhealer_mean_difference=permutation['observed_difference'], welch_t=float(welch.statistic), welch_two_sided_p=float(welch.pvalue), exact_two_sided_permutation_p=permutation['exact_probability'], add_one_two_sided_permutation_p=permutation['recorded_add_one_probability'], bootstrap_95_ci=np.quantile(draws, [0.025, 0.975]).tolist(), permutation=permutation)

def equivalence(values, healed):
    values, healed = (np.asarray(values, float), np.asarray(healed, bool))
    boundary = equivalence_boundary(values[healed], values[~healed])
    delta = float(values[healed].mean() - values[~healed].mean())
    se, df = (boundary['standard_error'], boundary['welch_df'])
    margins = []
    for margin in [0.05, 0.1, 0.2, 0.3, 0.5]:
        lower, upper = (float(t.sf((delta + margin) / se, df)), float(t.cdf((delta - margin) / se, df)))
        margins.append(dict(bound=margin, p_lower=lower, p_upper=upper, welch_df=df, equivalent_at_alpha_0_05=max(lower, upper) < 0.05))
    infimum = boundary['equivalence_margin_infimum']
    return dict(status='sensitivity_grid_only', alpha=0.05, margins=margins, smallest_equivalent_margin=dict(margin=infimum, multiple_of_observed_difference=infimum / abs(delta) if delta else None, note='Infimum M*=|difference|+t_(.95,Welch df)*SE; strict TOST p<.05 requires M>M*. This is not a clinical margin.'), exact_boundary=boundary, interpretation='Exploratory discovery-cohort precision sensitivity; no independently specified clinical equivalence margin.')

def revision_note(source_key, recomputed, reused=None):
    return dict(source_report=str(INPUTS[source_key].relative_to(ROOT)), source_sha256=sha256(INPUTS[source_key]), projection_source=str(INPUTS['projection'].relative_to(ROOT)), projection_sha256=sha256(INPUTS['projection']), recomputed=recomputed, reused=reused or [], script_sha256=sha256(Path(__file__)), no_model_training=True)

def revised_report(source_key, recomputed, reused=None):
    result = read_json(INPUTS[source_key])
    result['reference_protocol'] = result.pop('protocol', {})
    historical_keys = {'elapsed_seconds', 'checked_at_utc', 'source_sha256', 'discovery_dir', 'series_matrix', 'map_csv', 'contract_report', 'tar'}
    result['protocol'] = {key: value for key, value in result['reference_protocol'].items() if key not in historical_keys and (not key.endswith('sha256'))}
    result['protocol'].update(script_sha256=sha256(Path(__file__)), computation_dtype='float64', seed=0, deterministic_projection=True, projection_source=str(INPUTS['projection'].relative_to(ROOT)), projection_sha256=sha256(INPUTS['projection']), label_exchangeability='assumed; observational labels were not randomized')
    result['projection_revision'] = revision_note(source_key, recomputed, reused)
    return result

def build_effect_reports(cells, output):
    all_samples, all_patients = aggregate_scores(cells)
    samples = all_samples[all_samples.disease.isin(['DFU-healer', 'DFU-nonhealer'])].reset_index(drop=True)
    patients = all_patients[all_patients.healing_status.isin(['healed', 'not_healed'])].reset_index(drop=True)
    if len(samples) != 14 or len(patients) != 11 or patients.healing_status.eq('healed').sum() != 7:
        raise ValueError('unexpected 14-specimen/11-patient unit')
    h_sample = samples.disease.eq('DFU-healer').to_numpy()
    h_patient = patients.healing_status.eq('healed').to_numpy()
    sample = revised_report('sample_effect', ['all sample score effects, bootstrap, TOST'])
    sample['patient_level_inference_available'] = True
    sample['patient_mapping'] = dict(status='available', source=str(INPUTS['map'].relative_to(ROOT)), patient_level_inference_available=True, note='This report retains the historical specimen-unit diagnostic; 14 specimens belong to 11 patients.')
    sample['effect'] = effect(samples.score, h_sample)
    sample['equivalence'] = equivalence(samples.score, h_sample)
    sample['protocol'].update(exact_permutation_convention='exhaustive k/N; add-one stored separately', n_boot=30000, cell_to_sample_aggregation='float64 arithmetic mean within GSM')
    sample['scientific_scope'] = 'Historical specimen-unit sensitivity with repeated patients; primary inference uses 11 author-mapped patients.'
    write_json(output / 'patient_mapping_equivalence/report.json', sample)
    samples['tissue'] = 'Foot skin'
    samples['title_prefix'] = samples.title
    write_csv(output / 'patient_mapping_equivalence/sample_scores.csv', samples)
    patient = revised_report('patient_effect', ['cell-weighted and equal-specimen patient effects, bootstrap, TOST'])
    patient['effect_cell_weighted'] = effect(patients.score_cell_weighted, h_patient)
    patient['effect_unweighted_sample_means'] = effect(patients.score_sample_mean, h_patient)
    patient['effect_unweighted_sample_means']['note'] = 'Mean of specimen means within patient, then equal patient weighting; secondary estimand.'
    patient['equivalence'] = equivalence(patients.score_cell_weighted, h_patient)
    patient['sample_level_reference_not_rewritten'] = dict(healer_minus_nonhealer_mean_difference=sample['effect']['healer_minus_nonhealer_mean_difference'], source=str((output / 'patient_mapping_equivalence/report.json').relative_to(ROOT)), note='Independent revision; the historical source report remains unchanged.')
    patient['protocol'].update(exact_permutation_convention='exhaustive k/N; add-one stored separately', n_boot=30000)
    write_json(output / 'patient_unit_remap/report.json', patient)
    write_csv(output / 'patient_unit_remap/patient_scores.csv', patients)
    write_csv(output / 'projection/specimen_scores.csv', all_samples)
    write_csv(output / 'projection/patient_scores_all_arms.csv', all_patients)
    return (samples, patients, sample, patient)

def build_benchmarks(samples, patients, cells, legacy_theta, obs, output):
    full_scores = pd.read_csv(INPUTS['sample_scores'])
    positions = samples.set_index('gsm')
    if set(full_scores.gsm) != set(samples.gsm):
        raise ValueError('benchmark sample identifiers differ')
    full_scores['topic_simplex_theta0'] = full_scores.gsm.map(positions.score)
    h = full_scores.y_healer.to_numpy(bool)
    if not np.array_equal(h, full_scores.gsm.map(positions.disease).eq('DFU-healer')):
        raise ValueError('benchmark outcome order differs')
    topic = full_scores.topic_simplex_theta0.to_numpy()
    topic_loo = oriented_loo(topic, h)
    bench = revised_report('sample_benchmark', ['topic effects and LOSO AUC', 'all full-score cross-method correlations'], ['module/PCA/NMF fits, scores, summary and stability'])
    bench['patient_level_caveat'] = '14 specimens from 11 author-mapped patients; this sample-unit comparison is a historical sensitivity analysis.'
    bench['patient_level_inference_available'] = True
    descriptive = effect(topic, h)
    pooled = np.sqrt(((h.sum() - 1) * topic[h].var(ddof=1) + ((~h).sum() - 1) * topic[~h].var(ddof=1)) / (len(h) - 2))
    delta = descriptive['healer_minus_nonhealer_mean_difference']
    bench['summary'][METHODS[0]] = dict(cv=[dict(auc=auc(topic_loo, h), n=len(h), n_healer=int(h.sum()), n_nonhealer=int((~h).sum()))], full_sample_descriptive=[dict(mean_difference_healer_minus_nonhealer=abs(delta), cohens_d=abs(delta) / pooled, exact_two_sided_permutation_p=descriptive['exact_two_sided_permutation_p'], add_one_two_sided_permutation_p=descriptive['add_one_two_sided_permutation_p'], orientation_sign=1.0 if delta >= 0 else -1.0, absolute_mean_difference=abs(delta))], full_fit_details=[{}])
    vectors = {m: full_scores[m if m in full_scores else f'{m}_seed0'].to_numpy() for m in METHODS}
    bench['cross_method_spearman_representative'] = {f'{a}__{b}': float(spearmanr(vectors[a], vectors[b]).statistic) for a, b in itertools.combinations(METHODS, 2)}
    write_csv(output / 'representation_benchmark/sample_scores.csv', full_scores)
    write_json(output / 'representation_benchmark/report.json', bench)
    old_topic = np.array([legacy_theta[obs.gsm.eq(gsm).to_numpy()].mean(0)[0] for gsm in full_scores.gsm])
    recorded_topic = pd.read_csv(INPUTS['sample_scores']).topic_simplex_theta0.to_numpy()
    if not np.allclose(old_topic, recorded_topic, atol=1e-07, rtol=0):
        raise ValueError('legacy topic reconstruction disagrees with saved benchmark scores')
    old_loo = oriented_loo(old_topic, h)
    old_credit, new_credit = (pair_credit(old_loo, h), pair_credit(topic_loo, h))
    invariant = bool(np.array_equal(old_credit, new_credit))
    if not invariant:
        raise ValueError('sample AUC rank invariance failed; other OOF score vectors are required before paired intervals can be reused')
    proof = dict(old_new_cross_arm_pair_credits_identical=invariant, n_healing_nonhealing_pairs=int(old_credit.size), old_loo_auc=auc(old_loo, h), new_loo_auc=auc(topic_loo, h), legacy_score_reconstruction_max_abs=float(np.max(np.abs(old_topic - recorded_topic))), statement='AUC is the weighted mean of cross-arm pair credits. Identical credit matrices imply identical topic AUC for every arm-stratified resampling index multiset, hence identical topic-minus-other AUC bootstrap distributions when other scores and indices are unchanged.', limitation='Other sample OOF vectors were not saved. Their fitted results and paired intervals are reused from the hashed report, supported by this invariance proof; full-sample scores are never substituted for OOF scores.')
    rows = []
    for i, pi in enumerate(np.flatnonzero(h)):
        for j, ni in enumerate(np.flatnonzero(~h)):
            rows.append(dict(healed_gsm=full_scores.gsm.iloc[pi], not_healed_gsm=full_scores.gsm.iloc[ni], old_credit=old_credit[i, j], new_credit=new_credit[i, j]))
    write_csv(output / 'representation_inference/topic_auc_pair_invariance.csv', pd.DataFrame(rows))
    write_csv(output / 'representation_inference/topic_loso_scores.csv', pd.DataFrame(dict(gsm=full_scores.gsm, y_healer=h.astype(int), old_score=old_loo, deterministic_score=topic_loo)))
    inference = revised_report('sample_inference', ['topic LOSO AUC/bootstrap and complete 2002-label permutation'], ['other representations and permutations; paired AUC intervals retained by proven topic cross-arm rank invariance'])
    topic_boot, _, sample_draws = bootstrap_auc({METHODS[0]: topic_loo}, h)
    b = topic_boot[METHODS[0]]
    if not np.allclose(b['bootstrap_95_ci'], inference['auc'][METHODS[0]]['bootstrap_95_ci'], atol=1e-12):
        raise ValueError('recomputed sample AUC interval did not match rank-invariance expectation')
    inference['auc'][METHODS[0]] = dict(loso_auc=b['auc'], bootstrap_95_ci=b['bootstrap_95_ci'], bootstrap_sd=b['bootstrap_sd'], ci_includes_0_5=b['bootstrap_95_ci'][0] <= 0.5 <= b['bootstrap_95_ci'][1])
    inference['permutation_null'][METHODS[0]], sample_null = auc_permutation(topic, h)
    inference['projection_revision']['sample_bootstrap_invariance_proof'] = proof
    inference['patient_level_inference_available'] = True
    inference['scientific_scope'] = 'Specimen-unit diagnostic; repeated patients preclude independent-specimen interpretation.'
    inference['protocol'].update(n_boot=4000, permutation_probability_convention='AUC p_one_sided retains add-one; exact_probability also reported for repaired topic')
    write_json(output / 'representation_inference/report.json', inference)
    np.save(output / 'representation_inference/topic_auc_permutation_null.npy', sample_null)
    np.save(output / 'representation_inference/bootstrap_indices.npy', sample_draws)
    patient_scores = pd.read_csv(INPUTS['patient_scores'])
    lookup = patients.set_index('patient_id')
    if not patient_scores.patient_id.is_unique or set(patient_scores.patient_id) != set(patients.patient_id):
        raise ValueError('patient benchmark IDs differ')
    hp = patient_scores.healing_status.eq('healed').to_numpy()
    if not np.array_equal(hp, patient_scores.patient_id.map(lookup.healing_status).eq('healed')):
        raise ValueError('patient benchmark outcome order differs')
    xp = patient_scores.patient_id.map(lookup.score_cell_weighted).to_numpy()
    patient_scores['topic_simplex_theta0_lopo'] = oriented_loo(xp, hp)
    scores = {m: patient_scores[f'{m}_lopo'].to_numpy() for m in METHODS}
    patient_bench = revised_report('patient_benchmark', ['topic LOPO scores', 'all 4000 shared-index AUC bootstraps/pairs', 'topic exhaustive 330-label null'], ['module/PCA/NMF saved OOF scores and permutation nulls', 'unchanged deterministic fibroblast mixture semantics'])
    patient_bench['auc'], patient_bench['pairwise_auc_difference'], patient_draws = bootstrap_auc(scores, hp)
    patient_bench['permutation_null'][METHODS[0]], patient_null = auc_permutation(xp, hp)
    patient_bench['protocol'].update(n_boot=4000, permutation_probability_convention='AUC p_one_sided retains add-one; exact_probability also reported for repaired topic')
    patient_bench['projection_revision']['reused_non_topic_oof_scores'] = dict(source=str(INPUTS['patient_scores'].relative_to(ROOT)), sha256=sha256(INPUTS['patient_scores']), methods=METHODS[1:], join='unique author-mapped patient_id; outcome order verified')
    write_csv(output / 'patient_unit_representation_benchmark/patient_scores.csv', patient_scores)
    write_json(output / 'patient_unit_representation_benchmark/report.json', patient_bench)
    np.save(output / 'patient_unit_representation_benchmark/topic_auc_permutation_null.npy', patient_null)
    np.save(output / 'patient_unit_representation_benchmark/bootstrap_indices.npy', patient_draws)
    return (bench, inference, patient_bench)

def build_math_and_influence(cells, patients, samples, output, cut):
    fibro = cells[cells.celltype.eq('fibroblast')]
    x = fibro.topic_0.to_numpy(float)
    low, high = (float(x[x <= cut].mean()), float(x[x > cut].mean()))
    rows = [dict(specimen=gsm, **state_summary(block.topic_0, cut, low, high)) for gsm, block in fibro.groupby('gsm', sort=True)]
    report = revised_report('math', ['state algebra from unified input', 'patient exact permutation and analytic Welch/TOST boundary'])
    report['state_decomposition'] = dict(n_specimens=len(rows), n_fibroblasts=len(fibro), cut=cut, reference='Cell-pooled means of the two threshold bins; not fitted Gaussian means', low_reference=low, high_reference=high, empty_bin_specimens=sum((row['low_state_mean'] is None or row['high_state_mean'] is None for row in rows)), ranges={key: [min((row[key] for row in rows if row[key] is not None)), max((row[key] for row in rows if row[key] is not None))] for key in ('low_state_mean', 'high_state_mean', 'within_state_residual')}, specimen_weighted_residual_rms=float(np.sqrt(np.mean([row['within_state_residual'] ** 2 for row in rows]))), max_identity_error=max((row['identity_error'] for row in rows)), max_residual_identity_error=max((row['residual_identity_error'] for row in rows)), specimens=rows)
    h = patients.healing_status.eq('healed').to_numpy()
    values = patients.score_cell_weighted.to_numpy()
    report['patient_permutation'] = exhaustive_contrast(values, h)
    report['patient_equivalence'] = equivalence_boundary(values[h], values[~h])
    hs = samples.disease.eq('DFU-healer').to_numpy()
    report['specimen_permutation'] = exhaustive_contrast(samples.score, hs)
    report['specimen_equivalence'] = equivalence_boundary(samples.score[hs], samples.score[~hs])
    write_csv(output / 'mathematical_audit/state_decomposition.csv', pd.DataFrame(rows))
    write_json(output / 'mathematical_audit/report.json', report)
    robustness = revised_report('robustness', ['all-cell leave-one-patient-out outcome differences'], ['deterministic fibroblast patient semantics', 'independent temporal analyses'])
    omission = []
    for i, pid in enumerate(patients.patient_id):
        keep = np.arange(len(patients)) != i
        omission.append(dict(omitted_patient=pid.rsplit('_', 1)[-1], healing_minus_nonhealing=float(values[h & keep].mean() - values[~h & keep].mean())))
    robustness['patient']['patient_outcome_influence'] = dict(n_patients=len(values), observed_difference=float(values[h].mean() - values[~h].mean()), leave_one_patient_out=omission, omission_range=[min((r['healing_minus_nonhealing'] for r in omission)), max((r['healing_minus_nonhealing'] for r in omission))], interpretation='Influence diagnostic; no interval, significance test or independent validation.')
    robustness['patient']['source_fingerprints'][str((output / 'patient_unit_remap/patient_scores.csv').relative_to(ROOT))] = sha256(output / 'patient_unit_remap/patient_scores.csv')
    write_json(output / 'robustness_extensions/report.json', robustness)
    write_csv(output / 'robustness_extensions/patient_omission.csv', pd.DataFrame(omission))
    return (report, robustness)

def distribution_distances(a, b, cut):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        raise ValueError('distribution distance requires nonempty halves')
    support = np.unique(np.r_[a, b])
    sup = np.max(np.abs(np.searchsorted(np.sort(a), support, side='right') / len(a) - np.searchsorted(np.sort(b), support, side='right') / len(b)))
    return dict(absolute_mean_difference=abs(float(a.mean() - b.mean())), absolute_high_fraction_difference=abs(float((a > cut).mean() - (b > cut).mean())), wasserstein_distance=float(wasserstein_distance(a, b)), ecdf_supremum=float(sup))

def repeat_diagnostics(cells, output, cut, n_split=250):
    lookup = cells.groupby('gsm', sort=True)
    summaries = []
    within_rows = []
    for gsm, block in lookup:
        fibro = block.loc[block.celltype.eq('fibroblast'), 'topic_0'].to_numpy(float)
        summaries.append(dict(gsm=gsm, sample_title=block.sample_title.iloc[0], patient_id=block.patient_id.iloc[0], arm=block.arm.iloc[0], n_all=len(block), n_fibroblasts=len(fibro), all_cell_mean=float(block.topic_0.to_numpy(float).mean()), fibroblast_mean=float(fibro.mean()), high_state_fraction=float((fibro > cut).mean()), fibroblast_composition=len(fibro) / len(block)))
    table = pd.DataFrame(summaries)
    repeats = table[table.patient_id.duplicated(keep=False)]
    rng = np.random.default_rng(20260922)
    within = {}
    for specimen in repeats.itertuples(index=False):
        block = lookup.get_group(specimen.gsm)
        array = block.loc[block.celltype.eq('fibroblast'), 'topic_0'].to_numpy(float)
        draws = []
        for i in range(n_split):
            ordering = rng.permutation(len(array))
            midpoint = len(array) // 2
            values = distribution_distances(array[ordering[:midpoint]], array[ordering[midpoint:]], cut)
            draws.append(values)
            within_rows.append(dict(gsm=specimen.gsm, draw=i, **values))
        within[specimen.gsm] = {key: dict(median=float(np.median([row[key] for row in draws])), central_95_range=np.quantile([row[key] for row in draws], [0.025, 0.975]).tolist()) for key in draws[0]}
    pairs = []
    for patient, block in repeats.groupby('patient_id', sort=True):
        for (_, a), (_, b) in itertools.combinations(block.sort_values('sample_title').iterrows(), 2):
            x = lookup.get_group(a.gsm)
            y = lookup.get_group(b.gsm)
            observed = distribution_distances(x.loc[x.celltype.eq('fibroblast'), 'topic_0'], y.loc[y.celltype.eq('fibroblast'), 'topic_0'], cut)
            row = dict(patient_id=patient, arm=a.arm, gsm_a=a.gsm, gsm_b=b.gsm, sample_a=a.sample_title, sample_b=b.sample_title, n_fibroblasts_a=int(a.n_fibroblasts), n_fibroblasts_b=int(b.n_fibroblasts), mean_a=float(a.fibroblast_mean), mean_b=float(b.fibroblast_mean), high_fraction_a=float(a.high_state_fraction), high_fraction_b=float(b.high_state_fraction), composition_a=float(a.fibroblast_composition), composition_b=float(b.fibroblast_composition), absolute_all_cell_mean_difference=abs(float(a.all_cell_mean - b.all_cell_mean)), absolute_composition_difference=abs(float(a.fibroblast_composition - b.fibroblast_composition)), **observed)
            for key, value in observed.items():
                for suffix, gsm in (('a', a.gsm), ('b', b.gsm)):
                    row[f'{key}_split_half_{suffix}_median'] = within[gsm][key]['median']
                    row[f'{key}_split_half_{suffix}_upper975'] = within[gsm][key]['central_95_range'][1]
                reference = max(within[a.gsm][key]['central_95_range'][1], within[b.gsm][key]['central_95_range'][1])
                row[f'{key}_exceeds_both_split_half_upper975'] = bool(value > reference)
            pairs.append(row)
    report = dict(status='completed', n_repeated_pairs=len(pairs), n_repeated_patients=int(repeats.patient_id.nunique()), n_dfu_repeated_patients=sum((row['arm'] != 'Non-diabetic' for row in pairs)), cut=cut, paired_specimens=pairs, within_specimen_split_half=within, interpretation='Between-specimen disagreement is compared with conditional cell-partition variability in each specimen. Split halves are not biological replicates, confidence intervals, or a null distribution for a patient-level significance test. This is not an ICC or a clinical reliability estimate.', limitations=['Only five repeat pairs, including three DFU patients; anatomical microheterogeneity, preparation and annotation differences remain confounded.', "Random disjoint fibroblast halves preserve each specimen's own distribution and use half its cell count; their central range is a finite-cell sampling diagnostic.", 'A stable mean across repeated cell partitions cannot establish specimen or assay repeatability.'], protocol=dict(seed=20260922, split_half_draws=n_split, split_half_without_replacement=True, source_sha256=sha256(INPUTS['projection']), map_sha256=sha256(INPUTS['map']), script_sha256=sha256(Path(__file__))))
    write_csv(output / 'repeat_specimens/specimen_scores.csv', table)
    write_csv(output / 'repeat_specimens/paired_specimens.csv', pd.DataFrame(pairs))
    write_csv(output / 'repeat_specimens/split_half_draws.csv', pd.DataFrame(within_rows))
    write_json(output / 'repeat_specimens/report.json', report)
    return report

def build_design(output):
    with (output / 'design_power_simulation.log').open('w') as log_file:
        subprocess.run([sys.executable, str(INPUTS['design_script']), '--scores', str(output / 'representation_benchmark/sample_scores.csv'), '--benchmark', str(output / 'representation_benchmark/report.json'), '--output-dir', str(output / 'design_power_simulation')], cwd=ROOT, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    path = output / 'design_power_simulation/report.json'
    report = read_json(path)
    report['projection_revision'] = revision_note('design', ['all variance, correlation, power and design-effect scenarios'])
    report['scientific_unit'] = 'hypothetical independent unit under specimen-derived spread scenarios'
    report['question'] = 'What candidate independent-unit totals reach the simulation power target under the stated specimen-derived spread and correlation assumptions?'
    report['anchors']['source'] = 'Deterministic discovery specimen scores; repeated patients remain in these descriptive spread anchors, so they are not patient-population variance estimates.'
    report['anchors']['sd_ratio_healed_over_not_healed'] = report['anchors']['variance_ratio_healed_over_not_healed']
    report['anchors']['legacy_variance_ratio_field_is_sd_ratio'] = True
    report['anchors']['actual_variance_ratio_healed_over_not_healed'] = report['anchors']['sd_ratio_healed_over_not_healed'] ** 2
    report['assumptions'].extend(['The score rank-correlation median is used as a hypothetical latent paired-score correlation in the binormal AUC model; this is not an identified population parameter.', 'Finite Monte Carlo doubling/bisection search returns a candidate total, not a certified global minimum.', 'Patient translation uses stipulated ICC values; repeat-specimen diagnostics do not estimate ICC.'])
    report['not_supported'].append('Patient recruitment recommendations from these specimen-derived variance anchors or the stipulated ICC grid.')
    write_json(path, report)
    return report

def build_tables(output, bench, inference, patient_bench):
    table = pd.read_csv(INPUTS['inference_table'])
    for method in METHODS:
        mask = table.readout.eq(LABELS[method])
        score, null = (inference['auc'][method], inference['permutation_null'][method])
        table.loc[mask, ['auc', 'ci_low', 'ci_high', 'p_one_sided', 'null_mean']] = [score['loso_auc'], *score['bootstrap_95_ci'], null['p_one_sided'], null['null_auc_mean']]
    write_csv(output / 'tables/representation_inference.csv', table)
    table = pd.read_csv(INPUTS['patient_table'])
    for method in METHODS:
        mask = table.panel.eq('patient_unit_auc') & table.quantity.eq(LABELS[method])
        score, null = (patient_bench['auc'][method], patient_bench['permutation_null'][method])
        table.loc[mask, ['estimate', 'lower_95', 'upper_95', 'p_value']] = [score['auc'], *score['bootstrap_95_ci'], null['p_one_sided']]
    write_csv(output / 'tables/patient_unit_and_anatomy.csv', table)
    rows = [dict(readout_a=LABELS[a], readout_b=LABELS[b], spearman_rho=bench['cross_method_spearman_representative'][f'{a}__{b}']) for a, b in itertools.combinations(METHODS, 2)]
    write_csv(output / 'tables/representation_cross_method.csv', pd.DataFrame(rows))
    write_json(output / 'tables/provenance.json', dict(source_sha256={key: sha256(INPUTS[key]) for key in ('inference_table', 'patient_table', 'cross_table')}, paired_anatomy_row='Retained verbatim in numerical content from original patient/anatomy CSV; independent analysis not recalculated here.', auc_p_convention='p_one_sided and patient p_value use retained add-one AUC convention; updated all-cell mean-contrast reports instead prioritize exhaustive k/N.'))

def integration_text(output, report, sample, patient, bench, inference, patient_bench, math, robustness, repeat, design):
    p, s = (patient['effect_cell_weighted'], sample['effect'])
    pp, sp = (p['permutation'], s['permutation'])
    focal = next((row for row in repeat['paired_specimens'] if set([row['sample_a'], row['sample_b']]) == {'G7', 'G8'}))
    lines = ['# Deterministic measurement revision integration', '', 'All paths below are relative to outputs/scientific_revision_20260922/measurement/. Historical inputs are read only. Run: `python3 scripts/repair_measurement_projection.py`.', '', '## Authoritative input', '', '`projection/cells.parquet`, `projection/theta.npy`, `projection/obs.csv` and `projection/contract.json` define a single deterministic cell input. Aggregations use float64. cell_id is a source-order QC position tied to the recorded input hash, explicitly not a barcode. Optional reconstructed raw IDs retain their separate provenance.', '', 'The existing GSM5050530 raw-count probe verifies 2279 QC rows against all 15 deterministic coordinates (max absolute difference 3.5762786865234375e-07). It is evidence for the saved deterministic projection, not a claim that barcode IDs were present in historical artifacts.', '', '## Result replacements', '', f"11 patients, 7 healed/4 not healed: all-cell difference {p['healer_minus_nonhealer_mean_difference']:.12g}; 30000-draw bootstrap interval {p['bootstrap_95_ci']}; exhaustive p={pp['extreme_allocations']}/{pp['allocations']}={pp['exact_probability']:.12g}; add-one comparison={pp['recorded_add_one_probability']:.12g}.", '', f"Patient Welch SE={math['patient_equivalence']['standard_error']:.12g}, df={math['patient_equivalence']['welch_df']:.12g}; M*={math['patient_equivalence']['equivalence_margin_infimum']:.12g}. M* is the analytic infimum; strict p<.05 requires M>M*. This is not an MCID.", '', f"14 specimens (diagnostic only): difference={s['healer_minus_nonhealer_mean_difference']:.12g}; bootstrap={s['bootstrap_95_ci']}; exhaustive p={sp['extreme_allocations']}/{sp['allocations']}={sp['exact_probability']:.12g}; add-one={sp['recorded_add_one_probability']:.12g}; M*={math['specimen_equivalence']['equivalence_margin_infimum']:.12g}.", '', f"Equal-specimen-then-equal-patient difference={patient['effect_unweighted_sample_means']['healer_minus_nonhealer_mean_difference']:.12g}. Patient omission range={robustness['patient']['patient_outcome_influence']['omission_range']}.", '', f"Topic LOSO AUC={inference['auc'][METHODS[0]]['loso_auc']:.12g}; null mean={inference['permutation_null'][METHODS[0]]['null_auc_mean']:.12g}, add-one p={inference['permutation_null'][METHODS[0]]['p_one_sided']:.12g}, exhaustive p={inference['permutation_null'][METHODS[0]]['exact_probability']:.12g}.", '', f"Topic LOPO AUC={patient_bench['auc'][METHODS[0]]['auc']:.12g}; null mean={patient_bench['permutation_null'][METHODS[0]]['null_auc_mean']:.12g}, add-one p={patient_bench['permutation_null'][METHODS[0]]['p_one_sided']:.12g}, exhaustive p={patient_bench['permutation_null'][METHODS[0]]['exact_probability']:.12g}.", '', 'Sample topic cross-arm credits are exactly invariant (45/45), proving every fixed-score arm-stratified topic bootstrap AUC and paired difference is unchanged. Topic bootstrap itself is rerun; other sample OOF scores were not saved and their paired interval entries are reused with the mathematical proof in representation_inference/report.json. Patient OOF scores were saved: all patient AUC/pair bootstraps are actually recomputed from shared indices. Non-topic fits/permutation nulls are hash-referenced reuse; the encoder/PCA/NMF are not refitted.', '', f"Full-score Spearman correlations: {json.dumps(bench['cross_method_spearman_representative'])}", '', f"Design spread: healed SD={design['anchors']['sd_topic0_healed']:.12g}; not-healed SD={design['anchors']['sd_topic0_not_healed']:.12g}; median rank correlation={design['anchors']['median_between_representation_rank_correlation']:.12g}. All three simulation designs were recalculated. These are specimen-derived hypothetical spread/correlation scenarios, not estimated patient-population variances or recruitment recommendations.", '', f"Design totals: detect={json.dumps({k: v['total'] for k, v in design['design_a_detect_difference'].items()})}; equivalence={json.dumps({k: v['total'] for k, v in design['design_b_declare_equivalence'].items()})}; AUC={json.dumps({k: v['total'] for k, v in design['design_c_resolve_auc_difference'].items()})}.", '', f"Repeat G7/G8 (same patient): fibroblast means={focal['mean_a']:.12g}, {focal['mean_b']:.12g}; absolute difference={focal['absolute_mean_difference']:.12g}; high-state fractions={focal['high_fraction_a']:.12g}, {focal['high_fraction_b']:.12g}; Wasserstein={focal['wasserstein_distance']:.12g}; ECDF supremum={focal['ecdf_supremum']:.12g}. The split-half upper 97.5% mean differences are {focal['absolute_mean_difference_split_half_a_upper975']:.12g} and {focal['absolute_mean_difference_split_half_b_upper975']:.12g}. These are sampling diagnostic ranges, not biological CIs or ICC.", '', '## Compatible paths', '', '- patient_mapping_equivalence/report.json and sample_scores.csv replace the old 14-specimen files.', '- patient_unit_remap/report.json and patient_scores.csv replace the old 11-patient all-cell files.', '- representation_benchmark/report.json and sample_scores.csv replace full descriptive topic scores/correlations.', '- representation_inference/report.json replaces sample AUC inference.', '- patient_unit_representation_benchmark/report.json and patient_scores.csv replace patient AUC inference.', '- design_power_simulation/report.json replaces all old design scenarios.', '- mathematical_audit/report.json replaces patient mathematical audit; fibroblast decomposition is recomputed and unchanged within rounding.', '- robustness_extensions/report.json replaces patient all-cell omission; temporal and fibroblast-semantic branches are explicitly reused.', '- tables/representation_inference.csv, tables/patient_unit_and_anatomy.csv, tables/representation_cross_method.csv retain exporter schemas. The paired-anatomy row retains the original independent source values.', '- repeat_specimens/report.json, paired_specimens.csv, specimen_scores.csv and split_half_draws.csv provide the new repeat diagnostic.', '', '## Scope and verification', '', 'The independent 11-patient fibroblast ECDF branch already uses this deterministic source and is not replaced by the all-cell statistics. Gate/marker/context concerns require the separate controls revision. Bootstrap intervals condition on frozen scores; label enumeration assumes exchangeability in an observational cohort. No new subjects or independent validation are introduced.', '', f"Source hashes unchanged after run: {report['verification']['all_source_hashes_unchanged']}. All output JSON is finite; simplex/order checks and expected unit counts passed. See report.json for complete input/output hashes and old-to-new comparisons.", '']
    (output / 'INTEGRATION.md').write_text('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--raw-identity', type=Path, default=ROOT / 'outputs/scientific_revision_20260922/measurement_controls/raw_cell_identity.csv.gz')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output != DEFAULT_OUTPUT.resolve() and DEFAULT_OUTPUT.resolve() not in output.parents:
        raise ValueError('Outputs must remain inside the assigned measurement revision directory')
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()}
    projection, obs = (pd.read_parquet(INPUTS['projection']), pd.read_csv(INPUTS['obs']))
    mapping, legacy = (pd.read_csv(INPUTS['map']), np.load(INPUTS['legacy_theta']))
    raw = pd.read_csv(args.raw_identity, dtype={'barcode': str}) if args.raw_identity.exists() else None
    raw_manifest = None
    if raw is not None:
        source_hashes[str(args.raw_identity.relative_to(ROOT))] = sha256(args.raw_identity)
        manifest_path = args.raw_identity.parent / 'raw_cache_manifest.json'
        raw_manifest = read_json(manifest_path)
        if raw_manifest['cache_sha256'][args.raw_identity.name] != sha256(args.raw_identity):
            raise ValueError('raw identity does not match cache manifest hash')
        for key in ('projection', 'obs'):
            path = str(INPUTS[key].relative_to(ROOT))
            if raw_manifest['inputs_sha256'][path] != source_hashes[path]:
                raise ValueError('raw identity cache uses incompatible projection/obs inputs')
        source_hashes[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path)
    cells = validated_cells(projection, obs, mapping, raw)
    probe = read_json(INPUTS['probe'])
    for key in ('projection', 'obs', 'model', 'panel', 'legacy_theta'):
        path = str(INPUTS[key].relative_to(ROOT))
        if probe['source_sha256'][path] != source_hashes[path]:
            raise ValueError(f'input changed since independent probe: {path}')
    if raw is not None:
        verified = probe['single_specimen_raw_reprojection']
        selected = cells[cells.gsm.eq(verified['gsm'])]
        if selected.reconstructed_raw_cell_id.tolist() != verified['reconstructed_cell_ids_in_order']:
            raise ValueError('reconstructed barcode order differs from independent raw-count probe')
        if selected.sample_raw_row_zero_based.tolist() != verified['raw_qc_rows_zero_based']:
            raise ValueError('reconstructed pre-QC rows differ from independent raw-count probe')
    if len(cells) != 76461 or cells.gsm.nunique() != 25 or cells.patient_id.nunique() != 20:
        raise ValueError('unexpected discovery cell, specimen or patient count')
    if legacy.shape != (len(cells), len(TOPICS)):
        raise ValueError('legacy theta dimensions differ')
    (output / 'projection').mkdir(exist_ok=True)
    cells.to_parquet(output / 'projection/cells.parquet', index=False)
    np.save(output / 'projection/theta.npy', cells[TOPICS].to_numpy(np.float32))
    new_obs = obs.copy()
    for key in cells.columns:
        if key not in TOPICS and key not in new_obs:
            new_obs[key] = cells[key].to_numpy()
    write_csv(output / 'projection/obs.csv', new_obs)
    difference = legacy.astype(float) - cells[TOPICS].to_numpy(float)
    contract = dict(status='accepted', n_cells=len(cells), n_topics=len(TOPICS), n_specimens=25, n_patients=20, coordinate_definition='Frozen encoder softmax(mu), existing deterministic lineage parquet; no refit or stochastic draw', cell_id_definition='GSM:qc_row_N; zero-based within-GSM position after QC in the hashed source order, not a barcode or raw pre-QC row', raw_identity_included=raw is not None, raw_cache_manifest=raw_manifest, raw_identity_limitation='Reconstructed raw IDs, when included, join by identical QC/source order and per-GSM counts; historical artifacts did not store barcodes. The separately saved GSM5050530 probe verifies rowwise deterministic coordinates against raw input.', order_contract=dict(obs_gsm_equal=True, obs_arm_equal=True, unique_cell_ids=True, simplex_tolerance=2e-06), cell_order_sha256=sha256_bytes('\n'.join(cells.cell_id.astype(str)).encode()), raw_cell_order_sha256=sha256_bytes('\n'.join(cells.reconstructed_raw_cell_id.astype(str)).encode()) if raw is not None else None, single_gsm_raw_probe={k: probe['single_specimen_raw_reprojection'][k] for k in ('gsm', 'qc_cells', 'raw_archive_member', 'archive_member_sha256', 'deterministic_vs_lineage_parquet', 'cell_id_limitation')}, legacy_projection_difference=dict(maximum_absolute=float(np.max(abs(difference))), rms=float(np.sqrt(np.mean(difference ** 2)))), source_sha256=source_hashes, output_sha256={name: sha256(output / 'projection' / name) for name in ('cells.parquet', 'theta.npy', 'obs.csv')}, aggregation_dtype='float64', script_sha256=sha256(Path(__file__)))
    write_json(output / 'projection/contract.json', contract)
    print('Validated and wrote deterministic projection contract', flush=True)
    samples, patients, sample, patient = build_effect_reports(cells, output)
    bench, inference, patient_bench = build_benchmarks(samples, patients, cells, legacy, obs, output)
    cut = read_json(INPUTS['mixture'])['cohorts']['GSE165816_discovery']['gmm']['midpoint']
    math, robustness = build_math_and_influence(cells, patients, samples, output, cut)
    repeat = repeat_diagnostics(cells, output, cut)
    print('Rebuilt effects, AUC, mathematics, omissions, and repeat diagnostics', flush=True)
    design = build_design(output)
    build_tables(output, bench, inference, patient_bench)
    for path, expected in source_hashes.items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f'historical source changed while running: {path}')
    report = dict(status='completed', scope='SM-01 deterministic all-cell projection repair and SM-05 repeat-specimen diagnostic', source_sha256=source_hashes, script_sha256=sha256(Path(__file__)), source_identity=contract, old_to_new=dict(patient_effect=dict(old=read_json(INPUTS['patient_effect'])['effect_cell_weighted'], new=patient['effect_cell_weighted']), specimen_effect=dict(old=read_json(INPUTS['sample_effect'])['effect'], new=sample['effect']), patient_auc_null=dict(old=read_json(INPUTS['patient_benchmark'])['permutation_null'][METHODS[0]], new=patient_bench['permutation_null'][METHODS[0]]), specimen_auc_null=dict(old=read_json(INPUTS['sample_inference'])['permutation_null'][METHODS[0]], new=inference['permutation_null'][METHODS[0]])), verification=dict(all_source_hashes_unchanged=True, finite_json=True, expected_counts=True, topic_sample_bootstrap_pair_invariance=True, state_identity_error_below_1e12=math['state_decomposition']['max_identity_error'] < 1e-12), output_sha256={str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*')) if path.is_file() and path not in (output / 'report.json', output / 'INTEGRATION.md')})
    integration_text(output, report, sample, patient, bench, inference, patient_bench, math, robustness, repeat, design)
    report['output_sha256']['INTEGRATION.md'] = sha256(output / 'INTEGRATION.md')
    write_json(output / 'report.json', report)
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), patient_difference=patient['effect_cell_weighted']['healer_minus_nonhealer_mean_difference'], exact_patient_p=patient['effect_cell_weighted']['exact_two_sided_permutation_p'], raw_identity_included=raw is not None, source_hashes_unchanged=True), indent=2))
if __name__ == '__main__':
    main()
