#!/usr/bin/env python3
"""Reconstruct every fixed SDD output; no fitting, download or paper mutation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expansion.common.io import atomic_write_json, content_hash, read_json, sha256_file
from expansion.common.provenance import code_digest, code_files
from expansion.measurement import stability
from expansion.runner.jobs import method_base, read_jobs
from expansion.runner.queue import job_dir
from scripts.audit_saved_topic_null import training_nulls, fixed_distribution_crossentropy, model_crossentropy
from scripts.run_sdd_local_validation import require, stamp, topic_protocol, population_protocol, PREVIOUS, SPEC, implementation_hashes
from scripts.verify_local_convergence_followup import check_binding


def assert_metrics(saved, reconstructed, label, tolerance=2e-5):
    if isinstance(reconstructed, dict):
        require(set(saved) == set(reconstructed), f'{label} keys differ')
        return max((assert_metrics(saved[k], value, f'{label}.{k}', tolerance) for k, value in reconstructed.items()), default=0.)
    if isinstance(reconstructed, list):
        require(len(saved) == len(reconstructed), f'{label} lengths differ')
        return max((assert_metrics(a, b, label, tolerance) for a, b in zip(saved, reconstructed)), default=0.)
    if isinstance(reconstructed, (bool, str)) or reconstructed is None:
        require(saved == reconstructed, f'{label} categorical value differs')
        return 0.
    require(np.isfinite([saved, reconstructed]).all(), f'{label} is nonfinite')
    error = abs(float(saved) - float(reconstructed))
    require(error <= tolerance * max(1., abs(float(reconstructed))), f'{label} cannot be reconstructed ({error})')
    return error


def plateau_screen(evaluations, name):
    last = evaluations[-6:]
    movement = np.diff([r[name]['equal_patient_objective'] for r in last])
    return {'absolute_last_five_100step_movements': np.abs(movement).tolist(), 'threshold': .001,
        'screen_passed': len(movement) == 5 and bool(np.all(np.abs(movement) <= .001)),
        'scope': 'predeclared_engineering_screen_not_convergence_proof_no_stopping_or_selection'}


def verify_topic(run, panel_kind):
    from core_engines.topic_simplex_engine import TopicSimplexModel
    p = check_binding(run)
    require(p == topic_protocol(panel_kind), 'topic specification drift')
    fit = None
    if panel_kind == 'inherited':
        x, cells = stability.load_discovery_panel_counts(ROOT)
        panel = np.load(ROOT / stability.DISCOVERY / 'panel.npy', allow_pickle=True)
    else:
        x, cells, panel, fit = stability.load_training_panel_counts(ROOT,
            p['stability']['patient_partition'], p['stability']['panel_selection'])
    mask, selected, partition = stability.patient_partition(cells, p['stability']['patient_partition'])
    if fit is not None:
        partition['limitations'] = ['inherited_discovery_raw_gene_universe_and_cell_QC', 'supplied_patient_and_lineage_labels']
        partition['panel_fit'] = fit
    null = training_nulls(x, cells.patient_id.to_numpy(dtype=str), mask)
    original_panel = np.load(ROOT / stability.DISCOVERY / 'panel.npy', allow_pickle=True)
    reference = np.load(ROOT / stability.DISCOVERY / 'beta.npy', allow_pickle=False)
    rows, traces = [], {}
    for job in read_jobs(run / 'jobs.jsonl'):
        folder, steps = job_dir(run, job.job_id), job.params['steps']
        result = read_json(folder / 'result.json')
        loaded = stability.load_inference_checkpoint(folder, panel=panel, cell_ids=cells.reconstructed_raw_cell_id.to_numpy())
        training = loaded['metadata']['training_config']
        require(training['steps'] == steps and training['patient_partition'] == partition == result['patient_partition'],
                'topic training contract/partition mismatch')
        require(training['optimization_proportions_sha256'] == stability._matrix_hash(x[mask]), 'optimization input mismatch')
        require(loaded['metadata']['proportions_sha256'] == stability._matrix_hash(x), 'projection input mismatch')
        projection = stability.project_theta(loaded['state'], x, job.params['K'])
        require(np.max(np.abs(projection - loaded['theta'])) <= 1e-6, 'saved projection reload mismatch')
        trace, evaluation = result['training_trace'], result['evaluation_trace']
        require([r['step'] for r in trace] == list(range(1, steps + 1)), 'missing dense optimization trace')
        require([r['step'] for r in evaluation] == [0] + list(range(100, steps + 1, 100)), 'evaluation cadence drift')
        require(all(np.isfinite([r['loss'], r['recon_loss'], r['kl_div']]).all() for r in trace), 'nonfinite optimization trace')
        traces[(job.seed, steps)] = trace
        model = TopicSimplexModel(**loaded['metadata']['model_config'])
        model.load_state_dict(loaded['state'])
        reconstructed = {name: stability.evaluate_topic_model(model, x[index], cells.iloc[index].patient_id.to_numpy(dtype=str))
                         for name, index in selected.items()}
        error = max(assert_metrics(evaluation[-1][name], scores, 'topic-final-eval') for name, scores in reconstructed.items())
        try:
            match = (stability.match_topics(reference, loaded['beta']) if fit is None else
                     stability.match_panel_topics(reference, original_panel, loaded['beta'], panel))
            require(np.allclose(match['correlation'], result['match_to_reference']['correlation'], atol=1e-7, rtol=0),
                    'topic reference match mismatch')
        except ValueError as error:
            require(any(reason in str(error) for reason in ('constant topic rows', 'too few shared genes')) and
                result['match_to_reference'].get('status') == 'undefined_no_rescue', 'unexpected match failure')
            match = {'assignment': {}, 'panel_comparison': None}
        require(read_json(folder / 'fit_trace.json')['training_trace'] == trace, 'saved pre-match trace mismatch')
        assigned = match['assignment'].get(0)
        require(assigned == result['fibroblast_topic'], 'reference primary assignment differs')
        if result['primary_readout_status'] == 'available':
            values = loaded['theta'][:, assigned]
            fib = cells.celltype.eq('fibroblast').to_numpy()
            cut = stability.gmm_midpoint(values[mask & fib])
            require(abs(cut - result['cut']) <= 1e-10, 'readout cut not training-only')
            frame = pd.DataFrame({'unit': cells.loc[fib, 'patient_id'].to_numpy(),
                'specimen': cells.loc[fib, 'gsm'].to_numpy(), 'value': values[fib]})
            table = frame.groupby(['unit', 'specimen']).value.agg(mean='mean', n_cells='size', fraction=lambda v: float(np.mean(v > cut)))
            patients = table.groupby('unit').agg(mean=('mean', 'mean'), fraction=('fraction', 'mean'),
                                                n_cells=('n_cells', 'sum'), n_specimens=('mean', 'size'))
            require(len(result['patient_readouts']) == len(patients) and len(result['specimen_readouts']) == len(table),
                    'readout denominator mismatch')
            for record in result['patient_readouts']:
                for key in ('mean', 'fraction', 'n_cells', 'n_specimens'):
                    require(abs(patients.loc[record['unit'], key] - record[key]) <= 1e-10, 'patient readout mismatch')
            for record in result['specimen_readouts']:
                for key in ('mean', 'fraction', 'n_cells'):
                    require(abs(table.loc[(record['unit'], record['specimen']), key] - record[key]) <= 1e-10,
                            'specimen readout mismatch')
        require(result['semantics']['available'] is False and not partition['end_to_end_independent'], 'topic claims exceeded')
        beta, theta = loaded['beta'].astype(float), loaded['theta'].astype(float)
        cosine = np.sqrt(beta) @ np.sqrt(beta).T
        off = cosine[np.triu_indices(len(beta), 1)]
        partitions = {}
        for name, index in selected.items():
            weights, units = theta[index], cells.iloc[index].patient_id.to_numpy(dtype=str)
            scores = {'model': model_crossentropy(x[index], weights, beta),
                'uniform_decoder_mixture': fixed_distribution_crossentropy(x[index], beta.mean(0)),
                **{key: fixed_distribution_crossentropy(x[index], q) for key, q in null.items()}}
            by_patient = [{'patient_id': patient, **{key: float(score[units == patient].mean()) for key, score in scores.items()}}
                          for patient in sorted(set(units))]
            mean_scores = {key: float(np.mean([r[key] for r in by_patient])) for key in scores}
            mean_theta = weights.mean(0)
            partitions[name] = {'by_patient': by_patient, 'equal_patient_crossentropy': mean_scores,
                'model_minus_null': {key: mean_scores['model'] - value for key, value in mean_scores.items() if key != 'model'},
                'effective_mean_usage_topics': float(np.exp(-np.sum(mean_theta * np.log(mean_theta + 1e-10)))),
                'argmax_nonempty_topics': int(np.count_nonzero(np.bincount(weights.argmax(1), minlength=len(beta)))),
                'argmax_topic_counts': np.bincount(weights.argmax(1), minlength=len(beta)).tolist(),
                'per_topic_theta_std': weights.std(0).tolist(), 'plateau': plateau_screen(evaluation, name)}
        rows.append(dict(job_id=job.job_id, panel_kind=panel_kind, panel_size=len(panel), panel_identity=content_hash(panel.tolist(), 64),
            steps=steps, seed=job.seed, final_train_objective=evaluation[-1]['train']['equal_patient_objective'],
            final_held_objective=evaluation[-1]['held']['equal_patient_objective'],
            reference0_assigned_correlation=result['fibroblast_topic_correlation'],
            panel_comparison=match.get('panel_comparison'), reconstruction_max_error=error,
            decoder_cosine={'min': float(off.min()), 'median': float(np.median(off)), 'max': float(off.max())},
            partitions=partitions, done_seconds=read_json(folder / 'done.json')['seconds']))
    prefixes = []
    if panel_kind == 'inherited':
        old = read_jobs(PREVIOUS / 'runs/topic/jobs.jsonl')
        for seed in (0, 1):
            for low, high in ((20000, 40000),):
                a, b = traces[(seed, low)], traces[(seed, high)][:low]
                difference = max(abs(u[k] - v[k]) for u, v in zip(a, b) for k in ('loss', 'recon_loss', 'kl_div'))
                require(difference == 0, 'current same-seed budget trace prefix differs')
                prefixes.append(dict(seed=seed, lower=low, upper=high, max_abs_error=difference, historical=False))
            for job in [j for j in old if j.seed == seed]:
                low = job.params['steps']
                a = read_json(job_dir(PREVIOUS / 'runs/topic', job.job_id) / 'result.json')['training_trace']
                b = traces[(seed, 20000)][:low]
                difference = max(abs(u[k] - v[k]) for u, v in zip(a, b) for k in ('loss', 'recon_loss', 'kl_div'))
                require(difference == 0, 'compatible historical same-seed prefix differs')
                prefixes.append(dict(seed=seed, lower=low, upper=20000, max_abs_error=difference,
                    historical=True, scope='compatible_trace_only_old_receipt_not_reused'))
            jobs = read_jobs(run / 'jobs.jsonl')
            low_result, high_result = [read_json(job_dir(run, next(j.job_id for j in jobs if j.seed == seed and j.params['steps'] == step)) /
                                              'result.json')['evaluation_trace'] for step in (20000, 40000)]
            require(low_result == high_result[:len(low_result)], 'same-seed evaluation prefix differs')
    return dict(fits=rows, partition=partition, panel_fit=fit, prefix_checks=prefixes,
                cross_panel_raw_crossentropy_ranking_authorized=False)


def verify_population(run):
    from expansion.common.rng import derive_seed
    from expansion.population.metrics import evaluate
    from expansion.population.worker import cached_coordinates, load_dataset, reset_process_caches
    from expansion.population.tasks import build_task_data, TaskSpec, representation_mask, make_clock
    p = check_binding(run)
    require(p == population_protocol(), 'population scope drift')
    files = read_json(run / 'datasets.json')['GSE241132']
    reset_process_caches()
    adata = load_dataset(files['dataset_file'], files['dataset_sha256'])
    rows, max_error, sb_rows, rep_ids = [], 0., [], {}
    for job in read_jobs(run / 'jobs.jsonl'):
        folder, task = job_dir(run, job.job_id), TaskSpec.from_dict(job.task)
        result, base = read_json(folder / 'result.json'), method_base(p, job.method)
        mask = representation_mask(adata.obs, task)
        require(not mask[adata.obs.individual.astype(str).eq(task.held_individual)].any(), 'held donor shapes PCA')
        coords, info = cached_coordinates(adata, mask, p['representations'][job.representation], run / 'cache',
            dataset_sha256=files['dataset_sha256'], seed=p['representation_seed'], code_digest=job.params['code_digest'])
        identity = rep_ids.setdefault(task.held_individual, info['cache_key'])
        require(identity == info['cache_key'] == result['representation']['cache_key'], 'within-fold representation mismatch')
        data = build_task_data(coords, adata.obs, task, make_clock(adata.obs.time_value.unique(), 'rank'), min_cells=50)
        require({task.held_individual}.isdisjoint(pat for pat, _ in data.train), 'held donor enters method train')
        expected_steps = p['methods'][job.method].get('params', {}).get('steps')
        if expected_steps is not None:
            require(result['fit']['steps'] == expected_steps, 'flow alias budget did not reach fit')
        require(len(result['records']) == len(task.pairs) == 3, 'missing planned evaluation pairs')
        with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
            require(set(saved.files) == {f'pair{i}_own' for i in range(3)}, 'prediction inventory mismatch')
            for record in result['records']:
                pair = task.pairs[record['pair_index']]
                require(record['source_variant'] == 'own' and record['individual'] == pair.individual and
                        record['source_time'] == pair.source_time and record['target_time'] == pair.target_time, 'pair identity drift')
                eval_seed = derive_seed(task.dataset_id, task.kind, task.family, pair.individual, pair.source_time, pair.target_time)
                metrics = evaluate(saved[f'pair{record["pair_index"]}_own'], data.targets[(pair.individual, pair.target_time)],
                    seed=eval_seed, caps=p['evaluation']['caps'], n_projections=32)
                max_error = max(max_error, assert_metrics({key: record[key] for key in metrics}, metrics, 'population-scores', 2e-6))
                rows.append({**{key: record[key] for key in metrics}, 'job_id': job.job_id, 'method': base,
                    'method_alias': job.method, 'seed': job.seed, 'steps': expected_steps, 'held_individual': pair.individual,
                    'source_time': pair.source_time, 'target_time': pair.target_time, 'fit_seconds': result['fit_seconds'],
                    'stage_seconds': result['stage_seconds'], 'flow_stage_seconds': result['fit'].get('stage_seconds_synchronized')})
        if base == 'cfm_sb':
            batches = result['fit']['sb_batch_diagnostics']
            require(len(batches) == expected_steps and [r['step'] for r in batches] == list(range(1, expected_steps + 1)),
                    'SB alias lost actual batches')
            for batch in batches:
                require(batch['finite_log_plan'] and batch['finite_plan'] and batch['total_iterations'] ==
                        sum(s['iterations'] for s in batch['stages']), 'SB numerical accounting failed')
                require(batch['both_marginals_converged'] == (max(batch['row_l1'], batch['column_l1']) <= batch['tolerance']),
                        'SB convergence reported incorrectly')
            sb_rows.extend(batches)
    require(len(rows) == 180 and len(sb_rows) == 4320 and len(rep_ids) == 3, 'full population denominator mismatch')
    baseline = {(r['held_individual'], r['source_time'], r['target_time'], r['method']): r['energy']
                for r in rows if r['steps'] is None}
    for row in rows:
        row['energy_minus_baseline'] = {name: row['energy'] - baseline[(row['held_individual'], row['source_time'], row['target_time'], name)]
                                       for name in ('identity', 'mean_shift', 'location_scale', 'target_marginal')}
    return dict(jobs=60, paired_records=rows, metric_reconstruction_max_abs_error=max_error,
        sb_batches=len(sb_rows), sb_nonconverged=sum(not b['both_marginals_converged'] for b in sb_rows),
        sb_max_row_l1=max(b['row_l1'] for b in sb_rows), sb_max_column_l1=max(b['column_l1'] for b in sb_rows),
        representations_by_donor=rep_ids, inferential_unit='none_diagnostic_only',
        scope='three_healthy_acute_wound_donors_not_DFU_or_superiority_or_source_increment_validation')


def verify(root, out):
    from scripts.run_bounded_local_diagnostics import resource_violations, output_bytes
    root, out = Path(root).resolve(), Path(out).resolve()
    require(root in out.parents and not out.exists(), 'fresh scoped verification directory required')
    policy = read_json(root / 'resource_policy.json')
    require(policy['runner_sha256'] == sha256_file(ROOT / 'scripts/run_bounded_local_diagnostics.py') and
            policy['extension_sha256'] == sha256_file(ROOT / 'scripts/run_sdd_local_validation.py') and
            policy['spec_sha256'] == sha256_file(SPEC), 'SDD runner/spec drift')
    protected = read_json(root / 'protected_before.json')
    require(all(sha256_file(ROOT / k) == h for k, h in protected.items()), 'protected historical evidence drift')
    require(read_json(root / 'scientific_source.json') == {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files()},
            'scientific code drift')
    require(read_json(root / 'preparation.json')['implementation_hashes'] == implementation_hashes(), 'implementation dependency drift')
    profiles = {}
    for folder in sorted((root / 'profiles').iterdir()):
        s, w = read_json(folder / 'supervisor_result.json'), read_json(folder / 'worker_result.json')
        samples = read_json(folder / 'resource_samples.json')
        require(s['status'] == w['status'] == 'finished' and not s['violations'], 'failed profile')
        require(s['policy_sha256'] == sha256_file(root / 'resource_policy.json'), 'profile policy drift')
        require(all(not resource_violations(sample, policy, 0) for sample in samples), 'sample safety failed')
        require(w['rss_high_water_bytes'] <= 12 * 2**30 and s['seconds'] <= 600, 'profile highwater exceeded')
        require(w['runtime']['torch_threads'] == 4 and w['runtime']['interop_threads'] == 1, 'worker thread allocation drift')
        require(not Path(f'/proc/{s["pid"]}').exists(), 'worker still running')
        profiles[folder.name] = dict(supervisor=s, worker=w,
            peak_temperature_c=max(sample['gpu']['temperature_c'] for sample in samples),
            sampled_peak_gpu_mib=max((sample['gpu']['own_process_mib'] for sample in samples
                                     if sample['gpu']['own_process_mib'] is not None), default=0))
    require(sum(r['supervisor']['seconds'] for r in profiles.values()) <= 1800, 'lane wall budget exceeded')
    for name, expected in [('topic-inherited-replay', 4), ('topic-training-replay', 2), ('population-replay', 60)]:
        result = profiles[name]['worker']['result']
        require(result['done'] == result['failed'] == result['skipped_failed'] == 0 and result['skipped_done'] == expected,
                'replay retrained or missing completed jobs')
    report = dict(utc=stamp(), status='passed_bounded_diagnostic_only', code_digest=code_digest(),
        topic_inherited=verify_topic(root / 'runs/topic-inherited', 'inherited'),
        topic_training=verify_topic(root / 'runs/topic-training', 'training'),
        population=verify_population(root / 'runs/population'), profiles=profiles,
        protected_files_unchanged=len(protected), output_bytes=output_bytes(root), verifier_sha256=sha256_file(__file__),
        no_new_payload_downloads=True, no_remote_gpu=True, no_manuscript_or_publication_mutations=True)
    out.mkdir(parents=True, exist_ok=False)
    atomic_write_json(out / 'verification.json', report)
    return report


def seal(root):
    from scripts.run_bounded_local_diagnostics import output_bytes
    root = Path(root).resolve()
    require(not (root / 'FINAL_STATUS.json').exists(), 'existing seal is immutable')
    verification = read_json(root / 'verification/verification.json')
    regression = read_json(root / 'evidence/regression.json')
    require(verification['status'] == 'passed_bounded_diagnostic_only' and regression['exit_code'] == 0 and
        sha256_file(root / 'evidence/repository-tests-final.txt') == regression['log_sha256'], 'final checks incomplete')
    require(all(sha256_file(ROOT / k) == h for k, h in read_json(root / 'protected_before.json').items()), 'protected drift')
    require(read_json(root / 'scientific_source.json') == {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files()}, 'source drift')
    snapshot = root / 'source-checkpoint-final'
    manifest = read_json(snapshot / 'manifest.json')
    require(all(sha256_file(ROOT / k) == v['sha256'] for k, v in manifest['files'].items()), 'source snapshot drift')
    require(not list((root / 'runs').glob('*/jobs/*/running.json')), 'live job marker')
    for profile in verification['profiles'].values():
        require(not Path(f'/proc/{profile["supervisor"]["pid"]}').exists(), 'live worker')
    archive_sha = sha256_file(snapshot / 'authored-source.tar')
    require((snapshot / 'authored-source.tar.sha256').read_text().startswith(archive_sha), 'source archive checksum mismatch')
    status = dict(utc=stamp(), status='completed_bounded_SDD_diagnostic_only', code_digest=code_digest(), topic_fits=6,
        population_jobs=60, population_pairs=180, sb_batches=4320,
        sb_nonconverged=verification['population']['sb_nonconverged'], replay_new_fits=0,
        protected_files_unchanged=verification['protected_files_unchanged'], final_regression=regression,
        source_checkpoint_files=manifest['file_count'], internal_source_archive_sha256=archive_sha,
        supervised_seconds_excluding_replay=sum(v['supervisor']['seconds'] for k, v in verification['profiles'].items() if not k.endswith('replay')),
        maximum_rss_high_water_bytes=max(v['worker']['rss_high_water_bytes'] for v in verification['profiles'].values()),
        maximum_reserved_cuda_bytes=max(v['worker']['cuda_peak_reserved_bytes'] for v in verification['profiles'].values()),
        new_output_bytes_before_seal=output_bytes(root), no_new_payload_or_remote_GPU=True,
        no_manuscript_or_publication_mutation=True, scope='not_formal_replication_clinical_validation_or_software_release')
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file()
        and '__pycache__' not in p.parts and p.name != 'FINAL_STATUS.json' and p != root / 'evidence/evidence_hashes.json'}
    atomic_write_json(root / 'evidence/evidence_hashes.json', hashes)
    atomic_write_json(root / 'FINAL_STATUS.json', status)
    require(all(sha256_file(root / k) == h for k, h in hashes.items()), 'evidence drift during seal')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'seal'])
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    value = verify(args.experiment_root, args.out) if args.command == 'verify' else seal(args.experiment_root)
    print(json.dumps({k: value[k] for k in ('status', 'code_digest')}, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    main()
