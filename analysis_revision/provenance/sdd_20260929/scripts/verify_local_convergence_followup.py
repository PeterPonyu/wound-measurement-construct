#!/usr/bin/env python3
"""Reconstruct diagnostics from saved matrices/checkpoints; never refit models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expansion.analysis.aggregate import audit_run
from expansion.common.io import atomic_write_json, read_json, sha256_file
from expansion.common.provenance import code_digest, code_files, input_digests
from expansion.measurement import stability
from expansion.runner.approval import require_execution_approval
from expansion.runner.jobs import read_jobs
from expansion.runner.queue import job_dir
from scripts.run_bounded_local_diagnostics import resource_violations, output_bytes
from scripts.run_local_convergence_followup import require, measurement_protocol, population_protocol


def check_binding(run):
    protocol, provenance = read_json(run / 'protocol.json'), read_json(run / 'provenance.json')
    require(provenance['code_digest'] == code_digest(), 'scientific code drift')
    require(input_digests(ROOT, provenance['inputs']) == provenance['inputs'], 'discovery input drift')
    require_execution_approval(run, run / 'execution_approval.json', protocol=protocol,
        files=read_json(run / 'datasets.json'), provenance=provenance)
    require(audit_run(run)['complete'], 'run is incomplete or has corrupt outputs')
    return protocol


def verify_topic(run):
    from core_engines.topic_simplex_engine import TopicSimplexModel
    protocol = check_binding(run)
    require(protocol == measurement_protocol(), 'topic scope drift')
    proportions, cells = stability.load_discovery_panel_counts(ROOT)
    mask, selected, partition = stability.patient_partition(cells, protocol['stability']['patient_partition'])
    panel = np.load(ROOT / stability.DISCOVERY / 'panel.npy', allow_pickle=True)
    reference = np.load(ROOT / stability.DISCOVERY / 'beta.npy', allow_pickle=False)
    rows = []
    for job in read_jobs(run / 'jobs.jsonl'):
        folder, steps = job_dir(run, job.job_id), job.params['steps']
        result = read_json(folder / 'result.json')
        loaded = stability.load_inference_checkpoint(folder, panel=panel,
                       cell_ids=cells.reconstructed_raw_cell_id.to_numpy())
        config = loaded['metadata']['training_config']
        require(config['steps'] == steps and config['patient_partition'] == partition == result['patient_partition'],
                'training scope or partition record mismatch')
        require(config['optimization_proportions_sha256'] == stability._matrix_hash(proportions[mask]),
                'optimization did not bind training-patient input matrix')
        trace, evaluation = result['training_trace'], result['evaluation_trace']
        require([r['step'] for r in trace] == list(range(1, steps + 1)), 'missing dense optimizer trace')
        require([r['step'] for r in evaluation] == [0] + list(range(100, steps + 1, 100)), 'missing fixed evaluation times')
        require(all(np.isfinite([r['loss'], r['recon_loss'], r['kl_div']]).all() for r in trace), 'nonfinite trace')
        model = TopicSimplexModel(**loaded['metadata']['model_config'])
        model.load_state_dict(loaded['state'])
        recomputed_eval = {name: stability.evaluate_topic_model(model, proportions[index],
                    cells.iloc[index].patient_id.to_numpy(dtype=str)) for name, index in selected.items()}
        max_error = 0.
        for name, scores in recomputed_eval.items():
            saved = evaluation[-1][name]
            require(scores['cells'] == saved['cells'] and scores['patients'] == saved['patients'], 'evaluation denominator')
            for key in ('cell_mean_reconstruction', 'cell_mean_kl', 'cell_mean_objective',
                        'equal_patient_reconstruction', 'equal_patient_objective', 'mean_normalized_theta_entropy'):
                error = abs(scores[key] - saved[key])
                max_error = max(max_error, error)
                require(error <= 2e-5, f'saved final evaluation cannot be reconstructed: {key}')
        matched = stability.match_topics(reference, loaded['beta'])
        require(np.allclose(matched['correlation'], result['match_to_reference']['correlation'], rtol=0, atol=1e-7),
                'saved beta/reference match mismatch')
        assigned = matched['assignment'].get(0)
        require(assigned == result['fibroblast_topic'], 'primary assignment was rescued or changed')
        if result['primary_readout_status'] == 'available':
            values = loaded['theta'][:, assigned]
            fib = cells.celltype.eq('fibroblast').to_numpy()
            cut = stability.gmm_midpoint(values[mask & fib])
            require(abs(cut - result['cut']) <= 1e-10, 'mixture cut not fitted on training patients only')
            frame = pd.DataFrame({'unit': cells.loc[fib, 'patient_id'].to_numpy(),
                 'specimen': cells.loc[fib, 'gsm'].to_numpy(), 'value': values[fib]})
            table = frame.groupby(['unit', 'specimen']).value.agg(mean='mean', n_cells='size',
                                               fraction=lambda x: float(np.mean(x > cut)))
            patient = table.groupby('unit').agg(mean=('mean', 'mean'), fraction=('fraction', 'mean'),
                                               n_cells=('n_cells', 'sum'), n_specimens=('mean', 'size'))
            for record in result['patient_readouts']:
                for key in ('mean', 'fraction', 'n_cells', 'n_specimens'):
                    require(abs(patient.loc[record['unit'], key] - record[key]) <= 1e-10, 'patient readout mismatch')
                require(record['partition'] == ('held' if record['unit'] in partition['held_patients'] else 'train'),
                        'readout partition mismatch')
        require(result['semantics']['available'] is False and partition['end_to_end_independent'] is False,
                'convergence lane exports validation claims')
        last, prior = evaluation[-1], evaluation[-2]
        rows.append({'job_id': job.job_id, 'seed': job.seed, 'steps': steps,
            'done_seconds': read_json(folder / 'done.json')['seconds'],
            'training_last_loss': trace[-1]['loss'],
            'training_loss_last100_mean': float(np.mean([r['loss'] for r in trace[-100:]])),
            'training_loss_previous100_mean': float(np.mean([r['loss'] for r in trace[-200:-100]])),
            'final_train_objective': last['train']['equal_patient_objective'],
            'final_held_objective': last['held']['equal_patient_objective'],
            'held_last100_objective_delta': last['held']['equal_patient_objective'] - prior['held']['equal_patient_objective'],
            'held_theta_normalized_entropy': last['held']['mean_normalized_theta_entropy'],
            'held_argmax_counts': last['held']['argmax_topic_counts'],
            'held_topic_mean_loading': last['held']['topic_mean_loading'],
            'reference0_assigned_correlation': result['fibroblast_topic_correlation'],
            'reference0_assignment_gap': result['match_to_reference']['best_minus_second']['0'],
            'reload_max_abs_error': loaded['metadata']['reload_max_abs_error'],
            'evaluation_reconstruction_max_error': max_error,
            'patient_readouts': result['patient_readouts'], 'evaluation_trace': evaluation,
            'plateau_claim': 'not_established_fixed_budgets_only'})
    return {'fits': sorted(rows, key=lambda r: (r['steps'], r['seed'])), 'partition': partition,
        'denominators': {'all_cells': len(cells), 'all_patients': cells.patient_id.nunique(),
            'DFU_patients': cells.loc[cells.group.str.startswith('DFU'), 'patient_id'].nunique()},
        'scope': 'held_patients_excluded_from_optimizer_but_inherited_panel_not_end_to_end_clinical_validation'}


def verify_population(run):
    from expansion.common.rng import derive_seed
    from expansion.population.metrics import evaluate
    from expansion.population.worker import load_dataset, cached_coordinates, reset_process_caches
    from expansion.population.tasks import TaskSpec, build_task_data, make_clock, representation_mask
    protocol = check_binding(run)
    require(protocol == population_protocol(), 'paired population scope drift')
    manifest = read_json(run / 'datasets.json')['GSE241132']
    reset_process_caches()
    adata = load_dataset(manifest['dataset_file'], manifest['dataset_sha256'])
    rows, max_error = [], 0.
    for job in read_jobs(run / 'jobs.jsonl'):
        folder, task = job_dir(run, job.job_id), TaskSpec.from_dict(job.task)
        result = read_json(folder / 'result.json')
        mask = representation_mask(adata.obs, task)
        require(not mask[adata.obs.individual.astype(str).eq(task.held_individual)].any(), 'held donor in PCA training')
        coords, info = cached_coordinates(adata, mask, protocol['representations'][job.representation], run / 'cache',
            dataset_sha256=manifest['dataset_sha256'], seed=protocol['representation_seed'], code_digest=job.params['code_digest'])
        data = build_task_data(coords, adata.obs, task, make_clock(adata.obs.time_value.unique(), 'rank'), min_cells=50)
        require(task.held_individual not in {p for p, _ in data.train}, 'held donor in method training')
        with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
            for record in result['records']:
                pair = task.pairs[record['pair_index']]
                seed = derive_seed(task.dataset_id, task.kind, task.family, pair.individual, pair.source_time, pair.target_time)
                metrics = evaluate(saved[f'pair{record["pair_index"]}_own'], data.targets[(pair.individual, pair.target_time)],
                    seed=seed, caps=protocol['evaluation']['caps'], n_projections=32)
                for key, value in metrics.items():
                    if isinstance(value, bool):
                        require(record[key] == value, 'prediction finite flag mismatch')
                    else:
                        error = abs(value - record[key])
                        max_error = max(max_error, error)
                        require(error <= 2e-6 * max(1., abs(value)), f'saved-target metric mismatch: {key}')
                rows.append({'job_id': job.job_id, 'method': job.method, 'seed': job.seed,
                    'source_time': pair.source_time, 'target_time': pair.target_time, 'held_individual': pair.individual,
                    'energy': record['energy'], 'exact_w2': record['exact_w2'], 'mean_error': record['mean_error'],
                    'fit_seconds': result['fit_seconds'], 'stage_seconds': result['stage_seconds'],
                    'flow_stage_seconds': result['fit'].get('stage_seconds_synchronized'),
                    'steps': result['fit'].get('steps'), 'solver': result['fit'].get('sb_solver_summary')})
        if job.method == 'cfm_sb':
            batches = result['fit']['sb_batch_diagnostics']
            require(len(batches) == 120 and [b['step'] for b in batches] == list(range(1, 121)), 'SB batch inventory')
            for b in batches:
                require(b['finite_plan'] and b['finite_log_plan'] and b['total_iterations'] ==
                        sum(s['iterations'] for s in b['stages']), 'SB solver accounting mismatch')
                require(b['both_marginals_converged'] == (max(b['row_l1'], b['column_l1']) <= b['tolerance']),
                        'SB convergence misreported')
    return {'paired_records': rows, 'jobs': len(read_jobs(run / 'jobs.jsonl')), 'metric_reconstruction_max_abs_error': max_error,
            'scope': 'one_held_healthy_wound_donor_fixed120_step_cost_diagnostic_not_method_selection_or_DFU_validation'}


def verify_numerical(folder):
    rows = read_json(folder / 'sinkhorn.json')
    require(len(rows) == 18 and len({r['plan_key'] for r in rows}) == 18, 'numerical grid incomplete')
    with np.load(folder / 'sinkhorn_log_plans.npz', allow_pickle=False) as saved:
        for r in rows:
            # Reconstruct in the original dtype (CPU/CUDA summation can round
            # differently); tolerance is numerical, not scientific selection.
            p = torch.from_numpy(saved[r['plan_key']]).exp()
            row = float((p.sum(1) - 1 / p.shape[0]).abs().sum())
            col = float((p.sum(0) - 1 / p.shape[1]).abs().sum())
            require(abs(row - r['row_l1']) < 2e-6 and abs(col - r['column_l1']) < 2e-6
                    and abs(float(p.sum()) - r['mass']) < 2e-6, 'returned-plan reconstruction mismatch')
    batches = read_json(folder / 'synthetic_sb_fit.json')['sb_batch_diagnostics']
    require(len(batches) == 12 and any(not b['both_marginals_converged'] for b in batches),
            'deliberate exhausted training batches missing')
    return {'fixtures': rows, 'synthetic_batches': len(batches),
            'nonconverged_static': sum(not r['both_marginals_converged'] for r in rows)}


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(root in out.parents and not out.exists(), 'verification needs a fresh scoped directory')
    policy = read_json(root / 'resource_policy.json')
    require(sha256_file(ROOT / 'scripts/run_bounded_local_diagnostics.py') == policy['runner_sha256']
            and sha256_file(ROOT / 'scripts/run_local_convergence_followup.py') == policy['extension_sha256'], 'runner drift')
    reports = {}
    for folder in sorted((root / 'profiles').iterdir()):
        if not folder.is_dir():
            continue
        sup, worker = read_json(folder / 'supervisor_result.json'), read_json(folder / 'worker_result.json')
        samples = read_json(folder / 'resource_samples.json')
        require(sup['status'] == worker['status'] == 'finished' and not sup['violations'], 'failed resource batch')
        require(sup['runner_sha256'] == worker['runner_sha256'] == policy['runner_sha256']
                and sup['policy_sha256'] == sha256_file(root / 'resource_policy.json'), 'profile policy/runner binding drift')
        require(all(not resource_violations(s, policy, 0) for s in samples), 'sampled safety violation')
        require(worker['rss_high_water_bytes'] <= 12 * 2**30 and sup['seconds'] <= 600, 'high-water resource violation')
        require(worker['runtime']['torch_threads'] == 4 and worker['runtime']['interop_threads'] == 1, 'thread allocation drift')
        reports[folder.name] = {'supervisor': sup, 'worker': worker,
            'peak_temperature_c': max(s['gpu']['temperature_c'] for s in samples),
            'sampled_peak_gpu_mib': max((s['gpu']['own_process_mib'] for s in samples if s['gpu']['own_process_mib'] is not None), default=None)}
    require(sum(r['supervisor']['seconds'] for r in reports.values()) <= 1800, 'whole lane exceeded wall budget')
    for name, expected in (('topic-replay', 6), ('population-replay', 10)):
        result = reports[name]['worker']['result']
        require(result['done'] == result['failed'] == result['skipped_failed'] == 0 and result['skipped_done'] == expected,
                'completed-job replay retrained or failed')
    protected = read_json(root / 'protected_before.json')
    require(all(sha256_file(ROOT / p) == h for p, h in protected.items()), 'protected historical file changed')
    require(read_json(root / 'scientific_source.json') == {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files()},
            'scientific code changed after freeze')
    preservation = read_json(root / 'profiles/sinkhorn-audit/cuda_trace_preservation.json')
    require(preservation['all_state_tensors_bitwise_equal'] and preservation['cpu_and_cuda_rng_equal'],
            'CUDA trajectory preservation gate failed')
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed_diagnostic_only', 'code_digest': code_digest(),
        'topic': verify_topic(root / 'runs/topic'), 'population': verify_population(root / 'runs/population'),
        'numerical': verify_numerical(root / 'profiles/sinkhorn-audit'), 'profiles': reports,
        'protected_files_unchanged': len(protected), 'output_bytes': output_bytes(root),
        'cuda_trace_preservation': preservation, 'verifier_sha256': sha256_file(__file__),
        'no_new_downloads': True, 'no_remote_gpu': True, 'no_publication_mutations': True,
        'no_manuscript_changes': True, 'no_new_independent_review': True}
    out.mkdir(parents=True, exist_ok=False)
    atomic_write_json(out / 'verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = verify(args.experiment_root, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'code_digest', 'protected_files_unchanged', 'output_bytes')}, indent=2))


if __name__ == '__main__':
    main()
