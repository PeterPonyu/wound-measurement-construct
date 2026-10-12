#!/usr/bin/env python3
"""Prepare, execute and verify a fresh, fixed, low-cost local diagnostic lane.

Real-data jobs always pass the normal code/input-bound approval gate. The receipt
records the explicit user request, not an invented independent approval. No data
downloads, public writes, favorable-fit selection or optimizer resume are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expansion.common.io import atomic_save_npz, atomic_write_json, content_hash, read_json, sha256_file
from expansion.common.provenance import code_digest, code_files, input_digests

SPEC = ROOT / '.omx/plans/2026-09-29-local-convergence-followup-spec.md'
DISCOVERY = ROOT / 'output/local-resource-experiments-20260929'
STANDARD = ROOT / 'output/data-smoke-20260929/data/standardized'
USER_REQUEST = '你就做就行了，越是高杠杆，越是高转化率，越是高这个低成本的这种，你就越是给我做，往前推进。'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measurement_protocol():
    import pandas as pd
    from expansion.measurement import stability
    cells = pd.read_parquet(ROOT / stability.CELLS, columns=['patient_id', 'gsm', 'reconstructed_raw_cell_id'])
    held = ['NCOMMS2022_S11', 'NCOMMS2022_S07', 'NCOMMS2022_S02', 'NCOMMS2022_S04']
    return {'schema_version': 1, 'kind': 'measurement', 'protocol_id': 'local-patient-convergence-20260929-v1',
        'smoke_only': True, 'question': 'Fixed optimizer budgets under outcome-blind patient exclusion.',
        'cohorts': {'discovery': 'GSE165816', 'external': []},
        'semantics': {'families': [], 'discovery_only_families': []}, 'design': {'enabled': False},
        'analysis': {'report_mode': 'diagnostic_only', 'inferential_unit': 'none_diagnostic_only'},
        'stability': {'K': [15], 'seeds': [0, 1], 'step_budgets': [500, 2000, 5000],
            'batch_size': 128, 'lr': .001, 'kl_weight': .01, 'diagnostic_every': 100,
            'patient_partition': {'method': 'id_hash_v1', 'seed': 20260929, 'n_held_patients': 4,
                'expected_held_patients': held, 'evaluation_cells_per_patient': 128,
                'cell_identity_column': 'reconstructed_raw_cell_id',
                'expected_cell_order_identity': content_hash(cells.reconstructed_raw_cell_id.tolist(), 64)}},
        'not_claims': ['Not additional biological patients or independent clinical validation.',
            'Inherited full-discovery panel; not end-to-end held-patient independence.',
            'No healing/amputation labels consulted for splitting, scoring, tuning or stopping.',
            '5000 steps do not guarantee convergence; no favorable seed/step selection.']}


def population_protocol():
    p = read_json(ROOT / 'output/data-smoke-20260929/runs/population-conditioned-pilot-v1/protocol.json')
    p.update(protocol_id='local-paired-flow-cost-diagnostic-20260929-v1', smoke_only=True,
        status='bounded_paired_diagnostic_only',
        question='Fixed equal-step all-flow-method cost and numerical diagnostic on one held healthy wound donor.')
    p['analysis'] = {'report_mode': 'diagnostic_only', 'inferential_unit': 'none_diagnostic_only'}
    p['methods'] = {name: {'tasks': ['T-LIO']} for name in
                   ('cfm_independent', 'cfm_ot', 'cfm_sb', 'cfm_conditioned', 'target_marginal', 'location_scale')}
    p['flow_defaults'] = dict(steps=120, batch_size=64, hidden=64, depth=2, time_dim=16, lr=.001,
        rk4_steps_per_interval=20, sinkhorn_iterations=300, sinkhorn_tolerance=.001,
        sb_sigma_scale=.25, grad_clip=1., context_dropout=.2, profile_stages=True)
    p['evaluation']['caps'] = {name: (100 if name == 'exact_w2' else 200)
                               for name in ('energy', 'sliced_w2', 'exact_w2', 'c2st')}
    p['evaluation']['sliced_projections'] = 32
    p['stage_gate'] = {'stage_a_stops': True, 'stage_b_requires': 'fresh reviewed code/input-bound local receipt'}
    return p


def prepare(root):
    from expansion.measurement import stability
    from expansion.runner.approval import approval_binding
    from expansion.runner.cli import _load_contracts, plan_run
    from expansion.runner.jobs import read_jobs
    from scripts.run_bounded_local_diagnostics import validate_policy

    root = Path(root).resolve()
    require(ROOT / 'output' in root.parents and not (root / 'protected_before.json').exists(),
            'preparation requires a fresh output-root record')
    root.mkdir(parents=True, exist_ok=True)
    historical = read_json(DISCOVERY / 'protected_before.json')
    require(all(sha256_file(ROOT / k) == v for k, v in historical.items()), 'historical protection baseline already drifted')
    names = set(historical)
    for directory in (DISCOVERY, STANDARD / 'GSE241132'):
        names.update(str(p.relative_to(ROOT)) for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected = {k: sha256_file(ROOT / k) for k in sorted(names)}
    atomic_write_json(root / 'protected_before.json', protected)
    atomic_write_json(root / 'scientific_source.json', {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files()})
    policy = read_json(DISCOVERY / 'resource_policy.json')
    policy.update(experiment_root=str(root), runner_sha256=sha256_file(ROOT / 'scripts/run_bounded_local_diagnostics.py'),
                  extension_sha256=sha256_file(__file__), spec_sha256=sha256_file(SPEC),
                  runner_tests_sha256=sha256_file(ROOT / 'tests/test_convergence_followup.py'))
    validate_policy(policy)
    atomic_write_json(root / 'resource_policy.json', policy)
    contracts = _load_contracts(STANDARD, ['GSE241132'])
    contract = contracts['GSE241132']
    require(contract['contract']['status'] == 'accepted' and contract.get('probe'), 'accepted local contract/probe missing')
    require(sha256_file(contract['dataset_file']) == contract['dataset_sha256'], 'standard input drift')
    summaries = {}
    for name, protocol, material in (('topic', measurement_protocol(), {}), ('population', population_protocol(), contracts)):
        folder = root / 'runs' / name
        summaries[name] = plan_run(protocol, material, folder, root=ROOT)
        receipt = {**approval_binding(folder), 'approved': True, 'stage_a_reviewed': True,
            'approved_by': 'user_explicit_current_thread_instruction_recorded_by_codex',
            'approved_utc': stamp(), 'reviewed_dataset_ids': ['GSE165816' if name == 'topic' else 'GSE241132'],
            'review_notes': ('Current explicit user authorization: ' + USER_REQUEST +
                ' Scope limited by the pre-fit specification; no new downloads, remote GPU, formal grid or publication. '
                'Codex reviewed the actual frozen inputs/local accepted contract. Not an independent reviewer or author signature.')}
        atomic_write_json(folder / 'execution_approval.json', receipt)
        jobs = read_jobs(folder / 'jobs.jsonl')
        if name == 'topic':
            cells = __import__('pandas').read_parquet(ROOT / stability.CELLS)
            _, _, split = stability.patient_partition(cells, protocol['stability']['patient_partition'])
            atomic_write_json(root / 'patient_partition_frozen.json', split)
        else:
            held = {j.task['held_individual'] for j in jobs}
            require(len(jobs) == 10 and len(held) == 1, 'population plan does not match fixed diagnostic')
            atomic_write_json(root / 'population_tasks_frozen.json', {'jobs': [j.to_dict() for j in jobs],
                'held_individuals': sorted(held), 'selection': 'existing_dataset_ID_seeded_generator_max_held_1_outcome_blind',
                'dataset_sha256': contract['dataset_sha256'], 'stage_a_probe_limitations':
                'generic probe reports multi-member ZIP unsupported; accepted adapter/join evidence is separately preserved',
                'scope': 'one_held_healthy_wound_donor_not_DFU_validation_or_full_three_donor_benchmark'})
    atomic_write_json(root / 'preparation.json', {'utc': stamp(), 'spec_sha256': sha256_file(SPEC),
        'code_digest': code_digest(), 'plans': summaries, 'user_request': USER_REQUEST, 'protected_files': len(protected)})
    return summaries


def validate_followup_lane(args, root):
    """Fail closed if any sealed job or protocol exceeds the declared diagnostic."""
    from expansion.runner.jobs import read_jobs
    root = Path(root).resolve()
    if args.mode == 'sinkhorn_audit':
        require(not args.run_dir and not args.approval and args.max_jobs is None, 'numerical mode accepts no real-data queue')
        return
    name = 'topic' if args.mode == 'topic_convergence' else 'population'
    folder = root / 'runs' / name
    require(Path(args.run_dir).resolve() == folder and Path(args.approval).resolve() == folder / 'execution_approval.json',
            'follow-up paths must point to the sealed scoped run and receipt')
    require(args.max_jobs in range(1, 7 if name == 'topic' else 11), 'max-jobs exceeds the bounded queue')
    protocol, jobs = read_json(folder / 'protocol.json'), read_jobs(folder / 'jobs.jsonl')
    expected = measurement_protocol() if name == 'topic' else population_protocol()
    require(protocol == expected, 'follow-up protocol differs from fixed pre-fit scope')
    if name == 'topic':
        require(read_json(folder / 'datasets.json') == {} and len(jobs) == 6 and all(j.job_type == 'stability' for j in jobs)
                and {(j.params['K'], j.seed, j.params['steps']) for j in jobs} ==
                    {(15, s, b) for s in (0, 1) for b in (500, 2000, 5000)}, 'topic job grid mismatch')
    else:
        require(len(jobs) == 10 and all(j.job_type == 'population_fit' and j.dataset_id == 'GSE241132' for j in jobs)
            and {(j.method, j.seed) for j in jobs} == {(m, s) for m in
                ('cfm_independent', 'cfm_ot', 'cfm_sb', 'cfm_conditioned') for s in (0, 1)} |
                {('target_marginal', 0), ('location_scale', 0)}, 'paired population job grid mismatch')
        frozen = read_json(root / 'population_tasks_frozen.json')
        require([j.to_dict() for j in jobs] == frozen['jobs'], 'population task identities changed')


def sinkhorn_audit(out):
    import numpy as np
    import torch
    from expansion.population.methods.flows import _sinkhorn_log_plan
    rng = np.random.default_rng(8301)
    fixtures = {'uniform': np.zeros((19, 27)),
        'smooth': np.square(rng.normal(size=(19, 1)) - rng.normal(size=(1, 27))),
        'hard': np.square(np.linspace(-7, 7, 19)[:, None] - np.linspace(-4, 8, 27)[None, :])}
    records, plans = [], {}
    for name, cost in fixtures.items():
        for dtype in (torch.float32, torch.float64):
            for budget in (1, 30, 300):
                matrix = torch.tensor(cost, dtype=dtype, device='cuda')
                before = torch.cuda.get_rng_state().clone()
                log_plan, row, d = _sinkhorn_log_plan(matrix, .125, budget, .001, return_diagnostics=True)
                require(torch.equal(before, torch.cuda.get_rng_state()), 'solver consumed CUDA RNG')
                key = f'{name}_{str(dtype).split(".")[-1]}_{budget}'
                plans[key] = log_plan.cpu().numpy()
                records.append({'fixture': name, 'plan_key': key, 'target_budget': budget, **d})
    from expansion.population.methods import create
    from expansion.population.tasks import MethodInputs
    train = {(p, t): rng.normal(size=(32, 3)).astype('float32') + t for p in ('a', 'b') for t in (0., 1.)}
    method = create('cfm_sb', steps=12, batch_size=16, hidden=16, depth=1, time_dim=8,
                    sinkhorn_iterations=1, profile_stages=True)
    fit = method.fit(MethodInputs('T-LIO', train, {0.: 0., 1.: 1.}, None, 'held'), seed=9, device='cuda')
    require(len(fit['sb_batch_diagnostics']) == 12, 'synthetic SB lost training batches')
    import scipy.sparse as sp
    from expansion.measurement import stability
    x = sp.csr_matrix(rng.dirichlet(np.ones(16), size=32).astype('float32'))
    options = dict(K=3, seed=8, steps=9, batch_size=8, lr=.001, kl_weight=.01, device='cuda')
    plain = stability.train_topic_model(x, **options)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    sets = {name: {'proportions': x[:8], 'units': ['a'] * 4 + ['b'] * 4} for name in ('train', 'held')}
    logged = stability.train_topic_model(x, **options, diagnostic_inputs=sets, diagnostic_every=2)
    require(all(torch.equal(v, logged['state'][k]) for k, v in plain['state'].items()), 'CUDA diagnostic changed model trajectory')
    require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state()),
            'CUDA diagnostic changed CPU/CUDA RNG')
    atomic_write_json(out / 'cuda_trace_preservation.json', {'synthetic': True, 'steps': 9,
                     'all_state_tensors_bitwise_equal': True, 'cpu_and_cuda_rng_equal': True})
    atomic_write_json(out / 'sinkhorn.json', records)
    atomic_save_npz(out / 'sinkhorn_log_plans.npz', **plans)
    atomic_write_json(out / 'synthetic_sb_fit.json', fit)
    return {'fixtures': len(records), 'synthetic_sb_batches': 12,
            'nonconverged_static': sum(not r['both_marginals_converged'] for r in records),
            'scope': 'synthetic_numerical_only_no_method_ranking'}


def run_followup_worker(args, out):
    import functools
    import torch
    from expansion.runner import cli
    if args.mode == 'sinkhorn_audit':
        return sinkhorn_audit(out)
    from expansion.measurement import stability
    events, active = [], {'job_id': None}
    if args.mode == 'topic_convergence':
        for name in ('load_discovery_panel_counts', 'train_topic_model', 'project_theta', 'save_inference_checkpoint'):
            original = getattr(stability, name)

            def wrap(fn, label):
                @functools.wraps(fn)
                def instrumented(*a, **kw):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    try:
                        return fn(*a, **kw)
                    finally:
                        torch.cuda.synchronize()
                        events.append({'job_id': active['job_id'], 'stage': label,
                                       'seconds': time.perf_counter() - started})
                return instrumented
            setattr(stability, name, wrap(original, name))
    original_factory = cli.make_executor

    def factory(*a, **kw):
        execute = original_factory(*a, **kw)

        def checked(job, directory):
            active['job_id'] = job.job_id
            try:
                result = execute(job, directory)
                if args.mode == 'population_paired':
                    require(all(r['finite'] for r in result['records']), 'nonfinite held-target prediction')
                    if job.method == 'cfm_sb':
                        require(len(result['fit']['sb_batch_diagnostics']) == 120, 'missing SB batch records')
                else:
                    require(len(result['training_trace']) == job.params['steps'], 'missing optimizer step records')
            except Exception as error:
                # Stop this diagnostic lane immediately. Queue catches ordinary
                # Exceptions to continue general grids; this lane explicitly
                # retains the failed job and exits instead of dispatching more.
                atomic_write_json(directory / 'failed.json', {'job_id': job.job_id,
                    'error': f'{type(error).__name__}: {error}', 'utc': stamp(), 'scope': 'bounded_lane_abort_no_retry'})
                raise SystemExit(f'bounded follow-up aborted: {error}') from error
            atomic_write_json(out / 'stage_events.json', events)
            print(f'finished {job.job_id}: {job.method} seed={job.seed} budget={job.params.get("steps", "fixed120")}', flush=True)
            return result
        return checked
    cli.make_executor = factory
    return cli.execute_run(args.run_dir, root=ROOT, device='cuda', max_jobs=args.max_jobs,
                           approval_path=args.approval, retry_failed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare'])
    parser.add_argument('--experiment-root', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.experiment_root), indent=2))


if __name__ == '__main__':
    main()
