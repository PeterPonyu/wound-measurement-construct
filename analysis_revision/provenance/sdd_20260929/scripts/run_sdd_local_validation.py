#!/usr/bin/env python3
"""Fresh specification-bound SDD diagnostic plans and normal real-data dispatch."""
from __future__ import annotations

import argparse
import functools
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expansion.common.io import atomic_write_json, content_hash, read_json, sha256_file
from expansion.common.provenance import code_digest, code_files

SPEC = ROOT / '.omx/plans/2026-09-29-sdd-local-validation-spec.md'
PREVIOUS = ROOT / 'output/local-convergence-followup-20260929'
STANDARD = ROOT / 'output/data-smoke-20260929/data/standardized'
USER_REQUEST = ('我需要你把剩下的需要额外的很大的网络流量下载的那几个数据给我充分地计划好，'
                '并把它们涉及到的实验给我编排好。在此之外呢，我还需要你把现在能够额外补充上去的实验都给我处理掉。')
FLOW_BASES = ('cfm_independent', 'cfm_ot', 'cfm_sb', 'cfm_conditioned')
BASELINES = ('identity', 'mean_shift', 'location_scale', 'target_marginal')
PINNED = ['CDKN1A', 'FOS', 'JUNB', 'IL1B', 'TLR4', 'NLRP3', 'TNF', 'IFNG', 'MMP1', 'MMP3',
          'MMP13', 'CHI3L1', 'TIMP1', 'COL7A1', 'NRG1', 'ASPN', 'KRT14', 'KRT5', 'CD68',
          'CD163', 'PRG4', 'THY1', 'VEGFA']


def require(value, message):
    if not value:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def topic_protocol(panel):
    from scripts.run_local_convergence_followup import measurement_protocol
    p = measurement_protocol()
    p.update(protocol_id=f'sdd-budget-panel-{panel}-20260929-v1',
        question='Separate current-estimator budget sensitivity from training-panel fit sensitivity.')
    p['stability']['step_budgets'] = [20000, 40000] if panel == 'inherited' else [20000]
    p['stability']['retain_undefined_reference_match'] = True
    if panel == 'training':
        p['stability']['panel_selection'] = dict(method='training_dispersion_v1', n_top=7000,
            min_training_cells_per_gene=10, pinned_genes=PINNED)
    elif panel != 'inherited':
        raise ValueError('unknown panel scope')
    p['not_claims'] = ['No exact reproduction of historical CLI/sampler or biological/clinical validation.',
        'Training-only panel fit remains conditional on inherited full-discovery raw gene universe and cell QC.',
        'No cross-panel reconstruction ranking, held-out selection or reference-correlation tuning.']
    return p


def population_protocol():
    from scripts.run_local_convergence_followup import population_protocol as previous_protocol
    p = previous_protocol()
    p.update(protocol_id='sdd-all-donor-budget-paired-20260929-v1',
        question='Complete equal-step donor/budget diagnostic with all flows and fixed simple baselines.')
    p['tasks']['T-LIO']['max_held_individuals'] = 3
    p['methods'] = {f'{base}_s{steps}': {'base': base, 'params': {'steps': steps}, 'tasks': ['T-LIO']}
                    for steps in (120, 600) for base in FLOW_BASES}
    p['methods'].update({name: {'tasks': ['T-LIO']} for name in BASELINES})
    p['evaluation']['energy_scale_note'] = 'Descriptive within-fold scores, not confirmatory ranking.'
    p['evaluation']['mmd_note'] = 'Adaptive prediction-and-target kernel bandwidth; no cross-method MMD ranking/differences.'
    p['methods']['target_marginal']['sampling_note'] = 'Seed0 frozen equal-donor target draw, not sampling-free marginal.'
    return p


def implementation_hashes():
    names = ['scripts/run_sdd_local_validation.py', 'scripts/verify_sdd_local_validation.py',
        'scripts/run_bounded_local_diagnostics.py', 'scripts/run_local_convergence_followup.py',
        'scripts/verify_local_convergence_followup.py', 'scripts/audit_saved_topic_null.py',
        'scripts/finalize_local_convergence_followup.py', 'scripts/checkpoint_current_source.py',
        'tests/test_sdd_local_validation.py', 'tests/test_convergence_followup.py',
        '.omx/plans/2026-09-29-sdd-local-validation-spec.md']
    return {name: sha256_file(ROOT / name) for name in names}


def prepare(root):
    from expansion.runner.approval import approval_binding
    from expansion.runner.cli import _load_contracts, plan_run
    from expansion.runner.jobs import read_jobs
    from scripts.run_bounded_local_diagnostics import validate_policy

    root = Path(root).resolve()
    require(ROOT / 'output' in root.parents and not (root / 'protected_before.json').exists(), 'fresh scoped root required')
    root.mkdir(parents=True, exist_ok=True)
    protected = read_json(PREVIOUS / 'protected_before.json')
    for directory in (PREVIOUS, STANDARD / 'GSE241132'):
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                protected[str(path.relative_to(ROOT))] = sha256_file(path)
    require(all(sha256_file(ROOT / k) == h for k, h in protected.items()), 'historical baseline drift')
    # Recheck the previous final seal itself, without invoking its old code-bound receipt.
    inventory = read_json(PREVIOUS / 'evidence/evidence_hashes.json')
    require(all(sha256_file(PREVIOUS / k) == h for k, h in inventory.items()), 'previous sealed evidence drift')
    atomic_write_json(root / 'protected_before.json', protected)
    atomic_write_json(root / 'scientific_source.json', {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files()})
    policy = read_json(PREVIOUS / 'resource_policy.json')
    policy.update(experiment_root=str(root), runner_sha256=sha256_file(ROOT / 'scripts/run_bounded_local_diagnostics.py'),
        extension_sha256=sha256_file(__file__), spec_sha256=sha256_file(SPEC),
        runner_tests_sha256=sha256_file(ROOT / 'tests/test_sdd_local_validation.py'))
    validate_policy(policy)
    atomic_write_json(root / 'resource_policy.json', policy)
    from scripts.run_bounded_local_diagnostics import sample_host
    atomic_write_json(root / 'host_before.json', sample_host(root, gpu_required=False))
    contracts = _load_contracts(STANDARD, ['GSE241132'])
    require(contracts['GSE241132']['contract']['status'] == 'accepted' and contracts['GSE241132'].get('probe'),
            'local accepted Stage-A input/probe missing')
    require(sha256_file(contracts['GSE241132']['dataset_file']) ==
            '620aa6980e8ee5c6846532b65b22e860bca1faa1b64f54c17bac553315624848', 'GSE241132 input drift')
    summaries = {}
    for name, p, material in [('topic-inherited', topic_protocol('inherited'), {}),
                             ('topic-training', topic_protocol('training'), {}),
                             ('population', population_protocol(), contracts)]:
        folder = root / 'runs' / name
        summaries[name] = plan_run(p, material, folder, root=ROOT)
        jobs = read_jobs(folder / 'jobs.jsonl')
        if name == 'population':
            require(len(jobs) == 60 and {j.task['held_individual'] for j in jobs} == {'PWH26', 'PWH27', 'PWH28'},
                    'all-donor paired scope is not complete')
            require(all(len(j.task['pairs']) == 3 for j in jobs), 'missing time pairs')
        else:
            require(len(jobs) == (4 if name.endswith('inherited') else 2), 'topic scope is not complete')
        atomic_write_json(folder / 'execution_approval.json', {**approval_binding(folder), 'approved': True,
            'stage_a_reviewed': True, 'approved_by': 'user_explicit_current_thread_scope_recorded_by_codex',
            'approved_utc': stamp(), 'reviewed_dataset_ids': sorted({j.dataset_id for j in jobs}),
            'review_notes': USER_REQUEST + ' Local fixed SDD diagnostic only; excludes payload downloads, remote GPU, '
                'formal grid, publication and outcome tuning. Leader reviewed current inputs/Stage A; '
                'this operator receipt is not an external clinical/author signature.'})
        atomic_write_json(folder / 'frozen_jobs.json', [j.to_dict() for j in jobs])
    atomic_write_json(root / 'preparation.json', dict(utc=stamp(), code_digest=code_digest(),
        spec_sha256=sha256_file(SPEC), implementation_hashes=implementation_hashes(),
        user_request=USER_REQUEST, plans=summaries, protected_files=len(protected)))
    return summaries


def validate_lane(args, root):
    from expansion.runner.jobs import read_jobs
    root, run = Path(root).resolve(), Path(args.run_dir).resolve()
    name = run.name
    expected_names = ('topic-inherited', 'topic-training') if args.mode == 'sdd_topic' else ('population',)
    require(name in expected_names and run == root / 'runs' / name and
        Path(args.approval).resolve() == run / 'execution_approval.json', 'SDD sealed paths differ')
    expected = topic_protocol(name.split('-')[1]) if args.mode == 'sdd_topic' else population_protocol()
    require(read_json(run / 'protocol.json') == expected, 'SDD fixed protocol drift')
    jobs = read_jobs(run / 'jobs.jsonl')
    require([j.to_dict() for j in jobs] == read_json(run / 'frozen_jobs.json'), 'SDD frozen jobs drift')
    require(isinstance(args.max_jobs, int) and not isinstance(args.max_jobs, bool) and 1 <= args.max_jobs <= len(jobs),
            'SDD max-jobs out of fixed scope')
    require(read_json(root / 'preparation.json')['code_digest'] == code_digest(), 'SDD scientific code drift')
    require(read_json(root / 'preparation.json')['implementation_hashes'] == implementation_hashes(), 'SDD implementation dependency drift')
    require(read_json(root / 'resource_policy.json')['spec_sha256'] == sha256_file(SPEC), 'SDD specification drift')


def run_worker(args, out):
    from expansion.runner import cli
    import torch
    events = []
    factory = cli.make_executor

    def make(*a, **kw):
        execute = factory(*a, **kw)

        @functools.wraps(execute)
        def checked(job, directory):
            started = time.perf_counter()
            try:
                result = execute(job, directory)
                if job.job_type == 'stability':
                    require(len(result['training_trace']) == job.params['steps'], 'incomplete topic optimizer trace')
                else:
                    require(len(result['records']) == 3 and all(r['finite'] for r in result['records']),
                            'incomplete/nonfinite paired outputs')
                    if job.method.startswith('cfm_sb_'):
                        require(len(result['fit']['sb_batch_diagnostics']) == result['fit']['steps'], 'missing SB batch records')
                    if job.method == 'target_marginal':
                        result['baseline_sampling'] = {'seed': job.seed, 'generator_seed_sequence': [job.seed, 29],
                            'allocation_rule': 'equal_training_donor_cell_counts_remainder_in_sorted_donor_order',
                            'scope': 'fixed_single_draw_not_sampling_free_or_stochastic_stability_comparison'}
            except BaseException as error:
                atomic_write_json(directory / 'failed.json', dict(job_id=job.job_id, utc=stamp(),
                    error=f'{type(error).__name__}: {error}', scope='SDD_abort_no_retry'))
                raise SystemExit(f'SDD job failed; stop lane: {error}') from error
            torch.cuda.synchronize()
            events.append(dict(job_id=job.job_id, seconds=time.perf_counter() - started,
                               method=job.method, seed=job.seed, steps=job.params.get('steps', result.get('fit', {}).get('steps'))))
            atomic_write_json(out / 'job_events.json', events)
            print(f"finished {job.job_id}: {job.method} seed={job.seed} steps={events[-1]['steps']}", flush=True)
            return result
        return checked
    cli.make_executor = make
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
