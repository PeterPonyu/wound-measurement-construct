#!/usr/bin/env python3
"""Bounded local diagnostics; real-data execution always uses the normal receipt gate.

The supervisor controls only its new process group. Resource checks are sampled
operational limits, not cgroup hard quotas. No download, approval creation, retry,
or scientific claim selection is performed by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, payload):
    from expansion.common.io import atomic_write_json
    atomic_write_json(path, payload)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_policy(policy):
    positive = ('cpu_threads', 'rss_limit_gib', 'available_ram_floor_gib',
                'disk_free_floor_gib', 'output_limit_gib', 'wall_seconds',
                'sample_seconds', 'gpu_free_floor_mib', 'gpu_process_limit_mib',
                'gpu_temperature_limit_c')
    for key in positive:
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid resource policy: {key}')
    fraction = policy['cuda_allocator_fraction']
    if (not isinstance(policy['cpu_threads'], int) or policy['cpu_threads'] > 4
            or isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= .25):
        raise ValueError('invalid CPU threads or CUDA caching-allocator fraction')
    if policy['sample_seconds'] > 5 or policy['wall_seconds'] > 600 or policy['output_limit_gib'] > 2:
        raise ValueError('policy exceeds the bounded diagnostic lane')
    return policy


def tree_rss_bytes(pid):
    """Sum current RSS only for this process and its descendants; do not mutate them."""
    seen, pending, total = set(), [int(pid)], 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            text = Path(f'/proc/{current}/status').read_text()
            rss = next((line.split()[1] for line in text.splitlines() if line.startswith('VmRSS:')), '0')
            total += int(rss) * 1024
            children = Path(f'/proc/{current}/task/{current}/children').read_text().split()
            pending.extend(map(int, children))
        except FileNotFoundError:
            pass  # process exited during the snapshot
    return total


def output_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def sample_host(root, *, pid=None, gpu_required=True):
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    sample = {'utc': stamp(), 'available_ram_bytes': int(memory['MemAvailable'].split()[0]) * 1024,
              'disk_free_bytes': shutil.disk_usage(root).free,
              'output_bytes': output_bytes(root), 'process_tree_rss_bytes': tree_rss_bytes(pid) if pid else 0}
    if gpu_required:
        command = ['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu,temperature.gpu',
                   '--format=csv,noheader,nounits', '--id=0']
        line = subprocess.check_output(command, text=True, timeout=5).strip().split(', ')
        sample['gpu'] = dict(zip(('uuid', 'name', 'total_mib', 'free_mib', 'utilization_percent', 'temperature_c'), line))
        for key in ('total_mib', 'free_mib', 'utilization_percent', 'temperature_c'):
            sample['gpu'][key] = float(sample['gpu'][key])
        sample['gpu']['own_process_mib'] = None
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                             '--format=csv,noheader,nounits', '--id=0'], text=True, timeout=5)
        for row in processes.splitlines():
            fields = [item.strip() for item in row.split(',')]
            if pid is not None and len(fields) == 2 and fields[0] == str(pid):
                try:
                    sample['gpu']['own_process_mib'] = float(fields[1])
                except ValueError:
                    pass
    return sample


def resource_violations(sample, policy, elapsed):
    failures = []
    checks = ((elapsed > policy['wall_seconds'], 'wall_time_limit'),
              (sample['process_tree_rss_bytes'] > policy['rss_limit_gib'] * 2**30, 'process_tree_rss_limit'),
              (sample['available_ram_bytes'] < policy['available_ram_floor_gib'] * 2**30, 'host_ram_floor'),
              (sample['disk_free_bytes'] < policy['disk_free_floor_gib'] * 2**30, 'disk_free_floor'),
              (sample['output_bytes'] > policy['output_limit_gib'] * 2**30, 'new_output_limit'))
    failures.extend(name for failed, name in checks if failed)
    gpu = sample.get('gpu')
    if gpu:
        if gpu['free_mib'] < policy['gpu_free_floor_mib']:
            failures.append('gpu_free_floor')
        if gpu['temperature_c'] >= policy['gpu_temperature_limit_c']:
            failures.append('gpu_temperature_limit')
        if gpu.get('own_process_mib') is not None and gpu['own_process_mib'] > policy['gpu_process_limit_mib']:
            failures.append('gpu_own_process_limit')
    return failures


def validate_measurement_lane(run_dir):
    from expansion.common.io import read_json
    from expansion.runner.jobs import read_jobs
    run_dir = Path(run_dir)
    protocol = read_json(run_dir / 'protocol.json')
    jobs = read_jobs(run_dir / 'jobs.jsonl')
    expected = dict(K=[10, 15], seeds=[0, 1], steps=200, batch_size=128, lr=.001, kl_weight=.01)
    semantics = protocol.get('semantics', {})
    if (protocol.get('kind') != 'measurement' or protocol.get('smoke_only') is not True
            or protocol.get('cohorts') != {'discovery': 'GSE165816', 'external': []}
            or protocol.get('analysis') != {'report_mode': 'diagnostic_only', 'inferential_unit': 'none_diagnostic_only'}
            or any(protocol.get('stability', {}).get(k) != v for k, v in expected.items())
            or semantics.get('n_splits') != 2 or semantics.get('max_cells') != 400
            or semantics.get('min_cells') != 50 or semantics.get('min_positive') != 20
            or semantics.get('families') != [] or semantics.get('discovery_only_families') != []
            or protocol.get('design', {}).get('enabled') is not False
            or read_json(run_dir / 'datasets.json') != {}
            or len(jobs) != 4 or any(j.job_type != 'stability' or j.dataset_id != 'GSE165816' for j in jobs)
            or {(j.params.get('K'), j.seed) for j in jobs} != {(10, 0), (10, 1), (15, 0), (15, 1)}):
        raise ValueError('sealed plan exceeds the approved four-fit measurement diagnostic lane')


def configure_worker(policy):
    import torch
    torch.set_num_threads(policy['cpu_threads'])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('The approved diagnostic lane requires the verified local CUDA device')
    torch.cuda.set_per_process_memory_fraction(policy['cuda_allocator_fraction'], 0)
    torch.cuda.reset_peak_memory_stats(0)
    return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'allowed_cpu_ids': sorted(os.sched_getaffinity(0)), 'pid': os.getpid(),
            'cuda_allocator_fraction': policy['cuda_allocator_fraction'],
            'allocator_limit_scope': 'PyTorch caching allocator only; not all driver/context allocations',
            'env': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                        'NUMEXPR_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}}


def measurement(args, out):
    """Instrument existing functions without changing arguments, outputs or authorization."""
    import torch
    from expansion.measurement import stability, semantics
    from expansion.runner import cli

    validate_measurement_lane(args.run_dir)
    events, current = [], {'job_id': None, 'depth': 0}

    def instrument(module, name):
        original = getattr(module, name)

        @functools.wraps(original)
        def measured(*values, **kwargs):
            if name == 'train_topic_model':
                torch.cuda.synchronize()
            started, depth = time.perf_counter(), current['depth']
            current['depth'] += 1
            outcome = 'finished'
            try:
                return original(*values, **kwargs)
            except BaseException:
                outcome = 'failed'
                raise
            finally:
                if name == 'train_topic_model':
                    torch.cuda.synchronize()
                current['depth'] -= 1
                events.append({'job_id': current['job_id'], 'stage': name, 'nested_depth': depth,
                               'seconds': time.perf_counter() - started, 'status': outcome})
        setattr(module, name, measured)

    for name in ('load_discovery_panel_counts', 'train_topic_model', 'match_topics', 'project_theta',
                 'save_inference_checkpoint', 'gmm_midpoint'):
        instrument(stability, name)
    instrument(semantics, 'semantics_test')
    original_factory = cli.make_executor

    def factory(*values, **kwargs):
        execute = original_factory(*values, **kwargs)

        def profiled(job, folder):
            current['job_id'] = job.job_id
            started = time.perf_counter()
            try:
                return execute(job, folder)
            finally:
                torch.cuda.synchronize()
                events.append({'job_id': job.job_id, 'stage': 'whole_job', 'nested_depth': -1,
                               'seconds': time.perf_counter() - started})
                save(out / 'stage_events.json', events)
        return profiled
    cli.make_executor = factory
    return cli.execute_run(args.run_dir, root=ROOT, device='cuda', max_jobs=args.max_jobs,
                           approval_path=args.approval, retry_failed=False)


def synthetic(out):
    """Fixed tiny fixtures only; no real inputs, outcome selection or method ranking."""
    import numpy as np
    import torch
    from expansion.common.io import atomic_save_npz
    from expansion.population.methods import create
    from expansion.population.methods.flows import _sinkhorn_log_plan
    from expansion.population.tasks import MethodInputs
    from expansion.population.metrics import energy_distance

    rng = np.random.default_rng(7301)
    fixtures = {'uniform': np.zeros((24, 30)),
                'smooth': np.square(rng.normal(size=(24, 1)) - rng.normal(size=(1, 30))),
                'concentrated': np.square(np.linspace(-5, 5, 24)[:, None] - np.linspace(-5, 5, 30)[None, :])}
    residuals = []
    for name, cost in fixtures.items():
        for dtype in (torch.float32, torch.float64):
            for iterations in (30, 300):
                started = time.perf_counter()
                plan, reported = _sinkhorn_log_plan(torch.tensor(cost, dtype=dtype), .125, iterations, 1e-3)
                probability = plan.exp()
                row = float((probability.sum(1) - 1 / len(cost)).abs().sum())
                column = float((probability.sum(0) - 1 / cost.shape[1]).abs().sum())
                residuals.append({'fixture': name, 'dtype': str(dtype), 'target_iterations': iterations,
                                  'epsilon': .125, 'tolerance': 1e-3, 'row_l1': row, 'column_l1': column,
                                  'reported_row_l1': reported, 'mass': float(probability.sum()),
                                  'finite_log_plan': bool(torch.isfinite(plan).all()),
                                  'both_marginals_within_tolerance': max(row, column) <= 1e-3,
                                  'seconds': time.perf_counter() - started})
    save(out / 'sinkhorn_residuals.json', residuals)

    rng = np.random.default_rng(7302)
    train = {(individual, timepoint): (rng.normal(0, .5, (96, 2)) + [2 * timepoint, 0]).astype(np.float32)
             for individual in ('a', 'b') for timepoint in (0., 1., 2.)}
    source = rng.normal(0, .5, (96, 2)).astype(np.float32)
    target = source + [2., 0.]
    inputs = MethodInputs('T-LIO', train, {0.: 0., 1.: .5, 2.: 1.}, None, 'synthetic_held')
    atomic_save_npz(out / 'synthetic_inputs.npz', source=source, target=target,
                    **{f'{ind}_{int(t)}': x for (ind, t), x in train.items()})
    results, predictions = [], {}
    for steps in (40, 120):
        for name in ('cfm_independent', 'cfm_ot', 'cfm_sb', 'cfm_conditioned'):
            p = dict(hidden=48, depth=2, time_dim=8, steps=steps, batch_size=64, lr=.003,
                     rk4_steps_per_interval=20, sinkhorn_iterations=300, sinkhorn_tolerance=1e-3)
            method = create(name, **p)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            fit = method.fit(inputs, seed=7, device='cuda')
            torch.cuda.synchronize()
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            prediction = method.predict(source, individual='synthetic_held', source_time=0., target_time=1.)
            torch.cuda.synchronize()
            predict_seconds = time.perf_counter() - started
            started = time.perf_counter()
            metrics = {'energy': float(energy_distance(prediction, target)),
                       'translation_mean_error': float(np.linalg.norm(prediction.mean(0) - target.mean(0)))}
            score_seconds = time.perf_counter() - started
            key = f'{name}_{steps}'
            predictions[key] = prediction
            results.append({'method': name, 'steps': steps, 'seed': 7, 'params': p, 'fit': fit,
                            'fit_seconds_synchronized': fit_seconds, 'predict_seconds_synchronized': predict_seconds,
                            'score_seconds': score_seconds, 'metrics': metrics, 'prediction_shape': list(prediction.shape),
                            'finite': bool(np.isfinite(prediction).all()),
                            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            save(out / 'synthetic_fits.json', results)
            atomic_save_npz(out / 'synthetic_predictions.npz', **predictions)
            print(f'synthetic {key}: {fit_seconds:.3f}s fit; finite={results[-1]["finite"]}', flush=True)
            del method
            torch.cuda.empty_cache()
    if not all(r['finite'] and math.isfinite(r['fit']['loss_last']) for r in results):
        raise ValueError('synthetic fit/prediction finite check failed')
    if not all(r['finite_log_plan'] and abs(r['row_l1'] - r['reported_row_l1']) <= 1e-5 for r in residuals):
        raise ValueError('Sinkhorn finite/reported-row consistency check failed')
    return {'synthetic': True, 'fits': len(results), 'sinkhorn_checks': len(residuals),
            'marginal_exceedances': sum(not r['both_marginals_within_tolerance'] for r in residuals),
            'scope': 'synthetic numerical/fit-cost diagnostic; no real-data inference, method pruning or convergence claim'}


def worker(args):
    import torch
    policy, out = validate_policy(json.loads(Path(args.policy).read_text())), Path(args.out)
    started = time.perf_counter()
    report = {'started_utc': stamp(), 'runner_sha256': sha(__file__)}
    try:
        report['runtime'] = configure_worker(policy)
        if args.mode in ('topic_convergence', 'population_paired', 'sinkhorn_audit'):
            from scripts.run_local_convergence_followup import run_followup_worker
            report['result'] = run_followup_worker(args, out)
        elif args.mode in ('sdd_topic', 'sdd_population'):
            from scripts.run_sdd_local_validation import run_worker
            report['result'] = run_worker(args, out)
        else:
            report['result'] = measurement(args, out) if args.mode == 'measurement' else synthetic(out)
        report['status'] = 'failed' if report['result'].get('failed', 0) or report['result'].get('skipped_failed', 0) else 'finished'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(finished_utc=stamp(), end_to_end_seconds=time.perf_counter() - started,
                      rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if torch.cuda.is_initialized():
            report.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(out / 'worker_result.json', report)
    return int(report['status'] != 'finished')


def supervise(args):
    policy = validate_policy(json.loads(Path(args.policy).read_text()))
    root, out = Path(policy['experiment_root']).resolve(), Path(args.out).resolve()
    if root not in out.parents or out.exists():
        raise ValueError('profile output must be a fresh child directory of the frozen experiment root')
    if args.mode == 'measurement':
        if not args.run_dir or not args.approval or args.max_jobs not in (1, 3):
            raise ValueError('measurement requires a receipt, sealed run and max-jobs=1 or 3')
        if root not in Path(args.run_dir).resolve().parents or root not in Path(args.approval).resolve().parents:
            raise ValueError('run and receipt must belong to this experiment root')
        validate_measurement_lane(args.run_dir)
    if args.mode in ('topic_convergence', 'population_paired', 'sinkhorn_audit'):
        from scripts.run_local_convergence_followup import validate_followup_lane
        extension = ROOT / 'scripts/run_local_convergence_followup.py'
        if policy.get('extension_sha256') != sha(extension):
            raise ValueError('follow-up runner changed since freezing the policy')
        validate_followup_lane(args, root)
        previous = [json.loads(p.read_text()) for p in (root / 'profiles').glob('*/supervisor_result.json')]
        if any(p['status'] != 'finished' for p in previous):
            raise RuntimeError('a previous follow-up invocation failed; no automatic continuation/retry')
        if (not 0 < policy.get('lane_wall_budget_seconds', 0) <= 1800
                or sum(p['seconds'] for p in previous) + policy['wall_seconds'] > policy['lane_wall_budget_seconds']):
            raise ValueError('insufficient remaining supervised lane wall budget')
    if args.mode in ('sdd_topic', 'sdd_population'):
        from scripts.run_sdd_local_validation import validate_lane
        extension = ROOT / 'scripts/run_sdd_local_validation.py'
        if policy.get('extension_sha256') != sha(extension):
            raise ValueError('SDD runner changed since freezing the policy')
        validate_lane(args, root)
        previous = [json.loads(p.read_text()) for p in (root / 'profiles').glob('*/supervisor_result.json')]
        if any(p['status'] != 'finished' for p in previous):
            raise RuntimeError('a previous SDD invocation failed; no automatic continuation/retry')
        if (not 0 < policy.get('lane_wall_budget_seconds', 0) <= 1800
                or sum(p['seconds'] for p in previous) + policy['wall_seconds'] > policy['lane_wall_budget_seconds']):
            raise ValueError('insufficient remaining supervised SDD wall budget')
    if sha(__file__) != policy['runner_sha256']:
        raise ValueError('resource runner changed since freezing the policy')
    initial = sample_host(root)
    if resource_violations(initial, policy, 0):
        raise RuntimeError(f'host resource precondition failed: {resource_violations(initial, policy, 0)}')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'policy.json', policy)
    env = dict(os.environ, OMP_NUM_THREADS=str(policy['cpu_threads']), MKL_NUM_THREADS=str(policy['cpu_threads']),
               NUMEXPR_NUM_THREADS=str(policy['cpu_threads']), OPENBLAS_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--mode', args.mode,
               '--policy', str(Path(args.policy).resolve()), '--out', str(out)]
    if args.mode in ('measurement', 'topic_convergence', 'population_paired', 'sdd_topic', 'sdd_population'):
        command += ['--run-dir', str(Path(args.run_dir).resolve()), '--approval', str(Path(args.approval).resolve()),
                    '--max-jobs', str(args.max_jobs)]
    samples, violations, started = [initial], [], time.perf_counter()
    process, supervisor_error = None, None
    try:
        with (out / 'stdout.txt').open('x') as stdout, (out / 'stderr.txt').open('x') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            while process.poll() is None:
                sample = sample_host(root, pid=process.pid)
                samples.append(sample)
                violations = resource_violations(sample, policy, time.perf_counter() - started)
                if violations:
                    break
                try:
                    process.wait(timeout=policy['sample_seconds'])
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as error:
        supervisor_error = f'{type(error).__name__}: {error}'
        raise
    finally:
        if process is not None and process.poll() is None:
            # This PID is the new session leader created above, never an unrelated host PID.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        try:
            final = sample_host(root)
        except Exception as error:
            final = sample_host(root, gpu_required=False)
            supervisor_error = supervisor_error or f'final GPU monitor failed: {type(error).__name__}: {error}'
        samples.append(final)
        violations = sorted(set(violations + resource_violations(final, policy, time.perf_counter() - started)))
        save(out / 'resource_samples.json', samples)
        report = {'started_utc': initial['utc'], 'finished_utc': stamp(), 'mode': args.mode,
                  'pid': process.pid if process else None, 'returncode': process.returncode if process else None,
                  'seconds': time.perf_counter() - started, 'violations': violations,
                  'status': 'finished' if process and process.returncode == 0 and not violations and not supervisor_error else 'failed',
                  'supervisor_error': supervisor_error,
                  'sampling_scope': 'operational checks, not cgroup hard quotas; short peaks may precede stop',
                  'runner_sha256': sha(__file__), 'policy_sha256': sha(args.policy),
                  'sampled_peak_tree_rss_bytes': max(s['process_tree_rss_bytes'] for s in samples),
                  'new_output_bytes': final['output_bytes']}
        save(out / 'supervisor_result.json', report)
        print(json.dumps(report, indent=2), flush=True)
    return int(report['status'] != 'finished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', '_worker'))
    parser.add_argument('--mode', choices=('measurement', 'synthetic', 'topic_convergence', 'population_paired',
                                         'sinkhorn_audit', 'sdd_topic', 'sdd_population'), required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--approval')
    parser.add_argument('--max-jobs', type=int)
    args = parser.parse_args()
    return worker(args) if args.command == '_worker' else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
