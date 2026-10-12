#!/usr/bin/env python3
"""Run final regression or seal CLOSED local evidence; no fitting/publication."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expansion.common.io import atomic_write_json, read_json, sha256_file
from expansion.common.provenance import code_digest, code_files


def regression(root):
    evidence = root / 'evidence'
    evidence.mkdir(exist_ok=True)
    path = evidence / 'repository-tests-final.txt'
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='4')
    with path.open('x') as log:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=300)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'exit_code': result.returncode,
        'seconds': time.perf_counter() - started, 'log_sha256': sha256_file(path),
        'log_closed': True, 'command': [sys.executable, '-m', 'pytest', '-q']}
    match = re.search(r'(\d+) passed', path.read_text())
    report['passed_tests'] = int(match.group(1)) if match else None
    atomic_write_json(evidence / 'regression.json', report)
    print(path.read_text(), flush=True)
    return result.returncode


def seal(root):
    if (root / 'FINAL_STATUS.json').exists():
        raise FileExistsError('final seal already exists')
    evidence = root / 'evidence'
    regression = read_json(evidence / 'regression.json')
    if regression['exit_code'] != 0 or sha256_file(evidence / 'repository-tests-final.txt') != regression['log_sha256']:
        raise ValueError('final regression failed or its closed log changed')
    if not regression.get('passed_tests'):
        raise ValueError('final regression does not report executed passing tests')
    protected = read_json(root / 'protected_before.json')
    if any(sha256_file(ROOT / path) != h for path, h in protected.items()):
        raise ValueError('protected input/manuscript/archive drift')
    if read_json(root / 'scientific_source.json') != {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files()}:
        raise ValueError('scientific source drift since planning')
    report = read_json(root / 'verification-complete/verification.json')
    if report['status'] != 'passed_diagnostic_only':
        raise ValueError('final reconstruction failed')
    for folder in (root / 'profiles').iterdir():
        if not folder.is_dir():
            continue
        pid = read_json(folder / 'supervisor_result.json')['pid']
        if pid and Path(f'/proc/{pid}').exists():
            raise ValueError(f'profile worker {pid} has not exited; do not seal live evidence')
    if any((root / 'runs').glob('*/jobs/*/running.json')):
        raise ValueError('a job still has a running marker')
    snapshot = root / 'source-checkpoint-final'
    manifest = read_json(snapshot / 'manifest.json')
    if any(sha256_file(ROOT / path) != info['sha256'] for path, info in manifest['files'].items()):
        raise ValueError('authored-source snapshot no longer matches current files')
    archive_sha = sha256_file(snapshot / 'authored-source.tar')
    if not (snapshot / 'authored-source.tar.sha256').read_text().startswith(archive_sha):
        raise ValueError('source archive checksum mismatch')
    audit = read_json(root / 'saved-topic-null-audit-v2/audit.json')
    if audit['audit_script_sha256'] != sha256_file(ROOT / 'scripts/audit_saved_topic_null.py'):
        raise ValueError('post-hoc audit script drift')
    for name in ('fits', 'training_prefix_checks'):
        if audit[name] != read_json(root / 'saved-topic-null-audit/audit.json')[name]:
            raise ValueError('repeated saved-model audit produced different values')
    figure_path = root / 'diagnostic-figures-v2/figure_sources.json'
    figures = read_json(figure_path)
    if figures['null_audit_sha256'] != sha256_file(root / 'saved-topic-null-audit-v2/audit.json'):
        raise ValueError('diagnostic figures are not bound to the final null audit')
    if any(sha256_file(figure_path.parent / name) != h for name, h in figures['files'].items()):
        raise ValueError('diagnostic figure file changed')
    from expansion.runner.jobs import read_jobs
    from expansion.runner.queue import job_dir
    population = root / 'runs/population'
    sb = [read_json(job_dir(population, job.job_id) / 'result.json')['fit']['sb_batch_diagnostics']
          for job in read_jobs(population / 'jobs.jsonl') if job.method == 'cfm_sb']
    batches = [row for rows in sb for row in rows]
    if len(batches) != 240 or not all(row['both_marginals_converged'] for row in batches):
        raise ValueError('real SB summary differs from actual batch evidence')
    nonconverged = report['numerical']['nonconverged_static']
    from scripts.run_bounded_local_diagnostics import output_bytes
    status = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed_bounded_diagnostic_only',
        'code_digest': code_digest(), 'topic_fits': 6, 'population_jobs': 10, 'population_pairs': 30,
        'static_solver_fixtures': len(report['numerical']['fixtures']), 'retained_nonconverged_static': nonconverged,
        'real_sb_batches': len(batches), 'real_sb_batches_both_marginals_converged': len(batches),
        'replay_new_fits': 0, 'protected_files_unchanged': len(protected),
        'final_regression': regression, 'source_checkpoint_files': manifest['file_count'],
        'internal_source_archive_sha256': archive_sha,
        'supervised_seconds_excluding_replay': sum(v['supervisor']['seconds'] for k, v in report['profiles'].items()
                                                  if not k.endswith('replay')),
        'maximum_rss_high_water_bytes': max(v['worker']['rss_high_water_bytes'] for v in report['profiles'].values()),
        'maximum_reserved_cuda_bytes': max(v['worker']['cuda_peak_reserved_bytes'] for v in report['profiles'].values()),
        'new_output_bytes_before_seal': output_bytes(root),
        'evidence_policy': 'All computation/log writers exited before sealing; seal/status files excluded from their own hash inventory.',
        'no_new_data_or_remote_gpu': True, 'no_manuscript_or_publication_mutation': True,
        'new_independent_review': False, 'scope': 'not_a_formal_grid_clinical_validation_or_software_release'}
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'FINAL_STATUS.json'
              and p != evidence / 'evidence_hashes.json'}
    atomic_write_json(evidence / 'evidence_hashes.json', hashes)
    atomic_write_json(root / 'FINAL_STATUS.json', status)
    # Verify the inventory immediately, after atomic writes close.
    if any(sha256_file(root / name) != value for name, value in hashes.items()):
        raise ValueError('evidence changed during final seal')
    print(status, flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['regression', 'seal'])
    parser.add_argument('--experiment-root', required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root).resolve()
    if ROOT / 'output' not in root.parents:
        raise ValueError('use a task-owned output child')
    return regression(root) if args.command == 'regression' else seal(root)


if __name__ == '__main__':
    raise SystemExit(main())
