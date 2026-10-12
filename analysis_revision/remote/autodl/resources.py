"""Contain this task and its children; never alter another process or shared quota."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require_workloads_cleared(confirmed):
    if confirmed is not True:
        raise ValueError('prior PR/DV workloads must finish and downloaded results must be verified bytewise; '
                         'explicit --existing-workloads-cleared is required')


def _quota():
    """Read a container-visible v2 quota, falling back to the v1 CPU controller."""
    path = Path('/sys/fs/cgroup/cpu.max')
    try:
        words = path.read_text().split()
        if len(words) == 2 and words[0] != 'max':
            quota, period = map(int, words)
            if quota <= 0 or period <= 0:
                raise ValueError('invalid effective cgroup CPU quota')
            return quota / period
    except FileNotFoundError:
        pass
    for prefix in ('/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct'):
        path = Path(prefix)
        try:
            quota = int((path / 'cpu.cfs_quota_us').read_text())
            period = int((path / 'cpu.cfs_period_us').read_text())
        except FileNotFoundError:
            continue
        if quota > 0 and period > 0:
            return quota / period
    return None


def cpu_selection(allowed, maximum, *, requested=None, quota_cores=None):
    allowed = set(allowed)
    if type(maximum) is not int or maximum < 1 or not allowed:
        raise ValueError('CPU containment requires a positive limit and available affinity')
    limit = min(maximum, len(allowed))
    if quota_cores is not None:
        if quota_cores <= 0:
            raise ValueError('nonpositive CPU quota blocks execution')
        limit = min(limit, max(1, math.floor(quota_cores)))
    if requested is None:
        return sorted(allowed)[:limit]
    words = requested.split(',')
    if not words or any(not word.isascii() or not word.isdigit() for word in words):
        raise ValueError('AUTODL_CPUSET requires comma-separated numeric CPU IDs')
    selected = list(map(int, words))
    if len(selected) != len(set(selected)) or not set(selected) <= allowed or len(selected) > limit:
        raise ValueError('requested CPU set is duplicated, unavailable, or exceeds the effective CPU cap')
    return sorted(selected)


def apply_cpu_limit(root):
    policy = json.loads((Path(root) / 'remote/autodl/handoff.json').read_text())['cpu_policy']
    if policy['numerical_threads'] != 1 or policy['affinity_max_logical_cpus'] != 4:
        raise ValueError('this handoff requires one numerical thread and at most four logical CPUs')
    before = sorted(os.sched_getaffinity(0))
    quota = _quota()
    selected = cpu_selection(before, policy['affinity_max_logical_cpus'],
                             requested=os.environ.get('AUTODL_CPUSET'), quota_cores=quota)
    # Only this calling task changes its affinity; inherited child threads/processes
    # keep it. Launchers call this before importing numerical modules.
    os.sched_setaffinity(0, selected)
    if sorted(os.sched_getaffinity(0)) != selected:
        raise ValueError('CPU affinity containment did not take effect')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    return {'status': 'passed', 'affinity_before': before, 'affinity_selected': selected,
            'cgroup_quota_cores': quota, 'numerical_threads': 1,
            'hard_cgroup_quota': False, 'exclusive_cpu_reservation': False,
            'existing_processes_modified': False}
