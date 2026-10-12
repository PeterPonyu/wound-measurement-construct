"""Adversarial numerical and patient-isolation checks using generated data only."""
import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from expansion.common.io import content_hash
from expansion.measurement import stability, worker
from expansion.population.methods import create
from expansion.population.methods.flows import _sinkhorn_log_plan
from expansion.population.tasks import MethodInputs
from expansion.runner.jobs import plan_measurement_jobs


def legacy_plan(cost, epsilon, iterations, tolerance, stage_iterations=50):
    """Frozen pre-instrumentation valid-input algorithm for trajectory comparison."""
    n, m = cost.shape
    log_a = torch.full((n,), -math.log(n), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((m,), -math.log(m), device=cost.device, dtype=cost.dtype)
    f, g = torch.zeros(n, dtype=cost.dtype, device=cost.device), torch.zeros(m, dtype=cost.dtype, device=cost.device)
    schedule, current = [], max(float(cost.mean()), float(epsilon))
    while current > epsilon:
        schedule.append(current)
        current /= 2
    schedule.append(float(epsilon))
    error = float('inf')
    for stage, eps in enumerate(schedule):
        budget = iterations if stage == len(schedule) - 1 else stage_iterations
        for iteration in range(budget):
            f = eps * (log_a - torch.logsumexp((g[None, :] - cost) / eps, dim=1))
            g = eps * (log_b - torch.logsumexp((f[:, None] - cost) / eps, dim=0))
            if iteration % 10 == 9 or iteration == budget - 1:
                rows = torch.logsumexp((f[:, None] + g[None, :] - cost) / eps, dim=1)
                error = float((rows.exp() - log_a.exp()).abs().sum())
                if error < tolerance:
                    break
    return (f[:, None] + g[None, :] - cost) / float(epsilon), error


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('budget', [1, 30, 300])
def test_sinkhorn_returned_plan_complete_accounting_and_legacy_exactness(dtype, budget):
    cost = torch.tensor(np.square(np.linspace(-7, 7, 11)[:, None] - np.linspace(-4, 8, 17)[None, :]), dtype=dtype)
    expected, error = legacy_plan(cost, .125, budget, .001)
    before = torch.get_rng_state().clone()
    plan, reported, d = _sinkhorn_log_plan(cost, .125, budget, .001, return_diagnostics=True)
    assert torch.equal(plan, expected) and reported == error
    assert torch.equal(torch.get_rng_state(), before)
    p = plan.exp()
    assert d['row_l1'] == pytest.approx(float((p.sum(1) - 1 / 11).abs().sum()), abs=1e-7)
    assert d['column_l1'] == pytest.approx(float((p.sum(0) - 1 / 17).abs().sum()), abs=1e-7)
    assert d['mass'] == float(p.sum())
    assert d['total_iterations'] == d['warm_iterations'] + d['target_iterations']
    assert d['target_iterations'] <= budget
    assert d['stages'][-1]['target_stage'] and d['stages'][-1]['epsilon'] == .125
    assert d['both_marginals_converged'] == (max(d['row_l1'], d['column_l1']) <= .001)
    if budget == 1:
        assert d['target_budget_exhausted'] and d['nonconverged_at_target_budget']


@pytest.mark.parametrize('options', [dict(epsilon=0), dict(epsilon=-1), dict(epsilon=float('nan')),
    dict(tolerance=0), dict(tolerance=float('inf')), dict(iterations=0), dict(iterations=1.5),
    dict(iterations=True), dict(stage_iterations=0)])
def test_bad_solver_options_fail_before_schedule(options):
    p = dict(epsilon=.1, iterations=30, tolerance=.001, stage_iterations=50)
    p.update(options)
    with pytest.raises(ValueError):
        _sinkhorn_log_plan(torch.zeros(3, 4), **p)


@pytest.mark.parametrize('cost', [torch.empty(0, 3), torch.ones(3), torch.ones(3, 4, dtype=torch.long),
                                  torch.tensor([[float('nan')]]), torch.tensor([[float('inf')]]), torch.tensor([[-1.]])])
def test_bad_solver_costs_fail_closed(cost):
    with pytest.raises(ValueError):
        _sinkhorn_log_plan(cost, .1, 10, .001)


def test_every_sb_training_batch_retained_without_rng_or_prediction_change(monkeypatch):
    rng = np.random.default_rng(2)
    inputs = MethodInputs('T-LIO', {(p, t): rng.normal(size=(24, 3)).astype('float32') + t
                                    for p in ('a', 'b') for t in (0., 1.)}, {0.: 0., 1.: 1.}, None, 'held')
    p = dict(steps=12, batch_size=16, hidden=16, depth=1, time_dim=8, sinkhorn_iterations=1)
    first = create('cfm_sb', **p)
    info = first.fit(inputs, seed=3)
    records = info['sb_batch_diagnostics']
    assert len(records) == 12 and [r['step'] for r in records] == list(range(1, 13))
    assert info['sb_solver_summary']['batches'] == 12
    assert info['sb_solver_summary']['nonconverged_batches'] > 0
    original = _sinkhorn_log_plan

    def legacy_with_reporting(cost, eps, iterations, tolerance, **kw):
        plan, error = legacy_plan(cost, eps, iterations, tolerance)
        d = original(cost, eps, iterations, tolerance, return_diagnostics=True)[2]
        return (plan, error, d) if kw.get('return_diagnostics') else (plan, error)
    monkeypatch.setattr('expansion.population.methods.flows._sinkhorn_log_plan', legacy_with_reporting)
    second = create('cfm_sb', **p)
    second.fit(inputs, seed=3)
    for key, value in first.field.state_dict().items():
        assert torch.equal(value, second.field.state_dict()[key])
    source = inputs.train[('a', 0.)]
    assert np.array_equal(first.predict(source, individual='held', source_time=0., target_time=1.),
                          second.predict(source, individual='held', source_time=0., target_time=1.))


def partition_fixture():
    cells = pd.DataFrame({'cell_id': [f'c{i}' for i in range(40)], 'patient_id': np.repeat(['a', 'b', 'c', 'd'], 10),
                          'group': np.repeat(['healed', 'amputation', 'unknown', 'control'], 10)})
    held = sorted(['a', 'b', 'c', 'd'], key=lambda p: (content_hash(['outcome-blind-patient-partition-v1', 7, p], 64), p))[:1]
    config = dict(method='id_hash_v1', cell_identity_column='cell_id', seed=7, n_held_patients=1,
                  expected_held_patients=held, evaluation_cells_per_patient=4,
                  expected_cell_order_identity=content_hash(cells.cell_id.tolist(), 64))
    return cells, config


def test_patient_split_is_order_bound_outcome_blind_and_disjoint():
    cells, config = partition_fixture()
    train, rows, record = stability.patient_partition(cells, config)
    assert set(cells.loc[train, 'patient_id']).isdisjoint(record['held_patients'])
    assert len(rows['train']) == 12 and len(rows['held']) == 4
    assert not train[rows['held']].any() and train[rows['train']].all()
    cells['group'] = 'totally_changed_outcome'
    assert stability.patient_partition(cells, config)[2] == record
    with pytest.raises(ValueError, match='cell order'):
        stability.patient_partition(cells.iloc[::-1], config)
    with pytest.raises(ValueError, match='held-ID'):
        stability.patient_partition(cells, {**config, 'expected_held_patients': ['not_there']})


def test_deterministic_evaluation_preserves_full_training_trajectory_and_rng():
    rng = np.random.default_rng(3)
    x = sp.csr_matrix(rng.dirichlet(np.ones(10), size=32).astype('float32'))
    options = dict(K=3, seed=5, steps=11, batch_size=8, lr=.001, kl_weight=.01)
    first = stability.train_topic_model(x, **options)
    after = torch.get_rng_state().clone()
    sets = {name: {'proportions': x[:8], 'units': ['a'] * 4 + ['b'] * 4} for name in ('train', 'held')}
    second = stability.train_topic_model(x, **options, diagnostic_inputs=sets, diagnostic_every=3)
    assert torch.equal(after, torch.get_rng_state())
    assert all(torch.equal(v, second['state'][k]) for k, v in first['state'].items())
    assert first['losses'] == second['losses']
    assert [r['step'] for r in second['training_trace']] == list(range(1, 12))
    assert [r['step'] for r in second['evaluation_trace']] == [0, 3, 6, 9, 11]
    assert second['evaluation_trace'][-1]['held']['patients'] == 2


def test_budget_jobs_are_complete_distinct_and_ordered():
    protocol = dict(semantics={'families': []}, stability={'K': [15], 'seeds': [0, 1], 'step_budgets': [500, 2000, 5000]},
                    design={'enabled': False})
    jobs = plan_measurement_jobs(protocol, {})
    assert [(j.params['steps'], j.seed) for j in jobs] == [(b, s) for b in (500, 2000, 5000) for s in (0, 1)]
    assert len({j.job_id for j in jobs}) == 6
    with pytest.raises(ValueError):
        plan_measurement_jobs({**protocol, 'stability': {**protocol['stability'], 'step_budgets': [5, 5]}}, {})


def test_worker_passes_only_training_patient_rows_to_optimizer(monkeypatch):
    cells, config = partition_fixture()
    rng = np.random.default_rng(6)
    x = sp.csr_matrix(rng.dirichlet(np.ones(8), size=len(cells)).astype('float32'))
    mask, _, _ = stability.patient_partition(cells, config)
    monkeypatch.setattr(stability, 'load_discovery_panel_counts', lambda _: (x, cells))
    real_train = stability.train_topic_model

    def checked(values, **kwargs):
        assert (values != x[mask]).nnz == 0 and values.shape[0] == mask.sum()
        return real_train(values, **kwargs)
    monkeypatch.setattr(stability, 'train_topic_model', checked)
    monkeypatch.setattr(worker.np, 'load', lambda *a, **k: rng.dirichlet(np.ones(8), size=3))
    monkeypatch.setattr(stability, 'match_topics', lambda *a: {'assignment': {}, 'assigned_correlation': {},
                  'best_match': {0: 1}, 'best_correlation': {0: .3}, 'unmatched_reference': [0]})
    protocol = {'stability': dict(step_budgets=[3], K=[3], seeds=[0], patient_partition=config,
                 batch_size=8, lr=.001, kl_weight=.01, diagnostic_every=2), 'semantics': {}}
    result = worker.run_stability_job({'params': {'K': 3, 'steps': 3}, 'seed': 0}, protocol=protocol, root='.')
    assert result['patient_partition']['held_patients_excluded_from_optimization']
    assert result['steps'] == 3 and result['fibroblast_topic'] is None
    assert result['primary_readout_status'] == 'reference_topic_0_unmatched_no_primary_readout'


def test_stage_profiling_does_not_change_flow_state():
    rng = np.random.default_rng(4)
    inputs = MethodInputs('T-LIO', {(p, t): rng.normal(size=(16, 2)).astype('float32')
                          for p in ('a', 'b') for t in (0., 1.)}, {0.: 0., 1.: 1.}, None, 'held')
    params = dict(steps=4, hidden=12, depth=1, batch_size=8, time_dim=8)
    a, b = create('cfm_independent', **params), create('cfm_independent', **params, profile_stages=True)
    a.fit(inputs, seed=2)
    info = b.fit(inputs, seed=2)
    assert all(torch.equal(v, b.field.state_dict()[k]) for k, v in a.field.state_dict().items())
    assert all(v >= 0 for v in info['stage_seconds_synchronized'].values())


def test_followup_wrapper_rejects_grid_expansion_and_changed_paths(tmp_path, monkeypatch):
    import argparse
    from expansion.common.io import atomic_write_json
    from expansion.runner.jobs import write_jobs
    from scripts import run_local_convergence_followup as followup
    root, folder = tmp_path, tmp_path / 'runs/topic'
    folder.mkdir(parents=True)
    protocol = {'kind': 'measurement', 'semantics': {'families': []},
                'stability': {'K': [15], 'seeds': [0, 1], 'step_budgets': [500, 2000, 5000]},
                'design': {'enabled': False}}
    monkeypatch.setattr(followup, 'measurement_protocol', lambda: protocol)
    atomic_write_json(folder / 'protocol.json', protocol)
    atomic_write_json(folder / 'datasets.json', {})
    write_jobs(folder / 'jobs.jsonl', plan_measurement_jobs(protocol, {}))
    args = argparse.Namespace(mode='topic_convergence', run_dir=str(folder),
           approval=str(folder / 'execution_approval.json'), max_jobs=1)
    followup.validate_followup_lane(args, root)
    args.max_jobs = 7
    with pytest.raises(ValueError, match='max-jobs'):
        followup.validate_followup_lane(args, root)
    args.max_jobs, args.approval = 1, str(root / 'unrelated.json')
    with pytest.raises(ValueError, match='paths'):
        followup.validate_followup_lane(args, root)


def test_current_scoped_protocol_includes_all_four_flow_methods_and_two_baselines():
    from scripts.run_local_convergence_followup import population_protocol
    p = population_protocol()
    assert set(p['methods']) == {'cfm_independent', 'cfm_ot', 'cfm_sb', 'cfm_conditioned',
                                'location_scale', 'target_marginal'}
    assert p['flow_defaults']['steps'] == 120 and p['flow_defaults']['sinkhorn_iterations'] == 300
    assert p['analysis']['report_mode'] == 'diagnostic_only'
