"""Generated adversarial tests for fixed SDD scope and panel isolation."""
import argparse
import copy

import numpy as np
import pytest
import scipy.sparse as sp

from expansion.measurement import stability
from expansion.runner.jobs import plan_measurement_jobs, plan_population_jobs
from scripts import run_sdd_local_validation as sdd


def test_training_panel_held_mutations_have_no_effect_on_fit():
    rng = np.random.default_rng(411)
    x = rng.poisson(2, (36, 24))
    genes = np.array([f'GENE{i}' for i in range(24)])
    mask = np.arange(36) < 24
    config = dict(method='training_dispersion_v1', n_top=8, min_training_cells_per_gene=2, pinned_genes=['GENE23'])
    before = stability.select_training_panel(x, genes, mask, config)
    x[~mask] = rng.poisson(1000, (12, 24))
    after = stability.select_training_panel(x, genes, mask, config)
    assert np.array_equal(before[0], after[0]) and before[2] == after[2]
    assert 'GENE23' in before[1] and not before[2]['end_to_end_independent']


def test_training_support_excludes_held_only_pinned_gene():
    x = sp.csr_matrix([[2, 0, 3], [3, 0, 1], [1, 4, 1], [2, 5, 0]])
    config = dict(method='training_dispersion_v1', n_top=2, min_training_cells_per_gene=1, pinned_genes=['heldonly'])
    _, names, info = stability.select_training_panel(x, ['a', 'heldonly', 'c'], np.array([True, True, False, False]), config)
    assert 'heldonly' not in names and info['eligible_training_genes'] == 2


@pytest.mark.parametrize('bad', [np.array([0, 1]), np.array([False, False]), np.array([True]), np.array([True, True, True])])
def test_panel_masks_fail_closed(bad):
    with pytest.raises(ValueError):
        stability.select_training_panel([[1, 2], [2, 3]], ['a', 'b'], bad,
            dict(method='training_dispersion_v1', n_top=2, min_training_cells_per_gene=1, pinned_genes=[]))


def test_common_panel_matching_is_identity_aligned_not_column_position():
    a = np.array([[.1, .2, .7], [.5, .4, .1]])
    b = a[:, [2, 0, 1]]
    match = stability.match_panel_topics(a, ['a', 'b', 'c'], b, ['c', 'a', 'b'])
    assert match['assignment'] == {0: 0, 1: 1}
    assert match['assigned_correlation'][0] == pytest.approx(1)
    assert match['panel_comparison']['shared_genes'] == 3
    with pytest.raises(ValueError):
        stability.match_panel_topics(a, ['a', 'b', 'c'], b, ['x', 'y', 'c'])


def test_sdd_topic_scope_fixed_and_no_held_selection():
    a, b = sdd.topic_protocol('inherited'), sdd.topic_protocol('training')
    assert len(plan_measurement_jobs(a, {})) == 4 and len(plan_measurement_jobs(b, {})) == 2
    assert a['stability']['batch_size'] == b['stability']['batch_size'] == 128
    assert b['stability']['step_budgets'] == [20000]
    assert b['stability']['panel_selection']['pinned_genes'] == sdd.PINNED
    assert a['stability']['patient_partition'] == b['stability']['patient_partition']


def test_full_population_grid_has_all_methods_budgets_donors_and_baselines():
    p = sdd.population_protocol()
    donors = ['PWH26', 'PWH27', 'PWH28']
    pairs = [dict(source_time=a, target_time=b, individuals=donors) for a, b in [(0, 1), (1, 7), (7, 30)]]
    contract = dict(status='accepted', eligible={'T-MT': False, 'T-LIO': True},
                    lio_individuals=donors, consecutive_pairs=pairs)
    jobs = plan_population_jobs(p, {'GSE241132': dict(contract=contract, dataset_sha256='fixed')})
    assert len(jobs) == 60 and sum(len(j.task['pairs']) for j in jobs) == 180
    assert len({j.job_id for j in jobs}) == 60
    for donor in donors:
        own = [j for j in jobs if j.task['held_individual'] == donor]
        assert len(own) == 20
        for budget in (120, 600):
            assert {(p['methods'][j.method]['base'], j.seed) for j in own if j.method.endswith(f'_s{budget}')} == {
                (name, seed) for name in sdd.FLOW_BASES for seed in (0, 1)}
        assert {j.method for j in own if j.method in sdd.BASELINES} == set(sdd.BASELINES)


def test_sdd_rejects_scope_and_frozen_job_drift(tmp_path, monkeypatch):
    from expansion.common.io import atomic_write_json
    from expansion.runner.jobs import write_jobs
    run = tmp_path / 'runs/topic-inherited'
    run.mkdir(parents=True)
    p = dict(kind='measurement', semantics={'families': []}, design={'enabled': False},
             stability={'K': [3], 'seeds': [0, 1], 'step_budgets': [5, 9]})
    jobs = plan_measurement_jobs(p, {})
    monkeypatch.setattr(sdd, 'topic_protocol', lambda _: p)
    monkeypatch.setattr(sdd, 'code_digest', lambda: 'fixed')
    monkeypatch.setattr(sdd, 'sha256_file', lambda _: 'spec')
    monkeypatch.setattr(sdd, 'implementation_hashes', lambda: {'fixture': 'hash'})
    atomic_write_json(run / 'protocol.json', p)
    write_jobs(run / 'jobs.jsonl', jobs)
    atomic_write_json(run / 'frozen_jobs.json', [j.to_dict() for j in jobs])
    atomic_write_json(tmp_path / 'preparation.json', {'code_digest': 'fixed', 'implementation_hashes': {'fixture': 'hash'}})
    atomic_write_json(tmp_path / 'resource_policy.json', {'spec_sha256': 'spec'})
    args = argparse.Namespace(mode='sdd_topic', run_dir=str(run), approval=str(run / 'execution_approval.json'), max_jobs=1)
    sdd.validate_lane(args, tmp_path)
    args.max_jobs = 5
    with pytest.raises(ValueError, match='max-jobs'):
        sdd.validate_lane(args, tmp_path)
    args.max_jobs = 1
    changed = copy.deepcopy(p)
    changed['stability']['seeds'].append(2)
    atomic_write_json(run / 'protocol.json', changed)
    with pytest.raises(ValueError, match='protocol drift'):
        sdd.validate_lane(args, tmp_path)


def test_plateau_is_only_predeclared_screen_and_not_held_selection():
    from scripts.verify_sdd_local_validation import plateau_screen
    evaluations = [{'step': i * 100, 'train': {'equal_patient_objective': 5 + .0002 * i},
                    'held': {'equal_patient_objective': 4 + .01 * i}} for i in range(6)]
    assert plateau_screen(evaluations, 'train')['screen_passed']
    assert not plateau_screen(evaluations, 'held')['screen_passed']
    assert not plateau_screen(evaluations[:2], 'train')['screen_passed']


def test_saved_metric_verifier_detects_nonfinite_missing_and_numerical_corruption():
    from scripts.verify_sdd_local_validation import assert_metrics
    correct = {'patient_id': 'a', 'finite': True, 'scores': [1., 2., 3.]}
    assert assert_metrics(correct, correct, 'fixture') == 0
    for changed in ({'patient_id': 'a'}, {**correct, 'scores': [1., float('nan'), 3.]},
                    {**correct, 'scores': [1., 2., 4.]}, {**correct, 'finite': False}):
        with pytest.raises(ValueError):
            assert_metrics(changed, correct, 'fixture')


def test_training_panel_worker_uses_selected_panel_and_no_held_optimizer(monkeypatch):
    import pandas as pd
    from expansion.common.io import content_hash
    from expansion.measurement import worker
    cells = pd.DataFrame({'cell_id': [f'c{i}' for i in range(40)], 'patient_id': np.repeat(['a', 'b', 'c', 'd'], 10)})
    held = sorted(['a', 'b', 'c', 'd'], key=lambda pat: (content_hash(['outcome-blind-patient-partition-v1', 7, pat], 64), pat))[:1]
    split = dict(method='id_hash_v1', cell_identity_column='cell_id', seed=7, n_held_patients=1,
                 expected_held_patients=held, evaluation_cells_per_patient=4,
                 expected_cell_order_identity=content_hash(cells.cell_id.tolist(), 64))
    cells['celltype'] = 'other'
    rng = np.random.default_rng(813)
    x = sp.csr_matrix(rng.dirichlet(np.ones(8), size=len(cells)).astype('float32'))
    mask, _, _ = stability.patient_partition(cells, split)
    panel = np.array([f'g{i}' for i in range(8)])
    panel_fit = {'method': 'fixture', 'held_values_used_for_panel_fit': False}
    monkeypatch.setattr(stability, 'load_training_panel_counts', lambda *a: (x, cells, panel, panel_fit))
    real_train = stability.train_topic_model
    def checked(values, **kwargs):
        assert (values != x[mask]).nnz == 0
        return real_train(values, **kwargs)
    monkeypatch.setattr(stability, 'train_topic_model', checked)
    reference = rng.dirichlet(np.ones(8), size=3)
    monkeypatch.setattr(worker.np, 'load', lambda path, **kwargs: panel if str(path).endswith('panel.npy') else reference)
    p = dict(stability=dict(K=[3], seeds=[0], step_budgets=[3], patient_partition=split, panel_selection={},
                  batch_size=8, lr=.001, kl_weight=.01, diagnostic_every=2), semantics={})
    result = worker.run_stability_job({'params': {'K': 3, 'steps': 3}, 'seed': 0}, protocol=p, root='.')
    assert result['panel_fit'] == panel_fit
    assert not result['patient_partition']['end_to_end_independent']
    assert result['match_to_reference']['panel_comparison']['shared_genes'] == 8


def test_undefined_reference_match_does_not_discard_numeric_fit(monkeypatch, tmp_path):
    import pandas as pd
    from expansion.common.io import content_hash, read_json
    from expansion.measurement import worker
    rng = np.random.default_rng(316)
    cells = pd.DataFrame({'cell_id': [f'c{i}' for i in range(16)], 'patient_id': ['a'] * 8 + ['b'] * 8,
                          'gsm': ['s1'] * 8 + ['s2'] * 8, 'celltype': 'other'})
    held = sorted(['a', 'b'], key=lambda pat: (content_hash(['outcome-blind-patient-partition-v1', 7, pat], 64), pat))[:1]
    partition = dict(method='id_hash_v1', cell_identity_column='cell_id', seed=7, n_held_patients=1,
        expected_held_patients=held, evaluation_cells_per_patient=4,
        expected_cell_order_identity=content_hash(cells.cell_id.tolist(), 64))
    x = sp.csr_matrix(rng.dirichlet(np.ones(8), size=16).astype('float32'))
    panel = np.array([f'g{i}' for i in range(8)])
    monkeypatch.setattr(stability, 'load_discovery_panel_counts', lambda _: (x, cells))
    monkeypatch.setattr(worker.np, 'load', lambda path, **kwargs: panel if str(path).endswith('panel.npy') else np.ones((3, 8)) / 8)
    def undefined(*a):
        raise ValueError('constant topic rows have undefined correlation')
    monkeypatch.setattr(stability, 'match_topics', undefined)
    monkeypatch.setattr(worker, 'sha256_file', lambda _: 'fixture')
    monkeypatch.setattr(stability, 'save_inference_checkpoint', lambda folder, *a, **kw: {'reload_verified': True})
    protocol = dict(stability=dict(K=[3], seeds=[0], step_budgets=[3], patient_partition=partition,
        retain_undefined_reference_match=True, batch_size=8, lr=.001, kl_weight=.01, diagnostic_every=1), semantics={})
    result = worker.run_stability_job({'params': {'K': 3, 'steps': 3}, 'seed': 0}, protocol=protocol, root='.', out_dir=tmp_path)
    assert result['inference_checkpoint']['reload_verified']
    assert result['match_to_reference']['status'] == 'undefined_no_rescue'
    assert result['primary_readout_status'] == 'reference_topic_0_unmatched_no_primary_readout'
    assert len(read_json(tmp_path / 'fit_trace.json')['training_trace']) == 3
