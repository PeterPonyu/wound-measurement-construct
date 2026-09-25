"""Behavioral isolation and scale-invariance tests for the pair audit."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("patient_pair", ROOT / "scripts/assess_patient_pair_discrimination.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5423)
    panel = np.array(MOD.BENCH.MODULE_GENES + [f"extra_{i}" for i in range(11)])
    X = rng.gamma(1.5, 1, (11, len(panel)))
    X /= X.sum(1, keepdims=True)
    topic = rng.uniform(.02, .5, 11)
    y = np.array([1] * 7 + [0] * 4)
    ids = np.array([f"P{i}" for i in range(11)])
    return X, topic, y, panel, ids, [0, 7]


@pytest.fixture(scope="module")
def baseline(data):
    with threadpool_limits(limits=1):
        return MOD.fit_fold(*data)


@pytest.mark.parametrize("change", ["labels", "expressions", "both"])
def test_held_values_cannot_change_fit_preprocessing_component_or_orientation(data, baseline, change):
    X, topic, y, panel, ids, held = data
    X, topic, y = X.copy(), topic.copy(), y.copy()
    if change in ("labels", "both"):
        y[held] = y[held[::-1]]
    if change in ("expressions", "both"):
        X[held] = X[held[::-1]] * 1e6 + 123.0
        topic[held] = [999.0, -999.0]
    with threadpool_limits(limits=1):
        changed = MOD.fit_fold(X, topic, y, panel, ids, held)
    assert changed["training_ids"] == baseline["training_ids"]
    assert changed["training_input_sha256"] == baseline["training_input_sha256"]
    for method in MOD.METHODS:
        assert MOD.state_hash(changed["states"][method]) == MOD.state_hash(baseline["states"][method])


def test_holdout_transform_is_row_independent_and_does_not_mutate_fit(data, baseline):
    X, topic, _, _, _, held = data
    before = {m: MOD.state_hash(s) for m, s in baseline["states"].items()}
    together, _ = MOD.score_fold(baseline, X[held], topic[held])
    for i in range(2):
        alone, _ = MOD.score_fold(baseline, X[[held[i]]], topic[[held[i]]])
        for method in MOD.METHODS:
            np.testing.assert_allclose(alone[method][0], together[method][i], rtol=1e-10, atol=1e-12)
    assert before == {m: MOD.state_hash(s) for m, s in baseline["states"].items()}


def test_nmf_component_selection_ignores_arbitrary_positive_factor_scales(data, baseline):
    X, topic, _, _, _, held = data
    _, coords = MOD.score_fold(baseline, X[held], topic[held])
    state = baseline["states"]["nmf"]
    train = state["arrays"]["train_scores"]
    y = np.array(baseline["training_outcomes"])
    scales = np.geomspace(1e-5, 1e5, train.shape[1])
    changed = MOD.component_rule(train * scales, y, standardized=True)
    assert changed["component"] == state["meta"]["component"]
    assert changed["orientation"] == state["meta"]["orientation"]
    np.testing.assert_allclose(changed["training_contrasts"], state["meta"]["training_contrasts"], atol=1e-12)
    j = changed["component"]
    a = state["meta"]["orientation"] * coords["nmf"][:, j]
    b = changed["orientation"] * coords["nmf"][:, j] * scales[j]
    assert MOD.pair_credit(*a) == MOD.pair_credit(*b)


def test_positive_nmf_basis_rescaling_keeps_projection_and_ranking(data, baseline):
    X, _, _, _, _, held = data
    basis = baseline["states"]["nmf"]["arrays"]["components"]
    scales = np.geomspace(.001, 1000, len(basis))
    a = MOD.nmf_scores(X[held], basis)
    b = MOD.nmf_scores(X[held], basis * scales[:, None])
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    meta = baseline["states"]["nmf"]["meta"]
    j, sign = meta["component"], meta["orientation"]
    assert MOD.pair_credit(*(sign * a[:, j])) == MOD.pair_credit(*(sign * b[:, j]))


@pytest.mark.parametrize("scale,offset", [(0.01, -1.0), (9.0, 23.0), (1e3, -100.0)])
def test_within_fold_pair_ranking_is_positive_affine_invariant(data, baseline, scale, offset):
    X, topic, _, _, _, held = data
    scores, _ = MOD.score_fold(baseline, X[held], topic[held])
    for method in MOD.METHODS:
        assert MOD.pair_credit(*scores[method]) == MOD.pair_credit(*(scale * scores[method] + offset))
    assert MOD.pair_credit(1.0, 1.0) == .5


def test_pair_has_nine_training_patients_and_both_targets_are_excluded(data, baseline):
    assert len(baseline["training_ids"]) == 9
    assert not set(baseline["training_ids"]) & set(baseline["held_ids"])
    assert baseline["training_outcomes"].count(1) == 6
    assert baseline["training_outcomes"].count(0) == 3
    with pytest.raises(ValueError):
        MOD.fit_fold(*data[:-1], [0, 0])


def test_constant_nmf_component_is_finite_and_not_spuriously_selected():
    y = np.array([0, 0, 0, 1, 1, 1])
    values = np.column_stack([np.ones(6), [0, 1, 2, 8, 9, 10]])
    selected = MOD.component_rule(values, y, standardized=True)
    assert selected["component"] == 1
    assert selected["training_contrasts"][0] == 0
