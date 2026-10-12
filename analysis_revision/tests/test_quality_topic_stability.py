"""Cheap synthetic fail-closed checks; no human inputs, CUDA fits or downloads."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from expansion.runner.jobs import Job, plan_measurement_jobs
from scripts import run_quality_topic_stability as sidecar


def protocol():
    return json.loads(sidecar.CONFIG.read_text())


def synthetic_cells():
    return pd.DataFrame({"gsm": ["s0", "s0", "s1", "s2", "s2", "s2"],
        "patient_id": ["p0", "p0", "p0", "p1", "p1", "p1"],
        "celltype": ["fibroblast"] * 6, "group": ["DFU-H"] * 3 + ["H"] * 3,
        "reconstructed_raw_cell_id": [f"synthetic-cell-{i}" for i in range(6)]})


def job(seed=0, k=10):
    return Job("stability", "GSE165816", "topic_simplex", seed, "panel", {}, {"K": k, "steps": 40000}, "test")


def synthetic_result(seed=0, assigned=0):
    return {"K": 10, "seed": seed, "primary_match_rule": "hungarian_one_to_one",
        "primary_match_status": "unmatched" if assigned is None else "assigned", "fibroblast_topic": assigned,
        "fibroblast_topic_correlation": None if assigned is None else -.2,
        "rowwise_best_sensitivity": {"used_for_primary": False, "topic": 1}, "cut": None if assigned is None else .5,
        "mean_fraction_spearman": None,
        "match_to_reference": {"assignment": {} if assigned is None else {"0": assigned},
            "assigned_correlation": {} if assigned is None else {"0": -.2},
            "best_match": {"0": 1}, "best_correlation": {"0": .9},
            "unmatched_reference": list(range(15)) if assigned is None else list(range(1, 15)),
            "unmatched_new": list(range(10)) if assigned is None else list(range(1, 10))},
        "semantics": {"available": False, "reason": "synthetic unavailable", "exclusions": []},
        "patient_readouts": [{"unit": "p0", "mean": .5, "fraction": .5, "n_cells": 3, "n_specimens": 2},
                             {"unit": "p1", "mean": .4, "fraction": 1/3, "n_cells": 3, "n_specimens": 1}],
        "inference_checkpoint": {"reload_verified": True, "optimizer_resume_supported": False,
            "reload_max_abs_error": 0., "training_config": {"steps": 40000, "seed": seed, "device": "cuda"},
            "model_config": {"num_topics": 10}, "losses": [1.] * 41}}


def test_frozen_protocol_plans_exactly_nine_fits():
    p = protocol()
    sidecar.validate_protocol(p)
    jobs = plan_measurement_jobs(p, {"GSE165816": {"source": "discovery"}}, code_digest="synthetic")
    sidecar.validate_grid(jobs)
    assert len(jobs) == 9
    assert all(j.params["steps"] == 40000 for j in jobs)


@pytest.mark.parametrize("change", ["steps", "K", "seed_bool", "retention", "partition", "semantics", "design",
                                   "workers", "budget", "reserve", "output", "prognostic"])
def test_protocol_changes_fail_closed(change):
    p = protocol()
    if change == "steps": p["stability"]["steps"] = 4000
    elif change == "K": p["stability"]["K"] = [15]
    elif change == "seed_bool": p["stability"]["seeds"] = [False, 1, 2]
    elif change == "retention": p["stability"]["retain_undefined_reference_match"] = False
    elif change == "partition": p["stability"]["patient_partition"] = {}
    elif change == "semantics": p["semantics"]["families"] = ["topic"]
    elif change == "design": p["design"]["enabled"] = True
    elif change == "workers": p["resources"]["gpu_workers"] = 2
    elif change == "budget": p["resources"]["max_incremental_artifact_bytes"] += 1
    elif change == "reserve": p["resources"]["min_remaining_disk_bytes"] = 1
    elif change == "output": p["resources"]["output_root"] = "output/old"
    else: p["analysis"]["prognostic_tests"] = True
    with pytest.raises(ValueError): sidecar.validate_protocol(p)


def test_grid_missing_or_duplicated_jobs_cannot_aggregate():
    jobs = [job(seed=s, k=k) for k in (10, 15, 20) for s in (0, 1, 2)]
    for altered in (jobs[:-1], jobs[:-1] + [jobs[0]]):
        with pytest.raises(ValueError, match="grid"): sidecar.validate_grid(altered)


@pytest.mark.parametrize("model,effort", [(None, "xhigh"), ("gpt-6.1-sol", None), ("gpt-6-sol", "xhigh"),
                                        ("gpt-6.1-sol", "medium")])
def test_requested_model_is_not_runtime_verification(model, effort):
    with pytest.raises(ValueError, match="mismatch"):
        sidecar.validate_runtime({"id": "child", "session_id": "parent"},
                                 {"model": model, "effort": effort, "turn_id": "turn"})


def test_runtime_records_actual_and_requested_separately():
    record = sidecar.validate_runtime({"id": "child", "session_id": "parent", "parent_thread_id": "parent"},
                    {"model": "gpt-6.1-sol", "effort": "xhigh", "turn_id": "turn"})
    assert record["runtime_reported"]["thread_id"] == "child"
    assert record["runtime_reported"]["session_id"] == "parent"
    assert record["requested"]["reasoning_effort"] == "xhigh"


def test_forked_runtime_ignores_replayed_parent_metadata(monkeypatch, tmp_path):
    folder = tmp_path / "sessions/2026/10/09"
    folder.mkdir(parents=True)
    path = folder / "rollout-2026-10-09T00-00-00-child.jsonl"
    records = [
        {"type": "session_meta", "payload": {"id": "child", "session_id": "parent", "parent_thread_id": "parent"}},
        {"type": "session_meta", "payload": {"id": "parent", "session_id": "parent"}},
        {"type": "turn_context", "payload": {"model": "gpt-6.1-sol", "effort": "xhigh", "turn_id": "child-turn"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_THREAD_ID", "child")
    record = sidecar.runtime_record()
    assert record["runtime_reported"]["thread_id"] == "child"
    assert record["runtime_reported"]["turn_id"] == "child-turn"


def test_numeric_trace_adapter_preserves_fit_objects_and_options():
    state, beta = object(), object()
    options = {"steps": 40000, "K": 10, "seed": 2}
    def trainer(*args, **kwargs):
        assert kwargs == options
        return {"state": state, "beta": beta, "training_config": kwargs}
    result = sidecar.trace_metadata_adapter(trainer)(None, **options)
    assert result["state"] is state and result["beta"] is beta
    assert result["training_trace"] == result["evaluation_trace"] == []
    assert result["training_config"] == options


def test_trace_adapter_keeps_existing_traces():
    expected = {"training_trace": [{"step": 1}], "evaluation_trace": [{"step": 0}]}
    assert sidecar.trace_metadata_adapter(lambda: expected)() is expected
    assert expected["training_trace"] == [{"step": 1}]


@pytest.mark.parametrize("bad", ["reload", "steps", "error", "loss", "device", "rescue"])
def test_checkpoint_completion_fail_closed(bad):
    r = synthetic_result()
    m = r["inference_checkpoint"]
    if bad == "reload": m["reload_verified"] = False
    elif bad == "steps": m["training_config"]["steps"] = 4000
    elif bad == "error": m["reload_max_abs_error"] = float("nan")
    elif bad == "loss": m["losses"] = [1.] * 40
    elif bad == "device": m["training_config"]["device"] = "cpu"
    else: r["rowwise_best_sensitivity"]["used_for_primary"] = True
    with pytest.raises(ValueError): sidecar.validate_checkpoint_result(r, job())


def test_negative_and_unmatched_matches_are_not_filtered_or_rescued():
    assigned = sidecar.match_rows(job(), synthetic_result())
    assert len([r for r in assigned if r["row_kind"] == "reference_topic"]) == 15
    assert assigned[0]["assigned_correlation"] == -.2
    missing = sidecar.match_rows(job(), synthetic_result(assigned=None))
    assert missing[0]["new_topic"] is None and missing[0]["rowwise_best_topic"] == 1
    assert not missing[0]["rowwise_used_for_primary"]
    assert len(missing) == 25


def test_patient_pooling_equal_specimen_not_cell_pooled():
    cells = synthetic_cells()
    # p0: two low cells in s0, one high cell in s1 -> specimen mean .5,
    # not pooled-cell mean 1/3. This deliberately exposes unequal specimen sizes.
    theta = np.array([[0., 1.], [0., 1.], [1., 0.], [.2, .8], [.2, .8], [.8, .2]])
    rows = sidecar.patient_rows(job(), synthetic_result(), cells, theta)
    first = next(r for r in rows if r["patient_id"] == "p0" and r["readout_scope"] == "fibroblast_only")
    assert first["mean_loading"] == .5
    assert first["mean_loading"] != pytest.approx(theta[:3, 0].mean())
    assert first["n_cells"] == 3 and first["n_specimens"] == 2
    assert {r["readout_scope"] for r in rows} == {"fibroblast_only", "all_cells"}


def test_missing_primary_retains_every_person_and_both_readout_scopes():
    rows = sidecar.patient_rows(job(), synthetic_result(assigned=None), synthetic_cells(), np.ones((6, 2)) / 2)
    assert len(rows) == 4
    assert {r["patient_id"] for r in rows} == {"p0", "p1"}
    assert all(r["mean_loading"] is None and r["fraction_above_fibroblast_descriptive_cut"] is None for r in rows)


def test_seed_ranges_keep_missing_seeds_and_are_not_confidence_intervals():
    rows = []
    for seed, value in enumerate([0., None, .4]):
        rows.append({"K": 10, "patient_id": "p0", "readout_scope": "fibroblast_only", "seed": seed,
                     "mean_loading": value, "fraction_above_fibroblast_descriptive_cut": value})
    ranges = sidecar.seed_ranges(rows)
    assert ranges[0]["available_seeds"] == 2 and ranges[0]["missing_seeds"] == 1
    assert ranges[0]["min_over_available_seeds"] == 0.
    assert ranges[0]["range_over_available_seeds"] == .4
    assert not ranges[0]["complete_three_seed_readout"]
    with pytest.raises(ValueError, match="seed block"): sidecar.seed_ranges(rows[:2])


def test_resource_checks_fail_before_dispatch(monkeypatch, tmp_path):
    p = protocol()["resources"]
    monkeypatch.setattr(sidecar.shutil, "disk_usage", lambda _: SimpleNamespace(free=8 * 2**30))
    with pytest.raises(ValueError, match="reserve"): sidecar.resource_check(tmp_path, p, 1)
    monkeypatch.setattr(sidecar.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 2**30))
    with pytest.raises(ValueError, match="budget"):
        sidecar.resource_check(tmp_path, p, 512 * 2**20 + 1)
    assert sidecar.resource_check(tmp_path, p)["artifact_bytes"] == 0


def test_identity_conflicts_fail_closed():
    cells = synthetic_cells()
    cells.loc[0, "patient_id"] = "other"
    with pytest.raises(ValueError, match="multiple people"): sidecar.cohort_inventory(cells)


def test_unavailable_semantics_retains_all_comparisons_people_and_specimens():
    result = synthetic_result(assigned=None)
    tables = sidecar.tables_for_fit(job(), result, synthetic_cells(), np.ones((6, 2)) / 2)
    assert len(tables["semantics"]) == 5
    assert len(tables["patient_semantics"]) == 10
    assert len(tables["semantics_specimens"]) == 3
    assert all(r["difference_nats_per_cell"] is None for r in tables["patient_semantics"])
    assert all(r["cut_training_only_logit_scale"] is None for r in tables["semantics_specimens"])


def test_preparation_contract_never_self_approves_or_fits():
    import inspect
    source = inspect.getsource(sidecar.prepare)
    assert "execute_run(" not in source and "train_topic_model(" not in source
    assert "approval_request.json" in source
    assert "draft[\"approved\"] = True" not in source
