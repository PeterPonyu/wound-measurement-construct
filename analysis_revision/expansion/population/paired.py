"""Opt-in matched repeated evaluation and pickle-free CPU prediction replay.

One fit produces multiple *computational* draws, not more biological units.
Moment/context-based predictors see the full source before particle selection.
Neither targets nor predictions select the MMD kernel or representation.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..common.io import atomic_save_npz, canonical_json, content_hash, sanitize, sha256_file
from ..common.rng import derive_seed
from .infosets import resolve_variant
from .methods import create
from .metrics import PAIRED_EVALUATION_CONTRACT, evaluate, evaluation_contract_options, freeze_train_mmd_bandwidth
from .represent import (fit_pca, fit_standardizer, representation_arrays,
                        restore_representation, task_coordinates)
from .tasks import MethodInputs, TaskSpec, build_task_data, make_clock, method_inputs, representation_mask


def row_ids(digest, rows):
    return np.asarray([f"{digest}:{int(r)}" for r in rows], dtype=str)


def group_rows(obs, individual, time_value):
    return np.flatnonzero((obs.individual.astype(str).to_numpy() == str(individual))
                         & (obs.time_value.to_numpy(dtype=float) == float(time_value)))


def save_model(path, method, representation, *, rep_config, train_rows, binding, min_cells, clock_name):
    arrays = {f"rep_{k}": v for k, v in representation_arrays(representation).items()}
    groups = []
    for index, (key, cells) in enumerate(sorted(method.inputs.train.items())):
        tag = f"train{index}"
        arrays[tag] = np.asarray(cells, dtype=np.float64)
        arrays[f"{tag}_rows"] = np.asarray(train_rows[key], dtype=np.int64)
        groups.append({"tag": tag, "individual": key[0], "time": key[1]})
    meta = {"schema": "population-replay-v2", "binding": binding, "method": method.spec.name,
            "params": method.params, "variant": method.variant, "groups": groups,
            "clock": [[k, v] for k, v in sorted(method.inputs.clock.items())],
            "kind": method.inputs.kind, "held_time": method.inputs.held_time,
            "held_individual": method.inputs.held_individual,
            "min_cells_per_group": min_cells,
            "clock_name": clock_name,
            "representation": {"kind": representation.kind, "params": representation.params,
                               "config": rep_config, "array_names": list(representation_arrays(representation))},
            "field_keys": [], "purpose": "prediction replay only; optimizer/resume state is not saved"}
    if method.spec.family == "flow":
        for index, (key, tensor) in enumerate(method.field.state_dict().items()):
            arrays[f"field{index}"] = tensor.detach().cpu().numpy().copy()
            meta["field_keys"].append(key)
        meta["interval"] = method.interval
    atomic_save_npz(path, **arrays, metadata_json=np.array(canonical_json(meta)))


def load_model(path, *, expected_sha256, expected_binding):
    """Restore only registered implementations; no deserialization of executable objects."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("model replay SHA256 mismatch")
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata_json"]))
        if meta.get("schema") != "population-replay-v2" or meta["binding"] != expected_binding:
            raise ValueError("model replay binding differs from the approved job")
        repmeta = meta["representation"]
        rep = restore_representation(repmeta["kind"], repmeta["params"],
                                     {k: saved[f"rep_{k}"] for k in repmeta["array_names"]})
        train, rows = {}, {}
        for item in meta["groups"]:
            key, tag = (item["individual"], float(item["time"])), item["tag"]
            cells, indices = saved[tag], saved[f"{tag}_rows"]
            if (key in train or cells.ndim != 2 or cells.shape[1] != len(rep.coord_mean)
                    or not len(cells) or not np.isfinite(cells).all()
                    or indices.shape != (len(cells),) or indices.dtype.kind not in "iu"
                    or np.any(indices < 0) or len(set(indices.tolist())) != len(indices)):
                raise ValueError("invalid or duplicate saved training group")
            train[key], rows[key] = cells.copy(), indices.copy()
        if not train:
            raise ValueError("empty saved training view")
        method = create(meta["method"], **meta["params"])
        method.inputs = MethodInputs(meta["kind"], train, {float(k): float(v) for k, v in meta["clock"]},
                                     meta["held_time"], meta["held_individual"])
        method.variant = resolve_variant(meta["kind"], method.spec.variants)
        if method.variant != meta["variant"]:
            raise ValueError("saved method variant violates the information set")
        if method.spec.family == "flow":
            import torch
            from .methods.flows import DEFAULTS, VelocityField

            p = {**DEFAULTS, **method.params}
            dim = len(rep.coord_mean)
            method.device = torch.device("cpu")
            # Constructing the field must not perturb global training RNG state.
            with torch.random.fork_rng(devices=[]):
                method.field = VelocityField(dim, hidden=p["hidden"], depth=p["depth"], time_dim=p["time_dim"],
                                             context_dim=2 * dim if method.conditioned else 0)
            state = {key: torch.from_numpy(saved[f"field{i}"].copy()) for i, key in enumerate(meta["field_keys"])}
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("nonfinite saved field")
            for name, expected in method.field.named_buffers():
                if name not in state or not torch.equal(state[name], expected):
                    raise ValueError("saved deterministic field buffer differs from the approved architecture")
            method.field.load_state_dict(state, strict=True)
            method.field.eval()
            method.interval = float(meta["interval"])
            if not np.isfinite(method.interval) or method.interval <= 0:
                raise ValueError("invalid saved integration interval")
        elif meta["field_keys"]:
            raise ValueError("a baseline cannot carry an unrecognized neural field")
    return method, rep, rows, meta


def _expected_flow_interval(clock):
    spacing = np.diff(sorted(float(value) for value in clock.values()))
    return float(spacing.min()) if spacing.size else 1.0


def _check_representation_state(rep, adata, fit_rows, rep_config, *, seed):
    """Verify the saved coordinate state was deterministically fit on permitted rows.

    This is a representation audit, not a model fit or optimizer resume.  The
    standardizer is checked by exact train-row moments; PCA is reconstructed
    from the approved count rows with its frozen seed/configuration so a
    self-consistent but held-cell-derived state cannot pass replay.
    """
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    config = dict(rep_config)
    if rep.kind != config.get("kind"):
        raise ValueError("saved representation kind differs from the approved configuration")
    if rep.kind == "obsm_standardized":
        latent = np.asarray(adata.obsm[config["key"]], dtype=np.float64)[fit_rows]
        expected = fit_standardizer(latent)
    elif rep.kind == "pca_train":
        expected = fit_pca(sp.csr_matrix(adata.X)[fit_rows],
                           n_hvg=int(config.get("n_hvg", 2000)),
                           n_components=int(config.get("n_components", 30)),
                           max_fit_cells=int(config.get("max_fit_cells", 50_000)),
                           seed=int(seed))
    else:
        raise ValueError("unknown saved representation kind")
    if rep.params != expected.params:
        raise ValueError("saved representation parameters were not derived from the approved training rows")
    for name, expected_array in representation_arrays(expected).items():
        actual = getattr(rep, name)
        if actual is None or actual.shape != expected_array.shape:
            raise ValueError("saved representation state shape differs from the approved training fit")
        np.testing.assert_allclose(actual, expected_array, rtol=1e-10, atol=1e-10,
                                   err_msg="saved representation state differs from the approved training fit")


def _identity(arrays, prefix, rows, obs, digest):
    arrays[f"{prefix}_rows"] = np.asarray(rows, dtype=np.int64)
    arrays[f"{prefix}_ids"] = row_ids(digest, rows)
    arrays[f"{prefix}_obs_names"] = obs.index.to_numpy(dtype=str)[rows]
    arrays[f"{prefix}_metadata_json"] = np.array(obs.iloc[rows].to_json(orient="records"))


def run_paired_job(job, adata, *, protocol, device, out_dir):
    from .worker import _method_params, predict_evaluation_draw
    from ..runner.jobs import method_base

    options = evaluation_contract_options(protocol["evaluation"])
    if options["name"] != PAIRED_EVALUATION_CONTRACT:
        raise ValueError("paired runner needs its explicit v2 contract")
    task = TaskSpec.from_dict(job["task"])
    if task.training_individuals is not None:
        raise ValueError("paired-v2 count curves require a separately frozen cross-k scoring space")
    missing = set(options["identity_columns"]) - set(adata.obs.columns)
    if missing:
        raise ValueError(f"required original identity columns unavailable: {sorted(missing)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages, begun = {}, time.perf_counter()
    rep_config = dict(protocol["representations"][job["representation"]])
    mask = representation_mask(adata.obs, task)
    coords, rep_info, rep = task_coordinates(adata, mask, rep_config,
                                            seed=int(protocol.get("representation_seed", 0)), return_state=True)
    stages["representation"] = time.perf_counter() - begun
    clock_name = job["params"].get("clock", protocol["clock"]["primary"])
    data = build_task_data(coords, adata.obs, task, make_clock(adata.obs.time_value.unique(), clock_name),
                           min_cells=int(protocol.get("min_cells_per_group", 1)))
    train_rows = {key: group_rows(adata.obs, *key) for key in sorted(data.train)}
    training = np.concatenate(list(train_rows.values()))
    digest = job["params"]["dataset_sha256"]
    frozen = freeze_train_mmd_bandwidth(coords[training], row_ids(digest, training),
                                       seed=options["bandwidth_seed"], cap=options["bandwidth_cap"])
    # Save the full reference, not just a capped pilot, so the selected indices can be re-derived.
    arrays = {"mmd_training_coordinates": np.asarray(coords[training], dtype=np.float64),
              "mmd_training_rows": training, "mmd_training_ids": row_ids(digest, training),
              "representation_fit_rows": np.flatnonzero(mask)}
    method = create(method_base(protocol, job["method"]), **_method_params(protocol, job["method"]))
    if job["params"].get("evaluation_stochastic") != bool(getattr(method, "evaluation_stochastic", False)):
        raise ValueError("job evaluation-randomness marker differs from method implementation")

    def sync():
        if str(device).startswith("cuda"):
            import torch
            torch.cuda.synchronize(device)

    sync()
    tick = time.perf_counter()
    fit = method.fit(method_inputs(data), seed=int(job["seed"]), device=device)
    sync()
    fit_seconds = time.perf_counter() - tick
    stages["fit_including_coupling"] = fit_seconds
    binding = {"job_id": job["job_id"], "protocol_hash": job["protocol_hash"],
               "dataset_sha256": digest, "code_digest": job["params"].get("code_digest", "")}
    tick = time.perf_counter()
    save_model(out_dir / "model_replay.npz", method, rep, rep_config=rep_config,
               train_rows=train_rows, binding=binding, min_cells=int(protocol.get("min_cells_per_group", 1)),
               clock_name=clock_name)
    stages["model_io"] = time.perf_counter() - tick
    metrics = [protocol["evaluation"].get("primary_metric", "energy"), *protocol["evaluation"].get("secondary_metrics", [])]
    caps, projections = protocol["evaluation"]["caps"], int(protocol["evaluation"].get("sliced_projections", 128))
    records, items, predictions, prediction_cache = [], [], {}, {}

    def recorded_prediction(source, call, seed):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = predict_evaluation_draw(method, source, **call, draw_seed=seed)
        diagnostics = {"warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                       "solver": getattr(method, "last_prediction_diagnostics", None)}
        return value, diagnostics

    for pi, pair in enumerate(task.pairs):
        own_rows = group_rows(adata.obs, pair.individual, pair.source_time)
        target_rows = group_rows(adata.obs, pair.individual, pair.target_time)
        count = min(options["sample_size"], len(own_rows), len(target_rows))
        if count < 2:
            raise ValueError("paired evaluation needs at least two source and target cells")
        call = dict(individual=pair.individual, source_time=pair.source_time,
                    target_time=pair.target_time, anchor_time=pair.anchor_time)
        for draw_index in range(options["draws"]):
            draw_seed = derive_seed("paired-draw-v2", options["draw_seed"], task.dataset_id, task.kind, task.family,
                                    pair.individual, pair.source_time, pair.target_time, draw_index)
            score_seed = derive_seed("paired-score-v2", draw_seed)
            selected = np.sort(np.random.default_rng([draw_seed, 1]).choice(len(own_rows), count, replace=False))
            target_selection = np.sort(np.random.default_rng([draw_seed, 2]).choice(len(target_rows), count, replace=False))
            tr = target_rows[target_selection]
            variants = [("own", own_rows)]
            if task.family == "source_swap":
                sr = group_rows(adata.obs, task.substitute, pair.source_time)
                chosen = np.random.default_rng([draw_seed, 4]).choice(len(sr), len(own_rows),
                                                                      replace=len(sr) < len(own_rows))
                variants.append((f"substitute:{task.substitute}", sr[chosen]))
            for vi, (variant, full_rows) in enumerate(variants):
                full_source = coords[full_rows]
                key = f"pair{pi}_draw{draw_index}_v{vi}"
                context_key = f"pair{pi}_context" if variant == "own" else f"{key}_context"
                if context_key not in arrays:
                    arrays[context_key] = np.asarray(full_source, dtype=np.float64)
                    _identity(arrays, context_key, full_rows, adata.obs, digest)
                tick = time.perf_counter()
                cache_key = (pi, variant)
                stochastic = bool(getattr(method, "evaluation_stochastic", False))
                if not method.spec.uses_source:
                    prediction, diagnostics = recorded_prediction(full_source[selected], call, draw_seed)
                    origins = np.concatenate([train_rows[g][r] for g, r in method.last_draw_rows])
                else:
                    if cache_key not in prediction_cache or stochastic or variant != "own":
                        predicted, diagnostics = recorded_prediction(full_source, call, draw_seed)
                        if variant == "own" and not stochastic:
                            prediction_cache[cache_key] = (predicted, diagnostics)
                    else:
                        predicted, diagnostics = prediction_cache[cache_key]
                    if predicted.shape != full_source.shape:
                        raise ValueError("source-using predictor changed particle count or dimensions")
                    prediction, origins = predicted[selected], full_rows[selected]
                sync()
                stages["predict"] = stages.get("predict", 0.) + time.perf_counter() - tick
                if (prediction.shape != (count, coords.shape[1]) or not np.isfinite(prediction).all()
                        or len(origins) != count):
                    raise ValueError("nonfinite or unmatched prediction")
                tick = time.perf_counter()
                scores = evaluate(prediction, coords[tr], seed=score_seed, caps=caps, n_projections=projections,
                                  mmd_bandwidth=frozen["bandwidth"], return_indices=True, metric_names=metrics)
                stages["score"] = stages.get("score", 0.) + time.perf_counter() - tick
                record = {"pair_index": pi, "individual": pair.individual, "source_time": pair.source_time,
                          "target_time": pair.target_time, "anchor_time": pair.anchor_time, "source_variant": variant,
                          "draw_index": draw_index, "fit_seed": int(job["seed"]), "draw_seed": draw_seed,
                          "score_seed": score_seed, "evaluation_contract": options["name"],
                          "mmd_reference_hash": frozen["reference_hash"], "matched_particles": count,
                          "evaluation_stochastic": stochastic,
                          "prediction_diagnostics": diagnostics,
                          "target_selection_hash": content_hash(row_ids(digest, tr), length=64),
                          "source_selection_hash": content_hash(row_ids(digest, full_rows[selected]), length=64),
                          "prediction_identity_kind": ("initialized_source_row_not_biological_lineage" if method.spec.uses_source
                                                       else "sampled_training_target_row")}
                arrays[f"{key}_prediction"] = np.asarray(prediction, dtype=np.float64)
                arrays[f"{key}_target"] = np.asarray(coords[tr], dtype=np.float64)
                arrays[f"{key}_selected"] = selected
                _identity(arrays, f"{key}_source", full_rows[selected], adata.obs, digest)
                _identity(arrays, f"{key}_target", tr, adata.obs, digest)
                _identity(arrays, f"{key}_prediction", origins, adata.obs, digest)
                records.append({**record, **sanitize(scores)})
                items.append({"key": key, "context_key": context_key, "record": record, "scores": sanitize(scores)})
                predictions[key] = np.asarray(prediction, dtype=np.float32)
    meta = {"schema": PAIRED_EVALUATION_CONTRACT, "contract": options, "binding": binding, "task": task.to_dict(),
            "caps": caps, "n_projections": projections, "metric_names": metrics, "mmd_reference": frozen,
            "model_sha256": sha256_file(out_dir / "model_replay.npz"), "items": items,
            "scope": "paired computational draws; fold-local train geometry, not cross-fold distances or new donors"}
    tick = time.perf_counter()
    atomic_save_npz(out_dir / "evaluation_contract.npz", **arrays, metadata_json=np.array(canonical_json(meta)))
    atomic_save_npz(out_dir / "predictions.npz", **predictions)
    stages["evaluation_io"] = time.perf_counter() - tick
    return {"records": records, "fit": fit, "fit_seconds": fit_seconds, "variant": method.variant,
            "representation": rep_info, "device": str(device), "clock": clock_name,
            "stage_seconds": stages, "worker_seconds": time.perf_counter() - begun,
            "timing_scope": "synchronized actual cost, not equal steps or equal compute",
            "evaluation_contract": options["name"], "evaluation_draws": options["draws"], "mmd_reference": frozen,
            "model_artifact_sha256": meta["model_sha256"],
            "evaluation_artifact_sha256": sha256_file(out_dir / "evaluation_contract.npz"),
            "evaluation_scope": meta["scope"]}


def replay_paired_artifact(directory, *, expected_evaluation_sha256, expected_model_sha256,
                           expected_binding, expected_job, expected_protocol, adata=None, atol=1e-5, rtol=1e-5):
    """CPU prediction replay without predictor fitting or optimizer resumption.

    Input-backed PCA replay separately reconstructs the approved representation
    for its numerical provenance audit; standardizers use train-row moments.
    """
    from .worker import predict_evaluation_draw
    from .worker import _method_params
    from ..runner.jobs import Job, method_base, protocol_hash

    directory = Path(directory)
    path = directory / "evaluation_contract.npz"
    if sha256_file(path) != expected_evaluation_sha256:
        raise ValueError("paired evaluation SHA256 mismatch")
    approved_job = Job.from_dict(expected_job)
    binding = {"job_id": approved_job.job_id, "protocol_hash": approved_job.protocol_hash,
               "dataset_sha256": approved_job.params["dataset_sha256"],
               "code_digest": approved_job.params.get("code_digest", "")}
    if expected_binding != binding or approved_job.protocol_hash != protocol_hash(expected_protocol):
        raise ValueError("replay plan/protocol binding mismatch")
    method, rep, train_rows, modelmeta = load_model(directory / "model_replay.npz",
                                                    expected_sha256=expected_model_sha256,
                                                    expected_binding=expected_binding)
    task = TaskSpec.from_dict(approved_job.task)
    clock_name = approved_job.params.get("clock", expected_protocol["clock"]["primary"])
    config = expected_protocol["representations"][approved_job.representation]
    if rep.kind != config.get("kind"):
        raise ValueError("saved representation kind differs from the approved configuration")
    if (modelmeta["method"] != method_base(expected_protocol, approved_job.method)
            or modelmeta["params"] != _method_params(expected_protocol, approved_job.method)
            or modelmeta["representation"]["config"] != config
            or modelmeta["min_cells_per_group"] != int(expected_protocol.get("min_cells_per_group", 1))
            or modelmeta["kind"] != task.kind or modelmeta["held_time"] != task.held_time
            or modelmeta["held_individual"] != task.held_individual or modelmeta["clock_name"] != clock_name):
        raise ValueError("saved model semantic configuration differs from the approved plan")
    expected_clock = make_clock(adata.obs.time_value.unique() if adata is not None else method.inputs.clock.keys(), clock_name)
    if method.inputs.clock != expected_clock:
        raise ValueError("saved clock differs from the approved clock transformation")
    if method.spec.family == "flow" and not np.isclose(method.interval, _expected_flow_interval(expected_clock), rtol=0, atol=1e-12):
        raise ValueError("saved flow interval differs from the approved clock spacing")
    records, maximum = [], 0.
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata_json"]))
        if (meta.get("schema") != PAIRED_EVALUATION_CONTRACT or meta["binding"] != expected_binding
                or meta["model_sha256"] != expected_model_sha256):
            raise ValueError("paired replay belongs to another job/model")
        approved_evaluation = expected_protocol["evaluation"]
        approved_options = evaluation_contract_options(approved_evaluation)
        approved_metrics = [approved_evaluation.get("primary_metric", "energy"), *approved_evaluation.get("secondary_metrics", [])]
        if (meta["task"] != approved_job.task or meta["contract"] != approved_options
                or meta["caps"] != approved_evaluation["caps"] or meta["metric_names"] != approved_metrics
                or meta["n_projections"] != int(approved_evaluation.get("sliced_projections", 128))):
            raise ValueError("saved evaluation semantic configuration differs from the approved plan")
        options = evaluation_contract_options({"contract": meta["contract"], "caps": meta["caps"],
                                               "primary_metric": meta["metric_names"][0],
                                               "secondary_metrics": meta["metric_names"][1:]})
        for individual, time_value in method.inputs.train:
            if ((task.kind == "T-MT" and np.isclose(time_value, task.held_time))
                    or (task.kind == "T-LIO" and individual in {task.held_individual, task.substitute})):
                raise ValueError("saved model training view leaks a held identity/time")
        digest = expected_binding["dataset_sha256"]
        if not np.array_equal(saved["mmd_training_ids"], row_ids(digest, saved["mmd_training_rows"])):
            raise ValueError("MMD reference identity mismatch")
        frozen = freeze_train_mmd_bandwidth(saved["mmd_training_coordinates"], saved["mmd_training_ids"],
                                           seed=options["bandwidth_seed"], cap=options["bandwidth_cap"])
        if frozen != meta["mmd_reference"]:
            raise ValueError("frozen MMD reference mismatch")
        transformed = None
        if adata is not None:
            config = modelmeta["representation"]["config"]
            expected_fit_rows = np.flatnonzero(representation_mask(adata.obs, task))
            _check_representation_state(rep, adata, expected_fit_rows, config,
                                        seed=int(expected_protocol.get("representation_seed", 0)))
            transformed = (rep.transform_counts(adata.X) if rep.kind == "pca_train"
                           else rep.transform_latent(adata.obsm[config["key"]]))
            if not np.array_equal(saved["representation_fit_rows"], expected_fit_rows):
                raise ValueError("saved representation fit identities violate exclusion")
            data = build_task_data(transformed, adata.obs, task, method.inputs.clock,
                                   min_cells=modelmeta["min_cells_per_group"])
            if set(data.train) != set(method.inputs.train):
                raise ValueError("saved model differs from the permitted training groups")
            for key, rows in train_rows.items():
                np.testing.assert_array_equal(rows, group_rows(adata.obs, *key))
                np.testing.assert_allclose(transformed[rows], method.inputs.train[key], rtol=rtol, atol=atol)
            np.testing.assert_allclose(transformed[saved["mmd_training_rows"]], saved["mmd_training_coordinates"], rtol=rtol, atol=atol)
            expected_training_rows = np.concatenate([
                group_rows(adata.obs, individual, time_value)
                for individual, time_value in sorted(data.train)
            ]) if data.train else np.empty(0, dtype=np.int64)
            if not np.array_equal(saved["mmd_training_rows"], expected_training_rows):
                raise ValueError("MMD reference rows are not the complete approved training view")
        pairs = {(i, variant, d) for i in range(len(task.pairs)) for d in range(options["draws"])
                 for variant in (["own", f"substitute:{task.substitute}"] if task.family == "source_swap" else ["own"])}
        actual = [(i["record"]["pair_index"], i["record"]["source_variant"], i["record"]["draw_index"]) for i in meta["items"]]
        if len(actual) != len(set(actual)) or set(actual) != pairs:
            raise ValueError("paired artifact is incomplete or contains duplicate draws")
        cache = {}
        for item in meta["items"]:
            key, context, record = item["key"], item["context_key"], item["record"]
            pair = task.pairs[record["pair_index"]]
            if any(record[name] != getattr(pair, name) for name in ("individual", "source_time", "target_time", "anchor_time")):
                raise ValueError("paired replay record differs from the planned target")
            draw_seed = derive_seed("paired-draw-v2", options["draw_seed"], task.dataset_id, task.kind, task.family,
                                    pair.individual, pair.source_time, pair.target_time, record["draw_index"])
            if record["draw_seed"] != draw_seed or record["score_seed"] != derive_seed("paired-score-v2", draw_seed):
                raise ValueError("paired replay random streams do not match the frozen protocol")
            if (record["fit_seed"] != approved_job.seed or record["evaluation_contract"] != options["name"]
                    or record["mmd_reference_hash"] != frozen["reference_hash"]
                    or record["evaluation_stochastic"] != bool(getattr(method, "evaluation_stochastic", False))):
                raise ValueError("saved per-draw configuration differs from the approved plan")
            for role, array in (("prediction", saved[f"{key}_prediction"]), ("target", saved[f"{key}_target"]),
                                ("source", saved[f"{key}_selected"]), ("context", saved[context])):
                prefix = context if role == "context" else f"{key}_{role}"
                rows = saved[f"{prefix}_rows"]
                if (len(rows) != len(array) or not np.array_equal(saved[f"{prefix}_ids"], row_ids(digest, rows))):
                    raise ValueError("paired particle identity mismatch")
            if transformed is not None:
                for prefix, coordinates in ((context, saved[context]), (f"{key}_target", saved[f"{key}_target"])):
                    rows = saved[f"{prefix}_rows"]
                    np.testing.assert_allclose(coordinates, transformed[rows], rtol=rtol, atol=atol)
                    np.testing.assert_array_equal(saved[f"{prefix}_obs_names"], adata.obs.index.to_numpy(dtype=str)[rows])
                    if str(saved[f"{prefix}_metadata_json"]) != adata.obs.iloc[rows].to_json(orient="records"):
                        raise ValueError("original particle metadata differs from hash-bound input")
            selected = saved[f"{key}_selected"]
            source = saved[context]
            if (selected.dtype.kind not in "iu" or np.any(selected < 0) or np.any(selected >= len(source))
                    or not np.array_equal(saved[f"{key}_source_rows"], saved[f"{context}_rows"][selected])):
                raise ValueError("selected source particles do not match their full-source context")
            for role in ("source", "target"):
                if content_hash(saved[f"{key}_{role}_ids"], length=64) != record[f"{role}_selection_hash"]:
                    raise ValueError("paired selection hash mismatch")
            if transformed is not None:
                own_rows = group_rows(adata.obs, pair.individual, pair.source_time)
                target_rows = group_rows(adata.obs, pair.individual, pair.target_time)
                count = min(options["sample_size"], len(own_rows), len(target_rows))
                expected_selected = np.sort(np.random.default_rng([draw_seed, 1]).choice(len(own_rows), count, replace=False))
                target_selection = np.sort(np.random.default_rng([draw_seed, 2]).choice(len(target_rows), count, replace=False))
                expected_source = own_rows
                if record["source_variant"] != "own":
                    substitute = group_rows(adata.obs, task.substitute, pair.source_time)
                    drawn = np.random.default_rng([draw_seed, 4]).choice(len(substitute), len(own_rows),
                                                                        replace=len(substitute) < len(own_rows))
                    expected_source = substitute[drawn]
                np.testing.assert_array_equal(saved[f"{context}_rows"], expected_source)
                np.testing.assert_array_equal(selected, expected_selected)
                np.testing.assert_array_equal(saved[f"{key}_target_rows"], target_rows[target_selection])
                if record["matched_particles"] != count:
                    raise ValueError("paired particle count differs from the input")
            call = {k: record[k] for k in ("individual", "source_time", "target_time", "anchor_time")}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                if not method.spec.uses_source:
                    prediction = predict_evaluation_draw(method, source[selected], **call, draw_seed=record["draw_seed"])
                    origins = np.concatenate([train_rows[g][r] for g, r in method.last_draw_rows])
                else:
                    if context not in cache or bool(getattr(method, "evaluation_stochastic", False)):
                        cache[context] = predict_evaluation_draw(method, source, **call, draw_seed=record["draw_seed"])
                    prediction, origins = cache[context][selected], saved[f"{context}_rows"][selected]
            np.testing.assert_array_equal(origins, saved[f"{key}_prediction_rows"])
            np.testing.assert_allclose(prediction, saved[f"{key}_prediction"], rtol=rtol, atol=atol)
            maximum = max(maximum, float(np.max(np.abs(prediction - saved[f"{key}_prediction"]))))
            scores = sanitize(evaluate(saved[f"{key}_prediction"], saved[f"{key}_target"], seed=record["score_seed"],
                                      caps=meta["caps"], n_projections=meta["n_projections"],
                                      mmd_bandwidth=frozen["bandwidth"], return_indices=True, metric_names=meta["metric_names"]))
            if scores != item["scores"]:
                raise ValueError("saved paired score does not replay exactly")
            records.append({**record, **scores})
    return {"records": records, "prediction_max_abs_difference": maximum, "field_reintegrated": method.spec.family == "flow",
            "representation_restored": True, "input_revalidated": adata is not None, "device": "cpu",
            "representation_derivation_rechecked": adata is not None,
            "pca_state_rederived_for_audit": adata is not None and rep.kind == "pca_train",
            "atol": atol, "rtol": rtol, "no_fit_or_optimizer": rep.kind != "pca_train",
            "no_predictor_fit_or_optimizer": True,
            "replay_scope": "prediction/state consistency, not a causal certificate of neural training provenance"}


def summarize_paired_records(frame, protocol):
    """Collapse draws first, then seeds; preserve held-person/task blocks and adverse differences.

    Standard deviations are computational dispersion, never donor confidence intervals.
    No method ranking, p values, discovery claims or seed-based biological sample sizes.
    """
    from ..runner.queue import IntegrityError

    pairing = ["dataset_id", "task_hash", "representation", "clock", "pair_index", "source_variant", "draw_index"]
    if frame.duplicated(["job_id", "pair_index", "source_variant", "draw_index"]).any():
        raise IntegrityError("duplicate paired evaluation draw")
    for name in ("target_selection_hash", "source_selection_hash", "mmd_reference_hash", "matched_particles"):
        if name not in frame or (frame.groupby(pairing, dropna=False)[name].nunique(dropna=False) != 1).any():
            raise IntegrityError(f"paired methods do not share {name}")
    metrics = [protocol["evaluation"]["primary_metric"], *protocol["evaluation"].get("secondary_metrics", [])]
    fits = []
    identity = ["job_id", "dataset_id", "task_hash", "representation", "clock", "kind", "family",
                "method", "seed", "individual", "source_time", "target_time", "source_variant", "pair_index"]
    for keys, block in frame.groupby(identity, dropna=False, sort=True):
        if len(block) != protocol["evaluation"]["contract"]["draws"]:
            raise IntegrityError("incomplete paired computational draws")
        row = dict(zip(identity, keys))
        row.update(draws=len(block), fit_seconds=float(block.fit_seconds.iloc[0]), metrics={})
        for name in metrics:
            values = block[name].to_numpy(dtype=float)
            row["metrics"][name] = {"mean": float(values.mean()), "computational_sd": float(values.std()),
                                    "min": float(values.min()), "max": float(values.max())}
        fits.append(row)
    targets = []
    keys = [k for k in identity if k not in {"job_id", "seed"}]
    for values, block in frame.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row.update(fit_seeds=sorted(block.seed.unique().tolist()), independent_people=1, metrics={})
        for name in metrics:
            seed_means = block.groupby("seed")[name].mean().to_numpy()
            row["metrics"][name] = {"mean": float(seed_means.mean()),
                                    "training_seed_sd_of_draw_means": float(seed_means.std())}
        targets.append(row)
    job_costs = frame[["job_id", "method", "seed", "fit_seconds"]].drop_duplicates("job_id").to_dict(orient="records")
    return sanitize({"jobs": int(frame.job_id.nunique()), "evaluation_rows": len(frame), "job_costs": job_costs,
                     "fit_pair_summaries": fits, "target_summaries": targets,
                     "independent_people_in_dataset": frame.groupby("dataset_id").individual.nunique().to_dict(),
                     "scope": "draws averaged within fit; seeds averaged within target; targets are dependent donor/time blocks",
                     "inference_enabled": False, "method_ranking": False,
                     "draws_with_predictor_warnings": sum(bool(r.get("warnings")) for r in frame.prediction_diagnostics),
                     "draws_with_unresolved_ot_marginals": sum(r.get("solver") is not None and
                         not r["solver"]["both_marginals_within_audit_tolerance"] for r in frame.prediction_diagnostics),
                     "cost_note": "per-fit synchronized fit_seconds counted once, not once per draw"})
