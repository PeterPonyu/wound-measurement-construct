"""Execute one population-benchmark job: representation, fit, predictions and scores."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from ..common.io import atomic_save_npz, canonical_json, content_hash, sanitize, sha256_file
from ..common.rng import derive_seed
from ..data.standard import load_standard
from .methods import REGISTRY, create
from .metrics import PAIRED_EVALUATION_CONTRACT, energy_distance, evaluate, evaluation_contract_options, freeze_train_mmd_bandwidth
from .represent import task_coordinates
from ..runner.jobs import method_base, protocol_hash
from .tasks import TaskSpec, build_task_data, make_clock, method_inputs, representation_mask


def _mask_key(mask) -> str:
    packed = np.packbits(np.asarray(mask, dtype=bool))
    return content_hash({"n": int(len(mask)), "bits": packed.tobytes().hex()})


_DATASETS: dict = {}
_COORDINATES: dict = {}


def reset_process_caches():
    """Each invocation re-derives its inputs; never trust previous disk caches."""
    _DATASETS.clear()
    _COORDINATES.clear()


def load_dataset(path, sha256: str):
    """Load a standardized dataset once per process (jobs of one dataset run back to back)."""
    stat = Path(path).stat()
    key = (str(path), sha256, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _DATASETS:
        if sha256_file(path) != sha256:
            raise ValueError("dataset bytes differ from the approved job input")
        if len(_DATASETS) >= 2:
            _DATASETS.pop(next(iter(_DATASETS)))
        _DATASETS[key] = load_standard(path)
    return _DATASETS[key]


def cached_coordinates(adata, mask, rep_config: dict, cache_dir, *, dataset_sha256: str, seed: int, code_digest: str = ""):
    key = content_hash({"dataset": dataset_sha256, "mask": _mask_key(mask), "rep": rep_config, "seed": seed,
                        "code": code_digest})
    path = Path(cache_dir) / f"rep_{key}.npz"
    if key in _COORDINATES:
        coords, info = _COORDINATES[key]
        return coords, {**info, "cache_key": key, "cached": True, "cache_scope": "current_process"}
    coords, info = task_coordinates(adata, mask, rep_config, seed=seed)
    # Disk NPZ is diagnostic output only. A matching filename or a self-reported
    # digest cannot prove derivation from approved inputs. Existing files remain
    # untouched and are never loaded; process-local reuse starts after recompute.
    if not path.exists():
        atomic_save_npz(path, coords=coords, info=np.array(json.dumps(info)))
    if len(_COORDINATES) >= 16:
        _COORDINATES.pop(next(iter(_COORDINATES)))
    _COORDINATES[key] = (coords, info)
    return coords, {**info, "cache_key": key, "cached": False, "cache_scope": "recomputed"}


def _subsample(x, cap: int, rng):
    return x if len(x) <= cap else x[rng.choice(len(x), cap, replace=False)]


def task_features(data, pair, source) -> dict:
    """Moderators computed only from information available before the target is seen."""
    rng = np.random.default_rng(derive_seed("features", data.spec.dataset_id, pair.individual,
                                            pair.source_time, pair.target_time))
    clock = data.clock
    features = {"clock_gap": clock[pair.target_time] - clock[pair.source_time],
                "log_n_source": float(np.log(len(source))), "dimension": int(source.shape[1]),
                "n_train_individuals": len({individual for individual, _ in data.train})}
    small = _subsample(source, 400, rng)
    anchor_key = (pair.individual, pair.anchor_time)
    if pair.anchor_time is not None and anchor_key in data.train:
        features["source_anchor_energy"] = energy_distance(small, _subsample(data.train[anchor_key], 400, rng))
    peers = [cells for (individual, time), cells in sorted(data.train.items())
             if np.isclose(time, pair.source_time) and individual != pair.individual][:5]
    if peers:
        features["source_heterogeneity"] = float(np.mean([energy_distance(small, _subsample(c, 400, rng)) for c in peers]))
    if data.spec.kind == "T-LIO":
        shifts = [float(np.linalg.norm(data.train[(individual, pair.target_time)].mean(0) - cells.mean(0)))
                  for (individual, time), cells in sorted(data.train.items())
                  if np.isclose(time, pair.source_time) and (individual, pair.target_time) in data.train]
        if shifts:
            features["population_shift"] = float(np.mean(shifts))
    return features


def _method_params(protocol: dict, name: str) -> dict:
    params = dict(protocol["methods"][name].get("params", {}))
    if REGISTRY[method_base(protocol, name)].spec.family == "flow":
        params = {**protocol.get("flow_defaults", {}), **params}
    return params


def predict_evaluation_draw(method, source, *, draw_seed: int, **kwargs):
    """Evaluation-only seed adapter; never take a fit seed from a job.

    The shared Method API outside this module still calls its keyword ``seed``.
    TargetMarginal also exposes the literal ``draw_seed`` API. No trainer or
    public base interface is changed here.
    """
    if isinstance(draw_seed, bool) or not isinstance(draw_seed, int) or draw_seed < 0 or "seed" in kwargs:
        raise ValueError("predict_evaluation_draw requires a nonnegative draw_seed, not a fit seed keyword")
    keyword = "draw_seed" if method.spec.name == "target_marginal" else "seed"
    return method.predict(source, **kwargs, **{keyword: draw_seed})


def _group_rows(obs, individual, time_value):
    return np.flatnonzero((obs["individual"].astype(str).to_numpy() == str(individual))
                         & (obs["time_value"].to_numpy(dtype=float) == float(time_value)))


def _row_ids(dataset_sha256, rows):
    # These are exact positions in hash-bound standardized input, not invented barcodes.
    return np.asarray([f"{dataset_sha256}:{int(row)}" for row in rows], dtype=str)


def _identity_arrays(prefix, rows, obs, dataset_sha256):
    return {f"{prefix}_rows": np.asarray(rows, dtype=np.int64),
            f"{prefix}_ids": _row_ids(dataset_sha256, rows),
            f"{prefix}_obs_names": obs.index.to_numpy(dtype=str)[rows],
            f"{prefix}_metadata_json": np.array(obs.iloc[rows].to_json(orient="records"))}


def rescore_evaluation_artifact(path, *, expected_sha256=None) -> list:
    """Re-score opt-in coordinate/identity artifacts; does not fit or integrate a field."""
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("evaluation artifact SHA256 mismatch")
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        options = evaluation_contract_options({"contract": metadata["contract"]})
        frozen = freeze_train_mmd_bandwidth(saved["mmd_reference_coordinates"], saved["mmd_reference_ids"],
                                          seed=options["bandwidth_seed"], cap=options["bandwidth_cap"])
        reference = metadata["mmd_reference"]
        if frozen["reference_hash"] != reference["reference_hash"] or frozen["bandwidth"] != reference["bandwidth"]:
            raise ValueError("saved MMD reference differs from its frozen contract")
        records = []
        for item in metadata["pairs"]:
            key, record = item["key"], item["record"]
            for role in ("source", "target"):
                ids = saved[f"{key}_{role}_ids"]
                expected = _row_ids(metadata["dataset_sha256"], saved[f"{key}_{role}_rows"])
                if not np.array_equal(ids, expected) or len(ids) != len(saved[f"{key}_{role}"]):
                    raise ValueError(f"unaligned {role} identities in evaluation artifact")
            origin_ids = _row_ids(metadata["dataset_sha256"], saved[f"{key}_prediction_rows"])
            if (not np.array_equal(saved[f"{key}_prediction_ids"], origin_ids)
                    or len(origin_ids) != len(saved[f"{key}_prediction"])):
                raise ValueError("unaligned prediction origins in evaluation artifact")
            scores = evaluate(saved[f"{key}_prediction"], saved[f"{key}_target"], seed=record["score_seed"],
                              caps=metadata["caps"], n_projections=metadata["n_projections"],
                              mmd_bandwidth=reference["bandwidth"], return_indices=True,
                              metric_names=metadata["metric_names"])
            if scores.get("evaluation_indices") != item["scores"].get("evaluation_indices"):
                raise ValueError("scoring row selections differ from the saved evaluation contract")
            records.append({**record, **sanitize(scores)})
        return records


def run_population_job(job: dict, *, dataset_file, cache_dir, protocol: dict, device: str = "cpu",
                       out_dir=None, save_predictions: bool = True) -> dict:
    stages, begun = {}, time.perf_counter()
    options = evaluation_contract_options(protocol["evaluation"])
    if options is not None:
        if (job["params"].get("evaluation_contract") != options["name"]
                or job["params"].get("evaluation_draw_seed") != options["draw_seed"]
                or job.get("protocol_hash") != protocol_hash(protocol)):
            raise ValueError("opt-in evaluation requires a job bound to the same evaluation protocol")
        if not save_predictions or out_dir is None:
            raise ValueError("opt-in evaluation requires saved predictions and an output directory")
        if any((Path(out_dir) / name).exists() for name in ("predictions.npz", "evaluation_contract.npz", "model_replay.npz", "result.json")):
            raise FileExistsError("opt-in evaluation refuses to overwrite existing result artifacts")
    adata = load_dataset(dataset_file, job["params"]["dataset_sha256"])
    if options is not None and options["name"] == PAIRED_EVALUATION_CONTRACT:
        from .paired import run_paired_job

        return run_paired_job(job, adata, protocol=protocol, device=device, out_dir=out_dir)
    stages['load_dataset'] = time.perf_counter() - begun
    task = TaskSpec.from_dict(job["task"])
    rep_config = dict(protocol["representations"][job["representation"]])
    mask = representation_mask(adata.obs, task)
    tick = time.perf_counter()
    coords, rep_info = cached_coordinates(adata, mask, rep_config, cache_dir,
                                          dataset_sha256=job["params"]["dataset_sha256"],
                                          seed=int(protocol.get("representation_seed", 0)),
                                          code_digest=job["params"].get("code_digest", ""))
    stages['representation'] = time.perf_counter() - tick
    clock_name = job["params"].get("clock", protocol["clock"]["primary"])
    clock = make_clock(adata.obs["time_value"].unique(), clock_name)
    data = build_task_data(coords, adata.obs, task, clock, min_cells=int(protocol.get("min_cells_per_group", 1)))
    method = create(method_base(protocol, job["method"]), **_method_params(protocol, job["method"]))
    evaluation_arrays, evaluation_pairs = {}, []
    if options is not None:
        missing = set(options["identity_columns"]) - set(adata.obs.columns)
        if missing:
            raise ValueError(f"required original identity columns unavailable: {sorted(missing)}")
        training_rows = np.concatenate([_group_rows(adata.obs, *key) for key in sorted(data.train)])
        frozen_mmd = freeze_train_mmd_bandwidth(coords[training_rows],
                                               _row_ids(job["params"]["dataset_sha256"], training_rows),
                                               seed=options["bandwidth_seed"], cap=options["bandwidth_cap"])
        reference_rows = training_rows[frozen_mmd["selected_indices"]]
        evaluation_arrays.update(mmd_reference_coordinates=coords[reference_rows].astype(np.float64),
                                 mmd_reference_ids=_row_ids(job["params"]["dataset_sha256"], reference_rows),
                                 mmd_reference_rows=reference_rows,
                                 mmd_training_ids=_row_ids(job["params"]["dataset_sha256"], training_rows))
        metric_names = [protocol["evaluation"].get("primary_metric", "energy")]
        metric_names += list(protocol["evaluation"].get("secondary_metrics", []))
        if job["params"].get("evaluation_stochastic") != bool(getattr(method, "evaluation_stochastic", False)):
            raise ValueError("job evaluation-randomness marker differs from method implementation")
    def synchronize():
        if str(device).startswith('cuda'):
            import torch
            torch.cuda.synchronize(device)
    synchronize()
    started = time.perf_counter()
    fit_info = method.fit(method_inputs(data), seed=int(job["seed"]), device=device)
    synchronize()
    fit_seconds = time.perf_counter() - started
    stages['fit_including_coupling'] = fit_seconds
    caps = protocol["evaluation"]["caps"]
    projections = int(protocol["evaluation"].get("sliced_projections", 128))
    records, predictions = [], {}

    def retain_evaluation(key, prediction, source, source_rows, target, target_rows, record, scores):
        prediction_rows = source_rows
        origin = "initialized_source_row_not_biological_lineage"
        if not method.spec.uses_source:
            origin = "sampled_training_target_row"
            prediction_rows = np.concatenate([_group_rows(adata.obs, *group)[rows]
                                               for group, rows in method.last_draw_rows])
        if len(prediction_rows) != len(prediction):
            raise ValueError("prediction origins do not align with predicted particles")
        evaluation_arrays.update({f"{key}_prediction": np.asarray(prediction, dtype=np.float64),
                                  f"{key}_source": np.asarray(source, dtype=np.float64),
                                  f"{key}_target": np.asarray(target, dtype=np.float64),
                                  f"{key}_prediction_ids": _row_ids(job["params"]["dataset_sha256"], prediction_rows),
                                  f"{key}_prediction_rows": prediction_rows})
        for role, rows in (("source", source_rows), ("target", target_rows), ("prediction_origin", prediction_rows)):
            evaluation_arrays.update(_identity_arrays(f"{key}_{role}", rows, adata.obs,
                                                       job["params"]["dataset_sha256"]))
        evaluation_pairs.append({"key": key, "record": {**record, "prediction_identity_kind": origin},
                                 "scores": sanitize(scores)})

    for index, pair in enumerate(task.pairs):
        source = data.sources[(pair.individual, pair.source_time)]
        target = data.targets[(pair.individual, pair.target_time)]
        eval_seed = derive_seed(task.dataset_id, task.kind, task.family, pair.individual, pair.source_time, pair.target_time)
        if options is not None:
            draw_seed = derive_seed("evaluation-draw-v1", options["draw_seed"], task.dataset_id, task.kind,
                                    task.family, pair.individual, pair.source_time, pair.target_time)
            eval_seed = derive_seed("evaluation-score-v1", draw_seed)
            source_rows = _group_rows(adata.obs, pair.individual, pair.source_time)
            target_rows = _group_rows(adata.obs, pair.individual, pair.target_time)
        tick = time.perf_counter()
        call = dict(individual=pair.individual, source_time=pair.source_time,
                    target_time=pair.target_time, anchor_time=pair.anchor_time)
        prediction = (method.predict(source, **call, seed=int(job["seed"])) if options is None else
                      predict_evaluation_draw(method, source, **call, draw_seed=draw_seed))
        synchronize()
        elapsed = time.perf_counter() - tick
        stages['predict'] = stages.get('predict', 0.) + elapsed
        tick = time.perf_counter()
        score_options = ({} if options is None else {"mmd_bandwidth": frozen_mmd["bandwidth"],
                         "return_indices": True, "metric_names": metric_names})
        scores = evaluate(prediction, target, seed=eval_seed, caps=caps, n_projections=projections, **score_options)
        stages['score'] = stages.get('score', 0.) + time.perf_counter() - tick
        tick = time.perf_counter()
        features = {f"feature_{k}": v for k, v in task_features(data, pair, source).items()}
        stages['features'] = stages.get('features', 0.) + time.perf_counter() - tick
        common = {"pair_index": index, "individual": pair.individual, "source_time": pair.source_time,
                  "target_time": pair.target_time, "anchor_time": pair.anchor_time}
        if options is not None:
            common.update(evaluation_contract=options["name"], fit_seed=int(job["seed"]), draw_seed=draw_seed,
                          score_seed=eval_seed, mmd_reference_hash=frozen_mmd["reference_hash"],
                          evaluation_stochastic=bool(getattr(method, "evaluation_stochastic", False)))
            retain_evaluation(f"pair{index}_own", prediction, source, source_rows, target, target_rows,
                              {**common, "source_variant": "own"}, scores)
        records.append({**common, "source_variant": "own", "predict_seconds": elapsed, **scores, **features})
        if save_predictions:
            predictions[f"pair{index}_own"] = prediction.astype(np.float32)
        if task.family == "source_swap":
            # Leave-two-out: the substitute individual was excluded from fitting, like the held individual.
            cells = data.sources[(task.substitute, pair.source_time)]
            draw = np.random.default_rng(derive_seed("T-SRC-draw", task.dataset_id, pair.individual, task.substitute,
                                                     pair.source_time))
            if options is not None:
                draw = np.random.default_rng(derive_seed("evaluation-source-swap-v1", draw_seed, task.substitute))
            chosen = draw.choice(len(cells), size=len(source), replace=len(cells) < len(source))
            matched = cells[chosen]
            call = dict(individual=pair.individual, source_time=pair.source_time, target_time=pair.target_time)
            substitute = (method.predict(matched, **call, seed=int(job["seed"])) if options is None else
                          predict_evaluation_draw(method, matched, **call, draw_seed=draw_seed))
            sub_scores = evaluate(substitute, target, seed=eval_seed, caps=caps, n_projections=projections, **score_options)
            records.append({**common, "source_variant": f"substitute:{task.substitute}", **sub_scores})
            if save_predictions:
                safe = re.sub(r"[^A-Za-z0-9_]", "_", str(task.substitute))
                predictions[f"pair{index}_sub_{safe}"] = substitute.astype(np.float32)
                if options is not None:
                    sub_rows = _group_rows(adata.obs, task.substitute, pair.source_time)[chosen]
                    retain_evaluation(f"pair{index}_sub_{safe}", substitute, matched, sub_rows, target, target_rows,
                                      {**common, "source_variant": f"substitute:{task.substitute}"}, sub_scores)
    if save_predictions and out_dir is not None:
        tick = time.perf_counter()
        atomic_save_npz(Path(out_dir) / "predictions.npz", **predictions)
        if options is not None:
            metadata = {"contract": options, "dataset_sha256": job["params"]["dataset_sha256"],
                        "identity_level": "dataset_sha256_standard_row; original metadata retained when present",
                        "mmd_reference": frozen_mmd, "caps": caps, "n_projections": projections,
                        "metric_names": metric_names, "pairs": evaluation_pairs,
                        "scope": "one evaluation draw per pair; no fields, new donors or symmetric50 schedule"}
            atomic_save_npz(Path(out_dir) / "evaluation_contract.npz", **evaluation_arrays,
                            metadata_json=np.array(canonical_json(metadata)))
        stages['prediction_io'] = time.perf_counter() - tick
    result = {"records": records, "fit": fit_info, "fit_seconds": fit_seconds, "variant": method.variant,
            "representation": rep_info, "device": str(device), "clock": clock_name,
            'stage_seconds': stages, 'worker_seconds': time.perf_counter() - begun,
            'timing_scope': 'synchronized_fit_and_prediction_including_coupling_not_equal_wall_budget'}
    if options is not None:
        result.update(evaluation_contract=options["name"], mmd_reference=frozen_mmd,
                      evaluation_artifact_sha256=sha256_file(Path(out_dir) / "evaluation_contract.npz"),
                      evaluation_scope="single draw interface, not symmetric50 or independent biological validation")
    return result
