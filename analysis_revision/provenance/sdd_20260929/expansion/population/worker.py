"""Execute one population-benchmark job: representation, fit, predictions and scores."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from ..common.io import atomic_save_npz, content_hash, sha256_file
from ..common.rng import derive_seed
from ..data.standard import load_standard
from .methods import REGISTRY, create
from .metrics import energy_distance, evaluate
from .represent import task_coordinates
from ..runner.jobs import method_base
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


def run_population_job(job: dict, *, dataset_file, cache_dir, protocol: dict, device: str = "cpu",
                       out_dir=None, save_predictions: bool = True) -> dict:
    stages, begun = {}, time.perf_counter()
    adata = load_dataset(dataset_file, job["params"]["dataset_sha256"])
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
    for index, pair in enumerate(task.pairs):
        source = data.sources[(pair.individual, pair.source_time)]
        target = data.targets[(pair.individual, pair.target_time)]
        eval_seed = derive_seed(task.dataset_id, task.kind, task.family, pair.individual, pair.source_time, pair.target_time)
        tick = time.perf_counter()
        prediction = method.predict(source, individual=pair.individual, source_time=pair.source_time,
                                    target_time=pair.target_time, anchor_time=pair.anchor_time, seed=int(job["seed"]))
        synchronize()
        elapsed = time.perf_counter() - tick
        stages['predict'] = stages.get('predict', 0.) + elapsed
        tick = time.perf_counter()
        scores = evaluate(prediction, target, seed=eval_seed, caps=caps, n_projections=projections)
        stages['score'] = stages.get('score', 0.) + time.perf_counter() - tick
        tick = time.perf_counter()
        features = {f"feature_{k}": v for k, v in task_features(data, pair, source).items()}
        stages['features'] = stages.get('features', 0.) + time.perf_counter() - tick
        common = {"pair_index": index, "individual": pair.individual, "source_time": pair.source_time,
                  "target_time": pair.target_time, "anchor_time": pair.anchor_time}
        records.append({**common, "source_variant": "own", "predict_seconds": elapsed, **scores, **features})
        if save_predictions:
            predictions[f"pair{index}_own"] = prediction.astype(np.float32)
        if task.family == "source_swap":
            # Leave-two-out: the substitute individual was excluded from fitting, like the held individual.
            cells = data.sources[(task.substitute, pair.source_time)]
            draw = np.random.default_rng(derive_seed("T-SRC-draw", task.dataset_id, pair.individual, task.substitute,
                                                     pair.source_time))
            matched = cells[draw.choice(len(cells), size=len(source), replace=len(cells) < len(source))]
            substitute = method.predict(matched, individual=pair.individual, source_time=pair.source_time,
                                        target_time=pair.target_time, seed=int(job["seed"]))
            sub_scores = evaluate(substitute, target, seed=eval_seed, caps=caps, n_projections=projections)
            records.append({**common, "source_variant": f"substitute:{task.substitute}", **sub_scores})
            if save_predictions:
                safe = re.sub(r"[^A-Za-z0-9_]", "_", str(task.substitute))
                predictions[f"pair{index}_sub_{safe}"] = substitute.astype(np.float32)
    if save_predictions and out_dir is not None:
        tick = time.perf_counter()
        atomic_save_npz(Path(out_dir) / "predictions.npz", **predictions)
        stages['prediction_io'] = time.perf_counter() - tick
    return {"records": records, "fit": fit_info, "fit_seconds": fit_seconds, "variant": method.variant,
            "representation": rep_info, "device": str(device), "clock": clock_name,
            'stage_seconds': stages, 'worker_seconds': time.perf_counter() - begun,
            'timing_scope': 'synchronized_fit_and_prediction_including_coupling_not_equal_wall_budget'}
