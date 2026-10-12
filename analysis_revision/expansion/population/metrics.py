"""Distribution distances used to score predictions against held-out cells.

Energy distance (V-statistic, the primary metric of the existing studies), a
multi-bandwidth RBF MMD, sliced and exact Wasserstein-2, mean and covariance
(Bures) errors and a classifier two-sample AUC.  Subsampling uses separate
random streams for prediction and target so every method sees the same target
cells in a given comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ..common.io import content_hash


EVALUATION_CONTRACT = "train-side-v1"
PAIRED_EVALUATION_CONTRACT = "train-side-paired-v2"
METRIC_NAMES = ("energy", "mmd", "sliced_w2", "exact_w2", "mean_error", "bures", "c2st_auc")


def evaluation_contract_options(evaluation: dict) -> dict | None:
    """Explicit opt-in only; no draw count, fit seeds or thresholds are inferred."""
    raw = evaluation.get("contract")
    if raw is None:
        return None
    allowed = {"name", "draw_seed", "bandwidth_seed", "bandwidth_cap", "identity_columns"}
    if isinstance(raw, dict) and raw.get("name") == PAIRED_EVALUATION_CONTRACT:
        allowed |= {"draws", "sample_size"}
    if (not isinstance(raw, dict) or set(raw) - allowed
            or raw.get("name") not in {EVALUATION_CONTRACT, PAIRED_EVALUATION_CONTRACT}):
        raise ValueError("unknown population evaluation contract or contract fields")
    options = dict(raw)
    for name in ("draw_seed", "bandwidth_seed"):
        value = options.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evaluation contract requires an explicit nonnegative integer {name}")
    cap = options.setdefault("bandwidth_cap", 1000)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 2:
        raise ValueError("bandwidth_cap must be an integer >= 2")
    columns = options.setdefault("identity_columns", [])
    if (not isinstance(columns, list) or any(not isinstance(c, str) or not c for c in columns)
            or len(columns) != len(set(columns))):
        raise ValueError("identity_columns must be distinct nonempty column names")
    secondary = evaluation.get("secondary_metrics", [])
    if not isinstance(secondary, list):
        raise ValueError("opt-in secondary_metrics must be an explicit list")
    names = [evaluation.get("primary_metric", "energy"), *secondary]
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)) or set(names) - set(METRIC_NAMES):
        raise ValueError("opt-in evaluation needs distinct known metric names")
    if options["name"] == PAIRED_EVALUATION_CONTRACT:
        for name in ("draws", "sample_size"):
            value = options.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (2 if name == "sample_size" else 1):
                raise ValueError(f"paired evaluation requires explicit positive {name}")
        if options["draws"] > 1000:
            raise ValueError("paired evaluation draw count exceeds bounded preparation contract")
        caps = evaluation.get("caps", {})
        for name in ("energy", "exact_w2", "sliced_w2", "c2st"):
            cap = caps.get(name)
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < options["sample_size"]:
                raise ValueError("paired evaluation caps must explicitly retain every matched particle")
    return options


def freeze_train_mmd_bandwidth(training, reference_ids, *, seed: int, cap: int = 1000) -> dict:
    """Freeze one reference from permitted training coordinates, never predictions/targets.

    Callers decide the permitted train view. Reuse both the bandwidth and its
    reference hash across methods; this helper does not approve a scientific split.
    """
    train = np.asarray(training, dtype=np.float64)
    ids = np.asarray(reference_ids, dtype=str)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("MMD reference seed must be a nonnegative integer")
    if (train.ndim != 2 or len(train) < 2 or train.shape[1] == 0
            or not np.isfinite(train).all() or ids.shape != (len(train),)
            or len(set(ids.tolist())) != len(ids) or np.any(ids == "")):
        raise ValueError("MMD reference needs finite training rows with unique aligned identities")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 2:
        raise ValueError("MMD reference cap must be an integer >= 2")
    rows = np.arange(len(train))
    if len(rows) > cap:
        rows = np.sort(np.random.default_rng(seed).choice(len(rows), cap, replace=False))
    selected = train[rows]
    distances = cdist(selected, selected)
    values = distances[np.triu_indices(len(selected), k=1)]
    median = float(np.median(values))
    if not np.isfinite(median):
        raise ValueError("MMD training-reference distances must be finite")
    bandwidth = median if median > 0 else 1.0  # Same degenerate-reference fallback as the legacy heuristic.
    return {"bandwidth": bandwidth, "seed": int(seed), "cap": cap, "training_rows": len(train),
            "selected_indices": rows.tolist(), "reference_ids": ids[rows].tolist(),
            "reference_hash": content_hash({"ids": ids[rows], "coordinates": selected}, length=64)}


def _cap(x, cap, rng, *, return_indices: bool = False):
    x = np.asarray(x, dtype=np.float64)
    rows = np.arange(len(x))
    if cap and len(x) > cap:
        rows = np.sort(rng.choice(len(x), size=int(cap), replace=False))
    return (x[rows], rows) if return_indices else x[rows]


def _mean_pairwise_distance(a, b, chunk: int = 2048) -> float:
    total = 0.0
    for start in range(0, len(a), chunk):
        total += float(cdist(a[start:start + chunk], b).sum())
    return total / (len(a) * len(b))


def energy_distance(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(2.0 * _mean_pairwise_distance(x, y) - _mean_pairwise_distance(x, x) - _mean_pairwise_distance(y, y))


def median_bandwidth(x, y, rng, cap: int = 1000) -> float:
    pooled = _cap(np.vstack([x, y]), cap, rng)
    distances = cdist(pooled, pooled)
    values = distances[np.triu_indices_from(distances, k=1)]
    median = float(np.median(values)) if values.size else 1.0
    return median if median > 0 else 1.0


def mmd_rbf(x, y, *, bandwidth: float, scales=(0.5, 1.0, 2.0)) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dxx, dyy, dxy = cdist(x, x, "sqeuclidean"), cdist(y, y, "sqeuclidean"), cdist(x, y, "sqeuclidean")
    total = 0.0
    for scale in scales:
        gamma = 1.0 / (2.0 * (scale * bandwidth) ** 2)
        total += np.exp(-gamma * dxx).mean() + np.exp(-gamma * dyy).mean() - 2.0 * np.exp(-gamma * dxy).mean()
    return float(total / len(scales))


def sliced_w2(x, y, *, n_projections: int = 128, rng, n_quantiles: int = 200) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    directions = rng.normal(size=(int(n_projections), x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    levels = (np.arange(n_quantiles) + 0.5) / n_quantiles
    qx = np.quantile(x @ directions.T, levels, axis=0)
    qy = np.quantile(y @ directions.T, levels, axis=0)
    return float(np.sqrt(np.mean((qx - qy) ** 2)))


def exact_w2(x, y) -> float:
    import ot

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cost = ot.dist(x, y, metric="sqeuclidean")
    value = ot.emd2(np.full(len(x), 1.0 / len(x)), np.full(len(y), 1.0 / len(y)), cost, numItermax=1_000_000)
    return float(np.sqrt(max(float(value), 0.0)))


def mean_error(x, y) -> float:
    return float(np.linalg.norm(np.mean(x, axis=0) - np.mean(y, axis=0)))


def sqrtm_psd(matrix) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def _covariance(x) -> np.ndarray:
    return np.atleast_2d(np.cov(np.asarray(x, dtype=np.float64), rowvar=False, bias=True))


def bures_error(x, y) -> float:
    cx, cy = _covariance(x), _covariance(y)
    root = sqrtm_psd(cx)
    cross = sqrtm_psd(root @ cy @ root)
    return float(np.sqrt(max(np.trace(cx) + np.trace(cy) - 2.0 * np.trace(cross), 0.0)))


def c2st_auc(x, y, *, seed: int, folds: int = 5) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = min(len(x), len(y))
    if n < 2 * folds:
        return float("nan")
    rng = np.random.default_rng(seed)
    xs = np.asarray(x, dtype=np.float64)[rng.choice(len(x), n, replace=False)]
    ys = np.asarray(y, dtype=np.float64)[rng.choice(len(y), n, replace=False)]
    features = np.vstack([xs, ys])
    labels = np.r_[np.zeros(n), np.ones(n)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_predict(model, features, labels, cv=splitter, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, scores))


def evaluate(prediction, target, *, seed: int, caps: dict, n_projections: int = 128,
             mmd_bandwidth: float | None = None, return_indices: bool = False,
             metric_names=None) -> dict:
    """Legacy defaults remain adaptive/all-metric; new callers supply a frozen kernel.

    Explicit metric selection permits CPU contract checks without classifier
    fitting or transport optimization. It never changes legacy default metrics.
    """
    names = METRIC_NAMES if metric_names is None else tuple(metric_names)
    if not names or len(names) != len(set(names)) or set(names) - set(METRIC_NAMES):
        raise ValueError("metric_names must be distinct known metrics")
    if mmd_bandwidth is not None and (not np.isfinite(mmd_bandwidth) or mmd_bandwidth <= 0):
        raise ValueError("frozen MMD bandwidth must be finite and positive")
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    base = {"n_prediction": int(len(p)), "n_target": int(len(t))}
    if not (np.isfinite(p).all() and np.isfinite(t).all()):
        return {**base, "finite": False, **dict.fromkeys(names, None)}
    prediction_rng = np.random.default_rng([seed, 1])
    target_rng = np.random.default_rng([seed, 2])
    shared_rng = np.random.default_rng([seed, 3])
    capped, indices = {}, {}
    # Preserve the legacy cap order/random streams even for an explicit metric subset.
    for name, cap in (("energy", 2000), ("exact_w2", 1000), ("sliced_w2", 4000), ("c2st", 2000)):
        left, pi = _cap(p, caps.get(name, cap), prediction_rng, return_indices=True)
        right, ti = _cap(t, caps.get(name, cap), target_rng, return_indices=True)
        capped[name] = (left, right)
        indices[name] = {"prediction": pi.tolist(), "target": ti.tolist()}
    pe, te = capped["energy"]
    result = {**base, "finite": True}
    for name in METRIC_NAMES:
        if name not in names:
            continue
        if name == "energy":
            result[name] = energy_distance(pe, te)
        elif name == "mmd":
            bandwidth = median_bandwidth(pe, te, shared_rng) if mmd_bandwidth is None else float(mmd_bandwidth)
            result[name] = mmd_rbf(pe, te, bandwidth=bandwidth)
        elif name == "sliced_w2":
            # A frozen-kernel call cannot consume the projection stream via bandwidth estimation.
            rng = shared_rng if mmd_bandwidth is None else np.random.default_rng([seed, 3])
            result[name] = sliced_w2(*capped[name], n_projections=n_projections, rng=rng)
        elif name == "exact_w2":
            result[name] = exact_w2(*capped[name])
        elif name == "mean_error":
            result[name] = mean_error(p, t)
        elif name == "bures":
            result[name] = bures_error(p, t)
        elif name == "c2st_auc":
            result[name] = c2st_auc(*capped["c2st"], seed=seed)
    if mmd_bandwidth is not None:
        result["mmd_bandwidth"] = float(mmd_bandwidth)
    if return_indices:
        result["evaluation_indices"] = indices
    return result
