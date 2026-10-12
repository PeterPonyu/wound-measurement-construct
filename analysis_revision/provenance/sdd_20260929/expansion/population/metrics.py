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


def _cap(x, cap, rng):
    x = np.asarray(x, dtype=np.float64)
    if cap and len(x) > cap:
        return x[np.sort(rng.choice(len(x), size=int(cap), replace=False))]
    return x


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


def evaluate(prediction, target, *, seed: int, caps: dict, n_projections: int = 128) -> dict:
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    base = {"n_prediction": int(len(p)), "n_target": int(len(t))}
    if not (np.isfinite(p).all() and np.isfinite(t).all()):
        return {**base, "finite": False, "energy": None, "mmd": None, "sliced_w2": None, "exact_w2": None,
                "mean_error": None, "bures": None, "c2st_auc": None}
    prediction_rng = np.random.default_rng([seed, 1])
    target_rng = np.random.default_rng([seed, 2])
    shared_rng = np.random.default_rng([seed, 3])
    pe, te = _cap(p, caps.get("energy", 2000), prediction_rng), _cap(t, caps.get("energy", 2000), target_rng)
    pw, tw = _cap(p, caps.get("exact_w2", 1000), prediction_rng), _cap(t, caps.get("exact_w2", 1000), target_rng)
    ps, ts = _cap(p, caps.get("sliced_w2", 4000), prediction_rng), _cap(t, caps.get("sliced_w2", 4000), target_rng)
    pc, tc = _cap(p, caps.get("c2st", 2000), prediction_rng), _cap(t, caps.get("c2st", 2000), target_rng)
    return {**base, "finite": True,
            "energy": energy_distance(pe, te),
            "mmd": mmd_rbf(pe, te, bandwidth=median_bandwidth(pe, te, shared_rng)),
            "sliced_w2": sliced_w2(ps, ts, n_projections=n_projections, rng=shared_rng),
            "exact_w2": exact_w2(pw, tw),
            "mean_error": mean_error(p, t),
            "bures": bures_error(p, t),
            "c2st_auc": c2st_auc(pc, tc, seed=seed)}
