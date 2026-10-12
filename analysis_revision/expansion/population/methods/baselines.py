"""Information-set-matched simple predictors.

Missing-time variants use the target individual's own source and later anchor
('own_anchor'); leave-individual variants use displacements learned from other
individuals ('population').  All are deterministic given the job seed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..infosets import Info
from ..metrics import sqrtm_psd
from .base import Method, MethodSpec, anchor_cells, interpolation_weight, population_pairs, register

SOURCE = Info.OWN_SOURCE
OWN_ANCHOR = Info.OWN_SOURCE | Info.OWN_OTHER_TIMES
POPULATION = Info.OWN_SOURCE | Info.OTHERS_NONTARGET_TIMES | Info.OTHERS_TARGET_TIME
TWO_WAY = (("own_anchor", OWN_ANCHOR), ("population", POPULATION))


def _cov(x) -> np.ndarray:
    return np.atleast_2d(np.cov(x, rowvar=False, bias=True))


def _weighted_moments(groups):
    """Equal total weight per individual."""
    cells = np.vstack(groups)
    weights = np.concatenate([np.full(len(g), 1.0 / (len(groups) * len(g))) for g in groups])
    mean = weights @ cells
    centred = cells - mean
    return mean, (centred * weights[:, None]).T @ centred


def _gaussian_map(mean_a, cov_a, cov_b) -> np.ndarray:
    dim = len(mean_a)
    ridge = 1e-6 * max(float(np.trace(cov_a)) / dim, 1e-12)
    cov_a = cov_a + ridge * np.eye(dim)
    cov_b = cov_b + ridge * np.eye(dim)
    root = sqrtm_psd(cov_a)
    inverse_root = np.linalg.inv(root)
    return inverse_root @ sqrtm_psd(root @ cov_b @ root) @ inverse_root


def _midranks(values) -> np.ndarray:
    return (rankdata(values, axis=0, method="average") - 0.5) / values.shape[0]


def _quantiles_at(cells, levels) -> np.ndarray:
    """Empirical quantile function (inverted CDF): the exact 1-D optimal-transport map."""
    return np.column_stack([np.quantile(cells[:, j], levels[:, j], method="inverted_cdf")
                            for j in range(cells.shape[1])])


@register
class Identity(Method):
    spec = MethodSpec("identity", "baseline", (("source", SOURCE),), False, False, "Unchanged source (stay-still).")

    def _predict(self, source, **_):
        return source.copy()


@register
class MeanShift(Method):
    spec = MethodSpec("mean_shift", "baseline", TWO_WAY, False, False,
                      "Centroid translation: own source-anchor interpolation or mean training displacement.")

    def _predict(self, source, *, individual, source_time, target_time, anchor_time, **_):
        if self.variant == "own_anchor":
            alpha = interpolation_weight(self.inputs.clock, source_time, target_time, anchor_time)
            anchor = anchor_cells(self.inputs, individual, anchor_time)
            return source + alpha * (anchor.mean(axis=0) - source.mean(axis=0))
        shifts = [tgt.mean(axis=0) - src.mean(axis=0) for _, src, tgt in population_pairs(self.inputs, source_time, target_time)]
        return source + np.mean(shifts, axis=0)


@register
class LocationScale(Method):
    spec = MethodSpec("location_scale", "baseline", TWO_WAY, False, False,
                      "Coordinatewise mean and standard-deviation interpolation.")

    def _predict(self, source, *, individual, source_time, target_time, anchor_time, **_):
        mean_s, sd_s = source.mean(axis=0), source.std(axis=0)
        usable = sd_s > 1e-8
        if self.variant == "own_anchor":
            alpha = interpolation_weight(self.inputs.clock, source_time, target_time, anchor_time)
            anchor = anchor_cells(self.inputs, individual, anchor_time)
            mean = (1 - alpha) * mean_s + alpha * anchor.mean(axis=0)
            sd = (1 - alpha) * sd_s + alpha * anchor.std(axis=0)
            ratio = np.where(usable, sd / np.where(usable, sd_s, 1.0), 0.0)
            return mean + (source - mean_s) * ratio
        pairs = population_pairs(self.inputs, source_time, target_time)
        shift = np.mean([tgt.mean(axis=0) - src.mean(axis=0) for _, src, tgt in pairs], axis=0)
        log_ratio = np.mean([np.log(np.maximum(tgt.std(axis=0), 1e-8) / np.maximum(src.std(axis=0), 1e-8))
                             for _, src, tgt in pairs], axis=0)
        return mean_s + shift + (source - mean_s) * np.where(usable, np.exp(log_ratio), 0.0)


@register
class QuantileMap(Method):
    spec = MethodSpec("quantile_map", "baseline", TWO_WAY, False, False,
                      "Coordinatewise monotone (one-dimensional optimal transport) interpolation.")

    def _predict(self, source, *, individual, source_time, target_time, anchor_time, **_):
        levels = _midranks(source)
        if self.variant == "own_anchor":
            alpha = interpolation_weight(self.inputs.clock, source_time, target_time, anchor_time)
            anchor = anchor_cells(self.inputs, individual, anchor_time)
            return (1 - alpha) * source + alpha * _quantiles_at(anchor, levels)
        shifts = [_quantiles_at(tgt, levels) - _quantiles_at(src, levels)
                  for _, src, tgt in population_pairs(self.inputs, source_time, target_time)]
        return source + np.mean(shifts, axis=0)


@register
class OTDisplacement(Method):
    evaluation_stochastic = True  # Endpoint caps draw cells; this is not fit randomness.
    spec = MethodSpec("ot_displacement", "baseline", (("own_anchor", OWN_ANCHOR),), False, False,
                      "Exact optimal-transport (McCann) displacement interpolation between own source and anchor.")

    def _predict(self, source, *, individual, source_time, target_time, anchor_time, seed=0, **_):
        import ot
        from sklearn.neighbors import NearestNeighbors

        alpha = interpolation_weight(self.inputs.clock, source_time, target_time, anchor_time)
        anchor = anchor_cells(self.inputs, individual, anchor_time)
        cap = int(self.params.get("cap", 2000))
        rng = np.random.default_rng([seed, 17])
        kept = np.sort(rng.choice(len(source), cap, replace=False)) if len(source) > cap else np.arange(len(source))
        chosen = np.sort(rng.choice(len(anchor), cap, replace=False)) if len(anchor) > cap else np.arange(len(anchor))
        xs, yb = source[kept], anchor[chosen]
        plan = ot.emd(np.full(len(xs), 1.0 / len(xs)), np.full(len(yb), 1.0 / len(yb)), ot.dist(xs, yb),
                      numItermax=1_000_000)
        mapped = (plan @ yb) / plan.sum(axis=1, keepdims=True)
        displacement = np.empty_like(source)
        displacement[kept] = mapped - xs
        rest = np.setdiff1d(np.arange(len(source)), kept)
        if rest.size:
            neighbours = min(int(self.params.get("k", 15)), len(xs))
            _, index = NearestNeighbors(n_neighbors=neighbours).fit(xs).kneighbors(source[rest])
            displacement[rest] = (mapped - xs)[index].mean(axis=1)
        return source + alpha * displacement


@register
class GaussianGeodesic(Method):
    spec = MethodSpec("gaussian_geodesic", "baseline", TWO_WAY, False, False,
                      "Bures-Wasserstein (Gaussian optimal transport) map interpolation.")

    def _predict(self, source, *, individual, source_time, target_time, anchor_time, **_):
        if self.variant == "own_anchor":
            alpha = interpolation_weight(self.inputs.clock, source_time, target_time, anchor_time)
            anchor = anchor_cells(self.inputs, individual, anchor_time)
            mean_a, mean_b = source.mean(axis=0), anchor.mean(axis=0)
            mapped = mean_b + (source - mean_a) @ _gaussian_map(mean_a, _cov(source), _cov(anchor))
            return (1 - alpha) * source + alpha * mapped
        pairs = population_pairs(self.inputs, source_time, target_time)
        mean_s, cov_s = _weighted_moments([src for _, src, _ in pairs])
        mean_t, cov_t = _weighted_moments([tgt for _, _, tgt in pairs])
        return mean_t + (source - mean_s) @ _gaussian_map(mean_s, cov_s, cov_t)


@register
class TargetMarginal(Method):
    # Keep the legacy fit-seed classification/planner unchanged. Evaluation draws
    # are stochastic and are explicitly marked by the opt-in evaluation contract.
    evaluation_stochastic = True
    spec = MethodSpec("target_marginal", "baseline", (("population_target", Info.OTHERS_TARGET_TIME),), False, False,
                      "Equal-individual mixture of training cells at the target time; ignores the source.",
                      uses_source=False)

    def predict(self, source, *, draw_seed=None, **kwargs):
        """Support an explicit evaluation draw without changing legacy seed calls."""
        if draw_seed is not None:
            if isinstance(draw_seed, bool) or not isinstance(draw_seed, int) or draw_seed < 0:
                raise ValueError("draw_seed must be a nonnegative integer")
            if "seed" in kwargs:
                raise ValueError("provide draw_seed or legacy seed, not both")
            kwargs["seed"] = draw_seed
        return super().predict(source, **kwargs)

    def _predict(self, source, *, target_time, seed=0, **_):
        entries = [(key, np.asarray(cells, dtype=np.float64)) for key, cells in sorted(self.inputs.train.items())
                   if np.isclose(key[1], target_time)]
        pools = [cells for _, cells in entries]
        if not pools:
            raise ValueError("no training cells at the target time")
        total = len(source)
        counts = np.full(len(pools), total // len(pools))
        counts[: total % len(pools)] += 1
        rng = np.random.default_rng([seed, 29])
        draws, self.last_draw_rows = [], []
        for (key, cells), count in zip(entries, counts):
            if count > 0:
                rows = rng.choice(len(cells), size=int(count), replace=bool(count > len(cells)))
                draws.append(cells[rows])
                self.last_draw_rows.append((key, rows))
        return np.vstack(draws)


def _pooled_capped(groups, cap: int, rng):
    per = max(cap // len(groups), 1)
    cells, weights = [], []
    for group in groups:
        taken = group[np.sort(rng.choice(len(group), per, replace=False))] if len(group) > per else group
        cells.append(taken)
        weights.append(np.full(len(taken), 1.0 / (len(groups) * len(taken))))
    return np.vstack(cells), np.concatenate(weights)


@register
class PopulationOT(Method):
    evaluation_stochastic = True  # Training-endpoint caps are sampled at prediction.
    spec = MethodSpec("population_ot", "baseline", (("population", POPULATION),), False, False,
                      "Entropic OT between pooled training source and target cells, extended by nearest-neighbour displacement.")

    def _predict(self, source, *, source_time, target_time, seed=0, **_):
        import ot
        from sklearn.neighbors import NearestNeighbors

        pairs = population_pairs(self.inputs, source_time, target_time)
        cap = int(self.params.get("cap", 1500))
        rng = np.random.default_rng([seed, 31])
        xs, weights_s = _pooled_capped([src for _, src, _ in pairs], cap, rng)
        yt, weights_t = _pooled_capped([tgt for _, _, tgt in pairs], cap, rng)
        cost = ot.dist(xs, yt)
        cost = cost / max(float(cost.max()), 1e-12)
        plan = ot.sinkhorn(weights_s, weights_t, cost, reg=float(self.params.get("epsilon", 0.05)),
                           numItermax=int(self.params.get("iterations", 2000)))
        if not np.isfinite(plan).all() or plan.sum() <= 0:
            raise FloatingPointError("Sinkhorn plan is not finite")
        row_l1 = float(np.abs(plan.sum(axis=1) - weights_s).sum())
        column_l1 = float(np.abs(plan.sum(axis=0) - weights_t).sum())
        self.last_prediction_diagnostics = {
            "solver": "population_ot_sinkhorn", "row_l1": row_l1, "column_l1": column_l1,
            "mass": float(plan.sum()), "audit_tolerance": 1e-6,
            "both_marginals_within_audit_tolerance": max(row_l1, column_l1) <= 1e-6,
            "iteration_budget": int(self.params.get("iterations", 2000)),
            "scope": "returned_plan_marginals_not_a_convergence_proof_or_equal_cost"}
        mapped = (plan @ yt) / np.maximum(plan.sum(axis=1, keepdims=True), 1e-300)
        neighbours = min(int(self.params.get("k", 15)), len(xs))
        _, index = NearestNeighbors(n_neighbors=neighbours).fit(xs).kneighbors(source)
        return source + (mapped - xs)[index].mean(axis=1)


@register
class NearestIndividual(Method):
    spec = MethodSpec("nearest_individual", "baseline", (("population", POPULATION),), False, False,
                      "Mean displacement of the training individual with the most similar source summary (cosine).")

    def _predict(self, source, *, source_time, target_time, **_):
        context = np.r_[source.mean(axis=0), source.std(axis=0)]

        def similarity(cells):
            other = np.r_[cells.mean(axis=0), cells.std(axis=0)]
            return float(context @ other / max(np.linalg.norm(context) * np.linalg.norm(other), 1e-12))

        pairs = population_pairs(self.inputs, source_time, target_time)
        _, src, tgt = max(pairs, key=lambda item: (similarity(item[1]), item[0]))
        return source + (tgt.mean(axis=0) - src.mean(axis=0))
