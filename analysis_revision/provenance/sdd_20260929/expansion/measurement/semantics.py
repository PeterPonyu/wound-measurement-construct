"""Prevalence-versus-intensity test for a per-cell program score.

Conditional on supplied scores and lineage labels (not end-to-end cross-fitting),
each specimen gets model parameters fitted on half of its cells and is scored
on the other half (and vice versa, repeated over random splits).  Shared shapes
are estimated without the specimen's unit (patient), so a specimen never shapes
its own reference.  Models, each with the stated number of specimen-level
parameters:

  C1  shift of one shared shape (1)          C2  shift and rescaling of it (2)
  M1  re-weighting of two shared components (1)
  M2  re-weighting plus a common shift of both components (2)
  H1  components split at a fixed cut, weight = fraction above the cut (1)

The comparison is predictive (held-out log-likelihood); it describes which
statistical family fits specimen-to-specimen variation better.  It does not by
itself establish discrete biological states.

Exact zeros (common in NMF proportions) are a point mass that a continuous
density cannot represent.  Every model therefore shares the same two-part
likelihood: a Bernoulli term for the zero fraction (fitted on the same half)
plus the model's density for positive values, so zeros cannot favour one model.
Comparisons are descriptive: overlapping reference-training folds and multiple
programmes are not converted into calibrated sign-test p values here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar

from ..common.io import content_hash
from ..common.rng import np_rng

MODELS = ("C1", "C2", "M1", "M2", "H1")
COMPARISONS = (("M1", "C1"), ("M2", "C2"), ("M1", "C2"), ("H1", "C1"), ("M1", "H1"))
FLOOR = 1e-300
UNIFORM_WEIGHT = 1e-6
TAIL_WEIGHT = 1e-6
ESTIMATOR_VERSION = 'conditional-unit-density-v2-training-grid-exponential-tails'


def transform(values, scale: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('program scores must be finite')
    if scale == 'logit' and np.any((values < 0) | (values > 1)):
        raise ValueError('logit scores must be in [0, 1]')
    if scale == 'log' and np.any(values < 0):
        raise ValueError('log scores must be nonnegative')
    if scale == "logit":
        clipped = np.clip(values, 1e-7, 1 - 1e-7)
        return np.log(clipped / (1 - clipped))
    if scale == "log":
        return np.log(np.maximum(values, 0.0) + 1e-4)
    if scale == "identity":
        return values
    raise ValueError(f"unknown scale {scale!r}")


def kde_grid(values, grid, weights=None, *, support=None) -> np.ndarray:
    """Piecewise-linear interior density of unit integral; log_density adds fixed tails."""
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all() or weights.shape != values.shape
            or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('KDE requires finite values and nonnegative nonzero weights')
    if (grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all() or grid[1] <= grid[0]
            or not np.allclose(np.diff(grid), grid[1] - grid[0], rtol=1e-7, atol=1e-12)):
        raise ValueError('KDE grid must be finite, increasing and equally spaced')
    if values.min() < grid[0] or values.max() > grid[-1]:
        raise ValueError('training values outside KDE grid')
    weights = weights / weights.sum()
    effective = 1.0 / float(np.sum(weights ** 2))
    mean = float(np.sum(weights * values))
    sd = float(np.sqrt(np.sum(weights * (values - mean) ** 2)))
    dx = float(grid[1] - grid[0])
    bandwidth = max(1.06 * sd * effective ** (-0.2), dx)
    position = (values - grid[0]) / dx
    left = np.clip(np.floor(position).astype(int), 0, len(grid) - 2)
    fraction = np.clip(position - left, 0.0, 1.0)
    mass = (np.bincount(left, weights=weights * (1 - fraction), minlength=len(grid))
            + np.bincount(left + 1, weights=weights * fraction, minlength=len(grid)))
    density = gaussian_filter1d(mass, sigma=bandwidth / dx, mode="constant")
    if support is not None:
        low, high = support
        density[~((grid > low) & (grid <= high))] = 0.0
    density = np.maximum(density, FLOOR)
    density = density / np.trapezoid(density, grid)
    return (1.0 - UNIFORM_WEIGHT) * density + UNIFORM_WEIGHT / float(grid[-1] - grid[0])


def log_density(density, grid, x) -> np.ndarray:
    """Normalized interior plus two exponential tails (each TAIL_WEIGHT/2).

    Tail scale is grid width/8, fixed from training only. The piecewise density
    need not be continuous at the endpoints; endpoint values have zero mass.
    Unlike a constant exterior floor, the tail is an integrable probability law.
    """
    x = np.asarray(x, dtype=float)
    interior = np.log1p(-TAIL_WEIGHT) + np.log(np.maximum(np.interp(x, grid, density), FLOOR))
    tail_scale = float(grid[-1] - grid[0]) / 8.0
    distance = np.maximum(grid[0] - x, x - grid[-1])
    tails = np.log(TAIL_WEIGHT / (2 * tail_scale)) - distance / tail_scale
    return np.where((x < grid[0]) | (x > grid[-1]), tails, interior)


def shared_shapes(groups, grid, cut=None) -> dict:
    from sklearn.mixture import GaussianMixture

    pooled = np.concatenate(groups)
    shapes = {"shape": kde_grid(np.concatenate([g - np.median(g) for g in groups]), grid)}
    mixture = GaussianMixture(n_components=2, random_state=0).fit(pooled[:, None])
    order = np.argsort(mixture.means_.ravel())
    responsibility = mixture.predict_proba(pooled[:, None])
    shapes["low"] = kde_grid(pooled, grid, responsibility[:, order[0]] + 1e-12)
    shapes["high"] = kde_grid(pooled, grid, responsibility[:, order[1]] + 1e-12)
    shapes["cut"] = float(np.mean(mixture.means_.ravel())) if cut is None else float(cut)
    below, above = pooled[pooled <= shapes["cut"]], pooled[pooled > shapes["cut"]]
    shapes["hard_available"] = bool(len(below) > 1 and len(above) > 1)
    if shapes["hard_available"]:
        shapes["hard_low"] = kde_grid(below, grid, support=(-np.inf, shapes["cut"]))
        shapes["hard_high"] = kde_grid(above, grid, support=(shapes["cut"], np.inf))
    return shapes


def _mixture(weight, low, high):
    return np.logaddexp(np.log1p(-weight) + low, np.log(weight) + high)


def _fit_evaluate(fit, held, shapes: dict, grid) -> dict:
    lp = lambda density, x: log_density(density, grid, x)  # noqa: E731
    out = {}
    centre = float(np.median(fit))
    offsets = centre + np.linspace(-3.0, 3.0, 121)
    coarse = lp(shapes["shape"], fit[None, :] - offsets[:, None]).sum(axis=1)
    best = float(offsets[int(np.argmax(coarse))])
    shift = minimize_scalar(lambda m: -lp(shapes["shape"], fit - m).sum(), bounds=(best - 0.05, best + 0.05),
                            method="bounded").x
    out["C1"] = float(lp(shapes["shape"], held - shift).mean())
    shifts = centre + np.linspace(-3.0, 3.0, 61)
    scales = np.exp(np.linspace(-1.2, 1.2, 25))
    scaled = (fit[None, None, :] - shifts[:, None, None]) / scales[None, :, None]
    surface = (lp(shapes["shape"], scaled) - np.log(scales)[None, :, None]).sum(axis=2)
    i, j = np.unravel_index(int(np.argmax(surface)), surface.shape)
    out["C2"] = float((lp(shapes["shape"], (held - shifts[i]) / scales[j]) - np.log(scales[j])).mean())
    low, high = lp(shapes["low"], fit), lp(shapes["high"], fit)
    weight = minimize_scalar(lambda w: -_mixture(w, low, high).sum(), bounds=(1e-6, 1 - 1e-6), method="bounded").x
    out["M1"] = float(_mixture(weight, lp(shapes["low"], held), lp(shapes["high"], held)).mean())
    moves = np.linspace(-1.5, 1.5, 31)
    weights = np.linspace(0.005, 0.995, 67)
    low_moved = lp(shapes["low"], fit[None, :] - moves[:, None])
    high_moved = lp(shapes["high"], fit[None, :] - moves[:, None])
    surface = np.logaddexp(np.log1p(-weights)[None, :, None] + low_moved[:, None, :],
                           np.log(weights)[None, :, None] + high_moved[:, None, :]).sum(axis=2)
    i, j = np.unravel_index(int(np.argmax(surface)), surface.shape)
    out["M2"] = float(_mixture(weights[j], lp(shapes["low"], held - moves[i]), lp(shapes["high"], held - moves[i])).mean())
    if shapes["hard_available"]:
        above = float(np.clip(np.mean(fit > shapes["cut"]), 1e-4, 1 - 1e-4))
        out["H1"] = float(_mixture(above, lp(shapes["hard_low"], held), lp(shapes["hard_high"], held)).mean())
    else:
        out["H1"] = float("nan")
    return out


def summarize(table: pd.DataFrame) -> dict:
    if table.empty:
        return {'available': False, 'reason': 'no evaluable specimens after training/split checks', 'specimens': 0, 'units': 0}
    out = {"available": True, "specimens": int(len(table)), "units": int(table["unit"].nunique()),
           'inference_policy': 'descriptive_only_shared_training_folds_and_unfrozen_multiplicity'}
    for first, second in COMPARISONS:
        valid = table[[first, second]].notna().all(axis=1)
        difference = (table.loc[valid, first] - table.loc[valid, second]).to_numpy()
        unit_means = table.loc[valid].assign(d=difference).groupby('unit')['d'].mean()
        per_unit = unit_means.to_numpy()
        wins, unit_wins = int((difference > 0).sum()), int((per_unit > 0).sum())
        out[f"{first}_vs_{second}"] = {
            "specimens": int(len(difference)), "specimens_favour_first": wins,
            "specimen_sign_p": None,
            "median_difference_nats_per_cell": float(np.median(difference)) if len(difference) else None,
            "units": int(len(per_unit)), "units_favour_first": unit_wins,
            "unit_sign_p": None,
            'unit_mean_differences': unit_means.to_dict(),
            'unit_ties': int((per_unit == 0).sum()), 'unit_aggregation': 'equal_specimen_mean_within_unit'}
    return out


def _two_part(fit_zero, held_zero, fit_positive, held_positive, shapes, grid, zero_mass: bool) -> dict:
    """Held-out mean log-likelihood per cell, with a shared Bernoulli zero term when zero_mass is on."""
    if len(fit_positive) < 2 or len(held_positive) < 2:
        raise ValueError('insufficient positive observations in cell split')
    scores = _fit_evaluate(fit_positive, held_positive, shapes, grid)
    if not zero_mass:
        return scores
    fraction = float(np.clip(np.mean(fit_zero), 1e-4, 1 - 1e-4))
    n_zero, n_positive = int(np.sum(held_zero)), int(len(held_positive))
    total = n_zero + n_positive
    return {model: (n_zero * np.log(fraction) + n_positive * (np.log1p(-fraction) + value)) / total
            for model, value in scores.items()}


def semantics_test(values, specimens, units, *, scale: str = "logit", cut=None, n_splits: int = 10, seed: int = 0,
                   min_cells: int = 50, max_cells: int = 4000, zero_mass=None, min_positive: int = 20):
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError('scores must be one-dimensional')
    transformed = transform(raw, scale)
    for name, value, minimum in (('n_splits', n_splits, 1), ('min_cells', min_cells, 2),
                                 ('max_cells', max_cells, 4), ('min_positive', min_positive, 2)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    identifiers = {}
    for name, source in (('specimen', specimens), ('unit', units)):
        source = np.asarray(source)
        if source.ndim != 1 or len(source) != len(raw):
            raise ValueError(f'{name} identifiers must align with scores')
        text = pd.Series(source, dtype='string')
        if text.isna().any() or text.str.strip().str.lower().isin(['', 'none', 'nan', '<na>', 'null', 'nat']).any():
            raise ValueError(f'missing {name} identifier')
        identifiers[name] = text.astype(str).to_numpy()
    zero_mass = (scale in {"logit", "log"}) if zero_mass is None else bool(zero_mass)
    if zero_mass and np.any(raw < 0):
        raise ValueError('a zero-mass score cannot be negative')
    zero = raw == 0.0 if zero_mass else np.zeros(len(raw), dtype=bool)
    frame = pd.DataFrame({'x': transformed, 'raw': raw, 'zero': zero, **identifiers})
    if (frame.groupby('specimen')['unit'].nunique() > 1).any():
        raise ValueError('a specimen maps to more than one unit')
    context = {'estimator_version': ESTIMATOR_VERSION, 'scope': 'conditional_on_supplied_scores_and_lineage_labels',
               'end_to_end_cross_fitted': False, 'tail_policy': {'weight': TAIL_WEIGHT, 'scale': 'training_grid_width/8'},
               'score_transform': {'scale': scale, 'logit_clip': [1e-7, 1 - 1e-7], 'log_offset': 1e-4,
                                   'likelihood_scale': 'transformed_score_not_raw_score'},
               'zero_mass': zero_mass, 'reference_weighting': 'pooled_cells_after_per_specimen_cap',
               'split_aggregation': 'equal_direction_and_split_mean_within_specimen',
               'exclusions': [], 'folds': {}}
    sizes = frame.groupby("specimen").size()
    positives = frame[~frame["zero"]].groupby("specimen").size()
    eligible = [s for s in sizes.index if sizes[s] >= min_cells and positives.get(s, 0) >= min_positive]
    context['exclusions'] += [{'specimen': str(s), 'reason': 'insufficient cells or positive scores',
                              'n_cells': int(sizes[s]), 'n_positive': int(positives.get(s, 0))}
                             for s in sizes.index if s not in eligible]
    frame = frame[frame["specimen"].isin(eligible)]
    if frame["specimen"].nunique() < 3 or frame["unit"].nunique() < 2:
        return pd.DataFrame(), {**context, 'available': False, 'reason': 'fewer than 3 specimens or 2 units with enough cells'}
    cut_value = None if cut is None else float(transform([cut], scale)[0])
    groups, raw_means = {}, {}
    for specimen, block in frame.groupby("specimen"):
        index = np.arange(len(block))
        if len(index) > max_cells:
            index = np_rng('semantics-cap', seed, specimen).choice(index, max_cells, replace=False)
        if (~block['zero'].to_numpy()[index]).sum() < 4:
            context['exclusions'].append({'specimen': str(specimen), 'reason': 'fewer than four positives after cell cap'})
            continue
        groups[specimen] = (block["x"].to_numpy()[index], block["zero"].to_numpy()[index])
        raw_means[specimen] = float(block["raw"].mean())
    unit_of = frame.groupby("specimen")["unit"].first()
    rows = []
    for unit in sorted(unit_of.unique()):
        training_ids = sorted(s for s in groups if unit_of[s] != unit)
        training = [groups[s][0][~groups[s][1]] for s in training_ids]
        if len(training) < 2:
            context['folds'][str(unit)] = {'available': False, 'reason': 'fewer than two training specimens'}
            context['exclusions'] += [{'specimen': str(s), 'reason': 'insufficient training specimens in fold'}
                                      for s in groups if unit_of[s] == unit]
            continue
        pooled = np.concatenate(training)
        span = float(np.ptp(pooled))
        if not np.isfinite(span) or span <= 1e-12:
            context['folds'][str(unit)] = {'available': False, 'reason': 'constant or invalid training scores'}
            context['exclusions'] += [{'specimen': str(s), 'reason': 'constant or invalid training scores in fold'}
                                      for s in groups if unit_of[s] == unit]
            continue
        origin = float(np.median(pooled))
        # No held-unit score affects the origin, support, grid, tail scale or shapes.
        extent = span + 1.0 + 0.1 * span
        grid = np.linspace(-extent, extent, 6001)
        training = [g - origin for g in training]
        fold_cut = None if cut_value is None else cut_value - origin
        shapes = shared_shapes(training, grid, fold_cut)
        context['folds'][str(unit)] = {'available': True, 'training_specimens': training_ids,
            'training_units': sorted(set(map(str, unit_of.loc[training_ids]))), 'origin': origin,
            'grid_min': float(grid[0]), 'grid_max': float(grid[-1]), 'grid_points': len(grid),
            'reference_fingerprint': content_hash({'shapes': shapes, 'grid': grid, 'origin': origin}, 64)}
        for specimen in sorted(s for s in groups if unit_of[s] == unit):
            x, z = groups[specimen]
            x = x - origin
            totals = dict.fromkeys(MODELS, 0.0)
            valid_split = True
            for split in range(n_splits):
                order = np_rng('semantics-split', seed, specimen, split).permutation(len(x))
                half = len(x) // 2
                first, second = order[:half], order[half:]
                if min((~z[first]).sum(), (~z[second]).sum()) < 2:
                    valid_split = False
                    break
                for fit, held in ((first, second), (second, first)):
                    scores = _two_part(z[fit], z[held], x[fit][~z[fit]], x[held][~z[held]], shapes, grid, zero_mass)
                    for model, value in scores.items():
                        totals[model] += value / (2 * n_splits)
            if not valid_split:
                context['exclusions'].append({'specimen': str(specimen), 'reason': 'insufficient positives in a split'})
                continue
            if any(not np.isfinite(totals[model]) for model in MODELS if model != 'H1' or shapes['hard_available']):
                raise FloatingPointError(f'non-finite conditional scores for {specimen}')
            positive = x[~z]
            rows.append({"specimen": specimen, "unit": unit, "n_cells": int(len(x)), "zero_fraction": float(np.mean(z)),
                         'n_positive': int(len(positive)), 'n_splits': n_splits,
                         'out_of_domain_fraction': float(np.mean((positive < grid[0]) | (positive > grid[-1]))),
                         'hard_available': shapes['hard_available'],
                         "raw_mean": raw_means[specimen], "cut": shapes["cut"] + origin, **totals})
    table = pd.DataFrame(rows)
    return table, {**context, **summarize(table)}
