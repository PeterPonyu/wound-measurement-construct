"""Cross-dataset inference for method comparisons and source-information tests.

Task-level estimand: log(error_method / error_comparator), negative when the
method is better.  Seeds are averaged within a unit first.  The dataset is the
inferential unit: dataset means are pooled by REML random effects with a
modified Hartung-Knapp-Sidik-Jonkman interval, and win rates across datasets use
exact binomial intervals.  Cells, seeds and draws are never counted as units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

UNIT = ["dataset_id", "representation", "kind", "family", "held_time", "held_individual", "training_size", "repeat",
        "individual", "source_time", "target_time"]


PRIMARY_FAMILIES = ("missing_time", "leave_individual")


def unit_keys(df: pd.DataFrame) -> list:
    """Unit identity; the clock separates primary from sensitivity-clock jobs when present."""
    return UNIT + (["clock"] if "clock" in df.columns else [])


def unit_errors(df: pd.DataFrame, metric: str = "energy") -> pd.DataFrame:
    own = df[df["source_variant"] == "own"]
    features = [c for c in own.columns if c.startswith("feature_")]
    aggregation = {metric: "mean", **{name: "first" for name in features}}
    return own.groupby(unit_keys(own) + ["method"], dropna=False).agg(aggregation).reset_index()


def log_ratio_table(df: pd.DataFrame, comparator: str, metric: str = "energy") -> pd.DataFrame:
    errors = unit_errors(df, metric)
    keys = unit_keys(errors)
    reference = errors[errors["method"] == comparator][keys + [metric]].rename(columns={metric: "comparator_error"})
    merged = errors[errors["method"] != comparator].merge(reference, on=keys, how="inner")
    valid = (merged[metric] > 0) & (merged["comparator_error"] > 0) & np.isfinite(merged[metric]) & np.isfinite(merged["comparator_error"])
    excluded = merged.loc[~valid].groupby("method").size().to_dict()
    merged = merged[valid].copy()
    merged["log_ratio"] = np.log(merged[metric] / merged["comparator_error"])
    merged.attrs["excluded_error_pairs"] = excluded
    return merged


def dataset_summary(lr: pd.DataFrame, *, min_units_for_own_sd: int = 3) -> pd.DataFrame:
    """Average repeated tasks within individual before estimating each dataset's SE.

    Training folds still overlap; these SEs are working variance estimates, not
    evidence that cross-validation folds are independent biological replicates.
    """
    keys = ["dataset_id", "representation", "kind", "family", "method"]
    if "clock" in lr.columns:
        keys += ["clock"]
    individual = lr.groupby(keys + ["individual"], dropna=False)["log_ratio"].mean().reset_index()
    grouped = individual.groupby(keys, dropna=False)["log_ratio"]
    summary = grouped.agg(n_individuals="count", mean="mean", sd="std").reset_index()
    counts = lr.groupby(keys, dropna=False)["log_ratio"].count().rename("n_units").reset_index()
    summary = summary.merge(counts, on=keys)
    summary["se_pooled"] = False
    for _, index in summary.groupby(keys[1:], dropna=False).groups.items():
        block = summary.loc[index]
        usable = block[block["n_individuals"] >= 2]
        pooled = (float(np.sqrt(np.sum((usable["n_individuals"] - 1) * usable["sd"] ** 2) / np.sum(usable["n_individuals"] - 1)))
                  if len(usable) and np.sum(usable["n_individuals"] - 1) > 0 else float("nan"))
        few = block.index[block["n_individuals"] < min_units_for_own_sd]
        summary.loc[few, "sd"] = pooled
        summary.loc[few, "se_pooled"] = True
    summary["se"] = summary["sd"] / np.sqrt(summary["n_individuals"])
    return summary


def random_effects_meta(y, se, *, max_iter: int = 500, tol: float = 1e-12) -> dict:
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    if y.ndim != 1 or se.shape != y.shape or np.any(se < 0):
        raise ValueError("estimates and nonnegative standard errors must be equal-length vectors")
    usable = np.isfinite(y)
    dropped = int((~usable).sum())
    y, se = y[usable], se[usable]
    k = int(len(y))
    metadata = {"k": k, "excluded_nonfinite_estimates": dropped}
    if k == 0:
        return {**metadata, "descriptive_only": True, "reason": "no finite estimates"}
    if k == 1:
        # One dataset cannot support dataset-level inference: report the estimate only (no interval, no p).
        return {**metadata, "mu": float(y[0]), "descriptive_only": True, "method": "single dataset (descriptive)"}
    if not np.all(np.isfinite(se) & (se > 0)):
        return {**metadata, "mu": float(y.mean()), "descriptive_only": True,
                "method": "unweighted dataset mean (descriptive)",
                "reason": "zero or unavailable SE; no variance floor or selective dataset exclusion",
                "datasets_without_positive_se": int((~(np.isfinite(se) & (se > 0))).sum())}
    variance = se ** 2
    weights = 1.0 / variance
    fixed = float(np.sum(weights * y) / np.sum(weights))
    q = float(np.sum(weights * (y - fixed) ** 2))
    # Profile restricted negative log likelihood, including the tau^2=0 boundary.
    def objective(tau2):
        v = variance + tau2
        w = 1.0 / v
        mu = np.dot(w, y) / w.sum()
        return 0.5 * (np.log(v).sum() + np.log(w.sum()) + np.dot(w, (y - mu) ** 2))

    upper = max(float(np.var(y, ddof=1)), float(variance.max())) * 16
    fit = optimize.minimize_scalar(objective, bounds=(0.0, upper), method="bounded",
                                   options={"xatol": tol, "maxiter": max_iter})
    if not fit.success or not np.isfinite(fit.fun):
        return {**metadata, "mu": float(y.mean()), "descriptive_only": True, "converged": False,
                "reason": "REML optimization did not converge"}
    tau2 = 0.0 if objective(0.0) <= fit.fun else float(fit.x)
    weights = 1.0 / (variance + tau2)
    mu = float(np.sum(weights * y) / np.sum(weights))
    se_reml = float(np.sqrt(1.0 / np.sum(weights)))
    q_hk = float(np.sum(weights * (y - mu) ** 2) / (k - 1))
    se = se_reml * float(np.sqrt(max(1.0, q_hk)))
    critical = float(stats.t.ppf(0.975, k - 1))
    return {**metadata, "mu": mu, "se": se, "se_reml": se_reml, "tau2": float(tau2),
            "converged": True, "iterations": int(fit.nfev), "df": k - 1,
            "i2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0,
            "ci": [mu - critical * se, mu + critical * se], "p": float(2 * stats.t.sf(abs(mu / se), k - 1)),
            "method": "REML with modified Hartung-Knapp-Sidik-Jonkman interval"}


def clopper_pearson(k: int, n: int, level: float = 0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    low = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    high = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return (low, high)


def sign_test(k: int, n: int) -> float:
    return float(stats.binomtest(int(k), int(n), 0.5).pvalue) if n > 0 else float("nan")


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.05) -> dict:
    table = table.dropna()
    n, k = table.shape
    if n < 2 or k < 3:
        return {"available": False, "n": int(n), "k": int(k)}
    ranks = table.rank(axis=1, method="average")
    if (table.nunique(axis=1) == 1).all():
        return {"available": False, "n": int(n), "k": int(k), "reason": "all methods tied within every dataset"}
    statistic, p = stats.friedmanchisquare(*[table[column].to_numpy() for column in table.columns])
    q = float(stats.studentized_range.ppf(1 - alpha, k, 1e6)) / np.sqrt(2.0)
    return {"available": True, "n": int(n), "k": int(k), "statistic": float(statistic), "p": float(p),
            "average_ranks": {str(m): float(v) for m, v in ranks.mean().sort_values().items()},
            "critical_difference": float(q * np.sqrt(k * (k + 1) / (6.0 * n)))}


def holm(pvalues: dict, *, family_size: int | None = None) -> dict:
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("Holm correction requires finite p-values in [0,1]")
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered) if family_size is None else family_size
    if total < len(ordered):
        raise ValueError("family size cannot be smaller than the observed tests")
    running, adjusted = 0.0, {}
    for position, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - position) * float(value)))
        adjusted[key] = running
    return adjusted


def classify(low: float, high: float, delta: float, noninferiority: float) -> str:
    """Log-ratio interval (negative favours the method) against zero, the practical margin and the NI margin."""
    if high < -delta:
        return "better_by_practical_margin"
    if high < 0:
        return "better_below_practical_margin"
    if low > 0:
        return "worse"
    if high < noninferiority:
        return "not_worse_than_margin"
    return "inconclusive"


def primary_analysis(df: pd.DataFrame, protocol: dict, metric: str | None = None) -> dict:
    analysis = protocol["analysis"]
    metric = metric or protocol["evaluation"]["primary_metric"]
    base = df[(df["source_variant"] == "own") & (df["family"].isin(PRIMARY_FAMILIES))
              & (df["representation"] == protocol["representation"]["primary"])]
    if "clock" in base.columns:
        base = base[base["clock"].fillna(protocol["clock"]["primary"]) == protocol["clock"]["primary"]]
    results, pvalues = {}, {}
    family = analysis["primary_family"]
    planned_family = [f"{kind}:{method}" for kind in family["tasks"] for method in family["methods"]]
    for kind, comparator in analysis["primary_comparators"].items():
        subset = base[base["kind"] == kind]
        if subset.empty or comparator not in set(subset["method"]):
            continue
        ratios = log_ratio_table(subset, comparator, metric)
        summary = dataset_summary(ratios)
        for method in sorted(set(summary["method"])):
            rows = summary[summary["method"] == method]
            meta = random_effects_meta(rows["mean"], rows["se"])
            excluded = int(ratios.attrs["excluded_error_pairs"].get(method, 0))
            if excluded:
                meta = {"k": int(len(rows)), "mu": float(rows["mean"].mean()), "descriptive_only": True,
                        "reason": "nonpositive/nonfinite errors prevent complete log-ratio inference"}
            better, worse = int((rows["mean"] < 0).sum()), int((rows["mean"] > 0).sum())
            entry = {"kind": kind, "comparator": comparator, "method": method, "metric": metric,
                     "datasets": int(len(rows)), "meta_k": int(meta.get("k", 0)),
                     "datasets_with_pooled_se": int(rows["se_pooled"].sum()), "meta": meta,
                     "excluded_error_pairs": excluded,
                     "datasets_method_better": better, "datasets_method_worse": worse,
                     "datasets_tied": int(len(rows)) - better - worse,
                     "win_rate_ci": clopper_pearson(better, better + worse), "sign_p": sign_test(better, better + worse)}
            if meta.get("ci"):
                entry["classification"] = classify(meta["ci"][0], meta["ci"][1], analysis["practical_delta_log"],
                                                   analysis["noninferiority_log"])
            else:
                entry["classification"] = "descriptive_only"
            key = f"{kind}:{method}"
            results[key] = entry
            if key in planned_family and "p" in meta:
                pvalues[key] = meta["p"]
    for key, value in holm(pvalues, family_size=len(planned_family)).items():
        results[key]["p_holm"] = value
    results["_holm_family"] = {"planned": planned_family, "tested": sorted(pvalues),
                               "untested": sorted(set(planned_family) - set(pvalues)),
                               "multiplicity": len(planned_family),
                               "intervals": "pointwise, not familywise simultaneous",
                               "classification": "pointwise interval description, not a Holm-adjusted claim"}
    return results


def metric_agreement(df: pd.DataFrame, comparator: str, metrics) -> dict:
    """Share of units in which each metric agrees with the primary metric on the sign of the log ratio."""
    tables = {name: log_ratio_table(df, comparator, name).set_index(unit_keys(df) + ["method"])["log_ratio"] for name in metrics}
    primary = list(metrics)[0]
    out = {}
    for method in sorted({key[-1] for key in tables[primary].index}):
        reference = tables[primary][tables[primary].index.get_level_values("method") == method]
        out[method] = {}
        for name in metrics:
            other = tables[name].reindex(reference.index)
            usable = other.notna()
            out[method][name] = float((np.sign(other[usable]) == np.sign(reference[usable])).mean()) if usable.any() else None
    return out


def source_information(df: pd.DataFrame, metric: str = "energy") -> dict:
    """Own versus substituted source within leave-two-out fits; the held individual is the unit."""
    swap = df[df["family"] == "source_swap"]
    if swap.empty:
        return {}
    keys = ["job_id", "pair_index"]
    own = swap[swap["source_variant"] == "own"].set_index(keys)[metric].rename("own")
    substitute = swap[swap["source_variant"].str.startswith("substitute:")].set_index(keys)[metric].rename("substitute")
    labels = swap.drop_duplicates(keys).set_index(keys)[["dataset_id", "representation", "method", "individual"]]
    joined = pd.concat([own, substitute], axis=1).dropna().join(labels)
    joined = joined[(joined["own"] > 0) & (joined["substitute"] > 0)]
    joined["log_ratio"] = np.log(joined["own"] / joined["substitute"])
    per_individual = joined.groupby(["dataset_id", "representation", "method", "individual"])["log_ratio"].mean()
    out, pooled = {}, {}
    for (dataset, representation, method), values in per_individual.groupby(level=[0, 1, 2]):
        array = values.to_numpy()
        better, worse = int((array < 0).sum()), int((array > 0).sum())
        entry = {"individuals": int(len(array)), "own_better": better, "own_worse": worse,
                 "sign_p": sign_test(better, better + worse), "median_log_ratio": float(np.median(array))}
        if len(array) >= 6 and np.any(array != 0):
            entry["wilcoxon_p"] = float(stats.wilcoxon(array).pvalue)
        out[f"{dataset}:{representation}:{method}"] = entry
        pooled.setdefault((representation, method), []).append(
            (float(np.mean(array)), float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else float("nan")))
    for (representation, method), items in pooled.items():
        out[f"pooled:{representation}:{method}"] = random_effects_meta([m for m, _ in items], [s for _, s in items])
    return out


def clock_sensitivity(df: pd.DataFrame, primary: str, metric: str = "energy") -> dict:
    """Paired log(error at alternative clock / error at primary clock) per unit, method and clock."""
    if "clock" not in df.columns:
        return {}
    own = df[(df["source_variant"] == "own") & (df["family"].isin(PRIMARY_FAMILIES))].copy()
    own["clock"] = own["clock"].fillna(primary)
    errors = own.groupby(UNIT + ["method", "clock"], dropna=False)[metric].mean().reset_index()
    reference = errors[errors["clock"] == primary].drop(columns="clock").rename(columns={metric: "primary_error"})
    out = {}
    for clock in sorted(set(errors["clock"]) - {primary}):
        other = errors[errors["clock"] == clock].drop(columns="clock")
        merged = other.merge(reference, on=UNIT + ["method"], how="inner")
        merged = merged[(merged[metric] > 0) & (merged["primary_error"] > 0)]
        merged["log_ratio"] = np.log(merged[metric] / merged["primary_error"])
        for (representation, method), rows in merged.groupby(["representation", "method"]):
            by_dataset = rows.groupby("dataset_id")["log_ratio"].mean()
            out[f"{representation}:{clock}:{method}"] = {"units": int(len(rows)), "datasets": int(len(by_dataset)),
                                        "median_log_ratio": float(rows["log_ratio"].median()),
                                        "datasets_alternative_better": int((by_dataset < 0).sum()),
                                        "dataset_means": {str(k): float(v) for k, v in by_dataset.items()}}
    return out


def count_curve_analysis(df: pd.DataFrame, metric: str = "energy", *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Conditional resampling diagnostic, never equivalence or a required donor count.

    Match evaluation pairs/repeats/seeds across k; resample held individuals and
    paired within-individual seed/repeat cells. Overlapping training folds remain
    a limitation. Missing design evidence blocks any plateau designation.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    curve = df[(df["family"] == "count_curve") & (df["source_variant"] == "own")].copy()
    out = {}
    rng = np.random.default_rng(seed)
    groups = ["dataset_id", "representation"] + (["clock"] if "clock" in curve.columns else []) + ["method"]
    for identity, block in curve.groupby(groups, dropna=False):
        table = block.groupby(["held_individual", "training_size"])[metric].mean().unstack()
        sizes = sorted(int(k) for k in table.columns)
        issues = []
        if len(sizes) < 3:
            issues.append("fewer than three training sizes")
        required = ["seed", "repeat", "individual", "source_time", "target_time",
                    "stochastic", "budget_hash", "evaluation_hash", "training_individuals"]
        missing = [name for name in required if name not in block or block[name].isna().any()]
        if missing:
            issues.append(f"missing matched-design metadata: {missing}")
        elif sizes:
            eval_keys = ["held_individual", "repeat", "seed", "individual", "source_time", "target_time"]
            if block.duplicated(["training_size", *eval_keys]).any():
                issues.append("duplicate evaluation cells")
            identities = [set(block.loc[block.training_size == k, eval_keys].itertuples(index=False, name=None)) for k in sizes]
            if any(items != identities[0] for items in identities[1:]):
                issues.append("evaluation pairs/repeats/seeds differ across k")
            for name in ("budget_hash", "evaluation_hash", "stochastic"):
                if block[name].nunique() != 1:
                    issues.append(f"unmatched {name}")
            if block["stochastic"].any():
                seeds = block.groupby(["held_individual", "repeat", "training_size"])["seed"].nunique()
                if (seeds < 3).any():
                    issues.append("stochastic method has fewer than three training seeds per k cell")
            for (held, _), repeated in block.groupby(["held_individual", "repeat"]):
                previous = set()
                for k, cell in repeated.groupby("training_size", sort=True):
                    subsets = {tuple(sorted(v)) for v in cell["training_individuals"]}
                    if len(subsets) != 1:
                        issues.append("training subset differs within a k cell")
                        break
                    subset = set(next(iter(subsets)))
                    if len(subset) != k or held in subset or not previous.issubset(subset):
                        issues.append("training subsets are not nested held-out sets of size k")
                        break
                    previous = subset
        issues = sorted(set(issues))
        contract_ok = not issues
        steps = []
        for k, nxt in zip(sizes[:-1], sizes[1:]):
            paired = (table[k] - table[nxt]).dropna().to_numpy()
            if len(paired) < 2:
                continue
            spread = None
            if contract_ok:
                cell = block.groupby(["held_individual", "repeat", "seed", "training_size"])[metric].mean().unstack()
                delta = (cell[k] - cell[nxt]).dropna()
                by_individual = [v.to_numpy() for _, v in delta.groupby(level="held_individual")]
                draws = []
                for _ in range(n_boot):
                    sampled = rng.integers(0, len(by_individual), size=len(by_individual))
                    draws.append(np.mean([rng.choice(by_individual[i], size=len(by_individual[i]), replace=True).mean()
                                          for i in sampled]))
                within = (block[block.training_size.isin([k, nxt])]
                          .groupby(["held_individual", "repeat", "training_size", "seed"])[metric].mean()
                          .groupby(level=["held_individual", "repeat", "training_size"]).std(ddof=1))
                spread = float(within.mean()) if within.notna().any() else 0.0
            else:
                draws = paired[rng.integers(0, len(paired), size=(n_boot, len(paired)))].mean(axis=1)
            low, high = np.percentile(draws, [2.5, 97.5])
            steps.append({"from_k": k, "to_k": nxt, "unit_step": nxt - k == 1, "individuals": int(len(paired)),
                          "mean_improvement": float(paired.mean()), "interval": [float(low), float(high)],
                          "contains_zero": bool(low <= 0 <= high), "mean_within_cell_seed_sd": spread,
                          "unresolved_at_seed_scale": bool(spread is not None and abs(paired.mean()) < spread)})
        plateau = None
        for first, second in zip(steps[:-1], steps[1:]):
            if (contract_ok and first["unit_step"] and second["unit_step"] and first["to_k"] == second["from_k"]
                    and first["contains_zero"] and second["contains_zero"]
                    and not first["unresolved_at_seed_scale"] and not second["unresolved_at_seed_scale"]):
                plateau = first["from_k"]
                break
        out[":".join(map(str, identity))] = {"sizes": sizes, "steps": steps, "plateau_k": plateau,
            "matched_contract_passed": contract_ok, "contract_issues": issues,
            "interpretation": "conditional diagnostic only; not equivalence or a required donor count; training folds overlap",
            "representation_scope": "representation fitted on all non-held individuals; k counts dynamics-training individuals only",
            "mean_error_by_k": {str(k): float(table[k].mean()) for k in sizes}}
    return out
