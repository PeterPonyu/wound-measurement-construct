"""What statistical power means for each claim: exact sign tests, t-test detectable effects,
random-effects dataset requirements and pilot variance components."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def min_sign_p(n: int) -> float:
    return min(1.0, 2.0 * 0.5 ** int(n))


def sign_test_critical(n: int, alpha: float = 0.05):
    for k in range(int(n) // 2 + 1, int(n) + 1):
        if stats.binomtest(k, int(n), 0.5).pvalue <= alpha:
            return k
    return None


def sign_test_power(n: int, p_true: float, alpha: float = 0.05) -> float:
    critical = sign_test_critical(n, alpha)
    if critical is None:
        return 0.0
    upper = 1.0 - stats.binom.cdf(critical - 1, n, p_true)
    lower = stats.binom.cdf(n - critical, n, p_true)
    return float(upper + lower)


def t_test_mde(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    from statsmodels.stats.power import TTestIndPower

    return float(TTestIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=alpha, power=power))


def datasets_needed(delta: float, tau: float, sigma: float, tasks_per_dataset: int, alpha: float = 0.05,
                    power: float = 0.8) -> int:
    """Prospective normal approximation; delta is an absolute log error ratio."""
    if (not np.all(np.isfinite([delta, tau, sigma, tasks_per_dataset, alpha, power])) or delta <= 0
            or tau < 0 or sigma < 0 or tasks_per_dataset < 1 or int(tasks_per_dataset) != tasks_per_dataset
            or not 0 < alpha < 1 or not 0.5 < power < 1):
        raise ValueError("invalid prospective design parameters")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return max(2, int(math.ceil(z ** 2 * (tau ** 2 + sigma ** 2 / tasks_per_dataset) / delta ** 2)))


def variance_components(lr) -> dict:
    """Between-dataset (tau^2) and within-dataset (sigma^2) variance of pilot log ratios."""
    from .inference import random_effects_meta

    # Multiple time pairs/repeats from one individual are not new design units.
    if "individual" in lr.columns:
        lr = lr.groupby(["dataset_id", "individual"], dropna=False)["log_ratio"].mean().reset_index()
    grouped = lr.groupby("dataset_id")["log_ratio"]
    means, variances, sizes = grouped.mean(), grouped.var(ddof=1), grouped.count()
    usable = sizes > 1
    sigma2 = float(np.average(variances[usable], weights=sizes[usable] - 1)) if usable.any() else float("nan")
    meta = random_effects_meta(means, np.sqrt(sigma2 / sizes))
    return {"datasets": int(len(means)), "sigma2": sigma2, "tau2": float(meta.get("tau2", float("nan"))),
            "mean_tasks_per_dataset": float(sizes.mean()),
            "within_dataset_unit": "individual-averaged log ratio",
            "available": "tau2" in meta and np.isfinite(sigma2)}


def power_report(pilot=None) -> dict:
    report = {
        "sign_test": [{"n": n, "min_p": min_sign_p(n), "critical_wins": sign_test_critical(n),
                       "power": {str(p): sign_test_power(n, p) for p in (0.65, 0.75, 0.85)}}
                      for n in (3, 4, 5, 6, 8, 10, 12, 15, 19, 20, 30, 50, 100)],
        "t_test_mde": [{"n1": a, "n2": b, "mde_d": t_test_mde(a, b)}
                       for a, b in ((7, 4), (14, 8), (20, 20), (30, 30), (50, 50), (100, 100))],
        "exact_permutation_7_vs_4_min_p": 2 / math.comb(11, 7),
        "datasets_needed": [{"tau": tau, "sigma": 0.15, "tasks_per_dataset": t, "delta": d,
                             "error_ratio_factor": float(np.exp(d)), "reduction_if_better": float(1 - np.exp(-d)),
                             "datasets": datasets_needed(d, tau, 0.15, t)}
                            for tau in (0.1, 0.2, 0.3) for t in (3, 10) for d in (0.05, 0.1, 0.2)],
        "interpretation": ("The inferential unit determines power: datasets for method comparisons, individuals for "
                           "source-information and count-curve claims, patients for prognosis. Cells, seeds and "
                           "resampling draws improve measurement precision but add no units. Dataset requirements are "
                           "prospective normal approximations, not observed power or guaranteed sample sizes. Delta is a "
                           "log ratio of squared energy statistics: log(1.2) means a 1.2-fold ratio, or 16.7% reduction "
                           "in the favourable direction (8.7% in its square root), not a 20% reduction. Overlapping "
                           "training folds limit the working variance approximation; lack of significance is not equivalence."),
    }
    if pilot is not None:
        components = variance_components(pilot)
        report["pilot_components"] = components
        if not components["available"]:
            report["pilot_design_status"] = "unavailable: pilot cannot estimate finite between/within-dataset variance"
            return report
        tau = math.sqrt(max(components["tau2"], 0.0))
        sigma = math.sqrt(max(components["sigma2"], 0.0))
        tasks = max(int(round(components["mean_tasks_per_dataset"])), 1)
        report["pilot_datasets_needed"] = {str(d): datasets_needed(d, tau, sigma, tasks) for d in (0.1, 0.18, 0.3)}
    return report
