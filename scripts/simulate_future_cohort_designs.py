#!/usr/bin/env python3
"""Simulate future cohort designs."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def design_effect(samples_per_patient: float, icc: float) -> float:
    """Variance inflation from repeated samples within one patient."""
    return 1.0 + (samples_per_patient - 1.0) * icc

def welch_power(n1: int, n2: int, diff: float, sd1: float, sd2: float, alpha: float, n_sim: int, rng: np.random.Generator) -> float:
    """Simulated two-sided Welch power under unequal arm variances."""
    if n1 < 2 or n2 < 2:
        return 0.0
    a = rng.normal(diff, sd1, size=(n_sim, n1))
    b = rng.normal(0.0, sd2, size=(n_sim, n2))
    res = stats.ttest_ind(a, b, axis=1, equal_var=False)
    return float(np.mean(res.pvalue < alpha))

def tost_power(n1: int, n2: int, true_diff: float, margin: float, sd1: float, sd2: float, alpha: float, n_sim: int, rng: np.random.Generator) -> float:
    """Simulated power to declare equivalence within +/- margin."""
    if n1 < 2 or n2 < 2:
        return 0.0
    a = rng.normal(true_diff, sd1, size=(n_sim, n1))
    b = rng.normal(0.0, sd2, size=(n_sim, n2))
    ma, mb = (a.mean(axis=1), b.mean(axis=1))
    va, vb = (a.var(axis=1, ddof=1), b.var(axis=1, ddof=1))
    se = np.sqrt(va / n1 + vb / n2)
    se = np.where(se == 0, np.finfo(float).tiny, se)
    df = (va / n1 + vb / n2) ** 2 / ((va / n1) ** 2 / (n1 - 1) + (vb / n2) ** 2 / (n2 - 1))
    diff = ma - mb
    p_lower = 1.0 - stats.t.cdf((diff + margin) / se, df)
    p_upper = stats.t.cdf((diff - margin) / se, df)
    return float(np.mean(np.maximum(p_lower, p_upper) < alpha))

def _delta_for_auc(auc: float) -> float:
    """Binormal mean separation giving the requested AUC."""
    return math.sqrt(2.0) * stats.norm.ppf(min(max(auc, 1e-06), 1 - 1e-06))

def _auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC per simulation row, via ranks rather than pairs."""
    n_pos, n_neg = (pos.shape[1], neg.shape[1])
    both = np.concatenate([pos, neg], axis=1)
    order = np.argsort(both, axis=1, kind='stable')
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, both.shape[1] + 1, dtype=np.float64), both.shape), axis=1)
    rank_sum = ranks[:, :n_pos].sum(axis=1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def auc_difference_power(n_pos: int, n_neg: int, auc_a: float, auc_b: float, score_correlation: float, n_sim: int, rng: np.random.Generator) -> float:
    """Power for a paired interval on the AUC difference to exclude zero.

    The two representations score the same samples, so their errors are
    correlated. The correlation is imposed on the noise only; mixing the
    full scores would also mix the means and destroy the intended AUC gap.
    """
    if n_pos < 2 or n_neg < 2:
        return 0.0
    rho = float(np.clip(score_correlation, -0.999, 0.999))
    mix = math.sqrt(max(1.0 - rho * rho, 0.0))
    z1 = rng.standard_normal((n_sim, n_pos))
    z2 = rng.standard_normal((n_sim, n_pos))
    z3 = rng.standard_normal((n_sim, n_neg))
    z4 = rng.standard_normal((n_sim, n_neg))
    pos_a = _delta_for_auc(auc_a) + z1
    pos_b = _delta_for_auc(auc_b) + rho * z1 + mix * z2
    neg_a = z3
    neg_b = rho * z3 + mix * z4
    d = _auc_from_scores(pos_a, neg_a) - _auc_from_scores(pos_b, neg_b)
    se = d.std(ddof=1)
    if se == 0:
        return 0.0
    return float(np.mean(np.abs(d) - 1.96 * se > 0))

def smallest_n(fn, target: float, n_max: int, healer_fraction: float) -> dict:
    """Smallest total sample count reaching the target power.

    Doubling search then bisection; power is monotone in n up to simulation
    error, so a full integer scan only costs time.
    """

    def evaluate(total: int):
        n1 = max(int(round(total * healer_fraction)), 2)
        n2 = max(total - n1, 2)
        return (n1, n2, fn(n1, n2))
    low, total = (6, 6)
    while total <= n_max:
        n1, n2, power = evaluate(total)
        if power >= target:
            break
        low = total
        total = total * 2 if total * 2 <= n_max else n_max if total < n_max else n_max + 1
    if total > n_max:
        return {'total': None, 'n_healed': None, 'n_not_healed': None, 'power': None, 'note': f'not reached by n={n_max}'}
    high = total
    while high - low > 1:
        mid = (low + high) // 2
        _, _, power = evaluate(mid)
        if power >= target:
            high = mid
        else:
            low = mid
    n1, n2, power = evaluate(high)
    return {'total': high, 'n_healed': n1, 'n_not_healed': n2, 'power': power}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default='outputs/representation_benchmark/sample_scores.csv')
    ap.add_argument('--benchmark', default='outputs/representation_benchmark/report.json')
    ap.add_argument('--output-dir', default='outputs/analysis/design_power_simulation')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--target-power', type=float, default=0.8)
    ap.add_argument('--n-sim', type=int, default=4000)
    ap.add_argument('--n-max', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    scores = pd.read_csv(args.scores)
    healer = scores.loc[scores['y_healer'] == 1, 'topic_simplex_theta0'].to_numpy()
    nonhealer = scores.loc[scores['y_healer'] == 0, 'topic_simplex_theta0'].to_numpy()
    sd_healed = float(np.std(healer, ddof=1))
    sd_not_healed = float(np.std(nonhealer, ddof=1))
    observed_diff = float(healer.mean() - nonhealer.mean())
    healer_fraction = float(len(healer) / len(scores))
    benchmark = json.loads(Path(args.benchmark).read_text())
    cross = benchmark['cross_method_spearman_representative']
    score_correlation = float(np.median(list(cross.values())))
    anchors = {'source': 'current discovery samples; used only to set spread and balance', 'n_samples': int(len(scores)), 'healer_fraction': healer_fraction, 'sd_topic0_healed': sd_healed, 'sd_topic0_not_healed': sd_not_healed, 'variance_ratio_healed_over_not_healed': sd_healed / sd_not_healed, 'observed_sample_level_difference': observed_diff, 'median_between_representation_rank_correlation': score_correlation, 'pooled_sd_would_be': float(np.sqrt(((len(healer) - 1) * sd_healed ** 2 + (len(nonhealer) - 1) * sd_not_healed ** 2) / (len(healer) + len(nonhealer) - 2)))}
    log(f"anchors: sd_healed={sd_healed:.4f} sd_not_healed={sd_not_healed:.4f} ratio={anchors['variance_ratio_healed_over_not_healed']:.2f} healer_fraction={healer_fraction:.3f} score_corr={score_correlation:.3f}")
    detect_targets = [0.02, 0.03, round(abs(observed_diff), 4), 0.06, 0.08, 0.1]
    detect = {}
    for diff in sorted(set(detect_targets)):
        res = smallest_n(lambda n1, n2, d=diff: welch_power(n1, n2, d, sd_healed, sd_not_healed, args.alpha, args.n_sim, rng), args.target_power, args.n_max, healer_fraction)
        detect[f'difference_{diff:g}'] = res
        log(f"  detect {diff:g}: n={res['total']}")
    equivalence = {}
    for margin in [0.02, 0.03, 0.05, 0.075, 0.1]:
        res = smallest_n(lambda n1, n2, m=margin: tost_power(n1, n2, 0.0, m, sd_healed, sd_not_healed, args.alpha, args.n_sim, rng), args.target_power, args.n_max, healer_fraction)
        equivalence[f'margin_{margin:g}'] = res
        log(f"  equivalence at {margin:g}: n={res['total']}")
    auc_resolution = {}
    for gap in [0.05, 0.1, 0.15, 0.2]:
        res = smallest_n(lambda n1, n2, g=gap: auc_difference_power(n1, n2, 0.8 + g / 2, 0.8 - g / 2, score_correlation, args.n_sim, rng), args.target_power, args.n_max, healer_fraction)
        auc_resolution[f'auc_gap_{gap:g}'] = res
        log(f"  resolve AUC gap {gap:g}: n={res['total']}")

    def to_patients(total: int | None, samples_per_patient: float, icc: float, missing: float) -> int | None:
        if total is None:
            return None
        effective = total * design_effect(samples_per_patient, icc)
        patients = effective / samples_per_patient
        return int(math.ceil(patients / max(1.0 - missing, 1e-06)))
    grid = []
    headline_keys = {'detect_observed_difference': detect[f'difference_{round(abs(observed_diff), 4):g}']['total'], 'equivalence_margin_0.05': equivalence['margin_0.05']['total'], 'resolve_auc_gap_0.1': auc_resolution['auc_gap_0.1']['total']}
    for spp, icc, missing in itertools.product([1.0, 2.0, 3.0], [0.0, 0.3, 0.6], [0.0, 0.15]):
        entry = {'samples_per_patient': spp, 'within_patient_icc': icc, 'outcome_missing_fraction': missing, 'design_effect': design_effect(spp, icc)}
        for label, total in headline_keys.items():
            entry[f'patients_for_{label}'] = to_patients(total, spp, icc, missing)
        grid.append(entry)
    report = {'experiment': 'next_cohort_design_power_simulation', 'status': 'completed', 'scientific_unit': 'patient', 'patient_level_inference_available': False, 'is_a_design_calculation_not_a_result': True, 'question': 'How many independent patients does the next cohort need to resolve the comparisons the current cohort cannot?', 'anchors': anchors, 'design_a_detect_difference': detect, 'design_b_declare_equivalence': equivalence, 'design_c_resolve_auc_difference': auc_resolution, 'patient_translation_grid': grid, 'assumptions': ['arm-specific normal spread taken from the current discovery samples', 'unequal variances retained; a pooled SD would understate n', 'candidate margins and AUC gaps are inputs, not findings', 'repeated samples inflate variance through the design effect', 'no interim analyses and no multiplicity across the three designs'], 'not_supported': ['any of these margins as a clinical minimal important difference', 'a claim that the current cohort was adequately powered', 'reuse of these numbers as evidence about wound biology'], 'protocol': {'alpha': args.alpha, 'target_power': args.target_power, 'n_sim_per_point': args.n_sim, 'n_max_searched': args.n_max, 'seed': args.seed, 'scores_sha256': _sha256(args.scores), 'benchmark_sha256': _sha256(args.benchmark), 'script_sha256': _sha256(__file__), 'elapsed_seconds': time.time() - t0}}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    log(f'wrote {out}')
    print(json.dumps({'anchors': anchors, 'detect': detect, 'equivalence': equivalence, 'auc_resolution': auc_resolution}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
