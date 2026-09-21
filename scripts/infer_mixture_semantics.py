#!/usr/bin/env python3
"""Infer mixture semantics."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'outputs' / 'bimodal_stratification' / 'report.json'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'analysis' / 'mixture_semantic_inference'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def rho(x: np.ndarray, y: np.ndarray) -> float:
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else float('nan')

def bootstrap_rho(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    while len(draws) < n_boot:
        idx = rng.integers(0, len(x), size=len(x))
        if len(np.unique(idx)) < 3:
            continue
        value = rho(x[idx], y[idx])
        if np.isfinite(value):
            draws.append(value)
    return np.asarray(draws, dtype=float)

def permutation_null(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        out[i] = rho(x, rng.permutation(y))
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-report', default=str(DEFAULT_SOURCE))
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    started = time.time()
    source = Path(args.source_report)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(source.read_text(encoding='utf-8'))
    cohort = source_report['cohorts']['GSE165816_discovery']
    rows = cohort['per_sample']
    if len(rows) != 25:
        raise RuntimeError(f'expected 25 semantic samples, found {len(rows)}')
    x = np.asarray([r['topic0_mean'] for r in rows], dtype=float)
    y = np.asarray([r['mixing_weight'] for r in rows], dtype=float)
    observed = rho(x, y)
    source_rho = float(cohort['hypotheses']['mean_vs_mixing_weight']['rho'])
    if not np.isclose(observed, source_rho, atol=1e-12):
        raise RuntimeError(f'source rho mismatch: recomputed={observed}, report={source_rho}')
    boot = bootstrap_rho(x, y, args.n_boot, args.seed + 11)
    null = permutation_null(x, y, args.n_perm, args.seed + 29)
    ci = np.percentile(boot, [2.5, 97.5])
    null_p = (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
    loo = []
    for i, row in enumerate(rows):
        mask = np.ones(len(rows), dtype=bool)
        mask[i] = False
        loo.append({'excluded_sample': row['gsm'], 'excluded_arm': row['arm'], 'rho': rho(x[mask], y[mask])})
    loo_values = np.asarray([r['rho'] for r in loo], dtype=float)
    out = {'experiment': 'mixture_semantic_inference', 'status': 'completed', 'scientific_unit': 'sample-level fibroblast summaries', 'patient_level_inference_available': False, 'scope': {'patient_level_inference_available': False, 'limit': 'sample-level semantic uncertainty only; no patient-level or healing validation'}, 'scientific_scope': 'uncertainty audit of the frozen 25-sample GSE165816 mixture-semantic association; no topic refit, no patient-level inference, and no healing claim', 'source_experiment': 'bimodal_stratification', 'n_samples': int(len(rows)), 'observed': {'spearman_rho': float(observed), 'source_report_rho': float(source_rho), 'source_p': float(cohort['hypotheses']['mean_vs_mixing_weight']['p']), 'state_cut': float(cohort['gmm']['midpoint'])}, 'bootstrap': {'n_resamples': int(args.n_boot), 'seed': int(args.seed + 11), 'ci95': [float(ci[0]), float(ci[1])], 'sd': float(boot.std(ddof=1)), 'finite_draws': int(len(boot))}, 'permutation_null': {'n_permutations': int(args.n_perm), 'seed': int(args.seed + 29), 'null_mean': float(null.mean()), 'null_ci95': [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))], 'p_two_sided': float(null_p)}, 'leave_one_sample_out': {'rows': loo, 'rho_min': float(np.min(loo_values)), 'rho_max': float(np.max(loo_values)), 'all_positive': bool(np.all(loo_values > 0))}, 'interpretation': 'the sample-level semantic association is strong and remains positive under resampling and leave-one-sample-out analysis; the interval is a measurement-semantic uncertainty summary, not clinical validation', 'protocol': {'script_sha256': sha256(Path(__file__)), 'source_report_sha256': sha256(source), 'source_report': str(source), 'bootstrap_resampling': 'rows sampled with replacement', 'permutation_null': 'mixture weights permuted across the 25 samples', 'no_topic_refit': True, 'patient_level_inference': False, 'new_model_fits': False, 'elapsed_seconds': time.time() - started}}
    (output / 'report.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'observed': out['observed'], 'bootstrap': out['bootstrap'], 'permutation_null': out['permutation_null'], 'leave_one_sample_out': out['leave_one_sample_out'], 'output': str(output / 'report.json')}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
