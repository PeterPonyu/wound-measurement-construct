#!/usr/bin/env python3
"""Post-hoc saved-model diagnostics: training-only nulls, usage and redundancy.

No model is fitted and no experiment artifacts are changed. This is not an
outcome analysis or a model-selection procedure.
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expansion.common.io import atomic_save_npz, atomic_write_json, read_json, sha256_file
from expansion.measurement import stability
from expansion.runner.jobs import read_jobs
from expansion.runner.queue import job_dir, verify_done


def training_nulls(proportions, units, training_mask, smoothing=1e-6):
    x = stability._proportions(proportions)
    units, mask = np.asarray(units, dtype=str), np.asarray(training_mask, dtype=bool)
    if units.shape != mask.shape or mask.shape != (x.shape[0],) or not mask.any() or not 0 < smoothing < 1:
        raise ValueError('invalid training-only null inputs')
    selected = x[mask]
    cell_mean = np.asarray(selected.mean(axis=0), dtype=float).ravel()
    patient_means = [np.asarray(x[mask & (units == patient)].mean(axis=0), dtype=float).ravel()
                     for patient in sorted(set(units[mask]))]
    return {name: (1 - smoothing) * value + smoothing / x.shape[1]
            for name, value in (('training_cell_mean', cell_mean),
                                ('training_equal_patient_mean', np.mean(patient_means, axis=0)))}


def fixed_distribution_crossentropy(proportions, probabilities):
    x, q = stability._proportions(proportions), np.asarray(probabilities, dtype=float)
    if q.shape != (x.shape[1],) or not np.isfinite(q).all() or (q < 0).any() or not np.isclose(q.sum(), 1, atol=1e-5):
        raise ValueError('invalid fixed gene probability distribution')
    return -np.asarray(x @ np.log(q + 1e-10)).ravel()


def model_crossentropy(proportions, theta, beta, chunk=128):
    x, theta, beta = stability._proportions(proportions), np.asarray(theta, dtype=float), np.asarray(beta, dtype=float)
    if (theta.shape != (x.shape[0], beta.shape[0]) or beta.ndim != 2 or beta.shape[1] != x.shape[1]
            or not all(np.isfinite(a).all() and (a >= 0).all() and np.allclose(a.sum(1), 1, atol=1e-5)
                       for a in (theta, beta))):
        raise ValueError('model simplex arrays do not align')
    scores = []
    for start in range(0, x.shape[0], chunk):
        stop = start + chunk
        probability = theta[start:stop] @ beta
        scores.extend(-np.asarray(x[start:stop].multiply(np.log(probability + 1e-10)).sum(1)).ravel())
    return np.asarray(scores)


def audit(root, out):
    started = time.perf_counter()
    root, out = Path(root).resolve(), Path(out).resolve()
    if root not in out.parents or out.exists():
        raise ValueError('audit output must be a fresh experiment-root child')
    run = root / 'runs/topic'
    protocol = read_json(run / 'protocol.json')
    x, cells = stability.load_discovery_panel_counts(ROOT)
    mask, selected, partition = stability.patient_partition(cells, protocol['stability']['patient_partition'])
    baseline = training_nulls(x, cells.patient_id.to_numpy(dtype=str), mask)
    panel = np.load(ROOT / stability.DISCOVERY / 'panel.npy', allow_pickle=True)
    rows, traces, matrices = [], {}, {}
    for job in read_jobs(run / 'jobs.jsonl'):
        folder = job_dir(run, job.job_id)
        if not verify_done(folder, job):
            raise ValueError('audit encountered incomplete model')
        loaded = stability.load_inference_checkpoint(folder, panel=panel,
                         cell_ids=cells.reconstructed_raw_cell_id.to_numpy())
        result = read_json(folder / 'result.json')
        traces[(job.seed, job.params['steps'])] = result['training_trace']
        beta, theta = loaded['beta'], loaded['theta']
        sqrt_beta = np.sqrt(beta.astype(float))
        cosine = sqrt_beta @ sqrt_beta.T
        off = cosine[np.triu_indices(len(beta), 1)]
        matrices[job.job_id] = cosine
        partition_scores = {}
        for name, indices in selected.items():
            matrix, weights = x[indices], theta[indices].astype(float)
            scores = {'model': model_crossentropy(matrix, weights, beta),
                      'uniform_decoder_mixture': fixed_distribution_crossentropy(matrix, beta.astype(float).mean(0)),
                      **{key: fixed_distribution_crossentropy(matrix, q) for key, q in baseline.items()}}
            units = cells.iloc[indices].patient_id.to_numpy(dtype=str)
            by_patient = []
            for patient in sorted(set(units)):
                values = {key: float(score[units == patient].mean()) for key, score in scores.items()}
                by_patient.append({'patient_id': patient, **values,
                    'model_minus_null': {key: values['model'] - v for key, v in values.items() if key != 'model'}})
            mean_theta = weights.mean(0)
            mean_entropy = float(-np.sum(mean_theta * np.log(mean_theta + 1e-10)))
            cell_entropy = float(np.mean(-np.sum(weights * np.log(weights + 1e-10), axis=1)))
            partition_scores[name] = {'by_patient': by_patient,
                'equal_patient_crossentropy': {key: float(np.mean([p[key] for p in by_patient])) for key in scores},
                'mean_usage_entropy': mean_entropy, 'mean_cell_theta_entropy': cell_entropy,
                'entropy_gap_mean_usage_minus_cell': mean_entropy - cell_entropy,
                'effective_mean_usage_topics': float(np.exp(mean_entropy)),
                'dominant_mean_loading': float(mean_theta.max()),
                'per_topic_theta_std': weights.std(0).tolist(),
                'argmax_nonempty_topics': int(np.count_nonzero(np.bincount(weights.argmax(1), minlength=job.params['K'])))}
        rows.append({'job_id': job.job_id, 'seed': job.seed, 'steps': job.params['steps'],
            'decoder_sqrt_cosine_offdiagonal': {'min': float(off.min()), 'median': float(np.median(off)),
                                               'max': float(off.max())}, 'partitions': partition_scores})
    prefixes = []
    for seed in (0, 1):
        for lower, upper in ((500, 2000), (500, 5000), (2000, 5000)):
            a, b = traces[(seed, lower)], traces[(seed, upper)][:lower]
            error = max(abs(x[key] - y[key]) for x, y in zip(a, b) for key in ('loss', 'recon_loss', 'kl_div'))
            prefixes.append({'seed': seed, 'lower': lower, 'upper': upper, 'max_training_trace_abs_error': error,
                             'identical': error == 0.})
    out.mkdir(parents=True, exist_ok=False)
    atomic_save_npz(out / 'training_null_probabilities.npz', **baseline)
    atomic_save_npz(out / 'decoder_similarity.npz', **matrices)
    report = {'scope': 'post_hoc_descriptive_saved_model_diagnostic_not_selection_or_clinical_validation',
        'spec_sha256': sha256_file(ROOT / '.omx/plans/2026-09-29-saved-topic-null-audit-spec.md'),
        'audit_script_sha256': sha256_file(__file__), 'smoothing_uniform_gene_mass': 1e-6,
        'crossentropy_log_offset': 1e-10, 'partition': partition,
        'fits': sorted(rows, key=lambda r: (r['steps'], r['seed'])), 'training_prefix_checks': prefixes,
        'seconds': time.perf_counter() - started,
        'rss_high_water_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'interpretation': 'Lower reconstruction does not ensure stable biological topics. Near-uniform theta makes argmax usage ambiguous; '
                          'decoder similarity and usage are descriptive, not a stand-alone collapse test.'}
    atomic_write_json(out / 'audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = audit(args.experiment_root, args.out)
    print({'fits': len(result['fits']), 'all_prefixes_identical': all(r['identical'] for r in result['training_prefix_checks'])})


if __name__ == '__main__':
    main()
