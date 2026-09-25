#!/usr/bin/env python3
"""RT05 repair: one fitted scoring function for each held healed/nonhealed pair.

The fixed discovery panel and topic encoder are NOT externally validated.
Only the new fold preprocessing, component selection and orientation exclude
both held patients. The 28 pair credits are dependent descriptive observations.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import warnings
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.optimize import nnls
import sklearn
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / 'scripts/benchmark_expression_representations.py'
SPEC = importlib.util.spec_from_file_location('pair_benchmark24', BENCH_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
OUTPUT = ROOT / 'outputs/scientific_revision_20260923/patient_pair_discrimination'
CACHE = ROOT / 'outputs/scientific_revision_20260922/measurement_controls'
PROJECTION = ROOT / 'outputs/scientific_revision_20260922/measurement/projection'
LEGACY = ROOT / 'outputs/scientific_revision_20260922/measurement/patient_unit_representation_benchmark/report.json'
METHODS = ('topic_simplex_theta0', 'module_score', 'pca', 'nmf')
LABELS = ('Frozen topic', 'Module', 'PCA', 'NMF')
SEED, N_COMPONENTS, NMF_MAX_ITER, NMF_TOL = (0, 5, 10000, 0.0001)

def sha256(path):
    return BENCH._sha256(str(path))

def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()

def state_hash(state):
    record = {'meta': state['meta'], 'arrays': {key: array_hash(value) for key, value in state['arrays'].items()}}
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()

def component_rule(train_scores, y, standardized=False):
    """Scale-invariant NMF contrast = group mean difference / training SD.

    SD uses all training patients, ddof=1, not cells. Constant coordinates
    receive zero contrast. Ties select the first component deterministically.
    """
    scores = np.asarray(train_scores, dtype=np.float64)
    y = np.asarray(y)
    difference = scores[y == 1].mean(0) - scores[y == 0].mean(0)
    scale = scores.std(0, ddof=1) if standardized else np.ones(scores.shape[1])
    contrast = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
    selected = int(np.argmax(np.abs(contrast)))
    return {'component': selected, 'orientation': 1.0 if contrast[selected] >= 0 else -1.0, 'training_contrasts': contrast.tolist(), 'selection': 'training_SD_standardized_mean_difference' if standardized else 'training_mean_difference'}

def nmf_scores(X, components):
    """Project each row separately onto a fixed, unit-L2 NMF basis.

    Canonicalizing the basis removes H -> diag(s) H ambiguity before NNLS;
    the same mapping is used for training and both held patients. There is no
    target-dependent normalization, joint fit or target-dependent stopping rule.
    """
    basis = np.asarray(components, dtype=np.float64)
    norms = np.linalg.norm(basis, axis=1)
    basis = basis / np.where(norms > 0, norms, 1.0)[:, None]
    return np.vstack([nnls(basis.T, row, maxiter=1000)[0] for row in np.asarray(X, dtype=np.float64)])

def fit_fold(X, topic, y, panel, patient_ids, held):
    """Fit using the complement of a fixed pair; never inspect held values/labels."""
    held = np.asarray(held, dtype=int)
    if held.shape != (2,) or len(set(held)) != 2 or np.any(held < 0) or np.any(held >= len(y)):
        raise ValueError('two distinct valid held indices required')
    train = np.setdiff1d(np.arange(len(y)), held)
    xt = np.asarray(X[train], dtype=np.float64)
    tt = np.asarray(topic[train], dtype=np.float64)
    yt = np.asarray(y[train], dtype=int)
    if not np.isfinite(xt).all() or (xt < 0).any() or (not np.isfinite(tt).all()):
        raise ValueError('training inputs must be finite nonnegative expression and finite topic scores')
    if set(yt) != {0, 1} or min(np.sum(yt == 0), np.sum(yt == 1)) < 2:
        raise ValueError('each training outcome requires at least two patients')
    states = {}
    _, sign, delta = BENCH._orient(tt, yt)
    states[METHODS[0]] = {'meta': {'component': 0, 'orientation': sign, 'selection': 'prespecified_topic_0', 'training_abs_mean_difference': delta}, 'arrays': {'train_scores': tt[:, None]}}
    positions = pd.Index(np.asarray(panel).astype(str)).get_indexer(BENCH.MODULE_GENES)
    positions = positions[positions >= 0]
    if len(positions) < 5:
        raise ValueError('fewer than five preset module genes on frozen panel')
    mean, scale = (xt[:, positions].mean(0), xt[:, positions].std(0, ddof=1))
    scale[scale == 0] = 1.0
    module = ((xt[:, positions] - mean) / scale).mean(1)
    _, sign, delta = BENCH._orient(module, yt)
    states[METHODS[1]] = {'meta': {'component': -1, 'orientation': sign, 'selection': 'prespecified_module', 'training_abs_mean_difference': delta}, 'arrays': {'positions': positions, 'mean': mean, 'scale': scale, 'train_scores': module[:, None]}}
    k = min(N_COMPONENTS, len(train) - 1, xt.shape[1])
    scaler = StandardScaler()
    z = scaler.fit_transform(xt)
    pca = PCA(n_components=k, svd_solver='full', random_state=SEED)
    pca.fit(z)
    pca_train = pca.transform(z)
    states[METHODS[2]] = {'meta': component_rule(pca_train, yt), 'arrays': {'mean': scaler.mean_, 'scale': scaler.scale_, 'pca_mean': pca.mean_, 'components': pca.components_, 'explained_variance_ratio': pca.explained_variance_ratio_, 'train_scores': pca_train}}
    nmf = NMF(n_components=k, init='nndsvda', solver='cd', beta_loss='frobenius', random_state=SEED, max_iter=NMF_MAX_ITER, tol=NMF_TOL, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, shuffle=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmf.fit(xt)
    raw_basis = nmf.components_
    norms = np.linalg.norm(raw_basis, axis=1)
    basis = raw_basis / np.where(norms > 0, norms, 1.0)[:, None]
    train_w = nmf_scores(xt, basis)
    meta = component_rule(train_w, yt, standardized=True)
    meta.update(n_iter=int(nmf.n_iter_), fit_reconstruction_error=float(nmf.reconstruction_err_), warnings=[str(item.message) for item in caught], projection='independent row NNLS on training-fitted unit-L2 basis')
    states[METHODS[3]] = {'meta': meta, 'arrays': {'raw_components': raw_basis, 'basis_norms': norms, 'components': basis, 'train_scores': train_w}}
    return {'training_indices': train.tolist(), 'held_indices': held.tolist(), 'training_ids': [str(patient_ids[i]) for i in train], 'held_ids': [str(patient_ids[i]) for i in held], 'training_outcomes': yt.tolist(), 'training_input_sha256': {'expression': array_hash(xt), 'topic': array_hash(tt), 'outcomes': array_hash(yt)}, 'states': states}

def score_fold(fold, X_held, topic_held):
    """One unchanged fitted function per method scores both held patients."""
    X_held = np.asarray(X_held, dtype=np.float64)
    coordinates = {METHODS[0]: np.asarray(topic_held, dtype=float)[:, None]}
    a = fold['states'][METHODS[1]]['arrays']
    coordinates[METHODS[1]] = ((X_held[:, a['positions']] - a['mean']) / a['scale']).mean(1)[:, None]
    a = fold['states'][METHODS[2]]['arrays']
    coordinates[METHODS[2]] = ((X_held - a['mean']) / a['scale'] - a['pca_mean']) @ a['components'].T
    a = fold['states'][METHODS[3]]['arrays']
    coordinates[METHODS[3]] = nmf_scores(X_held, a['components'])
    scores = {}
    for method, coordinate in coordinates.items():
        meta = fold['states'][method]['meta']
        component = max(0, meta['component'])
        scores[method] = meta['orientation'] * coordinate[:, component]
        if not np.isfinite(scores[method]).all():
            raise ValueError('nonfinite held score')
    return (scores, coordinates)

def pair_credit(healed, nonhealed):
    return float(healed > nonhealed) + 0.5 * float(healed == nonhealed)

def load_unit():
    paths = {'counts': CACHE / 'discovery_raw_counts.npz', 'genes': CACHE / 'discovery_raw_genes.npy', 'identity': CACHE / 'raw_cell_identity.csv.gz', 'cache_manifest': CACHE / 'raw_cache_manifest.json', 'cells': PROJECTION / 'cells.parquet', 'projection_contract': PROJECTION / 'contract.json', 'panel': ROOT / 'outputs/expression_representation/panel.npy', 'checkpoint': ROOT / 'outputs/expression_representation/topic_model.pt', 'map': ROOT / 'cohort_metadata/GSE165816_ncomms_subject_sample_map.csv', 'legacy': LEGACY, 'benchmark24': BENCH_PATH, 'benchmark42': ROOT / 'scripts/benchmark_patient_readouts.py', 'normalization_code': ROOT / 'wound_models/data_adapter.py'}
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()}
    cache_contract = json.loads(paths['cache_manifest'].read_text())
    contract = json.loads(paths['projection_contract'].read_text())
    for key in ('counts', 'genes', 'identity'):
        if hashes[str(paths[key].relative_to(ROOT))] != cache_contract['cache_sha256'][paths[key].name]:
            raise ValueError(f'raw cache hash mismatch: {key}')
    if hashes[str(paths['cells'].relative_to(ROOT))] != contract['output_sha256']['cells.parquet']:
        raise ValueError('unified projection hash mismatch')
    for key in ('panel', 'checkpoint', 'map'):
        relative = str(paths[key].relative_to(ROOT))
        if hashes[relative] != contract['source_sha256'][relative]:
            raise ValueError(f'projection provenance mismatch: {key}')
    identity = pd.read_csv(paths['identity'])
    cells = pd.read_parquet(paths['cells'])
    if not np.array_equal(identity.raw_cell_id, cells.reconstructed_raw_cell_id):
        raise ValueError('raw identity and projection cell IDs do not align')
    if not np.array_equal(identity.saved_row_0, cells.source_row_zero_based):
        raise ValueError('raw identity and projection rows do not align')
    genes = np.load(paths['genes'], allow_pickle=False).astype(str)
    panel = np.load(paths['panel'], allow_pickle=True).astype(str)
    positions = pd.Index(genes).get_indexer(panel)
    if (positions < 0).any() or len(set(panel)) != len(panel):
        raise ValueError('missing/duplicated panel genes')
    mapping = pd.read_csv(paths['map'], dtype=str).set_index('sample_id', verify_integrity=True)
    mapped = mapping.loc[identity.gsm].reset_index()
    for key in ('patient_id', 'healing_status'):
        if not np.array_equal(mapped[key].fillna(''), cells[key].fillna('')):
            raise ValueError(f'author map disagrees with projection: {key}')
    eligible = identity.arm.isin(['DFU-healer', 'DFU-nonhealer']).to_numpy()
    raw = sp.load_npz(paths['counts'])
    if raw.shape != (len(identity), len(genes)):
        raise ValueError('raw cache shape mismatch')
    normalized = BENCH.library_normalize(raw[eligible][:, positions])
    del raw
    metadata = mapped.loc[eligible].reset_index(drop=True)
    if not metadata.tissue_site.eq('foot').all():
        raise ValueError('non-foot specimen entered outcome unit')
    topic = cells.loc[eligible, 'topic_0'].to_numpy(float)
    rows, X, T = ([], [], [])
    for pid, block in metadata.groupby('patient_id', sort=True):
        idx = block.index.to_numpy()
        if block.healing_status.nunique() != 1:
            raise ValueError('conflicting outcomes within patient')
        X.append(np.asarray(normalized[idx].astype(np.float64).mean(axis=0)).ravel())
        T.append(float(topic[idx].mean()))
        rows.append({'patient_id': pid, 'healing_status': block.healing_status.iloc[0], 'specimen_ids': '|'.join(sorted(block.sample_id.unique())), 'n_specimens': int(block.sample_id.nunique()), 'n_cells': int(len(block))})
    patients = pd.DataFrame(rows)
    y = patients.healing_status.eq('healed').to_numpy(int)
    if len(y) != 11 or int(y.sum()) != 7 or patients.n_specimens.sum() != 14:
        raise ValueError('expected 11 patients (7/4) from 14 DFU specimens')
    return (np.vstack(X), np.array(T), y, panel, patients, hashes)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out != OUTPUT and OUTPUT not in out.parents:
        raise ValueError('outputs must stay inside assigned patient_pair_discrimination directory')
    if (out / 'report.json').exists():
        raise FileExistsError('completed run exists; use a fresh subdirectory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fitted_states').mkdir(exist_ok=True)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    print('Loading and verifying cached counts, author map and deterministic projection', flush=True)
    X, topic, y, panel, patients, hashes = load_unit()
    patients.to_csv(out / 'patients.csv', index=False)
    np.savez_compressed(out / 'patient_inputs.npz', expression=X, topic=topic, outcome=y, panel=panel, patient_ids=patients.patient_id.to_numpy(str))
    legacy = json.loads(LEGACY.read_text())
    pairs = [(int(h), int(n)) for h in np.flatnonzero(y == 1) for n in np.flatnonzero(y == 0)]
    records, fold_records = ([], [])
    with threadpool_limits(limits=1):
        for number, held in enumerate(pairs):
            fold_started = time.perf_counter()
            fold = fit_fold(X, topic, y, panel, patients.patient_id.to_numpy(), held)
            before = {m: state_hash(s) for m, s in fold['states'].items()}
            scores, coordinates = score_fold(fold, X[list(held)], topic[list(held)])
            if before != {m: state_hash(s) for m, s in fold['states'].items()}:
                raise AssertionError('scoring mutated a fitted state')
            arrays, states = ({}, {})
            for method in METHODS:
                state = fold['states'][method]
                arrays.update({f'{method}__{k}': v for k, v in state['arrays'].items()})
                preprocessing = {k: array_hash(v) for k, v in state['arrays'].items() if k in {'mean', 'scale', 'positions', 'pca_mean', 'basis_norms'}}
                pre_hash = hashlib.sha256(json.dumps(preprocessing, sort_keys=True).encode()).hexdigest()
                states[method] = {**state['meta'], 'state_sha256': before[method], 'preprocessing_sha256': pre_hash, 'preprocessing_arrays': preprocessing}
                a, b = scores[method]
                credit = pair_credit(a, b)
                if pair_credit(2.75 * a - 1.0, 2.75 * b - 1.0) != credit:
                    raise AssertionError('within-pair positive affine invariance failed')
                records.append({'fold': number, 'method': method, 'held_healed_id': fold['held_ids'][0], 'held_nonhealed_id': fold['held_ids'][1], 'training_ids': '|'.join(fold['training_ids']), 'healed_score': float(a), 'nonhealed_score': float(b), 'credit': credit, 'component': state['meta']['component'], 'orientation': state['meta']['orientation'], 'preprocessing_sha256': pre_hash, 'state_sha256': before[method]})
            state = fold['states']['nmf']
            factor_scales = np.geomspace(0.01, 100.0, state['arrays']['train_scores'].shape[1])
            alternative = component_rule(state['arrays']['train_scores'] * factor_scales, np.array(fold['training_outcomes']), standardized=True)
            j = alternative['component']
            scaled = alternative['orientation'] * coordinates['nmf'][:, j] * factor_scales[j]
            if j != state['meta']['component'] or pair_credit(*scaled) != pair_credit(*scores['nmf']):
                raise AssertionError('NMF factor scale invariance failed')
            name = f'fitted_states/fold_{number:02d}.npz'
            np.savez_compressed(out / name, **arrays)
            fold_records.append({k: v for k, v in fold.items() if k != 'states'} | {'fold': number, 'held_outcomes': [1, 0], 'states': states, 'state_file': name, 'state_file_sha256': sha256(out / name), 'seconds': time.perf_counter() - fold_started, 'positive_affine_credit_invariant': True, 'positive_nmf_factor_scale_invariant': True, 'scoring_does_not_mutate_fit': True})
            print(f"Pair {number + 1}/28: {fold['held_ids']} completed", flush=True)
    scores_table = pd.DataFrame(records)
    scores_table.to_csv(out / 'pair_scores.csv', index=False)
    summary = []
    for method, label in zip(METHODS, LABELS):
        credits = scores_table.loc[scores_table.method.eq(method), 'credit'].to_numpy()
        if len(credits) != 28:
            raise AssertionError('each method must have all 28 held pairs')
        summary.append({'method': method, 'label': label, 'n_pairs': len(credits), 'wins': int(np.sum(credits == 1)), 'ties': int(np.sum(credits == 0.5)), 'losses': int(np.sum(credits == 0)), 'credit_sum': float(credits.sum()), 'auc': float(credits.mean()), 'legacy_lopo_auc': legacy['auc'][method]['auc']})
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    (out / 'folds.json').write_text(json.dumps(fold_records, indent=2, allow_nan=False) + '\n')
    report = {'status': 'complete', 'experiment': 'patient_pair_discrimination', 'started_at_utc': started_at, 'elapsed_seconds': time.perf_counter() - started, 'device': 'cpu', 'threads': 1, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}, 'input_sha256': hashes, 'script_sha256': sha256(Path(__file__)), 'test_sha256': sha256(ROOT / 'tests/test_patient_pair_discrimination.py'), 'patient_input_array_sha256': {'expression': array_hash(X), 'topic': array_hash(topic)}, 'normalization': 'fixed panel; benchmark24 library_normalize per-cell proportions; float64 cell-count-weighted patient means as benchmark42', 'protocol': {'patients': 11, 'healed': 7, 'nonhealed': 4, 'held_pairs': 28, 'training_patients_per_fold': 9, 'training_healed': 6, 'training_nonhealed': 3, 'seed': SEED, 'components': N_COMPONENTS, 'nmf_max_iter': NMF_MAX_ITER, 'nmf_tol': NMF_TOL, 'hyperparameters_fixed_before_evaluation': True, 'nmf_iteration_note': 'Historical cap 300; initial fixed cap 1000 reached the training convergence limit in folds 8, 9, 23. Raised uniformly to 10000 solely for training convergence; no held-out criterion used. Initial run retained in pilot_maxiter1000 when present.', 'nmf_projection': 'training-fitted basis; canonical unit-L2 rows; independent row NNLS, maxiter 1000', 'pca_selection': 'maximum absolute training component mean difference, as benchmark24', 'nmf_selection': 'maximum absolute training group mean difference / all-training-patient SD (ddof=1)', 'component_ties': 'first index', 'pair_ties': 'exact score equality earns 0.5', 'auc': 'mean of 28 within-fold healed/nonhealed pair credits; no cross-fold raw-score pooling'}, 'summary': summary, 'nmf_warning_folds': [r['fold'] for r in fold_records if r['states']['nmf']['warnings']], 'limitations': ['All 28 pairs reuse 11 patients and overlapping training sets; they are dependent, not 28 biological replicates.', 'No bootstrap confidence intervals, pair-binomial errors or p values are supplied.', 'Frozen topic encoder and panel were learned using discovery expression, including held patients. Fold-local isolation is not external or fully nested representation validation.', 'Legacy LOPO AUC describes a historical pooled-fold scoring procedure only; differences cannot establish representation superiority or isolate the contribution of each repair.', 'This retrospective discrimination audit does not demonstrate clinical usefulness, calibration, causality or equivalence.'], 'output_sha256': {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(out), 'summary': summary, 'elapsed_seconds': report['elapsed_seconds'], 'nmf_warning_folds': report['nmf_warning_folds']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
