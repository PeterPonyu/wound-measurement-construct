"""Retrain the neural topic model across seeds and topic numbers, then match topics.

The frozen discovery model is one training run.  These jobs ask whether the
fibroblast-associated topic and the measurement conclusions reappear across
training seeds and capacities; matching uses the Hungarian algorithm on the
Pearson correlation of square-rooted topic-gene rows.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..common.rng import derive_seed
from ..common.io import atomic_save_npz, atomic_write_json, content_hash, read_json, sha256_file

COUNTS = "outputs/scientific_revision_20260922/measurement_controls/discovery_raw_counts.npz"
GENES = "outputs/scientific_revision_20260922/measurement_controls/discovery_raw_genes.npy"
CELLS = "outputs/scientific_revision_20260922/measurement/projection/cells.parquet"
DISCOVERY = "outputs/direction1_dfu_mainline"


def _identifiers(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty ordered identifier vector")
    text = pd.Series(values).astype('string')
    invalid = text.str.strip().str.lower().isin(['', 'none', 'nan', '<na>', 'null', 'nat'])
    if text.isna().any() or invalid.any() or text.duplicated().any():
        raise ValueError(f"{name} contains missing or duplicate identifiers")
    return text.tolist()


def _proportions(values):
    values = sp.csr_matrix(values, dtype=np.float32)
    if not all(values.shape) or not np.isfinite(values.data).all() or np.any(values.data < 0):
        raise ValueError('proportions must have cells/genes and finite nonnegative entries')
    if not np.allclose(np.asarray(values.sum(axis=1)).ravel(), 1, rtol=0, atol=1e-5):
        raise ValueError('each proportions row must sum to one (no empty cells)')
    return values


def _matrix_hash(values):
    values = sp.csr_matrix(values, dtype=np.float32, copy=True)
    values.sum_duplicates()
    values.sort_indices()
    values.eliminate_zeros()
    digest = hashlib.sha256()
    for data in (np.asarray(values.shape, dtype='<i8'), values.indptr.astype('<i8'),
                 values.indices.astype('<i8'), values.data.astype('<f4')):
        digest.update(data.tobytes())
    return digest.hexdigest()


def load_discovery_panel_counts(root):
    from core_engines.data_adapter import library_normalize

    root = Path(root)
    counts = sp.load_npz(root / COUNTS).tocsr()
    genes = np.load(root / GENES, allow_pickle=True)
    panel = np.load(root / DISCOVERY / "panel.npy", allow_pickle=True)
    cells = pd.read_parquet(root / CELLS)
    if counts.shape[0] != len(cells):
        raise ValueError(f"count rows {counts.shape[0]} do not match cell table rows {len(cells)}")
    genes = np.asarray(_identifiers(genes, 'genes'))
    panel = np.asarray(_identifiers(panel, 'panel'))
    if counts.shape[1] != len(genes) or not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        raise ValueError('counts must align to genes and be finite/nonnegative')
    if not np.allclose(counts.data, np.round(counts.data), rtol=0, atol=1e-6):
        raise ValueError('discovery raw counts must be integer valued')
    for column in ('gsm', 'patient_id', 'celltype'):
        text = cells[column].astype('string')
        if text.isna().any() or text.str.strip().str.lower().isin(['', 'none', 'nan', '<na>', 'null', 'nat']).any():
            raise ValueError(f'missing {column} in discovery cells')
    if (cells.groupby('gsm')['patient_id'].nunique() > 1).any():
        raise ValueError('discovery specimen maps to multiple patients')
    index = pd.Index(genes.astype(str)).get_indexer(panel.astype(str))
    present = np.flatnonzero(index >= 0)
    sub = counts[:, index[present]].tocoo()
    mapped = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(counts.shape[0], len(panel)), dtype=np.float32)
    return _proportions(library_normalize(mapped)), cells


def select_training_panel(counts, genes, training_mask, config):
    """Fit dispersion/support on training rows only, within a supplied gene universe.

    The supplied raw-count universe and cell QC may already have been chosen on
    the discovery cohort. Removing held rows from this fit does not undo that
    earlier selection or validate donor/lineage identities.
    """
    from core_engines.data_adapter import select_target_genes

    counts = sp.csr_matrix(counts)
    genes = np.asarray(_identifiers(genes, 'genes'))
    mask = np.asarray(training_mask)
    if (mask.dtype != bool or mask.shape != (counts.shape[0],) or not mask.any()
            or counts.shape[1] != len(genes) or not np.isfinite(counts.data).all()
            or np.any(counts.data < 0) or not np.allclose(counts.data, np.round(counts.data), atol=1e-6, rtol=0)):
        raise ValueError('invalid training-only panel count/identity/mask inputs')
    for name in ('n_top', 'min_training_cells_per_gene'):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'invalid training-only panel {name}')
    if config.get('method') != 'training_dispersion_v1':
        raise ValueError('unknown training-only panel method')
    training = counts[mask]
    eligible = np.flatnonzero(np.asarray((training > 0).sum(0)).ravel() >= config['min_training_cells_per_gene'])
    if not len(eligible):
        raise ValueError('no training-supported genes for panel')
    local, panel = select_target_genes(training[:, eligible], genes[eligible], n_top=config['n_top'],
            pinned=config['pinned_genes'], exclude_technical=True, log_normalize=True)
    if not len(local):
        raise ValueError('no nontechnical or pinned training-supported genes')
    index = eligible[local]
    record = {'method': 'training_dispersion_v1', 'training_rows': int(mask.sum()),
        'training_counts_sha256': _matrix_hash(training), 'gene_universe_identity': content_hash(genes.tolist(), 64),
        'eligible_training_genes': len(eligible), 'n_selected_genes': len(index),
        'panel_identity': content_hash(panel.tolist(), 64), 'config': config,
        'held_values_used_for_panel_fit': False, 'end_to_end_independent': False,
        'limitations': ['inherited_discovery_raw_gene_universe_and_cell_QC', 'supplied_patient_and_lineage_labels']}
    return index, panel, record


def load_training_panel_counts(root, partition_config, panel_config):
    """Select first on training patients, then transform all supplied cells."""
    from core_engines.data_adapter import library_normalize

    root = Path(root)
    cells = pd.read_parquet(root / CELLS)
    training_mask, _, _ = patient_partition(cells, partition_config)
    counts = sp.load_npz(root / COUNTS).tocsr()
    genes = np.load(root / GENES, allow_pickle=True)
    index, panel, record = select_training_panel(counts, genes, training_mask, panel_config)
    mapped = counts[:, index]
    if np.any(np.asarray(mapped.sum(1)).ravel() == 0):
        raise ValueError('selected training panel gives empty cells; do not select a rescue panel using held data')
    return _proportions(library_normalize(mapped)), cells, panel, record


def match_panel_topics(beta_reference, reference_panel, beta_new, new_panel):
    """Descriptive matching on explicitly intersected genes; never select a fit.

    The full-discovery reference is not a held-patient biological gold standard.
    Its topic numbering is used only for a transparently conditional readout.
    """
    reference_panel = _identifiers(reference_panel, 'reference_panel')
    new_panel = _identifiers(new_panel, 'new_panel')
    a, b = np.asarray(beta_reference), np.asarray(beta_new)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != len(reference_panel) or b.shape[1] != len(new_panel):
        raise ValueError('topic matrices do not align to their panel identities')
    common = sorted(set(reference_panel) & set(new_panel))
    if len(common) < 2:
        raise ValueError('too few shared genes for descriptive topic matching')
    ai, bi = pd.Index(reference_panel).get_indexer(common), pd.Index(new_panel).get_indexer(common)
    # Do not renormalize rows: the existing Pearson square-root rule is scale invariant.
    matched = match_topics(a[:, ai], b[:, bi])
    matched['panel_comparison'] = {'shared_genes': len(common), 'reference_genes': len(reference_panel),
        'new_genes': len(new_panel), 'shared_gene_identity': content_hash(common, 64),
        'reference_retained_probability_mass': a[:, ai].sum(1).tolist(),
        'new_retained_probability_mass': b[:, bi].sum(1).tolist(),
        'scope': 'intersection_conditioned_descriptive_match_not_independent_validation_or_selection'}
    return matched


def patient_partition(cells, config):
    """Outcome-blind patient-ID split and fixed evaluation rows, bound to cell order.

    No outcome, group, lineage or fitted score is consulted. This does not make
    the inherited discovery panel or supplied patient identity independently
    validated. Held patients are excluded from every optimization batch.
    """
    ids = _identifiers(cells[config['cell_identity_column']].to_numpy(), 'cell_ids')
    patients = cells['patient_id'].astype('string')
    if patients.isna().any() or patients.str.strip().eq('').any():
        raise ValueError('missing patient identity in partition')
    unique = sorted(patients.unique().tolist())
    n, cap, seed = config['n_held_patients'], config['evaluation_cells_per_patient'], config['seed']
    if (config.get('method') != 'id_hash_v1' or isinstance(n, bool) or not isinstance(n, int)
            or not 1 <= n < len(unique) or isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
            or isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError('invalid fixed patient partition configuration')
    ordered = sorted(unique, key=lambda p: (content_hash(['outcome-blind-patient-partition-v1', seed, p], 64), p))
    held = ordered[:n]
    if held != config['expected_held_patients']:
        raise ValueError('patient partition differs from the frozen held-ID list')
    training_mask = ~patients.isin(held).to_numpy()
    row_patients = patients.to_numpy(dtype=str)
    selections = {}
    for name, mask in (('train', training_mask), ('held', ~training_mask)):
        selected = []
        for patient in sorted(set(row_patients[mask])):
            rows = np.flatnonzero((row_patients == patient) & mask)
            order = sorted(rows.tolist(), key=lambda i: (content_hash(['fixed-eval-cell-v1', seed, ids[i]], 64), ids[i]))
            selected.extend(order[:cap])
        selections[name] = np.asarray(selected, dtype=int)
    record = {'method': 'id_hash_v1', 'seed': seed, 'training_patients': sorted(set(row_patients[training_mask])),
              'held_patients': held, 'optimization_cells': int(training_mask.sum()),
              'held_cells': int((~training_mask).sum()), 'evaluation_cells_per_patient': cap,
              'cell_order_identity': content_hash(ids, 64),
              'patient_order_identity': content_hash(row_patients.tolist(), 64),
              'optimization_cell_identity': content_hash(np.asarray(ids)[training_mask].tolist(), 64),
              'evaluation_cell_ids': {k: np.asarray(ids)[v].tolist() for k, v in selections.items()},
              'selection_uses_outcomes': False, 'held_patients_excluded_from_optimization': True,
              'end_to_end_independent': False,
              'limitations': ['gene_panel_inherited_from_full_discovery', 'supplied_patient_and_lineage_labels']}
    if config.get('expected_cell_order_identity') != record['cell_order_identity']:
        raise ValueError('cell order differs from the frozen patient partition')
    return training_mask, selections, record


def evaluate_topic_model(model, proportions, units, *, chunk=128):
    """Deterministic eval-coordinate scores; not a stochastic ELBO estimate.

    Restores train/eval mode. No random draw or state update occurs; tests bind
    RNG, BatchNorm buffers and exact optimization trajectories.
    """
    import torch
    proportions = _proportions(proportions)
    units = np.asarray(units, dtype=str)
    if units.shape != (proportions.shape[0],) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('evaluation patient labels/chunk do not align')
    device, mode = next(model.parameters()).device, model.training
    arrays = []
    try:
        model.eval()
        with torch.no_grad():
            beta = model.decoder.beta
            for start in range(0, len(units), chunk):
                batch = torch.as_tensor(proportions[start:start + chunk].toarray(), device=device)
                mu, logvar = model.encoder(batch)
                theta = torch.softmax(mu, dim=-1)
                reconstruction = -torch.sum(batch * torch.log(theta @ beta + 1e-10), dim=-1)
                kl = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                entropy = -torch.sum(theta * torch.log(theta.clamp_min(1e-10)), dim=-1)
                arrays.append(torch.cat([reconstruction[:, None], kl[:, None], entropy[:, None], theta], 1).cpu().numpy())
    finally:
        model.train(mode)
    scores = np.vstack(arrays).astype(np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError('nonfinite deterministic topic diagnostic')
    patient_scores = []
    for patient in sorted(set(units)):
        subset = scores[units == patient]
        patient_scores.append({'patient_id': patient, 'n_cells': len(subset),
                               'reconstruction': float(subset[:, 0].mean()), 'kl': float(subset[:, 1].mean()),
                               'objective': float((subset[:, 0] + model.kl_weight * subset[:, 1]).mean())})
    theta = scores[:, 3:]
    return {'cells': len(units), 'patients': len(patient_scores), 'by_patient': patient_scores,
            'cell_mean_reconstruction': float(scores[:, 0].mean()), 'cell_mean_kl': float(scores[:, 1].mean()),
            'cell_mean_objective': float((scores[:, 0] + model.kl_weight * scores[:, 1]).mean()),
            'equal_patient_reconstruction': float(np.mean([p['reconstruction'] for p in patient_scores])),
            'equal_patient_objective': float(np.mean([p['objective'] for p in patient_scores])),
            'mean_normalized_theta_entropy': float(scores[:, 2].mean() / np.log(theta.shape[1])) if theta.shape[1] > 1 else 0.,
            'topic_mean_loading': theta.mean(0).tolist(),
            'argmax_topic_counts': np.bincount(theta.argmax(1), minlength=theta.shape[1]).tolist(),
            'scope': 'deterministic_softmax_mu_diagnostic_not_ELBO_or_outcome_validation'}


def train_topic_model(proportions, *, K: int, seed: int, steps: int, batch_size: int, lr: float, kl_weight: float,
                      device: str = "cpu", log_every: int = 1000, diagnostic_inputs=None,
                      diagnostic_every: int = 100) -> dict:
    import torch

    from core_engines.topic_simplex_engine import TopicSimplexModel

    proportions = _proportions(proportions)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < minimum
           for v, minimum in ((K, 1), (steps, 1), (batch_size, 2), (log_every, 1))):
        raise ValueError('K, steps, batch_size and log_every must be valid positive integers; batch >= 2')
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError('lr must be positive and kl_weight nonnegative, both finite')
    if diagnostic_inputs is not None:
        if (isinstance(diagnostic_every, bool) or not isinstance(diagnostic_every, int) or diagnostic_every < 1
                or set(diagnostic_inputs) != {'train', 'held'}):
            raise ValueError('invalid diagnostic sets or evaluation cadence')
        for values in diagnostic_inputs.values():
            matrix = _proportions(values['proportions'])
            if matrix.shape[1] != proportions.shape[1] or len(values['units']) != matrix.shape[0]:
                raise ValueError('diagnostic input dimensions do not align')
    torch.manual_seed(derive_seed("topic-init", seed, K))
    generator = torch.Generator().manual_seed(derive_seed("topic-batches", seed, K))
    model = TopicSimplexModel(input_dim=proportions.shape[1], num_topics=int(K), kl_weight=kl_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses, dense_trace, evaluation_trace = [], [], []

    def evaluate(step):
        evaluation_trace.append({'step': step, **{name: evaluate_topic_model(model, **values)
                                                 for name, values in diagnostic_inputs.items()}})
    if diagnostic_inputs is not None:
        evaluate(0)
    for step in range(int(steps)):
        rows = torch.randint(proportions.shape[0], (int(batch_size),), generator=generator).numpy()
        batch = torch.as_tensor(proportions[rows].toarray(), device=device)
        output = model(batch)
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError(f"non-finite topic-model loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        optimizer.step()
        if diagnostic_inputs is not None:
            dense_trace.append({'step': step + 1, **{name: float(output[name].detach().cpu())
                                                    for name in ('loss', 'recon_loss', 'kl_div')}})
            if (step + 1) % diagnostic_every == 0 or step == int(steps) - 1:
                evaluate(step + 1)
        if step % log_every == 0 or step == int(steps) - 1:
            losses.append(float(output["loss"].detach().cpu()))
    model.eval()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    beta = model.decoder.beta.detach().cpu().numpy().astype(np.float64)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError('non-finite trained model state')
    result = {"state": state, "beta": beta, "losses": losses, "K": int(K), "seed": int(seed), "steps": int(steps),
            "model_config": {"input_dim": proportions.shape[1], "num_topics": int(K), "hidden_dim": 256,
                             "kl_weight": float(kl_weight)},
            "training_config": {"steps": int(steps), "batch_size": int(batch_size), "lr": float(lr),
                                "seed": int(seed), "device": str(device)}}
    if diagnostic_inputs is not None:
        result.update(training_trace=dense_trace, evaluation_trace=evaluation_trace,
                      diagnostic_every=diagnostic_every)
    return result


def project_theta(state: dict, proportions, K: int, *, device: str = "cpu", chunk: int = 8192) -> np.ndarray:
    import torch

    from core_engines.topic_simplex_engine import TopicSimplexModel

    proportions = _proportions(proportions)
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError('projection chunk must be a positive integer')
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('non-finite model state')
    model = TopicSimplexModel(input_dim=proportions.shape[1], num_topics=int(K)).to(device)
    model.load_state_dict(state)
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, proportions.shape[0], chunk):
            batch = torch.as_tensor(proportions[start:start + chunk].toarray(), device=device)
            blocks.append(model(batch)["theta"].cpu().numpy())
    theta = np.vstack(blocks).astype(np.float32)
    if not np.isfinite(theta).all() or not np.allclose(theta.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError('invalid simplex projection')
    return theta


def match_topics(beta_ref, beta_new) -> dict:
    from scipy.optimize import linear_sum_assignment

    matrices = [np.asarray(beta_ref, dtype=np.float64), np.asarray(beta_new, dtype=np.float64)]
    for matrix in matrices:
        if (matrix.ndim != 2 or not all(matrix.shape) or matrix.shape[1] < 2
                or not np.isfinite(matrix).all() or np.any(matrix < 0)
                or np.any(matrix.sum(axis=1) <= 0)):
            raise ValueError('topic matrices must be finite nonnegative rows on the same gene panel')
        if np.any(np.std(np.sqrt(matrix), axis=1) <= 1e-15):
            raise ValueError('constant topic rows have undefined correlation')
    if matrices[0].shape[1] != matrices[1].shape[1]:
        raise ValueError('topic matrices use different gene panel widths')
    a, b = [np.sqrt(matrix) for matrix in matrices]
    a = (a - a.mean(axis=1, keepdims=True)) / a.std(axis=1, keepdims=True)
    b = (b - b.mean(axis=1, keepdims=True)) / b.std(axis=1, keepdims=True)
    correlation = np.clip(a @ b.T / a.shape[1], -1, 1)
    rows, cols = linear_sum_assignment(-correlation)
    best = np.argmax(correlation, axis=1)
    second = np.sort(correlation, axis=1)[:, -2] if b.shape[0] > 1 else None
    return {"assignment": {int(r): int(c) for r, c in zip(rows, cols)},
            "assigned_correlation": {int(r): float(correlation[r, c]) for r, c in zip(rows, cols)},
            "best_correlation": {int(r): float(correlation[r].max()) for r in range(correlation.shape[0])},
            "best_match": {int(r): int(best[r]) for r in range(correlation.shape[0])},
            "second_best_correlation": {r: float(second[r]) if second is not None else None for r in range(len(a))},
            "best_minus_second": {r: float(correlation[r, best[r]] - second[r]) if second is not None else None
                                  for r in range(len(a))},
            "best_minus_assigned": {int(r): float(correlation[r, best[r]] - correlation[r, c]) for r, c in zip(rows, cols)},
            "unmatched_reference": sorted(set(range(len(a))) - set(map(int, rows))),
            "unmatched_new": sorted(set(range(len(b))) - set(map(int, cols))),
            "correlation": correlation.tolist(), "primary_rule": "hungarian_one_to_one"}


def save_inference_checkpoint(folder, trained, proportions, theta, *, panel, cell_ids, provenance) -> dict:
    """Persist complete inference state without pickle; no optimizer-step resume is claimed."""
    folder = Path(folder)
    names = ('model_state.npz', 'topics.npz', 'checkpoint.json', 'checkpoint.sha256.json')
    if any((folder / name).exists() for name in names):
        raise FileExistsError(f'checkpoint already exists in {folder}')
    panel, cell_ids = _identifiers(panel, 'panel'), _identifiers(cell_ids, 'cell_ids')
    proportions = _proportions(proportions)
    if proportions.shape != (len(cell_ids), len(panel)):
        raise ValueError('checkpoint input dimensions do not match identity vectors')
    theta = np.asarray(theta, dtype=np.float32)
    if theta.shape != (len(cell_ids), trained['K']):
        raise ValueError('checkpoint theta dimensions do not match input')
    atomic_save_npz(folder / 'model_state.npz', **{k: v.detach().cpu().numpy() for k, v in trained['state'].items()})
    atomic_save_npz(folder / 'topics.npz', beta=trained['beta'].astype(np.float32), theta=theta,
                    panel=np.asarray(panel, dtype=str), cell_ids=np.asarray(cell_ids, dtype=str))
    metadata = {'schema_version': 1, 'kind': 'complete_inference_state', 'optimizer_resume_supported': False,
                'reload_verified': False,
                'model_config': trained['model_config'], 'training_config': trained['training_config'],
                'losses': trained['losses'], 'panel_identity': content_hash(panel, 64),
                'cell_order_identity': content_hash(cell_ids, 64), 'proportions_sha256': _matrix_hash(proportions),
                'provenance': provenance, 'reload_device': 'cpu', 'projection_chunk': 8192,
                'reload_tolerance': 1e-6}
    atomic_write_json(folder / 'checkpoint.json', metadata)
    atomic_write_json(folder / 'checkpoint.sha256.json', {name: sha256_file(folder / name) for name in names[:3]})
    loaded = _load_checkpoint_payload(folder, panel=panel, cell_ids=cell_ids)
    projected = project_theta(loaded['state'], proportions, trained['K'], device='cpu')
    error = float(np.max(np.abs(projected - theta)))
    if not np.isfinite(error) or error > metadata['reload_tolerance']:
        raise ValueError(f'CPU inference reload discrepancy {error}')
    metadata['reload_max_abs_error'] = error
    metadata['reload_verified'] = True
    atomic_write_json(folder / 'checkpoint.json', metadata)
    atomic_write_json(folder / 'checkpoint.sha256.json', {name: sha256_file(folder / name) for name in names[:3]})
    return metadata


def load_inference_checkpoint(folder, *, panel, cell_ids) -> dict:
    loaded = _load_checkpoint_payload(folder, panel=panel, cell_ids=cell_ids)
    metadata = loaded['metadata']
    error = metadata.get('reload_max_abs_error')
    if (metadata.get('reload_verified') is not True or not isinstance(error, (float, int))
            or not np.isfinite(error) or error > 1e-6 or error < 0):
        raise ValueError('checkpoint has not passed CPU reload verification')
    return loaded


def _load_checkpoint_payload(folder, *, panel, cell_ids) -> dict:
    """Only numerical NPZ tensors are deserialized; checksums are integrity, not authentication."""
    import torch
    from core_engines.topic_simplex_engine import TopicSimplexModel

    folder = Path(folder)
    panel, cell_ids = _identifiers(panel, 'panel'), _identifiers(cell_ids, 'cell_ids')
    hashes = read_json(folder / 'checkpoint.sha256.json')
    if set(hashes) != {'model_state.npz', 'topics.npz', 'checkpoint.json'}:
        raise ValueError('checkpoint checksum inventory mismatch')
    if any(not (folder / name).is_file() or sha256_file(folder / name) != digest for name, digest in hashes.items()):
        raise ValueError('checkpoint checksum mismatch')
    metadata = read_json(folder / 'checkpoint.json')
    if metadata.get('schema_version') != 1 or metadata.get('kind') != 'complete_inference_state':
        raise ValueError('unknown checkpoint schema')
    for values, key, name in ((panel, 'panel_identity', 'panel'), (cell_ids, 'cell_order_identity', 'cell_ids')):
        if content_hash(values, 64) != metadata[key]:
            raise ValueError(f'{name} identity mismatch')
    with np.load(folder / 'model_state.npz', allow_pickle=False) as arrays:
        state = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('checkpoint state is nonfinite')
    model = TopicSimplexModel(**metadata['model_config'])
    expected = model.state_dict()
    if set(expected) != set(state) or any(state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype
                                         for k in expected):
        raise ValueError('checkpoint state keys/shapes/dtypes mismatch')
    model.load_state_dict(state, strict=True)
    with np.load(folder / 'topics.npz', allow_pickle=False) as arrays:
        beta, theta = arrays['beta'], arrays['theta']
        if arrays['panel'].tolist() != list(panel) or arrays['cell_ids'].tolist() != list(cell_ids):
            raise ValueError('checkpoint arrays identity mismatch')
    k = metadata['model_config']['num_topics']
    for values, shape in ((beta, (k, len(panel))), (theta, (len(cell_ids), k))):
        if (values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all()
                or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-5)):
            raise ValueError('invalid checkpoint simplex arrays')
    if not np.allclose(model.decoder.beta.detach().numpy(), beta, rtol=0, atol=1e-6):
        raise ValueError('checkpoint decoder disagrees with beta')
    return {'state': state, 'beta': beta, 'theta': theta, 'metadata': metadata}


def gmm_midpoint(values) -> float:
    from sklearn.mixture import GaussianMixture

    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return float(np.mean(GaussianMixture(n_components=2, random_state=0).fit(values).means_))
