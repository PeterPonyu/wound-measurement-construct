#!/usr/bin/env python3
"""GSE241132 join contract, fold-internal scaling, and historical-output lock.

Scripts 18-22 used to share a copy-pasted join that reset the pandas index
after merging author metadata, then treated that index as a row into coh.X.
This module is the single implementation those scripts must call.

It also owns two protocol rules the 2026-09-17 review made mandatory:

  * mu / theta standardisation is estimated on training cells only
  * historical report.json files must not be overwritten by a repaired rerun
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.data_adapter import library_normalize, load_10x_zip_dir
from wound_models.topic_model import TopicModel
SAMPLE_RE = re.compile('(GSM\\d+)_(PWH\\d+)(D\\d+)\\.zip$')
D2COND = {'D0': 'Skin', 'D1': 'Wound1', 'D7': 'Wound7', 'D30': 'Wound30'}
COND_ORDER = ['Skin', 'Wound1', 'Wound7', 'Wound30']
COND_TIME = {'Skin': 0.0, 'Wound1': 1 / 3, 'Wound7': 2 / 3, 'Wound30': 1.0}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANALYSIS_ROOT = 'outputs/analysis'
HISTORICAL_OUTPUT_DIRS = ('outputs/human_temporal_flow', 'outputs/time_parameterisation', 'outputs/donor_transfer_flow', 'outputs/donor_conditioned_benchmark', 'outputs/negative_result_strength')

class AlignmentError(RuntimeError):
    """Raised when a barcode does not select the matrix row it claims."""

def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def parse_sample_map(per_gsm_dir: str) -> Dict[str, Dict[str, str]]:
    sample_of: Dict[str, Dict[str, str]] = {}
    for name in sorted(os.listdir(per_gsm_dir)):
        match = SAMPLE_RE.match(name)
        if match is None:
            continue
        gsm, donor, day = match.groups()
        cond = D2COND[day]
        sample_of[gsm] = {'donor': donor, 'cond': cond, 'label': donor + day}
    return sample_of

def barcodes_from_cells(cells: np.ndarray) -> np.ndarray:
    return np.array([str(cell).split(':', 1)[1] for cell in cells], dtype=object)

def join_from_zip_dir(coh, per_gsm_dir: str, meta_path: str, extra_cols: Sequence[str]=()) -> pd.DataFrame:
    sample_of = parse_sample_map(per_gsm_dir)
    obs = coh.obs.copy()
    for key in ('donor', 'cond', 'label'):
        obs[key] = obs['gsm'].map(lambda gsm, key=key: sample_of.get(gsm, {}).get(key))
    if obs['label'].isna().any():
        missing = sorted(set(obs.loc[obs['label'].isna(), 'gsm'].astype(str)))
        raise AlignmentError(f'GSM ids have no zip-name map: {missing[:8]}')
    raw_bc = barcodes_from_cells(coh.cells)
    obs['key'] = obs['label'].astype(str) + '_' + pd.Series(raw_bc).astype(str)
    obs['_raw_row'] = np.arange(len(obs), dtype=np.int64)
    meta = pd.read_csv(meta_path, sep='\t')
    wanted = [str(meta.columns[0]), 'newMainCellTypes']
    for col in extra_cols:
        if col in meta.columns and col not in wanted:
            wanted.append(col)
    meta = meta.loc[:, wanted]
    meta.columns = pd.Index(['key', 'celltype'] + list(wanted[2:]))
    joined = obs.merge(meta, on='key', how='inner', sort=False, validate='one_to_one')
    assert_row_contract(coh, joined)
    return joined

def assert_row_contract(coh, obs: pd.DataFrame) -> np.ndarray:
    if '_raw_row' not in obs.columns:
        raise AlignmentError('obs is missing _raw_row; refuse to guess matrix rows')
    row_idx = obs['_raw_row'].to_numpy(dtype=np.int64)
    raw_bc = barcodes_from_cells(coh.cells)
    expected = obs['key'].astype(str).str.rsplit('_', n=1).str[-1].to_numpy(dtype=object)
    if not np.array_equal(raw_bc[row_idx], expected):
        n_bad = int(np.sum(raw_bc[row_idx] != expected))
        raise AlignmentError(f'GSE241132 barcode-to-matrix row contract failed ({n_bad} rows)')
    return row_idx

def aligned_X(coh, obs: pd.DataFrame):
    return coh.X[assert_row_contract(coh, obs)]

def fold_standardize(mu: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Centre/scale using training cells only; apply to the full matrix."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape[0] != mu.shape[0]:
        raise ValueError('train_mask length must match mu rows')
    n_train = int(train_mask.sum())
    if n_train < 2:
        raise ValueError(f'scaler needs >=2 training cells, got {n_train}')
    mean = mu[train_mask].mean(axis=0, keepdims=True)
    std = mu[train_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled = ((mu - mean) / std).astype(np.float32)
    return (scaled, {'n_train_cells': n_train, 'n_applied_cells': int(mu.shape[0]), 'n_held_out_of_scaler': int((~train_mask).sum()), 'train_mean_abs': float(np.abs(scaled[train_mask].mean()))})

def paired_endpoint_pairs(donors: np.ndarray, conds: np.ndarray, donor_list: Sequence[str], segments: Sequence[Tuple[str, str]], min_cells: int=20) -> List[Tuple[str, str, str]]:
    pairs = []
    for donor in donor_list:
        for src, tgt in segments:
            n_src = int(np.sum((donors == donor) & (conds == src)))
            n_tgt = int(np.sum((donors == donor) & (conds == tgt)))
            if n_src >= min_cells and n_tgt >= min_cells:
                pairs.append((donor, src, tgt))
    if not pairs:
        raise ValueError('no donor-paired endpoints with enough cells')
    return pairs

def project_latent(X, genes, panel, model, dev, n_panel: int, key: str='mu') -> Tuple[np.ndarray, int]:
    if key not in {'mu', 'theta'}:
        raise ValueError(f'key must be mu or theta, got {key!r}')
    pos = pd.Index(genes).get_indexer(panel)
    present = np.flatnonzero(pos >= 0)
    sub = X[:, pos[present]].tocoo()
    n_cells = int(X.shape[0])
    mapped = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(n_cells, n_panel), dtype=np.int32)
    normalised = library_normalize(mapped)
    chunks = []
    with torch.no_grad():
        for start in range(0, n_cells, 4096):
            batch = torch.from_numpy(np.asarray(normalised[start:start + 4096].todense(), dtype=np.float32)).to(dev)
            chunks.append(model(batch)[key].cpu().numpy())
    return (np.vstack(chunks), int(present.size))

def load_discovery(discovery_dir: str):
    panel = np.load(os.path.join(discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopicModel(input_dim=n_panel, num_topics=n_topics).to(device)
    model.load_state_dict(state)
    model.eval()
    return (panel, beta, model, n_topics, n_panel, device)

def load_annotated_counts(per_gsm_dir: str, meta_path: str, extra_cols: Sequence[str]=()):
    coh = load_10x_zip_dir(per_gsm_dir)
    obs = join_from_zip_dir(coh, per_gsm_dir, meta_path, extra_cols=extra_cols)
    matrix = aligned_X(coh, obs)
    return (coh, obs, matrix)

def load_lineage_latent(per_gsm_dir: str, meta_path: str, discovery_dir: str, lineage: str='Fibroblast', extra_cols: Sequence[str]=(), key: str='mu'):
    coh, obs, matrix = load_annotated_counts(per_gsm_dir, meta_path, extra_cols=extra_cols)
    panel, beta, model, n_topics, n_panel, device = load_discovery(discovery_dir)
    keep = np.flatnonzero((obs['celltype'] == lineage).to_numpy())
    obs_kept = obs.iloc[keep].reset_index(drop=True)
    latent, covered = project_latent(matrix[keep], coh.genes, panel, model, device, n_panel, key=key)
    return {'latent': latent, 'obs': obs_kept, 'n_topics': n_topics, 'n_panel': n_panel, 'covered': covered, 'device': device, 'panel': panel, 'n_cells_joined': int(len(obs)), 'n_cells_loaded': int(len(coh.obs))}

def protocol_block(script_path: str, discovery_dir: str, seed: int, folds: Iterable[str], extra: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    block = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'fold_internal_standardization': True, 'held_out_target_not_in_scaler': True, 'code_sha256': file_sha256(script_path), 'helper_sha256': file_sha256(__file__), 'panel_sha256': file_sha256(os.path.join(discovery_dir, 'panel.npy')), 'model_sha256': file_sha256(os.path.join(discovery_dir, 'topic_model.pt')), 'seed': int(seed), 'folds': list(folds), 'analysis_output_root': ANALYSIS_ROOT}
    if extra:
        block.update(extra)
    return block

def preserve_reference_outputs(output_dir: str) -> None:
    abs_out = os.path.abspath(output_dir)
    for rel in HISTORICAL_OUTPUT_DIRS:
        if abs_out == os.path.abspath(os.path.join(PROJECT_ROOT, rel)):
            raise SystemExit(f'refusing to overwrite historical {rel}; write under {ANALYSIS_ROOT}/ instead')
