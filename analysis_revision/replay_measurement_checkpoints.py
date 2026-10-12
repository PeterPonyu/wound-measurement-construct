#!/usr/bin/env python3
"""Reload and reproject the nine measurement checkpoints without fitting."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from expansion.common.io import sanitize
from expansion.measurement import stability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    if dest == ROOT or not dest.is_relative_to(ROOT):
        raise ValueError('Choose a fresh output directory inside the portable closure')
    dest.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    proportions, cells = stability.load_discovery_panel_counts(ROOT)
    panel = np.load(ROOT/stability.DISCOVERY/'panel.npy', allow_pickle=True)
    reference = np.load(ROOT/stability.DISCOVERY/'beta.npy', allow_pickle=False)
    native = ROOT/'output/publication-quality-20261009/p1/run'
    jobs = [json.loads(line) for line in (native/'jobs.jsonl').read_text().splitlines()]
    if {(job['params']['K'],job['seed']) for job in jobs} != {(k,s) for k in (10,15,20) for s in (0,1,2)}:
        raise ValueError('The fixed capacity/seed grid differs')
    rows = []
    for job in sorted(jobs, key=lambda x:(x['params']['K'], x['seed'])):
        folder = native/'jobs'/job['job_id']
        result = json.loads((folder/'result.json').read_text())
        loaded = stability.load_inference_checkpoint(folder, panel=panel,
                    cell_ids=cells.reconstructed_raw_cell_id.to_numpy())
        if loaded['metadata'] != result['inference_checkpoint']:
            raise ValueError('Result/checkpoint metadata differ')
        if loaded['metadata']['proportions_sha256'] != stability._matrix_hash(proportions):
            raise ValueError('Checkpoint input proportions differ')
        projected = stability.project_theta(loaded['state'], proportions, job['params']['K'], device='cpu')
        error = float(np.max(np.abs(projected-loaded['theta'])))
        if not np.isfinite(error) or error > 1e-6:
            raise ValueError('Unchanged projection tolerance failed: '+str(error))
        matched = stability.match_topics(reference, loaded['beta'])
        if sanitize(matched) != result['match_to_reference']:
            raise ValueError('Hungarian match differs from saved reference')
        rows.append(dict(job_id=job['job_id'], K=job['params']['K'], seed=job['seed'],
                         projection_max_abs_error=error, match_reproduced=True))
        print(json.dumps(rows[-1]), flush=True)
    receipt = dict(status='passed', model_training_performed=False, network_access_performed=False,
                   checkpoints=rows, cells=len(cells), elapsed_seconds=time.time()-started)
    (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
