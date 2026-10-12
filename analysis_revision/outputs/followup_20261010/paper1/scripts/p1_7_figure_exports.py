#!/usr/bin/env python3
"""Export the selection-aware null draws used by the manuscript figure.

Recomputes, from the saved patient table, the statistic of P1-1: for every one
of the 330 allocations of 7 healed among 11 DFU patients, the maximum over the
15 topics of the healed/non-healed patient-mean ratio. Checks the observed
selection-adjusted p against p1_1_selection.json before writing.
"""
import itertools, json
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'
tab = pd.read_csv(R / 'p1_1_patient_readouts_all_topics.csv')
healed = tab.healed.astype(str).str.lower().eq('true').to_numpy()
rows = []
for readout in ['S', 'M', 'W']:
    V = tab[[f'{readout}_topic{k}' for k in range(15)]].to_numpy(float)
    stats = []
    for combo in itertools.combinations(range(len(tab)), int(healed.sum())):
        a = np.zeros(len(tab), bool); a[list(combo)] = True
        ratio = V[a].mean(0) / V[~a].mean(0)
        stats.append((ratio.max(), ratio[0], bool((a == healed).all())))
    stats = np.array(stats, dtype=object)
    observed = [s for s in stats if s[2]][0]
    null = np.array([s[0] for s in stats], float)
    p = float(np.mean(null >= observed[0] - 1e-12))
    rows += [dict(readout=readout, allocation=i, max_ratio=float(s[0]), topic0_ratio=float(s[1]), observed=s[2])
             for i, s in enumerate(stats)]
    saved = json.load(open(R / 'p1_1_selection.json'))
    key = {'S': 'S_all_cell_mean', 'M': 'M_fibroblast_mean', 'W': 'W_fibroblast_fraction_topic_cut'}[readout]
    print(readout, 'adjusted p', round(p, 4), 'saved', round(saved[key]['max_ratio']['selection_adjusted_p'], 4))
pd.DataFrame(rows).to_csv(R / 'p1_7_selection_null_draws.csv', index=False)
