# Patient-pair discrimination: Figure 8 and table input

Authoritative results are in this directory; `pilot_maxiter1000/` is a retained
convergence pilot, not the final run. No shared manuscript or export files changed.

- `summary.csv`: `method,label,n_pairs,wins,ties,losses,credit_sum,auc,legacy_lopo_auc`.
  Plot `auc` by `label`; do not attach historical confidence intervals or p values.
- `pair_scores.csv`: 112 rows (28 folds x 4 methods), with `fold,method`, exact
  `held_healed_id,held_nonhealed_id,training_ids`, `healed_score,nonhealed_score,credit`,
  selected `component,orientation`, and `preprocessing_sha256,state_sha256`.
- `folds.json`: exact held/training indices, patient IDs and outcomes, fitted-state
  fingerprints, state-file paths/hashes, convergence diagnostics and fold runtimes.
- `fitted_states/fold_00.npz` through `fold_27.npz`: saved preprocessing and fitted
  arrays. `patient_inputs.npz` and `patients.csv` preserve inputs and specimen mapping.
- `report.json`: protocol, input SHA256, package versions, runtime, results and limits.
- `test_results.xml`: 11 behavioral tests passed. Independently reloading the 28
  states reproduced all 112 score pairs and their credits; input/output hashes matched.

Final AUCs (mean of 28 within-fold credits): frozen topic 20/28 = 0.714286;
module 22/28 = 0.785714; PCA 14/28 = 0.500000; NMF (5 + 0.5 x 3)/28 = 0.232143.
NMF has 5 wins, 3 ties and 20 losses; historical pooled-fold LOPO AUC was 0.125.
The historical value describes another scoring procedure, not evidence of superiority.

Each fold trains on nine patients (six healed, three nonhealed); both held patients
are scored by the same fitted function. NMF uses training-SD-standardized component
selection, a unit-L2 training-fitted basis and row-independent NNLS projection.
The initial 1000-iteration cap caused training convergence warnings in folds 8, 9,
23. A uniform cap of 10000 was adopted solely for training convergence; all final
folds converged (maximum 2139 iterations), with unchanged pair AUCs. Final CPU runtime
including input loading and hashing was 8.0643 seconds, using one BLAS thread.

The 28 pairs share patients and overlapping training sets. No bootstrap intervals,
binomial pair errors or p values are supplied. The frozen topic encoder and panel
already used discovery expression, including held patients: this is an internal
scoring-procedure audit, not external or fully nested representation validation.
