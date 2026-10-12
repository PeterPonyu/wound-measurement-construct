# Deterministic measurement revision integration

All paths below are relative to outputs/scientific_revision_20260922/measurement/. Historical inputs are read only. Run: `/home/zeyufu/miniconda3/bin/python scripts/48_measurement_projection_repair.py`.

## Authoritative input

`projection/cells.parquet`, `projection/theta.npy`, `projection/obs.csv` and `projection/contract.json` define a single deterministic cell input. Aggregations use float64. cell_id is a source-order QC position tied to the recorded input hash, explicitly not a barcode. Optional reconstructed raw IDs retain their separate provenance.

The existing GSM5050530 raw-count probe verifies 2279 QC rows against all 15 deterministic coordinates (max absolute difference 3.5762786865234375e-07). It is evidence for the saved deterministic projection, not a claim that barcode IDs were present in historical artifacts.

## Result replacements

11 patients, 7 healed/4 not healed: all-cell difference 0.0292796678028; 30000-draw bootstrap interval [-0.0024419653327118623, 0.06670583026054017]; exhaustive p=90/330=0.272727272727; add-one comparison=0.274924471299.

Patient Welch SE=0.0194462525768, df=8.11383264025; M*=0.0653752492785. M* is the analytic infimum; strict p<.05 requires M>M*. This is not an MCID.

14 specimens (diagnostic only): difference=0.0457369544466; bootstrap=[0.012343447469591375, 0.08100828861444198]; exhaustive p=158/2002=0.0789210789211; add-one=0.0793809286071; M*=0.0796436635839.

Equal-specimen-then-equal-patient difference=0.0297317881994. Patient omission range=[0.014328632713224286, 0.036469582932468284].

Topic LOSO AUC=0.822222222222; null mean=0.561571761572, add-one p=0.0604093859211, exhaustive p=0.0599400599401.

Topic LOPO AUC=0.714285714286; null mean=0.539935064935, add-one p=0.299093655589, exhaustive p=0.29696969697.

Sample topic cross-arm credits are exactly invariant (45/45), proving every fixed-score arm-stratified topic bootstrap AUC and paired difference is unchanged. Topic bootstrap itself is rerun; other sample OOF scores were not saved and their paired interval entries are reused with the mathematical proof in representation_inference/report.json. Patient OOF scores were saved: all patient AUC/pair bootstraps are actually recomputed from shared indices. Non-topic fits/permutation nulls are hash-referenced reuse; the encoder/PCA/NMF are not refitted.

Full-score Spearman correlations: {"topic_simplex_theta0__module_score": 0.7098901098901099, "topic_simplex_theta0__pca": 0.17362637362637365, "topic_simplex_theta0__nmf": 0.6067041186941533, "module_score__pca": 0.4021978021978022, "module_score__nmf": 0.6755972603773723, "pca__nmf": 0.32001975491559737}

Design spread: healed SD=0.0517043832353; not-healed SD=0.0168580567519; median rank correlation=0.504450960446. All three simulation designs were recalculated. These are specimen-derived hypothetical spread/correlation scenarios, not estimated patient-population variances or recruitment recommendations.

Design totals: detect={"difference_0.02": 100, "difference_0.03": 47, "difference_0.0457": 21, "difference_0.06": 14, "difference_0.08": 9, "difference_0.1": 8}; equivalence={"margin_0.02": 110, "margin_0.03": 49, "margin_0.05": 20, "margin_0.075": 11, "margin_0.1": 8}; AUC={"auc_gap_0.05": 706, "auc_gap_0.1": 181, "auc_gap_0.15": 85, "auc_gap_0.2": 50}.

Repeat G7/G8 (same patient): fibroblast means=0.350415581639, 0.0920663940101; absolute difference=0.258349187629; high-state fractions=0.688053097345, 0.142237640937; Wasserstein=0.258349187629; ECDF supremum=0.578891924875. The split-half upper 97.5% mean differences are 0.0552537367531 and 0.0184609882655. These are sampling diagnostic ranges, not biological CIs or ICC.

## Compatible paths

- patient_mapping_equivalence/report.json and sample_scores.csv replace the old 14-specimen files.
- patient_unit_remap/report.json and patient_scores.csv replace the old 11-patient all-cell files.
- representation_benchmark/report.json and sample_scores.csv replace full descriptive topic scores/correlations.
- representation_inference/report.json replaces sample AUC inference.
- patient_unit_representation_benchmark/report.json and patient_scores.csv replace patient AUC inference.
- design_power_simulation/report.json replaces all old design scenarios.
- mathematical_audit/report.json replaces patient mathematical audit; fibroblast decomposition is recomputed and unchanged within rounding.
- robustness_extensions/report.json replaces patient all-cell omission; temporal and fibroblast-semantic branches are explicitly reused.
- tables/paper1_representation_inference.csv, tables/paper1_patient_unit_and_anatomy.csv, tables/paper1_representation_cross_method.csv retain exporter schemas. The paired-anatomy row retains the original independent source values.
- repeat_specimens/report.json, paired_specimens.csv, specimen_scores.csv and split_half_draws.csv provide the new repeat diagnostic.

## Scope and verification

The independent 11-patient fibroblast ECDF branch already uses this deterministic source and is not replaced by the all-cell statistics. Gate/marker/context concerns require the separate controls revision. Bootstrap intervals condition on frozen scores; label enumeration assumes exchangeability in an observational cohort. No new subjects or independent validation are introduced.

Source hashes unchanged after run: True. All output JSON is finite; simplex/order checks and expected unit counts passed. See report.json for complete input/output hashes and old-to-new comparisons.
