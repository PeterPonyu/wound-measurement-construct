# Frozen P1 compact table schema

All CSVs use one header row. Missing numeric values are empty fields with explicit status or missing-count columns, never numeric zero. Fit seeds and capacity variants are computational sensitivity, not new biological patients. All patient tables retain the supplied 20-person mapping, including 11 DFU patients. No prognostic tests or inferential confidence intervals are produced.

| Table | Key / expected rows | Values |
| --- | --- | --- |
| fits.csv | job_id / 9 | K, seed, optimizer_steps, reload_verified, reload_max_abs_error, primary_match_status, primary_new_topic, primary_correlation, semantics_available, cut_full_sample_fibroblasts, specimen_mean_fraction_spearman, mean_fraction_correlation_unit, unmatched_reference_topics, unmatched_new_topics |
| topic_matches.csv | job_id + row_kind + reference_topic/new_topic / variable | K, seed, status, reason, assigned_correlation, rowwise_best_topic, rowwise_best_correlation, best_minus_second, rowwise_used_for_primary (always false). Every 15 reference topics and every unmatched new topic is retained per fit. |
| patient_readouts.csv | K + seed + patient_id + readout_scope / 360 | job_id, group, is_DFU, n_cells, n_specimens, status, mean_loading, fraction_above_fibroblast_descriptive_cut, cut_full_sample_fibroblasts, threshold_scope, patient_aggregation |
| seed_ranges.csv | K + patient_id + readout_scope + metric / 240 | planned_seeds, available_seeds, missing_seeds, min_over_available_seeds, max_over_available_seeds, range_over_available_seeds, complete_three_seed_readout, interpretation |
| semantics.csv | job_id + comparison / 45 | K, seed, available, reason, specimens, units, specimens_favour_first, units_favour_first, median_difference_nats_per_cell, threshold_scope, validation_scope, exclusions_json |
| patient_semantics.csv | job_id + patient_id + comparison / 900 | K, seed, group, difference_nats_per_cell, status, reason, aggregation, prognostic_test (always false) |
| semantics_specimens.csv | job_id + specimen / 225 for current 25-specimen mapping | K, seed, patient_id, status, exclusions_json, n_cells_scored, n_splits, cut_training_only_logit_scale, hard_available, threshold_scope, C1, C2, M1, M2, H1 |

Readout scopes are `fibroblast_only` and `all_cells`. Patient readouts are equal specimen means within patient, not pooled-cell means. The all-cell fraction uses the fibroblast-derived full-sample descriptive cutoff; it is not fibroblast state prevalence. Full-sample descriptive readout cutoffs and patient-excluded H1 density cutoffs have distinct columns and must not be mixed. Semantics scores are logit-space log predictive density in nats per cell, conditional on full-discovery fitted scores/lineage labels; only reference density/threshold fitting excludes the scored patient's specimens.

Comparisons: `M1_vs_C1`, `M2_vs_C2`, `M1_vs_C2`, `H1_vs_C1`, `M1_vs_H1`. Positive differences favour the first density family descriptively. Missing comparisons remain present. Min/max ranges with fewer than 3 available seeds are labeled incomplete; they are not confidence intervals or selected best-fit estimates.

`verification.json` contains audited complete queue counts, nine fresh full CPU inference replay records, table row counts/SHA256, cohort counts, interpretation limits and final storage evidence.
