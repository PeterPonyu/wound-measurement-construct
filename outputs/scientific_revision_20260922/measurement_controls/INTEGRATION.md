# Measurement-control repair handoff

This directory contains a retrospective measurement audit. It does not add
independent biological validation and it does not refit the topic model.

## Raw-count column shuffle

`raw_shuffle_report.json` is based on 6000 cells,
with 240 cells selected from each of
25 discovery GSMs. The source is the integer
count cache `discovery_raw_counts.npz`, and each gene column is independently
permuted before library normalization and projection. The report records an
exact empirical marginal check for all 23572
columns, the same frozen checkpoint/panel in both branches, and explicit
`deterministic=True` `softmax(mu)` projection. It also reports library size,
detection, perplexity, HHI, maximum topic coordinate, and pre-softmax coordinate
concentration for both branches. Mean cell perplexity changes from
3.438302 to 11.230590; HHI changes
from 0.574026 to 0.146722.

Use these numbers for Figure 6E. Historical 3.68/14.43 came from a decoder
reconstruction shuffle and an asymmetric real/null projection; they are not
the new raw-count experiment. The new shuffled mean perplexity of 11.23 is
below the historical 0.95 K = 14.25 near-uniform threshold. This experiment
does not run a topology audit and does not establish that the historical
integrity gate rejects the shuffled input. Describe lower concentration only.
Gene shuffling also changes row library-size and detection distributions, so
the contrast does not isolate a biological co-expression mechanism from those
technical changes. It is not independent biological validation.

The identity/cache contract is in `raw_cache_manifest.json`. `raw_barcode` is
the barcode read from the source CSV; `raw_row_0` is its source column order;
`qc_row_0` is the post-QC GSM-local row; `source_order_cell_id` is explicitly a
synthetic order ID and must not be described as an original barcode linkage.

## Marker gate sensitivity

`gate_report.json` compares the original pooled gate, a frozen reference fitted
once on all 76,461 discovery cells, and a per-GSM adaptive reference. The
frozen gate exactly reproduces the pooled labels on the same cells; this is an
implementation check rather than independent identity validation. The adaptive
gate changes the marker reference population, so changes in fibroblast counts,
denominators, means, high-state fractions, specimen readouts, and the 11-patient
exact contrasts are sensitivity quantities. No gate has a supplied ground-truth
cell identity label.

Use `gate_patient_contrasts.csv` and `gate_specimen_changes.csv` for tables.
The fixed threshold is 0.184295848011971 with `>` as its operator.
The original pooled gate reproduces the existing cell-weighted 11-patient
results exactly. Zero-fibroblast rows are retained and reported rather than
silently dropped.

Per-GSM adaptation changes 7,044 labels
(9.21% of all cells).
Fibroblast calls change from 19,410
to 19,711, comprising
1,119 lost and
1,420 gained calls.
The largest absolute specimen changes are
0.037060 in
within-fibroblast mean and
0.069525 in
high-state fraction. No specimen or patient has zero fibroblasts in this run.

## DFU-only label shuffle

`dfu_label_report.json` first restricts the secondary specimen allocation to 14
DFU outcome-eligible specimens (9 healed, 5 non-healed), so the 11 healthy
specimens (9 patients) never enter a healing-label null. It reports 2002
exhaustive allocations. The primary result remains the existing 11-patient
exact test (7 healed, 4 non-healed; 330
allocations), with ties included using the documented tolerance. Specimen-level
results are secondary because repeated specimens share patients.

For Figure 6D use `dfu_specimen_exact_null.csv`, restricted to
`readout == high_state_fraction`: observed difference
0.238038232, absolute two-sided exact p
0.119880120, central 95% null interval
[-0.316657377, 0.322150711].
Replace the historical three-arm p=0.1358 and interval [-0.3183, +0.2874].
This exact enumeration uses 2,002 allocations, not 20,000 Monte Carlo shuffles.
The primary patient result remains p=0.369697 for high-state fraction;
specimen p values cannot be substituted for patient p values.

## Constructed continuous counterexample

`continuous_counterexample_report.json` is an illustrative construction from
the original specimen means and counts, not a fit to the cell-level data. It
must be described as a non-identification example rather than as a null model
or a new biological result. Its median correlation is
0.951144;
76% of the
200 construction draws reach the observed rho
0.932544. That percentage is not a calibrated p value.

## Historical numbers retained

The historical cell-weighted 11-patient contrasts remain in
`outputs/biological_expansion/measurement/patient_contrasts.csv`; this repair
checks and reproduces them. Keep these primary estimates. Replace the old
three-arm shuffle and decoder-proportion shuffle descriptions with the
explicitly identified new controls above. Cite the marker-reference comparison
as measurement sensitivity, with no independently verified identity reference.
