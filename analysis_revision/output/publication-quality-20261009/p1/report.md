# P1 fixed topic stability sidecar — 2026-10-09

All nine CUDA fits completed: K 10/15/20 × seeds 0/1/2 × 40,000 optimizer steps. Queue audit: 9 done, 0 failed/running/pending. Total execution was 1,177.55 seconds (19.63 minutes), including per-fit checkpoint reload, matching and conditional density analysis. The GPU worker exited and `worker.lock` was removed.

Full fresh CPU inference-state, all-cell projection and Hungarian-match replay passed for every fit. Every replay projection discrepancy was exactly 0.0. Completion/output hashes, frozen source/input manifest, table SHA256/row counts, and the separately reviewed verification helper/addendum hashes were rechecked after export. The canonical P1 subtree occupies approximately 106.3 MiB, below the 512 MiB bound; no raw matrices, runtimes, archives, Git commits/releases or public writes were created.

## Runtime and tests

Requested model/effort: `gpt-6.1-sol / xhigh`. Runtime-reported `turn_context`: `gpt-6.1-sol / xhigh`. Actual child thread: `01a1213b-95f3-7bc1-88f6-2225c233d102`. Fork session metadata carries parent session ID `01a1212f-da64-7fb0-8937-08c545eac5f8`. Runtime evidence is saved in `runtime_prepare.json`, `runtime_run.json` and `execution.json`; it was read from the child's actual session file, not inferred from requested parameters.

`python -m pytest -q tests/test_quality_topic_stability.py tests/test_expansion_measurement_guards.py --junitxml=output/publication-quality-20261009/p1/evidence/synthetic-tests.xml`: **74 passed in 6.03 seconds**. The exact-dtype output helper's synthetic check also passed, preserving strict rtol=0/atol=1e-12 and unavailable-primary rows. These unit tests used synthetic fixtures, not human data. The initial bare `pytest` invocation failed import collection; using `python -m pytest` corrected the invocation without shared-source edits.

## Results and adverse evidence

| K | Reference-topic-0 assigned correlations, seeds 0/1/2 | Unmatched topics retained |
| --- | --- | --- |
| 10 | 0.816832 / 0.938406 / 0.965161 | 5 reference topics per fit |
| 15 | 0.979762 / 0.982233 / 0.745977 | None |
| 20 | 0.987605 / 0.832379 / 0.603801 | 5 new topics per fit |

All nine primary reference-topic-0 assignments and their readouts were available. This is not a threshold-qualified stability success claim: assignment strength varied from 0.603801 to 0.987605, and every weaker/adverse and unmatched topic row was retained. No rowwise-best alternative replaced any primary assignment.

M1 versus C1 specimen-level median predictive density differences were positive in all nine fits; 13–16 of the same 20 mapped people descriptively favoured M1. M2 versus C2 medians were also positive in all nine fits. However, M1 versus C2 specimen-level medians were negative for **K=15/seed=1 (-0.001233 nats/cell)** and **K=20/seed=1 (-0.008090 nats/cell)**; only 8/20 and 9/20 people respectively favoured M1. These contrary rows remain in `semantics.csv` and `patient_semantics.csv`.

Specimen mean-versus-fraction Spearman correlations ranged from 0.905963 to 0.950262. Patient readouts were not uniformly insensitive to seed: the largest within-K three-seed fibroblast-only ranges were 0.277364 in mean loading and 0.364706 in fraction above the per-fit descriptive cutoff. No fit, K, patient or seed was selected based on these outcomes.

All 240 seed-range rows contain all three seeds. All 900 patient-density comparison rows were available. These remain computational sensitivity on **20 existing people, including 11 DFU patients**, 25 specimens, 76,461 cells and 19,410 fibroblast cells, not additional biological replication.

## Preserved initial failure and reviewed correction

The original frozen runner finished every fit but exited 1 during its post-fit table extraction: `ValueError: worker/table equal-specimen readout disagreement`. Its extractor cast float32 checkpoint loadings to float64 before averaging, whereas the existing worker preserved float32 mean arithmetic. A read-only historical-pilot check independently exposed a 4.719520876536598e-10 maximum difference before new outcome review.

The raw initial traceback and structured failure remain separately in `evidence/initial_verification_failure.txt` and `.json`. The leader reviewed and authorized `replay_tables_exact_dtype.py` and `precision_addendum.json`. That output-only helper restored exactly the worker's specimen and patient mean arithmetic, retained rtol=0/atol=1e-12, and executed the original complete audit/replay/table-generation logic. `verification.json` binds both SHA256 values. The frozen config, runner, tests, training plan, raw results and checkpoints were not edited. Leader review is operator-recorded engineering review, not independent author signoff.

## Tables for integration

| Table | Rows | Key |
| --- | --- | --- |
| fits.csv | 9 | job_id / K / seed |
| topic_matches.csv | 150 | job_id / row_kind / reference_topic or new_topic |
| patient_readouts.csv | 360 | K / seed / patient_id / readout_scope |
| seed_ranges.csv | 240 | K / patient_id / readout_scope / metric |
| semantics.csv | 45 | job_id / comparison |
| patient_semantics.csv | 900 | job_id / patient_id / comparison |
| semantics_specimens.csv | 225 | job_id / specimen |

Full columns/units and the unchanged integration contract are in `table_schema.md`. Missing values remain blank fields with explicit status/counts, never zeros. Patient readouts are equal-specimen means within person, not cell pooled. Fibroblast-only and all-cell scopes are distinct. The all-cell fraction uses the fibroblast-derived full-sample descriptive cutoff and is not fibroblast prevalence. H1 density thresholds use other patients only and are recorded on logit score scale separately from full-sample readout cutoffs.

Density results are conditional on full-discovery-fitted scores and supplied lineage labels, not end-to-end held-person validation or evidence of discrete biological states. Seeds/K variants are not patients; ranges are not confidence intervals. No new prognosis, healing, amputation, mechanism, MCID or hypothesis test was performed. Fixed-step completion does not establish optimizer convergence.

Changed source paths are only `config/quality_topic_stability_20261009.json`, `scripts/run_quality_topic_stability.py` and `tests/test_quality_topic_stability.py`. All other task artifacts are under the single new `output/publication-quality-20261009/p1/` tree. Shared expansion/core, manuscript, README, publication, AGENTS and historical outputs were not edited.
