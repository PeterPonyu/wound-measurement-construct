# Reproduction guide

## Data and biological units

The discovery analysis contains 25 specimens mapped to 20 patients. Healing inference uses 14 specimens from 11 patients (7 healed and 4 not healed).

Source counts and annotations are supplied by the following GEO records, under their source-study terms:

- [GSE165816](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165816)
- [GSE231643](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE231643)
- [GSE241132](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241132)
- [GSE223964](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223964)
- [GSE245703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245703)
- [GSE268834](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268834)
- [GSE248247](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248247)

Download counts into `data/raw/<accession>/`. Acquisition helpers are in scripts/fetch_*.sh. Human wound cell annotations must be joined by barcode before selecting fibroblasts. The GSE165816 clinical map is derived from [the source study's supplementary clinical workbook](https://doi.org/10.1038/s41467-021-27801-8); place that workbook at `cohort_metadata/source/41467_2021_27801_clinical_table.xlsx` if rebuilding the patient map. This source accession is used to fit the independent frozen representation; it does not supply a temporal healing target.

## Analysis order

Run from the archive root. Each program documents options with `--help`. Some programs consume completed outputs from earlier steps; preserve reference reports and use fresh output directories for alternative runs. A complete raw-data rerun requires count downloads and substantially more computation than rendering saved results.

1. Fit the 7,002-gene, 15-coordinate discovery representation with `python3 scripts/fit_expression_representation.py --steps 40000 --seed 0`. The generic CLI default is shorter than the manuscript's 40,000 steps. Hardware-dependent training may differ from the frozen reference weights.
2. Project the debridement cohort, compare lineage readouts, characterize state mixtures, and assess technical controls and ambient RNA. Run `scan_sample_covariates.py` for the entire 31-covariate screen and the threshold/capacity and negative-control programs for those sensitivity analyses.
3. Build or read the discovery patient map, infer the patient outcome contrast, benchmark patient readouts and assess paired anatomy. The source-study inventory includes more specimens than the selected foot subset; use the 11 mapped patients for healing inference.
4. Run the external cohort projection programs before the disjoint immune-compartment analysis. Cohort metadata describe tissue context, not independently adjudicated healing outcomes.
5. Run the representation and mixture inference programs after their source analyses. `assess_power_and_donor_geometry.py` supplies the immune-replication and observed-design-power summaries; anatomical contrasts remain exploratory. Future-cohort simulations are conditional planning calculations, not validated recruitment recommendations.
6. Run `assess_robustness.py` to recompute the fibroblast-count-weighted collapse to 20 patients, 10,000 patient bootstrap draws and the 11-patient outcome influence diagnostic. The omission range is not a new confidence interval.

## Verification scope

The archive verifier checks the checksum manifest, Python syntax, matching citation metadata and CLI imports. The full reproduction package additionally checks the frozen decoder, deterministic projection and training-only standardization. Manuscript builds check protected narrative sections, adjacent equation references, table citations, embedded fonts, complete citations and figure placement. Figure builds enforce black bold lettering, vector content and unchanged numerical source files.

The software-only archive contains programs, not pretrained weights or completed observations. Its checksum/import checks do not claim that a full raw-count pipeline has been rerun on every platform. Provenance fingerprints inside reports describe the original computation; MANIFEST.sha256.json hashes the files in the exact distributed artifact.

## Mathematical audit

Run `python3 -m unittest discover -s tests -v` for analytic implementation checks. Run `python3 scripts/audit_measurement_math.py --output-dir outputs/reruns/mathematical_audit` after producing its saved inputs. The full reproduction package includes those inputs; the software-only archive requires the preceding analysis steps. Historical report hashes describe the original computation, while the release manifest describes the current exported files. Documentation corrections are not a claim that every original model has been refitted.
